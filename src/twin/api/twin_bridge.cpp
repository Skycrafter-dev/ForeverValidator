// Twin API: the bridge from a twin result to FV's exact inputs (docs/twin/bridge.md).
//
// Pass 1 plays the twin winner's controls on FV optimized_cpu; pass 2 solves them tick by tick with a receding
// horizon: at tick t, from FV's committed state, the candidates are no edit, the steer moved by +-2^k / 65536
// (k = 0..10), the steer timing (the nearest later differing steer value within 5 ticks, or the previous tick's) and,
// rarely, a gas or brake flip where the twin's pedal changes within 3 ticks; each is scored on FV over the next
// `lookTicks` ticks (the rms position gap to the twin's path, the rest of the controls as they are). An edit is kept
// when it beats no edit by `margin` (a pedal shift: by `pedalMargin` over the best steer move); among the near-best
// (within 5%) the smallest. The closer of the two passes is returned as input events, replayed through FV's own
// input path first: its plan's controls and states must be the solved ones bit for bit.
#include <forevervalidator/experimental/twin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "validation/api/physics_sandbox_twin_test_access.h"

namespace forevervalidator::experimental::twin {

namespace {

using Access = twin_test::PhysicsSandboxTwinTestAccess;
using Controls = Access::TickControls;

PhysicsSandboxError BridgeError(std::string diagnostic) {
    PhysicsSandboxError error;
    error.code = PhysicsSandboxErrorCode::SimulationFailed;
    error.diagnostic = "twin bridge: " + std::move(diagnostic);
    return error;
}

struct Body {
    float pos[3];
};

double Gap(const PhysicsSandboxCarState &f, const Body &t) {
    const double x = double(f.position.x) - t.pos[0], y = double(f.position.y) - t.pos[1], z = double(f.position.z) - t.pos[2];
    return std::sqrt(x * x + y * y + z * z);
}

// A twin control word's vehicle controls (TwinControl: bits 0-17 signed steer in 1/65536, 18 gas, 19 brake).
Controls FromTwinControl(TwinControl c) {
    std::int32_t steer = static_cast<std::int32_t>(c & 0x3FFFFu);
    if (steer & 0x20000) steer -= 0x40000;
    return Controls{(c >> 18) & 1u ? 1.0f : 0.0f, (c >> 19) & 1u ? 1.0f : 0.0f, static_cast<float>(steer) / 65536.0f};
}

bool SameControls(const Controls &a, const Controls &b) {
    return a.gateA == b.gateA && a.gateB == b.gateB && a.steering == b.steering;
}

bool IsVehicleChannel(PhysicsSandboxInputAction a) {
    using A = PhysicsSandboxInputAction;
    return a == A::Accelerate || a == A::Gas || a == A::Brake || a == A::Steer || a == A::SteerLeft || a == A::SteerRight;
}

// The controls as input events for the window (branch, end]: tick t's at branch + 10 (t + 1) (FV's input state machine
// reads a tick's events at its end time); at the first tick the three channels set explicitly, then each change; the
// base inputs' other events in the window (respawns, race actions) kept. The sandbox's window starts at the branch time
// itself, whose events the tick ending at the branch already read: atBranch puts them back as they were.
std::vector<PhysicsSandboxInputEvent> WindowEvents(const std::vector<Controls> &c, std::int64_t branchMs,
                                                   const std::vector<PhysicsSandboxInputEvent> &base,
                                                   const std::vector<PhysicsSandboxInputEvent> &atBranch) {
    std::vector<PhysicsSandboxInputEvent> out = atBranch;
    auto sw = [](std::int32_t t, PhysicsSandboxInputAction a, bool on) {
        PhysicsSandboxInputEvent e;
        e.timeMs = t;
        e.action = a;
        e.value.kind = PhysicsSandboxInputValueKind::Switch;
        e.value.switchState = on ? PhysicsSandboxSwitchState::Pressed : PhysicsSandboxSwitchState::Released;
        return e;
    };
    for (std::size_t t = 0; t < c.size(); ++t) {
        const std::int32_t tm = static_cast<std::int32_t>(branchMs + 10 * static_cast<std::int64_t>(t + 1u));
        const bool first = t == 0u;
        if (first || c[t].gateA != c[t - 1].gateA) out.push_back(sw(tm, PhysicsSandboxInputAction::Accelerate, c[t].gateA != 0.0f));
        if (first || c[t].gateB != c[t - 1].gateB) out.push_back(sw(tm, PhysicsSandboxInputAction::Brake, c[t].gateB != 0.0f));
        if (first || c[t].steering != c[t - 1].steering) {
            PhysicsSandboxInputEvent e;
            e.timeMs = tm;
            e.action = PhysicsSandboxInputAction::Steer;
            e.value.kind = PhysicsSandboxInputValueKind::Analog;
            e.value.analog = static_cast<AnalogInputState>(std::lround(c[t].steering * 65536.0f));
            out.push_back(e);
        }
    }
    const std::int64_t endMs = branchMs + 10 * static_cast<std::int64_t>(c.size());
    for (const auto &e : base)
        if (e.timeMs > branchMs && e.timeMs <= endMs && !IsVehicleChannel(e.action)) out.push_back(e);
    std::stable_sort(out.begin(), out.end(), [](const auto &a, const auto &b) { return a.timeMs < b.timeMs; });
    return out;
}

class Bridge final : public TwinBridge {
public:
    Bridge(PhysicsSandbox box, std::shared_ptr<const TwinWorld> world, PhysicsSandboxState branch, TwinState start, std::int64_t branchMs,
           std::vector<PhysicsSandboxInputEvent> past)
        : box_(std::move(box)), world_(std::move(world)), branch_(std::move(branch)), start_(start), branchMs_(branchMs),
          past_(std::move(past)) {
        for (const auto &e : past_)
            if (e.timeMs == branchMs_) atBranch_.push_back(e);
    }

    PhysicsSandboxResult<TwinBridgeResult> Run(const std::vector<PhysicsSandboxInputEvent> &inputs, std::uint32_t ticks,
                                              const TwinBridgeOptions &options) noexcept override {
        using R = PhysicsSandboxResult<TwinBridgeResult>;
        try {
            return Solve(inputs, ticks, options);
        } catch (const std::bad_alloc &) {
            return R::Failure(BridgeError("allocation failed"));
        } catch (...) {
            return R::Failure(BridgeError("unexpected failure"));
        }
    }

private:
    PhysicsSandboxResult<TwinBridgeResult> Solve(const std::vector<PhysicsSandboxInputEvent> &inputs, std::uint32_t ticks,
                                                const TwinBridgeOptions &options) {
        using R = PhysicsSandboxResult<TwinBridgeResult>;
        if (ticks == 0u) return R::Failure(BridgeError("no ticks to follow"));
        // FV at the branch on the winner's inputs (its control plan: race actions, respawns): the branch's own events
        // up to the branch time (the ticks so far read them), the winner's after it
        std::vector<PhysicsSandboxInputEvent> full = past_;
        for (const auto &e : inputs)
            if (e.timeMs > branchMs_) full.push_back(e);
        if (!box_.RestoreState(branch_)) return R::Failure(BridgeError("restoring the branch"));
        if (auto r = box_.ReplaceInputs(full); !r) return R::Failure(BridgeError(r.Error().diagnostic));
        auto planned = box_.CaptureState();
        if (!planned) return R::Failure(BridgeError("capturing the branch"));
        const PhysicsSandboxState start = std::move(planned).Value();
        // the twin's path (the target) and controls, from the same state
        auto stream = TwinControlsFromEvents(*world_, inputs, 2610u + static_cast<std::uint32_t>(branchMs_), ticks);
        if (!stream) return R::Failure(BridgeError(stream.Error().diagnostic));
        std::vector<Body> target;
        std::vector<Controls> twinCtl;
        {
            TwinState s = start_;
            for (std::uint32_t t = 0; t < ticks && t < stream.Value().ticks.size(); ++t) {
                TwinEvents ev;
                if (TwinStep(*world_, s, stream.Value().ticks[t], &ev) != TwinStatus::Ok) break;
                std::uint32_t w[4];
                Body b{};
                if (TwinReadField(s, "body.pos", w, 4u) != 3u) return R::Failure(BridgeError("reading the twin's position"));
                std::memcpy(b.pos, w, sizeof b.pos);
                target.push_back(b);
                twinCtl.push_back(FromTwinControl(stream.Value().ticks[t]));
                if (ev.finished) break;
            }
        }
        const std::uint32_t n = static_cast<std::uint32_t>(target.size());
        if (n == 0u) return R::Failure(BridgeError("the twin made no tick"));
        TwinBridgeResult result;
        result.ticks = n;
        // pass 1
        std::vector<PhysicsSandboxCarState> passOne;
        double finalGap1 = 0.0;
        if (!box_.RestoreState(start)) return R::Failure(BridgeError("restoring"));
        for (std::uint32_t t = 0; t < n; ++t) {
            auto view = Access::AdvanceTickWithControls(box_, twinCtl[t]);
            if (!view) return R::Failure(BridgeError("pass 1: " + view.Error().diagnostic));
            passOne.push_back(view.Value().car);
            finalGap1 = Gap(view.Value().car, target[t]);
            result.passOneMaxDriftMeters = std::max(result.passOneMaxDriftMeters, finalGap1);
        }
        // pass 2
        std::vector<Controls> ctl = twinCtl;
        std::vector<PhysicsSandboxCarState> solved;
        double maxGap2 = 0.0, finalGap2 = 0.0;
        if (!SolvePass(start, target, twinCtl, options, &ctl, &solved, &maxGap2, &finalGap2, &result))
            return R::Failure(BridgeError(options.cancelled && options.cancelled() ? "cancelled" : "pass 2"));
        result.solved = maxGap2 < result.passOneMaxDriftMeters;
        const std::vector<Controls> &chosen = result.solved ? ctl : twinCtl;
        if (!result.solved) {
            result.steerEdits = result.steerShifts = result.pedalShifts = 0u;
            solved = std::move(passOne);
        }
        result.maxDriftMeters = result.solved ? maxGap2 : result.passOneMaxDriftMeters;
        result.finalDriftMeters = result.solved ? finalGap2 : finalGap1;
        // the chosen controls as events, replayed through FV's input path: the same controls and states bit for bit
        const std::int64_t solvedEndMs = branchMs_ + 10 * static_cast<std::int64_t>(n);
        auto events = WindowEvents(chosen, branchMs_, inputs, atBranch_);
        if (!box_.RestoreState(branch_)) return R::Failure(BridgeError("restoring the branch"));
        if (auto r = box_.ReplaceInputs(full); !r) return R::Failure(BridgeError(r.Error().diagnostic));
        if (auto r = box_.ReplaceInputWindow(branchMs_, solvedEndMs, events); !r) return R::Failure(BridgeError(r.Error().diagnostic));
        auto written = box_.ReadInputs();
        if (!written) return R::Failure(BridgeError("reading the solved inputs"));
        for (std::uint32_t t = 0; t < n; ++t) {
            Controls plan;
            if (!Access::NextTickControls(box_, &plan) || !SameControls(plan, chosen[t]))
                return R::Failure(BridgeError("the solved inputs' controls differ at tick " + std::to_string(t + 1u)));
            auto view = box_.AdvanceTicks(1u);
            if (!view) return R::Failure(BridgeError("replaying the solved inputs"));
            const auto &a = view.Value().car, &b = solved[t];
            if (std::memcmp(&a.position, &b.position, sizeof a.position) != 0 || std::memcmp(&a.linearSpeed, &b.linearSpeed, sizeof a.linearSpeed) != 0)
                return R::Failure(BridgeError("the solved inputs' states differ at tick " + std::to_string(t + 1u)));
        }
        result.inputs = std::move(written).Value();
        return R::Success(std::move(result));
    }

    // The receding horizon (see the file comment). Candidates are scored on `threads` sandbox clones.
    bool SolvePass(const PhysicsSandboxState &start, const std::vector<Body> &target, const std::vector<Controls> &twinCtl,
                   const TwinBridgeOptions &o, std::vector<Controls> *ctlOut, std::vector<PhysicsSandboxCarState> *solved, double *maxGap,
                   double *finalGap, TwinBridgeResult *result) {
        const std::uint32_t n = static_cast<std::uint32_t>(target.size());
        std::vector<Controls> &ctl = *ctlOut;
        const std::uint32_t threads = std::max(1u, o.threads);
        while (clones_.size() + 1u < threads) {
            auto c = ClonePhysicsSandbox(box_);
            if (!c) break;
            clones_.push_back(std::move(c).Value());
        }
        std::vector<PhysicsSandbox *> boxes{&box_};
        for (auto &c : clones_) boxes.push_back(&c);
        if (boxes.size() > threads) boxes.resize(threads);
        struct Cand {
            Controls c;
            double cost;   // the edit's size (a pedal flip counts 10)
            int kind;      // -1 no edit, 0 nudge, 1 steer timing, 2 pedal timing
            double score = 1e300;
        };
        PhysicsSandboxState state = start;
        for (std::uint32_t t = 0; t < n; ++t) {
            if (o.cancelled && o.cancelled()) return false;
            std::vector<Cand> cands{{ctl[t], 0.0, -1}};
            for (int sign : {1, -1})
                for (int k = 0; k <= 10; ++k) {
                    const double sv = double(ctl[t].steering) + sign * (1 << k) / 65536.0;
                    if (sv > 1.0 || sv < -1.0) continue;
                    Controls c = ctl[t];
                    c.steering = static_cast<float>(sv);
                    cands.push_back({c, (1 << k) / 65536.0, 0});
                }
            for (std::uint32_t k = 1; k <= 5u && t + k < n; ++k)
                if (ctl[t + k].steering != ctl[t].steering) {
                    Controls c = ctl[t];
                    c.steering = ctl[t + k].steering;
                    cands.push_back({c, std::fabs(double(c.steering) - ctl[t].steering), 1});
                    break;
                }
            if (t > 0u && ctl[t - 1u].steering != ctl[t].steering) {
                Controls c = ctl[t];
                c.steering = ctl[t - 1u].steering;
                cands.push_back({c, std::fabs(double(c.steering) - ctl[t].steering), 1});
            }
            for (int g = 0; g < 2; ++g) {
                auto gate = [g](const Controls &c) { return g == 0 ? c.gateA : c.gateB; };
                bool near = false;
                for (std::uint32_t k = 1; k <= 3u; ++k) {
                    if (t + k < n && gate(twinCtl[t + k]) != gate(ctl[t])) near = true;
                    if (t >= k && gate(twinCtl[t - k]) != gate(ctl[t])) near = true;
                }
                if (!near) continue;
                Controls c = ctl[t];
                (g == 0 ? c.gateA : c.gateB) = 1.0f - gate(ctl[t]);
                cands.push_back({c, 10.0, 2});
            }
            const std::uint32_t end = std::min(n, t + o.lookTicks);
            auto work = [&](std::size_t tid) {
                PhysicsSandbox &box = *boxes[tid];
                for (std::size_t i = tid; i < cands.size(); i += boxes.size()) {
                    if (!box.RestoreState(state)) continue;
                    double acc = 0.0;
                    bool ok = true;
                    for (std::uint32_t k = t; k < end; ++k) {
                        auto view = Access::AdvanceTickWithControls(box, k == t ? cands[i].c : ctl[k]);
                        if (!view) {
                            ok = false;
                            break;
                        }
                        const double g = Gap(view.Value().car, target[k]);
                        acc += g * g;
                    }
                    if (ok) cands[i].score = std::sqrt(acc / (end - t));
                }
            };
            if (boxes.size() > 1u) {
                std::vector<std::thread> pool;
                for (std::size_t i = 1; i < boxes.size(); ++i) pool.emplace_back(work, i);
                work(0u);
                for (auto &th : pool) th.join();
            } else {
                work(0u);
            }
            const double base = cands[0].score;
            const Cand *best = nullptr;
            double bestSteer = base;
            for (const auto &cd : cands)
                if ((cd.kind == 0 || cd.kind == 1) && cd.score < bestSteer) bestSteer = cd.score;
            if (bestSteer < base * (1.0 - o.margin)) {
                const double lim = bestSteer * 1.05;
                for (const auto &cd : cands)
                    if ((cd.kind == 0 || cd.kind == 1) && cd.score <= lim && (!best || cd.cost < best->cost)) best = &cd;
            }
            const double ref = best ? best->score : base;
            for (const auto &cd : cands)
                if (cd.kind == 2 && cd.score < ref * (1.0 - o.pedalMargin) && (!best || best->kind != 2 || cd.score < best->score)) best = &cd;
            if (best) {
                (best->kind == 2 ? result->pedalShifts : best->kind == 1 ? result->steerShifts : result->steerEdits) += 1u;
                ctl[t] = best->c;
            }
            if (!box_.RestoreState(state)) return false;
            auto view = Access::AdvanceTickWithControls(box_, ctl[t]);
            if (!view) return false;
            auto next = box_.CaptureState();
            if (!next) return false;
            state = std::move(next).Value();
            solved->push_back(view.Value().car);
            *finalGap = Gap(view.Value().car, target[t]);
            *maxGap = std::max(*maxGap, *finalGap);
        }
        return true;
    }

    PhysicsSandbox box_;
    std::vector<PhysicsSandbox> clones_;
    std::shared_ptr<const TwinWorld> world_;
    PhysicsSandboxState branch_;
    TwinState start_;
    std::int64_t branchMs_;
    std::vector<PhysicsSandboxInputEvent> past_;       // the branch's own events up to the branch time
    std::vector<PhysicsSandboxInputEvent> atBranch_;   // those at the branch time (the tick ending there read them)
};

}  // namespace

PhysicsSandboxResult<std::unique_ptr<TwinBridge>> CreateTwinBridge(const PhysicsSandbox &loaded) noexcept {
    using R = PhysicsSandboxResult<std::unique_ptr<TwinBridge>>;
    try {
        if (loaded.Backend() != SimulationBackend::OptimizedCpu)
            return R::Failure(BridgeError("the sandbox must run on optimized_cpu (FV's exact CPU physics)"));
        auto clone = ClonePhysicsSandbox(loaded);
        if (!clone) return R::Failure(std::move(clone).Error());
        auto world = CompileTwinWorld(loaded);
        if (!world) return R::Failure(std::move(world).Error());
        auto branch = loaded.CaptureState();
        if (!branch) return R::Failure(std::move(branch).Error());
        auto start = TwinImportState(*world.Value(), branch.Value());
        if (!start) return R::Failure(std::move(start).Error());
        auto view = loaded.ReadState();
        if (!view) return R::Failure(std::move(view).Error());
        auto inputs = loaded.ReadInputs();
        if (!inputs) return R::Failure(std::move(inputs).Error());
        const std::int64_t branchMs = static_cast<std::int64_t>(view.Value().timeMs);
        std::vector<PhysicsSandboxInputEvent> past;
        for (const auto &e : inputs.Value())
            if (e.timeMs <= branchMs) past.push_back(e);
        return R::Success(std::make_unique<Bridge>(std::move(clone).Value(), std::move(world).Value(), std::move(branch).Value(),
                                                   start.Value(), branchMs, std::move(past)));
    } catch (const std::bad_alloc &) {
        return R::Failure(BridgeError("allocation failed"));
    } catch (...) {
        return R::Failure(BridgeError("unexpected failure"));
    }
}

}  // namespace forevervalidator::experimental::twin
