// Architect C: the rival collision stage (cc_stage.h) in the real all-twin engine.
//
//   cc_enginebench PACKS REPLAY...     env CC_REPS=n (default 5), TWB_TICKS=N (cap, default 4000)
//
// Per replay: FV's hand-off state (T = 2600), the replay's own controls, policy spec1n0b1, the
// baseline tick (tw::step::StepTicks on the world's model), as forevervalidator-twin-step-bench.
// model.collision is (B) the world's own stage (src/twin/collide, as linked) or (C) CcCollision over
// the same world package and car shapes. Reported: collision cycles per tick (a timing wrapper,
// best of n, alternating B / C; PMU core cycles of the run x the collision calls'
// TSC share), all-twin cycles per tick, Detect calls per tick; checks: one pass
// with both stages on every call (bitwise), the end states of B, C and C with its cache flushed
// before every call (bitwise).
#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/experimental/twin.h>
#include <forevervalidator/native.h>

#include <x86intrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "twin_pmu.h"
#include "cc_stage.h"
#include "policy.h"
#include "stages.h"
#include "tw_collide.h"
#include "tw_step.h"
#include "twin_api_internal.h"
#include "twin_world_packs.h"

namespace fv = forevervalidator;
namespace fvx = forevervalidator::experimental;
namespace tx = forevervalidator::experimental::twin;
using Clock = std::chrono::steady_clock;

namespace tw::ccbench {
std::unique_ptr<step::CollisionStage> MakeBStage(const world::WorldPackView &view, const void *carShapes, std::size_t carBytes, std::string *err,
                                                 bool noiseNarrow);
}

namespace {

bool Same(const tw::step::SubstepContacts &a, const tw::step::SubstepContacts &b) {
    if (a.n != b.n || a.overflow != b.overflow) return false;
    for (int c = 0; c < a.n; ++c)
        if (std::memcmp(&a.c[c], &b.c[c], sizeof a.c[c]) != 0) return false;
    return true;
}

struct Timed : tw::step::CollisionStage {
    const tw::step::CollisionStage *inner = nullptr;
    mutable std::uint64_t tsc = 0, calls = 0;
    void Detect(const tw::model::TwinState &s, tw::step::SubstepContacts &out) const override { DetectCached(s, nullptr, out); }
    std::unique_ptr<tw::step::CollisionCache> NewCache() const override { return inner->NewCache(); }
    void DetectCached(const tw::model::TwinState &s, tw::step::CollisionCache *cache, tw::step::SubstepContacts &out) const override {
        const std::uint64_t a = __rdtsc();
        inner->DetectCached(s, cache, out);
        tsc += __rdtsc() - a;
        ++calls;
    }
};

// both stages on every call: B's answer drives the engine, C's is compared bitwise
struct Both : tw::step::CollisionStage {
    const tw::step::CollisionStage *b = nullptr, *c = nullptr;
    mutable std::unique_ptr<tw::step::CollisionCache> cb, cc;
    mutable std::uint64_t calls = 0, differ = 0, tb = 0, tc = 0;
    void Detect(const tw::model::TwinState &s, tw::step::SubstepContacts &out) const override { DetectCached(s, nullptr, out); }
    void DetectCached(const tw::model::TwinState &s, tw::step::CollisionCache *, tw::step::SubstepContacts &out) const override {
        thread_local tw::step::SubstepContacts oc;
        // the stage called second finds the call's data warm (an A/A run shows ~1.3x): alternate the order
        const std::uint64_t t0 = __rdtsc();
        if (calls & 1u) {
            c->DetectCached(s, cc.get(), oc);
            const std::uint64_t t1 = __rdtsc();
            b->DetectCached(s, cb.get(), out);
            tb += __rdtsc() - t1;
            tc += t1 - t0;
        } else {
            b->DetectCached(s, cb.get(), out);
            const std::uint64_t t1 = __rdtsc();
            c->DetectCached(s, cc.get(), oc);
            tc += __rdtsc() - t1;
            tb += t1 - t0;
        }
        ++calls;
        differ += !Same(out, oc);
    }
};

void RunReplay(const std::string &packDir, const std::string &path, std::uint32_t maxTicks, int reps) {
    tw::bench::Pmu pmu;
    if (!pmu.ok()) { std::printf("PMU (rdpmc) unavailable\n"); return; }
    const std::string name = path.substr(path.find_last_of('/') + 1);
    auto bytes = fv::ReadNativeReplayFile(path, fv::ReplayIdentity{path});
    auto source = fv::OpenInstalledPackDirectory(packDir);
    if (!bytes || !source) { std::printf("%s: open failed\n", name.c_str()); return; }
    fvx::PhysicsSandboxOptions options;
    options.backend = fv::SimulationBackend::OptimizedCpu;
    auto created = fvx::CreatePhysicsSandbox(std::move(source).Value(), options);
    if (!created) { std::printf("%s: sandbox failed\n", name.c_str()); return; }
    fvx::PhysicsSandbox sb = std::move(created).Value();
    auto loaded = sb.LoadReplay({bytes.Value().data(), bytes.Value().size()}, fv::ReplayIdentity{path});
    if (!loaded) { std::printf("%s: skip (%s)\n", name.c_str(), loaded.Error().diagnostic.c_str()); return; }
    auto handoff = sb.CaptureState();
    auto inputs = sb.ReadInputs();
    if (!handoff || !inputs) { std::printf("%s: capture/inputs failed\n", name.c_str()); return; }
    auto before = sb.ReadState();
    if (before && before.Value().durationMs > before.Value().timeMs)
        maxTicks = std::min<std::uint32_t>(maxTicks, static_cast<std::uint32_t>((before.Value().durationMs - before.Value().timeMs) / 10u));
    tw::step::PolicyConfig spec;
    auto world = tx::CompileTwinWorld(sb, fv::TwinPolicy::ContactSubstepsFull, fv::TwinCollision::Exact);
    const char *pol = std::getenv("CC_POLICY") ? std::getenv("CC_POLICY") : "spec1n0b1";
    if (!world || !tw::step::ParsePolicy(pol, &spec)) { std::printf("%s: compile failed\n", name.c_str()); return; }
    const tx::TwinWorld &w = *world.Value();
    auto imported = tx::TwinImportState(w, handoff.Value());
    auto stream = tx::TwinControlsFromEvents(w, inputs.Value(), 2610u, maxTicks);
    if (!imported || !stream || !w.Internal().packs) { std::printf("%s: import/controls failed\n", name.c_str()); return; }
    const tw::fv::TwinWorldPacks &packs = *w.Internal().packs;
    tw::collide_c::CcCollision cstage;
    std::string err;
    tw::collide_c::StageOptions copt;
    copt.noise = std::getenv("CC_NOISE") != nullptr;
    const char *sw = std::getenv("CC_SWITCH");
    if (sw && !tw::collide_c::ParseStageSwitch(sw, &copt)) { std::printf("bad CC_SWITCH\n"); return; }
    const auto i0 = Clock::now();
    if (!cstage.Init(packs.view, packs.shapes, &err, copt)) { std::printf("%s: C init failed: %s\n", name.c_str(), err.c_str()); return; }
    const double initMs = std::chrono::duration<double, std::milli>(Clock::now() - i0).count();
    // B: its current sources built here (same flags as C), or with CC_BLIB the FV twin library's stage
    std::unique_ptr<tw::step::CollisionStage> bstage;
    if (std::getenv("CC_BASE")) {   // the reference is C's stage with these options (variants against their base)
        auto base = std::make_unique<tw::collide_c::CcCollision>();
        tw::collide_c::StageOptions bo;
        if (!tw::collide_c::ParseStageSwitch(std::getenv("CC_BASE"), &bo) || !base->Init(packs.view, packs.shapes, &err, bo)) {
            std::printf("%s: base init failed\n", name.c_str());
            return;
        }
        bstage = std::move(base);
    } else if (!std::getenv("CC_BLIB")) {
        bstage = tw::ccbench::MakeBStage(packs.view, &packs.shapes, sizeof packs.shapes, &err, std::getenv("CC_BNOISE") != nullptr);
        if (!bstage) { std::printf("%s: B init failed: %s\n", name.c_str(), err.c_str()); return; }
    }
    const tw::step::CollisionStage *stages[2] = {bstage ? bstage.get() : w.Internal().model.collision, &cstage};
    const tw::model::TwinState s0 = tx::internal::ToModel(imported.Value());
    const auto *ticks = stream.Value().ticks.data();
    const std::size_t nt = stream.Value().ticks.size();

    // check pass: both stages on every call of B's trajectory (B drives); also timed: the same calls for
    // both, PMU cycles of the run split by each stage's TSC share, min of reps
    std::uint64_t checkCalls = 0, checkDiffer = 0;
    double same[2] = {1e30, 1e30};
    for (int r = 0; r < reps; ++r) {
        tw::step::Model model = w.Internal().model;
        model.policyConfig = &spec;
        Both both;
        both.b = stages[0];
        both.c = stages[1];
        both.cb = stages[0]->NewCache();
        both.cc = stages[1]->NewCache();
        model.collision = &both;
        tw::model::TwinState s = s0;
        tw::step::Events ev;
        const auto c0 = pmu.Read();
        const std::uint64_t r0 = __rdtsc();
        const std::uint32_t n = tw::step::StepTicks(model, s, ticks, nt, tw::step::StopRule{}, &ev, nullptr, nullptr);
        const std::uint64_t r1 = __rdtsc();
        const auto c1 = pmu.Read();
        const double runCycles = double(tw::bench::Pmu::Delta(c0, c1).cycles), span = double(r1 - r0), nn = double(std::max<std::uint32_t>(1u, n));
        same[0] = std::min(same[0], runCycles * double(both.tb) / span / nn);
        same[1] = std::min(same[1], runCycles * double(both.tc) / span / nn);
        checkCalls = both.calls;
        checkDiffer = both.differ;
    }
    // timed passes (alternating), then C with its cache flushed before every call
    double best[2] = {1e30, 1e30}, all[2] = {1e30, 1e30};
    double callsPerTick = 0.0, cpt[2] = {0.0, 0.0};
    std::uint32_t nTicks = 0;
    tw::model::TwinState end[3];
    for (int r = 0; r < 2 * reps + 1; ++r) {
        const int mode = r < 2 * reps ? (r & 1) : 2;   // 0 B, 1 C, 2 C flushed every call
        tw::step::Model model = w.Internal().model;
        model.policyConfig = &spec;
        Timed timed;
        timed.inner = stages[mode == 0 ? 0 : 1];
        model.collision = &timed;
        std::unique_ptr<tw::step::CollisionCache> cache = timed.NewCache();
        if (mode == 2 && cache) cache->flushEveryCall = true;
        tw::model::TwinState s = s0;
        tw::step::Events ev;
        // PMU core cycles of the whole run, split by the TSC share of the collision calls (B's method)
        const auto c0 = pmu.Read();
        const std::uint64_t r0 = __rdtsc();
        nTicks = tw::step::StepTicks(model, s, ticks, nt, tw::step::StopRule{}, &ev, nullptr, cache.get());
        const std::uint64_t r1 = __rdtsc();
        const auto c1 = pmu.Read();
        const double runCycles = double(tw::bench::Pmu::Delta(c0, c1).cycles);
        end[mode] = s;
        if (mode < 2) {
            best[mode] = std::min(best[mode], runCycles * (double(timed.tsc) / double(r1 - r0)) / std::max<std::uint32_t>(1u, nTicks));
            all[mode] = std::min(all[mode], runCycles / std::max<std::uint32_t>(1u, nTicks));
            callsPerTick = double(timed.calls) / std::max<std::uint32_t>(1u, nTicks);
            cpt[mode] = callsPerTick;
        }
    }
    const bool sameBC = std::memcmp(&end[0], &end[1], sizeof end[0]) == 0;
    const bool sameFlush = std::memcmp(&end[1], &end[2], sizeof end[0]) == 0;
    std::printf("%s [%s, B %s, C %s, best of %d]: %u ticks; collision cycles/tick on B's trajectory (both stages, same calls, alternating order): B %.0f, C %.0f "
                "(%.2fx); each on its own trajectory: B %.0f (%.2f calls/tick), C %.0f (%.2f calls/tick); all-twin cycles/tick B %.0f, C %.0f (C's collision share %.0f%%); "
                "check: %llu of %llu calls differ, end state B==C %s, C flushed==C %s\n",
                name.c_str(), pol, std::getenv("CC_BASE") ? std::getenv("CC_BASE") : bstage ? (std::getenv("CC_BNOISE") ? "built here + C's noise narrow" : "built here") : "FV twin library", sw ? sw : (copt.noise ? "noise-level" : "exact"), reps, nTicks, same[0], same[1],
                same[0] / same[1], best[0], cpt[0], best[1], cpt[1], all[0], all[1], 100.0 * best[1] / all[1], static_cast<unsigned long long>(checkDiffer),
                static_cast<unsigned long long>(checkCalls), sameBC ? "yes" : "NO", sameFlush ? "yes" : "NO");
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: cc_enginebench PACKS REPLAY...\n"); return 2; }
    const std::uint32_t maxTicks = std::getenv("TWB_TICKS") ? static_cast<std::uint32_t>(std::atoi(std::getenv("TWB_TICKS"))) : 4000u;
    const int reps = std::getenv("CC_REPS") ? std::max(1, std::atoi(std::getenv("CC_REPS"))) : 5;
    for (int i = 2; i < argc; ++i) RunReplay(argv[1], argv[i], maxTicks, reps);
    return 0;
}
