// Twin all-twin tick cost (M3 wired): per replay, from FV's hand-off state (T = 2600),
// the twin's TwinStep over the replay's own controls until the finish, against FV's own
// ticks (OptimizedCpu sandbox) over the same span. Collision is timed separately by a
// wrapping CollisionStage.
//   forevervalidator-twin-step-bench PACKS REPLAY...      env TWB_TICKS=N caps the ticks (default 4000)
//   env: TWB_REF (collision by the M3 commit's reference path), TWB_NOCACHE (no per-candidate cache),
//        TWB_STATS (stats pass), TWB_REPEAT=N (profiling), TWB_DUMP=DIR (pose streams for tw_m3check M3_POSES)
// Cycles = wall time x the core clock measured in the same run. Not a test: no pass/fail.
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

#include "policy.h"
#include "stages.h"
#include "tw_collide.h"
#include "cc_stage.h"
#include "tw_step.h"
#include "twin_api_internal.h"
#include "twin_world_packs.h"
#include "src/twin/common/twin_pmu.h"

namespace fv = forevervalidator;
namespace fvx = forevervalidator::experimental;
namespace tx = forevervalidator::experimental::twin;
using Clock = std::chrono::steady_clock;
using tw::bench::Pmu;

namespace {

double CoreGHz(int iters = 25000000) {
    std::uint64_t x = 0;
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        asm volatile("add $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0\n\tadd $1, %0" : "+r"(x));
    return 8.0 * iters / std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

struct TimedCollision : tw::step::CollisionStage {
    const tw::step::CollisionStage *inner = nullptr;
    bool ref = std::getenv("TWB_REF") != nullptr;   // the M3 commit's path (DetectShapesRef), same output
    const Pmu *pmu = nullptr;                        // A/B mode: PMU cycles around the calls
    mutable std::uint64_t pmuCycles = 0, pmuIns = 0;
    mutable std::uint64_t tsc = 0, calls = 0, contacts = 0, samePose = 0;
    mutable float lastKey[11] = {};
    mutable tw::collide::CollideStats st;
    mutable std::FILE *dump = nullptr;   // TWB_DUMP: the engine's shape poses per Detect call (tw_m3check M3_POSES)
    mutable std::FILE *cdump = nullptr;  // ... and its car poses (tw_m3check M3_REFD)
    void Detect(const tw::model::TwinState &s, tw::step::SubstepContacts &out) const override { DetectCached(s, nullptr, out); }
    std::unique_ptr<tw::step::CollisionCache> NewCache() const override { return inner->NewCache(); }
    bool bare = false;   // TWB_WRAPCHECK runs: only forward (measures this wrapper's own cost)
    void DetectCached(const tw::model::TwinState &s, tw::step::CollisionCache *cache, tw::step::SubstepContacts &out) const override {
        if (bare) { inner->DetectCached(s, cache, out); ++calls; return; }
        {   // how often a call repeats the previous call's pose bits (a last-result cache would answer it)
            float key[11];
            std::memcpy(key, s.pos, 12); std::memcpy(key + 3, s.quat, 16); std::memcpy(key + 7, s.absorb, 16);
            if (calls > 0 && std::memcmp(key, lastKey, sizeof key) == 0) ++samePose;
            std::memcpy(lastKey, key, sizeof key);
        }
        const auto *exact = ref ? dynamic_cast<const tw::collide::ExactCollision *>(inner) : nullptr;
        Pmu::Sample p0;
        if (pmu) p0 = pmu->Read();
        const std::uint64_t a = __rdtsc();
        if (exact != nullptr) {
            tw::collide::ShapePose sp[8];
            tw::collide::ShapePoses(exact->Car(), s.pos, s.quat, s.absorb, sp);
            exact->DetectShapesRef(sp, out);
        } else {
            inner->DetectCached(s, cache, out);
        }
        if (pmu) {
            const Pmu::Sample d = Pmu::Delta(p0, pmu->Read());
            pmuCycles += d.cycles;
            pmuIns += d.instructions;
        }
        tsc += __rdtsc() - a;
        ++calls;
        contacts += static_cast<std::uint64_t>(out.n);
        if (dump != nullptr)
            if (const auto *ec = dynamic_cast<const tw::collide::ExactCollision *>(inner)) {
                tw::collide::ShapePose sp[8];
                tw::collide::ShapePoses(ec->Car(), s.pos, s.quat, s.absorb, sp);
                std::fwrite(sp, sizeof sp, 1, dump);
                if (cdump != nullptr) {   // the car pose Detect read (pos, quat, absorb): the double reference's input
                    std::fwrite(s.pos, sizeof(float), 3, cdump);
                    std::fwrite(s.quat, sizeof(float), 4, cdump);
                    std::fwrite(s.absorb, sizeof(float), 4, cdump);
                }
            }
        static const bool statsPass = std::getenv("TWB_STATS") != nullptr;
        if (!statsPass) return;
        if (const auto *ec = dynamic_cast<const tw::collide::ExactCollision *>(inner)) {   // stats pass (TWB_STATS; pollutes the caches)
            tw::collide::ShapePose sp[8];
            tw::collide::ShapePoses(ec->Car(), s.pos, s.quat, s.absorb, sp);
            tw::step::SubstepContacts tmp;
            ec->DetectShapes(sp, tmp, &st);
        }
    }
};

void RunReplay(const std::string &packDir, const std::string &path, std::uint32_t maxTicks, double ghz) {  // maxTicks: capped to the replay
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

    // the race span: from the hand-off to the end of the replay, capped
    auto before = sb.ReadState();
    if (before && before.Value().durationMs > before.Value().timeMs)
        maxTicks = std::min<std::uint32_t>(maxTicks, static_cast<std::uint32_t>((before.Value().durationMs - before.Value().timeMs) / 10u));
    if (std::getenv("TWB_AB")) {   // A/B: spec1n0b1 only, current path vs the reference path, alternating, min of N each
        const int reps = std::max(1, std::atoi(std::getenv("TWB_AB")));
        tw::step::PolicyConfig specAb;
        auto world = tx::CompileTwinWorld(sb, fv::TwinPolicy::ContactSubstepsFull, fv::TwinCollision::Exact);
        if (!world || !tw::step::ParsePolicy(std::getenv("TWB_POLICY") ? std::getenv("TWB_POLICY") : "spec1n0b1", &specAb)) { std::printf("%s: compile failed\n", name.c_str()); return; }
        const tx::TwinWorld &w = *world.Value();
        auto imported = tx::TwinImportState(w, handoff.Value());
        auto stream = tx::TwinControlsFromEvents(w, inputs.Value(), 2610u, maxTicks);
        if (!imported || !stream) { std::printf("%s: import/controls failed\n", name.c_str()); return; }
        double best[2] = {1e30, 1e30}, insPmu[2] = {0, 0};
        // TWB_ALT="key=value,...": compare against the same stage built with other CollideOptions instead of the
        // M3 reference path (keys: lead, margin, compact, lanesetup, checksetup)
        std::unique_ptr<tw::collide::ExactCollision> alt;
        if (const char *spec = std::getenv("TWB_ALT")) {
            tw::collide::CollideOptions o = static_cast<const tw::collide::ExactCollision *>(w.Internal().model.collision)->Options();
            std::string sp(spec);
            for (std::size_t p = 0; p < sp.size();) {
                std::size_t q = sp.find(',', p);
                if (q == std::string::npos) q = sp.size();
                const std::string kv = sp.substr(p, q - p);
                const std::size_t eq = kv.find('=');
                const std::string k = kv.substr(0, eq);
                const double val = eq == std::string::npos ? 1.0 : std::atof(kv.c_str() + eq + 1);
                if (k == "lead") o.leadPoses = static_cast<float>(val);
                else if (k == "margin") o.unionMargin = static_cast<float>(val);
                else if (k == "compact") o.compactMemo = val != 0.0;
                else if (k == "lanesetup") o.laneSetup = val != 0.0;
                else if (k == "r7") o.closedWheelFace = val != 0.0;
                else if (k == "checksetup") o.checkSetup = o.checkFrames = val != 0.0;
                else std::printf("TWB_ALT: unknown key %s\n", k.c_str());
                p = q + 1;
            }
            alt = std::make_unique<tw::collide::ExactCollision>();
            std::string err;
            const auto &pk = *w.Internal().packs;
            if (!alt->Init(pk.view, pk.shapes, &err, o)) { std::printf("TWB_ALT: %s\n", err.c_str()); return; }
        }
        // TWB_CALT=NAME: the B side is architect C's stage with that FOREVERVALIDATOR_TWIN_COLLIDE name (e.g. c-legacy),
        // over the same world, so two C pipelines compare in one binary (the world's stage: FOREVERVALIDATOR_TWIN_COLLIDE)
        std::unique_ptr<tw::collide_c::CcCollision> calt;
        if (const char *name = std::getenv("TWB_CALT")) {
            tw::collide_c::StageOptions o;
            if (!tw::collide_c::ParseStageSwitch(name, &o)) { std::printf("TWB_CALT: unknown stage %s\n", name); return; }
            calt = std::make_unique<tw::collide_c::CcCollision>();
            std::string err;
            const auto &pk = *w.Internal().packs;
            if (!calt->Init(pk.view, pk.shapes, &err, o)) { std::printf("TWB_CALT: %s\n", err.c_str()); return; }
        }
        std::uint32_t nTicks = 0;
        double callsPerTick = 0, samePerTick = 0;
        // 2 reps timed runs (PMU core cycles of the run x the TSC share of the collision calls), then one run per
        // mode counting the calls' instructions with the PMU (rdpmc per call traps in a VM: those runs are not timed)
        for (int r = 0; r < 2 * reps + 2; ++r) {
            const int mode = r & 1;   // 0 current, 1 reference
            const bool pmuRun = r >= 2 * reps;
            tw::step::Model model = w.Internal().model;
            model.policyConfig = &specAb;
            TimedCollision timed;
            timed.inner = model.collision;
            timed.ref = mode == 1 && alt == nullptr;
            if (mode == 1 && alt != nullptr) timed.inner = alt.get();   // TWB_ALT: the B side is the stage with other options
            if (mode == 1 && calt != nullptr) { timed.inner = calt.get(); timed.ref = false; }
            Pmu pmu;
            if (pmuRun && pmu.ok()) timed.pmu = &pmu;
            model.collision = &timed;
            std::unique_ptr<tw::step::CollisionCache> cache = std::getenv("TWB_NOCACHE") ? nullptr : timed.NewCache();
            tw::model::TwinState st = tx::internal::ToModel(imported.Value());
            tw::step::Events ev;
            // core cycles of the whole run from the PMU (two reads), split by the TSC share spent in collision
            const bool wholeOk = !pmuRun && pmu.ok();
            const Pmu::Sample w0 = pmu.Read();
            const std::uint64_t r0 = __rdtsc();
            nTicks = tw::step::StepTicks(model, st, stream.Value().ticks.data(), stream.Value().ticks.size(), tw::step::StopRule{}, &ev, nullptr,
                                         cache.get());
            const std::uint64_t r1 = __rdtsc();
            const Pmu::Sample w1 = pmu.Read();
            if (wholeOk && !pmu.FullyScheduled()) std::printf("  warning: PMU counters multiplexed (another perf user): partial counts\n");
            if (wholeOk) {
                const double runCycles = double(Pmu::Delta(w0, w1).cycles);
                best[mode] = std::min(best[mode], runCycles * (double(timed.tsc) / double(r1 - r0)) / std::max<std::uint32_t>(1u, nTicks));
            }
            if (timed.pmu) insPmu[mode] = double(timed.pmuIns) / std::max<std::uint32_t>(1u, nTicks);
            callsPerTick = double(timed.calls) / std::max<std::uint32_t>(1u, nTicks);
            samePerTick = double(timed.samePose) / std::max<std::uint32_t>(1u, nTicks);
        }
        if (std::getenv("TWB_WRAPCHECK")) {   // whole-run PMU cycles per tick with the timing wrapper and with a bare one
            double wrapped = 1e30, bareRun = 1e30, coll = 1e30;
            for (int r = 0; r < 2 * reps; ++r) {
                tw::step::Model model = w.Internal().model;
                model.policyConfig = &specAb;
                TimedCollision timed;
                timed.inner = model.collision;
                timed.ref = false;
                timed.bare = (r & 1) != 0;
                model.collision = &timed;
                Pmu pmu;
                std::unique_ptr<tw::step::CollisionCache> cache = timed.NewCache();
                tw::model::TwinState st = tx::internal::ToModel(imported.Value());
                tw::step::Events ev;
                const Pmu::Sample w0 = pmu.Read();
                const std::uint64_t r0 = __rdtsc();
                const std::uint32_t nt = tw::step::StepTicks(model, st, stream.Value().ticks.data(), stream.Value().ticks.size(), tw::step::StopRule{}, &ev,
                                                             nullptr, cache.get());
                const std::uint64_t r1 = __rdtsc();
                const double cyc = double(Pmu::Delta(w0, pmu.Read()).cycles) / std::max<std::uint32_t>(1u, nt);
                if (timed.bare) bareRun = std::min(bareRun, cyc);
                else { wrapped = std::min(wrapped, cyc); coll = std::min(coll, cyc * double(timed.tsc) / double(r1 - r0)); }
            }
            std::printf("%s wrapper check: tick %.0f cycles with the timing wrapper, %.0f bare (wrapper %.0f per tick); collision as measured %.0f\n",
                        name.c_str(), wrapped, bareRun, wrapped - bareRun, coll);
        }
        if (alt && (alt->Options().checkSetup || alt->Options().checkFrames))
            std::printf("%s alt check: setup mismatches %llu, frame mismatches %llu\n", name.c_str(), static_cast<unsigned long long>(alt->SetupMismatches()),
                        static_cast<unsigned long long>(alt->FrameMismatches()));
        if (calt) {
            std::printf("%s [%s A/B, min of %d]: %u ticks, %.2f calls/tick; collision cycles/tick: current %.0f, %s %.0f (%.2fx); instructions/tick: current %.0f, %s %.0f\n",
                        name.c_str(), std::getenv("TWB_POLICY") ? std::getenv("TWB_POLICY") : "spec1n0b1", reps, nTicks, callsPerTick, best[0], std::getenv("TWB_CALT"),
                        best[1], best[1] / best[0], insPmu[0], std::getenv("TWB_CALT"), insPmu[1]);
            std::fflush(stdout);
            return;
        }
        std::printf(alt ? "%s [%s A/B, min of %d]: %u ticks, %.2f calls/tick (%.2f repeat the previous pose); collision cycles/tick: current %.0f, "
                          "alt %.0f (%.2fx); instructions/tick: current %.0f, alt %.0f\n"
                        : "%s [%s A/B, min of %d]: %u ticks, %.2f calls/tick (%.2f repeat the previous pose); collision cycles/tick: current %.0f, "
                          "reference %.0f (%.2fx); instructions/tick: current %.0f, reference %.0f\n",
                    name.c_str(), std::getenv("TWB_POLICY") ? std::getenv("TWB_POLICY") : "spec1n0b1", reps, nTicks, callsPerTick, samePerTick, best[0],
                    best[1], best[1] / best[0], insPmu[0], insPmu[1]);
        std::fflush(stdout);
        return;
    }
    // FV's own ticks over the same span (from the hand-off)
    const auto f0 = Clock::now();
    auto adv = sb.AdvanceTicks(maxTicks);
    const double fvNs = std::chrono::duration<double, std::nano>(Clock::now() - f0).count();
    const std::uint64_t fvTicks = (adv && before) ? adv.Value().tick - before.Value().tick : 0u;
    if (!adv) std::printf("%s: FV AdvanceTicks: %s\n", name.c_str(), adv.Error().diagnostic.c_str());

    const fv::TwinPolicy policies[2] = {fv::TwinPolicy::FvSubsteps, fv::TwinPolicy::ContactSubstepsFull};
    const char *pname[2] = {"fv", "spec1n0b1"};
    tw::step::PolicyConfig spec;
    if (!tw::step::ParsePolicy("spec1n0b1", &spec)) { std::printf("spec1n0b1: parse failed\n"); return; }
    for (int pi = 0; pi < 2; ++pi) {
        const auto c0 = Clock::now();
        auto world = tx::CompileTwinWorld(sb, policies[pi], fv::TwinCollision::Exact);
        const double compileMs = std::chrono::duration<double, std::milli>(Clock::now() - c0).count();
        if (!world) { std::printf("%s: CompileTwinWorld failed: %s\n", name.c_str(), world.Error().diagnostic.c_str()); return; }
        const tx::TwinWorld &w = *world.Value();
        auto imported = tx::TwinImportState(w, handoff.Value());
        if (!imported) { std::printf("%s: import failed: %s\n", name.c_str(), imported.Error().diagnostic.c_str()); return; }
        auto stream = tx::TwinControlsFromEvents(w, inputs.Value(), 2610u, maxTicks);
        if (!stream) { std::printf("%s: controls failed\n", name.c_str()); return; }
        tw::step::Model model = w.Internal().model;
        if (pi == 1) model.policyConfig = &spec;   // explicit: the policy grammar's spec1n0b1
        TimedCollision timed;
        timed.inner = model.collision;
        model.collision = &timed;
        tw::model::TwinState s = tx::internal::ToModel(imported.Value());
        // the candidate's collision cache (TWB_NOCACHE: none, the stage's per-thread memo)
        std::unique_ptr<tw::step::CollisionCache> cache = std::getenv("TWB_NOCACHE") ? nullptr : timed.NewCache();
        if (std::getenv("TWB_REPEAT")) {   // profiling: run the stepping N more times first (untimed)
            for (int r = 0; r < std::atoi(std::getenv("TWB_REPEAT")); ++r) {
                tw::model::TwinState s2 = s;
                tw::step::Events e2;
                tw::step::StepTicks(model, s2, stream.Value().ticks.data(), stream.Value().ticks.size(), tw::step::StopRule{}, &e2, nullptr,
                                    cache.get());
            }
            timed.tsc = timed.calls = timed.contacts = 0;
        }
        tw::step::Events ev;
        if (pi == 1 && std::getenv("TWB_DUMP")) {   // the engine's pose stream and its own world pack
            const std::string base = std::string(std::getenv("TWB_DUMP")) + "/" + name.substr(0, name.find('.'));
            timed.dump = std::fopen((base + ".poses").c_str(), "wb");
            timed.cdump = std::fopen((base + ".cposes").c_str(), "wb");
            if (w.Internal().packs)
                if (std::FILE *cf = std::fopen((base + ".carshapes").c_str(), "wb")) {
                    std::fwrite(&w.Internal().packs->shapes, sizeof w.Internal().packs->shapes, 1, cf);
                    std::fclose(cf);
                }
            std::string err;
            if (w.Internal().packs && !tw::world::SaveWorldPack((base + ".twpack").c_str(), w.Internal().packs->pack, &err))
                std::printf("%s: pack save failed: %s\n", name.c_str(), err.c_str());
        }
        const std::uint64_t r0 = __rdtsc();
        const auto t0 = Clock::now();
        const std::uint32_t n = tw::step::StepTicks(model, s, stream.Value().ticks.data(), stream.Value().ticks.size(), tw::step::StopRule{},
                                                    &ev, nullptr, cache.get());
        const double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
        if (timed.dump) { std::fclose(timed.dump); timed.dump = nullptr; }
        if (timed.cdump) { std::fclose(timed.cdump); timed.cdump = nullptr; }
        const double tscPerNs = double(__rdtsc() - r0) / ns;
        const double cycPerTick = ns * ghz / std::max<std::uint32_t>(1u, n);
        const double collCycPerTick = double(timed.tsc) / tscPerNs * ghz / std::max<std::uint32_t>(1u, n);
        const float *v = s.vel;
        std::printf("%s [%s]: end speed %.0f km/h; %u ticks (%s%s), %.2f substeps/tick, %.0f cycles/tick all-twin (collision %.0f = %.0f%%, %.2f calls/tick, "
                    "%.2f contacts/call), %.0f ticks/s; world compile %.0f ms\n",
                    name.c_str(), pname[pi], 3.6 * std::sqrt(double(v[0]) * v[0] + double(v[1]) * v[1] + double(v[2]) * v[2]), n, tw::step::StatusName(ev.status), ev.finished ? ", finished" : "",
                    double(ev.substeps) / std::max<std::uint32_t>(1u, n), cycPerTick, collCycPerTick, 100.0 * collCycPerTick / cycPerTick,
                    double(timed.calls) / std::max<std::uint32_t>(1u, n), timed.calls ? double(timed.contacts) / timed.calls : 0.0,
                    1e9 / (ns / std::max<std::uint32_t>(1u, n)), compileMs);
        std::printf("    collision per call: %.1f candidates, %.1f triangle tests, %.2f raw contacts; end pos (%.1f %.1f %.1f)\n",
                    double(timed.st.candidates) / std::max<std::uint64_t>(1, timed.st.calls), double(timed.st.triangleTests) / std::max<std::uint64_t>(1, timed.st.calls),
                    double(timed.st.rawContacts) / std::max<std::uint64_t>(1, timed.st.calls), s.pos[0], s.pos[1], s.pos[2]);
    }
    std::printf("%s [FV OptimizedCpu]: %llu ticks advanced, %.0f cycles/tick\n", name.c_str(), static_cast<unsigned long long>(fvTicks),
                fvNs * ghz / double(fvTicks ? fvTicks : 1u));
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: forevervalidator-twin-step-bench PACKS REPLAY...\n"); return 2; }
    const std::uint32_t maxTicks = std::getenv("TWB_TICKS") ? static_cast<std::uint32_t>(std::atoi(std::getenv("TWB_TICKS"))) : 4000u;
    const double ghz = CoreGHz();
    std::printf("core clock %.2f GHz (measured)\n", ghz);
    for (int i = 2; i < argc; ++i) RunReplay(argv[1], argv[i], maxTicks, ghz);
    return 0;
}
