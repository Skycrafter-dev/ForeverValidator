// The per-candidate collision cache is result-invariant (lead decision 2026-11-08).
//
//   forevervalidator-twin-collide-cache-tests PACKS REPLAY...     env TWC_TICKS=N caps the ticks (default 3000)
//
// Per replay, from FV's hand-off state (T = 2600), for policies fv and spec1n0b1,
// the twin runs tick by tick (TwinStep::Step) four times over the replay's controls:
//   A  no cache (null: the stage's per-thread memo);
//   B  its own cache (model.collision->NewCache()), kept across ticks;
//   C  its own cache with flushEveryCall (the cache is cold at every detection);
//   D  a shared stale cache: before every tick the same cache also serves a
//      detection at the pose of 25 ticks earlier, so B's working set is never
//      the one the tick left behind.
// Every tick's state must be byte-identical across A-D, and so must the
// contacts of a detection at the post-tick pose. The run stops at the finish
// or at the cap. Needs the game packs and replays (CTest label twin;assets).
#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/experimental/twin.h>
#include <forevervalidator/native.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "policy.h"
#include "stages.h"
#include "tw_collide.h"
#include "tw_step.h"
#include "twin_api_internal.h"

namespace fv = forevervalidator;
namespace fvx = forevervalidator::experimental;
namespace tx = forevervalidator::experimental::twin;

namespace {

int g_pass = 0, g_fail = 0;
void Check(bool ok, const std::string &what) {
    if (ok) {
        ++g_pass;
        return;
    }
    ++g_fail;
    std::printf("  FAIL %s\n", what.c_str());
}

bool SameContacts(const tw::step::SubstepContacts &a, const tw::step::SubstepContacts &b) {
    return a.n == b.n && a.overflow == b.overflow && std::memcmp(a.c, b.c, sizeof(a.c[0]) * static_cast<std::size_t>(a.n)) == 0;
}

void RunReplay(const std::string &packDir, const std::string &path, std::uint32_t maxTicks) {
    const std::string name = path.substr(path.find_last_of('/') + 1);
    std::printf("%s\n", name.c_str());
    auto bytes = fv::ReadNativeReplayFile(path, fv::ReplayIdentity{path});
    auto source = fv::OpenInstalledPackDirectory(packDir);
    Check(bytes && source, name + ": open");
    if (!bytes || !source) return;
    fvx::PhysicsSandboxOptions options;
    options.backend = fv::SimulationBackend::OptimizedCpu;
    auto created = fvx::CreatePhysicsSandbox(std::move(source).Value(), options);
    Check(static_cast<bool>(created), name + ": sandbox");
    if (!created) return;
    fvx::PhysicsSandbox sb = std::move(created).Value();
    auto loaded = sb.LoadReplay({bytes.Value().data(), bytes.Value().size()}, fv::ReplayIdentity{path});
    if (!loaded) {
        std::printf("  skip (%s)\n", loaded.Error().diagnostic.c_str());
        return;
    }
    auto handoff = sb.CaptureState();
    auto inputs = sb.ReadInputs();
    Check(handoff && inputs, name + ": capture / inputs");
    if (!handoff || !inputs) return;
    auto before = sb.ReadState();
    if (before && before.Value().durationMs > before.Value().timeMs)
        maxTicks = std::min<std::uint32_t>(maxTicks, static_cast<std::uint32_t>((before.Value().durationMs - before.Value().timeMs) / 10u));

    tw::step::PolicyConfig spec;
    Check(tw::step::ParsePolicy("spec1n0b1", &spec), "spec1n0b1 parses");
    const fv::TwinPolicy policies[2] = {fv::TwinPolicy::FvSubsteps, fv::TwinPolicy::ContactSubstepsFull};
    const char *pname[2] = {"fv", "spec1n0b1"};
    for (int pi = 0; pi < 2; ++pi) {
        auto world = tx::CompileTwinWorld(sb, policies[pi], fv::TwinCollision::Exact);
        Check(static_cast<bool>(world), name + ": CompileTwinWorld");
        if (!world) return;
        const tx::TwinWorld &w = *world.Value();
        auto imported = tx::TwinImportState(w, handoff.Value());
        auto stream = tx::TwinControlsFromEvents(w, inputs.Value(), 2610u, maxTicks);
        Check(imported && stream, name + ": import / controls");
        if (!imported || !stream) return;
        tw::step::Model model = w.Internal().model;
        if (pi == 1) model.policyConfig = &spec;
        const tw::step::CollisionStage &coll = *model.collision;
        std::unique_ptr<tw::step::CollisionCache> cb = coll.NewCache(), cc = coll.NewCache(), cd = coll.NewCache();
        Check(cb && cc && cd, name + ": the stage makes caches");
        if (!cb || !cc || !cd) return;
        cc->flushEveryCall = true;
        tw::step::CollisionCache *caches[4] = {nullptr, cb.get(), cc.get(), cd.get()};
        tw::model::TwinState s[4];
        for (auto &x : s) x = tx::internal::ToModel(imported.Value());
        std::vector<tw::model::TwinState> past;   // D's stale poses (run A's states)
        const auto &ticks = stream.Value().ticks;
        std::uint32_t ticksRun = 0, stateDiffs = 0, contactDiffs = 0, firstDiff = ~0u;
        bool finished = false;
        tw::step::SubstepContacts scratch, ref, got;
        for (std::size_t i = 0; i < ticks.size() && !finished; ++i) {
            if (past.size() >= 25u) coll.DetectCached(past[past.size() - 25u], cd.get(), scratch);
            tw::step::Status st[4];
            tw::step::Events ev[4];
            for (int r = 0; r < 4; ++r) st[r] = tw::step::Step(model, s[r], ticks[i], &ev[r], caches[r]);
            ++ticksRun;
            coll.DetectCached(s[0], nullptr, ref);
            for (int r = 1; r < 4; ++r) {
                const bool same = st[r] == st[0] && std::memcmp(&s[r], &s[0], sizeof s[0]) == 0 && ev[r].finished == ev[0].finished;
                if (!same) {
                    ++stateDiffs;
                    if (firstDiff == ~0u) firstDiff = static_cast<std::uint32_t>(i);
                }
                coll.DetectCached(s[r], caches[r], got);
                if (!SameContacts(got, ref)) ++contactDiffs;
            }
            if (st[0] != tw::step::Status::Ok) break;
            finished = ev[0].finished;
            past.push_back(s[0]);
            if (stateDiffs) break;   // the runs have split: later ticks compare different states
        }
        std::printf("  [%s] %u ticks%s: state differences %u (first at tick %d), contact differences %u\n", pname[pi], ticksRun,
                    finished ? " (finished)" : "", stateDiffs, firstDiff == ~0u ? -1 : static_cast<int>(firstDiff), contactDiffs);
        for (int r = 1; r < 4; ++r)   // the caches were exercised: B reuses, C refills at every call, D refills often
            if (const auto *ec = dynamic_cast<const tw::collide::ExactCache *>(caches[r]))
                std::printf("    cache %c: %llu refills, %llu reuses, %llu last-pose hits\n", 'A' + r, static_cast<unsigned long long>(ec->refills),
                            static_cast<unsigned long long>(ec->reuses), static_cast<unsigned long long>(ec->lastHits));
        Check(ticksRun > 0u, name + " [" + pname[pi] + "]: ticks ran");
        Check(stateDiffs == 0u, name + " [" + pname[pi] + "]: every tick's state equal with and without the cache");
        Check(contactDiffs == 0u, name + " [" + pname[pi] + "]: post-tick contacts equal with and without the cache");
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s PACKS REPLAY...\n", argv[0]);
        return 2;
    }
    const std::uint32_t maxTicks = std::getenv("TWC_TICKS") ? static_cast<std::uint32_t>(std::atoi(std::getenv("TWC_TICKS"))) : 3000u;
    for (int i = 2; i < argc; ++i) RunReplay(argv[1], argv[i], maxTicks);
    std::printf("%d passed, %d failed\n%s\n", g_pass, g_fail, g_fail ? "RESULT: FAIL" : "RESULT: PASS");
    return g_fail ? 1 : 0;
}
