// Twin session creation from an FV-loaded replay, up to the first twin tick.
//
//   forevervalidator-twin-handoff-tests PACKS REPLAY...
//
// Per replay (FV optimized_cpu sandbox; LoadReplay runs FV's prestart through
// T = 2600, the twin hand-off):
//  1. CompileTwinWorld compiles the world package, car pack, water and race
//     map (fv/twin_world_build.cpp); TwRaceMapValid holds;
//  2. the car pack's fs/lds pairs equal M1's TwCarPack from the StadiumCar
//     JSON (catches drift between the two car compiles);
//  3. TwinImportState: the imported state equals FV's state at the hand-off.
//     ImportGolden's refusals are hard errors; then every word ImportGolden
//     reads round-trips: ExportGolden(imported) == FV's state words
//     (w.peer_z: z only, as in M1). tickMs = 2600;
//  4. TwinStep on the first twin tick (T = 2610, controls from
//     TwinControlsFromEvents) runs all-twin (M3 collision): Ok, tickMs = 2610;
//  5. with a no-contact collision stand-in, one tick runs on the imported
//     state: Ok, tickMs = 2610, the car moved (forces, integration, race
//     layer and response all ran).
// Needs the game packs and replays, so it is not a quick test.
#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/experimental/twin.h>
#include <forevervalidator/native.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "stages.h"
#include "src/twin/common/twin_fv_state.h"
#include "tw_compile_car.h"
#include "tw_race.h"
#include "tw_state_golden.h"
#include "tw_step.h"
#include "twin_api_internal.h"
#include "twin_world_packs.h"
#include "validation/api/physics_sandbox_twin_test_access.h"

#ifndef TW_STADIUMCAR_JSON
#define TW_STADIUMCAR_JSON "src/twin/model/stadiumcar-values.json"
#endif

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

struct NoContacts : tw::step::CollisionStage {
    void Detect(const tw::model::TwinState &, tw::step::SubstepContacts &out) const override {
        out.n = 0;
        out.overflow = false;
    }
};

std::uint32_t Bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

void RunReplay(const std::string &packDir, const std::string &path, const tw::model::TwCarPack &json) {
    const std::string name = path.substr(path.find_last_of('/') + 1);
    auto bytes = fv::ReadNativeReplayFile(path, fv::ReplayIdentity{path});
    auto source = fv::OpenInstalledPackDirectory(packDir);
    if (!bytes || !source) {
        Check(false, name + ": open");
        return;
    }
    fvx::PhysicsSandboxOptions options;
    options.backend = fv::SimulationBackend::OptimizedCpu;
    auto created = fvx::CreatePhysicsSandbox(std::move(source).Value(), options);
    if (!created) {
        Check(false, name + ": sandbox");
        return;
    }
    fvx::PhysicsSandbox sb = std::move(created).Value();
    auto loaded = sb.LoadReplay({bytes.Value().data(), bytes.Value().size()}, fv::ReplayIdentity{path});
    if (!loaded) {
        std::printf("  skip %s: FV load failed (%s)\n", name.c_str(), loaded.Error().diagnostic.c_str());
        return;
    }
    auto state = sb.CaptureState();
    if (!state) {
        Check(false, name + ": capture");
        return;
    }
    const auto clone = fvx::twin_test::PhysicsSandboxTwinTestAccess::RuntimeClone(state.Value());
    if (!clone) {
        Check(false, name + ": runtime clone");
        return;
    }

    // 1. world
    auto world = tx::CompileTwinWorld(sb);
    Check(world.HasValue(), name + ": CompileTwinWorld: " + (world ? "" : world.Error().diagnostic));
    if (!world) return;
    const tx::TwinWorld &w = *world.Value();
    const auto &packs = w.Internal().packs;
    Check(packs != nullptr && w.Info().worldPackReady && w.Info().carPackReady, name + ": packs ready");
    if (!packs) return;
    Check(tw::race::TwRaceMapValid(packs->race), name + ": TwRaceMapValid");
    Check(w.Internal().model.car == &packs->car && w.Internal().model.race == &packs->race &&
                  w.Internal().model.collision == packs->Stage(),
          name + ": tick model points into the packs, collision included");
    // The product default is C's noise stage ("c"); Exact selects B's stage ("b", the reference).
    const char *envStage = std::getenv("FOREVERVALIDATOR_TWIN_COLLIDE");
    if (envStage == nullptr || *envStage == '\0') {
        Check(w.Info().collisionStage == "c" && w.Internal().model.collision == &packs->collisionC,
              name + ": default collision stage is C's noise stage");
        auto exact = tx::CompileTwinWorld(sb, fv::TwinPolicy::FvSubsteps, fv::TwinCollision::Exact);
        Check(exact && exact.Value()->Info().collisionStage == "b" &&
                      exact.Value()->Internal().model.collision == &exact.Value()->Internal().packs->collision,
              name + ": TwinCollision::Exact is B's stage");
    }

    // 2. car scales vs M1's JSON pack
    Check(packs->golden.fsGroundedBits == json.fsGroundedBits && packs->golden.ldsGroundedBits == json.ldsGroundedBits &&
                  packs->golden.fsAirborneBits == json.fsAirborneBits &&
                  packs->golden.ldsAirborneBits == json.ldsAirborneBits,
          name + ": fs/lds pairs = TwCarPack (JSON)");

    // 3. import at the hand-off
    auto imported = tx::TwinImportState(w, state.Value());
    Check(imported.HasValue(), name + ": TwinImportState: " + (imported ? "" : imported.Error().diagnostic));
    if (!imported) return;
    tw::model::TwinState s = tx::internal::ToModel(imported.Value());
    Check(s.tickMs == 2600u, name + ": hand-off tick " + std::to_string(s.tickMs));
    std::vector<std::uint32_t> fvWords(twin_fv::StateWordCount());
    twin_fv::ExtractState(*clone, fvWords.data());
    std::map<std::string, std::pair<std::uint32_t, std::uint32_t>> at;
    std::uint32_t off = 0u;
    for (const auto &f : twin_fv::StateFields()) {
        at[f.name] = {off, f.count};
        off += f.count;
    }
    std::uint32_t compared = 0u, words = 0u;
    std::string differ;
    tw::model::ExportGolden(s, packs->golden, [&](const char *field, const std::uint32_t *v, std::uint32_t count) {
        if (std::strncmp(field, "ctl.", 4) == 0) return;  // the tick's input echo, not state
        const auto it = at.find(field);
        if (it == at.end()) {
            differ += std::string(" ") + field + "(absent)";
            return;
        }
        if (it->second.second != count) {
            differ += std::string(" ") + field + "(count)";
            return;
        }
        const bool peerZ = std::strcmp(field, "w.peer_z") == 0;
        for (std::uint32_t k = 0; k < count; ++k) {
            if (peerZ && k % 3u != 2u) continue;   // x/y never read by FV physics (M1)
            ++words;
            if (fvWords[it->second.first + k] != v[k]) {
                differ += std::string(" ") + field + "[" + std::to_string(k) + "]";
                break;
            }
        }
        ++compared;
    });
    Check(differ.empty(), name + ": imported state == FV at the hand-off (" + std::to_string(compared) + " fields, " +
                                  std::to_string(words) + " words); differ:" + differ);

    // 4. first twin tick, all-twin (M3 collision stage)
    auto inputs = sb.ReadInputs();
    Check(inputs.HasValue(), name + ": inputs");
    if (!inputs) return;
    auto stream = tx::TwinControlsFromEvents(w, inputs.Value(), 10u, 262u);
    Check(stream.HasValue() && stream.Value().ticks.size() == 262u, name + ": control stream from T=10");
    if (!stream) return;
    const tx::TwinControl first = stream.Value().ticks[260];   // tick T = 2610
    tx::TwinState t = imported.Value();
    tx::TwinEvents ev;
    const tx::TwinStatus st = tx::TwinStep(w, t, first, &ev);
    const tw::model::TwinState tm = tx::internal::ToModel(t);
    Check(st == tx::TwinStatus::Ok && tm.tickMs == 2610u,
          name + ": TwinStep T=2610 -> " + tx::TwinStatusName(st) + ", tickMs " + std::to_string(tm.tickMs));

    // 5. the same tick with a no-contact collision stand-in
    tw::step::Model model = w.Internal().model;
    NoContacts none;
    model.collision = &none;
    model.policy = tw::step::Policy::FvSubsteps;
    tw::model::TwinState u = s;
    tw::step::Events sev;
    const tw::step::Status ss = tw::step::Step(model, u, first, &sev);
    const bool moved = Bits(u.pos[0]) != Bits(s.pos[0]) || Bits(u.pos[1]) != Bits(s.pos[1]) || Bits(u.pos[2]) != Bits(s.pos[2]);
    Check(ss == tw::step::Status::Ok && u.tickMs == 2610u && moved && sev.substeps >= 1u,
          name + ": one tick with no contacts: " + tw::step::StatusName(ss) + ", T=" + std::to_string(u.tickMs) + ", " +
                  std::to_string(sev.substeps) + " substeps, moved " + (moved ? "yes" : "no"));
    std::printf("%s: %zu tris, %zu triggers, %zu spawns, laps %u; import ok (%u fields); T=2610 %s; stand-in tick %s, %u substeps\n",
                name.c_str(), packs->view.tris.size(), packs->view.triggers.size(), packs->view.spawns.size(),
                packs->race.requiredLaps, compared, tx::TwinStatusName(st), tw::step::StatusName(ss), sev.substeps);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s PACKS REPLAY...\n", argv[0]);
        return 2;
    }
    tw::model::TwCarPack json;
    std::string err;
    if (!tw::model::CompileCarFromJson(TW_STADIUMCAR_JSON, &json, &err)) {
        std::fprintf(stderr, "car json: %s\n", err.c_str());
        return 2;
    }
    for (int i = 2; i < argc; ++i) RunReplay(argv[1], argv[i], json);
    std::printf("%d passed, %d failed\n%s\n", g_pass, g_fail, g_fail ? "RESULT: FAIL" : "RESULT: PASS");
    return g_fail ? 1 : 0;
}
