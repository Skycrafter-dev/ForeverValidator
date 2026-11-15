// Twin collision stage choice: one name -> which stage and its options.
//
// Used by the world build (src/twin/fv/twin_world_build.cpp) and by every ISA
// variant's tick entry (tw_isa_tick_entry.cpp), so both build the same stage.
// Header-only on purpose: in a variant the noise hook below is that variant's
// own copy (isolate_variant.sh renames it).
//
// Names (docs/twin/scoreboard.md, "Verdict" and "Headline"):
//   c          architect C's stage, noise-level narrow phase. The PRODUCT
//              DEFAULT (TwinCollision::Noise): passes the acceptance test with
//              the fv policy and is the fastest stage (x3.77 on 100-tick windows).
//   b          B's ExactCollision: FV's contacts bit for bit. The REFERENCE
//              (TwinCollision::Exact).
//   Experiments (selected only through FOREVERVALIDATOR_TWIN_COLLIDE):
//   b-noise    B's stage with C's noise-level narrow phase
//   b-patch    B's stage with merged coplanar patches (relaxation R1)
//   b-r7       B's stage with closed-form wheel FACE contacts (relaxation R7)
//   c-exact | c-nomerge | c-deepest | c-nobody | c-legacy   C's stage variants (collide_c ParseStageSwitch;
//              c-legacy: c's pipeline before the 2026-11-09 collision round, same bits, for A/B)
//   atlas      architect E's contact-atlas stage (gen2 S1, src/twin/atlas/atlas_stage.h): free / T0 / closed-form
//              FACE per shape, B's exact stage (later D's per-plane fallback) for the rest
//
// FOREVERVALIDATOR_TWIN_COLLIDE, when set and not empty, overrides the API's
// choice for the whole process (tests, benches, the scoreboard). An unknown
// value fails the world build.
#ifndef TW_ISA_COLLISION_CHOICE_H
#define TW_ISA_COLLISION_CHOICE_H

#include <cstdlib>
#include <string>

#include "cc_stage.h"
#include "tw_collide.h"

namespace tw::collision_choice {

constexpr const char *kEnv = "FOREVERVALIDATOR_TWIN_COLLIDE";
constexpr const char *kNoise = "c";   // product default
constexpr const char *kExact = "b";   // reference

struct Choice {
    std::string name;                  // canonical name (above)
    bool useC = false;                 // C's stage (else B's ExactCollision)
    bool useAtlas = false;             // E's atlas stage (gen2 S1)
    collide_c::StageOptions c{};       // when useC
    collide::CollideOptions b{};       // when !useC (b-noise: the narrow hook)
};

// The effective name: the environment's value when set and not empty, else `apiName`.
inline std::string Effective(const char *apiName) {
    const char *env = std::getenv(kEnv);
    return (env != nullptr && *env != '\0') ? std::string(env) : std::string(apiName);
}

// Resolves `name` (canonical, see above) onto `base` options. False for an unknown name.
inline bool Resolve(const std::string &name, const collide::CollideOptions &base, Choice *out) {
    Choice ch;
    ch.name = name;
    ch.b = base;
    if (name == kExact) {
    } else if (name == "b-noise") {
        ch.b.narrow = &collide_c::noise::Narrow;
    } else if (name == "b-patch") {
        ch.b.patches = collide::kPatchMerged;
    } else if (name == "b-r7") {
        ch.b.closedWheelFace = true;
    } else if (name == "atlas") {
        ch.useAtlas = true;
    } else if (collide_c::ParseStageSwitch(name.c_str(), &ch.c)) {
        ch.useC = true;
    } else {
        return false;
    }
    *out = ch;
    return true;
}

inline std::string UnknownDiagnostic(const std::string &name) {
    return "unknown twin collision stage '" + name + "' (" + kEnv +
           ": c, b, b-noise, b-patch, b-r7, c-exact, c-nomerge, c-deepest, c-nobody, c-legacy, atlas)";
}

}  // namespace tw::collision_choice

#endif
