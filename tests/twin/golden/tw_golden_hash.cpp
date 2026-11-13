// Twin golden hashes (design.md §7, milestone M9). FV-free.
//
//   tw_golden_hash run   components|engine [--policy fv|contact-full|contact|cap3]   print "<key> <hash>" lines
//   tw_golden_hash check components|engine [--goldens FILE]     compare with tests/twin/golden/goldens.txt
//
// Stage "components": the deterministic pieces that exist today, run in this
// build's ISA: state lane round trips (L = 1/4/8/16), the control-stream
// builder (race layer) and the lane math on a fixed grid. Stage "engine":
// the golden-hash suite proper (12 scenarios x 2,000 ticks per policy). It
// needs TwinStep; while the engine is the stub (tw::step::Info()) the stage
// prints not_implemented and exits 77 (CTest SKIP).
//
// Cross-build identity: tests/twin/ci/determinism.sh runs `run` in every
// build of the matrix and diffs the outputs; `check` pins the values to the
// file so that one build alone also detects drift.
//
// goldens.txt lines: "<model-id> <key> <hex hash>"; '#' starts a comment.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "tw_controls.h"
#include "tw_fpenv.h"
#include "tw_isa.h"
#include "tw_math.h"
#include "tw_race.h"
#include "tw_state.h"
#include "tw_step.h"

#ifndef TW_GOLDEN_FILE
#define TW_GOLDEN_FILE "tests/twin/golden/goldens.txt"
#endif

// 1: built without FP contraction (the variants must equal the baseline); 0: FMA build.
#ifndef TW_FP_STRICT
#define TW_FP_STRICT 1
#endif

namespace {

using tw::model::TwinState;

constexpr const char *kComponentsId = "components-1";

struct Fnv {
    std::uint64_t h = 1469598103934665603ull;
    void Byte(unsigned char b) {
        h ^= b;
        h *= 1099511628211ull;
    }
    void U32(std::uint32_t v) {
        for (int i = 0; i < 4; ++i) Byte(static_cast<unsigned char>(v >> (8 * i)));
    }
    void U64(std::uint64_t v) {
        U32(static_cast<std::uint32_t>(v));
        U32(static_cast<std::uint32_t>(v >> 32));
    }
    void F32(float f) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        if ((b & 0x7f800000u) == 0x7f800000u && (b & 0x007fffffu) != 0u) b = 0x7fc00000u;  // canonical NaN
        U32(b);
    }
};

struct Lcg {
    std::uint64_t s;
    explicit Lcg(std::uint64_t seed) : s(seed * 6364136223846793005ull + 1442695040888963407ull) {}
    std::uint32_t Next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(s >> 32);
    }
    float Uniform(float lo, float hi) {  // integer-generated, exact on every build
        return lo + (hi - lo) * (static_cast<float>(Next() >> 8) * (1.0f / 16777216.0f));
    }
};

using Lines = std::vector<std::pair<std::string, std::uint64_t>>;

// ---- components ------------------------------------------------------------------
TwinState SyntheticState(std::uint32_t seed) {
    TwinState s;
    tw::model::TwinStateClear(&s);
    Lcg rng(seed);
    std::size_t n = 0;
    const tw::model::TwFieldInfo *fields = tw::model::TwinFields(&n);
    unsigned char *base = reinterpret_cast<unsigned char *>(&s);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::uint16_t k = 0; k < fields[i].count; ++k) {
            unsigned char *p = base + fields[i].offset + fields[i].elementBytes * k;
            if (fields[i].type == tw::model::TwType::F32) {
                const float v = rng.Uniform(-1000.0f, 1000.0f);
                std::memcpy(p, &v, 4);
            } else {
                const std::uint32_t v = rng.Next();
                std::memcpy(p, &v, fields[i].elementBytes);
            }
        }
    }
    return s;
}

template <int L>
std::uint64_t LaneRoundTrip(const std::vector<TwinState> &states, bool *equal) {
    static tw::model::TwinLanes<L> lanes;
    Fnv h;
    for (std::size_t first = 0; first < states.size(); first += L) {
        for (int lane = 0; lane < L; ++lane) tw::model::TwinLoadLane(&lanes, lane, states[first + lane]);
        for (int lane = 0; lane < L; ++lane) {
            TwinState back;
            tw::model::TwinStoreLane(lanes, lane, &back);
            if (std::memcmp(&back, &states[first + lane], sizeof(TwinState)) != 0) *equal = false;
            h.U64(tw::step::CanonicalHash(back));
        }
    }
    return h.h;
}

std::uint64_t ControlsHash(bool *ok) {
    using namespace tw::race;
    auto sw = [](std::int32_t t, TwAction a, bool press) {
        return TwInputEvent{t, a, TwValueKind::Switch, press ? TwSwitch::Pressed : TwSwitch::Released, 0};
    };
    auto an = [](std::int32_t t, TwAction a, std::int32_t v) {
        return TwInputEvent{t, a, TwValueKind::Analog, TwSwitch::Released, v};
    };
    std::vector<TwInputEvent> e = {sw(0, TwAction::RaceRunning, true), sw(0, TwAction::Accelerate, true)};
    Lcg rng(7);
    for (std::int32_t t = 50; t < 20000; t += 10 + static_cast<std::int32_t>(rng.Next() % 300u)) {
        switch (rng.Next() % 5u) {
        case 0: e.push_back(an(t, TwAction::Steer, static_cast<std::int32_t>(rng.Next() % 131073u) - 65536)); break;
        case 1: e.push_back(sw(t, TwAction::SteerLeft, rng.Next() & 1u)); break;
        case 2: e.push_back(sw(t, TwAction::Brake, rng.Next() & 1u)); break;
        case 3: e.push_back(sw(t, TwAction::Accelerate, rng.Next() & 1u)); break;
        default: e.push_back(sw(t, TwAction::Respawn, rng.Next() & 1u)); break;
        }
    }
    TwTimeline timeline;
    TwControlStream stream;
    std::string error;
    *ok = TwBuildControls(e.data(), e.size(), timeline, 10u, 2260u, &stream, &error);
    if (!*ok) std::fprintf(stderr, "controls: %s\n", error.c_str());
    Fnv h;
    for (const TwinControl c : stream.ticks) h.U32(c);
    return h.h;
}

template <int W>
std::uint64_t MathGrid() {
    using F = tw::math::vf<W>;
    constexpr int kFunctions = 12, kInputs = 4096;
    alignas(64) float x[W], y[W], out[W];
    static float all[kFunctions][kInputs];  // hashed in (function, input) order, independent of W
    Lcg rng(11);
    for (int block = 0; block < kInputs / W; ++block) {
        for (int i = 0; i < W; ++i) {
            x[i] = rng.Uniform(-12.0f, 12.0f);
            y[i] = rng.Uniform(-1.0f, 1.0f);
        }
        const F vx = F::load(x), vy = F::load(y);
        const F results[] = {tw::math::sin(vx), tw::math::cos(vx), tw::math::sincos_small(vy * F(0.785f)).cos,
                             tw::math::asin(vy), tw::math::acos(vy), tw::math::atan2(vy, vx),
                             tw::math::exp(vx), tw::math::sqrt(tw::math::abs(vx)), tw::math::rsqrt(tw::math::abs(vx)),
                             tw::math::floor(vx), tw::math::fmod_pow2(vx * F(100.0f), F(32.0f)),
                             vx * vy + vy / (vx + F(13.0f))};
        static_assert(sizeof(results) / sizeof(results[0]) == kFunctions, "function count");
        for (int f = 0; f < kFunctions; ++f) {
            results[f].store(out);
            for (int i = 0; i < W; ++i) all[f][block * W + i] = out[i];
        }
    }
    Fnv h;
    for (int f = 0; f < kFunctions; ++f)
        for (int i = 0; i < kInputs; ++i) h.F32(all[f][i]);
    return h.h;
}

bool RunComponents(Lines *lines) {
    tw::math::FpGuard guard;
    bool ok = true;
    std::vector<TwinState> states;
    for (std::uint32_t i = 0; i < 64u; ++i) states.push_back(SyntheticState(i + 1u));
    bool equal = true;
    const std::uint64_t l1 = LaneRoundTrip<1>(states, &equal);
    const std::uint64_t l4 = LaneRoundTrip<4>(states, &equal);
    const std::uint64_t l8 = LaneRoundTrip<8>(states, &equal);
    const std::uint64_t l16 = LaneRoundTrip<16>(states, &equal);
    if (!equal || l1 != l4 || l1 != l8 || l1 != l16) {
        std::fprintf(stderr, "components: lane round trip differs across widths\n");
        ok = false;
    }
    lines->push_back({"components/state.lanes", l1});
    bool controlsOk = false;
    lines->push_back({"components/race.controls", ControlsHash(&controlsOk)});
    ok = ok && controlsOk;
    const std::uint64_t m1 = MathGrid<1>(), m4 = MathGrid<4>(), m8 = MathGrid<8>(), m16 = MathGrid<16>();
    if (m1 != m4 || m1 != m8 || m1 != m16) {
        // With FMA contraction (FOREVERVALIDATOR_TWIN_CONTRACT=fast, measurement only) the compiler
        // may contract the scalar and the vector forms differently: reported, not failed.
        std::fprintf(stderr, "components: lane math differs across widths%s\n",
                     TW_FP_STRICT ? "" : " (FMA build: informational)");
        ok = ok && !TW_FP_STRICT;
    }
    lines->push_back({"components/math.grid", m1});
    if (!TW_FP_STRICT) {
        lines->push_back({"components/math.grid.w4", m4});
        lines->push_back({"components/math.grid.w8", m8});
        lines->push_back({"components/math.grid.w16", m16});
    }
    lines->push_back({"components/math.canary", tw::math::contraction_canary() ? 1u : 0u});
    return ok;
}

// ---- engine ------------------------------------------------------------------------
struct Scenario {
    const char *name;
    std::uint32_t seed;
    std::uint32_t ticks;
};

// Placeholder scenario list (12 x 2,000 ticks). When M5 lands, each start
// state and its world/car packs come from FV-free pack files written by the
// FV adapter (design.md §7); the names are kept so the goldens stay keyed.
const Scenario kScenarios[] = {
        {"flat-full-throttle", 1u, 2000u}, {"flat-steer-sweep", 2u, 2000u}, {"brake-to-stop", 3u, 2000u},
        {"drift-left", 4u, 2000u},         {"jump-landing", 5u, 2000u},     {"wall-scrape", 6u, 2000u},
        {"kerb-edge", 7u, 2000u},          {"booster", 8u, 2000u},          {"freewheel", 9u, 2000u},
        {"respawn-chain", 10u, 2000u},     {"checkpoint-finish", 11u, 2000u}, {"high-speed-324kmh", 12u, 2000u},
};

// 0 ok, 77 engine not implemented
int RunEngine(tw::step::Policy policy, Lines *lines) {
    const tw::step::EngineInfo info = tw::step::Info();
    tw::step::Model model;
    model.policy = policy;
    for (const Scenario &sc : kScenarios) {
        TwinState s = SyntheticState(sc.seed);
        std::vector<tw::step::Control> controls(sc.ticks);
        Lcg rng(sc.seed);
        for (auto &c : controls) c = tw::race::TwCtlMake(static_cast<std::int32_t>(rng.Next() % 131073u) - 65536,
                                                         rng.Next() & 1u, (rng.Next() & 7u) == 0u);
        tw::step::Events ev;
        tw::step::StepTicks(model, s, controls.data(), controls.size(), tw::step::StopRule{false}, &ev);
        if (ev.status == tw::step::Status::NotImplemented) {
            std::printf("engine: not_implemented (model %s): %s\n", info.modelId, info.missing);
            return 77;
        }
        if (ev.status != tw::step::Status::Ok) {
            std::fprintf(stderr, "engine %s: %s\n", sc.name, tw::step::StatusName(ev.status));
            return 1;
        }
        lines->push_back({std::string("engine/") + tw::step::PolicyName(policy) + "/" + sc.name,
                          tw::step::CanonicalHash(s)});
    }
    return 0;
}

// Stage "isa": every ISA variant of the tick path (isa/tw_isa.h) built into
// this binary and runnable on this CPU must give the baseline's selftest
// hash (lane math at every width, exact-fast test incl. AVX-512 packets).
// One golden line; per-variant lines are printed for the record.
bool RunIsa(Lines *lines) {
    const tw::isa::TickTable *base = tw::isa::Table(tw::isa::Kind::Baseline);
    const std::uint64_t reference = base->selftest();
    bool ok = true;
    for (const tw::isa::Kind kind : {tw::isa::Kind::Baseline, tw::isa::Kind::Avx2, tw::isa::Kind::Avx512}) {
        const tw::isa::TickTable *table = tw::isa::Table(kind);
        if (table == nullptr) {
            std::printf("isa %s: not built\n", tw::isa::KindName(kind));
            continue;
        }
        if (!tw::isa::Supported(kind)) {
            std::printf("isa %s: not supported by this CPU\n", tw::isa::KindName(kind));
            continue;
        }
        const bool layout = table->layout() == base->layout();
        const std::uint64_t h = table->selftest();
        // Without contraction (the default) every variant equals the baseline bit for bit (a gate);
        // with FMA contraction (measurement builds) they may differ, which is only reported.
        std::printf("isa %-8s selftest %016llx layout %s %s\n", table->name, static_cast<unsigned long long>(h),
                    layout ? "ok" : "DIFFERENT",
                    h == reference ? "equal" : (TW_FP_STRICT ? "DIFFERENT" : "differs (FMA build: informational)"));
        ok = ok && layout && (h == reference || !TW_FP_STRICT);
        if (!TW_FP_STRICT && kind != tw::isa::Kind::Baseline) lines->push_back({std::string("isa/selftest.") + table->name, h});
    }
    const char *why = nullptr;
    const tw::isa::Kind chosen = tw::isa::Select(&why);
    std::printf("isa selected: %s (%s)\n", tw::isa::KindName(chosen), why ? why : "");
    lines->push_back({"isa/selftest", reference});
    return ok;
}

std::string ModelIdFor(const std::string &stage) {
    return stage == "components" || stage == "isa" ? kComponentsId : tw::step::Info().modelId;
}

int Usage() {
    std::fprintf(stderr,
                 "usage: tw_golden_hash run|check components|engine|isa [--policy fv|contact-full|contact|cap3] "
                 "[--goldens FILE]\n");
    return 2;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) return Usage();
    const std::string mode = argv[1], stage = argv[2];
    std::string goldens = TW_GOLDEN_FILE;
    std::vector<tw::step::Policy> policies = {tw::step::Policy::FvSubsteps, tw::step::Policy::ContactSubstepsFull,
                                              tw::step::Policy::ContactSubsteps, tw::step::Policy::CappedSubsteps3};
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--goldens") && i + 1 < argc) {
            goldens = argv[++i];
        } else if (!std::strcmp(argv[i], "--policy") && i + 1 < argc) {
            const std::string p = argv[++i];
            if (p == "fv") policies = {tw::step::Policy::FvSubsteps};
            else if (p == "contact") policies = {tw::step::Policy::ContactSubsteps};
            else if (p == "contact-full") policies = {tw::step::Policy::ContactSubstepsFull};
            else if (p == "cap3") policies = {tw::step::Policy::CappedSubsteps3};
            else return Usage();
        } else {
            return Usage();
        }
    }
    if ((mode != "run" && mode != "check") || (stage != "components" && stage != "engine" && stage != "isa"))
        return Usage();

    Lines lines;
    if (stage == "components") {
        if (!RunComponents(&lines)) return 1;
    } else if (stage == "isa") {
        if (!RunIsa(&lines)) return 1;
    } else {
        for (const tw::step::Policy p : policies) {
            const int rc = RunEngine(p, &lines);
            if (rc != 0) return rc;
        }
    }
    const std::string id = ModelIdFor(stage);
    if (mode == "run") {
        for (const auto &l : lines) std::printf("%s %s %016llx\n", id.c_str(), l.first.c_str(),
                                                static_cast<unsigned long long>(l.second));
        return 0;
    }

    std::ifstream in(goldens);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", goldens.c_str());
        return 1;
    }
    std::map<std::string, std::uint64_t> expected;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string fileId, key, hex;
        if (!(ls >> fileId >> key >> hex) || fileId != id) continue;
        expected[key] = std::strtoull(hex.c_str(), nullptr, 16);
    }
    int bad = 0;
    for (const auto &l : lines) {
        const auto it = expected.find(l.first);
        if (it == expected.end()) {
            std::printf("MISSING %s %s %016llx (no golden for model %s)\n", id.c_str(), l.first.c_str(),
                        static_cast<unsigned long long>(l.second), id.c_str());
            ++bad;
        } else if (it->second != l.second) {
            std::printf("DIFF    %s %s %016llx (golden %016llx)\n", id.c_str(), l.first.c_str(),
                        static_cast<unsigned long long>(l.second), static_cast<unsigned long long>(it->second));
            ++bad;
        } else {
            std::printf("ok      %s %s %016llx\n", id.c_str(), l.first.c_str(),
                        static_cast<unsigned long long>(l.second));
        }
    }
    return bad == 0 ? 0 : 1;
}
