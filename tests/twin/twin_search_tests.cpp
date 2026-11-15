// Twin search (M13): host candidate generation on synthetic inputs, no game
// assets. Equality with the GPU's candidates is checked end to end by
// tools/twin/twin_search_bench --compare cuda (docs/twin/m13-search.md);
// this test pins determinism and the generator's contract, and a hash of
// 4,000 candidates (it changes only if FV's CUDA mutation code changes, in
// which case re-run the CUDA comparison and update kCandidateHash).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "src/twin/fv/twin_search_candidates.h"

namespace {

namespace sim = forevervalidator::simulation;
int g_fail = 0;

void Check(bool ok, const char *what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

constexpr std::uint64_t kCandidateHash = 0x68c9446d5d94ace8ull;

sim::CudaSearchExecutorConfiguration MakeConfiguration() {
    sim::CudaSearchExecutorConfiguration c;
    c.tickDurationMs = 10u;
    c.branchTimeMs = 1000;
    c.evaluationStartTimeMs = 1010;
    c.evaluationEndTimeMs = 3000;
    c.maximumBatchSize = 64u;
    // Accelerate held, analog steering every 50 ms, a brake tap.
    c.baselineInputs.push_back({0, 1u, 1u, 1});
    for (std::int32_t t = 0; t <= 4000; t += 50) {
        c.baselineInputs.push_back({t, 4u, 2u, (t * 37) % 131073 - 65536});
    }
    c.baselineInputs.push_back({2000, 3u, 1u, 1});
    c.baselineInputs.push_back({2100, 3u, 1u, 0});
    std::stable_sort(c.baselineInputs.begin(), c.baselineInputs.end(),
                     [](const sim::CudaSearchInputEvent &a, const sim::CudaSearchInputEvent &b) {
                         return a.timeMs < b.timeMs;
                     });
    c.maximumEventCount = c.baselineInputs.size() + 256u;
    sim::CudaSearchModifierConfiguration random;
    random.kind = sim::CudaSearchModifierKind::RandomSteering;
    random.window = {1010, 2500, 11u};
    c.modifiers.push_back(random);
    sim::CudaSearchModifierConfiguration insertion;
    insertion.kind = sim::CudaSearchModifierKind::InputInsertion;
    insertion.window = {1200, 2800, 12u};
    insertion.steering = {1u, 1u, 3u, 100};
    insertion.brake = {1u, 0u, 1u, 50};
    insertion.analogMinimum = -65536;
    insertion.analogMaximum = 65536;
    c.modifiers.push_back(insertion);
    sim::CudaSearchModifierConfiguration existing;
    existing.kind = sim::CudaSearchModifierKind::ExistingEvent;
    existing.window = {1010, 2900, 13u};
    existing.minimumCount = 1u;
    existing.maximumCount = 4u;
    existing.timeParameterMs = 50;
    existing.analogMinimum = -20000;
    existing.analogMaximum = 20000;
    c.modifiers.push_back(existing);
    return c;
}

std::uint64_t Hash(std::uint64_t h, const sim::twin_search::GeneratedCandidate &c) {
    auto mix = [&](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            h ^= (v >> (8 * i)) & 0xffu;
            h *= 1099511628211ull;
        }
    };
    mix(static_cast<std::uint64_t>(c.status));
    mix(c.active ? 1u : 0u);
    mix(c.mutationCount);
    mix(c.events.size());
    for (const auto &e : c.events) {
        mix(static_cast<std::uint32_t>(e.timeMs));
        mix(e.action);
        mix(e.valueKind);
        mix(static_cast<std::uint32_t>(e.value));
    }
    return h;
}

bool Same(const sim::twin_search::GeneratedCandidate &a, const sim::twin_search::GeneratedCandidate &b) {
    return Hash(1469598103934665603ull, a) == Hash(1469598103934665603ull, b);
}

}  // namespace

int main() {
    std::string diagnostic;
    sim::twin_search::MutationProgram program;
    Check(sim::twin_search::PrepareMutationProgram(MakeConfiguration(), &program, &diagnostic), "prepare");
    if (g_fail != 0) {
        std::fprintf(stderr, "%s\n", diagnostic.c_str());
        return 1;
    }
    Check(program.mutableFromTimeMs == 1010, "mutable boundary is the tick after the branch");
    Check(!program.immutablePrefix.empty() && !program.immutableTail.empty(), "prefix and tail split");
    Check(program.baselineInputsCanonical, "canonical baseline");

    sim::twin_search::CandidateGenerator generator(program), other(program);
    sim::twin_search::GeneratedCandidate a, b;
    generator.Generate(0u, true, &a);
    Check(a.active && a.mutationCount == 0u && a.events.size() == program.baselineInputs.size(),
          "baseline candidate is the baseline");

    std::uint64_t h = 1469598103934665603ull;
    std::uint32_t active = 0u;
    for (std::uint64_t id = 1u; id <= 4000u; ++id) {
        generator.Generate(id, false, &a);
        other.Generate(id, false, &b);
        if (!Same(a, b)) {
            Check(false, "same id gives the same candidate on another generator");
            break;
        }
        if (a.active) ++active;
        h = Hash(h, a);
    }
    // Re-generating an earlier id after others gives the same candidate (no hidden state).
    generator.Generate(17u, false, &a);
    other.Generate(17u, false, &b);
    Check(Same(a, b), "candidate 17 regenerated identically");
    Check(active > 3900u, "nearly every candidate mutates");
    std::printf("twin search candidates: 4000 generated, %u active, hash %016llx\n", active,
                static_cast<unsigned long long>(h));
    if (kCandidateHash != 0u) Check(h == kCandidateHash, "candidate hash (FV CUDA mutation semantics)");
    std::printf("%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
