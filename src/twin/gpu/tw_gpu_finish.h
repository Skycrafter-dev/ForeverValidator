// Twin GPU engine: race::TwRefineFinish (the sub-tick finish bracket, R4.2) as a resumable state machine, so the
// lane runner (tw_gpu_lockstep.cuh) runs each probe as one substep of a pass. Host and device (TW_HD): searchq
// --gpu checks it against race::TwRefineFinish before a run.
//
//   FinishBracket b;
//   float h;
//   if (b.Begin(sub, &h)) {                 // false: no bracket (TwRefineFinish's argument checks)
//       for (bool r = probe(h); b.Next(r, &h); r = probe(h)) {}
//       if (b.Valid()) TwStoreFinish(s, b.LowerNs());
//   }
#ifndef TW_GPU_FINISH_H
#define TW_GPU_FINISH_H

#include <cmath>
#include <cstdint>
#include <limits>

#include "tw_race.h"

namespace tw::gpu {

struct FinishBracket {
    double lower, upper, startNs;
    std::uint64_t cand;
    bool first;

    // TwRefineFinish up to its first probe (the whole completing substep): its argument checks; *h the probe's step
    TW_HD bool Begin(const race::TwFinishSubstep &sub, float *h) {
        constexpr double kNsPerS = 1000000000.0;
        if (!std::isfinite(sub.substepStartSeconds) || sub.substepStartSeconds < 0.0 || !std::isfinite(sub.substepDurationSeconds) ||
            !(sub.substepDurationSeconds > 0.0f))
            return false;
        const std::uint64_t tickStartNs = static_cast<std::uint64_t>(sub.tickMs) * 1000000u;
        const double lo = static_cast<double>(tickStartNs) + sub.substepStartSeconds * kNsPerS;
        const double up = lo + static_cast<double>(sub.substepDurationSeconds) * kNsPerS;
        if (!std::isfinite(lo) || !std::isfinite(up) || lo < 0.0 || !(lo < up) ||
            up > static_cast<double>(std::numeric_limits<std::uint64_t>::max() - 1u) || up >= 9007199254740992.0)
            return false;
        lower = lo;
        upper = up;
        startNs = lo;
        first = true;
        *h = sub.substepDurationSeconds;
        return true;
    }
    // the probe's answer (the race completed in the probe); true: probe again with step *h; false: done (Valid)
    TW_HD bool Next(bool completed, float *h) {
        constexpr double kNsPerS = 1000000000.0;
        if (first) {
            first = false;
            if (!completed) {   // TwRefineFinish returns false: no bracket
                lower = 1.0;
                upper = 0.0;
                return false;
            }
        } else if (completed) {
            upper = static_cast<double>(cand);
        } else {
            lower = static_cast<double>(cand);
        }
        for (;;) {
            const std::uint64_t firstInterior = race::detail::FloorNs(lower) + 1u;
            const std::uint64_t upperCeil = race::detail::CeilNs(upper);
            if (upperCeil == 0u || firstInterior >= upperCeil) return false;
            const std::uint64_t lastInterior = upperCeil - 1u;
            cand = firstInterior + (lastInterior - firstInterior) / 2u;
            const double partial = (static_cast<double>(cand) - startNs) / kNsPerS;
            if (partial <= 0.0) {
                lower = static_cast<double>(cand);
                continue;
            }
            *h = static_cast<float>(partial);
            return true;
        }
    }
    // after Next returned false: TwRefineFinish's result and its lower bound
    TW_HD bool Valid() const { return lower <= upper && race::detail::FloorNs(lower) < race::detail::CeilNs(upper); }
    TW_HD std::uint64_t LowerNs() const { return race::detail::FloorNs(lower); }
};

}  // namespace tw::gpu

#endif
