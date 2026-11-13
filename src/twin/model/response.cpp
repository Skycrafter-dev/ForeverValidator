// Twin contact response: guarded, non-inlined entry (math.md §4: the FP mode
// is set once per API call, in a function the compiler cannot inline).
#include "response.h"

#include "tw_fpenv.h"

namespace tw::model {

__attribute__((noinline)) static void RespondNoGuard(const CarForceConsts &k, TwinState &s,
                                                     const M3 &Rw, const RespContact *c, int n, RespTick &tick,
                                                     RespOut &out) {
    Respond(k, s, Rw, c, n, tick, out);
}

void RespondGuarded(const CarForceConsts &k, TwinState &s, const M3 &Rw, const RespContact *c,
                    int n, RespTick &tick, RespOut &out) {
    tw::math::FpGuard guard;
    RespondNoGuard(k, s, Rw, c, n, tick, out);
}

}  // namespace tw::model
