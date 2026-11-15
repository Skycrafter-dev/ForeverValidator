// Gate 1' collision reference: see tw_collide_ref.h. The body (tw_collide_ref_impl.inc) is compiled for double
// and for the stochastic-rounding float (src/twin/ref/tw_sr.h).
#include "tw_collide_ref.h"

#include <algorithm>
#include <cmath>

namespace tw::collide::ref {

namespace impl_d {
using T = double;
#include "tw_collide_ref_impl.inc"
}  // namespace impl_d

namespace impl_sr {
using T = tw::sr::sr;
#include "tw_collide_ref_impl.inc"
}  // namespace impl_sr

void Detect(const ExactCollision &stage, const double pos[3], const double quat[4], const double absorb[4], DetectionD *out) {
    impl_d::DetectImpl(stage, pos, quat, absorb, out);
}

void DetectSR(const ExactCollision &stage, const sr::sr pos[3], const sr::sr quat[4], const sr::sr absorb[4], DetectionD *out) {
    impl_sr::DetectImpl(stage, pos, quat, absorb, out);
}

}  // namespace tw::collide::ref
