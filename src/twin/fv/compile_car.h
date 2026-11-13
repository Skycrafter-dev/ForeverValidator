// Twin FV adapter: StadiumCar force constants from FV's flat static
// configuration (BuildCudaHostStaticConfiguration, called in place).
// Only src/twin/fv/ includes FV headers (design.md §8.3).
#ifndef TW_FV_COMPILE_CAR_H
#define TW_FV_COMPILE_CAR_H

#include <cstdint>
#include <string>
#include <vector>

#include "car_force_consts.h"

namespace forevervalidator::simulation {
struct CudaHostStaticConfiguration;
}

namespace tw::fv {

// Fills `out`. Refuses (returns false, reason in `diag`) anything the kernel
// does not model: handling model != 5, wheel force mode != 2, != 4 wheels,
// non-diagonal body inertia, Ball force fields, non-power-of-two fake-contact
// periods, curves above 15 keys or unsorted. Model 6 paths that never fire for
// StadiumCar (side-speed kill for lateralForceScale < 0, large steer angles,
// pointImpulseAngularYScale, slippingAccelScale) are kept, behind per-car flags.
bool CompileCarForceConsts(const forevervalidator::simulation::CudaHostStaticConfiguration &cfg,
                           tw::model::CarForceConsts *out, std::string *diag);

// Water zone cells are copied into `cells` (the zone points into it).
bool CompileWaterZone(const forevervalidator::simulation::CudaHostStaticConfiguration &cfg,
                      tw::model::WaterZone *out, std::vector<std::uint8_t> *cells, std::string *diag);

}  // namespace tw::fv

#endif
