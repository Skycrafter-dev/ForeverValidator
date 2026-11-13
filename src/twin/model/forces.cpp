// Twin StadiumCar force kernel: non-inlined entries. The FP mode is set once
// per API call by the caller (FpGuard, math.md §4); CarSubstepGuarded is the
// self-guarded L = 1 convenience entry.
#include "forces.h"

#include "tw_fpenv.h"

namespace tw::model {

__attribute__((noinline)) void CarSubstepNoGuard(const CarForceConsts &k, const WaterZone *water, TwinState &s,
                                                 const CarTickIn &in, float h, CarForceOut &out) {
    CarSubstep(k, water, s, in, h, out);
}

void CarSubstepGuarded(const CarForceConsts &k, const WaterZone *water, TwinState &s, const CarTickIn &in, float h,
                       CarForceOut &out) {
    tw::math::FpGuard guard;
    CarSubstepNoGuard(k, water, s, in, h, out);
}

__attribute__((noinline)) void CarAirSubstepNoGuard(const CarForceConsts &k, const WaterZone *water, TwinState &s,
                                                    const CarTickIn &in, float h, CarForceOut &out) {
    CarAirSubstep(k, water, s, in, h, out);
}

#if TW_HAVE_AVX512F
__attribute__((noinline)) void CarAirSubstep16Lanes(const CarForceConsts &k, const WaterZone *water, TwinLanes<16> &s,
                                                    const CarTickInL<16> &in, const float (&h)[16], CarForceOutL<16> &out,
                                                    std::uint32_t active) {
    CarAirSubstepL<16>(k, water, s, in, h, out, active);
}
#endif

__attribute__((noinline)) void CarSubstep4(const CarForceConsts &k, const WaterZone *water, TwinLanes<4> &s,
                                           const CarTickInL<4> &in, const float (&h)[4], CarForceOutL<4> &out,
                                           std::uint32_t active) {
    CarSubstepL<4>(k, water, s, in, h, out, active);
}

#if TW_HAVE_AVX2
__attribute__((noinline)) void CarSubstep8(const CarForceConsts &k, const WaterZone *water, TwinLanes<8> &s,
                                           const CarTickInL<8> &in, const float (&h)[8], CarForceOutL<8> &out,
                                           std::uint32_t active) {
    CarSubstepL<8>(k, water, s, in, h, out, active);
}
#endif

#if TW_HAVE_AVX512F
__attribute__((noinline)) void CarSubstep16(const CarForceConsts &k, const WaterZone *water, TwinLanes<16> &s,
                                            const CarTickInL<16> &in, const float (&h)[16], CarForceOutL<16> &out,
                                            std::uint32_t active) {
    CarSubstepL<16>(k, water, s, in, h, out, active);
}
#endif

}  // namespace tw::model
