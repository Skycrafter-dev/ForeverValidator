#include <array>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(__i386__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

#include "engine/core/binary32_math.h"
#include "engine/core/func_keys_real.h"
#include "engine/scene/scene_vehicle_car.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_model3_vehicle_forces.h"
#include "simulation/runtime/replay_deterministic_execution.h"

namespace {

using forevervalidator::simulation::OptimizedCpuBinary32MathPath;
using forevervalidator::simulation::OptimizedCpuEvaluateModel3CurveForDifferential;
using forevervalidator::simulation::OptimizedCpuModel3VehicleForceContext;
using forevervalidator::simulation::SelectOptimizedCpuBinary32MathPathForActiveExecution;

std::size_t completedCurveCases = 0u;
std::size_t completedRoutingCases = 0u;
std::size_t completedFenvCases = 0u;

std::uint32_t FloatBits(float value) noexcept {
    std::uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float FloatFromBits(std::uint32_t bits) noexcept {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool Fail(const std::string &message) {
    std::fprintf(stderr, "model3_vehicle_differential: %s\n", message.c_str());
    return false;
}

float EvaluateReference(
        CFuncKeysReal &curve,
        float input,
        bool convertSpeedToKmh,
        bool forceConstantInterpolation) {
    if (forceConstantInterpolation) {
        curve.SetInterpolation(CFuncKeysReal::Constant);
    }
    if (convertSpeedToKmh) {
        input = Binary32::FromDouble(
                static_cast<double>(input) * static_cast<double>(3.6f));
    }
    unsigned long keyIndex = 0ul;
    float output = 0.0f;
    curve.GetValue(input, output, keyIndex);
    return output;
}

bool RunOneCurveCase(
        const char *name,
        const std::vector<CFuncKeysReal::Key> &keys,
        CFuncKeysReal::ERealInterp interpolation,
        float input,
        bool convertSpeedToKmh,
        bool forceConstantInterpolation,
        OptimizedCpuBinary32MathPath mathPath) {
    CFuncKeysReal reference;
    CFuncKeysReal optimized;
    reference.SetKeys(keys, interpolation);
    optimized.SetKeys(keys, interpolation);

    const int roundingBefore = std::fegetround();
#if defined(__i386__) || defined(__x86_64__)
    const unsigned int mxcsrControlBefore = _mm_getcsr() & 0xffc0u;
#endif
    const float referenceValue = EvaluateReference(
            reference,
            input,
            convertSpeedToKmh,
            forceConstantInterpolation);
    const float optimizedValue = OptimizedCpuEvaluateModel3CurveForDifferential(
            optimized,
            input,
            convertSpeedToKmh,
            forceConstantInterpolation,
            mathPath);
    ++completedCurveCases;

    if (FloatBits(referenceValue) != FloatBits(optimizedValue)) {
        char diagnostic[256];
        std::snprintf(
                diagnostic,
                sizeof(diagnostic),
                "%s input=%08x speed=%d constant=%d path=%u ref=%08x opt=%08x",
                name,
                FloatBits(input),
                convertSpeedToKmh ? 1 : 0,
                forceConstantInterpolation ? 1 : 0,
                static_cast<unsigned>(mathPath),
                FloatBits(referenceValue),
                FloatBits(optimizedValue));
        return Fail(diagnostic);
    }
    if (reference.Interpolation() != optimized.Interpolation()) {
        return Fail(std::string(name) + " interpolation mutation differs");
    }
    if (std::fegetround() != roundingBefore) {
        return Fail(std::string(name) + " changed the C rounding mode");
    }
#if defined(__i386__) || defined(__x86_64__)
    if ((_mm_getcsr() & 0xffc0u) != mxcsrControlBefore) {
        return Fail(std::string(name) + " changed MXCSR control bits");
    }
#endif
    return true;
}

bool RunCurveCases(OptimizedCpuBinary32MathPath selectedPath) {
    const std::vector<std::vector<CFuncKeysReal::Key>> definitions = {
            {},
            {{-0.0f, FloatFromBits(0x80000000u)}},
            {
                    {-100.0f, -7.0f},
                    {-1.0f, FloatFromBits(0x80000000u)},
                    {0.0f, 0.0f},
                    {1.0e-5f, std::numeric_limits<float>::denorm_min()},
                    {1.0f, 1.25f},
                    {100.0f, std::numeric_limits<float>::max()},
            },
            {
                    {-std::numeric_limits<float>::infinity(), -1.0f},
                    {0.0f, 2.0f},
                    {std::numeric_limits<float>::infinity(), 3.0f},
            },
    };

    std::vector<float> inputs = {
            FloatFromBits(0xff800000u),
            -std::numeric_limits<float>::max(),
            -100.0f,
            -1.0f,
            -1.0e-5f,
            FloatFromBits(0x80000001u),
            FloatFromBits(0x80000000u),
            0.0f,
            std::numeric_limits<float>::denorm_min(),
            1.0e-5f,
            1.0f,
            100.0f,
            std::numeric_limits<float>::max(),
            std::numeric_limits<float>::infinity(),
            FloatFromBits(0x7fc00001u),
            FloatFromBits(0xffc12345u),
            FloatFromBits(0x7f800001u),
    };
    for (float key : {-100.0f, -1.0f, 0.0f, 1.0e-5f, 1.0f, 100.0f}) {
        const float lower = Binary32::FromDouble(
                static_cast<double>(key) - static_cast<double>(1.0e-5f));
        const float upper = Binary32::FromDouble(
                static_cast<double>(key) + static_cast<double>(1.0e-5f));
        inputs.push_back(std::nextafter(
                lower, -std::numeric_limits<float>::infinity()));
        inputs.push_back(lower);
        inputs.push_back(std::nextafter(
                lower, std::numeric_limits<float>::infinity()));
        inputs.push_back(std::nextafter(
                upper, -std::numeric_limits<float>::infinity()));
        inputs.push_back(upper);
        inputs.push_back(std::nextafter(
                upper, std::numeric_limits<float>::infinity()));
    }

    std::uint32_t bits = 0x6d2b79f5u;
    for (unsigned index = 0u; index < 1024u; ++index) {
        bits = bits * 1664525u + 1013904223u;
        inputs.push_back(FloatFromBits(bits));
    }

    const OptimizedCpuBinary32MathPath paths[] = {
            OptimizedCpuBinary32MathPath::Reference,
            selectedPath,
    };
    for (std::size_t definitionIndex = 0u;
         definitionIndex < definitions.size();
         ++definitionIndex) {
        for (CFuncKeysReal::ERealInterp interpolation :
             {CFuncKeysReal::Linear, CFuncKeysReal::Constant}) {
            for (float input : inputs) {
                for (bool convertSpeed : {false, true}) {
                    for (bool forceConstant : {false, true}) {
                        for (OptimizedCpuBinary32MathPath path : paths) {
                            if (!RunOneCurveCase(
                                        "curve",
                                        definitions[definitionIndex],
                                        interpolation,
                                        input,
                                        convertSpeed,
                                        forceConstant,
                                        path)) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

class CustomComputeForcesCallback final
        : public CHmsItem::CCallbackComputeForces {
public:
    void ComputeForces(CHmsItem *, float) override { ++calls; }
    unsigned calls = 0u;
};

class DerivedCar final : public CSceneVehicleCar {};
class DerivedTuning final : public CSceneVehicleCarTuning {};

template<typename Car, typename Tuning>
struct RoutingFixture {
    std::array<CFuncKeysReal, 6> curves;
    Tuning tuning;
    CSceneVehicleTunings tunings;
    Car car;

    RoutingFixture() {
        for (CFuncKeysReal &curve : curves) {
            curve.SetKeys({{0.0f, 1.0f}, {100.0f, 2.0f}},
                          CFuncKeysReal::Linear);
        }
        tuning.handlingModel = CSceneVehicleCarHandlingModel_Lateral;
        tuning.maxSideFrictionFromSpeedCurve.Bind(curves[0]);
        tuning.rolloverLateralFromSpeedCurve.Bind(curves[1]);
        tuning.rolloverLateralCoefFromAngleCurve.Bind(curves[2]);
        tuning.slipResponse.accelFromSpeedCurve.Bind(curves[3]);
        tuning.steering.driveTorqueFromSpeedCurve.Bind(curves[4]);
        tuning.steering.slowDownFromSpeedCurve.Bind(curves[5]);
        tunings.SetActiveTuning(tuning);
        car.TuningsSet(&tunings);
        car.EnablePhysicsUpdates(1);
    }
};

bool ExpectInactive(
        const char *name,
        CSceneVehicleCar &car,
        OptimizedCpuBinary32MathPath path,
        CHmsItem::CCallback *enabledCallback) {
    CHmsItem *item = car.HmsItem();
    CHmsItem::CCallback *before =
            item->CallbackGet(CHmsItem::ECallback_ComputeForces);
    OptimizedCpuModel3VehicleForceContext context;
    context.BeginTick(car, path, enabledCallback);
    ++completedRoutingCases;
    if (context.IsTickEligible() ||
        context.WouldUseSpecializationFor(item)) {
        return Fail(std::string(name) + " selected specialization");
    }
    if (item->CallbackGet(CHmsItem::ECallback_ComputeForces) != before) {
        return Fail(std::string(name) + " changed callback");
    }
    return true;
}

bool RunRoutingCases(OptimizedCpuBinary32MathPath selectedPath) {
    RoutingFixture<CSceneVehicleCar, CSceneVehicleCarTuning> exact;
    CHmsItem *item = exact.car.HmsItem();
    CHmsItem::CCallback *canonical =
            item->CallbackGet(CHmsItem::ECallback_ComputeForces);

    if (!ExpectInactive(
                "reference-math", exact.car,
                OptimizedCpuBinary32MathPath::Reference,
                canonical)) {
        return false;
    }

    if (selectedPath == OptimizedCpuBinary32MathPath::X86Sse2) {
        OptimizedCpuModel3VehicleForceContext context;
        context.BeginTick(exact.car, selectedPath, canonical);
        ++completedRoutingCases;
        if (!context.IsTickEligible() ||
            !context.WouldUseSpecializationFor(item) ||
            item->CallbackGet(CHmsItem::ECallback_ComputeForces) != canonical) {
            return Fail("exact canonical Model3 route was not selected");
        }

        exact.tuning.handlingModel =
                CSceneVehicleCarHandlingModel_RadiusSteering;
        if (context.WouldUseSpecializationFor(item)) {
            return Fail("post-capture handling change did not fall back");
        }
        exact.tuning.handlingModel = CSceneVehicleCarHandlingModel_Lateral;

        CustomComputeForcesCallback custom;
        item->CallbackSet(CHmsItem::ECallback_ComputeForces, &custom);
        if (context.WouldUseSpecializationFor(item)) {
            return Fail("post-capture callback change did not fall back");
        }
        item->CallbackSet(CHmsItem::ECallback_ComputeForces, canonical);

        exact.tuning.handlingModel =
                CSceneVehicleCarHandlingModel_RadiusSteering;
        if (!ExpectInactive(
                    "wrong-handling", exact.car, selectedPath, canonical)) {
            return false;
        }
        exact.tuning.handlingModel = CSceneVehicleCarHandlingModel_Lateral;

        exact.tuning.steering.slowDownFromSpeedCurve.Reset();
        if (!ExpectInactive(
                    "unbound-required-curve",
                    exact.car,
                    selectedPath,
                    canonical)) {
            return false;
        }
        exact.tuning.steering.slowDownFromSpeedCurve.Bind(exact.curves[5]);

        exact.car.EnablePhysicsUpdates(0);
        if (!ExpectInactive(
                    "suppressed-callback", exact.car, selectedPath, nullptr)) {
            return false;
        }
        exact.car.EnablePhysicsUpdates(1);
        canonical = item->CallbackGet(CHmsItem::ECallback_ComputeForces);

        context.BeginTick(exact.car, selectedPath, canonical);
        ++completedRoutingCases;
        if (!context.WouldUseSpecializationFor(item)) {
            return Fail("context did not reactivate after suppression");
        }

        item->CallbackSet(CHmsItem::ECallback_ComputeForces, &custom);
        if (!ExpectInactive(
                    "custom-callback", exact.car, selectedPath, canonical)) {
            return false;
        }
        item->CallbackSet(CHmsItem::ECallback_ComputeForces, canonical);

        item->SetSceneMobilOwner(nullptr);
        if (!ExpectInactive(
                    "owner-mismatch", exact.car, selectedPath, canonical)) {
            return false;
        }
        item->SetSceneMobilOwner(&exact.car);

        RoutingFixture<DerivedCar, CSceneVehicleCarTuning> derivedCar;
        CHmsItem *derivedCarItem = derivedCar.car.HmsItem();
        if (!ExpectInactive(
                    "derived-car",
                    derivedCar.car,
                    selectedPath,
                    derivedCarItem->CallbackGet(
                            CHmsItem::ECallback_ComputeForces))) {
            return false;
        }
        RoutingFixture<CSceneVehicleCar, DerivedTuning> derivedTuning;
        CHmsItem *derivedTuningItem = derivedTuning.car.HmsItem();
        if (!ExpectInactive(
                    "derived-tuning",
                    derivedTuning.car,
                    selectedPath,
                    derivedTuningItem->CallbackGet(
                            CHmsItem::ECallback_ComputeForces))) {
            return false;
        }
    }
    return true;
}

bool RunFenvCases(OptimizedCpuBinary32MathPath selectedPath) {
    if (SelectOptimizedCpuBinary32MathPathForActiveExecution() != selectedPath) {
        return Fail("active selector changed unexpectedly");
    }
    ++completedFenvCases;

    if (std::fesetround(FE_DOWNWARD) != 0) {
        return Fail("could not set FE_DOWNWARD");
    }
    ++completedFenvCases;
    if (SelectOptimizedCpuBinary32MathPathForActiveExecution() !=
        OptimizedCpuBinary32MathPath::Reference) {
        std::fesetround(FE_TONEAREST);
        return Fail("non-RNE environment selected native Model3 math");
    }
    if (std::fesetround(FE_TONEAREST) != 0) {
        return Fail("could not restore FE_TONEAREST");
    }

#if defined(__i386__) || defined(__x86_64__)
    if (selectedPath == OptimizedCpuBinary32MathPath::X86Sse2) {
        const unsigned int base = _mm_getcsr();
        _mm_setcsr(base | 0x3fu);
        ++completedFenvCases;
        if (SelectOptimizedCpuBinary32MathPathForActiveExecution() !=
            selectedPath) {
            _mm_setcsr(base);
            return Fail("sticky MXCSR flags disabled native Model3 math");
        }
        _mm_setcsr(base);

        const unsigned int incompatible[] = {
                base | 0x0040u,
                base & ~0x0080u,
                base | 0x2000u,
                base | 0x4000u,
                base | 0x6000u,
                base | 0x8000u,
        };
        for (unsigned int value : incompatible) {
            _mm_setcsr(value);
            ++completedFenvCases;
            if (SelectOptimizedCpuBinary32MathPathForActiveExecution() !=
                OptimizedCpuBinary32MathPath::Reference) {
                _mm_setcsr(base);
                return Fail("incompatible MXCSR selected native Model3 math");
            }
            _mm_setcsr(base);
        }
    }
#endif
    return true;
}

}  // namespace

int main(void) {
    if (SelectOptimizedCpuBinary32MathPathForActiveExecution() !=
        OptimizedCpuBinary32MathPath::Reference) {
        std::fprintf(stderr, "native Model3 math selected outside scope\n");
        return 1;
    }

    tmnf::simulation::DeterministicExecutionScope deterministicScope;
    if (!deterministicScope.Established()) {
        std::fprintf(stderr, "could not establish deterministic execution\n");
        return 1;
    }
    const OptimizedCpuBinary32MathPath selectedPath =
            SelectOptimizedCpuBinary32MathPathForActiveExecution();

    if (!RunFenvCases(selectedPath) || !RunCurveCases(selectedPath) ||
        !RunRoutingCases(selectedPath)) {
        return 1;
    }

    std::printf(
            "model3_curve_cases=%zu routing_cases=%zu fenv_cases=%zu "
            "binary32_path=%s result=identical\n",
            completedCurveCases,
            completedRoutingCases,
            completedFenvCases,
            selectedPath == OptimizedCpuBinary32MathPath::X86Sse2
                    ? "x86_sse2"
                    : "reference");
    return 0;
}
