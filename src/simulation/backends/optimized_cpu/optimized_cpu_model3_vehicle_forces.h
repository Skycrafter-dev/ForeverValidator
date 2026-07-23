#pragma once

#include "engine/physics/dynamics/hms_item.h"
#include "simulation/backends/optimized_cpu/optimized_cpu_binary32_math.h"

class CFuncKeysReal;
class CSceneVehicleCar;
class CSceneVehicleCarTuning;
struct CHmsCorpus;

namespace forevervalidator::simulation {

// Per-runtime routing state for the exact native Model3 force specialization.
// It never replaces the item's callback: a mismatch therefore falls through to
// the authoritative callback before any vehicle-owned state is changed.
class OptimizedCpuModel3VehicleForceContext final {
public:
    void BeginTick(
            CSceneVehicleCar &car,
            OptimizedCpuBinary32MathPath mathPath,
            CHmsItem::CCallback *enabledComputeForcesCallback) noexcept;
    void Reset(void) noexcept;

    bool IsTickEligible(void) const noexcept { return tickEligible_; }
    bool WouldUseSpecializationFor(
            const CHmsItem *item) const noexcept;
    bool TryComputeOwnerForces(CHmsCorpus *corpus, float dt);

private:
    CSceneVehicleCar *car_ = nullptr;
    CHmsItem *item_ = nullptr;
    CSceneVehicleCarTuning *tuning_ = nullptr;
    CHmsItem::CCallback *canonicalCallback_ = nullptr;
    bool stableEligible_ = false;
    bool tickEligible_ = false;
};

// Focused differential entry point. Production force evaluation uses this same
// evaluator after the context has established the exact Model3 route.
float OptimizedCpuEvaluateModel3CurveForDifferential(
        CFuncKeysReal &curve,
        float input,
        bool convertSpeedToKmh,
        bool forceConstantInterpolation,
        OptimizedCpuBinary32MathPath mathPath) noexcept;

}  // namespace forevervalidator::simulation
