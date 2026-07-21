#include "simulation/runtime/replay_vehicle_wheel_surfaces.h"
#include "engine/rendering/plug_tree.h"
#include "simulation/runtime/replay_vehicle_collision_model.h"
void ReplayVehicleWheelSurfaces::Reset(CSceneVehicleCar *car) {
    SanitizeWheelBorrowedPointers(car);
    wheelBindings.clear();
}

void ReplayVehicleWheelSurfaces::MarkWheelSurfaceUpdated(
        CSceneVehicleCar *car,
        CSceneVehicleCar::SSimulationWheel *wheel) {
    if (car == nullptr || wheel == nullptr) {
        return;
    }
    for (u32 i = 0u; i < car->WheelGetCount(); ++i) {
        if (&car->WheelAt(i) == wheel) {
            WheelSurfaceBinding *binding = BindingAt(i);
            if (binding != nullptr) {
                binding->movedByUpdateSurface = true;
            }
            return;
        }
    }
}

void ReplayVehicleWheelSurfaces::BeginWheelFromReplayState(
        u32 wheelIndex,
        CSceneVehicleCar::SSimulationWheel *wheel) {
    if (wheel == nullptr) {
        return;
    }
    WheelSurfaceBinding *binding = EnsureBinding(wheelIndex);
    CPlugTree *localTree = binding->ownedLocalTree.get();
    localTree->SetLocation(wheel->surfaceHandler.CurrentPose());
    wheel->surfaceHandler.BindTree(localTree);
}

void ReplayVehicleWheelSurfaces::UpdateLocalIsoFromWheel(
        u32 wheelIndex,
        const CSceneVehicleCar::SSimulationWheel *wheel) {
    CPlugTree *tree = LocalTree(wheelIndex);
    if (tree != nullptr && wheel != nullptr) {
        tree->SetLocation(wheel->surfaceHandler.CurrentPose());
    }
}

void ReplayVehicleWheelSurfaces::BindCollisionShapes(
        const ReplayVehicleCollisionModel &collisionModel,
        const VehicleWheelSetDefinition &wheelDefinitions,
        CSceneVehicleCar &car) {
    for (u32 wheelIndex = 0u;
         wheelIndex < car.WheelGetCount();
         ++wheelIndex) {
        CSceneVehicleCar::SSimulationWheel &wheel = car.WheelAt(wheelIndex);
        if (wheelIndex >= wheelDefinitions.wheels.size()) {
            UseLocalHandlerTree(wheelIndex, &wheel);
            continue;
        }
        const VehicleWheelDefinition &definition =
                wheelDefinitions.wheels[wheelIndex];
        CPlugTree *tree = collisionModel.Shape(definition.collisionRole);
        if (tree == nullptr) {
            UseLocalHandlerTree(wheelIndex, &wheel);
            continue;
        }
        SetBoundSourceTreeLocation(tree, &wheel.surfaceHandler.CurrentPose());
        if (ShouldPreserveBoundWheelRestBox(wheelIndex)) {
            PreserveBoundWheelSurfaceRestBox(tree, &wheel);
        }
        BindWheelToSourceTree(wheelIndex, tree, &wheel);
    }
}

ReplayVehicleWheelSurfaces::WheelSurfaceBinding *
ReplayVehicleWheelSurfaces::EnsureBinding(u32 wheelIndex) {
    const std::size_t requiredSize = static_cast<std::size_t>(wheelIndex) + 1u;
    if (wheelBindings.size() < requiredSize) {
        wheelBindings.resize(requiredSize);
    }
    WheelSurfaceBinding &binding = wheelBindings[wheelIndex];
    if (!binding.ownedLocalTree) {
        binding.ownedLocalTree = std::make_unique<CPlugTree>();
        CPlugTree::SFlags state;
        state.visible = false;
        state.pickableVisual = false;
        state.castsShadows = false;
        state.rooted = false;
        state.locationDirty = false;
        binding.ownedLocalTree->ApplyLoadedState(state);
    }
    return &binding;
}

ReplayVehicleWheelSurfaces::WheelSurfaceBinding *
ReplayVehicleWheelSurfaces::BindingAt(u32 wheelIndex) {
    return wheelIndex < wheelBindings.size()
            ? &wheelBindings[wheelIndex]
            : nullptr;
}

const ReplayVehicleWheelSurfaces::WheelSurfaceBinding *
ReplayVehicleWheelSurfaces::BindingAt(u32 wheelIndex) const {
    return wheelIndex < wheelBindings.size()
            ? &wheelBindings[wheelIndex]
            : nullptr;
}

CPlugTree *ReplayVehicleWheelSurfaces::LocalTree(u32 wheelIndex) {
    WheelSurfaceBinding *binding = BindingAt(wheelIndex);
    return binding != nullptr ? binding->ownedLocalTree.get() : nullptr;
}

void ReplayVehicleWheelSurfaces::UseLocalHandlerTree(
        u32 wheelIndex,
        CSceneVehicleCar::SSimulationWheel *wheel) {
    CPlugTree *tree = LocalTree(wheelIndex);
    if (tree != nullptr && wheel != nullptr) {
        wheel->surfaceHandler.BindTree(tree);
    }
}

void ReplayVehicleWheelSurfaces::BindWheelToSourceTree(
        u32 wheelIndex,
        CPlugTree *tree,
        CSceneVehicleCar::SSimulationWheel *wheel) {
    if (BindingAt(wheelIndex) == nullptr || tree == nullptr || wheel == nullptr) {
        return;
    }
    wheel->surfaceHandler.BindTree(tree);
    UpdateLocalIsoFromWheel(wheelIndex, wheel);
}

bool ReplayVehicleWheelSurfaces::ShouldPreserveBoundWheelRestBox(
        u32 wheelIndex) const {
    const WheelSurfaceBinding *binding = BindingAt(wheelIndex);
    return binding != nullptr && !binding->movedByUpdateSurface;
}

void ReplayVehicleWheelSurfaces::SetBoundSourceTreeLocation(
        CPlugTree *tree,
        const GmIso4 *iso) {
    if (tree != nullptr && iso != nullptr) {
        tree->SetLocation(*iso);
    }
}

void ReplayVehicleWheelSurfaces::PreserveBoundWheelSurfaceRestBox(
        CPlugTree *tree,
        const CSceneVehicleCar::SSimulationWheel *wheel) {
    if (tree != nullptr && wheel != nullptr) {
        GmBoxAligned bounds = tree->Box();
        bounds.center = wheel->forceApplicationPoint;
        tree->SetTreeBounds(bounds);
    }
}

void ReplayVehicleWheelSurfaces::SanitizeWheelBorrowedPointers(
        CSceneVehicleCar *car) {
    if (car == nullptr) {
        return;
    }
    for (u32 wheelIndex = 0u;
         wheelIndex < car->WheelGetCount();
         ++wheelIndex) {
        CSceneVehicleCar::SSimulationWheel &wheel = car->WheelAt(wheelIndex);
        wheel.surfaceHandler.BindTree(nullptr);
        wheel.realTimeState.peerCorpusId = {};
    }
}

void ReplayVehicleWheelSurfaces::OnWheelSurfaceUpdated(
        CSceneVehicleCar &car,
        CSceneVehicleCar::SSimulationWheel &wheel) {
    MarkWheelSurfaceUpdated(&car, &wheel);
}

ReplayVehicleWheelSurfaces::RuntimeClone
ReplayVehicleWheelSurfaces::CaptureRuntimeClone() const {
    RuntimeClone clone;
    clone.movedByUpdateSurface.reserve(wheelBindings.size());
    for (const WheelSurfaceBinding &binding : wheelBindings) {
        clone.movedByUpdateSurface.push_back(binding.movedByUpdateSurface);
    }
    return clone;
}

bool ReplayVehicleWheelSurfaces::CanRestoreRuntimeClone(
        const RuntimeClone &clone,
        const CSceneVehicleCar &car) const noexcept {
    return clone.movedByUpdateSurface.size() == wheelBindings.size() &&
           clone.movedByUpdateSurface.size() == car.WheelGetCount();
}

void ReplayVehicleWheelSurfaces::RestoreRuntimeClone(
        const RuntimeClone &clone,
        CSceneVehicleCar &car) noexcept {
    for (u32 index = 0u; index < car.WheelGetCount(); ++index) {
        wheelBindings[index].movedByUpdateSurface =
                clone.movedByUpdateSurface[index];
        CSceneVehicleCar::SSimulationWheel &wheel = car.WheelAt(index);
        CPlugTree *tree = wheel.surfaceHandler.Tree();
        if (tree != nullptr) {
            tree->SetLocation(wheel.surfaceHandler.CurrentPose());
        }
    }
}
