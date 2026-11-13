#pragma once

#include <optional>
#include <string>
#include <vector>

#include "engine/core/engine_types.h"

struct CPlugFilePack;

struct InstalledVehicleAssetReference {
    std::string logicalPath;
    std::string selectedPath;
    u32 classId = 0u;
    u32 collectorIndex = 0xffffffffu;

    bool IsValid() const {
        return !logicalPath.empty() && !selectedPath.empty() && classId != 0u;
    }
};

struct InstalledVehicleCameraSet {
    std::vector<InstalledVehicleAssetReference> cameras;
    u32 defaultCameraIndex = 0xffffffffu;
};

struct InstalledVehicleAssetGraph {
    InstalledVehicleAssetReference collector;
    InstalledVehicleAssetReference mobil;
    InstalledVehicleAssetReference solid;
    InstalledVehicleAssetReference tuning;
    InstalledVehicleAssetReference materials;
    InstalledVehicleAssetReference vehicleStruct;
    InstalledVehicleAssetReference fakeContactBitmap;
    InstalledVehicleAssetReference fakeContactTexture;
    std::vector<InstalledVehicleAssetReference> raceCameras;

    static std::optional<InstalledVehicleAssetGraph> ResolveFromPack(
            CPlugFilePack &pack);
    static std::optional<InstalledVehicleCameraSet>
    ResolveRaceCamerasFromPack(CPlugFilePack &pack);
    bool IsComplete() const;
};
