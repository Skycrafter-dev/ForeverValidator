#include <forevervalidator/native.h>

#include <new>
#include <utility>

#include "engine/camera/race_camera_internal.h"
#include "format/archive/archive_class_ids.h"
#include "format/camera/race_camera_gbx_reader.h"
#include "format/pack/installed/byte_buffer.h"
#include "format/pack/installed/installed_pack_key_catalog.h"
#include "format/pack/installed/plug_file_pack.h"
#include "format/pack/installed_vehicle_asset_graph.h"
#include "platform/native/installed_pack_asset_source_internal.h"
#include "platform/native/native_system_file_operations.h"

namespace forevervalidator {
namespace {

ValidationError CameraEnvironmentError(
        ValidationErrorCode code,
        ValidationFailureReason reason,
        const std::string &asset,
        std::string diagnostic) {
    ValidationError error;
    error.category = code == ValidationErrorCode::AllocationFailed
            ? ValidationErrorCategory::Allocation
            : code == ValidationErrorCode::InvalidArgument
                    ? ValidationErrorCategory::InvalidInput
                    : ValidationErrorCategory::Asset;
    error.code = code;
    error.stage = ValidationStage::AssetLoading;
    error.reason = reason;
    error.relatedAsset = asset;
    error.diagnostic = std::move(diagnostic);
    return error;
}

Result<AssetBytes> ReadPackAsset(
        const native_detail::InstalledPackRoot &root,
        const std::string &identifier) {
    AssetRequest request;
    request.logicalIdentifier = identifier;
    return native_detail::ReadInstalledPackAsset(root, request);
}

bool HasDecodedCameraClass(
        const camera::detail::EnvironmentConfig &environment,
        u32 classId) {
    switch (classId) {
    case TMNF_CLASS_CGameControlCamera:
        return environment.internal.has_value();
    case TMNF_CLASS_CGameControlCameraTrackManiaRace:
        return environment.race.has_value();
    case TMNF_CLASS_CGameControlCameraTrackManiaRace2:
        return environment.race2.has_value();
    case TMNF_CLASS_CGameControlCameraTrackManiaRace3:
        return environment.race3.has_value();
    default:
        return false;
    }
}

void AppendDecodedCameraClass(
        camera::detail::EnvironmentConfig *environment,
        camera::detail::EnvironmentConfig decoded,
        u32 classId,
        std::string sourceName,
        u32 collectorIndex) {
    switch (classId) {
    case TMNF_CLASS_CGameControlCamera:
        environment->internal = std::move(decoded.internal);
        environment->internalResources.push_back(
                {std::move(sourceName), collectorIndex, *environment->internal});
        break;
    case TMNF_CLASS_CGameControlCameraTrackManiaRace:
        environment->race = std::move(decoded.race);
        environment->raceResources.push_back(
                {std::move(sourceName), collectorIndex, *environment->race});
        break;
    case TMNF_CLASS_CGameControlCameraTrackManiaRace2:
        environment->race2 = std::move(decoded.race2);
        environment->race2Resources.push_back(
                {std::move(sourceName), collectorIndex, *environment->race2});
        break;
    case TMNF_CLASS_CGameControlCameraTrackManiaRace3:
        environment->race3 = std::move(decoded.race3);
        environment->race3Resources.push_back(
                {std::move(sourceName), collectorIndex, *environment->race3});
        break;
    default:
        break;
    }
}

} // namespace

Result<camera::RaceCameraEnvironment> LoadInstalledRaceCameraEnvironment(
        const std::string &packDirectory,
        const std::string &packName) noexcept {
    tmnf::platform::InstallNativeSystemFileOperations();
    try {
        if (packName.empty() ||
            packName.find_first_of("/\\") != std::string::npos ||
            packName.find('\0') != std::string::npos ||
            packName == "." || packName == "..") {
            return Result<camera::RaceCameraEnvironment>::Failure(
                    CameraEnvironmentError(
                            ValidationErrorCode::InvalidArgument,
                            ValidationFailureReason::InvalidAssetIdentifier,
                            packName,
                            "installed camera pack name is invalid"));
        }

        Result<native_detail::InstalledPackRoot> resolved =
                native_detail::ResolveInstalledPackRoot(packDirectory);
        if (!resolved) {
            return Result<camera::RaceCameraEnvironment>::Failure(
                    std::move(resolved).Error());
        }
        native_detail::InstalledPackRoot root = std::move(resolved).Value();

        Result<AssetBytes> packlist = ReadPackAsset(root, "packlist.dat");
        if (!packlist) {
            return Result<camera::RaceCameraEnvironment>::Failure(
                    std::move(packlist).Error());
        }
        const std::string packIdentifier = packName + ".pak";
        Result<AssetBytes> pak = ReadPackAsset(root, packIdentifier);
        if (!pak) {
            return Result<camera::RaceCameraEnvironment>::Failure(
                    std::move(pak).Error());
        }

        AssetBytes packlistBytes = std::move(packlist).Value();
        AssetBytes pakBytes = std::move(pak).Value();
        InstalledPackKeyCatalog keys;
        CPlugFilePack pack;
        if (!keys.LoadFromMemory(
                    packlistBytes.data(), packlistBytes.size(), "") ||
            !pack.OpenFromMemory(
                    pakBytes.data(), pakBytes.size(), keys,
                    packName.c_str())) {
            return Result<camera::RaceCameraEnvironment>::Failure(
                    CameraEnvironmentError(
                            ValidationErrorCode::AssetLoadingFailed,
                            ValidationFailureReason::InstalledPackInvalid,
                            packIdentifier,
                            "installed camera pack could not be authenticated or opened"));
        }

        const auto cameraSet =
                InstalledVehicleAssetGraph::ResolveRaceCamerasFromPack(pack);
        if (!cameraSet) {
            return Result<camera::RaceCameraEnvironment>::Failure(
                    CameraEnvironmentError(
                            ValidationErrorCode::AssetLoadingFailed,
                            ValidationFailureReason::DefaultVehicleUnavailable,
                            packIdentifier,
                            "vehicle collector has no supported race-camera references"));
        }

        camera::detail::EnvironmentConfig environment;
        environment.sourceName = packName + ".pak";
        environment.defaultCollectorIndex = cameraSet->defaultCameraIndex;
        for (const InstalledVehicleAssetReference &asset : cameraSet->cameras) {
            ByteBuffer bytes;
            std::string decodeError;
            camera::detail::EnvironmentConfig decoded;
            if (!pack.ExtractPath(asset.selectedPath.c_str(), &bytes) ||
                !camera::detail::DecodeRaceCameraProfileGbx(
                        bytes.Data(), bytes.Size(), &decoded,
                        &decodeError) ||
                !HasDecodedCameraClass(decoded, asset.classId)) {
                return Result<camera::RaceCameraEnvironment>::Failure(
                        CameraEnvironmentError(
                                ValidationErrorCode::AssetLoadingFailed,
                                ValidationFailureReason::InstalledPackInvalid,
                                asset.logicalPath,
                                decodeError.empty()
                                        ? "race-camera GBX could not be extracted or its root class differs from the pack index"
                                        : std::move(decodeError)));
            }
            AppendDecodedCameraClass(
                    &environment, std::move(decoded), asset.classId,
                    asset.logicalPath, asset.collectorIndex);
        }
        if (!environment.race && !environment.race2 && !environment.race3 &&
            !environment.internal) {
            return Result<camera::RaceCameraEnvironment>::Failure(
                    CameraEnvironmentError(
                            ValidationErrorCode::AssetLoadingFailed,
                            ValidationFailureReason::DefaultVehicleUnavailable,
                            packIdentifier,
                            "no supported camera profile was decoded"));
        }
        return Result<camera::RaceCameraEnvironment>::Success(
                camera::detail::RaceCameraEnvironmentFactory::Create(
                        std::move(environment)));
    } catch (const std::bad_alloc &) {
        return Result<camera::RaceCameraEnvironment>::Failure(
                CameraEnvironmentError(
                        ValidationErrorCode::AllocationFailed,
                        ValidationFailureReason::AllocationFailed,
                        packName,
                        "allocation failed while loading race-camera resources"));
    } catch (...) {
        return Result<camera::RaceCameraEnvironment>::Failure(
                CameraEnvironmentError(
                        ValidationErrorCode::UnexpectedFailure,
                        ValidationFailureReason::UnexpectedFailure,
                        packName,
                        "unexpected failure while loading race-camera resources"));
    }
}

} // namespace forevervalidator
