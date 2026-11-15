#include "format/pack/installed_vehicle_asset_graph.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <new>
#include <utility>

#include "format/archive/archive_class_ids.h"
#include "format/archive/archive_binary.h"
#include "format/archive/tmnf_gbx_body_reader.h"
#include "format/pack/installed/byte_buffer.h"
#include "format/pack/installed/plug_file_pack.h"
#include "format/static_solid/static_scene_archive_loader.h"
#include "format/static_solid/static_solid_archive_payload_stream.h"
#include "format/vehicle_tuning/default_vehicle_archive_schema.h"

namespace {

using TmnfFormat::ArchiveBinary::ReadU32LE;

constexpr u32 VehicleCollectorDefaultCameraChunk = 0x0301c006u;
constexpr u32 VehicleCollectorCamerasChunk = 0x0301c009u;
constexpr u32 VehicleCollectorCameraArchiveVersion = 10u;
constexpr u32 MaximumVehicleCollectorCameras = 64u;

bool ParseReferenceTable(
        CPlugFilePack &pack,
        const InstalledVehicleAssetReference &asset,
        GbxBodyReferenceTable *references) {
    if (references == nullptr || !asset.IsValid()) {
        return false;
    }
    const CPlugFileFidContainer_SFileDesc *file =
            pack.FindFileDescByPath(asset.selectedPath.c_str());
    if (file == nullptr || file->classId != asset.classId) {
        return false;
    }

    ByteBuffer extracted;
    u32 classId = 0u;
    if (pack.ExtractPath(asset.selectedPath.c_str(), &extracted) &&
        extracted.Size() <= UINT32_MAX &&
        GbxBodyOffsetReader::TryParseWithReferences(
                extracted.Data(),
                static_cast<u32>(extracted.Size()),
                &classId,
                references)) {
        return classId == asset.classId;
    }
    if (!file->IsEncryptedPayload()) {
        return false;
    }

    StaticSolidArchivePayload payload;
    if (!payload.BuildEncryptedPackFilePayload({
                *file,
                pack.dataStart,
                0u,
                0u,
                asset.logicalPath.c_str(),
                asset.selectedPath.c_str()})) {
        return false;
    }
    const std::size_t payloadOffset = payload.PackPayloadOffset();
    if (payloadOffset > pack.bytes.Size() ||
        payload.RawByteCount() > pack.bytes.Size() - payloadOffset) {
        return false;
    }
    CGameCtnReplayStaticSolidDecodedPayload header;
    if (!CGameCtnReplayStaticSolidArchivePayloadReader::
                DecodeReferenceTablePrefixWithStreamFeedback(
                        pack.key,
                        payload,
                        StaticSolidArchiveRawBytes::FromBytes(
                                pack.bytes.Data() + payloadOffset,
                                payload.RawByteCount()),
                        &header) ||
        !GbxBodyOffsetReader::TryParseWithReferences(
                header.Bytes(),
                header.ByteCount(),
                &classId,
                references)) {
        return false;
    }
    return classId == asset.classId;
}

bool FindUniquePackClass(
        CPlugFilePack &pack,
        u32 classId,
        InstalledVehicleAssetReference *out) {
    if (out == nullptr) {
        return false;
    }
    *out = InstalledVehicleAssetReference{};
    const CPlugFileFidContainer_SFileDesc *found = nullptr;
    for (const CPlugFileFidContainer_SFileDesc &file : pack.files) {
        if (file.classId != classId) {
            continue;
        }
        if (found != nullptr) {
            return false;
        }
        found = &file;
    }
    std::array<char, 512u> path{};
    if (found == nullptr ||
        !pack.FileDescPlainPath(found, path.data(), path.size())) {
        return false;
    }
    try {
        out->logicalPath = path.data();
        out->selectedPath = path.data();
        out->classId = classId;
    } catch (const std::bad_alloc &) {
        *out = InstalledVehicleAssetReference{};
        return false;
    }
    return true;
}

bool ResolveUniqueDeclaredClass(
        CPlugFilePack &pack,
        const InstalledVehicleAssetReference &source,
        const GbxBodyReferenceTable &references,
        u32 targetClassId,
        InstalledVehicleAssetReference *out) {
    if (out == nullptr) {
        return false;
    }
    *out = InstalledVehicleAssetReference{};
    bool found = false;
    for (const GbxBodyExternalReference &reference :
         references.externalReferences) {
        if (reference.IsResource()) {
            continue;
        }
        std::string logicalPath;
        std::array<char, 512u> selectedPath{};
        if (!references.ResolvePlainPathForReference(
                    source.logicalPath, reference, &logicalPath) ||
            !pack.SelectedPathForPlainRef(
                    logicalPath.c_str(),
                    selectedPath.data(),
                    selectedPath.size())) {
            continue;
        }
        const CPlugFileFidContainer_SFileDesc *file =
                pack.FindFileDescByPath(selectedPath.data());
        if (file == nullptr || file->classId != targetClassId) {
            continue;
        }
        if (found) {
            return false;
        }
        try {
            out->logicalPath = std::move(logicalPath);
            out->selectedPath = selectedPath.data();
            out->classId = targetClassId;
        } catch (const std::bad_alloc &) {
            *out = InstalledVehicleAssetReference{};
            return false;
        }
        found = true;
    }
    return found;
}

bool IsRaceCameraClass(u32 classId) {
    return classId == TMNF_CLASS_CGameControlCamera ||
           classId == TMNF_CLASS_CGameControlCameraTrackManiaRace ||
           classId == TMNF_CLASS_CGameControlCameraTrackManiaRace2 ||
           classId == TMNF_CLASS_CGameControlCameraTrackManiaRace3;
}

bool ParseCollectorCameraIndices(
        CPlugFilePack &pack,
        const InstalledVehicleAssetReference &source,
        const GbxBodyReferenceTable &references,
        u32 *defaultCameraIndex,
        std::vector<u32> *cameraNodeIndices) {
    if (defaultCameraIndex == nullptr || cameraNodeIndices == nullptr ||
        references.bodyOffset > UINT32_MAX || !source.IsValid()) {
        return false;
    }
    ByteBuffer bytes;
    if (!pack.ExtractPath(source.selectedPath.c_str(), &bytes) ||
        references.bodyOffset > bytes.Size()) {
        return false;
    }

    std::optional<std::size_t> camerasChunk;
    for (std::size_t offset = references.bodyOffset;
         offset + 12u <= bytes.Size(); ++offset) {
        if (ReadU32LE(bytes.Data() + offset) !=
                    VehicleCollectorCamerasChunk ||
            ReadU32LE(bytes.Data() + offset + 4u) !=
                    VehicleCollectorCameraArchiveVersion) {
            continue;
        }
        const u32 count = ReadU32LE(bytes.Data() + offset + 8u);
        const std::size_t payloadBytes = static_cast<std::size_t>(count) * 4u;
        if (count == 0u || count > MaximumVehicleCollectorCameras ||
            offset + 12u > bytes.Size() ||
            payloadBytes > bytes.Size() - (offset + 12u)) {
            continue;
        }
        bool valid = true;
        for (u32 index = 0u; index < count; ++index) {
            const u32 nodeIndex = ReadU32LE(
                    bytes.Data() + offset + 12u + index * 4u);
            const auto declared = std::find_if(
                    references.externalReferences.begin(),
                    references.externalReferences.end(),
                    [nodeIndex](const GbxBodyExternalReference &reference) {
                        return reference.nodeIndex == nodeIndex;
                    });
            if (declared == references.externalReferences.end()) {
                valid = false;
                break;
            }
        }
        if (!valid || camerasChunk.has_value()) {
            return false;
        }
        camerasChunk = offset;
    }
    if (!camerasChunk) {
        return false;
    }

    std::optional<u32> decodedDefault;
    for (std::size_t offset = references.bodyOffset;
         offset + 8u <= *camerasChunk; ++offset) {
        if (ReadU32LE(bytes.Data() + offset) !=
                VehicleCollectorDefaultCameraChunk) {
            continue;
        }
        const u32 value = ReadU32LE(bytes.Data() + offset + 4u);
        if (decodedDefault.has_value()) {
            return false;
        }
        decodedDefault = value;
    }

    const u32 count = ReadU32LE(bytes.Data() + *camerasChunk + 8u);
    if (!decodedDefault || *decodedDefault >= count) {
        return false;
    }
    cameraNodeIndices->clear();
    cameraNodeIndices->reserve(count);
    for (u32 index = 0u; index < count; ++index) {
        cameraNodeIndices->push_back(ReadU32LE(
                bytes.Data() + *camerasChunk + 12u + index * 4u));
    }
    *defaultCameraIndex = *decodedDefault;
    return true;
}

bool ResolveDeclaredRaceCameras(
        CPlugFilePack &pack,
        const InstalledVehicleAssetReference &source,
        const GbxBodyReferenceTable &references,
        const std::vector<u32> &cameraNodeIndices,
        std::vector<InstalledVehicleAssetReference> *out) {
    if (out == nullptr) {
        return false;
    }
    out->clear();
    for (std::size_t collectorIndex = 0u;
         collectorIndex < cameraNodeIndices.size(); ++collectorIndex) {
        const u32 nodeIndex = cameraNodeIndices[collectorIndex];
        const auto foundReference = std::find_if(
                references.externalReferences.begin(),
                references.externalReferences.end(),
                [nodeIndex](const GbxBodyExternalReference &reference) {
                    return reference.nodeIndex == nodeIndex;
                });
        if (foundReference == references.externalReferences.end()) {
            return false;
        }
        const GbxBodyExternalReference &reference = *foundReference;
        if (reference.IsResource()) {
            continue;
        }
        std::string logicalPath;
        std::array<char, 512u> selectedPath{};
        if (!references.ResolvePlainPathForReference(
                    source.logicalPath, reference, &logicalPath) ||
            !pack.SelectedPathForPlainRef(
                    logicalPath.c_str(), selectedPath.data(),
                    selectedPath.size())) {
            continue;
        }
        const CPlugFileFidContainer_SFileDesc *file =
                pack.FindFileDescByPath(selectedPath.data());
        if (file == nullptr || !IsRaceCameraClass(file->classId)) {
            continue;
        }
        InstalledVehicleAssetReference camera;
        camera.logicalPath = std::move(logicalPath);
        camera.selectedPath = selectedPath.data();
        camera.classId = file->classId;
        camera.collectorIndex = static_cast<u32>(collectorIndex);
        out->push_back(std::move(camera));
    }
    return true;
}

}  // namespace

std::optional<InstalledVehicleAssetGraph>
InstalledVehicleAssetGraph::ResolveFromPack(CPlugFilePack &pack) {
    try {
        InstalledVehicleAssetGraph graph;
        GbxBodyReferenceTable collectorReferences;
        std::vector<u32> cameraNodeIndices;
        u32 defaultCameraIndex = 0xffffffffu;
        GbxBodyReferenceTable mobilReferences;
        GbxBodyReferenceTable materialReferences;
        GbxBodyReferenceTable bitmapReferences;
        const bool resolved =
                FindUniquePackClass(
                        pack,
                        TMNF_CLASS_CGameCtnCollectorVehicle,
                        &graph.collector) &&
                ParseReferenceTable(
                        pack, graph.collector, &collectorReferences) &&
                ParseCollectorCameraIndices(
                        pack, graph.collector, collectorReferences,
                        &defaultCameraIndex, &cameraNodeIndices) &&
                ResolveDeclaredRaceCameras(
                        pack, graph.collector, collectorReferences,
                        cameraNodeIndices,
                        &graph.raceCameras) &&
                ResolveUniqueDeclaredClass(
                        pack,
                        graph.collector,
                        collectorReferences,
                        TMNF_CLASS_CSceneVehicleCar,
                        &graph.mobil) &&
                ParseReferenceTable(pack, graph.mobil, &mobilReferences) &&
                ResolveUniqueDeclaredClass(
                        pack,
                        graph.mobil,
                        mobilReferences,
                        TMNF_CLASS_CPlugSolid,
                        &graph.solid) &&
                ResolveUniqueDeclaredClass(
                        pack,
                        graph.mobil,
                        mobilReferences,
                        TMNF_CLASS_CSceneVehicleTunings,
                        &graph.tuning) &&
                ResolveUniqueDeclaredClass(
                        pack,
                        graph.mobil,
                        mobilReferences,
                        TMNF_CLASS_CMwRefBuffer,
                        &graph.materials) &&
                ResolveUniqueDeclaredClass(
                        pack,
                        graph.mobil,
                        mobilReferences,
                        TMNF_CLASS_CSceneVehicleStruct,
                        &graph.vehicleStruct) &&
                ParseReferenceTable(
                        pack, graph.materials, &materialReferences) &&
                ResolveUniqueDeclaredClass(
                        pack,
                        graph.materials,
                        materialReferences,
                        TMNF_CLASS_CPlugBitmap,
                        &graph.fakeContactBitmap) &&
                ParseReferenceTable(
                        pack, graph.fakeContactBitmap, &bitmapReferences) &&
                ResolveUniqueDeclaredClass(
                        pack,
                        graph.fakeContactBitmap,
                        bitmapReferences,
                        TMNF_CLASS_CPlugFileTga,
                        &graph.fakeContactTexture);
        if (!resolved || !graph.IsComplete()) {
            return std::nullopt;
        }
        return graph;
    } catch (const std::bad_alloc &) {
        return std::nullopt;
    }
}

std::optional<InstalledVehicleCameraSet>
InstalledVehicleAssetGraph::ResolveRaceCamerasFromPack(CPlugFilePack &pack) {
    try {
        InstalledVehicleAssetReference collector;
        GbxBodyReferenceTable references;
        InstalledVehicleCameraSet cameraSet;
        std::vector<u32> cameraNodeIndices;
        if (!FindUniquePackClass(
                    pack, TMNF_CLASS_CGameCtnCollectorVehicle, &collector) ||
            !ParseReferenceTable(pack, collector, &references) ||
            !ParseCollectorCameraIndices(
                    pack, collector, references,
                    &cameraSet.defaultCameraIndex, &cameraNodeIndices) ||
            !ResolveDeclaredRaceCameras(
                    pack, collector, references, cameraNodeIndices,
                    &cameraSet.cameras) ||
            cameraSet.cameras.empty()) {
            return std::nullopt;
        }
        return cameraSet;
    } catch (const std::bad_alloc &) {
        return std::nullopt;
    }
}

bool InstalledVehicleAssetGraph::IsComplete() const {
    return collector.IsValid() &&
           collector.classId == TMNF_CLASS_CGameCtnCollectorVehicle &&
           mobil.IsValid() && mobil.classId == TMNF_CLASS_CSceneVehicleCar &&
           solid.IsValid() && solid.classId == TMNF_CLASS_CPlugSolid &&
           tuning.IsValid() &&
           tuning.classId == TMNF_CLASS_CSceneVehicleTunings &&
           materials.IsValid() && materials.classId == TMNF_CLASS_CMwRefBuffer &&
           vehicleStruct.IsValid() &&
           vehicleStruct.classId == TMNF_CLASS_CSceneVehicleStruct &&
           fakeContactBitmap.IsValid() &&
           fakeContactBitmap.classId == TMNF_CLASS_CPlugBitmap &&
           fakeContactTexture.IsValid() &&
           fakeContactTexture.classId == TMNF_CLASS_CPlugFileTga;
}
