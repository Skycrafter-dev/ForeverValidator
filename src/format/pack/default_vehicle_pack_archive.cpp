#include "format/pack/default_vehicle_pack_archive.h"
#include "format/materials/tga_image.h"
#include "format/vehicle_tuning/vehicle_tuning_archive.h"
#include <array>
#include <stddef.h>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

#include "format/archive/classic_archive_reader.h"
#include "format/pack/installed/plug_file_pack.h"
#include "format/pack/installed_vehicle_asset_graph.h"
#include "format/pack/installed/plug_file_pack_crypto_constants.h"
#include "format/archive/scene_vehicle_car_archive_schema.h"
#include "format/vehicle_tuning/default_vehicle_archive_schema.h"
#include "format/archive/tmnf_archive_ids.h"
#include "format/archive/tmnf_gbx_body_reader.h"
#include <new>
constexpr u32 DefaultVehicleMaterialNaturalIdCount = 31u;
constexpr u32 ClassicArchiveEndMarker = 0xfacade01u;
constexpr u32 ClassicArchiveSkipMarker = 0x534b4950u;
constexpr std::array<const char *, OfficialVehicleWheelCount>
        DefaultVehicleWheelSurfaceNames{
                "FLSurf",
                "FRSurf",
                "RRSurf",
                "RLSurf",
        };

static int extract_pack_path_bytes(
        CPlugFilePack &pack,
        const char *path,
        std::vector<uint8_t> *out) {
    if (path == nullptr || out == nullptr) {
        return 0;
    }
    out->clear();
    ByteBuffer bytes;
    if (!pack.ExtractPath(path, &bytes) || bytes.Empty()) {
        return 0;
    }
    try {
        out->assign(bytes.Data(), bytes.Data() + bytes.Size());
    } catch (const std::bad_alloc &) {
        out->clear();
        return 0;
    }
    return 1;
}

static int parse_default_vehicle_mobil_member(
        const std::vector<uint8_t> &bytes,
        ReplayVehicleSourceDefinition *packData) {
    u32 classId = 0u;
    u32 bodyOffset = 0u;
    if (packData == nullptr ||
        bytes.size() > UINT32_MAX ||
        !GbxBodyOffsetReader::TryParse(bytes.data(),
                                       (u32)bytes.size(),
                                       &classId,
                                       &bodyOffset) ||
        classId != TMNF_CLASS_CSceneVehicleCar) {
        return 0;
    }
    constexpr size_t PhysicalParameterByteCount = 8u * sizeof(float);
    constexpr size_t TerminalChunkByteCount =
            sizeof(u32) + PhysicalParameterByteCount + sizeof(u32);
    if (bytes.size() < TerminalChunkByteCount ||
        bytes.size() - TerminalChunkByteCount < bodyOffset) {
        return 0;
    }
    TmnfFormat::ClassicArchiveReader archive;
    u32 chunkId = 0u;
    u32 endMarker = 0u;
    VehicleInitialParameters parameters;
    if (!archive.Open(bytes.data(), bytes.size(), 0u) ||
        !archive.Seek(bytes.size() - TerminalChunkByteCount) ||
        !archive.ReadU32(chunkId) ||
        chunkId != ArchiveChunkIdValue(
                           CSceneVehicleCarArchiveChunkId::PhysicalParams) ||
        !archive.ReadF32(parameters.linearSpeedCap) ||
        !archive.ReadF32(parameters.reverseGearSpeedThreshold) ||
        !archive.ReadF32(parameters.waterBoxLocal.center.x) ||
        !archive.ReadF32(parameters.waterBoxLocal.center.y) ||
        !archive.ReadF32(parameters.waterBoxLocal.center.z) ||
        !archive.ReadF32(parameters.waterBoxLocal.halfExtents.x) ||
        !archive.ReadF32(parameters.waterBoxLocal.halfExtents.y) ||
        !archive.ReadF32(parameters.waterBoxLocal.halfExtents.z) ||
        !archive.ReadU32(endMarker) ||
        endMarker != ClassicArchiveEndMarker || archive.Remaining() != 0u) {
        return 0;
    }
    packData->initialParameters = parameters;
    return 1;
}

static int parse_default_vehicle_struct_member(
        const std::vector<uint8_t> &bytes,
        ReplayVehicleSourceDefinition *packData) {
    u32 classId = 0u;
    u32 offset = 0u;
    if (packData == nullptr ||
        bytes.size() > UINT32_MAX ||
        !GbxBodyOffsetReader::TryParse(bytes.data(),
                                       (u32)bytes.size(),
                                       &classId,
                                       &offset) ||
        classId != TMNF_CLASS_CSceneVehicleStruct ||
        offset > bytes.size()) {
        return 0;
    }
    static constexpr size_t VehicleStructStoredCMwIdNameLimit = 32u;
    TmnfFormat::ClassicArchiveReader archive;
    if (!archive.Open(bytes.data(),
                      bytes.size(),
                      VehicleStructStoredCMwIdNameLimit) ||
        !archive.Seek(offset)) {
        return 0;
    }
    u32 chunkId = 0u;
    u32 count = 0u;
    if (!archive.ReadU32(chunkId) ||
        chunkId != ArchiveChunkIdValue(
                           CSceneVehicleStructArchiveChunkId::
                                   SimulationWheelBuffer) ||
        !archive.ReadU32(count) || count != OfficialVehicleWheelCount) {
        return 0;
    }
    for (u32 i = 0u; i < count; i++) {
        u32 side = 0u;
        u32 bucket = 0u;
        TmnfFormat::ClassicArchiveIdentifier surface;
        if (!archive.ReadU32(side) ||
            !archive.ReadU32(bucket) ||
            !archive.ReadIdentifier(surface)) {
            return 0;
        }
        if (surface.text != DefaultVehicleWheelSurfaceNames[i]) {
            return 0;
        }
        packData->wheelContactSettings[i] = VehicleWheelContactSettings{
                VehicleWheelSurfaceIdForIndex(i),
                side != 0u,
                ((bucket) == 0u
            ? VehicleWheelAxle::Rear
            : VehicleWheelAxle::Front),
        };
    }
    return packData->HasCompleteWheelContactSettings() ? 1 : 0;
}

static int skip_unknown_archive_chunk(
        TmnfFormat::ClassicArchiveReader &archive) {
    u32 marker = 0u;
    u32 byteCount = 0u;
    return archive.ReadU32(marker) && marker == ClassicArchiveSkipMarker &&
           archive.ReadU32(byteCount) && archive.Skip(byteCount);
}

static int parse_vehicle_material_chunks(
        TmnfFormat::ClassicArchiveReader &archive,
        VehicleMaterialDefinition *materialOut) {
    if (materialOut == nullptr) {
        return 0;
    }
    VehicleMaterialDefinition material;
    bool hasBitmapAndPeriod = false;
    bool hasBlendableValues = false;
    bool hasNaturalIdAndFakeContact = false;
    bool hasFeedbackResponse = false;
    for (;;) {
        u32 chunkId = 0u;
        if (!archive.ReadU32(chunkId)) {
            return 0;
        }
        if (chunkId == ClassicArchiveEndMarker) {
            break;
        }
        switch (chunkId) {
            case ArchiveChunkIdValue(
                    CSceneVehicleMaterialArchiveChunkId::
                            FakeContactBitmapAndPeriod): {
                u32 textureReference = 0u;
                if (hasBitmapAndPeriod ||
                    !archive.ReadU32(textureReference) ||
                    !archive.ReadF32(material.fakeContactPeriodX) ||
                    !archive.ReadF32(material.fakeContactPeriodZ)) {
                    return 0;
                }
                material.usesFakeContactTexture =
                        textureReference != UINT32_MAX;
                hasBitmapAndPeriod = true;
                break;
            }
            case ArchiveChunkIdValue(
                    CSceneVehicleMaterialArchiveChunkId::BlendableValues):
                if (hasBlendableValues ||
                    !archive.ReadF32(material.blendableValues.x) ||
                    !archive.ReadF32(material.blendableValues.w) ||
                    !archive.ReadF32(material.blendableValues.y) ||
                    !archive.ReadF32(material.blendableValues.z)) {
                    return 0;
                }
                hasBlendableValues = true;
                break;
            case ArchiveChunkIdValue(
                    CSceneVehicleMaterialArchiveChunkId::
                            NaturalIdAndFakeContact): {
                std::uint8_t naturalId = 0u;
                if (hasNaturalIdAndFakeContact ||
                    !archive.ReadU8(naturalId) ||
                    !archive.ReadF32(material.fakeContactSpeedScale) ||
                    !archive.ReadF32(material.fakeContactDepthMax)) {
                    return 0;
                }
                material.naturalId = naturalId;
                hasNaturalIdAndFakeContact = true;
                break;
            }
            case ArchiveChunkIdValue(
                    CSceneVehicleMaterialArchiveChunkId::FeedbackResponse):
                if (hasFeedbackResponse ||
                    !archive.ReadF32(material.feedbackSpeedDivisor) ||
                    !archive.ReadF32(material.feedbackScale)) {
                    return 0;
                }
                hasFeedbackResponse = true;
                break;
            default:
                if (!skip_unknown_archive_chunk(archive)) {
                    return 0;
                }
                break;
        }
    }
    if (!hasBitmapAndPeriod || !hasBlendableValues ||
        !hasNaturalIdAndFakeContact || !hasFeedbackResponse ||
        material.naturalId >= DefaultVehicleMaterialNaturalIdCount) {
        return 0;
    }
    *materialOut = material;
    return 1;
}

static int parse_default_vehicle_materials_member(
        const std::vector<uint8_t> &bytes,
        ReplayVehicleSourceDefinition *packData) {
    u32 classId = 0u;
    u32 bodyOffset = 0u;
    if (packData == nullptr ||
        bytes.size() > UINT32_MAX ||
        !GbxBodyOffsetReader::TryParse(bytes.data(),
                                       (u32)bytes.size(),
                                       &classId,
                                       &bodyOffset) ||
        classId != TMNF_CLASS_CMwRefBuffer) {
        return 0;
    }
    VehicleMaterialSetDefinition materialSet;
    try {
        materialSet.materials.reserve(DefaultVehicleMaterialNaturalIdCount);
    } catch (const std::bad_alloc &) {
        return 0;
    }
    TmnfFormat::ClassicArchiveReader archive;
    u32 rootClassId = 0u;
    u32 elementClassId = 0u;
    u32 serializationVersion = 0u;
    u32 allocationIncrement = 0u;
    u32 materialCount = 0u;
    if (!archive.Open(bytes.data(), bytes.size(), 0u) ||
        !archive.Seek(bodyOffset) ||
        !archive.ReadU32(rootClassId) || rootClassId != TMNF_CLASS_CMwRefBuffer ||
        !archive.ReadU32(elementClassId) ||
        elementClassId != TMNF_CLASS_CSceneVehicleMaterial ||
        !archive.ReadU32(serializationVersion) || serializationVersion != 1u ||
        !archive.ReadU32(allocationIncrement) || allocationIncrement == 0u ||
        !archive.ReadU32(materialCount) || materialCount == 0u ||
        materialCount > DefaultVehicleMaterialNaturalIdCount) {
        return 0;
    }
    u32 previousNodeReference = 0u;
    for (u32 materialIndex = 0u;
         materialIndex < materialCount;
         materialIndex++) {
        u32 nodeReference = 0u;
        u32 nodeClassId = 0u;
        VehicleMaterialDefinition material;
        if (!archive.ReadU32(nodeReference) ||
            nodeReference <= previousNodeReference ||
            !archive.ReadU32(nodeClassId) ||
            nodeClassId != TMNF_CLASS_CSceneVehicleMaterial ||
            !parse_vehicle_material_chunks(archive, &material)) {
            return 0;
        }
        previousNodeReference = nodeReference;
        materialSet.materials.push_back(material);
    }
    u32 rootEndMarker = 0u;
    if (!archive.ReadU32(rootEndMarker) ||
        rootEndMarker != ClassicArchiveEndMarker ||
        archive.Remaining() != 0u) {
        return 0;
    }
    materialSet.materialIndexByNaturalId.assign(
            DefaultVehicleMaterialNaturalIdCount,
            0u);
    for (u32 i = 0u; i < materialSet.materials.size(); i++) {
        materialSet.materialIndexByNaturalId[
                materialSet.materials[i].naturalId] = i;
    }
    packData->materials = std::move(materialSet);
    return 1;
}

static int parse_default_vehicle_fake_contact_texture(
        const std::vector<uint8_t> &bytes,
        ReplayVehicleSourceDefinition *packData) {
    if (packData == nullptr || !packData->materials.has_value()) {
        return 0;
    }
    std::optional<TgaFormat::TrueColorImage> image =
            TgaFormat::DecodeUncompressedTrueColor(bytes.data(), bytes.size());
    if (!image.has_value() ||
        !image->HasDimensions(
                static_cast<std::uint16_t>(VehicleFakeContactTextureWidth),
                static_cast<std::uint16_t>(VehicleFakeContactTextureHeight),
                static_cast<std::uint8_t>(
                        VehicleFakeContactTextureBytesPerPixel * 8u))) {
        return 0;
    }
    VehicleFakeContactTextureDefinition texture;
    texture.width = image->width;
    texture.height = image->height;
    texture.rgbPixels = std::move(image->storedPixels);
    packData->materials->fakeContactTexture = std::move(texture);
    return 1;
}

std::optional<DefaultVehiclePackData>
DefaultVehiclePackArchive::LoadFromPack(
        CPlugFilePack &pack) {
    std::optional<InstalledVehicleAssetGraph> assets =
            InstalledVehicleAssetGraph::ResolveFromPack(pack);
    return assets.has_value()
            ? LoadFromPack(pack, *assets)
            : std::nullopt;
}

std::optional<DefaultVehiclePackData>
DefaultVehiclePackArchive::LoadFromPack(
        CPlugFilePack &pack,
        const InstalledVehicleAssetGraph &assets) {
// GCC's -Wmaybe-uninitialized produces a false positive here: it cannot prove
// that the nested std::optional<VehicleMaterialSetDefinition> inside `result`
// only ever holds a fully constructed VehicleFakeContactTextureDefinition
// vector. Clang does not emit this warning and does not support the flag, so
// the suppression is scoped to GCC only.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    if (!assets.IsComplete()) {
        return std::nullopt;
    }
    try {
        std::vector<uint8_t> carBytes;
        std::vector<uint8_t> materialBytes;
        std::vector<uint8_t> structBytes;
        std::vector<uint8_t> tuningBytes;
        std::vector<uint8_t> fakeContactTextureBytes;
        DefaultVehiclePackData result;
        const bool loaded =
                extract_pack_path_bytes(
                        pack, assets.tuning.selectedPath.c_str(), &tuningBytes) &&
                extract_pack_path_bytes(pack, assets.mobil.selectedPath.c_str(),
                                        &carBytes) &&
                extract_pack_path_bytes(pack, assets.materials.selectedPath.c_str(),
                                        &materialBytes) &&
                extract_pack_path_bytes(pack, assets.vehicleStruct.selectedPath.c_str(),
                                        &structBytes) &&
                extract_pack_path_bytes(
                        pack,
                        assets.fakeContactTexture.selectedPath.c_str(),
                        &fakeContactTextureBytes) &&
                parse_default_vehicle_mobil_member(carBytes, &result.vehicle) &&
                parse_default_vehicle_materials_member(materialBytes,
                                                       &result.vehicle) &&
                parse_default_vehicle_struct_member(structBytes,
                                                    &result.vehicle) &&
                parse_default_vehicle_fake_contact_texture(
                        fakeContactTextureBytes, &result.vehicle);
        std::optional<ReplayVehicleTuningDefinition> tuning =
                VehicleTuningArchive::Decode(tuningBytes);
        if (!loaded || !tuning.has_value()) {
            return std::nullopt;
        }
        result.tuning = std::move(*tuning);
        return result;
    } catch (const std::bad_alloc &) {
        return std::nullopt;
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
}
