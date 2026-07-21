#ifndef TMNF_REPLAY_VEHICLE_BODY_H
#define TMNF_REPLAY_VEHICLE_BODY_H

#include <optional>

#include "engine/physics/dynamics/replay_dyna_parameters.h"
#include "simulation/runtime/replay_dyna_frame_state.h"
#include "engine/physics/dynamics/hms_corpus.h"
#include "engine/physics/dynamics/hms_dyna.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/scene/plug_solid.h"
class ReplayVehicleBody {
public:
    using RuntimeClone = CHmsDyna::RuntimeClone;
    ReplayVehicleBody();

    void InitializeAtSpawn(const ReplayDynaParameters &parameters,
                           const GmIso4 &spawnLocation);
    void SetSpawnLocation(const std::optional<GmIso4> &location);
    void ConstructItem(const CHmsItem::Properties &properties);
    void BuildCorpus();
    ReplayDynaParameters CaptureDynaParameters() const;
    ReplayDynaFrameState CaptureCurrentFrame() const;
    ReplayDynaFrameState CaptureWriteState() const;
    void InstallDynaParameters(const ReplayDynaParameters &parameters);
    void InstallPhysicalParameters(const ReplayDynaParameters &parameters);
    void InstallEmptyCollisionTree();

    CHmsDyna &Dyna();
    CHmsItem &Item();
    CPlugSolid &Solid();
    CHmsCorpus &Corpus();
    RuntimeClone CaptureRuntimeClone() const;
    bool PrepareRuntimeCloneRestore(const RuntimeClone &clone);
    void RestoreRuntimeClone(RuntimeClone clone) noexcept;

private:
    CHmsDyna dyna;
    CHmsItem item;
    CHmsCorpus corpus;
};

#endif // TMNF_REPLAY_VEHICLE_BODY_H
