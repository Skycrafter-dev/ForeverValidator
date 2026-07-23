#ifndef TMNF_REPLAY_MAP_SCENE_H
#define TMNF_REPLAY_MAP_SCENE_H

#include <memory>

#include "engine/physics/collision/hms_collision_manager.h"
#include "engine/scene/replay_scene_placements.h"
#include "engine/scene/replay_static_scene_corpuses.h"
#include "engine/scene/static_scene_model.h"
class CGameCtnChallenge;
class CGameCtnChallengeConstruction;
class CSceneVehicleCar;
class CTrackManiaRace;

enum class ReplayMapSceneResult {
    Ready,
    Inactive,
    ChallengeUnavailable,
    StaticCorpusConstructionFailed,
    MissingModelSources,
    CorpusCountMismatch,
    StationaryCorpusInstallFailed,
};

class ReplayMapScene {
public:
    ReplayMapScene() = default;
    ~ReplayMapScene();

    ReplayMapScene(const ReplayMapScene &) = delete;
    ReplayMapScene &operator=(const ReplayMapScene &) = delete;

    void Reset(CTrackManiaRace &race);
    ReplayMapSceneResult PreloadChallenge(
            CGameCtnChallengeConstruction &construction);
    ReplayMapSceneResult InstallModels(StaticSceneModelCollection models);
    void Activate();
    bool IsActive() const { return active_; }

    ReplayMapSceneResult EnsureReady(CTrackManiaRace &race);
    ReplayMapSceneResult SelectCollisionZone(
            CHmsCollisionManagerSZone &localZone,
            CSceneVehicleCar *vehicle,
            CTrackManiaRace &race,
            CHmsCollisionManagerSZone *&selectedZone);
    bool FirstStartLineSpawnLocation(GmIso4 &location) const;
    const CHmsCollisionManagerSZone &PersistentCollisionZoneForTesting(
            void) const noexcept {
        return persistentCollisionZone_;
    }

private:
    ReplayMapSceneResult BuildStaticCorpuses();
    ReplayMapSceneResult InstallStationaryCorpuses(
            CSceneVehicleCar *vehicle,
            CTrackManiaRace &race);

    std::unique_ptr<CGameCtnChallenge> challenge_;
    ReplaySceneBlockPlacements blockPlacements_;
    StaticSceneModelCollection models_;
    ReplayStaticCorpusCollection staticCorpuses_;
    ReplayDedicatedCollisionCorpusCollection dedicatedCollisionCorpuses_;
    CHmsCollisionManagerSZone persistentCollisionZone_{0u, nullptr};
    bool active_ = false;
    bool ready_ = false;
    bool collisionZoneConstructed_ = false;
    bool stationaryCorpusesInstalled_ = false;
};

#endif
