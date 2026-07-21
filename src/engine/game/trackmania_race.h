#ifndef TMNF_TRACKMANIA_RACE_H
#define TMNF_TRACKMANIA_RACE_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "engine/core/engine_types.h"
#include "engine/game/game_ctn_block.h"
#include "engine/game/game_ctn_types.h"
#include "engine/physics/collision/hms_collision.h"
#include "engine/physics/dynamics/hms_item.h"
#include "engine/game/replay_checkpoint_trigger.h"
#include "engine/game/trackmania_player.h"
#include "engine/scene/scene_mobil.h"
#include "engine/scene/scene_vehicle_car.h"
class ReplayStaticCorpusCollection;

struct ReplayRaceProgress {
    u32 installedTriggerCount = 0u;
    u32 preparedEventCount = 0u;
    u32 checkpointCount = 0u;
    u32 finishCount = 0u;
    u32 freewheelClearCount = 0u;
    u32 lastBlockRole = 0u;
    u32 lastPrepareTimeMs = 0u;
    u32 lastAcceptedBlockId = 0u;
    u32 lastContactBlockId = 0u;
    u32 currentLapCheckpointCount = 0u;
    u32 totalCheckpointEventCount = 0u;
    u32 completedLapCount = 0u;
    u32 requiredLapCount = 1u;
    u32 requiredCheckpointCount = 0u;
    bool raceCompleted = false;
};

enum EFigures : u32 {
    EFigures_Unknown = 0u,
};

struct ReplayStuntSimulationState {
    u32 tickTimeMs = 0u;
    u32 inputQueryTimeOffsetMs = 0u;
    bool raceStart = false;
    bool finishRace = false;
    GmIso4 vehicleLocation{};
    float forwardSpeed = 0.0f;
    float sideSpeed = 0.0f;
    bool hasWheelContact = false;
    bool hasBodyContact = false;
    float bodyContactVerticalAngle = 0.0f;
    float bodyContactHorizontalAngle = 0.0f;
    bool noGroundFrictionGuard = false;
    std::array<u32, 6u> inputLastChangeTimeMs{};
};

struct ReplayStuntEvent {
    EFigures figure = EFigures_Unknown;
    u32 degree = 0u;
    u32 score = 0u;
    float bonus = 0.0f;
    bool straightLanding = false;
    bool reverseLanding = false;
    bool masterJump = false;
    u32 chain = 0u;
};

class CTrackManiaRace : public ReplayCheckpointContactObserver {
public:
    struct ReplayStuntInputSnapshot {
        u32 tickTimeMs = 0u;
        std::array<u32, 6u> lastChangeTimeMs{};
    };

    struct RuntimeClone {
        CTrackManiaPlayer player{};
        std::vector<std::uint8_t> checkpointSlotsPassed;
        std::optional<GmIso4> playerSpawnLocation;
        std::optional<GmIso4> lastAcceptedSpawnLocation;
        bool currentSpawnLocationInitialized = false;
        u32 preparedEventTimeMs = 0u;
        EChallengePlayMode replayPlayMode = EChallengePlayMode::Race;
        u32 replayNbLaps = 1u;
        ReplayRaceProgress progress{};
        bool replayStuntsEnabled = false;
        bool replayStuntStateAvailable = false;
        u32 replayStuntsTimeLimitMs = 0u;
        u32 replayStuntsRaceStartTimeMs = 0u;
        ReplayStuntSimulationState replayStuntState{};
        std::array<ReplayStuntInputSnapshot, 32u> replayStuntInputHistory{};
        std::size_t replayStuntInputHistorySize = 0u;
        std::array<GmIso4, 20u> replayStuntLocationHistory{};
        std::size_t replayStuntLocationHistorySize = 0u;
        GmIso4 replayStuntPreviousLocation{};
        GmIso4 replayStuntTakeoffLocation{};
        GmVec3 replayStuntRotation{};
        float replayStuntLandingDirection = 0.0f;
        u32 replayStuntTakeoffTick = UINT32_MAX;
        u32 replayStuntLandingTick = UINT32_MAX;
        u32 replayStuntPreviousLandingTick = UINT32_MAX;
        u32 replayStuntChain = 0u;
        u32 replayStuntComboWindowMs = 0u;
        bool replayStuntInProgress = false;
        bool replayStuntMasterJump = false;
        bool replayStuntBadLanding = false;
        std::optional<u32> replayStuntScoreAtTimeLimit;
        std::array<u32, 39u> replayStuntFigureScores{};
        u32 stuntsScore = 0u;
        std::vector<ReplayStuntEvent> stuntEvents;
    };
    static float s_MasterJumpFactor;
    static unsigned long s_ReverseBonus;
    static unsigned long s_TimeBonus;
    static unsigned long s_InterComboDelay;
    static unsigned long s_MinStuntTime;
    static unsigned long s_MinGrindTime;
    static unsigned long s_MaxGrindInterval;
    static float s_MaxGrindRotation;
    static float s_FigureRepeatMalus;
    static unsigned long s_FigureRepeatCompensation;
    static float s_ChainBonus1;
    static float s_ChainBonus2;
    static float s_ChainBonus3;
    static float s_ChainBonus4;
    static float s_ChainBonus5;
    static unsigned long s_StuntRespawnPenalty;

    void ResetValidationSession();
    void BindVehicle(CSceneVehicleCar *vehicle);
    void BindCheckpointCourse(
            ReplayStaticCorpusCollection *course);
    void SetInitialSpawnLocation(const GmIso4 &spawnIso);
    bool HasRespawnLocation() const;
    const GmIso4 &RespawnLocation() const;
    const ReplayRaceProgress &Progress() const { return progress_; }

    CTrackManiaPlayer *GetPlayerFromMobil(CSceneMobil *mobil);
    CTrackManiaPlayer *GetPlayingPlayer(void);
    CGameCtnBlock *GetBlockFromCheckpointMobil(CSceneMobil *mobil);
    void InternalPrepareEvent(CTrackManiaPlayer *player);
    virtual void OnCheckpoint(CTrackManiaPlayer *player,
                              CGameCtnBlock *block);
    virtual void OnFinishLine(CTrackManiaPlayer *player,
                              CGameCtnBlock *block);
    void PrepareCheckpoints();
    void InitNbLapsAndCheckpoints(unsigned long nbLaps);
    virtual int InternalOnCheckpoint(
            unsigned long raceTime,
            unsigned long score,
            unsigned long checkpointIndex,
            unsigned long checkpointSlot,
            CTrackManiaPlayerInfo *playerInfo,
            const GmIso4 *spawnIso,
            const CSceneMobil *triggerMobil,
            int playSound);
    void SetReplayChallengePlayMode(EChallengePlayMode playMode);

    int IsReverseLanding(void);
    int IsStraightLanding(void);
    unsigned long GetTimePenalty(unsigned long timeMs);
    void UpdateStuntTime(void);
    virtual void DisplayStuntMessages(
            EFigures figure,
            unsigned long degree,
            unsigned long score,
            float bonus,
            int straightLanding,
            int reverseLanding,
            int masterJump,
            unsigned long chain);
    int IsStuntTimeOver(unsigned long tickTimeMs);
    int IsMasterJump(unsigned long startTimeMs,
                     unsigned long endTimeMs);
    void ResetStunts(void);
    void UpdateStunts(void);
    virtual void ComputeStunt(void);

    void ConfigureReplayStuntsSimulation(bool enabled,
                                         u32 timeLimitMs);
    void SetReplayStuntSimulationState(
            const ReplayStuntSimulationState &state);
    void ApplyReplayStuntTimePenalty(unsigned long overtimeMs);
    void ApplyReplayStuntRespawnPenalty();
    void ApplyReplayStuntRespawnPenalty(unsigned long tickTimeMs);
    u32 StuntsScore() const { return stuntsScore_; }
    const std::vector<ReplayStuntEvent> &StuntEvents() const {
        return stuntEvents_;
    }

    void OnCheckpointContact(CHmsItem &item,
                             CHmsPhysicalContact &contact) override;
    RuntimeClone CaptureRuntimeClone() const;
    bool PrepareRuntimeCloneRestore(const RuntimeClone &clone);
    void RestoreRuntimeClone(RuntimeClone clone) noexcept;

private:
    static constexpr std::size_t ReplayStuntInputHistoryCount = 32u;
    static constexpr std::size_t ReplayStuntLocationHistoryCount = 20u;

    void StoreSpawnLocation(const GmIso4 &spawnIso);
    void EnsurePreviousSpawnLocationInitialized(const GmIso4 &spawnIso);
    void ResetCheckpointSlots();
    void ClearVehicleFreewheelState();
    void PushReplayStuntInputSnapshot();
    void PushReplayStuntVehicleLocation();

    CTrackManiaPlayer player{};
    CSceneVehicleCar *vehicle = nullptr;
    ReplayStaticCorpusCollection *checkpointCourse = nullptr;
    std::vector<std::uint8_t> checkpointSlotsPassed_{};
    std::optional<GmIso4> playerSpawnLocation_;
    std::optional<GmIso4> lastAcceptedSpawnLocation_;
    bool currentSpawnLocationInitialized_ = false;
    u32 preparedEventTimeMs_ = 0u;
    EChallengePlayMode replayPlayMode_ = EChallengePlayMode::Race;
    u32 replayNbLaps_ = 1u;
    ReplayRaceProgress progress_{};

    bool replayStuntsEnabled_ = false;
    bool replayStuntStateAvailable_ = false;
    u32 replayStuntsTimeLimitMs_ = 0u;
    u32 replayStuntsRaceStartTimeMs_ = 0u;
    ReplayStuntSimulationState replayStuntState_{};
    std::array<ReplayStuntInputSnapshot,
               ReplayStuntInputHistoryCount> replayStuntInputHistory_{};
    std::size_t replayStuntInputHistorySize_ = 0u;
    std::array<GmIso4,
               ReplayStuntLocationHistoryCount> replayStuntLocationHistory_{};
    std::size_t replayStuntLocationHistorySize_ = 0u;
    GmIso4 replayStuntPreviousLocation_{};
    GmIso4 replayStuntTakeoffLocation_{};
    GmVec3 replayStuntRotation_{};
    float replayStuntLandingDirection_ = 0.0f;
    u32 replayStuntTakeoffTick_ = UINT32_MAX;
    u32 replayStuntLandingTick_ = UINT32_MAX;
    u32 replayStuntPreviousLandingTick_ = UINT32_MAX;
    u32 replayStuntChain_ = 0u;
    u32 replayStuntComboWindowMs_ = 0u;
    bool replayStuntInProgress_ = false;
    bool replayStuntMasterJump_ = false;
    bool replayStuntBadLanding_ = false;
    std::optional<u32> replayStuntScoreAtTimeLimit_;
    std::array<u32, 39u> replayStuntFigureScores_{};
    u32 stuntsScore_ = 0u;
    std::vector<ReplayStuntEvent> stuntEvents_;

    friend struct CTrackManiaRaceTestPeer;
};

#endif
