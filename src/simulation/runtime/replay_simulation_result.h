#ifndef TMNF_REPLAY_SIMULATION_RESULT_H
#define TMNF_REPLAY_SIMULATION_RESULT_H

enum class ReplayMapSceneResult;

enum class ReplaySimulationRunResult {
    Success,
    InvalidControlTimeline,
    MapStartUnavailable,
    StaticSceneConstructionFailed,
    StaticSceneSourcesMissing,
    StaticSceneCorpusCountMismatch,
    StaticSceneInstallFailed,
    VehicleCollisionModelFailed,
    ObservationAllocationFailed,
    DeterministicExecutionUnavailable,
    CudaUnavailable,
    CudaInitializationFailed,
    CudaExecutionFailed,
    TwinUnavailable,
    TwinNotImplemented,
    TwinExecutionFailed,
};

ReplaySimulationRunResult MapReplaySceneResult(
        ReplayMapSceneResult result);

#endif
