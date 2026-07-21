#include "forevervalidator/json.h"

#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <new>
#include <string>

namespace forevervalidator {
namespace {

const char *ValidationStatusName(ValidationStatus status) {
    switch (status) {
    case ValidationStatus::Valid: return "valid";
    case ValidationStatus::ValidPrefix: return "valid_prefix";
    case ValidationStatus::WrongSimulation: return "wrong_simulation";
    case ValidationStatus::IncompleteStandaloneRun:
        return "incomplete_validation_run";
    case ValidationStatus::RaceCompletionUnavailable:
        return "race_completion_unavailable";
    case ValidationStatus::ExpectingCompletedRace:
        return "expecting_completed_race";
    case ValidationStatus::RaceTimeMismatch: return "race_time_mismatch";
    case ValidationStatus::StuntsScoreMismatch:
        return "stunts_score_mismatch";
    case ValidationStatus::RespawnCountMismatch:
        return "respawn_count_mismatch";
    case ValidationStatus::RespawnExpectationUnavailable:
        return "respawn_expectation_unavailable";
    case ValidationStatus::ObservationError:
        return "trajectory_observation_error";
    case ValidationStatus::IncompatibleReplayVersion:
        return "incompatible_replay_version";
    case ValidationStatus::InputUnavailable:
        return "validation_input_unavailable";
    case ValidationStatus::TMInterfaceReplay:
        return "tminterface_replay";
    }
    return "unexpected_validation_status";
}

const char *ReplayProvenanceName(ReplayProvenance provenance) {
    switch (provenance) {
    case ReplayProvenance::Unmarked: return "Unmarked";
    case ReplayProvenance::TMInterface: return "TMInterface";
    }
    return "Unmarked";
}

const char *InputGhostMatchName(InputGhostMatch match) {
    switch (match) {
    case InputGhostMatch::Unavailable: return "Unavailable";
    case InputGhostMatch::Match: return "Match";
    case InputGhostMatch::Mismatch: return "Mismatch";
    }
    return "Unavailable";
}

int ValidationOutcomeCode(ValidationOutcome outcome) {
    switch (outcome) {
    case ValidationOutcome::Invalid: return 0;
    case ValidationOutcome::Valid: return 1;
    case ValidationOutcome::WrongSimulation: return 2;
    case ValidationOutcome::Unavailable:
    case ValidationOutcome::Error:
        return -1;
    }
    return -1;
}

std::uint32_t ObservationErrorCode(
        const std::optional<ObservationError> &error) {
    if (!error.has_value()) {
        return 0u;
    }
    switch (*error) {
    case ObservationError::NonFiniteDistance: return 2u;
    case ObservationError::ReplayMetadataUnavailable: return 3u;
    }
    return 0u;
}

class JsonText {
public:
    bool Append(const char *text) {
        value_ += text;
        return true;
    }

    bool AppendChar(char ch) {
        value_.push_back(ch);
        return true;
    }

    bool AppendFormat(const char *format, ...) {
        char stack[512];
        va_list args;
        va_start(args, format);
        va_list copy;
        va_copy(copy, args);
        const int needed = std::vsnprintf(stack, sizeof(stack), format, args);
        va_end(args);
        if (needed < 0) {
            va_end(copy);
            return false;
        }
        if (static_cast<std::size_t>(needed) < sizeof(stack)) {
            va_end(copy);
            return Append(stack);
        }
        std::string heap(static_cast<std::size_t>(needed) + 1u, '\0');
        std::vsnprintf(heap.data(), heap.size(), format, copy);
        va_end(copy);
        value_.append(heap.data(), static_cast<std::size_t>(needed));
        return true;
    }

    bool AppendJsonString(const std::string &text) {
        AppendChar('"');
        for (const char value : text) {
            const unsigned char codeUnit = static_cast<unsigned char>(value);
            char escaped[8];
            switch (codeUnit) {
            case '"': Append("\\\""); break;
            case '\\': Append("\\\\"); break;
            case '\b': Append("\\b"); break;
            case '\f': Append("\\f"); break;
            case '\n': Append("\\n"); break;
            case '\r': Append("\\r"); break;
            case '\t': Append("\\t"); break;
            default:
                if (codeUnit < 0x20u) {
                    std::snprintf(escaped, sizeof(escaped),
                                  "\\u%04x",
                                  static_cast<unsigned>(codeUnit));
                    Append(escaped);
                } else {
                    AppendChar(value);
                }
                break;
            }
        }
        return AppendChar('"');
    }

    std::string Take() { return std::move(value_); }

private:
    std::string value_;
};

void AppendValidationMessageJson(
        JsonText &json,
        const ValidationReport &report) {
    switch (report.status) {
    case ValidationStatus::WrongSimulation:
        json.AppendFormat(
                "\"Deviates : time=%d, dist=%.9g\"",
                report.maxDeviationTimeMs.value_or(-1),
                static_cast<double>(report.maxDeviationDistance));
        break;
    case ValidationStatus::ExpectingCompletedRace:
        json.Append("\"Expecting completed race\"");
        break;
    case ValidationStatus::RaceTimeMismatch:
        json.Append("\"RaceTime mismatch\"");
        break;
    case ValidationStatus::RespawnCountMismatch:
        json.Append("\"Respawn count mismatch\"");
        break;
    case ValidationStatus::ObservationError:
        json.AppendFormat(
                "\"trajectory observation error %u\"",
                ObservationErrorCode(report.observationError));
        break;
    case ValidationStatus::IncompatibleReplayVersion:
        json.Append(
                "\"Replay version: TMr.6 is not compatible with current "
                "game version: TMr.7\"");
        break;
    case ValidationStatus::TMInterfaceReplay:
        json.Append("\"TMInterface replay is invalid\"");
        break;
    default:
        json.Append("null");
        break;
    }
}

void AppendDeviationJson(
        JsonText &json,
        const std::optional<ValidationDeviation> &deviation) {
    if (!deviation.has_value()) {
        json.Append("null");
        return;
    }
    const ValidationDeviation &value = *deviation;
    json.AppendFormat("{\"sample_index\":%u", value.comparisonOrdinal);
    json.AppendFormat(",\"ghost_t_ms\":%d", value.ghostTimeMs);
    json.AppendFormat(",\"validation_physics_t_ms\":%d",
                      value.simulationTimeMs);
    json.AppendFormat(",\"simulation_state_t_ms\":%d",
                      value.simulationTimeMs);
    json.AppendFormat(",\"distance\":%.9g",
                      static_cast<double>(value.distance));
    json.AppendFormat(",\"simulated_position\":[%.9g,%.9g,%.9g]",
                      static_cast<double>(value.simulatedPosition.x),
                      static_cast<double>(value.simulatedPosition.y),
                      static_cast<double>(value.simulatedPosition.z));
    json.AppendFormat(",\"simulated_write_position\":[%.9g,%.9g,%.9g]",
                      static_cast<double>(value.writePosition.x),
                      static_cast<double>(value.writePosition.y),
                      static_cast<double>(value.writePosition.z));
    json.AppendFormat(",\"target_position\":[%.9g,%.9g,%.9g]",
                      static_cast<double>(value.targetPosition.x),
                      static_cast<double>(value.targetPosition.y),
                      static_cast<double>(value.targetPosition.z));
    json.Append(",\"distance_source\":\"trajectory_observation\"}");
}

void AppendValidationResultJson(
        JsonText &json,
        const ValidationReport &report) {
    json.Append("\"status\":");
    json.AppendJsonString(ValidationStatusName(report.status));
    json.Append(",\"validate_result_code\":");
    const int validationCode = ValidationOutcomeCode(report.outcome);
    if (validationCode >= 0) {
        json.AppendFormat("%d", validationCode);
    } else {
        json.Append("null");
    }
    json.Append(",\"message\":");
    AppendValidationMessageJson(json, report);
    json.AppendFormat(",\"measured_sample_count\":%u",
                      report.measuredSamples);
    json.AppendFormat(",\"expected_sample_count\":%u",
                      report.expectedSamples);
    json.AppendFormat(",\"max_deviation\":%.9g",
                      static_cast<double>(report.maxDeviation));
    json.AppendFormat(",\"max_deviation_time_ms\":%d",
                      report.maxDeviationTimeMs.value_or(-1));
    json.AppendFormat(",\"max_deviation_distance\":%.9g",
                      static_cast<double>(report.maxDeviationDistance));
    json.AppendFormat(",\"compared_exact_ghost_state_count\":%u",
                      report.comparedExactGhostStateCount);
    json.AppendFormat(",\"wrong_simulation\":%s",
                      report.wrongSimulation ? "true" : "false");
    json.Append(",\"input_ghost_match\":");
    json.AppendJsonString(InputGhostMatchName(report.inputGhostMatch));
    json.Append(",\"first_divergence\":");
    AppendDeviationJson(json, report.firstDivergence);
    json.Append(",\"first_exact_deviation\":");
    AppendDeviationJson(json, report.firstExactDeviation);
}

void AppendValidationMetadataJson(
        JsonText &json,
        const ValidationMetadata &metadata) {
    json.Append(",\"replay_file_metadata\":{");
    json.Append("\"replay_provenance\":");
    json.AppendJsonString(ReplayProvenanceName(metadata.replayProvenance));
    json.Append(",\"map_environment\":");
    json.AppendJsonString(MapEnvironmentName(metadata.mapEnvironment));
    json.Append(",\"vehicle_model\":");
    json.AppendJsonString(VehicleModelName(metadata.vehicleModel));
    json.Append(",\"play_mode\":");
    if (metadata.playMode.has_value()) {
        json.AppendJsonString(PlayModeName(*metadata.playMode));
    } else {
        json.Append("null");
    }
    json.Append(",\"expected_stunts_score\":");
    if (metadata.expectedStuntsScore.has_value()) {
        json.AppendFormat("%u", *metadata.expectedStuntsScore);
    } else {
        json.Append("null");
    }
    json.Append(",\"input_status\":1");
    json.Append(",\"input_parse_failed\":false");
    json.Append(",\"state_status\":1");
    json.Append(",\"state_parse_failed\":false");
    json.AppendFormat(",\"sample_count\":%zu", metadata.sampleCount);
    json.AppendFormat(",\"sample_period_ms\":%u", metadata.samplePeriodMs);
    json.AppendFormat(",\"sample_stride\":%zu",
                      metadata.encodedGhostSampleByteCount);
    json.AppendFormat(",\"state_buffer_size\":%zu",
                      metadata.encodedGhostStateByteCount);
    json.AppendFormat(",\"input_duration_ms\":%d",
                      metadata.inputDurationMs);
    json.AppendFormat(",\"expected_race_time_ms\":%d",
                      metadata.expectedRaceTimeMs.value_or(0));
    json.AppendFormat(",\"expected_respawns\":%d",
                      metadata.expectedRespawns.value_or(
                              std::numeric_limits<std::int32_t>::min()));
    json.AppendFormat(",\"action_count\":%zu", metadata.actionCount);
    json.AppendFormat(",\"event_count\":%zu", metadata.eventCount);
    json.AppendFormat(",\"requested_samples\":%u",
                      metadata.requestedSamples);
    json.AppendFormat(",\"expected_samples\":%zu",
                      metadata.expectedSamples);
    json.Append("}");
}

void AppendSimulationOutcomeJson(
        JsonText &json,
        const SimulationOutcome &simulation) {
    json.Append(",\"simulation_outcome\":{");
    json.Append("\"race_completed\":");
    if (simulation.raceCompleted.has_value()) {
        json.Append(*simulation.raceCompleted ? "true" : "false");
    } else {
        json.Append("null");
    }
    json.Append(",\"race_time_ms\":");
    if (simulation.raceTimeMs.has_value()) {
        json.AppendFormat("%d", *simulation.raceTimeMs);
    } else {
        json.Append("null");
    }
    json.Append(",\"stunts_score\":");
    if (simulation.stuntsScore.has_value()) {
        json.AppendFormat("%d", *simulation.stuntsScore);
    } else {
        json.Append("null");
    }
    json.AppendFormat(",\"respawn_count\":%u}", simulation.respawnCount);
}

ValidationError SerializationError(const ReplayIdentity &identity) {
    ValidationError error;
    error.category = ValidationErrorCategory::Serialization;
    error.code = ValidationErrorCode::SerializationFailed;
    error.stage = ValidationStage::Serialization;
    error.reason = ValidationFailureReason::SerializationFailed;
    error.replay = identity;
    error.diagnostic = "validation JSON serialization failed";
    return error;
}

}  // namespace

Result<std::string> SerializeValidationReport(
        const ValidationReport &report) noexcept {
    try {
        JsonText json;
        json.Append("{");
        AppendValidationResultJson(json, report);
        AppendValidationMetadataJson(json, report.metadata);
        AppendSimulationOutcomeJson(json, report.simulation);
        json.Append(",\"schema\":\"forevervalidator-result-v1\"");
        json.Append(",\"replay\":");
        json.AppendJsonString(report.replay.name);
        json.AppendFormat(",\"valid\":%s", report.valid ? "true" : "false");
        json.Append("}");
        return Result<std::string>::Success(json.Take());
    } catch (const std::bad_alloc &) {
        ValidationError error = SerializationError(report.replay);
        error.category = ValidationErrorCategory::Allocation;
        error.code = ValidationErrorCode::AllocationFailed;
        error.reason = ValidationFailureReason::AllocationFailed;
        return Result<std::string>::Failure(std::move(error));
    } catch (...) {
        return Result<std::string>::Failure(
                SerializationError(report.replay));
    }
}

Result<std::string> SerializeValidationError(
        const ValidationError &error) noexcept {
    try {
        JsonText json;
        json.Append("{\"schema\":\"forevervalidator-error-v1\"");
        json.Append(",\"category\":");
        json.AppendJsonString(ValidationErrorCategoryName(error.category));
        json.Append(",\"code\":");
        json.AppendJsonString(ValidationErrorCodeName(error.code));
        json.Append(",\"stage\":");
        json.AppendJsonString(ValidationStageName(error.stage));
        json.Append(",\"reason\":");
        json.AppendJsonString(ValidationFailureReasonName(error.reason));
        json.Append(",\"replay\":");
        json.AppendJsonString(error.replay.name);
        const int parserCode = error.stage == ValidationStage::ReplayDecoding
                ? ValidationLegacyParserCode(error.reason)
                : -1;
        json.AppendFormat(",\"parser_code\":%d", parserCode);
        json.AppendFormat(",\"simulation_code\":%d",
                          ValidationLegacySimulationCode(error.reason));
        json.AppendFormat(",\"cause_code\":%d",
                          ValidationLegacyCauseCode(error.reason));
        json.Append(",\"related_asset\":");
        json.AppendJsonString(error.relatedAsset);
        json.Append(",\"diagnostic\":");
        json.AppendJsonString(error.diagnostic);
        json.Append("}");
        return Result<std::string>::Success(json.Take());
    } catch (const std::bad_alloc &) {
        ValidationError serializationError = SerializationError(error.replay);
        serializationError.category = ValidationErrorCategory::Allocation;
        serializationError.code = ValidationErrorCode::AllocationFailed;
        serializationError.reason = ValidationFailureReason::AllocationFailed;
        return Result<std::string>::Failure(std::move(serializationError));
    } catch (...) {
        return Result<std::string>::Failure(
                SerializationError(error.replay));
    }
}

}  // namespace forevervalidator
