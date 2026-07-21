#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <forevervalidator/json.h>
#include <forevervalidator/native.h>
#include <forevervalidator/validation.h>

namespace {

namespace fs = std::filesystem;
using forevervalidator::AssetBytes;
using forevervalidator::ByteView;
using forevervalidator::ReplayIdentity;
using forevervalidator::Result;
using forevervalidator::ValidationContext;
using forevervalidator::ValidationError;
using forevervalidator::ValidationOptions;

struct ValidationOutput {
    int exitCode;
    std::string json;
    forevervalidator::ReplayProvenance replayProvenance;
    forevervalidator::InputGhostMatch inputGhostMatch;
};

using ValidationAttempt = Result<ValidationOutput>;
using NativeFileResult = Result<AssetBytes>;

bool HasGbxExtension(const fs::path &path) {
    std::string extension = path.extension().string();
    if (extension.size() != 4u) {
        return false;
    }
    return extension[0] == '.' &&
           (extension[1] == 'g' || extension[1] == 'G') &&
           (extension[2] == 'b' || extension[2] == 'B') &&
           (extension[3] == 'x' || extension[3] == 'X');
}

bool IsDirectory(const fs::path &path) {
    std::error_code error;
    return fs::is_directory(path, error) && !error;
}

bool AppendReplaysFromDirectory(
        std::vector<fs::path> &replays,
        const fs::path &directory) {
    std::error_code error;
    fs::recursive_directory_iterator iterator(directory, error);
    if (error) {
        return false;
    }
    const fs::recursive_directory_iterator end;
    for (; iterator != end; iterator.increment(error)) {
        if (error) {
            return false;
        }
        if (!iterator->is_regular_file(error)) {
            if (error) {
                return false;
            }
            continue;
        }
        if (HasGbxExtension(iterator->path())) {
            replays.push_back(iterator->path());
        }
    }
    return true;
}

bool MakeParentDirectories(const fs::path &path) {
    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return true;
    }
    std::error_code error;
    fs::create_directories(parent, error);
    return !error;
}

bool WriteTextFile(const fs::path &path, const std::string &text) {
    if (!MakeParentDirectories(path)) {
        return false;
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    return output.good();
}

fs::path BatchOutputPath(
        const fs::path &outputDirectory,
        const fs::path &replayPath,
        std::size_t index) {
    char prefix[32];
    std::snprintf(prefix, sizeof(prefix), "%03u-",
                  static_cast<unsigned>(index));
    return outputDirectory /
            (std::string(prefix) + replayPath.filename().string() + ".json");
}

void PrintUsage(const char *program) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  %s --pak-dir DIR [--backend BACKEND] [--batch-size N] REPLAY [--out PATH]\n"
                 "  %s --pak-dir DIR [--backend BACKEND] [--batch-size N] --out-dir DIR REPLAY_OR_DIRECTORY [REPLAY_OR_DIRECTORY ...]\n"
                 "  BACKEND: reference, optimized-cpu, batched; N defaults to 10\n",
                 program,
                 program);
}

void ReportValidationError(const ValidationError &error) {
    if (!error.diagnostic.empty()) {
        std::fprintf(stderr, "%s", error.diagnostic.c_str());
    } else {
        std::fprintf(stderr, "replay validation failed");
    }
    std::fprintf(stderr,
                 " category=%s reason=%s\n",
                 forevervalidator::ValidationErrorCategoryName(error.category),
                 forevervalidator::ValidationFailureReasonName(error.reason));
}

int AttemptExitCode(const ValidationAttempt &attempt) {
    return attempt.HasValue()
            ? attempt.Value().exitCode
            : forevervalidator::ValidationErrorExitCode(attempt.Error());
}

const std::string &AttemptJson(const ValidationAttempt &attempt) {
    static const std::string empty;
    return attempt.HasValue() ? attempt.Value().json : empty;
}

const char *InputGhostMatchName(forevervalidator::InputGhostMatch match) {
    switch (match) {
    case forevervalidator::InputGhostMatch::Unavailable:
        return "Unavailable";
    case forevervalidator::InputGhostMatch::Match:
        return "Match";
    case forevervalidator::InputGhostMatch::Mismatch:
        return "Mismatch";
    }
    return "Unavailable";
}

void ReportReplayClassification(const ValidationAttempt &attempt) {
    if (!attempt.HasValue() ||
        attempt.Value().replayProvenance !=
                forevervalidator::ReplayProvenance::TMInterface) {
        return;
    }
    std::fprintf(stderr,
                 "TMInterface replay: input-vs-ghost=%s\n",
                 InputGhostMatchName(attempt.Value().inputGhostMatch));
}

ValidationAttempt ValidateLoadedReplay(
        ValidationContext &context,
        const NativeFileResult &file,
        const ReplayIdentity &identity,
        const ValidationOptions &options = {}) {
    if (!file) {
        return ValidationAttempt::Failure(file.Error());
    }

    Result<forevervalidator::ValidationReport> validation =
            forevervalidator::ValidateReplay(
                    context,
                    ByteView{file.Value().data(), file.Value().size()},
                    identity,
                    options);
    if (!validation) {
        return ValidationAttempt::Failure(std::move(validation).Error());
    }

    const bool valid = validation.Value().valid;
    Result<std::string> serialization =
            forevervalidator::SerializeValidationReport(validation.Value());
    if (!serialization) {
        return ValidationAttempt::Failure(std::move(serialization).Error());
    }
    return ValidationAttempt::Success(ValidationOutput{
            valid ? 0 : 1,
            std::move(serialization).Value(),
            validation.Value().metadata.replayProvenance,
            validation.Value().inputGhostMatch});
}

ValidationAttempt ValidateReplayPath(
        ValidationContext &context,
        const fs::path &replayPath,
        const ValidationOptions &options = {}) {
    const ReplayIdentity identity{replayPath.string()};
    NativeFileResult file = forevervalidator::ReadNativeReplayFile(
            identity.name, identity);
    return ValidateLoadedReplay(context, file, identity, options);
}

const char *ValidationResultName(int result) {
    if (result == 0) {
        return "valid";
    }
    if (result == 1) {
        return "invalid";
    }
    return "error";
}

std::optional<forevervalidator::SimulationBackend> ParseBackend(
        const char *value) {
    if (std::strcmp(value, "reference") == 0) {
        return forevervalidator::SimulationBackend::Reference;
    }
    if (std::strcmp(value, "optimized-cpu") == 0) {
        return forevervalidator::SimulationBackend::OptimizedCpu;
    }
    if (std::strcmp(value, "batched") == 0) {
        return forevervalidator::SimulationBackend::Batched;
    }
    return std::nullopt;
}

std::optional<std::size_t> ParseBatchSize(const char *value) {
    const char *end = value + std::strlen(value);
    std::size_t result = 0u;
    const std::from_chars_result parsed = std::from_chars(value, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end || result == 0u) {
        return std::nullopt;
    }
    return result;
}

ValidationAttempt SerializeValidationAttempt(
        forevervalidator::ReplayValidationAttempt attempt) {
    if (!attempt) {
        return ValidationAttempt::Failure(std::move(attempt).Error());
    }
    forevervalidator::ValidationReport report = std::move(attempt).Value();
    Result<std::string> serialization =
            forevervalidator::SerializeValidationReport(report);
    if (!serialization) {
        return ValidationAttempt::Failure(std::move(serialization).Error());
    }
    return ValidationAttempt::Success(ValidationOutput{
            report.valid ? 0 : 1,
            std::move(serialization).Value(),
            report.metadata.replayProvenance,
            report.inputGhostMatch});
}

}  // namespace

int main(int argc, char **argv) {
    std::optional<fs::path> outputPath;
    std::optional<fs::path> outputDirectory;
    std::optional<fs::path> packDirectory;
    std::vector<fs::path> replays;
    bool repeatSameProcess = false;
    std::size_t batchSize = 10u;
    std::optional<forevervalidator::SimulationBackend> selectedBackend;

    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--out") == 0 && index + 1 < argc) {
            outputPath.emplace(argv[++index]);
        } else if (std::strcmp(argv[index], "--out-dir") == 0 &&
                   index + 1 < argc) {
            outputDirectory.emplace(argv[++index]);
        } else if (std::strcmp(argv[index], "--pak-dir") == 0 &&
                   index + 1 < argc) {
            packDirectory.emplace(argv[++index]);
        } else if (std::strcmp(argv[index], "--repeat-same-process") == 0) {
            repeatSameProcess = true;
        } else if (std::strcmp(argv[index], "--backend") == 0 &&
                   index + 1 < argc) {
            selectedBackend = ParseBackend(argv[++index]);
            if (!selectedBackend.has_value()) {
                PrintUsage(argv[0]);
                return 64;
            }
        } else if (std::strcmp(argv[index], "--batch-size") == 0 &&
                   index + 1 < argc) {
            const std::optional<std::size_t> parsed =
                    ParseBatchSize(argv[++index]);
            if (!parsed.has_value()) {
                PrintUsage(argv[0]);
                return 64;
            }
            batchSize = *parsed;
        } else if (argv[index][0] == '-') {
            PrintUsage(argv[0]);
            return 64;
        } else if (IsDirectory(argv[index])) {
            if (!AppendReplaysFromDirectory(replays, argv[index])) {
                std::fprintf(stderr, "could not scan replay directory %s\n",
                             argv[index]);
                return 65;
            }
        } else {
            replays.emplace_back(argv[index]);
        }
    }

    if (!packDirectory.has_value() || packDirectory->empty() ||
        replays.empty() ||
        (outputPath.has_value() && outputDirectory.has_value())) {
        PrintUsage(argv[0]);
        return 64;
    }
    std::sort(replays.begin(), replays.end());

    const bool singleRun = !outputDirectory.has_value() && replays.size() == 1u;
    if (!singleRun && !outputDirectory.has_value()) {
        PrintUsage(argv[0]);
        return 64;
    }

    const std::string packDirectoryText = packDirectory->string();
    Result<forevervalidator::AssetSource> source =
            forevervalidator::OpenInstalledPackDirectory(packDirectoryText);
    if (!source) {
        ReportValidationError(source.Error());
        if (outputDirectory.has_value()) {
            std::fprintf(stderr,
                         "could not prepare installed-pack validation context\n");
        }
        return forevervalidator::ValidationErrorExitCode(source.Error());
    }
    Result<ValidationContext> contextResult =
            forevervalidator::CreateValidationContext(
                    std::move(source).Value());
    if (!contextResult) {
        ReportValidationError(contextResult.Error());
        return forevervalidator::ValidationErrorExitCode(
                contextResult.Error());
    }
    ValidationContext context = std::move(contextResult).Value();
    ValidationOptions validationOptions;
    validationOptions.backend = selectedBackend.value_or(
            singleRun ? forevervalidator::SimulationBackend::Reference
                      : forevervalidator::SimulationBackend::Batched);

    if (singleRun) {
        ValidationAttempt attempt = ValidateReplayPath(
                context, replays.front(), validationOptions);
        ReportReplayClassification(attempt);
        if (!attempt) {
            ReportValidationError(attempt.Error());
        }
        const int exitCode = AttemptExitCode(attempt);
        if (exitCode <= 1) {
            if (outputPath.has_value() &&
                !WriteTextFile(*outputPath, AttemptJson(attempt))) {
                const std::string outputText = outputPath->string();
                std::fprintf(stderr, "could not write output %s\n",
                             outputText.c_str());
                return 67;
            }
            const std::string &json = AttemptJson(attempt);
            std::fwrite(json.data(), 1u, json.size(), stdout);
            std::fputc('\n', stdout);
        }
        return exitCode;
    }

    unsigned valid = 0u;
    unsigned invalid = 0u;
    unsigned errors = 0u;
    for (std::size_t batchBegin = 0u; batchBegin < replays.size();) {
        const std::size_t replayCount = std::min(
                batchSize, replays.size() - batchBegin);
        std::vector<AssetBytes> loadedReplays;
        std::vector<forevervalidator::ReplayValidationRequest> requests;
        loadedReplays.reserve(replayCount);
        requests.reserve(repeatSameProcess ? replayCount * 2u : replayCount);
        for (std::size_t offset = 0u; offset < replayCount; ++offset) {
            const fs::path &replay = replays[batchBegin + offset];
            const ReplayIdentity identity{replay.string()};
            NativeFileResult file = forevervalidator::ReadNativeReplayFile(
                    identity.name, identity);
            if (!file) {
                ReportValidationError(file.Error());
                return forevervalidator::ValidationErrorExitCode(file.Error());
            }
            loadedReplays.push_back(std::move(file).Value());
            requests.push_back({
                    ByteView{loadedReplays.back().data(),
                             loadedReplays.back().size()},
                    identity});
            if (repeatSameProcess) {
                requests.push_back(requests.back());
            }
        }

        Result<forevervalidator::ReplayBatchReport> batch =
                forevervalidator::ValidateReplayBatch(
                        context, requests, validationOptions);
        if (!batch) {
            ReportValidationError(batch.Error());
            return forevervalidator::ValidationErrorExitCode(batch.Error());
        }
        std::vector<ValidationAttempt> attempts;
        attempts.reserve(batch.Value().attempts.size());
        for (auto &attempt : batch.Value().attempts) {
            attempts.push_back(SerializeValidationAttempt(std::move(attempt)));
        }

        for (std::size_t offset = 0u; offset < replayCount; ++offset) {
            const std::size_t index = batchBegin + offset;
            const fs::path output = BatchOutputPath(
                    *outputDirectory, replays[index], index);
            const std::string replayText = replays[index].string();
            std::fprintf(stderr, "validate: %s\n", replayText.c_str());

            const std::size_t attemptIndex = repeatSameProcess
                    ? offset * 2u : offset;
            ValidationAttempt &attempt = attempts[attemptIndex];
            ReportReplayClassification(attempt);
            if (!attempt) {
                ReportValidationError(attempt.Error());
            }
            int result = AttemptExitCode(attempt);
            if (repeatSameProcess) {
                ValidationAttempt &second = attempts[attemptIndex + 1u];
                if (result != AttemptExitCode(second) ||
                    AttemptJson(attempt) != AttemptJson(second)) {
                    std::fprintf(stderr,
                                 "same-process replay result mismatch for %s\n",
                                 replayText.c_str());
                    result = 71;
                }
            }
            if (result <= 1 && !WriteTextFile(output, AttemptJson(attempt))) {
                result = 67;
            }

            std::fprintf(stderr, "result: %s -> %s",
                         replayText.c_str(), ValidationResultName(result));
            if (result > 1) {
                std::fprintf(stderr, " (%d)", result);
            }
            std::fputc('\n', stderr);
            if (result == 0) {
                ++valid;
            } else if (result == 1) {
                ++invalid;
            } else {
                ++errors;
            }
        }
        batchBegin += replayCount;
    }

    std::printf("{\"schema\":\"forevervalidator-batch-v1\","
                "\"total\":%u,\"valid\":%u,\"invalid\":%u,\"error\":%u}\n",
                valid + invalid + errors,
                valid,
                invalid,
                errors);
    if (errors != 0u) {
        return 2;
    }
    return invalid == 0u ? 0 : 1;
}
