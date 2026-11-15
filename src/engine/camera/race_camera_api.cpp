#include <forevervalidator/camera.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>

#include "engine/camera/race_camera_internal.h"

namespace forevervalidator::camera {
namespace {

constexpr std::array<RaceCameraProfileInfo, 4u> ProfileInfos{{
        {RaceCameraProfile::Race, 0x24085000u,
         "CGameControlCameraTrackManiaRace", std::nullopt},
        {RaceCameraProfile::Race2, 0x24086000u,
         "CGameControlCameraTrackManiaRace2", 1u},
        {RaceCameraProfile::Race3, 0x24087000u,
         "CGameControlCameraTrackManiaRace3", 2u},
        {RaceCameraProfile::Internal, 0x0306b000u,
         "CGameControlCamera", 3u},
}};

} // namespace

const RaceCameraProfileInfo &ProfileInfo(RaceCameraProfile profile) noexcept {
    const std::size_t index = static_cast<std::size_t>(profile);
    return ProfileInfos[index < ProfileInfos.size() ? index : 0u];
}

class RaceCameraEnvironment::Impl {
  public:
    explicit Impl(detail::EnvironmentConfig value)
        : config(std::move(value)) {}

    detail::EnvironmentConfig config;
};

RaceCameraEnvironment::RaceCameraEnvironment() = default;
RaceCameraEnvironment::~RaceCameraEnvironment() = default;
RaceCameraEnvironment::RaceCameraEnvironment(
        const RaceCameraEnvironment &) noexcept = default;
RaceCameraEnvironment &RaceCameraEnvironment::operator=(
        const RaceCameraEnvironment &) noexcept = default;
RaceCameraEnvironment::RaceCameraEnvironment(
        RaceCameraEnvironment &&) noexcept = default;
RaceCameraEnvironment &RaceCameraEnvironment::operator=(
        RaceCameraEnvironment &&) noexcept = default;

RaceCameraEnvironment::RaceCameraEnvironment(
        std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}

bool RaceCameraEnvironment::HasProfile(
        RaceCameraProfile profile) const noexcept {
    if (!impl_) {
        return false;
    }
    switch (profile) {
    case RaceCameraProfile::Race:
        return !impl_->config.raceResources.empty();
    case RaceCameraProfile::Race2:
        return !impl_->config.race2Resources.empty();
    case RaceCameraProfile::Race3:
        return !impl_->config.race3Resources.empty();
    case RaceCameraProfile::Internal:
        return !impl_->config.internalResources.empty();
    }
    return false;
}

std::string_view RaceCameraEnvironment::SourceName() const noexcept {
    return impl_ ? std::string_view(impl_->config.sourceName)
                 : std::string_view{};
}

std::size_t RaceCameraEnvironment::ResourceCount(
        RaceCameraProfile profile) const noexcept {
    if (!impl_) {
        return 0u;
    }
    switch (profile) {
    case RaceCameraProfile::Race:
        return impl_->config.raceResources.size();
    case RaceCameraProfile::Race2:
        return impl_->config.race2Resources.size();
    case RaceCameraProfile::Race3:
        return impl_->config.race3Resources.size();
    case RaceCameraProfile::Internal:
        return impl_->config.internalResources.size();
    }
    return 0u;
}

std::string_view RaceCameraEnvironment::ResourceName(
        RaceCameraProfile profile,
        std::size_t resourceIndex) const noexcept {
    if (!impl_) {
        return {};
    }
    switch (profile) {
    case RaceCameraProfile::Race:
        return resourceIndex < impl_->config.raceResources.size()
                       ? std::string_view(
                                 impl_->config.raceResources[resourceIndex]
                                         .sourceName)
                       : std::string_view{};
    case RaceCameraProfile::Race2:
        return resourceIndex < impl_->config.race2Resources.size()
                       ? std::string_view(
                                 impl_->config.race2Resources[resourceIndex]
                                         .sourceName)
                       : std::string_view{};
    case RaceCameraProfile::Race3:
        return resourceIndex < impl_->config.race3Resources.size()
                       ? std::string_view(
                                 impl_->config.race3Resources[resourceIndex]
                                         .sourceName)
                       : std::string_view{};
    case RaceCameraProfile::Internal:
        return resourceIndex < impl_->config.internalResources.size()
                       ? std::string_view(
                                 impl_->config.internalResources[resourceIndex]
                                         .sourceName)
                       : std::string_view{};
    }
    return {};
}

std::optional<RaceCameraResourceSelection>
RaceCameraEnvironment::ResourceForCollectorIndex(
        std::uint32_t collectorIndex) const noexcept {
    if (!impl_) {
        return std::nullopt;
    }
    for (std::size_t index = 0u;
         index < impl_->config.raceResources.size(); ++index) {
        if (impl_->config.raceResources[index].collectorIndex ==
                collectorIndex) {
            return RaceCameraResourceSelection{
                    RaceCameraProfile::Race, index, collectorIndex};
        }
    }
    for (std::size_t index = 0u;
         index < impl_->config.race2Resources.size(); ++index) {
        if (impl_->config.race2Resources[index].collectorIndex ==
                collectorIndex) {
            return RaceCameraResourceSelection{
                    RaceCameraProfile::Race2, index, collectorIndex};
        }
    }
    for (std::size_t index = 0u;
         index < impl_->config.race3Resources.size(); ++index) {
        if (impl_->config.race3Resources[index].collectorIndex ==
                collectorIndex) {
            return RaceCameraResourceSelection{
                    RaceCameraProfile::Race3, index, collectorIndex};
        }
    }
    for (std::size_t index = 0u;
         index < impl_->config.internalResources.size(); ++index) {
        if (impl_->config.internalResources[index].collectorIndex ==
                collectorIndex) {
            return RaceCameraResourceSelection{
                    RaceCameraProfile::Internal, index, collectorIndex};
        }
    }
    return std::nullopt;
}

std::optional<RaceCameraResourceSelection>
RaceCameraEnvironment::DefaultResource() const noexcept {
    return impl_ ? ResourceForCollectorIndex(
                           impl_->config.defaultCollectorIndex)
                 : std::nullopt;
}

RaceCameraEnvironment detail::RaceCameraEnvironmentFactory::Create(
        EnvironmentConfig config) {
    if (!config.race && !config.raceResources.empty()) {
        config.race = config.raceResources.back().config;
    }
    if (!config.race2 && !config.race2Resources.empty()) {
        config.race2 = config.race2Resources.back().config;
    }
    if (!config.race3 && !config.race3Resources.empty()) {
        config.race3 = config.race3Resources.back().config;
    }
    if (!config.internal && !config.internalResources.empty()) {
        config.internal = config.internalResources.back().config;
    }
    if (config.race && config.raceResources.empty()) {
        config.raceResources.push_back(
                {config.sourceName, 0xffffffffu, *config.race});
    }
    if (config.race2 && config.race2Resources.empty()) {
        config.race2Resources.push_back(
                {config.sourceName, 0xffffffffu, *config.race2});
    }
    if (config.race3 && config.race3Resources.empty()) {
        config.race3Resources.push_back(
                {config.sourceName, 0xffffffffu, *config.race3});
    }
    if (config.internal && config.internalResources.empty()) {
        config.internalResources.push_back(
                {config.sourceName, 0xffffffffu, *config.internal});
    }
    return RaceCameraEnvironment(
            std::make_shared<const RaceCameraEnvironment::Impl>(
                    std::move(config)));
}

const detail::EnvironmentConfig &
detail::RaceCameraEnvironmentFactory::Config(
        const RaceCameraEnvironment &environment) {
    if (!environment.impl_) {
        throw std::invalid_argument("race camera environment is empty");
    }
    return environment.impl_->config;
}

class RaceCameraSession::Impl {
  public:
    Impl(const RaceCameraEnvironment &environment,
         RaceCameraProfile profile,
         std::optional<std::size_t> resourceIndex)
        : controller(detail::CreateController(
                  detail::RaceCameraEnvironmentFactory::Config(environment),
                  profile, resourceIndex)) {
        if (!controller) {
            throw std::invalid_argument("unsupported race camera profile");
        }
    }
    explicit Impl(std::unique_ptr<detail::Controller> copied)
        : controller(std::move(copied)) {}

    std::unique_ptr<detail::Controller> controller;
};

RaceCameraSession::RaceCameraSession(
        const RaceCameraEnvironment &environment,
        RaceCameraProfile profile)
    : impl_(std::make_unique<Impl>(environment, profile, std::nullopt)) {}

RaceCameraSession::RaceCameraSession(
        const RaceCameraEnvironment &environment,
        RaceCameraProfile profile,
        std::size_t resourceIndex)
    : impl_(std::make_unique<Impl>(
              environment, profile, resourceIndex)) {}

RaceCameraSession::RaceCameraSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

RaceCameraSession::~RaceCameraSession() = default;

RaceCameraSession RaceCameraSession::Clone() const {
    return RaceCameraSession(
            std::make_unique<Impl>(impl_->controller->Clone()));
}

RaceCameraSession::State RaceCameraSession::CaptureState() const {
    State state;
    state.profile = Profile();
    detail::StateArchive archive(state, false);
    impl_->controller->VisitState(archive);
    state.size = archive.Size();
    return state;
}

void RaceCameraSession::RestoreState(const State &state) {
    if (state.profile != Profile() || state.size > state.bytes.size())
        throw std::invalid_argument("incompatible camera controller state");
    auto copy = state;
    detail::StateArchive archive(copy, true);
    // Validate on a clone so malformed input cannot partially restore a session.
    auto restored = impl_->controller->Clone();
    restored->VisitState(archive);
    if (archive.Size() != state.size)
        throw std::invalid_argument("invalid camera controller state size");
    impl_->controller = std::move(restored);
}

RaceCameraSession::RaceCameraSession(RaceCameraSession &&) noexcept = default;

RaceCameraSession &
RaceCameraSession::operator=(RaceCameraSession &&) noexcept = default;

RaceCameraProfile RaceCameraSession::Profile() const noexcept {
    return impl_->controller->Profile();
}

void RaceCameraSession::Reset(const RaceCameraVehicleState &vehicle) {
    impl_->controller->Reset(detail::ToInternal(vehicle));
}

RaceCameraOutput RaceCameraSession::Evaluate(const RaceCameraQuery &query) {
    return detail::ToPublic(detail::RaceCameraSessionAccess::Evaluate(*this, query));
}

detail::CameraFrame detail::RaceCameraSessionAccess::Evaluate(
        RaceCameraSession &session, const RaceCameraQuery &query) {
    return session.impl_->controller->Evaluate(
            detail::ToInternal(query.vehicle), query.segmentCollision);
}

} // namespace forevervalidator::camera
