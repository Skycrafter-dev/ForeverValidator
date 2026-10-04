#ifndef FOREVERVALIDATOR_VALIDATION_PLANNING_SEARCH_EVENT_CAPACITY_H
#define FOREVERVALIDATOR_VALIDATION_PLANNING_SEARCH_EVENT_CAPACITY_H

#include <forevervalidator/experimental/physics_sandbox.h>
#include <forevervalidator/experimental/search_limits.h>

#include <limits>
#include <type_traits>

namespace forevervalidator::experimental::search_limits {

inline bool AddEventCapacity(std::size_t amount, std::size_t *capacity) {
    if (capacity == nullptr || *capacity > kMaximumSearchInputEvents ||
        amount > kMaximumSearchInputEvents - *capacity) {
        return false;
    }
    *capacity += amount;
    return true;
}

inline bool MaximumEventCapacity(
        std::size_t baselineCount,
        const std::vector<PhysicsSandboxCudaModifier> &modifiers,
        std::uint32_t tickDurationMs,
        std::size_t *capacity) {
    if (capacity == nullptr || tickDurationMs == 0u ||
        baselineCount > kMaximumSearchInputEvents) return false;
    *capacity = baselineCount;
    for (const PhysicsSandboxCudaModifier &modifier : modifiers) {
        bool valid = std::visit(
                [&](const auto &value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<
                                          T,
                                          PhysicsSandboxCudaSmoothSteeringModifier>) {
                        if (value.radiusMs < 0 ||
                            value.radiusMs %
                                            static_cast<std::int64_t>(
                                                    tickDurationMs) !=
                                    0) {
                            return false;
                        }
                        if (static_cast<std::uint64_t>(value.radiusMs / tickDurationMs) >
                            (kMaximumSearchInputEvents - 2u) / 2u) return false;
                        const std::uint64_t perDeformation =
                                static_cast<std::uint64_t>(
                                        value.radiusMs /
                                        tickDurationMs) *
                                        2u +
                                2u;
                        if (value.deformationCount != 0u &&
                            perDeformation >
                                    std::numeric_limits<std::size_t>::max() /
                                            value.deformationCount) {
                            return false;
                        }
                        return AddEventCapacity(
                                static_cast<std::size_t>(
                                        perDeformation *
                                        value.deformationCount),
                                capacity);
                    } else if constexpr (std::is_same_v<
                                                 T,
                                                 PhysicsSandboxCudaInputInsertionModifier>) {
                        const std::uint64_t operations =
                                (value.steering.enabled
                                         ? static_cast<std::uint64_t>(value.steering.maximumCount)
                                         : 0u) +
                                (value.accelerate.enabled
                                         ? value.accelerate.maximumCount
                                         : 0u) +
                                (value.brake.enabled
                                         ? value.brake.maximumCount
                                         : 0u);
                        if (operations >
                            std::numeric_limits<std::size_t>::max() / 2u) {
                            return false;
                        }
                        return AddEventCapacity(
                                static_cast<std::size_t>(operations * 2u),
                                capacity);
                    } else {
                        return true;
                    }
                },
                modifier);
        if (!valid) {
            return false;
        }
    }
    return true;
}

}  // namespace forevervalidator::experimental::search_limits

#endif
