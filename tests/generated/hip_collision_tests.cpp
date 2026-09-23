// Generated from tests/cuda_collision_tests.cpp by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_collision_certification.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main() {
    using namespace forevervalidator::simulation;
    using hip::collision::HipCollision;
    constexpr std::array<std::uint32_t, 12u> order = {
            7u, 2u, 11u, 0u, 5u, 9u,
            1u, 10u, 4u, 8u, 3u, 6u};
    std::vector<HipCollision> collisions;
    for (std::uint32_t id : order) {
        HipCollision collision;
        collision.contactPoint = {
                static_cast<float>(id),
                static_cast<float>(11u - id),
                static_cast<float>(id & 1u)};
        collision.impulseNormal = {
                0.01f * static_cast<float>(id),
                1.0f,
                -0.02f * static_cast<float>(id)};
        collision.materialA = id;
        collisions.push_back(collision);
    }
    const HipCollisionOrderingExecution sorted =
            ExecuteHipCollisionOrderingForCertification(collisions);
    if (!sorted.success ||
        sorted.collisions.size() != collisions.size()) {
        std::cerr << sorted.diagnostic << '\n';
        return 1;
    }
    for (std::uint32_t index = 0u;
         index < sorted.collisions.size(); ++index) {
        const std::uint32_t expected =
                static_cast<std::uint32_t>(
                        sorted.collisions.size() - 1u - index);
        if (sorted.collisions[index].materialA != expected) {
            std::cerr << "HIP collision response ordering diverged at "
                      << index << " actual="
                      << sorted.collisions[index].materialA << '\n';
            return 1;
        }
    }
    const auto overflow =
            ExecuteHipCollisionOrderingForCertification(
                    std::vector<HipCollision>(
                            hip::collision::CollisionCapacity + 1u));
    if (overflow.success ||
        overflow.diagnostic.find("capacity") ==
                std::string::npos) {
        std::cerr << "HIP collision ordering overflow was not explicit\n";
        return 1;
    }

    HipCandidateState state;
    state.body.current.rotation.Set(
            GmQuat{0.91354543f, 0.0f, 0.40673664f, 0.0f});
    state.body.current.position =
            {1277.009277f, 80.64061f, 1381.608643f};
    std::vector<HipVehicleCollisionShape> shapes(2u);
    shapes[0].localPose.rotation.Set(
            GmQuat{0.98480773f, 0.17364818f, 0.0f, 0.0f});
    shapes[0].localPose.translation = {0.3f, -0.2f, 0.7f};
    shapes[0].parentShapeIndex = UINT32_MAX;
    shapes[0].wheelIndex = UINT32_MAX;
    shapes[1].localPose.rotation.Set(
            GmQuat{0.96592581f, 0.0f, 0.0f, 0.25881904f});
    shapes[1].localPose.translation = {0.0f, 1.0f, -0.4f};
    shapes[1].parentShapeIndex = 0u;
    shapes[1].wheelIndex = UINT32_MAX;

    GmIso4 bodyPose{
            state.body.current.rotation,
            state.body.current.position};
    GmIso4 parentWorld;
    parentWorld.SetMult(shapes[0].localPose, bodyPose);
    GmIso4 expected;
    expected.SetMult(shapes[1].localPose, parentWorld);
    const HipShapeWorldPoseExecution hierarchy =
            ExecuteHipShapeWorldPoseForCertification(
                    shapes, state, 1u);
    if (!hierarchy.success ||
        std::memcmp(
                &hierarchy.worldPose, &expected,
                sizeof(expected)) != 0) {
        std::cerr
                << "HIP nested shape pose changed transform order: "
                << hierarchy.diagnostic << '\n';
        return 1;
    }
    return 0;
}
