// Generated from src/simulation/backends/cuda/cuda_dynamics_certification.cu by tools/hipify_backend.py. Do not edit.
#include "hip/hip_runtime.h"
#include "simulation/backends/hip/generated/hip_dynamics_certification.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <new>
#include <vector>

#include "engine/physics/dynamics/hms_dyna.h"
#include "simulation/backends/hip/generated/hip_dynamics.cuh"

namespace forevervalidator::simulation {
namespace {

constexpr std::uint32_t StateCount = 32768u;

struct DynamicsInput {
    HipDynamicBodyState body{};
    HipFixedArray<
            GmVec3,
            HipCollisionReplacementOverflowCapacity>
            overflowReplacements{};
    float dt = 0.0f;
};

struct DynamicsOutput {
    CHmsDyna::CHmsStateDyna preCollision{};
    CHmsDyna::CHmsStateDyna postCollision{};
};

std::uint32_t Next(std::uint32_t &state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

float RandomFloat(std::uint32_t &state, float scale) {
    const std::int32_t value =
            static_cast<std::int32_t>(Next(state) >> 8u) -
            0x007fffff;
    return static_cast<float>(value) *
           (scale / 8388608.0f);
}

GmVec3 RandomVector(std::uint32_t &state, float scale) {
    return {
            RandomFloat(state, scale),
            RandomFloat(state, scale),
            RandomFloat(state, scale),
    };
}

GmMat3 RandomMatrix(std::uint32_t &state, float scale) {
    return {
            RandomVector(state, scale),
            RandomVector(state, scale),
            RandomVector(state, scale),
    };
}

DynamicsInput MakeInput(std::uint32_t index) {
    std::uint32_t random = 0x9e3779b9u ^ (index * 0x85ebca6bu);
    DynamicsInput input;
    input.body.parameters.mass =
            100.0f + static_cast<float>(Next(random) % 2000u);
    input.body.parameters.bodyInertiaLike =
            RandomMatrix(random, 0.02f);
    input.body.parameters.localCenterOfMass =
            RandomVector(random, 2.0f);
    input.body.dynamicType =
            index % 11u == 0u
            ? CHmsDyna::EDynamicType_Frozen
            : (index % 7u == 0u
                       ? CHmsDyna::EDynamicType_LinearOnly
                       : CHmsDyna::EDynamicType_FullAngularDynamics);
    input.body.maxAngularSpeed.present = (index & 3u) == 0u;
    input.body.maxAngularSpeed.value =
            0.25f + static_cast<float>(index % 100u) * 0.1f;

    CHmsDyna::CHmsStateDyna &state = input.body.current;
    state.rotationQuat = {
            RandomFloat(random, 1.0f),
            RandomFloat(random, 1.0f),
            RandomFloat(random, 1.0f),
            RandomFloat(random, 1.0f)};
    if (state.rotationQuat.w == 0.0f &&
        state.rotationQuat.x == 0.0f &&
        state.rotationQuat.y == 0.0f &&
        state.rotationQuat.z == 0.0f) {
        state.rotationQuat.w = 1.0f;
    }
    state.rotationQuat.Normalize();
    state.rotation.Set(state.rotationQuat);
    state.position = RandomVector(random, 1000.0f);
    state.linearSpeed = RandomVector(random, 300.0f);
    state.linearCorrectionSpeed = RandomVector(random, 2.0f);
    state.angularSpeed =
            index % 13u == 0u
            ? GmVec3{}
            : RandomVector(random, 30.0f);
    state.force = RandomVector(random, 10000.0f);
    state.torque = RandomVector(random, 10000.0f);
    state.inverseInertiaWorld = RandomMatrix(random, 0.05f);
    state.tweakedLinearSpeedValid = (index & 1u) != 0u;
    state.tweakedLinearSpeed = RandomVector(random, 100.0f);
    input.dt = 0.001f *
               static_cast<float>((index % 40u) + 1u);
    const std::uint32_t replacementCount =
            index % (HipCollisionReplacementCapacity + 1u);
    input.body.collisionReplacements.count = std::min(
            replacementCount,
            static_cast<std::uint32_t>(
                    HipCollisionReplacementInlineCapacity));
    input.overflowReplacements.count =
            replacementCount -
            input.body.collisionReplacements.count;
    for (std::uint32_t replacement = 0u;
         replacement < replacementCount;
         ++replacement) {
        const GmVec3 value = RandomVector(random, 0.5f);
        if (replacement <
            input.body.collisionReplacements.count) {
            input.body.collisionReplacements.values[replacement] =
                    value;
        } else {
            input.overflowReplacements.values[
                    replacement -
                    input.body.collisionReplacements.count] = value;
        }
    }
    return input;
}

CHmsDyna MakeCpuDyna(const DynamicsInput &input) {
    CHmsDyna::RuntimeClone clone;
    if (input.body.maxAngularSpeed.present) {
        clone.maxAngularSpeed = input.body.maxAngularSpeed.value;
    }
    clone.dynaParams = input.body.parameters;
    clone.currentState = input.body.current;
    clone.writeState = input.body.write;
    clone.tempState = input.body.temporary;
    clone.pendingCollisionReplacements.assign(
            input.body.collisionReplacements.values,
            input.body.collisionReplacements.values +
                    input.body.collisionReplacements.count);
    clone.pendingCollisionReplacements.insert(
            clone.pendingCollisionReplacements.end(),
            input.overflowReplacements.values,
            input.overflowReplacements.values +
                    input.overflowReplacements.count);
    clone.dynamicType = static_cast<CHmsDyna::EDynamicType>(
            input.body.dynamicType);
    clone.isDynamicActive = input.body.dynamicActive;
    CHmsDyna dyna;
    dyna.RestoreRuntimeClone(std::move(clone));
    return dyna;
}

DynamicsOutput CpuExpected(const DynamicsInput &input) {
    DynamicsOutput result;
    CHmsDyna pre = MakeCpuDyna(input);
    pre.DoPreCollisionDynamic(input.dt);
    result.preCollision = pre.CurrentState();
    CHmsDyna post = MakeCpuDyna(input);
    post.DoPostCollisionDynamic();
    result.postCollision = post.CurrentState();
    return result;
}

bool SameFloat(float left, float right) {
    std::uint32_t leftBits = 0u;
    std::uint32_t rightBits = 0u;
    std::memcpy(&leftBits, &left, sizeof(left));
    std::memcpy(&rightBits, &right, sizeof(right));
    return leftBits == rightBits;
}

bool SameVector(const GmVec3 &left, const GmVec3 &right) {
    return SameFloat(left.x, right.x) &&
           SameFloat(left.y, right.y) &&
           SameFloat(left.z, right.z);
}

bool SameMatrix(const GmMat3 &left, const GmMat3 &right) {
    return SameVector(left.basisX, right.basisX) &&
           SameVector(left.basisY, right.basisY) &&
           SameVector(left.basisZ, right.basisZ);
}

bool SameState(const CHmsDyna::CHmsStateDyna &left,
               const CHmsDyna::CHmsStateDyna &right) {
    return SameFloat(left.rotationQuat.w, right.rotationQuat.w) &&
           SameFloat(left.rotationQuat.x, right.rotationQuat.x) &&
           SameFloat(left.rotationQuat.y, right.rotationQuat.y) &&
           SameFloat(left.rotationQuat.z, right.rotationQuat.z) &&
           SameMatrix(left.rotation, right.rotation) &&
           SameVector(left.position, right.position) &&
           SameVector(left.linearSpeed, right.linearSpeed) &&
           SameVector(left.linearCorrectionSpeed,
                      right.linearCorrectionSpeed) &&
           SameVector(left.angularSpeed, right.angularSpeed) &&
           SameVector(left.force, right.force) &&
           SameVector(left.torque, right.torque) &&
           SameMatrix(left.inverseInertiaWorld,
                      right.inverseInertiaWorld) &&
           left.tweakedLinearSpeedValid ==
                   right.tweakedLinearSpeedValid &&
           SameVector(left.tweakedLinearSpeed,
                      right.tweakedLinearSpeed);
}

__global__ void DynamicsKernel(
        const DynamicsInput *inputs,
        DynamicsOutput *outputs,
        std::uint32_t count) {
    const std::uint32_t index =
            blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    HipDynamicBodyState preBody = inputs[index].body;
    auto preOverflow = inputs[index].overflowReplacements;
    hip::dynamics::PreCollision(
            preBody, preOverflow, inputs[index].dt);
    outputs[index].preCollision = preBody.current;
    HipDynamicBodyState postBody = inputs[index].body;
    hip::dynamics::PostCollision(
            postBody, inputs[index].overflowReplacements);
    outputs[index].postCollision = postBody.current;
}

}  // namespace

HipDynamicsCertificationResult
CertifyHipPreCollisionDynamics() noexcept {
    HipDynamicsCertificationResult result;
    try {
        std::vector<DynamicsInput> inputs(StateCount);
        std::vector<DynamicsOutput> expected(StateCount);
        std::vector<DynamicsOutput> actual(StateCount);
        for (std::uint32_t index = 0u; index < StateCount; ++index) {
            inputs[index] = MakeInput(index);
            expected[index] = CpuExpected(inputs[index]);
        }

        DynamicsInput *deviceInputs = nullptr;
        DynamicsOutput *deviceOutputs = nullptr;
        hipError_t error = hipMalloc(
                reinterpret_cast<void **>(&deviceInputs),
                inputs.size() * sizeof(DynamicsInput));
        if (error == hipSuccess) {
            error = hipMalloc(
                    reinterpret_cast<void **>(&deviceOutputs),
                    actual.size() *
                            sizeof(DynamicsOutput));
        }
        if (error == hipSuccess) {
            error = hipMemcpy(
                    deviceInputs, inputs.data(),
                    inputs.size() * sizeof(DynamicsInput),
                    hipMemcpyHostToDevice);
        }
        if (error == hipSuccess) {
            constexpr std::uint32_t Threads = 128u;
            DynamicsKernel<<<(StateCount + Threads - 1u) / Threads,
                             Threads>>>(
                    deviceInputs, deviceOutputs, StateCount);
            error = hipGetLastError();
        }
        if (error == hipSuccess) {
            error = hipDeviceSynchronize();
        }
        if (error == hipSuccess) {
            error = hipMemcpy(
                    actual.data(), deviceOutputs,
                    actual.size() *
                            sizeof(DynamicsOutput),
                    hipMemcpyDeviceToHost);
        }
        hipFree(deviceOutputs);
        hipFree(deviceInputs);
        if (error != hipSuccess) {
            result.diagnostic =
                    std::string("HIP dynamics certification failed: ") +
                    hipGetErrorName(error) + " (" +
                    hipGetErrorString(error) + ")";
            return result;
        }

        result.checkedStates = StateCount;
        result.checkedFields =
                static_cast<std::uint64_t>(StateCount) * 45u;
        for (std::uint32_t index = 0u; index < StateCount; ++index) {
            if (!SameState(expected[index].preCollision,
                           actual[index].preCollision) ||
                !SameState(expected[index].postCollision,
                           actual[index].postCollision)) {
                result.firstMismatchState = index;
                result.diagnostic =
                        "HIP pre-collision dynamics mismatch at state " +
                        std::to_string(index);
                return result;
            }
        }
        result.success = true;
        result.diagnostic =
                "HIP pre/post-collision dynamics certification passed";
        return result;
    } catch (const std::bad_alloc &) {
        result.diagnostic =
                "HIP dynamics certification host allocation failed";
        return result;
    } catch (...) {
        result.diagnostic =
                "unexpected HIP dynamics certification failure";
        return result;
    }
}

}  // namespace forevervalidator::simulation
