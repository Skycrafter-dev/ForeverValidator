#include "simulation/backends/cuda/cuda_search_executor.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using namespace forevervalidator::simulation;

using Op = CudaSearchConditionOpcode;
using Source = CudaSearchConditionValue;

void Constant(CudaSearchConditionConfiguration &condition, double value) {
    condition.instructions.push_back({Op::Constant, Source::Speed, value});
}

void Vector(CudaSearchConditionConfiguration &condition,
            double x, double y, double z) {
    condition.instructions.push_back(
            {Op::ConstantVector, Source::Speed, x, y, z});
}

void SourceValue(CudaSearchConditionConfiguration &condition,
                 Op opcode, Source source) {
    condition.instructions.push_back({opcode, source});
}

bool Evaluate(const CudaSearchConditionConfiguration &condition,
              const CudaCandidateState &state,
              bool *result,
              std::uint32_t stuntPoints = 0u,
              double timeMs = 0.0) {
    std::string diagnostic;
    if (!EvaluateCudaSearchConditionForTesting(
                condition, state, 17u, 123.0, timeMs, stuntPoints,
                result, &diagnostic)) {
        std::cerr << (diagnostic.empty()
                              ? "CUDA condition test execution failed"
                              : diagnostic)
                  << '\n';
        return false;
    }
    return true;
}

CudaSearchConditionConfiguration GeometryCondition() {
    CudaSearchConditionConfiguration condition;
    SourceValue(condition, Op::Vector, Source::Position);
    Constant(condition, 0.0);
    Constant(condition, 0.0);
    Constant(condition, 0.0);
    condition.instructions.push_back({Op::ComposeVector});
    Constant(condition, 4.0);
    Constant(condition, 4.0);
    Constant(condition, 4.0);
    condition.instructions.push_back({Op::ComposeVector});
    condition.instructions.push_back({Op::InsideBox});

    SourceValue(condition, Op::Vector, Source::Velocity);
    condition.instructions.push_back({Op::Normalize});
    Constant(condition, 1.0);
    Constant(condition, 0.0);
    Constant(condition, 0.0);
    condition.instructions.push_back({Op::Direction});
    condition.instructions.push_back({Op::Dot});
    Constant(condition, 0.9);
    condition.instructions.push_back({Op::GreaterOrEqual});
    condition.instructions.push_back({Op::LogicalAnd});

    SourceValue(condition, Op::RotationSource, Source::CarRotation);
    Constant(condition, 0.0);
    Constant(condition, 0.0);
    Constant(condition, 0.0);
    condition.instructions.push_back({Op::Rotation});
    condition.instructions.push_back({Op::RotationDistance});
    Constant(condition, 0.1);
    condition.instructions.push_back({Op::Less});
    condition.instructions.push_back({Op::LogicalAnd});
    return condition;
}

bool CheckGeometryAndRotation() {
    CudaCandidateState state{};
    state.body.current.position = {1.0f, 0.0f, 0.0f};
    state.body.current.linearSpeed = {10.0f, 0.0f, 0.0f};
    state.body.current.rotationQuat = {1.0f, 0.0f, 0.0f, 0.0f};
    const CudaSearchConditionConfiguration condition = GeometryCondition();
    bool result = false;
    if (!Evaluate(condition, state, &result) || !result) {
        std::cerr << "typed CUDA geometry condition rejected matching state\n";
        return false;
    }
    state.body.current.linearSpeed = {-10.0f, 0.0f, 0.0f};
    if (!Evaluate(condition, state, &result) || result) {
        std::cerr << "typed CUDA dot condition accepted opposite direction\n";
        return false;
    }
    state.body.current.linearSpeed = {10.0f, 0.0f, 0.0f};
    constexpr float halfRoot = 0.7071067811865476f;
    state.body.current.rotationQuat = {halfRoot, 0.0f, halfRoot, 0.0f};
    if (!Evaluate(condition, state, &result) || result) {
        std::cerr << "typed CUDA rotation condition ignored rotation distance\n";
        return false;
    }
    return true;
}

CudaSearchConditionConfiguration PrismScalarCondition() {
    CudaSearchConditionConfiguration condition;
    condition.prisms.push_back({1u, 0u, 4u});
    condition.prismVertices = {
            {-2.0, -2.0}, {2.0, -2.0}, {2.0, 2.0}, {-2.0, 2.0}};
    SourceValue(condition, Op::Vector, Source::Position);
    Vector(condition, 0.0, 0.0, 0.0);
    Constant(condition, 5.0);
    condition.instructions.push_back({Op::InsidePrism, Source::Speed, 0.0});

    SourceValue(condition, Op::Scalar, Source::StuntPoints);
    SourceValue(condition, Op::Scalar, Source::SimulationTime);
    Constant(condition, 50.0);
    condition.instructions.push_back({Op::WeightedBlend});
    Constant(condition, 50.0);
    condition.instructions.push_back({Op::PercentRatio});
    condition.instructions.push_back({Op::Multiply});
    Constant(condition, 10.0);
    condition.instructions.push_back({Op::Subtract});
    condition.instructions.push_back({Op::Absolute});
    Constant(condition, 0.0);
    Constant(condition, 10.0);
    condition.instructions.push_back({Op::Clamp});
    Constant(condition, 2.0);
    condition.instructions.push_back({Op::Maximum});
    Constant(condition, 2.0);
    condition.instructions.push_back({Op::GreaterOrEqual});
    condition.instructions.push_back({Op::LogicalAnd});
    return condition;
}

bool CheckPrismAndScalarComposition() {
    CudaCandidateState state{};
    state.body.current.position = {0.0f, 1.0f, 0.0f};
    const CudaSearchConditionConfiguration condition = PrismScalarCondition();
    bool result = false;
    if (!Evaluate(condition, state, &result, 10u, 20.0) || !result) {
        std::cerr << "CUDA prism/scalar condition rejected matching state\n";
        return false;
    }
    state.body.current.position.y = 6.0f;
    if (!Evaluate(condition, state, &result, 10u, 20.0) || result) {
        std::cerr << "CUDA prism condition accepted point beyond depth\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    return CheckGeometryAndRotation() && CheckPrismAndScalarComposition()
            ? 0 : 1;
}
