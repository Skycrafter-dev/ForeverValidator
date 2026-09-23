// Generated from src/simulation/backends/cuda/cuda_backend.h by tools/hipify_backend.py. Do not edit.
#ifndef FOREVERVALIDATOR_HIP_BACKEND_H
#define FOREVERVALIDATOR_HIP_BACKEND_H

#include <forevervalidator/validation.h>

namespace forevervalidator::simulation {

struct HipArithmeticCertification {
    bool passed = false;
    std::uint64_t checkedValues = 0u;
    std::uint64_t mismatchedValues = 0u;
    std::uint32_t firstMismatchOperation = 0u;
    std::uint32_t firstMismatchInput = 0u;
    std::uint32_t expectedBits = 0u;
    std::uint32_t actualBits = 0u;
    std::string diagnostic;
};

HipBackendDiagnostics QueryHipRuntimeDiagnostics() noexcept;
HipArithmeticCertification CertifyHipArithmetic(
        std::uint32_t sampleCount) noexcept;

}  // namespace forevervalidator::simulation

#endif
