#ifndef FOREVERVALIDATOR_CUDA_BACKEND_H
#define FOREVERVALIDATOR_CUDA_BACKEND_H

#include <forevervalidator/validation.h>

namespace forevervalidator::simulation {

// The generic, ahead-of-time CUDA search kernels support Pascal GP104 and
// newer devices. The optional per-map specialization is compiled from LTO IR
// whose baseline is Volta/Turing-era code, so keep that optimization on the
// narrower capability range and transparently use the generic kernels below
// it.
constexpr bool IsCudaComputeCapabilitySupported(int major,
                                                int minor) noexcept {
    return major > 6 || (major == 6 && minor >= 1);
}

constexpr bool IsCudaSessionSpecializationSupported(int major,
                                                     int minor) noexcept {
    return major > 7 || (major == 7 && minor >= 5);
}

struct CudaArithmeticCertification {
    bool passed = false;
    std::uint64_t checkedValues = 0u;
    std::uint64_t mismatchedValues = 0u;
    std::uint32_t firstMismatchOperation = 0u;
    std::uint32_t firstMismatchInput = 0u;
    std::uint32_t expectedBits = 0u;
    std::uint32_t actualBits = 0u;
    std::string diagnostic;
};

CudaBackendDiagnostics QueryCudaRuntimeDiagnostics() noexcept;
CudaArithmeticCertification CertifyCudaArithmetic(
        std::uint32_t sampleCount) noexcept;

}  // namespace forevervalidator::simulation

#endif
