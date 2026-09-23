#include "simulation/backends/hip/generated/hip_backend.h"

#include <iostream>
#include <string_view>

int main(int argc, char **argv) {
    const auto device = forevervalidator::QueryHipBackendDiagnostics();
    if (argc == 2 && std::string_view(argv[1]) == "--unavailable") {
        if (device.status != forevervalidator::HipBackendStatus::NoDevice ||
            device.diagnostic.empty()) {
            std::cerr << "masked HIP device did not report NoDevice: "
                      << device.diagnostic << '\n';
            return 1;
        }
        return 0;
    }
    if (!device.IsReady()) {
        std::cerr << device.diagnostic << '\n';
        return 1;
    }
    const auto result =
            forevervalidator::simulation::CertifyHipArithmetic(1000000u);
    if (!result.passed) {
        std::cerr << result.diagnostic
                  << " checked=" << result.checkedValues
                  << " mismatches=" << result.mismatchedValues
                  << " operation=" << result.firstMismatchOperation
                  << " input=0x" << std::hex << result.firstMismatchInput
                  << " expected=0x" << result.expectedBits
                  << " actual=0x" << result.actualBits << '\n';
        return 1;
    }
    std::cout << result.diagnostic
              << " checked=" << result.checkedValues << '\n';
    return 0;
}
