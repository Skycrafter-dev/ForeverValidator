// Generated from tests/cuda_dynamics_tests.cpp by tools/hipify_backend.py. Do not edit.
#include "simulation/backends/hip/generated/hip_dynamics_certification.h"

#include <iostream>

int main() {
    const auto result = forevervalidator::simulation::
            CertifyHipPreCollisionDynamics();
    if (!result.success) {
        std::cerr << result.diagnostic << '\n';
        return 1;
    }
    std::cout << result.diagnostic
              << " checked_states=" << result.checkedStates
              << " checked_fields=" << result.checkedFields << '\n';
    return 0;
}
