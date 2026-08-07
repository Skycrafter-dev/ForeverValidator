#include "simulation/backends/cuda/cuda_search_executor.h"

#include <string>

int main() {
    forevervalidator::simulation::CudaSearchExecutorConfiguration
            configuration;
    std::string diagnostic;
    auto executor =
            forevervalidator::simulation::CudaSearchExecutor::Create(
                    configuration, &diagnostic);
    return executor ? 1 : 0;
}
