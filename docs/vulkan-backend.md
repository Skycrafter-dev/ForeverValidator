# Vulkan compute backend

Vulkan is an optional simulation and resident-search backend alongside CUDA;
it does not replace CUDA or the CPU implementations.

## Build

```sh
cmake -S . -B build/gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFOREVERVALIDATOR_ENABLE_CUDA=ON \
  -DFOREVERVALIDATOR_ENABLE_VULKAN=ON \
  -DFOREVERVALIDATOR_BUILD_TESTS=ON \
  -DFOREVERVALIDATOR_BUILD_BENCHMARKS=ON
cmake --build build/gpu
```

The two backend switches are independent. Disable CUDA to build without the
CUDA toolkit; disable both switches for a CPU-only build. Vulkan requires
Vulkan development headers and its loader. Precompiled SPIR-V is embedded, so
ordinary builds do not require the Slang compiler.

## Selection and compatibility

Select `SimulationBackend::Cuda` or `SimulationBackend::Vulkan` explicitly.
Use the corresponding `QueryCudaBackendDiagnostics` or
`QueryVulkanBackendDiagnostics` call to distinguish a compiled backend from a
usable runtime/device. An explicit GPU selection must not silently become a
CPU simulation.

The experimental resident-search API retains its `PhysicsSandboxCuda*` names
for source compatibility. Its implementation dispatches using the owning
sandbox's backend. CUDA session specialization remains CUDA-only; Vulkan uses
its own shader pipelines and bounded submissions.

## Verification

`forevervalidator-backend-routing-tests` covers backend registration.
`forevervalidator-vulkan-timeline-tests` exercises Vulkan execution. Keep the
existing CUDA arithmetic, dynamics, collision, race and timeline tests enabled
in combined builds.

```sh
build/gpu/cuda_replay_parity PACKS REPLAY both
build/gpu/vulkan_replay_parity PACKS REPLAY both
```

These tools check recorded and modified replay execution against the reference
state, including state restoration and exact finish timing. Hardware support
must be validated on the intended device, not inferred from compilation.

## Shader sources

`tools/generate_vulkan_*` and `tools/compile_vulkan_slang_shader.py` regenerate
the checked-in shader artifacts. The two `vulkan_steady_velocity_*.patch` files
are inputs to the search shader generator, not disposable investigation output.
Keep generated sources and SPIR-V synchronized when changing shared physics.
