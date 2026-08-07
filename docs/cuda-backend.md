# CUDA backend

ForeverValidator has an optional NVIDIA CUDA backend for exact
Stadium/StadiumCar simulation. It is selected with `SimulationBackend::Cuda`
or `--backend cuda`; it never resolves to a CPU backend.

## Build and hardware

CUDA is off by default, so the normal CPU-only build has no CUDA toolkit or
runtime dependency:

```sh
cmake -S . -B build/cpu -G Ninja \
  -DFOREVERVALIDATOR_BUILD_TESTS=ON
cmake --build build/cpu
```

Enable it explicitly with a CUDA C++17 compiler:

```sh
cmake -S . -B build/cuda -G Ninja \
  -DFOREVERVALIDATOR_ENABLE_CUDA=ON \
  -DFOREVERVALIDATOR_BUILD_TESTS=ON \
  -DFOREVERVALIDATOR_BUILD_BENCHMARKS=ON
cmake --build build/cuda
```

The generic runtime requires an NVIDIA device with compute capability 6.1 or
newer. The optional per-map session-specialized search kernel requires compute
capability 7.5; supported older devices automatically use the precompiled
generic search kernel.
The default CMake configuration emits real `sm_61` code for Pascal plus
forward-compatible `compute_61` PTX for newer GPUs. Release builders can set
`CMAKE_CUDA_ARCHITECTURES` explicitly to select a native or multi-architecture
matrix.
Device discovery reports the selected device, compute capability, driver and
runtime versions, and global memory. A missing toolkit, runtime, device, or
supported architecture produces a typed CUDA error. A CPU-only executable
rejects `Cuda` with `cuda_not_compiled`.

## Scope

CUDA supports Stadium maps using the StadiumCar in Race, Platform, Puzzle,
and Stunts modes, including multilap races. Other map/vehicle routes are
rejected with `cuda_unsupported_simulation_scope` before simulation.

The device transition includes control timing, powertrain, wheels,
suspension, forces, rigid-body integration, static collision traversal and
response, materials, fake contacts, water, boosters, checkpoints, finish,
laps, respawns, and the complete stunt state machine and event stream.
Candidate state includes the candidate identifier, validation seed,
candidate-owned RNG state, control cursor, race state, every vehicle frame,
and every future-affecting dynamic field.

## Architecture

Map actors, surfaces, materials, mesh/octree data, broad-phase cells,
checkpoint metadata, tuning curves, transmission tables, collision shapes,
force fields, and water data are flattened deterministically into
pointer-free packed buffers. Scene and configuration hashes control retained
device uploads; they are uploaded once and reused by subsequent timeline
launches.

Mutable state is a standard-layout, trivially-copyable, pointer-free
`CudaCandidateState`. All variable data is bounded:

| Data | Capacity |
| --- | ---: |
| Wheels | 4 |
| Collision replacements | 64 |
| Checkpoint slots | 1,024 |
| Stunt events | 2,048 |
| Detected contacts per candidate | 512 |
| Candidate batch | 4,096 |
| Ticks per submitted batch | 100,000,000 |
| Observations per submitted batch | 10,000,000 |

Packing and execution reject schema mismatches and every capacity overflow.
Each CUDA block owns one independent candidate timeline. The kernel loops
over the full control stream, so there is no per-tick host synchronization.
Final state, observations, outcomes, events, and scores are copied back once;
winner selection uses the exact returned candidate data.

CUDA is compiled with fused multiply-add disabled, precise division and
square root, gradual underflow, and no fast math. The corresponding CPU
reference uses native binary32 operations with contraction disabled.

## Correctness

The permanent test suite covers state conversion and overflow handling,
packed configuration, 12 million arithmetic checks, pre/post-collision
dynamics, collision response ordering, timeline batching, candidate
ownership, cancellation, malformed inputs, stunt histories/events/penalties,
and exact synthetic state transitions.

`cuda_replay_parity` additionally performs a complete CPU transition and an
actual CUDA transition at every replay tick, compares all 58,208 bytes of
candidate state, and verifies capture/restoration around every check. On
2026-07-26 it passed recorded and deterministically mutated timelines for:

| Replay | Mode | Ticks per variant | Bytes checked per variant |
| --- | --- | ---: | ---: |
| 7186162 | Puzzle | 1,255 | 73,051,040 |
| 7186170 | Platform | 1,546 | 89,989,568 |
| 7216920 | Race | 3,530 | 205,474,240 |
| 7191978 | Stunts | 8,296 | 482,893,568 |

Each run also certified a 32-candidate, 1,000-tick mutated batch against
independent CPU transitions, including final states and the selected winner.
CUDA and OptimizedCpu produced byte-identical validation JSON for those four
replays. The production Puzzle timeline and all CUDA unit kernels completed
Compute Sanitizer memcheck with zero errors and zero leaked device bytes.

Run the corpus proof with:

```sh
build/cuda/cuda_replay_parity PACKS REPLAY both
```

## Performance

Winner-selection complexity, differential coverage, and current search
microbenchmarks are documented in
[CUDA winner reduction](cuda-winner-reduction.md).

`cuda_backend_benchmark` reports immutable packing/upload, allocation,
timeline transfer, kernel, synchronization, device memory, throughput, and a
parallel CPU comparison. On an RTX 5060 (compute capability 12.0), replay
7186162, 256 CUDA candidates and 32 parallel CPU candidates, each for 1,000
ticks:

| Metric | Result |
| --- | ---: |
| CUDA initialization | 885.444 ms |
| Scene pack / upload | 17.512 / 146.254 ms |
| Configuration pack / upload | 0.029 / 0.165 ms |
| Timeline transfer | 3.190 ms |
| Kernel / synchronization | 1,623.797 / 1,623.805 ms |
| CUDA batch wall time | 1,657.993 ms |
| Immutable / peak timeline memory | 11,875,312 / 47,559,684 bytes |
| CUDA throughput | 154,403.508 ticks/s |
| Parallel CPU throughput | 83,792.409 ticks/s |
| CUDA throughput ratio | 1.843x |

The benchmark uses independently mutated steering per candidate and reports
the exact winner. Initialization depends strongly on map size, storage, GPU,
driver, compiler architecture, and CPU core count.

## Limitations

The CUDA backend is NVIDIA-only, single-device, and intentionally limited to
Stadium/StadiumCar. Bounds above are hard failures rather than dynamic device
allocation. The execution API is synchronous; its cancellation flag is
checked every tick but is currently supplied before launch. Candidate batch
execution and differential hooks are internal developer APIs rather than
stable public library interfaces.
