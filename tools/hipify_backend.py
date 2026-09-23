#!/usr/bin/env python3
"""Regenerate the HIP backend from the canonical CUDA sources."""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
CUDA = ROOT / "src/simulation/backends/cuda"
HIP = ROOT / "src/simulation/backends/hip/generated"
EXCLUDED = {
    "cuda_driver_shim.cpp",
    "cuda_session_specialization.cpp",
    "cuda_vehicle_cpu_reference.cpp",
    "cuda_vehicle_cpu_reference.h",
}
SUFFIXES = {".cu", ".cuh", ".cpp", ".h"}
HIPIFY_VERSION = "7.2.0"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one HIP overlay anchor, found {source.count(old)}: {old[:70]!r}")
    return source.replace(old, new)


def translate(source: pathlib.Path, hipify: str) -> bytes:
    result = subprocess.run(
        [hipify, "-quiet-warnings", str(source)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = result.stdout.decode("utf-8")
    output = output.replace("simulation/backends/cuda/", "simulation/backends/hip/generated/")
    output = re.sub(r"Cuda|CUDA|cuda", lambda match: {
        "Cuda": "Hip", "CUDA": "HIP", "cuda": "hip"
    }[match.group()], output)
    output = output.replace(
        "#include <cub/device/device_reduce.cuh>",
        '#include "simulation/backends/hip/hip_reduce_adapter.h"',
    )
    output = output.replace("__CUDACC__", "__HIPCC__")
    if source.name == "cuda_backend.cu":
        output = replace_once(
            output,
            "    error = hipGetDeviceCount(&result.deviceCount);\n"
            "    if (error != hipSuccess) {",
            "    error = hipGetDeviceCount(&result.deviceCount);\n"
            "    if (error == hipErrorNoDevice) {\n"
            "        result.status = HipBackendStatus::NoDevice;\n"
            "        result.diagnostic = \"HIP runtime reported no HIP-capable devices\";\n"
            "        return result;\n"
            "    }\n"
            "    if (error != hipSuccess) {",
        )
        capability_check = (
            "    if (!HipBackendDiagnostics::SupportsComputeCapability(\n"
            "            properties.major, properties.minor)) {"
        )
        output = replace_once(output,
            capability_check,
            "#if defined(__HIP_PLATFORM_NVIDIA__)\n" + capability_check,
        )
        capability_end = (
            '                "HIP backend requires compute capability 5.0 or newer";\n'
            "        return result;\n"
            "    }"
        )
        output = replace_once(output, capability_end, capability_end + "\n#endif")
        output = replace_once(output,
            '    std::snprintf(\n            buffer, sizeof(buffer),\n'
            '            "HIP device %d ready: %s, compute capability %d.%d, "',
            '#if defined(__HIP_PLATFORM_NVIDIA__)\n'
            '    std::snprintf(\n            buffer, sizeof(buffer),\n'
            '            "HIP device %d ready: %s, compute capability %d.%d, "',
        )
        output = replace_once(output,
            '            static_cast<unsigned long long>(result.totalGlobalMemoryBytes));\n'
            '    result.diagnostic = buffer;',
            '            static_cast<unsigned long long>(result.totalGlobalMemoryBytes));\n'
            '#else\n'
            '    std::snprintf(buffer, sizeof(buffer),\n'
            '                  "HIP device %d ready: %s, driver %d, runtime %d, "\n'
            '                  "%llu bytes global memory",\n'
            '                  result.selectedDevice, properties.name,\n'
            '                  result.driverVersion, result.runtimeVersion,\n'
            '                  static_cast<unsigned long long>(result.totalGlobalMemoryBytes));\n'
            '#endif\n'
            '    result.diagnostic = buffer;',
        )
    if source.name == "cuda_search_executor.cu":
        output = replace_once(
            output,
            '#include "simulation/backends/hip/generated/hip_search_executor.h"',
            '#include "simulation/backends/hip/generated/hip_search_executor.h"\n'
            '#include "simulation/backends/hip/hip_search_arch_adapter.h"',
        )
        output = replace_once(
            output,
            '__global__ __launch_bounds__(\n'
            '        SimulationBlockSize,\n'
            '        MinimumBlocksPerSm) void SimulateSearchCandidatesKernel(',
            '__global__ FOREVERVALIDATOR_HIP_SEARCH_LAUNCH_BOUNDS(\n'
            '        SimulationBlockSize,\n'
            '        MinimumBlocksPerSm) void SimulateSearchCandidatesKernel(',
        )
    if source.name == "cuda_collision.cuh":
        warp_guard = "#if __HIP_ARCH__ >= 800"
        if output.count(warp_guard) != 2:
            raise ValueError("expected two NVIDIA warp reduction guards")
        output = output.replace(
            warp_guard,
            "#if defined(__HIP_PLATFORM_NVIDIA__) && "
            "defined(__HIP_ARCH__) && (__HIP_ARCH__ >= 800)",
        )
    output = (
        "// Generated from " + source.relative_to(ROOT).as_posix() +
        " by tools/hipify_backend.py. Do not edit.\n" + output
    )
    return output.encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hipify", default="hipify-perl")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    hipify = shutil.which(args.hipify)
    if hipify is None:
        parser.error(f"hipify tool not found: {args.hipify}")
    version = subprocess.run(
        [hipify, "-version"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    ).stdout.strip()
    if version != f"HIP version {HIPIFY_VERSION}":
        parser.error(
            f"expected hipify-perl {HIPIFY_VERSION}, got {version!r}"
        )
    sources = sorted(
        (path, HIP / path.name.replace("cuda", "hip"))
        for path in CUDA.iterdir()
        if path.suffix in SUFFIXES and path.name not in EXCLUDED
    )
    for test in (
        "cuda_timeline_executor_tests.cpp",
        "cuda_search_branch_state_tests.cu",
        "cuda_search_winner_reduction_tests.cu",
        "cuda_modifier_event_ops_tests.cpp",
        "cuda_candidate_events_tests.cpp",
        "cuda_state_layout_tests.cpp",
        "cuda_static_configuration_tests.cpp",
        "cuda_dynamics_tests.cpp",
        "cuda_stunt_tests.cpp",
        "cuda_collision_tests.cpp",
        "cuda_race_tests.cpp",
        "cuda_finish_time_origin_tests.cu",
    ):
        sources.append((
            ROOT / "tests" / test,
            ROOT / "tests/generated" / test.replace("cuda", "hip"),
        ))
    differences = []
    expected = {destination for _, destination in sources}
    for directory in (HIP, ROOT / "tests/generated"):
        if directory.is_dir():
            for destination in directory.iterdir():
                if destination.is_file() and destination not in expected:
                    if args.check:
                        differences.append(str(destination.relative_to(ROOT)))
                    else:
                        destination.unlink()
    for source, destination in sources:
        generated = translate(source, hipify)
        if args.check:
            if not destination.exists() or destination.read_bytes() != generated:
                differences.append(str(destination.relative_to(ROOT)))
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.read_bytes() != generated:
                destination.write_bytes(generated)
    if differences:
        print("Out-of-date HIP files:\n" + "\n".join(differences), file=sys.stderr)
        return 1
    print(f"HIP generation {'checked' if args.check else 'updated'}: {len(sources)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
