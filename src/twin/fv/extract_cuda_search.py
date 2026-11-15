#!/usr/bin/env python3
"""Extract FV's CUDA search mutation code for the host (twin search, M13).

The twin's CPU search must generate exactly the candidates FV's GPU search
generates for the same seeds. Rather than keep a hand-made copy, this script
copies the mutation functions out of
src/simulation/backends/cuda/cuda_search_executor.cu at build time, the same
way tools/generate_vulkan_search_shader.py feeds the Vulkan backend, and
strips the CUDA qualifiers. The functions are plain C++ (no thread indices,
atomics or intrinsics); the script refuses to emit anything that is not.

    extract_cuda_search.py SOURCE.cu OUTPUT.inc
    extract_cuda_search.py --device SOURCE.cu OUTPUT.inc
    extract_cuda_search.py --judge SOURCE.cu OUTPUT.inc

--device: the mutation functions only (IsAnalog .. ApplyModifier), with their
CUDA qualifiers kept, for the GPU twin session's generation on the device
(src/twin/fv/twin_search_device.cu); FV's generator class becomes
DeviceMt19937Fv (the device file's own DeviceMt19937 is checked against it).

--judge: the evaluator's types and functions with their CUDA qualifiers, for
the twin GPU engine's judge (src/twin/gpu/tw_gpu_judge.cuh): the evaluator
configuration and its enums (cuda_search_executor.h), DeviceSample
(cuda_search_winner_selection.cuh), the custom volume (cuda_custom_volume.cuh)
and the per-tick evaluation (condition programs, scripted objectives,
EvaluateState, the checkpoint counters). They name FV's state types; the
judge declares stand-ins with the members they read. The other files are
found next to SOURCE.cu.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Top-level entities, in source order. Each must exist exactly once.
ENTITIES = (
    "CanonicalBaselineInputs",
    "IsAnalog",
    "IsSwitch",
    "IsSteerAction",
    "SaturateAnalog",
    "SameEvent",
    "class DeviceMt19937",
    "ShuffleIndices",
    "NormalizeEvents",
    "EffectiveChangeCount",
    "SteeringStateAt",
    "SwitchStateAt",
    "PushEvent",
    "AnalogEvent",
    "SwitchEvent",
    "CollectEligible",
    "ApplyModifier",
    "ContainsVolume",
    "SegmentEntry",
    "struct DeviceConditionValue",
    "ConditionLength",
    "ConditionLocalSpeed",
    "ConditionAngles",
    "ConditionSource",
    "struct DeviceExpressionResult",
    "EvaluateExpression",
    "EvaluateCondition",
    "UpdateScriptedSample",
    "EntrySpeed",
    "namespace distance_target",
    "EvaluateState",
    "struct CheckpointProgressBefore",
    "CheckpointEventAccepted",
    "DistanceTargetReached",
    "MaximizesScore",
)

# --device: the generation's functions (a contiguous run of ENTITIES).
DEVICE_ENTITIES = ENTITIES[ENTITIES.index("IsAnalog"):ENTITIES.index("ApplyModifier") + 1]

# --judge: (file next to SOURCE.cu, entities in its source order, namespace to wrap them in or None).
JUDGE_FILES = (
    ("cuda_search_executor.h", (
        "enum CudaSearchEvaluatorKind",
        "enum CudaSearchConditionOpcode",
        "enum CudaSearchConditionValue",
        "struct CudaSearchConditionInstruction",
        "lines CudaSearchMaximumScriptedObjectives CudaSearchEntrySpeed",
        "struct CudaSearchScriptedObjective",
        "struct CudaSearchEvaluatorConfiguration",
    ), None),
    ("cuda_search_winner_selection.cuh", (
        "lines InvalidCandidateSlot InvalidCandidateSlot",
        "struct DeviceSample",
    ), None),
    ("cuda_custom_volume.cuh", (
        "struct Point",
        "Project",
        "Contains",
        "AddCandidate",
        "SegmentEntry",
    ), "cuda::custom_volume"),
    ("cuda_search_executor.cu", ENTITIES[ENTITIES.index("ContainsVolume"):ENTITIES.index("MaximizesScore")], None),
)

FORBIDDEN = re.compile(
    r"\b(blockIdx|threadIdx|blockDim|gridDim|__syncthreads|__shfl\w*|atomic\w*|"
    r"__ldg|cudaMalloc|cudaMemcpy|__global__|__shared__|warpSize)\b")


def find_start(lines: list[str], entity: str) -> int:
    if entity.startswith("lines "):
        pattern = re.compile(rf"^(?:inline )?constexpr [\w:]+ {entity.split()[1]}\b")
    elif entity.startswith("enum "):
        pattern = re.compile(rf"^enum class {entity.split()[1]}\b")
    elif entity.startswith("class ") or entity.startswith("struct "):
        keyword, name = entity.split()
        pattern = re.compile(rf"^{keyword} {name}\b")
    elif entity.startswith("namespace "):
        name = entity.split()[1]
        pattern = re.compile(rf"^namespace {name} \{{")
    else:
        pattern = re.compile(
            rf"^(?:__host__ __device__ |__device__ |)[A-Za-z_][\w:<>,\s\*&]*?\b{entity}\(")
    hits = [i for i, line in enumerate(lines) if pattern.match(line)]
    if len(hits) != 1:
        raise SystemExit(f"extract_cuda_search: '{entity}' found {len(hits)} times (expected 1)")
    return hits[0]


def find_end(lines: list[str], start: int) -> int:
    depth = 0
    opened = False
    for i in range(start, len(lines)):
        code = re.sub(r"//.*", "", lines[i])
        code = re.sub(r"'(\\.|[^'])'", "''", code)
        code = re.sub(r'"(\\.|[^"])*"', '""', code)
        for ch in code:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth == 0:
                    return i
    raise SystemExit(f"extract_cuda_search: unbalanced braces from line {start + 1}")


def find_lines_end(lines: list[str], start: int, entity: str) -> int:
    last = re.compile(rf"^(?:inline )?constexpr [\w:]+ {entity.split()[2]}\b")
    for i in range(start, len(lines)):
        if last.match(lines[i]):
            return i
    raise SystemExit(f"extract_cuda_search: '{entity}' has no last line")


def extract_judge(source: Path, output: Path) -> int:
    out = [
        "// GENERATED by src/twin/fv/extract_cuda_search.py --judge from FV's CUDA search",
        "// (cuda_search_executor.h/.cu, cuda_search_winner_selection.cuh, cuda_custom_volume.cuh). Do not edit.",
        "// Device copies of FV's search evaluator (qualifiers kept).",
        "",
    ]
    for name, entities, namespace in JUDGE_FILES:
        lines = (source.parent / name).read_text().splitlines()
        if namespace:
            out.append(f"namespace {namespace} {{")
        previous = -1
        for entity in entities:
            start = find_start(lines, entity)
            if start <= previous:
                raise SystemExit(f"extract_cuda_search: '{entity}' is out of source order")
            end = find_lines_end(lines, start, entity) if entity.startswith("lines ") else find_end(lines, start)
            previous = end
            block = lines[start:end + 1]
            if entity.split()[0] in ("class", "struct", "enum"):
                block[-1] = block[-1] if block[-1].rstrip().endswith(";") else block[-1] + ";"
            text = "\n".join(block)
            bad = FORBIDDEN.search(text)
            if bad:
                raise SystemExit(f"extract_cuda_search: '{entity}' uses device-only '{bad.group(0)}'")
            out.append(f"// ---- {entity} ({name}:{start + 1}-{end + 1})")
            out.append(text)
            out.append("")
        if namespace:
            out.append(f"}}  // namespace {namespace}")
            out.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    new = "\n".join(out) + "\n"
    if not output.exists() or output.read_text() != new:
        output.write_text(new)
    return 0


def main() -> int:
    args = sys.argv[1:]
    if len(args) == 3 and args[0] == "--judge":
        return extract_judge(Path(args[1]), Path(args[2]))
    device = bool(args) and args[0] == "--device"
    if device:
        args = args[1:]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    source = Path(args[0])
    lines = source.read_text().splitlines()
    out = [
        "// GENERATED by src/twin/fv/extract_cuda_search.py from",
        f"// {source.as_posix().split('/src/', 1)[-1] if '/src/' in source.as_posix() else source.name}. Do not edit.",
        "// Device copies of FV's CUDA search mutation functions (qualifiers kept)." if device else
        "// Host copies of FV's CUDA search mutation functions (CUDA qualifiers removed).",
        "",
    ]
    previous = -1
    for entity in (DEVICE_ENTITIES if device else ENTITIES):
        start = find_start(lines, entity)
        if start <= previous:
            raise SystemExit(f"extract_cuda_search: '{entity}' is out of source order")
        end = find_end(lines, start)
        previous = end
        block = lines[start:end + 1]
        if entity.startswith("class ") or entity.startswith("struct "):
            block[-1] = block[-1] if block[-1].rstrip().endswith(";") else block[-1] + ";"
        text = "\n".join(block)
        bad = FORBIDDEN.search(text)
        if bad:
            raise SystemExit(f"extract_cuda_search: '{entity}' uses device-only '{bad.group(0)}'")
        if device and entity == "class DeviceMt19937":
            # the device copy brings its own generator (same draws, verified against this one at start): FV's class
            # stays as the reference, renamed
            text = re.sub(r"\bDeviceMt19937\b", "DeviceMt19937Fv", text)
        if not device:
            text = text.replace("__host__ __device__ ", "").replace("__device__ ", "")
            text = text.replace("__forceinline__ ", "inline ").replace("__noinline__ ", "")
        out.append(f"// ---- {entity} (cuda_search_executor.cu:{start + 1}-{end + 1})")
        out.append(text)
        out.append("")
    Path(args[1]).parent.mkdir(parents=True, exist_ok=True)
    new = "\n".join(out) + "\n"
    target = Path(args[1])
    if not target.exists() or target.read_text() != new:
        target.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
