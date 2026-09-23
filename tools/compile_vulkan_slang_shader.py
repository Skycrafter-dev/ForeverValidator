#!/usr/bin/env python3
"""Compile a Slang compute entry with the exact FP32 Vulkan contract."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slangc", type=Path, required=True)
    parser.add_argument("--spirv-dis", type=Path, required=True)
    parser.add_argument("--spirv-as", type=Path, required=True)
    parser.add_argument("--spirv-val", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def inject_exact_float_controls(assembly: str) -> str:
    lines = assembly.splitlines(keepends=True)
    entry_ids = []
    for line in lines:
        match = re.search(r"\bOpEntryPoint\s+GLCompute\s+(%\S+)", line)
        if match is not None:
            entry_ids.append(match.group(1))
    if len(entry_ids) != 1:
        raise RuntimeError(
            f"expected one compute entry point, found {len(entry_ids)}")
    entry_id = entry_ids[0]

    capability_index = max(
        (index for index, line in enumerate(lines)
         if " OpCapability " in line),
        default=-1,
    )
    if capability_index < 0:
        raise RuntimeError("SPIR-V assembly has no capability section")
    if not any("OpCapability FloatControls2" in line for line in lines):
        lines.insert(
            capability_index + 1,
            "               OpCapability FloatControls2\n",
        )

    if not any("SPV_KHR_float_controls2" in line for line in lines):
        extension_index = max(
            (index for index, line in enumerate(lines)
             if " OpExtension " in line),
            default=capability_index + 1,
        )
        lines.insert(
            extension_index + 1,
            '               OpExtension "SPV_KHR_float_controls2"\n',
        )

    mode = (
        f"               OpExecutionModeId {entry_id} "
        "FPFastMathDefault %float %uint_0\n"
    )
    if not any("FPFastMathDefault" in line for line in lines):
        local_size_index = next(
            (index for index, line in enumerate(lines)
             if f"OpExecutionMode {entry_id} LocalSize" in line),
            -1,
        )
        if local_size_index < 0:
            raise RuntimeError("SPIR-V assembly has no LocalSize execution mode")
        lines.insert(local_size_index + 1, mode)
    return "".join(lines)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="forevervalidator-slang-") as value:
        temporary = Path(value)
        raw_spirv = temporary / "shader.raw.spv"
        assembly = temporary / "shader.spvasm"
        patched_assembly = temporary / "shader.exact.spvasm"

        subprocess.run(
            [
                str(args.slangc), str(args.source),
                "-entry", args.entry,
                "-stage", "compute",
                "-target", "spirv",
                "-profile", "spirv_1_5",
                "-O2",
                "-o", str(raw_spirv),
            ],
            check=True,
        )
        subprocess.run(
            [str(args.spirv_dis), str(raw_spirv), "-o", str(assembly)],
            check=True,
        )
        patched_assembly.write_text(
            inject_exact_float_controls(
                assembly.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(args.spirv_as), "--target-env", "vulkan1.2",
                str(patched_assembly), "-o", str(args.output),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(args.spirv_val), "--target-env", "vulkan1.2",
                str(args.output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
