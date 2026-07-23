#!/usr/bin/env python3
"""Run the fixed OptimizedCpu benchmark and retain reproducible raw evidence."""

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    if not path.is_file():
        raise RuntimeError(f"missing input: {path}")
    size = path.stat().st_size
    if size != expected["size"]:
        raise RuntimeError(
            f"size mismatch for {path}: expected {expected['size']}, got {size}"
        )
    actual_hash = sha256(path)
    if actual_hash != expected["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {actual_hash}")
    return {"path": str(path), "size": size, "sha256": actual_hash}


def parse_result(output):
    fields = {}
    for token in output.strip().split():
        key, separator, value = token.partition("=")
        if not separator:
            raise RuntimeError(f"unrecognized benchmark output token: {token}")
        fields[key] = value
    samples = [int(value) for value in fields.pop("samples_ns").split(",")]
    result = {key: int(value) for key, value in fields.items() if key != "backend"}
    result["backend"] = fields["backend"]
    result["samples_ns"] = samples
    return result


def run_capture(command, cwd=None):
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def read_optional(path):
    try:
        return pathlib.Path(path).read_text(encoding="ascii").strip()
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        return None


def upper_median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--benchmark", required=True, type=pathlib.Path)
    parser.add_argument("--packs", required=True, type=pathlib.Path)
    parser.add_argument("--replay", required=True, type=pathlib.Path)
    parser.add_argument("--cpu", required=True, type=int)
    parser.add_argument("--build-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    verified_replay = verify_file(arguments.replay, manifest["workload"]["replay"])
    verified_packs = []
    for expected in manifest["workload"]["packs"]:
        verified_packs.append(
            verify_file(arguments.packs / expected["name"], expected)
        )
    benchmark_path = arguments.benchmark.resolve()
    if not benchmark_path.is_file():
        raise RuntimeError(f"missing benchmark executable: {benchmark_path}")

    compile_database = arguments.build_dir / "compile_commands.json"
    compile_command = None
    if compile_database.is_file():
        entries = json.loads(compile_database.read_text(encoding="utf-8"))
        for entry in entries:
            if entry.get("file", "").endswith("tools/optimized_cpu_benchmark.cpp"):
                compile_command = entry.get("command") or entry.get("arguments")
                break

    run_records = []
    combined_samples = {"reference": [], "optimized-cpu": []}
    fingerprints = set()
    for order in protocol["backend_orders"]:
        for backend in order:
            command = [
                "taskset",
                "-c",
                str(arguments.cpu),
                str(benchmark_path),
                str(arguments.packs.resolve()),
                str(arguments.replay.resolve()),
                backend,
                str(protocol["tick_count"]),
                str(protocol["warmup_count"]),
                str(protocol["repetition_count"]),
            ]
            completed = subprocess.run(
                command,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            parsed = parse_result(completed.stdout)
            if parsed["backend"] != backend:
                raise RuntimeError("benchmark reported the wrong backend")
            if len(parsed["samples_ns"]) != protocol["repetition_count"]:
                raise RuntimeError("benchmark reported the wrong sample count")
            fingerprints.add(parsed["final_fingerprint"])
            combined_samples[backend].extend(parsed["samples_ns"])
            run_records.append(
                {
                    "order": order,
                    "command": command,
                    "stderr": completed.stderr,
                    "result": parsed,
                }
            )
    if len(fingerprints) != 1:
        raise RuntimeError("backend final-state fingerprints differ")

    medians = {
        backend: upper_median(samples)
        for backend, samples in combined_samples.items()
    }
    speedup = medians["reference"] / medians["optimized-cpu"]
    target = manifest["target"]
    target_met = (
        medians["optimized-cpu"] <= target["maximum_wall_time_ns"]
        and speedup >= target["minimum_median_speedup"]
    )

    repository = pathlib.Path(__file__).resolve().parent.parent
    evidence = {
        "schema": "forevervalidator-optimized-cpu-measurement-v1",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "manifest": {
            "path": str(arguments.manifest.resolve()),
            "sha256": sha256(arguments.manifest),
        },
        "source": {
            "repository": str(repository),
            "commit": run_capture(["git", "rev-parse", "HEAD"], repository),
            "status_porcelain": run_capture(
                ["git", "status", "--porcelain"], repository
            ),
        },
        "build": {
            "directory": str(arguments.build_dir.resolve()),
            "benchmark": str(benchmark_path),
            "benchmark_sha256": sha256(benchmark_path),
            "compile_command": compile_command,
            "cmake_cache_sha256": sha256(arguments.build_dir / "CMakeCache.txt")
            if (arguments.build_dir / "CMakeCache.txt").is_file()
            else None,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu": arguments.cpu,
            "requested_affinity": [arguments.cpu],
            "runner_affinity": sorted(os.sched_getaffinity(0)),
            "lscpu": run_capture(["lscpu"]),
            "governor": read_optional(
                f"/sys/devices/system/cpu/cpu{arguments.cpu}/cpufreq/scaling_governor"
            ),
            "energy_performance_preference": read_optional(
                f"/sys/devices/system/cpu/cpu{arguments.cpu}/cpufreq/energy_performance_preference"
            ),
            "boost": read_optional("/sys/devices/system/cpu/cpufreq/boost"),
        },
        "inputs": {"replay": verified_replay, "packs": verified_packs},
        "protocol": protocol,
        "runs": run_records,
        "combined_samples_ns": combined_samples,
        "result": {
            "reference_median_ns": medians["reference"],
            "optimized_cpu_median_ns": medians["optimized-cpu"],
            "median_speedup": speedup,
            "final_fingerprint": next(iter(fingerprints)),
            "target_met": target_met,
        },
    }

    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(arguments.output)
    sys.stdout.write(rendered)
    return 0 if target_met else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"run_optimized_cpu_benchmark.py: {error}", file=sys.stderr)
        sys.exit(1)
