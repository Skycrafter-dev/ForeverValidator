#!/usr/bin/env python3

"""Run and validate a production/profiling CUDA benchmark pair."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from typing import Any


# These fields describe the workload, logical ordering, progress, and result.
# Unlike timing and profiling overhead, they must be exactly identical when
# hot-path counters are enabled.
PARITY_KEYS = (
    "repetition",
    "candidates",
    "evaluated_candidates",
    "qualifying_candidates",
    "closest_target_distance",
    "calibrated_batch_size",
    "baseline_input_events",
    "best_input_events",
    "total_mutation_count",
    "timeline_ticks",
    "branch_time_ms",
    "mutable_from_time_ms",
    "modifier_from_time_ms",
    "input_events_per_second",
    "synthetic_input_start_ms",
    "normalized_input_events",
    "batch_capacity",
    "sort_candidates_by_locality",
    "reuse_baseline_prefixes",
    "deduplicate_low_entropy_inputs",
    "forced_minimum_blocks_per_sm",
    "existing_minimum_count",
    "existing_maximum_count",
    "modifier",
    "mutation_pipeline",
    "evaluator",
    "cancelled",
    "best_changed",
    "best_is_mutation",
    "best_candidate_id",
    "best_evaluation_tick",
    "best_score",
    "best_time_ms",
    "best_detail_0",
    "best_detail_1",
    "best_mutation_count",
    "mutation_improvement_count",
    "best_state_fingerprint",
    "best_input_count",
    "best_input_fingerprint",
    "baseline_prefix_reuse_active",
    "candidate_deduplication_active",
    "simulated_candidate_count",
    "deduplicated_candidate_count",
)

# Profiling legitimately changes these totals or kernel characteristics. They
# remain required diagnostics, but this paired runner does not compare them.
# Production-off resource parity is checked separately against the PR09 build.
RESOURCE_DIAGNOSTIC_KEYS = (
    "simulation_threads_per_block",
    "simulation_minimum_blocks_per_sm",
    "simulation_registers_per_thread",
    "simulation_local_bytes_per_thread",
    "simulation_active_blocks_per_sm",
    "simulation_theoretical_occupancy",
    "resident_device_bytes",
    "mutation_device_bytes",
    "candidate_input_device_bytes",
    "mutation_scratch_device_bytes",
    "baseline_prefix_device_bytes",
    "candidate_prefix_device_bytes",
    "candidate_deduplication_device_bytes",
    "winner_selection_device_bytes",
    "host_to_device_bytes",
    "device_to_host_bytes",
    "initial_host_to_device_bytes",
    "baseline_device_to_host_bytes",
)

TIMING_DIAGNOSTIC_KEYS = (
    "kernel_ms",
    "wall_ms",
    "attempts_per_second",
    "score_initialization_kernel_ms",
    "mutation_kernel_ms",
    "simulation_kernel_ms",
    "finish_refinement_kernel_ms",
    "finish_refinement_tick_equivalents",
    "winner_kernel_ms",
    "winner_reduction_kernel_ms",
    "winner_state_capture_kernel_ms",
    "finalization_kernel_ms",
    "simulation_kernel_ns_per_tick",
    "simulation_kernel_ticks_per_second",
)

PROVENANCE_KEYS = (
    "hot_path_metrics_requested",
    "hot_path_metrics_collected",
    "hot_path_metrics_complete",
    "hot_path_metrics_timing_is_production",
    "hot_path_forced_runtime_generic_kernel",
)

HOT_PATH_COUNTER_KEYS = (
    "hot_path_physically_simulated_candidate_count",
    "hot_path_first_simulation_tick_sum",
    "hot_path_first_simulation_tick_minimum",
    "hot_path_first_simulation_tick_maximum",
    "hot_path_executed_tick_count",
    "hot_path_completed_tick_count",
    "hot_path_physics_substep_count",
    "hot_path_maximum_substeps_per_tick",
    "hot_path_collision_detect_count",
    "hot_path_surface_cache_eligible_count",
    "hot_path_surface_cache_reuse_count",
    "hot_path_surface_cache_refresh_count",
    "hot_path_surface_cache_refresh_failure_count",
    "hot_path_mesh_cache_reuse_count",
    "hot_path_acceleration_cell_visit_count",
    "hot_path_acceleration_surface_visit_count",
    "hot_path_octree_cell_visit_count",
    "hot_path_cached_triangle_leaf_visit_count",
    "hot_path_triangle_test_count",
    "hot_path_triangle_hit_count",
    "hot_path_raw_contact_count",
    "hot_path_response_sort_call_count",
    "hot_path_response_sort_item_count",
    "hot_path_maximum_response_sort_item_count",
    "hot_path_ground_force_pass_count",
    "hot_path_air_force_pass_count",
    "hot_path_water_force_pass_count",
    "hot_path_physics_callback_disabled_force_pass_count",
    "hot_path_zero_dynamics_force_pass_count",
)

REQUIRED_BASE_KEYS = (
    PARITY_KEYS + RESOURCE_DIAGNOSTIC_KEYS + TIMING_DIAGNOSTIC_KEYS
)

REFERENCE_PROVENANCE = {
    "hot_path_metrics_requested": False,
    "hot_path_metrics_collected": False,
    "hot_path_metrics_complete": False,
    "hot_path_metrics_timing_is_production": True,
    "hot_path_forced_runtime_generic_kernel": False,
}

PROFILED_PROVENANCE = {
    "hot_path_metrics_requested": True,
    "hot_path_metrics_collected": True,
    "hot_path_metrics_complete": True,
    "hot_path_metrics_timing_is_production": False,
    "hot_path_forced_runtime_generic_kernel": True,
}


class PairValidationError(RuntimeError):
    """A benchmark invocation or its paired contract was invalid."""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse_single_row(stdout: str, label: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise PairValidationError(
            f"{label} benchmark emitted {len(lines)} non-empty stdout "
            "lines; expected exactly one JSON row"
        )
    try:
        row = json.loads(
            lines[0],
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise PairValidationError(
            f"{label} benchmark did not emit valid strict JSON: {error}"
        ) from error
    if not isinstance(row, dict):
        raise PairValidationError(
            f"{label} benchmark JSON row must be an object"
        )
    return row


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def run_benchmark(command: Sequence[str], label: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise PairValidationError(
            f"could not start {label} benchmark "
            f"({_display_command(command)}): {error}"
        ) from error
    if completed.returncode != 0:
        details = [
            f"{label} benchmark exited with code {completed.returncode}",
            f"command: {_display_command(command)}",
        ]
        if completed.stderr.strip():
            details.append(f"stderr:\n{completed.stderr.rstrip()}")
        if completed.stdout.strip():
            details.append(f"stdout:\n{completed.stdout.rstrip()}")
        raise PairValidationError("\n".join(details))
    return parse_single_row(completed.stdout, label)


def _missing_keys(row: dict[str, Any], keys: Sequence[str]) -> list[str]:
    return [key for key in keys if key not in row]


def _nonnegative_integer(
    row: dict[str, Any], key: str, label: str, problems: list[str]
) -> int | None:
    if key not in row:
        return None
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        problems.append(
            f"{label}.{key} must be a non-negative JSON integer; "
            f"got {value!r}"
        )
        return None
    return value


def _same_json_value(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def validate_pair(
    reference: dict[str, Any], profiled: dict[str, Any]
) -> None:
    problems: list[str] = []

    for label, row in (("reference", reference), ("profiled", profiled)):
        required_keys = REQUIRED_BASE_KEYS + PROVENANCE_KEYS
        if label == "profiled":
            required_keys += HOT_PATH_COUNTER_KEYS
        missing = _missing_keys(row, required_keys)
        if missing:
            problems.append(
                f"{label} row is missing required keys: "
                + ", ".join(missing)
            )
        forbidden = sorted(
            key
            for key in row
            if key.startswith("hot_path_")
            and ("empty_air" in key or "mesh_shape_mask" in key)
        )
        if forbidden:
            problems.append(
                f"{label} row contains PR13/discarded counters: "
                + ", ".join(forbidden)
            )

    for key in PARITY_KEYS:
        if key not in reference or key not in profiled:
            continue
        if not _same_json_value(reference[key], profiled[key]):
            problems.append(
                f"semantic parity mismatch for {key}: "
                f"reference={reference[key]!r}, profiled={profiled[key]!r}"
            )

    for label, row, expected in (
        ("reference", reference, REFERENCE_PROVENANCE),
        ("profiled", profiled, PROFILED_PROVENANCE),
    ):
        for key, expected_value in expected.items():
            if key not in row:
                continue
            actual = row[key]
            if not isinstance(actual, bool) or actual is not expected_value:
                problems.append(
                    f"{label}.{key} must be {expected_value!r}; "
                    f"got {actual!r}"
                )

    for key in HOT_PATH_COUNTER_KEYS:
        value = _nonnegative_integer(reference, key, "reference", problems)
        if value not in (None, 0):
            problems.append(
                f"reference.{key} must be zero when collection is disabled; "
                f"got {value}"
            )

    counters = {
        key: _nonnegative_integer(profiled, key, "profiled", problems)
        for key in HOT_PATH_COUNTER_KEYS
    }
    evaluated = _nonnegative_integer(
        profiled, "evaluated_candidates", "profiled", problems
    )
    timeline = _nonnegative_integer(
        profiled, "timeline_ticks", "profiled", problems
    )

    if (
        any(value is None for value in counters.values())
        or evaluated is None
        or timeline is None
    ):
        if problems:
            raise PairValidationError(
                "hot-path pair validation failed:\n  - "
                + "\n  - ".join(problems)
            )
        return

    physical = counters["hot_path_physically_simulated_candidate_count"]
    first_sum = counters["hot_path_first_simulation_tick_sum"]
    first_min = counters["hot_path_first_simulation_tick_minimum"]
    first_max = counters["hot_path_first_simulation_tick_maximum"]
    executed = counters["hot_path_executed_tick_count"]
    completed = counters["hot_path_completed_tick_count"]
    substeps = counters["hot_path_physics_substep_count"]
    maximum_substeps = counters["hot_path_maximum_substeps_per_tick"]
    detects = counters["hot_path_collision_detect_count"]
    eligible = counters["hot_path_surface_cache_eligible_count"]
    reuse = counters["hot_path_surface_cache_reuse_count"]
    refresh = counters["hot_path_surface_cache_refresh_count"]
    failures = counters["hot_path_surface_cache_refresh_failure_count"]
    tests = counters["hot_path_triangle_test_count"]
    hits = counters["hot_path_triangle_hit_count"]
    sort_calls = counters["hot_path_response_sort_call_count"]
    sort_items = counters["hot_path_response_sort_item_count"]
    maximum_sort_items = counters[
        "hot_path_maximum_response_sort_item_count"
    ]
    ground = counters["hot_path_ground_force_pass_count"]
    air = counters["hot_path_air_force_pass_count"]
    water = counters["hot_path_water_force_pass_count"]
    callbacks_disabled = counters[
        "hot_path_physics_callback_disabled_force_pass_count"
    ]
    zero_dynamics = counters["hot_path_zero_dynamics_force_pass_count"]

    if physical > evaluated:
        problems.append(
            "physically simulated candidates exceed evaluated candidates: "
            f"{physical} > {evaluated}"
        )
    if completed != executed:
        problems.append(
            f"completed ticks must equal executed ticks: {completed} != "
            f"{executed}"
        )
    if executed > physical * timeline:
        problems.append(
            "executed ticks exceed the candidate/timeline bound: "
            f"{executed} > {physical} * {timeline}"
        )
    if not (substeps == detects == sort_calls):
        problems.append(
            "physics substeps, collision detects, and response sorts must "
            f"match: {substeps}, {detects}, {sort_calls}"
        )
    if eligible != reuse + refresh:
        problems.append(
            "surface-cache accounting must satisfy eligible = reuse + "
            f"refresh: {eligible} != {reuse} + {refresh}"
        )
    force_total = ground + air + callbacks_disabled + zero_dynamics
    if force_total != substeps:
        problems.append(
            "force-path accounting must equal physics substeps: "
            f"{ground} + {air} + {callbacks_disabled} + {zero_dynamics} "
            f"!= {substeps}"
        )
    if hits > tests:
        problems.append(f"triangle hits exceed tests: {hits} > {tests}")
    if failures > refresh:
        problems.append(
            f"surface-cache refresh failures exceed refreshes: "
            f"{failures} > {refresh}"
        )
    if water > ground + air:
        problems.append(
            f"water force passes exceed ground + air passes: "
            f"{water} > {ground + air}"
        )
    if maximum_substeps > substeps:
        problems.append(
            "maximum substeps for one tick exceed total substeps: "
            f"{maximum_substeps} > {substeps}"
        )
    if maximum_sort_items > sort_items:
        problems.append(
            "maximum response-sort items exceed total sorted items: "
            f"{maximum_sort_items} > {sort_items}"
        )

    if physical == 0:
        if (first_sum, first_min, first_max) != (0, 0, 0):
            problems.append(
                "zero physical candidates require zero first-tick "
                f"statistics; got sum/min/max="
                f"{first_sum}/{first_min}/{first_max}"
            )
    else:
        if timeline == 0:
            problems.append(
                "a profiled row with physical candidates requires a "
                "positive timeline"
            )
        elif first_max >= timeline:
            problems.append(
                f"maximum first simulation tick is outside the timeline: "
                f"{first_max} >= {timeline}"
            )
        if first_min > first_max:
            problems.append(
                f"minimum first simulation tick exceeds maximum: "
                f"{first_min} > {first_max}"
            )
        if first_sum < physical * first_min:
            problems.append(
                "first simulation tick sum is below its minimum bound: "
                f"{first_sum} < {physical} * {first_min}"
            )
        if first_sum > physical * first_max:
            problems.append(
                "first simulation tick sum exceeds its maximum bound: "
                f"{first_sum} > {physical} * {first_max}"
            )
        if executed + first_sum > physical * timeline:
            problems.append(
                "executed ticks plus skipped prefix ticks exceed the "
                f"timeline bound: {executed} + {first_sum} > "
                f"{physical} * {timeline}"
            )

    if problems:
        raise PairValidationError(
            "hot-path pair validation failed:\n  - "
            + "\n  - ".join(problems)
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "run one production AOT CUDA benchmark and one instrumented "
            "generic-AOT benchmark, then validate exact semantic parity"
        )
    )
    parser.add_argument("benchmark", help="CUDA search benchmark executable")
    parser.add_argument("packs", help="installed TrackMania Packs directory")
    parser.add_argument("replay", help="input Replay.Gbx path")
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help=(
            "remaining cuda_search_benchmark arguments; a leading -- "
            "separator is optional"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    benchmark_args = list(args.benchmark_args)
    if benchmark_args[:1] == ["--"]:
        benchmark_args.pop(0)
    if "--hot-path-metrics" in benchmark_args:
        print(
            "error: --hot-path-metrics is controlled by this paired runner",
            file=sys.stderr,
        )
        return 2

    base_command = [args.benchmark, args.packs, args.replay, *benchmark_args]
    try:
        reference = run_benchmark(base_command, "reference")
        profiled = run_benchmark(
            [*base_command, "--hot-path-metrics"], "profiled"
        )
        validate_pair(reference, profiled)
    except PairValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    json.dump(
        {
            "schema": "forevervalidator-cuda-hotpath-pair-v1",
            "reference": reference,
            "profiled": profiled,
        },
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
