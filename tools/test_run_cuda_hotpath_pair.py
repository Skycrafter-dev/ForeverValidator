#!/usr/bin/env python3

import contextlib
import copy
import io
import json
import subprocess
import unittest
from unittest import mock

import run_cuda_hotpath_pair as pair


def valid_rows():
    reference = {
        key: 0
        for key in (
            pair.REQUIRED_BASE_KEYS + pair.PROVENANCE_KEYS
        )
    }
    reference.update(
        {
            "repetition": 0,
            "candidates": 4,
            "evaluated_candidates": 4,
            "qualifying_candidates": 2,
            "closest_target_distance": None,
            "calibrated_batch_size": 4,
            "baseline_input_events": 3,
            "best_input_events": 3,
            "timeline_ticks": 10,
            "branch_time_ms": 5000,
            "mutable_from_time_ms": 5000,
            "modifier_from_time_ms": 5000,
            "input_events_per_second": 10,
            "synthetic_input_start_ms": 5000,
            "normalized_input_events": 3,
            "batch_capacity": 4,
            "sort_candidates_by_locality": True,
            "reuse_baseline_prefixes": True,
            "deduplicate_low_entropy_inputs": True,
            "forced_minimum_blocks_per_sm": 16,
            "existing_minimum_count": 1,
            "existing_maximum_count": 16,
            "modifier": "random-steering",
            "mutation_pipeline": "optimized",
            "evaluator": "velocity",
            "cancelled": False,
            "best_changed": True,
            "best_is_mutation": True,
            "best_candidate_id": 2,
            "best_evaluation_tick": 7,
            "best_score": 12.5,
            "best_time_ms": 5100,
            "best_detail_0": 1.25,
            "best_detail_1": 2.5,
            "best_mutation_count": 1,
            "mutation_improvement_count": 1,
            "best_state_fingerprint": 1234567890123456789,
            "best_input_count": 3,
            "best_input_fingerprint": 9876543210987654321,
            "baseline_prefix_reuse_active": True,
            "candidate_deduplication_active": True,
            "simulated_candidate_count": 3,
            "deduplicated_candidate_count": 1,
            **pair.REFERENCE_PROVENANCE,
        }
    )

    profiled = copy.deepcopy(reference)
    profiled.update(pair.PROFILED_PROVENANCE)
    profiled.update({key: 0 for key in pair.HOT_PATH_COUNTER_KEYS})
    profiled.update(
        {
            "resident_device_bytes": 4096,
            "device_to_host_bytes": 2048,
            "kernel_ms": 2.0,
            "wall_ms": 3.0,
            "hot_path_physically_simulated_candidate_count": 3,
            "hot_path_first_simulation_tick_sum": 3,
            "hot_path_first_simulation_tick_minimum": 0,
            "hot_path_first_simulation_tick_maximum": 2,
            "hot_path_executed_tick_count": 27,
            "hot_path_completed_tick_count": 27,
            "hot_path_physics_substep_count": 27,
            "hot_path_maximum_substeps_per_tick": 1,
            "hot_path_collision_detect_count": 27,
            "hot_path_surface_cache_eligible_count": 27,
            "hot_path_surface_cache_reuse_count": 20,
            "hot_path_surface_cache_refresh_count": 7,
            "hot_path_surface_cache_refresh_failure_count": 1,
            "hot_path_mesh_cache_reuse_count": 5,
            "hot_path_acceleration_cell_visit_count": 40,
            "hot_path_acceleration_surface_visit_count": 30,
            "hot_path_octree_cell_visit_count": 20,
            "hot_path_cached_triangle_leaf_visit_count": 8,
            "hot_path_triangle_test_count": 100,
            "hot_path_triangle_hit_count": 10,
            "hot_path_raw_contact_count": 30,
            "hot_path_response_sort_call_count": 27,
            "hot_path_response_sort_item_count": 30,
            "hot_path_maximum_response_sort_item_count": 2,
            "hot_path_ground_force_pass_count": 20,
            "hot_path_air_force_pass_count": 5,
            "hot_path_water_force_pass_count": 2,
            "hot_path_physics_callback_disabled_force_pass_count": 1,
            "hot_path_zero_dynamics_force_pass_count": 1,
        }
    )
    return reference, profiled


class HotPathPairTests(unittest.TestCase):
    def test_main_runs_reference_then_profiled_and_emits_pair(self):
        reference, profiled = valid_rows()
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            row = reference if len(calls) == 1 else profiled
            return subprocess.CompletedProcess(
                command, 0, json.dumps(row) + "\n", ""
            )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(pair.subprocess, "run", side_effect=fake_run):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                stderr
            ):
                result = pair.main(
                    [
                        "benchmark.exe",
                        "Packs",
                        "run.Replay.Gbx",
                        "4",
                        "10",
                        "1",
                        "--minimum-blocks",
                        "16",
                    ]
                )

        self.assertEqual(result, 0, stderr.getvalue())
        self.assertEqual(len(calls), 2)
        self.assertNotIn("--hot-path-metrics", calls[0])
        self.assertEqual(calls[1][-1], "--hot-path-metrics")
        output = json.loads(stdout.getvalue())
        self.assertEqual(
            output["schema"], "forevervalidator-cuda-hotpath-pair-v1"
        )
        self.assertEqual(output["reference"], reference)
        self.assertEqual(output["profiled"], profiled)

    def test_semantic_mismatch_names_the_field(self):
        reference, profiled = valid_rows()
        profiled["best_state_fingerprint"] += 1
        with self.assertRaisesRegex(
            pair.PairValidationError, "best_state_fingerprint"
        ):
            pair.validate_pair(reference, profiled)

    def test_counter_invariant_has_clear_diagnostic(self):
        reference, profiled = valid_rows()
        profiled["hot_path_collision_detect_count"] -= 1
        with self.assertRaisesRegex(
            pair.PairValidationError,
            "physics substeps, collision detects, and response sorts",
        ):
            pair.validate_pair(reference, profiled)

    def test_reference_may_omit_uncollected_counters(self):
        reference, profiled = valid_rows()
        self.assertFalse(
            any(key in reference for key in pair.HOT_PATH_COUNTER_KEYS)
        )
        pair.validate_pair(reference, profiled)

    def test_profiled_row_requires_every_counter(self):
        reference, profiled = valid_rows()
        del profiled["hot_path_raw_contact_count"]
        with self.assertRaisesRegex(
            pair.PairValidationError,
            "profiled row is missing required keys: hot_path_raw_contact_count",
        ):
            pair.validate_pair(reference, profiled)

    def test_present_reference_counter_must_be_zero(self):
        reference, profiled = valid_rows()
        reference["hot_path_raw_contact_count"] = 1
        with self.assertRaisesRegex(
            pair.PairValidationError,
            "reference.hot_path_raw_contact_count must be zero",
        ):
            pair.validate_pair(reference, profiled)

    def test_requires_exactly_one_json_row(self):
        with self.assertRaisesRegex(
            pair.PairValidationError, "2 non-empty stdout lines"
        ):
            pair.parse_single_row("{}\n{}\n", "profiled")

    def test_rejects_pr13_empty_air_counter(self):
        reference, profiled = valid_rows()
        profiled["hot_path_empty_air_certificate_success_count"] = 1
        with self.assertRaisesRegex(
            pair.PairValidationError, "PR13/discarded counters"
        ):
            pair.validate_pair(reference, profiled)


if __name__ == "__main__":
    unittest.main()
