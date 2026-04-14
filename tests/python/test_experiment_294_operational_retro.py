"""Tests for Exp 294: operational retrospective for milestone 2026.04.21.

Spec coverage: REQ-OPS-001, REQ-OPS-002, REQ-OPS-003, REQ-OPS-004,
               SCENARIO-OPS-001, SCENARIO-OPS-002, SCENARIO-OPS-003,
               SCENARIO-OPS-004, SCENARIO-OPS-005, SCENARIO-OPS-006
"""

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers — locate the script and the results root
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RETRO_SCRIPT = SCRIPTS_DIR / "experiment_294_operational_retro.py"
RETRO_RESULT = RESULTS_DIR / "operational_retro_2026_04_21.json"


# ---------------------------------------------------------------------------
# REQ-OPS-001: Retro artifact schema
# SCENARIO-OPS-001: All required top-level fields are present and typed.
# ---------------------------------------------------------------------------


class TestRetroArtifactSchema:
    """REQ-OPS-001: retro artifact must conform to the canonical schema."""

    REQUIRED_FIELDS = {
        "milestone": str,
        "generated_at": str,
        "experiments_in_scope": list,
        "experiments_with_results": list,
        "total_wall_time_minutes": (int, float),
        "experiments_completed": int,
        "exp_per_hour": (int, float),
        "gpu_utilization_distribution": dict,
        "action_item_audit": list,
        "carry_over_rate_pct": (int, float),
        "slowest_experiments": list,
        "bottlenecks_identified": list,
        "structural_action_taken": dict,
        "delta_vs_prior_milestone": dict,
    }

    def test_result_file_exists(self) -> None:
        """SCENARIO-OPS-001: retro result JSON must exist on disk."""
        assert RETRO_RESULT.exists(), f"Missing {RETRO_RESULT}"

    def test_result_is_valid_json(self) -> None:
        """SCENARIO-OPS-001: retro result must be parseable JSON."""
        data = json.loads(RETRO_RESULT.read_text())
        assert isinstance(data, dict)

    def test_required_fields_present(self) -> None:
        """SCENARIO-OPS-001: all required top-level keys are present."""
        data = json.loads(RETRO_RESULT.read_text())
        for field in self.REQUIRED_FIELDS:
            assert field in data, f"Missing required field: {field}"

    def test_required_fields_typed(self) -> None:
        """SCENARIO-OPS-001: all required top-level keys have correct types."""
        data = json.loads(RETRO_RESULT.read_text())
        for field, expected_type in self.REQUIRED_FIELDS.items():
            assert isinstance(data[field], expected_type), (
                f"Field '{field}' should be {expected_type}, "
                f"got {type(data[field])}"
            )

    def test_milestone_value(self) -> None:
        """SCENARIO-OPS-001: milestone must identify 2026.04.21."""
        data = json.loads(RETRO_RESULT.read_text())
        assert data["milestone"] == "2026.04.21"

    def test_generated_at_is_iso8601(self) -> None:
        """SCENARIO-OPS-001: generated_at must be an ISO-8601 timestamp."""
        from datetime import datetime

        data = json.loads(RETRO_RESULT.read_text())
        ts = data["generated_at"]
        # Accept YYYY-MM-DDTHH:MM:SSZ or offset variants
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"generated_at is not ISO-8601: {ts!r}")


# ---------------------------------------------------------------------------
# REQ-OPS-002: Carry-over rate computation
# SCENARIO-OPS-002: Rate is a percentage; 0–100 inclusive; computed from
#   action item audit counts.
# ---------------------------------------------------------------------------


class TestCarryOverRateComputation:
    """REQ-OPS-002: carry-over rate is computed correctly from the audit."""

    def _load(self) -> dict[str, Any]:
        return json.loads(RETRO_RESULT.read_text())

    def test_carry_over_rate_in_valid_range(self) -> None:
        """SCENARIO-OPS-002: carry-over rate must be 0–100."""
        data = self._load()
        rate = data["carry_over_rate_pct"]
        assert 0.0 <= rate <= 100.0, f"carry_over_rate_pct out of range: {rate}"

    def test_carry_over_rate_consistent_with_audit(self) -> None:
        """SCENARIO-OPS-002: rate = deferred / total * 100."""
        data = self._load()
        audit = data["action_item_audit"]
        total = len(audit)
        assert total > 0, "action_item_audit must not be empty"
        deferred = sum(1 for item in audit if item["resolution"] == "deferred")
        expected_rate = round(deferred / total * 100, 1)
        assert abs(data["carry_over_rate_pct"] - expected_rate) < 0.5, (
            f"carry_over_rate_pct {data['carry_over_rate_pct']} != "
            f"computed {expected_rate}"
        )

    def test_carry_over_rate_improved_vs_prior(self) -> None:
        """SCENARIO-OPS-002: milestone 2026.04.21 carry-over must be < 100%.

        Three consecutive milestones had 100% carry-over.  This milestone
        resolved at least one action item, so the rate must be below 100%.
        """
        data = self._load()
        assert data["carry_over_rate_pct"] < 100.0, (
            "Carry-over rate is still 100% — no action items were resolved"
        )


# ---------------------------------------------------------------------------
# REQ-OPS-003: Action item resolution tracking
# SCENARIO-OPS-003: Each item has id, description, resolution, evidence.
# ---------------------------------------------------------------------------


class TestActionItemResolutionTracking:
    """REQ-OPS-003: action item audit entries have required structure."""

    ITEM_REQUIRED_KEYS = {"id", "description", "resolution", "evidence"}
    VALID_RESOLUTIONS = {"resolved", "deferred", "new"}

    def _load(self) -> dict[str, Any]:
        return json.loads(RETRO_RESULT.read_text())

    def test_audit_is_non_empty_list(self) -> None:
        """SCENARIO-OPS-003: action_item_audit must contain ≥ 1 entry."""
        data = self._load()
        assert isinstance(data["action_item_audit"], list)
        assert len(data["action_item_audit"]) >= 1

    def test_each_item_has_required_keys(self) -> None:
        """SCENARIO-OPS-003: every audit entry must have all required keys."""
        data = self._load()
        for i, item in enumerate(data["action_item_audit"]):
            missing = self.ITEM_REQUIRED_KEYS - set(item.keys())
            assert not missing, f"Action item {i} missing keys: {missing}"

    def test_each_resolution_is_valid(self) -> None:
        """SCENARIO-OPS-003: resolution must be resolved|deferred|new."""
        data = self._load()
        for item in data["action_item_audit"]:
            assert item["resolution"] in self.VALID_RESOLUTIONS, (
                f"Invalid resolution {item['resolution']!r} for item {item['id']!r}"
            )

    def test_all_four_prior_retro_items_present(self) -> None:
        """SCENARIO-OPS-003: all four 2026.04.20 action items are audited."""
        data = self._load()
        ids = {item["id"] for item in data["action_item_audit"]}
        expected = {
            "RETRO-2026-04-20-A",
            "RETRO-2026-04-20-B",
            "RETRO-2026-04-20-C",
            "RETRO-2026-04-20-D",
        }
        assert expected.issubset(ids), f"Missing action item IDs: {expected - ids}"

    def test_dual_gpu_item_present(self) -> None:
        """SCENARIO-OPS-003: DualGPURunner item must be in audit."""
        data = self._load()
        descriptions = [item["description"].lower() for item in data["action_item_audit"]]
        assert any("dualgpu" in d or "dual_gpu" in d or "dual gpu" in d for d in descriptions), (
            "DualGPURunner action item not found in audit"
        )

    def test_checkpointing_item_present(self) -> None:
        """SCENARIO-OPS-003: per-question checkpointing item must be in audit."""
        data = self._load()
        descriptions = [item["description"].lower() for item in data["action_item_audit"]]
        assert any("checkpoint" in d for d in descriptions), (
            "Per-question checkpointing action item not found in audit"
        )

    def test_apple_benchmark_item_present(self) -> None:
        """SCENARIO-OPS-003: Apple adversarial benchmark item must be in audit."""
        data = self._load()
        descriptions = [item["description"].lower() for item in data["action_item_audit"]]
        assert any("apple" in d for d in descriptions), (
            "Apple adversarial benchmark action item not found in audit"
        )

    def test_cuda_ort_item_present(self) -> None:
        """SCENARIO-OPS-003: CUDA ORT batch_size item must be in audit."""
        data = self._load()
        descriptions = [item["description"].lower() for item in data["action_item_audit"]]
        assert any("cuda" in d or "ort" in d or "batch" in d for d in descriptions), (
            "CUDA ORT batch_size action item not found in audit"
        )

    def test_evidence_is_string(self) -> None:
        """SCENARIO-OPS-003: evidence field must be a non-empty string."""
        data = self._load()
        for item in data["action_item_audit"]:
            assert isinstance(item["evidence"], str), (
                f"evidence for {item['id']!r} is not a string"
            )
            assert len(item["evidence"]) > 0, (
                f"evidence for {item['id']!r} is empty"
            )


# ---------------------------------------------------------------------------
# REQ-OPS-004: GPU utilization fields
# SCENARIO-OPS-004: Distribution histogram with 0GPU/1GPU/2GPU counts;
#   each experiment entry records gpu_count.
# ---------------------------------------------------------------------------


class TestGPUUtilizationFields:
    """REQ-OPS-004: GPU utilization distribution must be present and valid."""

    def _load(self) -> dict[str, Any]:
        return json.loads(RETRO_RESULT.read_text())

    def test_gpu_distribution_has_all_tiers(self) -> None:
        """SCENARIO-OPS-004: distribution must include 0gpu, 1gpu, 2gpu keys."""
        data = self._load()
        dist = data["gpu_utilization_distribution"]
        for key in ("0gpu", "1gpu", "2gpu"):
            assert key in dist, f"Missing GPU distribution key: {key}"

    def test_gpu_distribution_values_are_non_negative_ints(self) -> None:
        """SCENARIO-OPS-004: distribution values must be non-negative integers."""
        data = self._load()
        dist = data["gpu_utilization_distribution"]
        for key, val in dist.items():
            assert isinstance(val, int) and val >= 0, (
                f"GPU distribution {key}={val!r} is not a non-negative int"
            )

    def test_gpu_distribution_sums_to_experiment_count(self) -> None:
        """SCENARIO-OPS-004: sum of GPU bins equals experiments_completed."""
        data = self._load()
        dist = data["gpu_utilization_distribution"]
        total_in_dist = dist["0gpu"] + dist["1gpu"] + dist["2gpu"]
        assert total_in_dist == data["experiments_completed"], (
            f"GPU distribution total {total_in_dist} != "
            f"experiments_completed {data['experiments_completed']}"
        )

    def test_experiment_entries_have_gpu_count(self) -> None:
        """SCENARIO-OPS-004: each experiment entry must have gpu_count field."""
        data = self._load()
        for exp in data["experiments_in_scope"]:
            assert "gpu_count" in exp, (
                f"Experiment entry {exp.get('experiment_id', '?')} "
                "missing gpu_count"
            )
            assert exp["gpu_count"] in (0, 1, 2), (
                f"gpu_count must be 0, 1, or 2; got {exp['gpu_count']!r}"
            )


# ---------------------------------------------------------------------------
# REQ-OPS-004 (continued): Structural root-cause analysis fields
# SCENARIO-OPS-005: structural_action_taken has required sub-fields.
# ---------------------------------------------------------------------------


class TestStructuralRootCauseFields:
    """REQ-OPS-004: structural_action_taken must document concrete steps."""

    STRUCTURAL_REQUIRED = {
        "description",
        "stories_created",
        "story_paths",
    }

    def _load(self) -> dict[str, Any]:
        return json.loads(RETRO_RESULT.read_text())

    def test_structural_action_has_required_fields(self) -> None:
        """SCENARIO-OPS-005: structural_action_taken must have all keys."""
        data = self._load()
        action = data["structural_action_taken"]
        missing = self.STRUCTURAL_REQUIRED - set(action.keys())
        assert not missing, f"structural_action_taken missing: {missing}"

    def test_stories_created_is_positive_int(self) -> None:
        """SCENARIO-OPS-005: stories_created must be a positive integer."""
        data = self._load()
        n = data["structural_action_taken"]["stories_created"]
        assert isinstance(n, int) and n > 0, (
            f"stories_created must be a positive int, got {n!r}"
        )

    def test_story_paths_are_strings(self) -> None:
        """SCENARIO-OPS-005: story_paths must be a list of strings."""
        data = self._load()
        paths = data["structural_action_taken"]["story_paths"]
        assert isinstance(paths, list)
        for p in paths:
            assert isinstance(p, str), f"story_paths entry is not a string: {p!r}"

    def test_story_files_actually_exist(self) -> None:
        """SCENARIO-OPS-005: every story path in structural_action_taken must exist."""
        data = self._load()
        repo_root = RESULTS_DIR.parent
        paths = data["structural_action_taken"]["story_paths"]
        for p in paths:
            full = repo_root / p
            assert full.exists(), f"Story file listed in retro does not exist: {full}"


# ---------------------------------------------------------------------------
# REQ-OPS-002: Slowest-experiment list
# SCENARIO-OPS-006: top-5 slowest experiments are identified.
# ---------------------------------------------------------------------------


class TestSlowestExperiments:
    """REQ-OPS-002: slowest experiment list must be computed and ranked."""

    def _load(self) -> dict[str, Any]:
        return json.loads(RETRO_RESULT.read_text())

    def test_slowest_experiments_is_non_empty_list(self) -> None:
        """SCENARIO-OPS-006: slowest_experiments must be a non-empty list."""
        data = self._load()
        assert isinstance(data["slowest_experiments"], list)
        assert len(data["slowest_experiments"]) > 0

    def test_each_slowest_entry_has_required_keys(self) -> None:
        """SCENARIO-OPS-006: each entry must have rank, label, duration_minutes."""
        data = self._load()
        required = {"rank", "label", "duration_minutes"}
        for entry in data["slowest_experiments"]:
            missing = required - set(entry.keys())
            assert not missing, f"Slowest entry missing keys: {missing}"

    def test_slowest_experiments_ordered_by_descending_duration(self) -> None:
        """SCENARIO-OPS-006: rank 1 must have the largest duration_minutes."""
        data = self._load()
        entries = data["slowest_experiments"]
        if len(entries) < 2:
            return  # single entry is trivially sorted
        durations = [e["duration_minutes"] for e in entries]
        assert durations == sorted(durations, reverse=True), (
            f"slowest_experiments not sorted descending: {durations}"
        )

    def test_slowest_count_at_most_five(self) -> None:
        """SCENARIO-OPS-006: top-5 list must not exceed 5 entries."""
        data = self._load()
        assert len(data["slowest_experiments"]) <= 5


# ---------------------------------------------------------------------------
# Script existence & basic structure
# ---------------------------------------------------------------------------


class TestScriptExists:
    """Verify the retro script is present and importable."""

    def test_script_file_exists(self) -> None:
        """Retro script must exist at scripts/experiment_294_operational_retro.py."""
        assert RETRO_SCRIPT.exists(), f"Missing script: {RETRO_SCRIPT}"

    def test_script_defines_main(self) -> None:
        """Script must define a main() or if-name-main entry point."""
        source = RETRO_SCRIPT.read_text()
        assert "def main(" in source or '__name__ == "__main__"' in source, (
            "Script must define main() or if __name__ == '__main__' block"
        )

    def test_script_references_action_item_audit(self) -> None:
        """Script must reference action_item_audit in its output."""
        source = RETRO_SCRIPT.read_text()
        assert "action_item_audit" in source

    def test_script_references_carry_over_rate(self) -> None:
        """Script must compute carry_over_rate_pct."""
        source = RETRO_SCRIPT.read_text()
        assert "carry_over_rate_pct" in source

    def test_script_references_gpu_utilization_distribution(self) -> None:
        """Script must compute gpu_utilization_distribution."""
        source = RETRO_SCRIPT.read_text()
        assert "gpu_utilization_distribution" in source
