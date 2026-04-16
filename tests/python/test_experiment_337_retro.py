"""Tests for Exp 337: operational retrospective for milestone 2026.04.24.

Spec coverage: REQ-RETRO-003,
               SCENARIO-RETRO-005, SCENARIO-RETRO-006

Written test-first (before the implementation script exists) per the NEW-001
pattern established in Exp 325.  These tests validate the artifact produced by
scripts/experiment_337_retro.py.

Key properties under test:
- load_retro_artifact(path) validates schema and raises on errors
- n_experiments is an int in [10, 20] for this milestone (Exps 325-336 = 12)
- bottlenecks_identified is a non-empty list
- action_items is a non-empty list; each entry has id, description, estimated_impact_pct
- carry_over is a list of items from the 2026.04.23 retro with resolved booleans
- estimated_next_milestone_speedup_pct is a float (can be 0 or negative, but honest)
- retro_001_resolved: bool — was the 45-min timeout wrapper actually shipped?
- actual_speedup_pct: float — measured speedup vs prior milestone 40.6 min/exp
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RETRO_SCRIPT = SCRIPTS_DIR / "experiment_337_retro.py"
RETRO_RESULT = RESULTS_DIR / "operational_retro_2026_04_24.json"

# Prior milestone baseline for speedup computation.
# Derived from experiment_319_retro.py: 691 total min / 17 experiments = 40.647...
PRIOR_MILESTONE_MEAN_MIN = 40.6


# ---------------------------------------------------------------------------
# Shared loader — load_retro_artifact
#
# Public API documented in REQ-RETRO-003.  Must validate schema and raise
# typed exceptions so callers can distinguish missing files from corrupt JSON
# from schema mismatches.
# ---------------------------------------------------------------------------


def load_retro_artifact(path: Path) -> dict[str, Any]:
    """Load and schema-validate the 2026.04.24 retro artifact.

    Raises FileNotFoundError if the path does not exist.
    Raises ValueError for invalid JSON, non-dict root, or missing required keys.

    Why a separate validator: REQ-RETRO-003 requires this function to be
    testable as a standalone entry point, matching the pattern established in
    Exp 319's test_experiment_319_retro.py.
    """
    if not path.exists():
        raise FileNotFoundError(f"Retro artifact not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in retro artifact: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Retro artifact must be a JSON object")
    _validate_schema(data)
    return data


def _validate_schema(data: dict[str, Any]) -> None:
    """Validate the required top-level keys are present.

    Why centralised: both load_retro_artifact and individual test classes call
    this so the contract is enforced in one place.
    """
    required = {
        "schema",
        "milestone",
        "generated_at",
        "n_experiments",
        "total_wall_time_min",
        "mean_time_per_exp_min",
        "slowest_experiment",
        "retro_001_resolved",
        "retro_002_resolved",
        "actual_speedup_pct",
        "estimated_next_milestone_speedup_pct",
        "carry_over",
        "action_items",
        "bottlenecks_identified",
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Retro artifact missing required keys: {missing}")


# ---------------------------------------------------------------------------
# REQ-RETRO-003: Artifact file and top-level schema
# SCENARIO-RETRO-005 / SCENARIO-RETRO-006
# ---------------------------------------------------------------------------


class TestRetroArtifactSchema:
    """REQ-RETRO-003: the 2026.04.24 retro artifact must conform to its schema."""

    REQUIRED_FIELDS: dict[str, Any] = {
        "schema": str,
        "milestone": str,
        "generated_at": str,
        "n_experiments": int,
        "total_wall_time_min": (int, float),
        "mean_time_per_exp_min": (int, float),
        "slowest_experiment": str,
        "retro_001_resolved": bool,
        "retro_002_resolved": bool,
        "actual_speedup_pct": (int, float),
        "estimated_next_milestone_speedup_pct": (int, float),
        "carry_over": list,
        "action_items": list,
        "bottlenecks_identified": list,
    }

    def test_result_file_exists(self) -> None:
        """SCENARIO-RETRO-005: retro result JSON must exist on disk."""
        assert RETRO_RESULT.exists(), f"Missing {RETRO_RESULT}"

    def test_result_is_valid_json(self) -> None:
        """SCENARIO-RETRO-005: retro result must be parseable JSON."""
        data = json.loads(RETRO_RESULT.read_text())
        assert isinstance(data, dict)

    def test_required_fields_present(self) -> None:
        """SCENARIO-RETRO-005: all required top-level keys are present."""
        data = json.loads(RETRO_RESULT.read_text())
        for field in self.REQUIRED_FIELDS:
            assert field in data, f"Missing required field: {field}"

    def test_required_fields_typed(self) -> None:
        """SCENARIO-RETRO-005: all required fields have the correct types."""
        data = json.loads(RETRO_RESULT.read_text())
        for field, expected_type in self.REQUIRED_FIELDS.items():
            assert isinstance(data[field], expected_type), (
                f"Field '{field}' should be {expected_type}, got {type(data[field])}"
            )

    def test_milestone_value(self) -> None:
        """SCENARIO-RETRO-005: milestone must identify 2026.04.24."""
        data = json.loads(RETRO_RESULT.read_text())
        assert data["milestone"] == "2026.04.24"

    def test_schema_field_is_string(self) -> None:
        """SCENARIO-RETRO-005: schema field must be a non-empty string."""
        data = json.loads(RETRO_RESULT.read_text())
        assert isinstance(data["schema"], str) and len(data["schema"]) > 0

    def test_generated_at_is_iso8601(self) -> None:
        """SCENARIO-RETRO-005: generated_at must be an ISO-8601 timestamp."""
        from datetime import datetime

        data = json.loads(RETRO_RESULT.read_text())
        ts = data["generated_at"]
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"generated_at is not ISO-8601: {ts!r}")

    def test_slowest_experiment_is_non_empty_string(self) -> None:
        """SCENARIO-RETRO-005: slowest_experiment must name an experiment."""
        data = json.loads(RETRO_RESULT.read_text())
        s = data["slowest_experiment"]
        assert isinstance(s, str) and len(s) > 0


# ---------------------------------------------------------------------------
# load_retro_artifact function contract
# ---------------------------------------------------------------------------


class TestLoadRetroArtifact:
    """REQ-RETRO-003: load_retro_artifact must validate schema and raise correctly."""

    def test_load_returns_dict(self) -> None:
        """SCENARIO-RETRO-005: loader returns a dict for a valid artifact."""
        data = load_retro_artifact(RETRO_RESULT)
        assert isinstance(data, dict)

    def test_load_raises_file_not_found(self, tmp_path: Path) -> None:
        """SCENARIO-RETRO-005: raises FileNotFoundError for a nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_retro_artifact(tmp_path / "nonexistent.json")

    def test_load_raises_value_error_for_invalid_json(self, tmp_path: Path) -> None:
        """SCENARIO-RETRO-005: raises ValueError for malformed JSON."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_retro_artifact(bad)

    def test_load_raises_value_error_for_missing_keys(self, tmp_path: Path) -> None:
        """SCENARIO-RETRO-005: raises ValueError when required keys are absent."""
        sparse = tmp_path / "sparse.json"
        sparse.write_text(json.dumps({"milestone": "2026.04.24"}))
        with pytest.raises(ValueError, match="missing required keys"):
            load_retro_artifact(sparse)

    def test_load_raises_value_error_for_non_dict(self, tmp_path: Path) -> None:
        """SCENARIO-RETRO-005: raises ValueError when top level is not a dict."""
        arr = tmp_path / "array.json"
        arr.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="JSON object"):
            load_retro_artifact(arr)


# ---------------------------------------------------------------------------
# REQ-RETRO-003: n_experiments field
# ---------------------------------------------------------------------------


class TestNExperimentsField:
    """REQ-RETRO-003: n_experiments must count Exps 325-336 (= 12)."""

    def test_n_experiments_is_positive_int(self) -> None:
        """SCENARIO-RETRO-005: n_experiments must be a positive integer."""
        data = load_retro_artifact(RETRO_RESULT)
        n = data["n_experiments"]
        assert isinstance(n, int) and n > 0

    def test_n_experiments_in_range(self) -> None:
        """SCENARIO-RETRO-005: milestone 2026.04.24 had 10-20 experiments."""
        data = load_retro_artifact(RETRO_RESULT)
        n = data["n_experiments"]
        assert 10 <= n <= 20, (
            f"n_experiments={n} outside expected range [10, 20] for milestone 2026.04.24"
        )

    def test_mean_consistent_with_total(self) -> None:
        """SCENARIO-RETRO-006: mean_time_per_exp_min = total_wall_time_min / n_experiments."""
        data = load_retro_artifact(RETRO_RESULT)
        computed = data["total_wall_time_min"] / data["n_experiments"]
        assert abs(data["mean_time_per_exp_min"] - computed) < 0.5, (
            f"mean_time_per_exp_min {data['mean_time_per_exp_min']:.2f} "
            f"!= total/n = {computed:.2f}"
        )

    def test_total_wall_time_positive(self) -> None:
        """SCENARIO-RETRO-006: total_wall_time_min must be positive."""
        data = load_retro_artifact(RETRO_RESULT)
        assert data["total_wall_time_min"] > 0


# ---------------------------------------------------------------------------
# REQ-RETRO-003: bottlenecks_identified field
# ---------------------------------------------------------------------------


class TestBottlenecksIdentified:
    """REQ-RETRO-003: bottlenecks_identified must be a non-empty list of dicts."""

    REQUIRED_KEYS = {"name", "duration_min", "pct_total"}

    def test_bottlenecks_is_non_empty_list(self) -> None:
        """SCENARIO-RETRO-005: must have at least 1 bottleneck entry."""
        data = load_retro_artifact(RETRO_RESULT)
        assert isinstance(data["bottlenecks_identified"], list)
        assert len(data["bottlenecks_identified"]) >= 1

    def test_each_bottleneck_has_required_keys(self) -> None:
        """SCENARIO-RETRO-005: every bottleneck entry must have name, duration_min, pct_total."""
        data = load_retro_artifact(RETRO_RESULT)
        for i, b in enumerate(data["bottlenecks_identified"]):
            missing = self.REQUIRED_KEYS - set(b.keys())
            assert not missing, f"Bottleneck entry {i} missing keys: {missing}"

    def test_name_is_non_empty_string(self) -> None:
        """SCENARIO-RETRO-005: name must be a non-empty string."""
        data = load_retro_artifact(RETRO_RESULT)
        for b in data["bottlenecks_identified"]:
            assert isinstance(b["name"], str) and len(b["name"]) > 0

    def test_duration_min_is_positive(self) -> None:
        """SCENARIO-RETRO-005: duration_min must be a positive number."""
        data = load_retro_artifact(RETRO_RESULT)
        for b in data["bottlenecks_identified"]:
            assert isinstance(b["duration_min"], (int, float)) and b["duration_min"] > 0

    def test_pct_total_in_valid_range(self) -> None:
        """SCENARIO-RETRO-005: pct_total must be in (0, 100]."""
        data = load_retro_artifact(RETRO_RESULT)
        for b in data["bottlenecks_identified"]:
            v = b["pct_total"]
            assert isinstance(v, (int, float)) and 0.0 < v <= 100.0

    def test_at_least_three_bottlenecks(self) -> None:
        """SCENARIO-RETRO-005: retro should report top 3 slowest experiments."""
        data = load_retro_artifact(RETRO_RESULT)
        assert len(data["bottlenecks_identified"]) >= 3

    def test_first_bottleneck_is_longest(self) -> None:
        """SCENARIO-RETRO-005: first entry must have the largest duration_min."""
        data = load_retro_artifact(RETRO_RESULT)
        bottlenecks = data["bottlenecks_identified"]
        if len(bottlenecks) < 2:
            return
        first_dur = bottlenecks[0]["duration_min"]
        for b in bottlenecks[1:]:
            assert first_dur >= b["duration_min"], (
                f"First bottleneck ({first_dur} min) shorter than later one ({b['duration_min']} min)"
            )


# ---------------------------------------------------------------------------
# REQ-RETRO-003: action_items field
# Action items format: {id, description, estimated_impact_pct, ...}
# ---------------------------------------------------------------------------


class TestActionItems:
    """REQ-RETRO-003: action_items must have required structure."""

    REQUIRED_KEYS = {"id", "description", "estimated_impact_pct"}
    VALID_STATUSES = {"carried_forward", "resolved", "new"}

    def test_action_items_is_non_empty_list(self) -> None:
        """SCENARIO-RETRO-005: action_items must contain at least 1 entry."""
        data = load_retro_artifact(RETRO_RESULT)
        assert isinstance(data["action_items"], list)
        assert len(data["action_items"]) >= 1

    def test_each_item_has_required_keys(self) -> None:
        """SCENARIO-RETRO-005: every action item must have id, description, estimated_impact_pct."""
        data = load_retro_artifact(RETRO_RESULT)
        for i, item in enumerate(data["action_items"]):
            missing = self.REQUIRED_KEYS - set(item.keys())
            assert not missing, f"Action item {i} missing keys: {missing}"

    def test_all_ids_are_strings(self) -> None:
        """SCENARIO-RETRO-005: id must be a non-empty string."""
        data = load_retro_artifact(RETRO_RESULT)
        for item in data["action_items"]:
            assert isinstance(item["id"], str) and len(item["id"]) > 0

    def test_all_descriptions_are_strings(self) -> None:
        """SCENARIO-RETRO-005: description must be a non-empty string."""
        data = load_retro_artifact(RETRO_RESULT)
        for item in data["action_items"]:
            assert isinstance(item["description"], str) and len(item["description"]) > 0

    def test_estimated_impact_pct_is_numeric(self) -> None:
        """SCENARIO-RETRO-005: estimated_impact_pct must be a number in [0, 100]."""
        data = load_retro_artifact(RETRO_RESULT)
        for item in data["action_items"]:
            v = item["estimated_impact_pct"]
            assert isinstance(v, (int, float)), (
                f"Item {item['id']!r}: estimated_impact_pct is {type(v)!r}"
            )
            assert 0.0 <= v <= 100.0, (
                f"Item {item['id']!r}: estimated_impact_pct={v} out of [0, 100]"
            )

    def test_new_items_present(self) -> None:
        """SCENARIO-RETRO-005: at least one NEW-* action item from this milestone."""
        data = load_retro_artifact(RETRO_RESULT)
        new_items = [i for i in data["action_items"] if i["id"].startswith("NEW-")]
        assert len(new_items) >= 1, "No NEW-* action items found for this milestone"


# ---------------------------------------------------------------------------
# REQ-RETRO-003: carry_over field
# carry_over is a list of {id, description, resolved: bool} from 2026.04.23 retro
# ---------------------------------------------------------------------------


class TestCarryOver:
    """REQ-RETRO-003: carry_over must document 2026.04.23 retro items with resolved flags."""

    REQUIRED_KEYS = {"id", "description", "resolved"}

    def test_carry_over_is_list(self) -> None:
        """SCENARIO-RETRO-005: carry_over must be a list."""
        data = load_retro_artifact(RETRO_RESULT)
        assert isinstance(data["carry_over"], list)

    def test_carry_over_non_empty(self) -> None:
        """SCENARIO-RETRO-005: carry_over must reference at least RETRO-001 and RETRO-002."""
        data = load_retro_artifact(RETRO_RESULT)
        assert len(data["carry_over"]) >= 2

    def test_each_entry_has_required_keys(self) -> None:
        """SCENARIO-RETRO-005: every carry_over entry must have id, description, resolved."""
        data = load_retro_artifact(RETRO_RESULT)
        for i, entry in enumerate(data["carry_over"]):
            missing = self.REQUIRED_KEYS - set(entry.keys())
            assert not missing, f"carry_over entry {i} missing keys: {missing}"

    def test_resolved_field_is_bool(self) -> None:
        """SCENARIO-RETRO-005: resolved must be a boolean for every carry_over entry."""
        data = load_retro_artifact(RETRO_RESULT)
        for entry in data["carry_over"]:
            assert isinstance(entry["resolved"], bool), (
                f"carry_over entry {entry['id']!r}: resolved is {type(entry['resolved'])!r}"
            )

    def test_retro_001_in_carry_over(self) -> None:
        """SCENARIO-RETRO-005: RETRO-001 must appear in carry_over."""
        data = load_retro_artifact(RETRO_RESULT)
        ids = {e["id"] for e in data["carry_over"]}
        assert "RETRO-001" in ids, "RETRO-001 not found in carry_over"

    def test_retro_002_in_carry_over(self) -> None:
        """SCENARIO-RETRO-005: RETRO-002 must appear in carry_over."""
        data = load_retro_artifact(RETRO_RESULT)
        ids = {e["id"] for e in data["carry_over"]}
        assert "RETRO-002" in ids, "RETRO-002 not found in carry_over"

    def test_retro_001_resolved_true(self) -> None:
        """SCENARIO-RETRO-005: RETRO-001 was implemented in Exp 325, so resolved=True."""
        data = load_retro_artifact(RETRO_RESULT)
        entry = next(e for e in data["carry_over"] if e["id"] == "RETRO-001")
        assert entry["resolved"] is True, (
            "RETRO-001 was implemented in Exp 325 but carry_over marks it unresolved"
        )

    def test_retro_002_resolved_true(self) -> None:
        """SCENARIO-RETRO-005: RETRO-002 was implemented in Exp 326, so resolved=True."""
        data = load_retro_artifact(RETRO_RESULT)
        entry = next(e for e in data["carry_over"] if e["id"] == "RETRO-002")
        assert entry["resolved"] is True, (
            "RETRO-002 was implemented in Exp 326 but carry_over marks it unresolved"
        )


# ---------------------------------------------------------------------------
# REQ-RETRO-003: retro_001_resolved and retro_002_resolved fields
# SCENARIO-RETRO-005
# ---------------------------------------------------------------------------


class TestRetroResolvedFlags:
    """REQ-RETRO-003: retro_001_resolved and retro_002_resolved must be booleans."""

    def test_retro_001_resolved_is_bool(self) -> None:
        """SCENARIO-RETRO-005: retro_001_resolved must be a boolean."""
        data = load_retro_artifact(RETRO_RESULT)
        assert isinstance(data["retro_001_resolved"], bool)

    def test_retro_002_resolved_is_bool(self) -> None:
        """SCENARIO-RETRO-005: retro_002_resolved must be a boolean."""
        data = load_retro_artifact(RETRO_RESULT)
        assert isinstance(data["retro_002_resolved"], bool)

    def test_retro_001_resolved_is_true(self) -> None:
        """SCENARIO-RETRO-005: RETRO-001 timeout wrapper was shipped in Exp 325."""
        data = load_retro_artifact(RETRO_RESULT)
        assert data["retro_001_resolved"] is True, (
            "retro_001_resolved should be True: run_experiment_with_timeout.sh was "
            "implemented in Exp 325 per conductor log 2026-04-15 01:51 UTC"
        )

    def test_retro_002_resolved_is_true(self) -> None:
        """SCENARIO-RETRO-005: RETRO-002 DualGPUMonitor was shipped in Exp 326."""
        data = load_retro_artifact(RETRO_RESULT)
        assert data["retro_002_resolved"] is True, (
            "retro_002_resolved should be True: DualGPUMonitor was implemented "
            "in Exp 326 per conductor log 2026-04-15 02:04 UTC"
        )


# ---------------------------------------------------------------------------
# REQ-RETRO-003: actual_speedup_pct field
# SCENARIO-RETRO-006
# ---------------------------------------------------------------------------


class TestActualSpeedupPct:
    """REQ-RETRO-003: actual_speedup_pct must measure improvement vs prior 40.6 min/exp."""

    def test_actual_speedup_is_float(self) -> None:
        """SCENARIO-RETRO-006: actual_speedup_pct must be numeric."""
        data = load_retro_artifact(RETRO_RESULT)
        assert isinstance(data["actual_speedup_pct"], (int, float))

    def test_actual_speedup_consistent_with_mean(self) -> None:
        """SCENARIO-RETRO-006: actual_speedup_pct = (40.6 - mean) / 40.6 * 100."""
        data = load_retro_artifact(RETRO_RESULT)
        mean = data["mean_time_per_exp_min"]
        expected = round((PRIOR_MILESTONE_MEAN_MIN - mean) / PRIOR_MILESTONE_MEAN_MIN * 100, 1)
        actual = round(data["actual_speedup_pct"], 1)
        assert abs(actual - expected) < 1.0, (
            f"actual_speedup_pct={actual} inconsistent with "
            f"mean {mean:.1f} min/exp (expected ~{expected})"
        )

    def test_actual_speedup_positive(self) -> None:
        """SCENARIO-RETRO-006: this milestone ran faster than the prior one (positive speedup).

        With all four action items (RETRO-001/002, NEW-001/002) resolved in Exps 325-327,
        the milestone mean should be well below the prior 40.6 min/exp baseline.
        """
        data = load_retro_artifact(RETRO_RESULT)
        assert data["actual_speedup_pct"] > 0.0, (
            f"actual_speedup_pct={data['actual_speedup_pct']} is not positive; "
            "expected improvement after resolving all RETRO items"
        )

    def test_actual_speedup_exceeds_estimate(self) -> None:
        """SCENARIO-RETRO-006: actual speedup should exceed the 27% estimate from Exp 319.

        The Exp 319 retro estimated 27% improvement if all four items were implemented.
        All four were implemented this milestone, so actual should be >= 27%.
        """
        data = load_retro_artifact(RETRO_RESULT)
        assert data["actual_speedup_pct"] >= 27.0, (
            f"actual_speedup_pct={data['actual_speedup_pct']} is below the 27% "
            "estimate from Exp 319, even though all RETRO items were implemented"
        )


# ---------------------------------------------------------------------------
# REQ-RETRO-003: estimated_next_milestone_speedup_pct field
# Unlike Exp 319, this field is allowed to be 0 or even negative.
# ---------------------------------------------------------------------------


class TestEstimatedNextSpeedup:
    """REQ-RETRO-003: estimated_next_milestone_speedup_pct is honest (can be 0 or negative)."""

    def test_estimated_speedup_is_numeric(self) -> None:
        """SCENARIO-RETRO-005: must be a numeric type."""
        data = load_retro_artifact(RETRO_RESULT)
        assert isinstance(data["estimated_next_milestone_speedup_pct"], (int, float))

    def test_estimated_speedup_in_bounded_range(self) -> None:
        """SCENARIO-RETRO-005: must be in [-100, 100] (negative allowed; speedup bounded)."""
        data = load_retro_artifact(RETRO_RESULT)
        val = data["estimated_next_milestone_speedup_pct"]
        assert -100.0 <= val <= 100.0, (
            f"estimated_next_milestone_speedup_pct={val} out of [-100, 100]"
        )


# ---------------------------------------------------------------------------
# Script structure validation
# ---------------------------------------------------------------------------


class TestScriptExists:
    """The retro script must be present and have the required structure."""

    def test_script_file_exists(self) -> None:
        """Retro script must exist at scripts/experiment_337_retro.py."""
        assert RETRO_SCRIPT.exists(), f"Missing script: {RETRO_SCRIPT}"

    def test_script_defines_main(self) -> None:
        """Script must define a main() entry point."""
        source = RETRO_SCRIPT.read_text()
        assert "def main(" in source or '__name__ == "__main__"' in source

    def test_script_references_action_items(self) -> None:
        """Script must reference action_items in its output."""
        source = RETRO_SCRIPT.read_text()
        assert "action_items" in source

    def test_script_references_n_experiments(self) -> None:
        """Script must reference n_experiments."""
        source = RETRO_SCRIPT.read_text()
        assert "n_experiments" in source

    def test_script_references_bottlenecks(self) -> None:
        """Script must reference bottlenecks_identified."""
        source = RETRO_SCRIPT.read_text()
        assert "bottlenecks_identified" in source

    def test_script_references_retro_001_resolved(self) -> None:
        """Script must compute retro_001_resolved."""
        source = RETRO_SCRIPT.read_text()
        assert "retro_001_resolved" in source

    def test_script_references_retro_002_resolved(self) -> None:
        """Script must compute retro_002_resolved."""
        source = RETRO_SCRIPT.read_text()
        assert "retro_002_resolved" in source

    def test_script_references_actual_speedup(self) -> None:
        """Script must compute actual_speedup_pct."""
        source = RETRO_SCRIPT.read_text()
        assert "actual_speedup_pct" in source

    def test_script_references_carry_over(self) -> None:
        """Script must build carry_over list."""
        source = RETRO_SCRIPT.read_text()
        assert "carry_over" in source

    def test_script_references_experiment_337(self) -> None:
        """Script must reference experiment 337."""
        source = RETRO_SCRIPT.read_text()
        assert "337" in source
