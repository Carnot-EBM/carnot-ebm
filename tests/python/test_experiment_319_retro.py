"""Tests for Exp 319: operational retrospective for milestone 2026.04.23.

Spec coverage: REQ-OPS-001, REQ-OPS-002, REQ-OPS-003, REQ-OPS-004, REQ-OPS-005,
               SCENARIO-OPS-001, SCENARIO-OPS-002, SCENARIO-OPS-003,
               SCENARIO-OPS-004, SCENARIO-OPS-005, SCENARIO-OPS-006,
               SCENARIO-OPS-007
"""

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RETRO_SCRIPT = SCRIPTS_DIR / "experiment_319_retro.py"
RETRO_RESULT = RESULTS_DIR / "operational_retro_2026_04_23.json"


# ---------------------------------------------------------------------------
# Shared loader (load_retro_artifact)
#
# This function validates that the file exists and has the correct JSON schema.
# It is the public API documented in the task spec.
# ---------------------------------------------------------------------------


def load_retro_artifact(path: Path) -> dict[str, Any]:
    """Load and schema-validate the 2026.04.23 retro artifact.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the file is not valid JSON or is missing required keys.

    Why a separate validator: the task spec requires this function to be
    explicitly testable as an entry point (not just inline in tests).
    """
    if not path.exists():
        raise FileNotFoundError(f"Retro artifact not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in retro artifact: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Retro artifact must be a JSON object")
    _validate_top_level_schema(data)
    return data


def _validate_top_level_schema(data: dict[str, Any]) -> None:
    """Check all required top-level keys are present.

    Why: centralises schema validation so test classes and the loader
    both enforce the same contract.
    """
    required = {
        "schema",
        "milestone",
        "generated_at",
        "n_experiments",
        "total_wall_time_minutes",
        "avg_minutes_per_experiment",
        "bottlenecks_identified",
        "improvements_implemented",
        "action_items",
        "carry_over_from_previous_retro",
        "estimated_next_milestone_speedup_pct",
    }
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Retro artifact missing required keys: {missing}")


# ---------------------------------------------------------------------------
# REQ-OPS-001: Artifact file and top-level schema
# SCENARIO-OPS-001: JSON is parseable; all required fields present and typed.
# ---------------------------------------------------------------------------


class TestRetroArtifactSchema:
    """REQ-OPS-001: retro artifact must conform to the canonical schema."""

    REQUIRED_FIELDS: dict[str, Any] = {
        "schema": str,
        "milestone": str,
        "generated_at": str,
        "n_experiments": int,
        "total_wall_time_minutes": (int, float),
        "avg_minutes_per_experiment": (int, float),
        "bottlenecks_identified": list,
        "improvements_implemented": list,
        "action_items": list,
        "carry_over_from_previous_retro": dict,
        "estimated_next_milestone_speedup_pct": (int, float),
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
        """SCENARIO-OPS-001: all required fields have the correct types."""
        data = json.loads(RETRO_RESULT.read_text())
        for field, expected_type in self.REQUIRED_FIELDS.items():
            assert isinstance(data[field], expected_type), (
                f"Field '{field}' should be {expected_type}, "
                f"got {type(data[field])}"
            )

    def test_milestone_value(self) -> None:
        """SCENARIO-OPS-001: milestone must identify 2026.04.23."""
        data = json.loads(RETRO_RESULT.read_text())
        assert data["milestone"] == "2026.04.23"

    def test_generated_at_is_iso8601(self) -> None:
        """SCENARIO-OPS-001: generated_at must be an ISO-8601 timestamp."""
        from datetime import datetime

        data = json.loads(RETRO_RESULT.read_text())
        ts = data["generated_at"]
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"generated_at is not ISO-8601: {ts!r}")

    def test_schema_field_is_string(self) -> None:
        """SCENARIO-OPS-001: schema version field must be a non-empty string."""
        data = json.loads(RETRO_RESULT.read_text())
        assert isinstance(data["schema"], str) and len(data["schema"]) > 0


# ---------------------------------------------------------------------------
# load_retro_artifact function tests
# SCENARIO-OPS-001: the public loader validates schema and raises on errors.
# ---------------------------------------------------------------------------


class TestLoadRetroArtifact:
    """REQ-OPS-001: load_retro_artifact must validate schema and raise correctly."""

    def test_load_returns_dict(self) -> None:
        """SCENARIO-OPS-001: loader returns a dict for a valid artifact."""
        data = load_retro_artifact(RETRO_RESULT)
        assert isinstance(data, dict)

    def test_load_raises_file_not_found(self, tmp_path: Path) -> None:
        """SCENARIO-OPS-001: raises FileNotFoundError for a nonexistent path."""
        with pytest.raises(FileNotFoundError):
            load_retro_artifact(tmp_path / "nonexistent.json")

    def test_load_raises_value_error_for_invalid_json(self, tmp_path: Path) -> None:
        """SCENARIO-OPS-001: raises ValueError for malformed JSON."""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_retro_artifact(bad)

    def test_load_raises_value_error_for_missing_keys(self, tmp_path: Path) -> None:
        """SCENARIO-OPS-001: raises ValueError when required keys are absent."""
        sparse = tmp_path / "sparse.json"
        sparse.write_text(json.dumps({"milestone": "2026.04.23"}))
        with pytest.raises(ValueError, match="missing required keys"):
            load_retro_artifact(sparse)

    def test_load_raises_value_error_for_non_dict(self, tmp_path: Path) -> None:
        """SCENARIO-OPS-001: raises ValueError when top level is not a dict."""
        arr = tmp_path / "array.json"
        arr.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="JSON object"):
            load_retro_artifact(arr)


# ---------------------------------------------------------------------------
# REQ-OPS-002: n_experiments field
# SCENARIO-OPS-002: count of experiments in this milestone is a positive int.
# ---------------------------------------------------------------------------


class TestNExperimentsField:
    """REQ-OPS-002: n_experiments must reflect the actual experiment count."""

    def test_n_experiments_is_positive_int(self) -> None:
        """SCENARIO-OPS-002: n_experiments must be a positive integer."""
        data = load_retro_artifact(RETRO_RESULT)
        n = data["n_experiments"]
        assert isinstance(n, int) and n > 0, f"n_experiments={n!r} not a positive int"

    def test_n_experiments_reasonable_range(self) -> None:
        """SCENARIO-OPS-002: milestone 2026.04.23 had 13–20 experiments."""
        data = load_retro_artifact(RETRO_RESULT)
        n = data["n_experiments"]
        assert 13 <= n <= 20, (
            f"n_experiments={n} outside expected range [13, 20] for this milestone"
        )

    def test_avg_minutes_consistent_with_total(self) -> None:
        """SCENARIO-OPS-002: avg = total_wall_time_minutes / n_experiments."""
        data = load_retro_artifact(RETRO_RESULT)
        computed_avg = data["total_wall_time_minutes"] / data["n_experiments"]
        assert abs(data["avg_minutes_per_experiment"] - computed_avg) < 0.5, (
            f"avg_minutes_per_experiment {data['avg_minutes_per_experiment']:.2f} "
            f"!= total/n = {computed_avg:.2f}"
        )

    def test_total_wall_time_positive(self) -> None:
        """SCENARIO-OPS-002: total_wall_time_minutes must be positive."""
        data = load_retro_artifact(RETRO_RESULT)
        assert data["total_wall_time_minutes"] > 0


# ---------------------------------------------------------------------------
# REQ-OPS-003: bottlenecks_identified field
# SCENARIO-OPS-003: list of bottleneck dicts with name, duration_min, pct_total.
# ---------------------------------------------------------------------------


class TestBottlenecksIdentified:
    """REQ-OPS-003: bottlenecks_identified must be a list of structured dicts."""

    BOTTLENECK_REQUIRED = {"name", "duration_min", "pct_total"}

    def _load(self) -> dict[str, Any]:
        return load_retro_artifact(RETRO_RESULT)

    def test_bottlenecks_is_non_empty_list(self) -> None:
        """SCENARIO-OPS-003: bottlenecks_identified must have at least 1 entry."""
        data = self._load()
        assert isinstance(data["bottlenecks_identified"], list)
        assert len(data["bottlenecks_identified"]) >= 1

    def test_each_bottleneck_has_required_keys(self) -> None:
        """SCENARIO-OPS-003: every bottleneck entry must have name, duration_min, pct_total."""
        data = self._load()
        for i, b in enumerate(data["bottlenecks_identified"]):
            missing = self.BOTTLENECK_REQUIRED - set(b.keys())
            assert not missing, f"Bottleneck entry {i} missing keys: {missing}"

    def test_name_is_non_empty_string(self) -> None:
        """SCENARIO-OPS-003: name must be a non-empty string."""
        data = self._load()
        for b in data["bottlenecks_identified"]:
            assert isinstance(b["name"], str) and len(b["name"]) > 0

    def test_duration_min_is_positive_number(self) -> None:
        """SCENARIO-OPS-003: duration_min must be a positive number."""
        data = self._load()
        for b in data["bottlenecks_identified"]:
            assert isinstance(b["duration_min"], (int, float)) and b["duration_min"] > 0

    def test_pct_total_in_valid_range(self) -> None:
        """SCENARIO-OPS-003: pct_total must be between 0 and 100 inclusive."""
        data = self._load()
        for b in data["bottlenecks_identified"]:
            assert isinstance(b["pct_total"], (int, float))
            assert 0.0 < b["pct_total"] <= 100.0, (
                f"pct_total={b['pct_total']} out of valid range (0, 100]"
            )

    def test_top_bottleneck_is_longest(self) -> None:
        """SCENARIO-OPS-003: first bottleneck should have the largest duration_min."""
        data = self._load()
        bottlenecks = data["bottlenecks_identified"]
        if len(bottlenecks) < 2:
            return
        first_dur = bottlenecks[0]["duration_min"]
        for b in bottlenecks[1:]:
            assert first_dur >= b["duration_min"], (
                f"First bottleneck ({first_dur} min) is shorter than "
                f"a later one ({b['duration_min']} min)"
            )

    def test_at_least_three_bottlenecks(self) -> None:
        """SCENARIO-OPS-003: milestone should report top 3 slowest experiments."""
        data = self._load()
        assert len(data["bottlenecks_identified"]) >= 3


# ---------------------------------------------------------------------------
# REQ-OPS-004: action_items field
# SCENARIO-OPS-004: list with id, description, status, owner; RETRO-001 and
#   RETRO-002 present with status carried_forward or resolved.
# ---------------------------------------------------------------------------


class TestActionItems:
    """REQ-OPS-004: action_items must have required structure and contain RETRO items."""

    ITEM_REQUIRED_KEYS = {"id", "description", "status", "owner"}
    VALID_STATUSES = {"carried_forward", "resolved", "new"}

    def _load(self) -> dict[str, Any]:
        return load_retro_artifact(RETRO_RESULT)

    def test_action_items_is_non_empty_list(self) -> None:
        """SCENARIO-OPS-004: action_items must contain at least 1 entry."""
        data = self._load()
        assert isinstance(data["action_items"], list)
        assert len(data["action_items"]) >= 1

    def test_each_item_has_required_keys(self) -> None:
        """SCENARIO-OPS-004: every action item must have id, description, status, owner."""
        data = self._load()
        for i, item in enumerate(data["action_items"]):
            missing = self.ITEM_REQUIRED_KEYS - set(item.keys())
            assert not missing, f"Action item {i} missing keys: {missing}"

    def test_each_status_is_valid(self) -> None:
        """SCENARIO-OPS-004: status must be carried_forward, resolved, or new."""
        data = self._load()
        for item in data["action_items"]:
            assert item["status"] in self.VALID_STATUSES, (
                f"Item {item['id']!r} has invalid status {item['status']!r}"
            )

    def test_retro_001_present(self) -> None:
        """SCENARIO-OPS-004: RETRO-001 (45-min timeout) must appear in action_items."""
        data = self._load()
        ids = {item["id"] for item in data["action_items"]}
        assert "RETRO-001" in ids, "RETRO-001 not found in action_items"

    def test_retro_002_present(self) -> None:
        """SCENARIO-OPS-004: RETRO-002 (GPU monitor) must appear in action_items."""
        data = self._load()
        ids = {item["id"] for item in data["action_items"]}
        assert "RETRO-002" in ids, "RETRO-002 not found in action_items"

    def test_retro_001_status(self) -> None:
        """SCENARIO-OPS-004: RETRO-001 must be carried_forward or resolved."""
        data = self._load()
        item = next(i for i in data["action_items"] if i["id"] == "RETRO-001")
        assert item["status"] in ("carried_forward", "resolved"), (
            f"RETRO-001 has unexpected status {item['status']!r}"
        )

    def test_retro_002_status(self) -> None:
        """SCENARIO-OPS-004: RETRO-002 must be carried_forward or resolved."""
        data = self._load()
        item = next(i for i in data["action_items"] if i["id"] == "RETRO-002")
        assert item["status"] in ("carried_forward", "resolved"), (
            f"RETRO-002 has unexpected status {item['status']!r}"
        )

    def test_all_ids_are_strings(self) -> None:
        """SCENARIO-OPS-004: id must be a non-empty string."""
        data = self._load()
        for item in data["action_items"]:
            assert isinstance(item["id"], str) and len(item["id"]) > 0

    def test_all_descriptions_are_strings(self) -> None:
        """SCENARIO-OPS-004: description must be a non-empty string."""
        data = self._load()
        for item in data["action_items"]:
            assert isinstance(item["description"], str) and len(item["description"]) > 0

    def test_all_owners_are_strings(self) -> None:
        """SCENARIO-OPS-004: owner must be a non-empty string."""
        data = self._load()
        for item in data["action_items"]:
            assert isinstance(item["owner"], str) and len(item["owner"]) > 0

    def test_new_item_present_for_this_milestone(self) -> None:
        """SCENARIO-OPS-004: at least one NEW-* action item from this milestone."""
        data = self._load()
        new_items = [i for i in data["action_items"] if i["id"].startswith("NEW-")]
        assert len(new_items) >= 1, "No NEW-* action items found for this milestone"


# ---------------------------------------------------------------------------
# REQ-OPS-005: improvements_implemented field
# SCENARIO-OPS-005: improvements implemented THIS milestone (not prior ones).
# ---------------------------------------------------------------------------


class TestImprovementsImplemented:
    """REQ-OPS-005: improvements_implemented must list this-milestone changes only."""

    IMPROVEMENT_REQUIRED_KEYS = {"name", "experiment", "description"}

    def _load(self) -> dict[str, Any]:
        return load_retro_artifact(RETRO_RESULT)

    def test_improvements_is_non_empty_list(self) -> None:
        """SCENARIO-OPS-005: improvements_implemented must have at least 1 entry."""
        data = self._load()
        assert isinstance(data["improvements_implemented"], list)
        assert len(data["improvements_implemented"]) >= 1

    def test_each_improvement_has_required_keys(self) -> None:
        """SCENARIO-OPS-005: every improvement must have name, experiment, description."""
        data = self._load()
        for i, imp in enumerate(data["improvements_implemented"]):
            missing = self.IMPROVEMENT_REQUIRED_KEYS - set(imp.keys())
            assert not missing, f"Improvement entry {i} missing keys: {missing}"

    def test_each_name_is_non_empty_string(self) -> None:
        """SCENARIO-OPS-005: name must be a non-empty string."""
        data = self._load()
        for imp in data["improvements_implemented"]:
            assert isinstance(imp["name"], str) and len(imp["name"]) > 0

    def test_each_description_is_non_empty_string(self) -> None:
        """SCENARIO-OPS-005: description must be a non-empty string."""
        data = self._load()
        for imp in data["improvements_implemented"]:
            assert isinstance(imp["description"], str) and len(imp["description"]) > 0

    def test_experiment_field_references_milestone_range(self) -> None:
        """SCENARIO-OPS-005: experiment numbers should be in milestone range (307-324).

        Allows 306 since ExperimentTemplate was the final experiment in the prior
        milestone and its benefit is counted as infrastructure used throughout this one.
        """
        data = self._load()
        for imp in data["improvements_implemented"]:
            exp_raw = imp.get("experiment")
            if exp_raw is None:
                continue
            # Allow ints, strings like "307", "Exp 307", or "N/A"
            if isinstance(exp_raw, str) and exp_raw.lower() in ("n/a", "prior"):
                continue
            if isinstance(exp_raw, (int, float)):
                exp_num = int(exp_raw)
            else:
                import re
                m = re.search(r"\d+", str(exp_raw))
                if not m:
                    continue
                exp_num = int(m.group())
            assert 306 <= exp_num <= 330, (
                f"experiment={exp_raw!r} references number {exp_num} "
                "outside expected milestone range [306, 330]"
            )

    def test_z3_gated_improvement_present(self) -> None:
        """SCENARIO-OPS-005: Z3-gated repair (Exp 312) must be in improvements."""
        data = self._load()
        names_lower = [imp["name"].lower() for imp in data["improvements_implemented"]]
        assert any("z3" in n for n in names_lower), (
            "Z3-gated repair improvement not found in improvements_implemented"
        )

    def test_jepa_gate_improvement_present(self) -> None:
        """SCENARIO-OPS-005: JEPA fast-path gate (Exp 308) must be in improvements."""
        data = self._load()
        names_lower = [imp["name"].lower() for imp in data["improvements_implemented"]]
        assert any("jepa" in n for n in names_lower), (
            "JEPA fast-path gate improvement not found in improvements_implemented"
        )


# ---------------------------------------------------------------------------
# REQ-OPS-001: estimated_next_milestone_speedup_pct field
# SCENARIO-OPS-006: speedup estimate is a float in [0, 100].
# ---------------------------------------------------------------------------


class TestEstimatedSpeedup:
    """REQ-OPS-001: estimated_next_milestone_speedup_pct must be a valid percentage."""

    def _load(self) -> dict[str, Any]:
        return load_retro_artifact(RETRO_RESULT)

    def test_speedup_is_float_or_int(self) -> None:
        """SCENARIO-OPS-006: speedup estimate must be numeric."""
        data = self._load()
        val = data["estimated_next_milestone_speedup_pct"]
        assert isinstance(val, (int, float)), f"speedup is {type(val)!r}, not numeric"

    def test_speedup_in_valid_range(self) -> None:
        """SCENARIO-OPS-006: speedup must be in [0, 100]."""
        data = self._load()
        val = data["estimated_next_milestone_speedup_pct"]
        assert 0.0 <= val <= 100.0, (
            f"estimated_next_milestone_speedup_pct={val} out of [0, 100]"
        )

    def test_speedup_nonzero(self) -> None:
        """SCENARIO-OPS-006: speedup must be > 0 (action items have nonzero impact)."""
        data = self._load()
        val = data["estimated_next_milestone_speedup_pct"]
        assert val > 0.0, "estimated_next_milestone_speedup_pct should be > 0"


# ---------------------------------------------------------------------------
# REQ-OPS-003: carry_over_from_previous_retro field
# SCENARIO-OPS-007: prior retro and carried items are documented.
# ---------------------------------------------------------------------------


class TestCarryOverFromPreviousRetro:
    """REQ-OPS-003: carry_over_from_previous_retro must document prior retro state."""

    CARRY_REQUIRED_KEYS = {"prior_milestone", "items_carried_forward", "items_resolved"}

    def _load(self) -> dict[str, Any]:
        return load_retro_artifact(RETRO_RESULT)

    def test_carry_over_has_required_keys(self) -> None:
        """SCENARIO-OPS-007: carry_over_from_previous_retro must have all required keys."""
        data = self._load()
        carry = data["carry_over_from_previous_retro"]
        missing = self.CARRY_REQUIRED_KEYS - set(carry.keys())
        assert not missing, f"carry_over_from_previous_retro missing: {missing}"

    def test_prior_milestone_is_2026_04_22(self) -> None:
        """SCENARIO-OPS-007: prior milestone must be 2026.04.22."""
        data = self._load()
        carry = data["carry_over_from_previous_retro"]
        assert carry["prior_milestone"] == "2026.04.22", (
            f"prior_milestone={carry['prior_milestone']!r}, expected '2026.04.22'"
        )

    def test_items_carried_forward_is_list(self) -> None:
        """SCENARIO-OPS-007: items_carried_forward must be a list."""
        data = self._load()
        assert isinstance(data["carry_over_from_previous_retro"]["items_carried_forward"], list)

    def test_items_resolved_is_list(self) -> None:
        """SCENARIO-OPS-007: items_resolved must be a list."""
        data = self._load()
        assert isinstance(data["carry_over_from_previous_retro"]["items_resolved"], list)

    def test_retro_001_appears_in_carried_or_resolved(self) -> None:
        """SCENARIO-OPS-007: RETRO-001 must appear in items_carried_forward or items_resolved."""
        data = self._load()
        carry = data["carry_over_from_previous_retro"]
        all_mentioned = (
            str(carry["items_carried_forward"]) + str(carry["items_resolved"])
        )
        assert "RETRO-001" in all_mentioned, (
            "RETRO-001 not mentioned in carry_over_from_previous_retro"
        )

    def test_retro_002_appears_in_carried_or_resolved(self) -> None:
        """SCENARIO-OPS-007: RETRO-002 must appear in items_carried_forward or items_resolved."""
        data = self._load()
        carry = data["carry_over_from_previous_retro"]
        all_mentioned = (
            str(carry["items_carried_forward"]) + str(carry["items_resolved"])
        )
        assert "RETRO-002" in all_mentioned, (
            "RETRO-002 not mentioned in carry_over_from_previous_retro"
        )


# ---------------------------------------------------------------------------
# Script existence and structural validation
# ---------------------------------------------------------------------------


class TestScriptExists:
    """Verify the retro script is present and has the required structure."""

    def test_script_file_exists(self) -> None:
        """Retro script must exist at scripts/experiment_319_retro.py."""
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

    def test_script_references_improvements(self) -> None:
        """Script must reference improvements_implemented."""
        source = RETRO_SCRIPT.read_text()
        assert "improvements_implemented" in source

    def test_script_references_estimated_speedup(self) -> None:
        """Script must compute estimated_next_milestone_speedup_pct."""
        source = RETRO_SCRIPT.read_text()
        assert "estimated_next_milestone_speedup_pct" in source

    def test_script_references_retro_001(self) -> None:
        """Script must handle RETRO-001 action item."""
        source = RETRO_SCRIPT.read_text()
        assert "RETRO-001" in source

    def test_script_references_retro_002(self) -> None:
        """Script must handle RETRO-002 action item."""
        source = RETRO_SCRIPT.read_text()
        assert "RETRO-002" in source
