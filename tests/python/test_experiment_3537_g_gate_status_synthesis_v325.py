"""Tests for experiment_3537_g_gate_status_synthesis_v325.py.

REQ-GATE-001: Every capstone must emit g1..g4 booleans + unmet_gates.
REQ-GATE-002: honest_verdict must start with 'complete:'.
REQ-GATE-003: inference_substrate must be aggregation_from_upstream_artifacts.
REQ-GATE-004: Absent or flagged_adversarial artifacts must contribute null values.
REQ-GATE-005: depth_forcing_function_can_relax = p01_defensible AND G2-in-motion.
REQ-GATE-006: random_seed must be 20260531 (NOT the experiment number 3537).
REQ-GATE-007: gate_status_v325_ready must be True.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Load the module under test
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _PROJECT_ROOT / "scripts" / "experiment_3537_g_gate_status_synthesis_v325.py"
_PUBLICATION_GATE_STATE = _PROJECT_ROOT / "ops" / "publication_gate_state.json"
_spec = importlib.util.spec_from_file_location("exp3537", _SCRIPT)
exp3537 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(exp3537)  # type: ignore[union-attr]

_load_artifact = exp3537._load_artifact
build_synthesis = exp3537.build_synthesis
main = exp3537.main

# Required schema fields per CLAUDE.md "Principle-Annotated Artifact Fields"
REQUIRED_FIELDS = [
    "honest_verdict",
    "inference_substrate",
    "g1", "g2", "g3", "g4",
    "unmet_gates",
    "p01_route1_graph_coloring_verdict",
    "p01_route1_headroom_preserved",
    "p01_route1_beats_strong_baseline",
    "p01_sudoku_energy_power_visible",
    "p01_route2_corpus_had_headroom_exp3530",
    "p01_route2_fair_verdict",
    "p01_route2_corpus_had_headroom",
    "p01_route2_flip_count",
    "p01_route2_delta",
    "p01_has_clean_defensible_verdict",
    "aggregation_positive_promoted",
    "self_learning_deployed_rule",
    "g2_package_status",
    "depth_forcing_function_can_relax",
    "gate_status_v325_ready",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
]


# ---------------------------------------------------------------------------
# _load_artifact tests (REQ-GATE-004)
# ---------------------------------------------------------------------------

def test_load_artifact_absent_returns_none(tmp_path: Path) -> None:
    """REQ-GATE-004: missing artifact returns None without raising."""
    assert _load_artifact(tmp_path / "nonexistent.json") is None


def test_load_artifact_flagged_adversarial_returns_none(tmp_path: Path) -> None:
    """REQ-GATE-004: flagged_adversarial=True artifact is excluded (fabrication gate)."""
    p = tmp_path / "flagged.json"
    p.write_text(json.dumps({"flagged_adversarial": True, "honest_verdict": "complete: ok"}))
    assert _load_artifact(p) is None


def test_load_artifact_not_flagged_returns_dict(tmp_path: Path) -> None:
    """REQ-GATE-004: flagged_adversarial=False artifact is loaded normally."""
    payload = {"flagged_adversarial": False, "val": 99}
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(payload))
    result = _load_artifact(p)
    assert result is not None
    assert result["val"] == 99


def test_load_artifact_no_flag_field_returns_dict(tmp_path: Path) -> None:
    """REQ-GATE-004: artifact without flagged_adversarial field is treated as clean."""
    payload = {"honest_verdict": "complete: x", "n": 7}
    p = tmp_path / "clean.json"
    p.write_text(json.dumps(payload))
    assert _load_artifact(p) == payload


def test_load_artifact_malformed_json_returns_none(tmp_path: Path) -> None:
    """REQ-GATE-004: malformed JSON returns None gracefully (no exception)."""
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert _load_artifact(p) is None


# ---------------------------------------------------------------------------
# build_synthesis() structural tests (REQ-GATE-001..007)
# ---------------------------------------------------------------------------

def test_build_synthesis_all_required_fields() -> None:
    """REQ-GATE-001: build_synthesis() returns all required schema fields."""
    result = build_synthesis()
    for field in REQUIRED_FIELDS:
        assert field in result, f"Missing required field: {field!r}"


def test_honest_verdict_starts_with_complete() -> None:
    """REQ-GATE-002: honest_verdict must start with 'complete:' (terminal prefix)."""
    result = build_synthesis()
    assert result["honest_verdict"].startswith("complete:"), (
        f"honest_verdict={result['honest_verdict']!r} does not start with 'complete:'"
    )


def test_inference_substrate_is_aggregation() -> None:
    """REQ-GATE-003: inference_substrate must declare aggregation_from_upstream_artifacts."""
    result = build_synthesis()
    assert result["inference_substrate"] == "aggregation_from_upstream_artifacts"


def test_gate_status_v325_ready_is_true() -> None:
    """REQ-GATE-007: gate_status_v325_ready is always True (terminal completion flag)."""
    result = build_synthesis()
    assert result["gate_status_v325_ready"] is True


def test_random_seed_is_20260531() -> None:
    """REQ-GATE-006: random_seed must be 20260531, not the experiment number 3537.

    The exp3502 tautology fix: adversarial_verify flags random_seed==experiment_id
    as TAUTOLOGY.  The seed must be a distinct fixed value.
    """
    result = build_synthesis()
    assert result["random_seed"] == 20260531, (
        f"random_seed={result['random_seed']} — must be 20260531, not the experiment number"
    )
    assert result["random_seed"] != result["experiment"], (
        "random_seed must NOT equal experiment id (TAUTOLOGY fabrication gate)"
    )


def test_g_gates_are_booleans() -> None:
    """REQ-GATE-001: g1..g4 are Python booleans."""
    result = build_synthesis()
    for g in ("g1", "g2", "g3", "g4"):
        assert isinstance(result[g], bool), f"{g} is not a bool: {result[g]!r}"


def test_unmet_gates_is_list() -> None:
    """REQ-GATE-001: unmet_gates is a list (possibly empty when all gates pass)."""
    result = build_synthesis()
    assert isinstance(result["unmet_gates"], list)


def test_depth_forcing_function_can_relax_is_bool() -> None:
    """REQ-GATE-005: depth_forcing_function_can_relax is a boolean."""
    result = build_synthesis()
    assert isinstance(result["depth_forcing_function_can_relax"], bool)


def test_p01_has_clean_defensible_verdict_is_bool() -> None:
    """p01_has_clean_defensible_verdict is a boolean (never None)."""
    result = build_synthesis()
    assert isinstance(result["p01_has_clean_defensible_verdict"], bool)


def test_duration_s_is_positive() -> None:
    """duration_s must be a positive float (aggregation is sub-second but non-zero)."""
    result = build_synthesis()
    assert isinstance(result["duration_s"], float)
    assert result["duration_s"] > 0.0


def test_field_provenance_covers_required_fields() -> None:
    """field_provenance carries principle annotations for all required fields."""
    result = build_synthesis()
    prov = result.get("field_provenance", {})
    missing = [f for f in REQUIRED_FIELDS if f not in prov]
    assert not missing, f"field_provenance missing principles for: {missing}"


# ---------------------------------------------------------------------------
# Fabrication gate: flagged exp3528 must produce null graph-coloring fields
# ---------------------------------------------------------------------------

def test_exp3528_flagged_yields_null_graph_coloring_fields() -> None:
    """REQ-GATE-004: when exp3528 is flagged_adversarial, all graph-coloring fields are null.

    This is the core fabrication gate test for .325: the graph-coloring Route-1
    result (vanilla_descent=0.2, beats_strong_baseline) must NOT leak into
    headline fields when the source artifact is flagged.
    """
    result = build_synthesis()
    # exp3528 is known to be flagged_adversarial=True in .325
    assert result["p01_route1_graph_coloring_verdict"] is None, (
        "p01_route1_graph_coloring_verdict must be null when exp3528 is flagged"
    )
    assert result["p01_route1_headroom_preserved"] is None
    assert result["p01_route1_beats_strong_baseline"] is None


# ---------------------------------------------------------------------------
# Positive P0.1 signal: exp3529 Sudoku is clean and shows energy power
# ---------------------------------------------------------------------------

def test_exp3529_sudoku_energy_power_visible() -> None:
    """exp3529 is clean and energy_power_gradient_present=True — positive P0.1 signal."""
    result = build_synthesis()
    # exp3529 is not flagged — energy power must be reported
    assert result["p01_sudoku_energy_power_visible"] is True, (
        "p01_sudoku_energy_power_visible should be True from exp3529"
    )


# ---------------------------------------------------------------------------
# Route-2 informative negative: exp3531 headroom present + flip_count > 0
# ---------------------------------------------------------------------------

def test_exp3531_route2_has_headroom() -> None:
    """exp3531 reports corpus_oracle_exceeds_sc=True (headroom present)."""
    result = build_synthesis()
    assert result["p01_route2_corpus_had_headroom"] is True, (
        "p01_route2_corpus_had_headroom should be True from exp3531"
    )


def test_exp3531_route2_flip_count_nonzero() -> None:
    """exp3531 flip_count_best_vs_sc=3 — not degenerate (FALSE_NEGATIVE_RISK check)."""
    result = build_synthesis()
    assert result["p01_route2_flip_count"] is not None
    assert result["p01_route2_flip_count"] > 0, (
        "flip_count must be >0 for the Route-2 test to be non-degenerate"
    )


def test_exp3531_route2_delta_is_negative() -> None:
    """exp3531 delta < 0 — informative negative (energy loses to SC on headroom corpus)."""
    result = build_synthesis()
    assert result["p01_route2_delta"] is not None
    assert result["p01_route2_delta"] < 0.0, (
        "Route-2 delta should be negative (energy loses to SC in .325 test)"
    )


# ---------------------------------------------------------------------------
# Depth-Over-Breadth relax: P0.1 defensible + G2 external-in-motion
# ---------------------------------------------------------------------------

def test_p01_has_clean_defensible_verdict_true() -> None:
    """P0.1 has a clean defensible verdict via Sudoku (exp3529) and Route-2 (exp3531)."""
    result = build_synthesis()
    assert result["p01_has_clean_defensible_verdict"] is True, (
        "p01_has_clean_defensible_verdict should be True: Sudoku clean + Route-2 informative"
    )


def test_depth_forcing_function_can_relax_true() -> None:
    """REQ-GATE-005: depth_forcing_function_can_relax=True when P0.1 defensible + G2 in-motion."""
    result = build_synthesis()
    assert result["depth_forcing_function_can_relax"] is True, (
        "depth_forcing_function_can_relax should be True given P0.1 defensible + G2 in-motion"
    )


# ---------------------------------------------------------------------------
# Secondary headline: exp3532 promoted aggregation
# ---------------------------------------------------------------------------

def test_aggregation_positive_promoted_contains_auroc() -> None:
    """exp3532 promoted aggregation AUROC is reported (mean_auroc=0.9234)."""
    result = build_synthesis()
    promo = result["aggregation_positive_promoted"]
    assert promo is not None, "aggregation_positive_promoted should not be null"
    assert "mean_auroc" in promo, f"Expected 'mean_auroc' in {promo!r}"


# ---------------------------------------------------------------------------
# G2 status: exp3534 package clean + external-in-motion
# ---------------------------------------------------------------------------

def test_g2_package_status_not_missing() -> None:
    """g2_package_status must not be the 'missing' sentinel (exp3534 is present)."""
    result = build_synthesis()
    assert result["g2_package_status"] != "exp3534_missing"


def test_g2_tracks_operator_publication_gate_state() -> None:
    """G2 follows the external operator gate state, not a hard-coded test assumption."""
    state = json.loads(_PUBLICATION_GATE_STATE.read_text())
    result = build_synthesis()
    assert result["g2"] is bool(state.get("g2_independent_reproducer", False))


def test_g2_not_auto_flipped_by_package_match() -> None:
    """The exp3534 package can be present while the external G2 gate remains false."""

    class FakePublicationGate:
        @staticmethod
        def evaluate() -> dict:
            return {
                "gates": {
                    "G1": {"pass": True},
                    "G2": {"pass": False},
                    "G3": {"pass": True},
                    "G4": {"pass": True},
                },
                "unmet_gates": ["G2"],
            }

    with patch.object(exp3537, "_load_publication_gate", return_value=FakePublicationGate):
        result = build_synthesis()

    assert result["g2_package_status"] != "exp3534_missing"
    assert result["g2"] is False, (
        "G2 must remain false until publication_gate.py reports the operator-updated "
        "external reproducer state."
    )


# ---------------------------------------------------------------------------
# main() integration: writes a readable JSON artifact to disk
# ---------------------------------------------------------------------------

def test_main_writes_valid_json(tmp_path: Path) -> None:
    """main() writes a valid JSON artifact with all required fields."""
    out = tmp_path / "experiment_3537_g_gate_status_synthesis_v325.json"
    with patch.object(exp3537, "OUT_PATH", out):
        main()
    assert out.exists()
    data = json.loads(out.read_text())
    for field in REQUIRED_FIELDS:
        assert field in data, f"main() output missing field: {field!r}"


def test_main_honest_verdict_starts_with_complete(tmp_path: Path) -> None:
    """REQ-GATE-002: main() output honest_verdict starts with 'complete:'."""
    out = tmp_path / "experiment_3537_g_gate_status_synthesis_v325.json"
    with patch.object(exp3537, "OUT_PATH", out):
        main()
    data = json.loads(out.read_text())
    assert data["honest_verdict"].startswith("complete:")


def test_main_availability_summary_present(tmp_path: Path) -> None:
    """main() output includes availability_summary for all 7 .325 upstream artifacts."""
    out = tmp_path / "experiment_3537_g_gate_status_synthesis_v325.json"
    with patch.object(exp3537, "OUT_PATH", out):
        main()
    data = json.loads(out.read_text())
    avail = data.get("availability_summary", {})
    for exp_id in ("exp3528", "exp3529", "exp3530", "exp3531", "exp3532", "exp3533", "exp3534"):
        assert exp_id in avail, f"availability_summary missing {exp_id}"


def test_main_exp3528_flagged_in_availability(tmp_path: Path) -> None:
    """availability_summary must report exp3528 as skipped_flagged_adversarial."""
    out = tmp_path / "experiment_3537_g_gate_status_synthesis_v325.json"
    with patch.object(exp3537, "OUT_PATH", out):
        main()
    data = json.loads(out.read_text())
    assert data["availability_summary"]["exp3528"] == "skipped_flagged_adversarial", (
        "exp3528 is known-flagged in .325 and must appear as skipped_flagged_adversarial"
    )
