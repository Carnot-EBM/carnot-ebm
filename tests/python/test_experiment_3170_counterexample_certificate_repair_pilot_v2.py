"""Tests for Exp 3170 counterexample-certificate repair pilot v2.

Spec refs: REQ-VERIFY-3170, SCENARIO-VERIFY-3170.

WHY THESE TESTS EXIST
---------------------
The certificate repair pilot is the input package for future model repair
calls.  A broken package — missing counterexamples, missing MCS, wrong row
types — would cause repair calls to start from incomplete evidence.  These
tests verify that the module produces a complete, correct, and replayable
artifact given a range of upstream artifact scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import counterexample_certificate_repair_pilot_v2 as mod


# ---------------------------------------------------------------------------
# Required artifact fields (mirrors the module constant for symmetry check)
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = {
    "counterexample_certificate_repair_pilot_v2_ready",
    "exact_row_count",
    "counterexample_count",
    "certificate_records",
    "bounded_frontier_records",
    "unavailable_certificate_fields",
    "prior_repair_candidates_scored",
    "exact_accept_count",
    "repair_call_required_for_next_step",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_standard_headers(root: Path) -> None:
    """Write minimal repo-root text files so source tracing doesn't break."""
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No fake repair claims\n", encoding="utf-8")
    (root / "openspec" / "capabilities" / "verification").mkdir(parents=True, exist_ok=True)
    (root / "openspec" / "capabilities" / "verification" / "spec.md").write_text(
        "REQ-VERIFY-3170\nSCENARIO-VERIFY-3170\n"
        "results/experiment_3170_counterexample_certificate_repair_pilot_v2.json\n",
        encoding="utf-8",
    )


def _write_module_stub(root: Path) -> None:
    """Write a minimal module stub so sha256 tracing of the module file works."""
    module_dir = root / "python" / "carnot" / "verify"
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "counterexample_certificate_repair_pilot_v2.py").write_text(
        "# stub\n", encoding="utf-8"
    )


def _write_exp3111(root: Path, override: dict[str, Any] | None = None) -> None:
    """Write a minimal exp3111 certified coherence artifact."""
    payload: dict[str, Any] = {
        "artifact": "experiment_3111_certified_coherence_z3_mcs_feedback_v3",
        "certificate_count": 4,
        "certificates": [
            {
                "fixture_id": "resyn-3084-arith-003",
                "exact_label": "INVALID",
                "task_family": "arithmetic_code_assertions",
                "coherence_status": "incoherent",
                "solver_authority": "python_ast_runtime_execution",
                "minimal_correction_set": {"from": 47, "kind": "replace_claimed_value", "to": 43},
                "unsat_core": ["computed_value", "claimed_value"],
            },
            {
                "fixture_id": "resyn-3084-smt-000",
                "exact_label": "UNSAT",
                "task_family": "smt_constraints",
                "coherence_status": "incoherent",
                "solver_authority": "z3_solver",
                "minimal_correction_set": {
                    "constraints_to_relax": ["constraint_0"],
                    "kind": "remove_conflicting_constraint",
                },
                "unsat_core": ["constraint_0", "constraint_1"],
            },
            {
                "fixture_id": "resyn-3084-smt-001",
                "exact_label": "SAT",
                "task_family": "smt_constraints",
                "coherence_status": "coherent",
                "solver_authority": "z3_solver",
                "minimal_correction_set": {},
                "unsat_core": [],
            },
            {
                "fixture_id": "resyn-3084-repair-json-000",
                "exact_label": "REPAIRABLE",
                "task_family": "repairable_invalid_candidates",
                "coherence_status": "incoherent",
                "solver_authority": "python_json_parser",
                "minimal_correction_set": {
                    "edits": [{"operation": "insert_delimiter", "token": ","}],
                    "kind": "json_token_edit",
                },
                "unsat_core": [],
            },
        ],
    }
    if override:
        payload.update(override)
    _write_json(root, mod.EXP3111_REL_PATH, payload)


def _write_exp3125(root: Path, override: dict[str, Any] | None = None) -> None:
    """Write a minimal exp3125 prefix-closed deterministic bound pilot artifact."""
    payload: dict[str, Any] = {
        "artifact": "experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1",
        "bound_width": 0.0020155392,
        "explored_mass": 1.0,
        "explored_prefix_count": 10,
        "accepted_prefix_count": 2,
        "constraint_families": [
            "json_like_answer_shape",
            "answer_label_match",
            "bounded_score_invariant",
            "forbidden_unknown_token",
        ],
        "fixture_details": [
            {"expected_answer": "VALID", "expected_score": 1, "fixture_id": "pc-3125-valid", "required_tag": None},
            {"expected_answer": "INVALID", "expected_score": 0, "fixture_id": "pc-3125-invalid", "required_tag": None},
        ],
        "frontier_rows": [
            {"depth": 0, "fixture_id": "pc-3125-valid", "prefix": [], "probability_mass": 0.5, "reason": "prefix_has_satisfying_extension", "status": "viable"},
            {"depth": 1, "fixture_id": "pc-3125-valid", "prefix": ["{"], "probability_mass": 0.3, "reason": "prefix_has_satisfying_extension", "status": "viable"},
            {"depth": 1, "fixture_id": "pc-3125-valid", "prefix": ["x"], "probability_mass": 0.2, "reason": "no_satisfying_extension", "status": "pruned"},
            {"depth": 0, "fixture_id": "pc-3125-invalid", "prefix": [], "probability_mass": 0.5, "reason": "prefix_has_satisfying_extension", "status": "viable"},
            {"depth": 1, "fixture_id": "pc-3125-invalid", "prefix": ["{"], "probability_mass": 0.3, "reason": "prefix_has_satisfying_extension", "status": "viable"},
            {"depth": 1, "fixture_id": "pc-3125-invalid", "prefix": ["y"], "probability_mass": 0.2, "reason": "no_satisfying_extension", "status": "pruned"},
        ],
    }
    if override:
        payload.update(override)
    _write_json(root, mod.EXP3125_REL_PATH, payload)


def _write_exp3136(root: Path) -> None:
    """Write a minimal exp3136 false accept autopsy artifact."""
    _write_json(
        root,
        mod.EXP3136_REL_PATH,
        {
            "artifact": "experiment_3136_false_accept_root_cause_autopsy_v1",
            "false_accept_row_ids": ["resyn-3084-arith-003", "resyn-3084-smt-000"],
        },
    )


def _write_exp3137(root: Path) -> None:
    """Write a minimal exp3137 exact safe accept contract artifact."""
    _write_json(
        root,
        mod.EXP3137_REL_PATH,
        {
            "artifact": "experiment_3137_exact_safe_accept_abstain_contract_v1",
            "acceptance_contract_v1_ready": True,
        },
    )


def _write_exp3138(root: Path) -> None:
    """Write a minimal exp3138 canonical grounding pilot artifact."""
    _write_json(
        root,
        mod.EXP3138_REL_PATH,
        {
            "artifact": "experiment_3138_canonical_answer_vericot_grounding_pilot_v1",
            "canonical_grounding_pilot_v1_ready": True,
            "regression_rows_evaluated": 2,
            "false_accept_rows_blocked": 2,
        },
    )


def _write_exp3168(root: Path, gate_state: str = "blocked_flagged_verifier") -> None:
    """Write a minimal exp3168 repair gate decision artifact."""
    _write_json(
        root,
        mod.EXP3168_REL_PATH,
        {
            "artifact": "experiment_3168_repair_gate_decision_v3",
            "repair_gate_decision_v3_ready": True,
            "repair_gate_state": gate_state,
            "gated_skip": gate_state != "unblocked",
        },
    )


def _write_exp3169(
    root: Path,
    *,
    gated_skip: bool = True,
    repair_attempts: list[dict[str, Any]] | None = None,
) -> None:
    """Write a minimal exp3169 repair ladder materializer artifact."""
    _write_json(
        root,
        mod.EXP3169_REL_PATH,
        {
            "artifact": "experiment_3169_repair_ladder_materializer_v4",
            "repair_ladder_materializer_v4_ready": True,
            "gate_state": "blocked_flagged_verifier" if gated_skip else "unblocked",
            "gated_skip": gated_skip,
            "repair_attempts": repair_attempts or [],
            "repair_attempt_count": len(repair_attempts or []),
        },
    )


def _write_all_sources(
    root: Path,
    *,
    include_exp3111: bool = True,
    include_exp3125: bool = True,
    include_exp3169: bool = True,
    exp3169_repair_attempts: list[dict[str, Any]] | None = None,
) -> None:
    """Write all standard upstream sources for happy-path tests."""
    _write_standard_headers(root)
    _write_module_stub(root)
    if include_exp3111:
        _write_exp3111(root)
    if include_exp3125:
        _write_exp3125(root)
    _write_exp3136(root)
    _write_exp3137(root)
    _write_exp3138(root)
    _write_exp3168(root)
    if include_exp3169:
        _write_exp3169(root, repair_attempts=exp3169_repair_attempts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRequiredFields:
    """Verify all required schema fields are present in the artifact."""

    def test_req_verify_3170_spec_anchor_exists(self) -> None:
        """REQ-VERIFY-3170: OpenSpec declares the certificate-first repair pilot."""
        spec = (mod.REPO_ROOT / "openspec/capabilities/verification/spec.md").read_text(
            encoding="utf-8"
        )

        assert "REQ-VERIFY-3170" in spec
        assert "SCENARIO-VERIFY-3170" in spec
        assert mod.OUTPUT_REL_PATH.as_posix() in spec
        assert "counterexample_certificate_repair_pilot_v2_ready" in spec
        assert "unavailable_certificate_fields" in spec

    def test_all_required_fields_present(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-3170: artifact must contain every required field."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        missing = REQUIRED_FIELDS - set(artifact.keys())
        assert not missing, f"Artifact missing required fields: {sorted(missing)}"

    def test_required_fields_match_module_constant(self) -> None:
        """Module's REQUIRED_FIELDS constant must match test's expected set."""
        assert mod.REQUIRED_FIELDS == REQUIRED_FIELDS


class TestHonestVerdict:
    """Verify the honest_verdict satisfies the terminal-prefix discipline."""

    TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped_")

    def test_verdict_has_terminal_prefix(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-3170: honest_verdict must start with a terminal prefix."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        verdict = artifact["honest_verdict"]
        assert any(verdict.startswith(p) for p in self.TERMINAL_PREFIXES), (
            f"honest_verdict lacks terminal prefix: {verdict!r}"
        )

    def test_verdict_contains_ready_field(self, tmp_path: Path) -> None:
        """Verdict string should reflect that the pilot is ready."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert "counterexample_certificate_repair_pilot_v2_ready=true" in artifact["honest_verdict"]


class TestCertificateRecords:
    """Verify the certificate_records list contains the correct pilot rows."""

    def test_certificate_uses_exp3111_mcs_when_available(self, tmp_path: Path) -> None:
        """REQ-VERIFY-3170: certificate MCS values come from exact source artifacts."""
        _write_all_sources(tmp_path)
        exp3111_path = tmp_path / mod.EXP3111_REL_PATH
        exp3111 = json.loads(exp3111_path.read_text(encoding="utf-8"))
        for cert in exp3111["certificates"]:
            if cert["fixture_id"] == "resyn-3084-arith-003":
                cert["diagnostics"] = {"claimed_value": 47, "computed_value": 41}
                cert["minimal_correction_set"] = {
                    "from": 47,
                    "kind": "replace_claimed_value",
                    "to": 41,
                }
        exp3111_path.write_text(json.dumps(exp3111), encoding="utf-8")

        artifact = mod.build_artifact(repo_root=tmp_path)
        by_id = {r["row_id"]: r for r in artifact["certificate_records"]}

        assert by_id["resyn-3084-arith-003"]["mcs"]["to"] == 41
        assert by_id["resyn-3084-arith-003"]["minimal_failing_assignment"][
            "computed_value"
        ] == 41

    def test_five_pilot_rows_built(self, tmp_path: Path) -> None:
        """Pilot must include 2 false-accept + 1 satisfiable-drift + 2 fragment-code rows."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        records = artifact["certificate_records"]
        assert len(records) == 5, f"Expected 5 certificate records, got {len(records)}"

    def test_false_accept_rows_present(self, tmp_path: Path) -> None:
        """Both known false-accept rows must appear in certificate_records."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        row_ids = {r["row_id"] for r in artifact["certificate_records"]}
        assert "resyn-3084-arith-003" in row_ids
        assert "resyn-3084-smt-000" in row_ids

    def test_satisfiable_drift_row_present(self, tmp_path: Path) -> None:
        """Satisfiable-drift anchor row must appear in certificate_records."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        row_ids = {r["row_id"] for r in artifact["certificate_records"]}
        assert "resyn-3084-smt-001" in row_ids

    def test_fragment_code_rows_present(self, tmp_path: Path) -> None:
        """Both fragment-code rows must appear in certificate_records."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        row_ids = {r["row_id"] for r in artifact["certificate_records"]}
        assert "resyn-3084-repair-json-000" in row_ids
        assert "resyn-3084-repair-json-003" in row_ids

    def test_row_types_correct(self, tmp_path: Path) -> None:
        """Each row has the correct row_type classification."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        by_id = {r["row_id"]: r for r in artifact["certificate_records"]}

        assert by_id["resyn-3084-arith-003"]["row_type"] == "false_accept"
        assert by_id["resyn-3084-smt-000"]["row_type"] == "false_accept"
        assert by_id["resyn-3084-smt-001"]["row_type"] == "satisfiable_drift"
        assert by_id["resyn-3084-repair-json-000"]["row_type"] == "fragment_code"
        assert by_id["resyn-3084-repair-json-003"]["row_type"] == "fragment_code"

    def test_false_accept_rows_have_mcs(self, tmp_path: Path) -> None:
        """False-accept rows must have a non-empty MCS so repair calls are bounded."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        by_id = {r["row_id"]: r for r in artifact["certificate_records"]}

        assert by_id["resyn-3084-arith-003"]["mcs"], "arith-003 must have MCS"
        assert by_id["resyn-3084-smt-000"]["mcs"], "smt-000 must have MCS"

    def test_satisfiable_drift_row_empty_mcs(self, tmp_path: Path) -> None:
        """Satisfiable-drift row must have an empty MCS (nothing to correct)."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        by_id = {r["row_id"]: r for r in artifact["certificate_records"]}
        assert by_id["resyn-3084-smt-001"]["mcs"] == {}

    def test_fragment_code_rows_have_mcs(self, tmp_path: Path) -> None:
        """Fragment-code rows must have a non-empty MCS specifying the token edit."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        by_id = {r["row_id"]: r for r in artifact["certificate_records"]}
        assert by_id["resyn-3084-repair-json-000"]["mcs"], "repair-json-000 must have MCS"
        assert by_id["resyn-3084-repair-json-003"]["mcs"], "repair-json-003 must have MCS"

    def test_verifier_to_rerun_specified(self, tmp_path: Path) -> None:
        """Every certificate record must name the verifier to rerun."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        for rec in artifact["certificate_records"]:
            assert rec["verifier_to_rerun"], f"Row {rec['row_id']} missing verifier_to_rerun"

    def test_exact_label_specified(self, tmp_path: Path) -> None:
        """Every certificate record must specify its exact label."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        for rec in artifact["certificate_records"]:
            assert rec["exact_label"], f"Row {rec['row_id']} missing exact_label"


class TestCounterexampleCount:
    """Verify counterexample_count reflects concrete failing assignments."""

    def test_counterexample_count_positive(self, tmp_path: Path) -> None:
        """At least the two false-accept rows provide concrete counterexamples."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        # At minimum the 2 false-accept rows and 2 fragment-code rows have non-empty
        # minimal_failing_assignment
        assert artifact["counterexample_count"] >= 2

    def test_exact_row_count_is_five(self, tmp_path: Path) -> None:
        """exact_row_count must equal the number of certificate records."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["exact_row_count"] == len(artifact["certificate_records"])


class TestBoundedFrontierRecords:
    """Verify BEAVER-style bounded frontier records are built from exp3125."""

    def test_frontier_records_built_when_exp3125_present(self, tmp_path: Path) -> None:
        """bounded_frontier_records must be non-empty when exp3125 is available."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["bounded_frontier_records"], "Expected frontier records from exp3125"

    def test_frontier_records_per_fixture(self, tmp_path: Path) -> None:
        """There should be one frontier record per fixture in exp3125."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        frontier_ids = {r["fixture_id"] for r in artifact["bounded_frontier_records"]}
        assert "pc-3125-valid" in frontier_ids
        assert "pc-3125-invalid" in frontier_ids

    def test_bound_width_propagated(self, tmp_path: Path) -> None:
        """bound_width from exp3125 must appear in every frontier record."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        for rec in artifact["bounded_frontier_records"]:
            assert rec["bound_width"] > 0.0, f"Frontier record {rec['fixture_id']} has zero bound_width"

    def test_frontier_records_empty_when_exp3125_absent(self, tmp_path: Path) -> None:
        """bounded_frontier_records must be empty when exp3125 is absent."""
        _write_all_sources(tmp_path, include_exp3125=False)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["bounded_frontier_records"] == []

    def test_unavailable_field_recorded_when_exp3125_absent(self, tmp_path: Path) -> None:
        """Missing exp3125 must appear in unavailable_certificate_fields."""
        _write_all_sources(tmp_path, include_exp3125=False)
        artifact = mod.build_artifact(repo_root=tmp_path)
        unavailable = artifact["unavailable_certificate_fields"]
        assert any("exp3125" in f for f in unavailable), (
            "Expected exp3125 absence to be noted in unavailable_certificate_fields"
        )


class TestUnavailableFields:
    """Verify that unavailable fields are enumerated honestly."""

    def test_live_logprobs_always_unavailable(self, tmp_path: Path) -> None:
        """Live model logprobs are always unavailable in this CPU-only task."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        unavailable = artifact["unavailable_certificate_fields"]
        assert any("live_model_logprobs" in f for f in unavailable)

    def test_satisfiable_drift_empty_assignment_noted(self, tmp_path: Path) -> None:
        """The empty minimal_failing_assignment for satisfiable-drift must be noted."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        unavailable = artifact["unavailable_certificate_fields"]
        assert any("satisfiable_drift" in f for f in unavailable)

    def test_gated_skip_noted_in_unavailable(self, tmp_path: Path) -> None:
        """Gated skip in exp3169 must appear in unavailable_certificate_fields."""
        _write_all_sources(tmp_path, include_exp3169=True)
        artifact = mod.build_artifact(repo_root=tmp_path)
        unavailable = artifact["unavailable_certificate_fields"]
        assert any("exp3169" in f or "prior_repair_candidates" in f for f in unavailable)


class TestPriorRepairCandidates:
    """Verify that prior repair candidate scoring works correctly."""

    def test_zero_scored_when_gate_blocked(self, tmp_path: Path) -> None:
        """When exp3169 gate is blocked with no repair_attempts, scored=0."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["prior_repair_candidates_scored"] == 0

    def test_zero_scored_when_exp3169_absent(self, tmp_path: Path) -> None:
        """When exp3169 is absent entirely, scored=0 and a note is in unavailable."""
        _write_all_sources(tmp_path, include_exp3169=False)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["prior_repair_candidates_scored"] == 0

    def test_scored_when_repair_attempts_present(self, tmp_path: Path) -> None:
        """When exp3169 has repair_attempts, the relevant certs are scored."""
        attempts = [
            {
                "row_id": "resyn-3084-arith-003",
                "verdict": "repair_accepted",
                "repaired_label": "INVALID",
                "accepted": True,
            }
        ]
        _write_all_sources(tmp_path, exp3169_repair_attempts=attempts)
        # Override gated_skip to False to match the repair attempt scenario
        _write_exp3169(tmp_path, gated_skip=False, repair_attempts=attempts)

        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["prior_repair_candidates_scored"] == 1

    def test_exact_accept_increments_when_repair_accepted(self, tmp_path: Path) -> None:
        """exact_accept_count must increment when a repair is accepted and label matches."""
        attempts = [
            {
                "row_id": "resyn-3084-arith-003",
                "verdict": "repair_accepted",
                "repaired_label": "INVALID",
                "accepted": True,
            }
        ]
        _write_all_sources(tmp_path)
        _write_exp3169(tmp_path, gated_skip=False, repair_attempts=attempts)

        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["exact_accept_count"] == 1

    def test_exact_accept_rejects_label_mismatch(self, tmp_path: Path) -> None:
        """REQ-VERIFY-3170: accepted repairs count only when exact labels match."""
        attempts = [
            {
                "row_id": "resyn-3084-arith-003",
                "verdict": "repair_accepted",
                "repaired_label": "VALID",
                "accepted": True,
            }
        ]
        _write_all_sources(tmp_path)
        _write_exp3169(tmp_path, gated_skip=False, repair_attempts=attempts)

        artifact = mod.build_artifact(repo_root=tmp_path)
        by_id = {r["row_id"]: r for r in artifact["certificate_records"]}

        assert artifact["exact_accept_count"] == 0
        assert by_id["resyn-3084-arith-003"]["prior_repair_score"][
            "exact_label_matches"
        ] is False


class TestRepairCallRequired:
    """Verify repair_call_required_for_next_step reflects the gate state."""

    def test_repair_required_when_gate_blocked(self, tmp_path: Path) -> None:
        """If no repairs were accepted, repair_call_required=True."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["repair_call_required_for_next_step"] is True

    def test_repair_not_required_when_all_accepted(self, tmp_path: Path) -> None:
        """If all 5 rows are accepted, repair_call_required_for_next_step=False."""
        # Build 5 repair attempts, one for each row, all accepted
        all_row_ids = [
            "resyn-3084-arith-003",
            "resyn-3084-smt-000",
            "resyn-3084-smt-001",
            "resyn-3084-repair-json-000",
            "resyn-3084-repair-json-003",
        ]
        exact_labels = {
            "resyn-3084-arith-003": "INVALID",
            "resyn-3084-smt-000": "UNSAT",
            "resyn-3084-smt-001": "SAT",
            "resyn-3084-repair-json-000": "REPAIRABLE",
            "resyn-3084-repair-json-003": "REPAIRABLE",
        }
        attempts = [
            {
                "row_id": rid,
                "verdict": "repair_accepted",
                "repaired_label": exact_labels[rid],
                "accepted": True,
            }
            for rid in all_row_ids
        ]
        _write_all_sources(tmp_path)
        _write_exp3169(tmp_path, gated_skip=False, repair_attempts=attempts)

        artifact = mod.build_artifact(repo_root=tmp_path)
        # All 5 rows accepted -> not required.
        assert artifact["repair_call_required_for_next_step"] is False
        assert "repair_call_required_for_next_step=false" in artifact["honest_verdict"]


class TestInferenceSubstrate:
    """Verify inference_substrate declares CPU-only exact authority work."""

    def test_no_live_inference(self, tmp_path: Path) -> None:
        """inference_substrate.no_live_llm_inference must be True."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        substrate = artifact["inference_substrate"]
        assert substrate["no_live_llm_inference"] is True

    def test_executes_models_false(self, tmp_path: Path) -> None:
        """inference_substrate.executes_models must be False."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["inference_substrate"]["executes_models"] is False

    def test_live_model_calls_zero(self, tmp_path: Path) -> None:
        """inference_substrate.live_model_calls must be 0."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["inference_substrate"]["live_model_calls"] == 0


class TestSourceArtifacts:
    """Verify source_artifacts list traces to the correct upstream files."""

    def test_source_artifacts_list_nonempty(self, tmp_path: Path) -> None:
        """source_artifacts must be a non-empty list."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["source_artifacts"]

    def test_required_sources_marked_required(self, tmp_path: Path) -> None:
        """Exact sources plus spec and tests must be marked in source_artifacts."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        required_roles = {
            "verification_openspec",
            "exp3111_certified_coherence_mcs",
            "exp3125_prefix_closed_deterministic_bound",
            "exp3136_false_accept_autopsy",
            "exp3137_exact_safe_accept_contract",
            "exp3138_canonical_grounding_pilot",
            "exp3170_tests",
        }
        src_by_role = {s["role"]: s for s in artifact["source_artifacts"]}
        for role in required_roles:
            assert role in src_by_role, f"Missing required source role: {role}"
            assert src_by_role[role]["required"] is True, f"Role {role} not marked required"

    def test_present_sources_have_sha256(self, tmp_path: Path) -> None:
        """Every present source artifact must have a non-None sha256."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        for src in artifact["source_artifacts"]:
            if src["present"]:
                assert src["sha256"] is not None, (
                    f"Source {src['path']} is present but has no sha256"
                )


class TestWriteArtifact:
    """Verify write_artifact produces a valid JSON file at the expected path."""

    def test_write_creates_file(self, tmp_path: Path) -> None:
        """write_artifact must create the output file."""
        _write_all_sources(tmp_path)
        output_path = tmp_path / mod.OUTPUT_REL_PATH
        returned_path = mod.write_artifact(repo_root=tmp_path, output_path=output_path)
        assert returned_path.exists()

    def test_written_file_is_valid_json(self, tmp_path: Path) -> None:
        """Output file must be parseable JSON."""
        _write_all_sources(tmp_path)
        output_path = tmp_path / mod.OUTPUT_REL_PATH
        mod.write_artifact(repo_root=tmp_path, output_path=output_path)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_written_file_has_all_required_fields(self, tmp_path: Path) -> None:
        """Written JSON must contain all required schema fields."""
        _write_all_sources(tmp_path)
        output_path = tmp_path / mod.OUTPUT_REL_PATH
        mod.write_artifact(repo_root=tmp_path, output_path=output_path)
        data = json.loads(output_path.read_text(encoding="utf-8"))
        missing = REQUIRED_FIELDS - set(data.keys())
        assert not missing, f"Written artifact missing fields: {sorted(missing)}"


class TestDurationIsReal:
    """Verify duration_s is recorded and plausible for CPU work."""

    def test_duration_positive(self, tmp_path: Path) -> None:
        """duration_s must be > 0 (real wall-clock time, not fabricated zero)."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["duration_s"] > 0.0

    def test_duration_not_implausibly_long(self, tmp_path: Path) -> None:
        """duration_s must be < 5s for CPU-only certificate construction."""
        _write_all_sources(tmp_path)
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["duration_s"] < 5.0, (
            f"CPU-only task took {artifact['duration_s']:.3f}s — implausibly long"
        )


class TestEdgeCases:
    """Cover edge-case branches for 100% line coverage."""

    def test_exp3125_with_empty_fixture_details_noted_in_unavailable(self, tmp_path: Path) -> None:
        """When exp3125 has an empty fixture_details list, the gap is noted as unavailable."""
        _write_all_sources(tmp_path, include_exp3125=False)
        # Write exp3125 with empty fixture_details (present but useless)
        _write_exp3125(tmp_path, override={"fixture_details": []})
        artifact = mod.build_artifact(repo_root=tmp_path)
        assert artifact["bounded_frontier_records"] == []
        unavailable = artifact["unavailable_certificate_fields"]
        assert any("fixture_details" in f for f in unavailable)

    def test_repair_attempt_for_unknown_row_id_is_skipped(self, tmp_path: Path) -> None:
        """A repair_attempt whose row_id matches no certificate record is silently skipped."""
        attempts = [
            {"row_id": "no-such-row", "verdict": "repair_accepted", "repaired_label": "X", "accepted": True}
        ]
        _write_all_sources(tmp_path)
        _write_exp3169(tmp_path, gated_skip=False, repair_attempts=attempts)
        artifact = mod.build_artifact(repo_root=tmp_path)
        # The unknown row_id is skipped; scored count stays 0
        assert artifact["prior_repair_candidates_scored"] == 0

    def test_exp3111_absent_noted_in_unavailable(self, tmp_path: Path) -> None:
        """When exp3111 is absent, its MCS unavailability is noted."""
        _write_all_sources(tmp_path, include_exp3111=False)
        artifact = mod.build_artifact(repo_root=tmp_path)
        unavailable = artifact["unavailable_certificate_fields"]
        assert any("exp3111" in f or "solver_mcs_records" in f for f in unavailable)

    def test_build_artifact_with_no_repo_root_uses_default(self) -> None:
        """build_artifact() with repo_root=None must fall back to REPO_ROOT without error."""
        # This exercises the ``if repo_root is None: repo_root = REPO_ROOT`` branch.
        # We call it for real because REPO_ROOT is the actual project root.
        artifact = mod.build_artifact()  # no repo_root kwarg
        assert "counterexample_certificate_repair_pilot_v2_ready" in artifact

    def test_write_artifact_with_default_output_path(self, tmp_path: Path) -> None:
        """write_artifact() with output_path=None uses the default path under repo_root."""
        _write_all_sources(tmp_path)
        returned_path = mod.write_artifact(repo_root=tmp_path, output_path=None)
        # The default path is repo_root / OUTPUT_REL_PATH
        expected = tmp_path / mod.OUTPUT_REL_PATH
        assert returned_path == expected
        assert returned_path.exists()
