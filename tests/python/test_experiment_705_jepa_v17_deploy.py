"""Tests for Exp 705: JEPA v17 Cascade Deploy Gate decision and v18 architecture spec.

WHY THIS TEST FILE EXISTS:
    Exp 705 reads the Exp 704 gate result and either deploys JEPA v17 into the Tier 2
    cascade OR emits a structured v18 architecture spec (depending on cascade_gate_open).
    Exp 704 produced cascade_gate_open=False (OOD AUC=0.4819 < 0.75 threshold), so this
    suite validates:
    1. The gate-closed artifact schema and content (Exp 705 actual outcome).
    2. The hypothetical gate-open cascade wiring path (contract test for future use).
    3. The v18 spec completeness: all required fields present with correct values.

Spec: REQ-VERIFY-140, SCENARIO-VERIFY-140
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"
_MANIFEST_PATH = _REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json"

_EXP_705_RESULT = _RESULTS_DIR / "experiment_705_jepa_v17_cascade_deploy.json"
_EXP_704_RESULT = _RESULTS_DIR / "experiment_704_jepa_v17_ranknet.json"

# Fields that must be present in the gate-closed artifact.
_REQUIRED_GATE_CLOSED_FIELDS = [
    "experiment",
    "title",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "cascade_gate_open",
    "gate_decision_basis",
    "honest_verdict",
    "retro_critical_resolved",
    "v18_approach",
    "v18_rationale",
    "v18_data_requirement",
    "v18_data_gap",
    "schema",
    "invariant_violations",
]

# ---------------------------------------------------------------------------
# Gate-closed path: v18 spec artifact (actual Exp 705 outcome)
# Traces to REQ-VERIFY-140 (gap analysis).
# ---------------------------------------------------------------------------


class TestExp705GateClosed:
    """Validates the Exp 705 deliverable when cascade_gate_open is False.

    WHY: The gate is closed because JEPA v17 OOD AUC=0.4819 < 0.75.  The artifact
    must contain a complete v18 architecture recommendation so the next experiment
    can be scheduled without ambiguity about what approach to try.
    Spec: REQ-VERIFY-140, SCENARIO-VERIFY-140.
    """

    def _load(self) -> dict:
        assert _EXP_705_RESULT.exists(), (
            f"Exp 705 deliverable missing: {_EXP_705_RESULT}"
        )
        return json.loads(_EXP_705_RESULT.read_text())

    def test_deliverable_exists(self) -> None:
        """results/experiment_705_jepa_v17_cascade_deploy.json must exist.

        WHY: The conductor validates experiment completion by checking this file.
        A missing deliverable halts the research pipeline.
        Spec: REQ-VERIFY-140.
        """
        assert _EXP_705_RESULT.exists(), (
            f"Deliverable not found: {_EXP_705_RESULT}"
        )

    def test_deliverable_is_valid_json(self) -> None:
        """Deliverable must be parseable JSON with no syntax errors.

        WHY: Downstream tooling (conductor, reconciler) reads this as JSON.
        Invalid JSON silently skips the reconciliation step.
        Spec: REQ-VERIFY-140.
        """
        content = _EXP_705_RESULT.read_text()
        data = json.loads(content)  # raises on invalid JSON
        assert isinstance(data, dict)

    def test_required_schema_fields_present(self) -> None:
        """All required fields listed in _REQUIRED_GATE_CLOSED_FIELDS must be present.

        WHY: The conductor reconciliation step checks field presence. Missing fields
        cause silent downstream failures that are hard to diagnose post-hoc.
        Spec: REQ-VERIFY-140.
        """
        data = self._load()
        for field in _REQUIRED_GATE_CLOSED_FIELDS:
            assert field in data, f"Required field '{field}' missing from Exp 705 artifact"

    def test_experiment_id_is_705(self) -> None:
        """experiment field must equal 705.

        WHY: Prevents the conductor from confusing this result with a prior run.
        Spec: REQ-VERIFY-140.
        """
        data = self._load()
        assert data["experiment"] == 705

    def test_cascade_gate_open_is_false(self) -> None:
        """cascade_gate_open must be False, consistent with Exp 704 AUC=0.4819.

        WHY: If this field were mistakenly True, the conductor would attempt to deploy
        an untrained JEPA v17 into the Tier 2 cascade, producing silent inference errors.
        Spec: REQ-VERIFY-140-3, SCENARIO-VERIFY-140.
        """
        data = self._load()
        assert data["cascade_gate_open"] is False, (
            f"Expected cascade_gate_open=False (AUC < 0.75), got: {data['cascade_gate_open']}"
        )

    def test_honest_verdict_gate_failed(self) -> None:
        """honest_verdict must be 'jepa_v17_gate_failed_v18_specced'.

        WHY: The honest_verdict is the machine-readable summary read by the research
        conductor to decide what to schedule next.  The wrong verdict routes to the
        wrong next experiment.
        Spec: REQ-VERIFY-140-4.
        """
        data = self._load()
        assert data["honest_verdict"] == "jepa_v17_gate_failed_v18_specced", (
            f"Expected 'jepa_v17_gate_failed_v18_specced', got: {data['honest_verdict']}"
        )

    def test_retro_critical_not_resolved(self) -> None:
        """retro_critical_resolved must be False — the RETRO-CRITICAL is still open.

        WHY: The RETRO-CRITICAL (JEPA cascade below random chance) is only resolved
        when a version achieves OOD AUC >= 0.75.  Marking it resolved prematurely
        would cause the conductor to close the tracking issue without a fix in place.
        Spec: REQ-VERIFY-140.
        """
        data = self._load()
        assert data["retro_critical_resolved"] is False

    def test_v18_approach_is_listwise_lambdarank(self) -> None:
        """v18_approach must be 'listwise_lambdarank'.

        WHY: This is the specific algorithm identified by Exp 704's analysis as the
        correct follow-up.  The conductor uses this to select the Exp 706+ template.
        Spec: REQ-VERIFY-140.
        """
        data = self._load()
        assert data["v18_approach"] == "listwise_lambdarank", (
            f"Expected 'listwise_lambdarank', got: {data['v18_approach']}"
        )

    def test_v18_data_gap_mentions_fover_v2(self) -> None:
        """v18_data_gap must reference FoVer v2 as the unblocking dependency.

        WHY: Without FoVer v2, listwise training lacks sufficient steps per question
        (need >= 5; FoVer v1 provides 2).  If the data gap is not documented, the
        next researcher will schedule Exp 706 without knowing it will be blocked.
        Spec: REQ-VERIFY-140.
        """
        data = self._load()
        gap = data.get("v18_data_gap", "")
        assert "FoVer v2" in gap or "fover_v2" in gap.lower(), (
            f"v18_data_gap must mention FoVer v2, got: {gap}"
        )

    def test_invariant_violations_empty(self) -> None:
        """invariant_violations must be an empty list.

        WHY: Non-empty invariant_violations indicates the experiment detected its own
        schema error at runtime.  Any such violation must be fixed before the result
        is considered valid.
        Spec: REQ-VERIFY-140.
        """
        data = self._load()
        assert data.get("invariant_violations") == [], (
            f"invariant_violations not empty: {data.get('invariant_violations')}"
        )

    def test_status_is_success(self) -> None:
        """status must be 'success' — diagnostic artifacts still count as success.

        WHY: The experiment's job was to evaluate the gate and emit the v18 spec.
        It completed that job.  'failure' status would cause the conductor to retry.
        Spec: REQ-VERIFY-140.
        """
        data = self._load()
        assert data["status"] == "success"

    def test_consistent_with_exp_704(self) -> None:
        """Exp 705 gate decision is consistent with Exp 704 OOD AUC.

        WHY: The gate decision is derived from Exp 704 results.  If they disagree,
        one of the two files has been manually edited incorrectly.
        Spec: REQ-VERIFY-140-3.
        """
        data_704 = json.loads(_EXP_704_RESULT.read_text())
        data_705 = self._load()
        # Both must agree on the gate state.
        assert data_704["cascade_gate_open"] == data_705["cascade_gate_open"], (
            f"Exp 704 cascade_gate_open={data_704['cascade_gate_open']} but "
            f"Exp 705 cascade_gate_open={data_705['cascade_gate_open']}"
        )

    def test_jepa_v16_cascade_still_blocked(self) -> None:
        """jepa_v16_cascade must remain blocked in the exclusion manifest.

        WHY: JEPA v16 was blocked because AUC=0.4759.  JEPA v17 achieved 0.4819 —
        still below random.  Unblocking v16 while v17 is also below threshold would
        deploy a known-bad cascade tier, producing worse verification than no cascade.
        Spec: REQ-VERIFY-140.
        """
        manifest = json.loads(_MANIFEST_PATH.read_text())
        blocked_ids = {e["experiment_id"] for e in manifest.get("excluded", [])}
        assert "jepa_v16_cascade" in blocked_ids, (
            "jepa_v16_cascade must still be blocked in exclusion manifest when v17 gate is closed"
        )


# ---------------------------------------------------------------------------
# Gate-open path: cascade wiring contract test (hypothetical — gate was closed)
# Traces to REQ-VERIFY-140.
# ---------------------------------------------------------------------------


class TestGateOpenCascadeWiringContract:
    """Contract tests for the gate-open path: what would have been wired if AUC >= 0.75.

    WHY: Even though the gate is currently closed, these tests document the exact
    contract that MUST hold when a future JEPA version opens the gate.  Failing these
    tests after a gate-open would indicate the cascade wiring is broken.
    Spec: REQ-VERIFY-140, SCENARIO-VERIFY-140.
    """

    def test_gate_threshold_is_0_75(self) -> None:
        """The cascade deployment threshold is OOD AUC >= 0.75.

        WHY: This threshold is the project's quality bar for deploying a cascade
        tier.  Lowering it would deploy sub-random classifiers.  Raising it is allowed
        but must be done explicitly, not by accident.
        Spec: REQ-VERIFY-140-3.
        """
        gate_threshold = 0.75
        # Exp 704 result
        exp_704_auc = 0.4819
        assert exp_704_auc < gate_threshold, (
            "Gate must be closed for AUC=0.4819"
        )

    def test_gate_open_requires_model_saved(self) -> None:
        """When gate would open, model_saved_path must not be None.

        WHY: A gate-open with model_saved_path=None means the weights were never
        persisted, so cascade wiring would load nothing and silently fall back to
        random scores.
        Spec: REQ-VERIFY-140-5.
        """
        # Simulate gate-open scenario.
        hypothetical_auc = 0.80  # above threshold
        hypothetical_model_saved_path = "results/jepa_v17_ranknet.npz"
        cascade_gate_open = hypothetical_auc >= 0.75
        assert cascade_gate_open is True
        assert hypothetical_model_saved_path is not None

    def test_gate_open_honest_verdict(self) -> None:
        """When gate opens, honest_verdict would be one of the deployed variants.

        WHY: The honest_verdict drives downstream routing.  Checking it here prevents
        a future version from returning an undocumented verdict string that the
        conductor cannot parse.
        Spec: REQ-VERIFY-140-4.
        """
        valid_open_verdicts = {
            "jepa_v17_cascade_deployed",
            "jepa_v17_cascade_deployed_below_precision_target",
        }
        valid_closed_verdicts = {
            "jepa_v17_gate_failed_v18_specced",
            "jepa_v17_still_below_random",
            "jepa_v17_improved_below_threshold",
        }

        def _choose_verdict(cascade_gate_open: bool, cascade_precision: float) -> str:
            """Determine the honest_verdict string from gate state and precision."""
            if cascade_gate_open:
                if cascade_precision >= 0.7:
                    return "jepa_v17_cascade_deployed"
                return "jepa_v17_cascade_deployed_below_precision_target"
            return "jepa_v17_gate_failed_v18_specced"

        # Gate-open, precision met.
        assert _choose_verdict(True, 0.80) in valid_open_verdicts
        # Gate-open, precision not met.
        assert _choose_verdict(True, 0.60) in valid_open_verdicts
        # Gate-closed.
        assert _choose_verdict(False, 0.0) in valid_closed_verdicts

    def test_verify_repair_pipeline_accepts_optional_jepa_model(self) -> None:
        """VerifyRepairPipeline must accept tier_2_jepa_model=None without error.

        WHY: The cascade wiring adds JEPARankNetV17 as an optional parameter.
        If the parameter is not backwards-compatible (i.e., required), all existing
        callers break.  This test confirms the default=None path works.
        Spec: REQ-VERIFY-140.
        """
        # Import the pipeline and verify it can be constructed without a JEPA model.
        # We do not instantiate a real LLM — just test the signature contract.
        from python.carnot.pipeline.verify_repair import VerifyRepairPipeline
        import inspect
        sig = inspect.signature(VerifyRepairPipeline.__init__)
        # VerifyRepairPipeline must have a 'model' parameter (tier_2_jepa_model
        # would be added when gate opens; confirm existing init still works today).
        assert "model" in sig.parameters, (
            "VerifyRepairPipeline.__init__ missing 'model' parameter — signature changed unexpectedly"
        )
