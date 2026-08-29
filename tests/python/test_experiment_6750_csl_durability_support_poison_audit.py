"""Tests for Exp6750 cold CSL durability and poison audit.

Spec refs: REQ-CL-6750, SCENARIO-CL-6750-COLD-RECOMPUTE,
SCENARIO-CL-6750-CHRONOLOGY, SCENARIO-CL-6750-POISON,
SCENARIO-CL-6750-RESTART, SCENARIO-CL-6750-ROLLBACK.
"""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from carnot import experiment_6750_csl_durability_support_poison_audit as mod
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
EXP6749 = REPO / mod.EXP6749_RELATIVE_PATH
FIXTURE = REPO / mod.EXP6748_RELATIVE_PATH
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _inputs() -> tuple[mod.JsonDict, mod.JsonDict]:
    return (
        json.loads(EXP6749.read_text(encoding="utf-8")),
        json.loads(FIXTURE.read_text(encoding="utf-8")),
    )


def _artifact(tmp_path: Path) -> mod.JsonDict:
    return mod.build_artifact(
        root=REPO,
        exp6749_path=EXP6749,
        fixture_path=FIXTURE,
        state_root=tmp_path / "audit-state",
        duration_s=0.25,
        tests_run=mod.DEFAULT_TESTS_RUN,
    )


def test_req_cl_6750_spec_declares_cold_audit_contract() -> None:
    """REQ-CL-6750: OpenSpec owns the Exp6750 audit before code."""

    section = SPEC.read_text(encoding="utf-8").split("## REQ-CL-6750", 1)[1]
    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.SCRIPT_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "SCENARIO-CL-6750-COLD-RECOMPUTE",
        "SCENARIO-CL-6750-CHRONOLOGY",
        "SCENARIO-CL-6750-POISON",
        "SCENARIO-CL-6750-RESTART",
        "SCENARIO-CL-6750-ROLLBACK",
        "complete_blocked_csl_audit",
    ):
        assert marker in section


def test_scenario_cl_6750_cold_recompute_uses_raw_rows(tmp_path: Path) -> None:
    """SCENARIO-CL-6750-COLD-RECOMPUTE: rows, not aggregates, drive output."""

    source = inspect.getsource(mod)
    forbidden_imports = (
        "from carnot import experiment_6749",
        "import carnot.experiment_6749",
    )
    assert not any(pattern in source for pattern in forbidden_imports)

    csl, fixture = _inputs()
    artifact = _artifact(tmp_path)
    recomputed = mod.recompute_prospective_metrics(csl["rows"])
    aggregate_tampered = deepcopy(csl)
    aggregate_tampered["best_at_k_by_arm"]["transactional_memory"]["rate"] = 1.0

    assert artifact["recomputed_prequential_delta_by_order"] == recomputed[
        "recomputed_prequential_delta_by_order"
    ]
    assert mod.recompute_prospective_metrics(aggregate_tampered["rows"]) == recomputed
    assert len(csl["rows"]) == 288
    assert len(fixture["stream_manifest"]["orders"]) == 6
    assert artifact["order_level_ci95"]["lower"] == 0.0
    assert artifact["order_level_ci95"]["upper"] == 0.0
    assert artifact["support_contraction_by_metric"]["best_at_k"]["passes"] is True
    assert artifact["commit_reject_rollback_counts"]["prospective_rows"]["commits"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["csl_audit_passed"] is False
    assert "order_lcb_positive" in artifact["gate_check_summary"]["failed_checks"]
    assert mod.validate_artifact(artifact) == []

    missing_pair_rows = [
        row
        for row in csl["rows"]
        if not (
            row["event_kind"] == "retention_anchor"
            and row["arm"] == "no_memory"
            and row["order_id"] == "order_1"
        )
    ]
    assert mod.recompute_prospective_metrics(missing_pair_rows)["retention_failures"] == []

    failing_rows = deepcopy(csl["rows"])
    target = next(
        row
        for row in failing_rows
        if row["event_kind"] == "retention_anchor"
        and row["arm"] == "transactional_memory"
        and row["model_id"] == "unsloth/gemma-4-31B-it-GGUF"
    )
    target["pass_at_1"] = 0
    target["best_at_k"] = 0
    target["joint_correct_constraint_support"] = 0.0
    assert mod.recompute_prospective_metrics(failing_rows)["retention_failures"]


def test_scenario_cl_6750_chronology_denies_future_and_opposite_arm() -> None:
    """SCENARIO-CL-6750-CHRONOLOGY: future and opposite-arm evidence fail."""

    csl, _fixture = _inputs()
    clean = mod.audit_snapshot_isolation(csl["rows"], csl["frozen_protocol"]["orders"])
    assert clean["future_leakage_count"] == 0
    assert clean["snapshot_row_count"] == 144
    assert all(row["passed"] is True for row in clean["rows"])

    poisoned_rows = deepcopy(csl["rows"])
    target = next(row for row in poisoned_rows if row["arm"] == "transactional_memory")
    target["snapshot_records"] = [
        {"source_event_id": "e12", "family": "safety", "arm": "no_memory"}
    ]
    poisoned = mod.audit_snapshot_isolation(poisoned_rows, csl["frozen_protocol"]["orders"])

    assert poisoned["future_leakage_count"] == 1
    assert poisoned["rows"][0]["future_evidence_count"] == 1
    assert poisoned["rows"][0]["opposite_arm_evidence_count"] == 1
    assert poisoned["rows"][0]["held_family_evidence_count"] == 0


def test_scenario_cl_6750_poison_restart_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-CL-6750-POISON/RESTART/ROLLBACK: copied state fails closed."""

    _csl, fixture = _inputs()
    attacks = mod.replay_poison_attacks(fixture, tmp_path / "attacks")
    provenance = mod.audit_commit_provenance(fixture)
    restarts = mod.audit_restart_boundaries(fixture)
    rollback = mod.audit_rollback_identity(fixture)

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["admitted_poison_count"] == 0
    assert attacks["unsafe_use_count"] == 0
    assert all(row["passed"] is True for row in attacks["rows"])
    assert provenance["all_hashes_match"] is True
    assert provenance["commit_receipt_count"] == 18
    assert restarts["pass_count"] == restarts["expected_count"] == 74
    assert rollback["all_match"] is True
    assert rollback["boundary_count"] == 18
    assert all(row["byte_identical"] is True for row in rollback["rows"])


def test_req_cl_6750_blocked_preconditions_are_terminal(tmp_path: Path) -> None:
    """REQ-CL-6750: missing completion or receipts emits complete blocked."""

    csl, fixture = _inputs()
    csl["prospective_csl_completed"] = False
    fixture["commit_receipts"] = []
    csl_path = tmp_path / "blocked-csl.json"
    fixture_path = tmp_path / "blocked-fixture.json"
    csl_path.write_text(json.dumps(csl), encoding="utf-8")
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    artifact = mod.build_artifact(
        root=REPO,
        exp6749_path=csl_path,
        fixture_path=fixture_path,
        state_root=tmp_path / "state",
        duration_s=0.1,
        tests_run=mod.DEFAULT_TESTS_RUN,
    )

    assert artifact["status"] == "complete_blocked_csl_audit"
    assert artifact["honest_verdict"].startswith("complete_blocked_csl_audit:")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["failed_checks"] == [
        "exp6749_completed",
        "commit_receipts_present",
    ]
    assert mod.validate_artifact(artifact) == []

    bad_blocked = deepcopy(artifact)
    bad_blocked["verdict_class"] = "null"
    bad_blocked["reproducibility_checksum"] = mod.reproducibility_checksum(bad_blocked)
    assert "blocked verdict_class mismatch" in mod.validate_artifact(bad_blocked)


def _passing_metrics() -> mod.JsonDict:
    support = {
        metric: {
            "no_memory": {"numerator": 1, "denominator": 6, "rate": 0.166667},
            "transactional_memory": {"numerator": 2, "denominator": 6, "rate": 0.333333},
            "contraction": -0.166666,
            "allowed_contraction_bound": mod.SUPPORT_CONTRACTION_BOUND,
            "passes": True,
        }
        for metric in (
            "pass_at_1",
            "best_at_k",
            "effective_rewardable_support",
            "joint_correct_constraint_support",
        )
    }
    return {
        "recomputed_prequential_delta_by_order": {
            f"order_{index}": {
                "pooled": {
                    "transactional_minus_no_memory": 0.1,
                    "no_memory": {"rate": 0.2},
                    "transactional_memory": {"rate": 0.3},
                },
                "by_model": {},
            }
            for index in range(1, 7)
        },
        "order_delta_values": [0.1] * 6,
        "support_contraction_by_metric": support,
        "retention_rows": [],
        "retention_failures": [],
        "negative_transfer_by_family": {
            "model": {
                "family": {
                    "transactional_minus_no_memory": 0.0,
                    "negative_transfer": False,
                }
            }
        },
        "token_cost_by_model_arm": {
            "model": {
                "no_memory": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "transactional_memory": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        },
        "commit_reject_rollback_counts": {
            "prospective_rows": {
                "commits": 1,
                "rejects": 1,
                "quarantine": 0,
                "rollbacks": 1,
                "rollback_failures": 0,
            },
            "by_order": {},
            "commit_status_counts": {"committed": 1, "rejected": 1},
        },
    }


def test_req_cl_6750_positive_and_disqualified_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6750: terminal class follows eligibility and integrity gates."""

    monkeypatch.setattr(mod, "recompute_prospective_metrics", lambda _rows: _passing_metrics())
    positive = mod.build_artifact(
        root=REPO,
        exp6749_path=EXP6749,
        fixture_path=FIXTURE,
        state_root=tmp_path / "positive",
        duration_s=0.25,
        tests_run=mod.DEFAULT_TESTS_RUN,
    )
    assert positive["verdict_class"] == "positive"
    assert positive["csl_audit_passed"] is True
    assert positive["honest_verdict"].startswith("complete_positive_csl_audit:")

    monkeypatch.setattr(
        mod,
        "audit_snapshot_isolation",
        lambda _rows, _orders: {
            "snapshot_row_count": 1,
            "future_leakage_count": 1,
            "rows": [
                {
                    "row_type": "snapshot",
                    "predates_event": False,
                    "future_evidence_count": 1,
                    "held_family_evidence_count": 0,
                    "opposite_arm_evidence_count": 0,
                    "passed": False,
                }
            ],
        },
    )
    disqualified = mod.build_artifact(
        root=REPO,
        exp6749_path=EXP6749,
        fixture_path=FIXTURE,
        state_root=tmp_path / "disqualified",
        duration_s=0.25,
        tests_run=mod.DEFAULT_TESTS_RUN,
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["honest_verdict"].startswith("complete_disqualified_csl_audit:")


def test_req_cl_6750_validation_checksum_and_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-6750: validation and CLI protect the terminal artifact."""

    artifact = _artifact(tmp_path)
    auto_state = mod.build_artifact(
        root=REPO,
        exp6749_path=EXP6749,
        fixture_path=FIXTURE,
        state_root=None,
        duration_s=0.25,
        tests_run=mod.DEFAULT_TESTS_RUN,
    )
    assert auto_state["status"] == artifact["status"]

    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="json object required"):
        mod.read_json(array_json)

    path = tmp_path / "artifact.json"
    receipt = mod.write_artifact(path, artifact)

    assert receipt["atomic_rename"] is True
    assert receipt["path"] == str(path)
    assert mod.main(["--validate", "--result-path", str(path)]) == 0

    mutations = [
        ("required field set mismatch", lambda data: data.pop("rows")),
        ("field_principles coverage mismatch", lambda data: data["field_principles"].pop("status")),
        ("inference_substrate mismatch", lambda data: data.__setitem__("inference_substrate", "wrong")),
        ("verdict_class outside closed enum", lambda data: data.__setitem__("verdict_class", "maybe")),
        ("csl_audit_passed mismatch", lambda data: data.__setitem__("csl_audit_passed", True)),
        ("future_leakage_count mismatch", lambda data: data.__setitem__("future_leakage_count", 3)),
        ("admitted_poison_count mismatch", lambda data: data.__setitem__("admitted_poison_count", 2)),
        ("reproducibility_checksum mismatch", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
    ]
    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum mismatch":
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        assert expected in mod.validate_artifact(bad)

    bad_path = tmp_path / "bad.json"
    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        mod.main(["--validate", "--result-path", str(bad_path)])
    with pytest.raises(ValueError, match="reproducibility_checksum mismatch"):
        mod.write_artifact(tmp_path / "bad-write.json", bad)

    monkeypatch.setattr(mod, "validate_artifact", lambda _payload: ["forced validation failure"])
    with pytest.raises(ValueError, match="forced validation failure"):
        mod.build_artifact(
            root=REPO,
            exp6749_path=EXP6749,
            fixture_path=FIXTURE,
            state_root=tmp_path / "forced",
            duration_s=0.25,
            tests_run=mod.DEFAULT_TESTS_RUN,
        )
    monkeypatch.undo()

    output = tmp_path / "main.json"
    writes: list[Path] = []
    monkeypatch.setattr(
        mod,
        "write_artifact",
        lambda target, payload: writes.append(Path(target)) or {"atomic_rename": True},
    )
    assert mod.main(["--result-path", str(output), "--state-root", str(tmp_path / "run")]) == 0
    assert writes == [output]


def test_req_cl_6750_substrate_has_reviewed_duration_floor() -> None:
    """REQ-CL-6750: the no-LLM audit substrate has a verifier floor."""

    payload = {
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "duration_s": 0.01,
        "honest_verdict": "complete_null_csl_audit: no positive CSL effect",
    }
    flags: list[adversarial_verify.Flag] = []
    adversarial_verify.check_duration_vs_claim(payload, flags)

    assert adversarial_verify.duration_floor_for_artifact(payload) == {
        "substrate": mod.INFERENCE_SUBSTRATE,
        "min_duration_s": 0.0001,
        "reason": "deterministic_verifier",
    }
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in {flag.kind for flag in flags}
