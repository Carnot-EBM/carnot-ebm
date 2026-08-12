"""Tests for Exp6350 V547 bounded terminal handoff.

Spec refs: REQ-INFRA-6350, SCENARIO-INFRA-6350-1,
SCENARIO-INFRA-6350-2, SCENARIO-INFRA-6350-3,
SCENARIO-INFRA-6350-4, SCENARIO-INFRA-6350-5,
SCENARIO-INFRA-6350-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6350_v547_bounded_terminal_handoff as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _report() -> dict[str, object]:
    return mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )


def test_req_infra_6350_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6350: OpenSpec records the handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6350") :]

    for marker in (
        "SCENARIO-INFRA-6350-1",
        "SCENARIO-INFRA-6350-2",
        "SCENARIO-INFRA-6350-3",
        "SCENARIO-INFRA-6350-4",
        "SCENARIO-INFRA-6350-5",
        "SCENARIO-INFRA-6350-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "live_autoregressive_generation_by_task",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6350_v546_denominator_and_flagged_evidence() -> None:
    """SCENARIO-INFRA-6350-1 and 2: exact V546 rows and flags are preserved."""

    report = _report()

    assert mod.validate_report(report) == []
    assert report["queued_v546_task_ids"] == list(mod.EXPECTED_V546_TASK_IDS)
    assert set(report["terminal_v546_artifacts_by_task"]) == set(mod.EXPECTED_V546_TASK_IDS)
    assert list(report["blocked_v546_tasks"]) == ["exp6341-prospective-prefix-utility-ab"]

    matrix = report["terminal_v546_artifacts_by_task"]
    exp6337 = matrix["exp6337-v546-bounded-terminal-handoff"]
    assert exp6337["terminal_class"] == "flagged"
    assert exp6337["clean_promotion_attempted"] is False

    flagged = report["flagged_v546_artifacts_and_reasons"]
    assert list(flagged) == ["exp6337-v546-bounded-terminal-handoff"]
    flag_text = json.dumps(flagged["exp6337-v546-bounded-terminal-handoff"], sort_keys=True)
    assert "DURATION_TOO_SHORT" in flag_text
    assert "METHODOLOGY" in flag_text

    blocked = report["blocked_v546_tasks"]["exp6341-prospective-prefix-utility-ab"]
    assert blocked["status_raw"] == "blocked"
    assert blocked["explicit_gate_block"] is True
    assert blocked["missing_artifact"] is False


def test_scenario_infra_6350_substrate_and_boundary_receipts() -> None:
    """SCENARIO-INFRA-6350-3 and 5: substrate and science boundaries stay narrow."""

    report = _report()
    substrate = report["inference_substrate_classification_by_task"]
    live = report["live_autoregressive_generation_by_task"]

    assert substrate["exp6340-parser-jit-semantic-diversity-canary"]["class"] == (
        "live_autoregressive_generation"
    )
    assert substrate["exp6344-counterexample-factor-proposal-calibration"]["class"] == (
        "deterministic_replay_with_gguf_receipts"
    )
    assert substrate["exp6345-prospective-certified-factor-evolution-ab"]["class"] == (
        "tokenizer_only_exact_replay"
    )
    assert substrate["exp6348-arc-default-off-action-influence-ab"]["class"] == (
        "live_autoregressive_generation"
    )
    assert live["exp6344-counterexample-factor-proposal-calibration"]["invoked"] is False
    assert live["exp6345-prospective-certified-factor-evolution-ab"]["invoked"] is False
    assert live["exp6348-arc-default-off-action-influence-ab"]["invoked"] is True

    parser = report["closed_parser_jit_receipt"]
    assert parser["closed"] is True
    assert parser["exp6340_semantic_diversity_gain_score"] == 0.0
    assert parser["exp6341_gate_blocked"] is True

    learning = report["qualified_certified_learning_receipt"]
    assert learning["qualified_closed"] is True
    assert learning["qualification"] == "closed_inside_synthetic_and_deterministic_replay_bounds"
    assert learning["live_generation_claim"] is False

    gaps = report["open_live_generation_and_consumer_gaps"]
    assert gaps["live_factor_proposal_generation_open"] is True
    assert gaps["future_consumer_value_open"] is True

    arc = report["arc_no_solve_receipt"]
    assert arc["solve_claim_count"] == 0
    assert arc["no_solve_boundary_preserved"] is True


def test_scenario_infra_6350_v547_roadmap_checks_fail_closed() -> None:
    """SCENARIO-INFRA-6350-4: dirty V547 identities and contracts fail closed."""

    data, identity = mod.load_v547_roadmap(REPO)
    clean = mod.validate_v547_roadmap_data(
        data,
        retired_exp_ids=mod.load_retired_exp_ids(REPO / mod.EXCLUSION_MANIFEST_RELATIVE_PATH),
    )

    assert identity["milestone"] == mod.MILESTONE_V547
    assert identity["task_count"] == len(mod.EXPECTED_V547_TASK_IDS)
    assert identity["proposal_task_count"] == len(mod.EXPECTED_V547_PROPOSAL_TASK_IDS)
    assert clean["v547_id_collision_check"]["ok"] is True
    assert clean["v547_deliverable_checks"]["ok"] is True
    assert clean["v547_dependency_checks"]["ok"] is True
    assert clean["v547_structured_gate_checks"]["ok"] is True
    assert clean["v547_prior_failure_checks"]["ok"] is True
    assert clean["v547_llm_model_policy_checks"]["ok"] is True
    assert clean["prompt_contract_checks"]["ok"] is True

    dirty = copy.deepcopy(data)
    tasks = dirty["tasks"]
    tasks[1]["id"] = tasks[0]["id"]
    tasks[2]["deliverable"] = "not-results.txt"
    tasks[3]["requires"] = [tasks[3]["id"], "exp9999-missing"]
    tasks[4]["gated_on"] = [
        {
            "upstream": tasks[0]["id"],
            "artifact_field": "not_declared",
            "op": "==",
            "value": 1.0,
        }
    ]
    tasks[5]["prior_failures"] = [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
    tasks[6]["agent_type"] = "codex"
    tasks[6]["model"] = "opus"
    tasks[2]["prompt"] = tasks[2]["prompt"].replace("MODEL_SPECS, ", "")
    tasks[2]["prompt"] = tasks[2]["prompt"].replace(
        "unsloth/Qwen3.6-35B-A3B-GGUF", "Bad/Unexpected-GGUF"
    )
    tasks[3]["prompt"] = tasks[3]["prompt"].replace(
        "Do NOT push. Do NOT modify scripts/research_conductor.py.",
        "Do NOT push.",
    )

    dirty_result = mod.validate_v547_roadmap_data(dirty, retired_exp_ids={9999})

    assert dirty_result["schema_validation"]["ok"] is False
    assert dirty_result["v547_id_collision_check"]["ok"] is False
    assert dirty_result["v547_deliverable_checks"]["ok"] is False
    assert dirty_result["v547_dependency_checks"]["ok"] is False
    assert dirty_result["v547_structured_gate_checks"]["ok"] is False
    assert dirty_result["v547_prior_failure_checks"]["ok"] is False
    assert dirty_result["v547_llm_model_policy_checks"]["ok"] is False
    assert dirty_result["prompt_contract_checks"]["ok"] is False
    assert any(
        row["reason"] == "non_mandated_gguf_id"
        for row in dirty_result["v547_llm_model_policy_checks"]["model_policy_failures"]
    )


def test_scenario_infra_6350_report_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6350-6: output is annotated, checksummed, and atomic."""

    report = _report()

    assert report["status"] == "complete_with_flagged_boundaries"
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert report["llm_call_count"] == 0
    assert report["verifier_is_oracle"] is False
    assert report["random_seeds"]["used"] == []
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["honest_verdict"].startswith("complete_with_flagged_boundaries:")

    bad = copy.deepcopy(report)
    bad["llm_call_count"] = {"value": 0}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "llm_call_count must be bare 0" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be false" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["flagged_v546_artifacts_and_reasons"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6337 flag must be preserved" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["inference_substrate_classification_by_task"][
        "exp6345-prospective-certified-factor-evolution-ab"
    ]["class"] = "live_autoregressive_generation"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6345 must not be live autoregressive generation" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["honest_verdict"] = "ok"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert any("missing field_principles entry" in err for err in mod.validate_report(bad))

    bad = copy.deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_report(bad)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=REPO, write=True, command_receipts=None: {
            "status": f"complete-{date}"
        },
    )
    assert mod.main(["--date", "20260812"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_infra_6350_helper_edges_and_external_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6350: helper edges fail closed without fabricating evidence."""

    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    malformed = tmp_path / "bad.json"
    malformed.write_text("{bad", encoding="utf-8")
    assert str(mod.read_json_mapping(malformed)[1]["error"]).startswith("json_error:")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(scalar)[1]["error"] == "json_not_mapping"

    assert mod._roadmap_tasks({"tasks": "bad"}) == []
    assert mod.classify_substrate({}, "x")["class"] == "unknown_missing_payload"
    assert mod.classify_substrate(
        {"inference_substrate": "deterministic_synthetic_evalue_replay_exact_oracle_no_llm"},
        "x",
    )["class"] == "synthetic_replay"

    terminal, blocked, missing = mod.classify_v546_evidence(tmp_path)
    assert terminal == {}
    assert blocked == {}
    assert set(missing) == set(mod.EXPECTED_V546_TASK_IDS)

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]
    receipt_path.write_text(json.dumps({"cmd": 7}), encoding="utf-8")
    assert mod.read_external_test_receipts() == [{"command": "cmd", "exit_code": 7}]
    receipt_path.write_text("{bad", encoding="utf-8")
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    writes: list[dict[str, object]] = []

    def fake_write_report(
        report: dict[str, object], root: Path = REPO, *, env: object = None
    ) -> Path:
        writes.append(report)
        return tmp_path / mod.RESULT_RELATIVE_PATH.name

    monkeypatch.setattr(mod, "write_report", fake_write_report)
    report = mod.run(
        date="20260812",
        root=REPO,
        write=True,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert writes and report["status"] == "complete_with_flagged_boundaries"
