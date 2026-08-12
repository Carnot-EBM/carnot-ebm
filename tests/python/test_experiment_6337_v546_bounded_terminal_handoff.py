"""Tests for Exp6337 V546 bounded terminal handoff.

Spec refs: REQ-INFRA-6337, SCENARIO-INFRA-6337-1,
SCENARIO-INFRA-6337-2, SCENARIO-INFRA-6337-3,
SCENARIO-INFRA-6337-4, SCENARIO-INFRA-6337-5,
SCENARIO-INFRA-6337-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6337_v546_bounded_terminal_handoff as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def test_req_infra_6337_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6337: OpenSpec records the bounded handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6337") :]

    for token in (
        "REQ-INFRA-6337",
        "SCENARIO-INFRA-6337-1",
        "SCENARIO-INFRA-6337-2",
        "SCENARIO-INFRA-6337-3",
        "SCENARIO-INFRA-6337-4",
        "SCENARIO-INFRA-6337-5",
        "SCENARIO-INFRA-6337-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "llm_call_count",
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenarios_6337_preserve_missing_exp6323_and_terminal_v545_paths() -> None:
    """SCENARIO-INFRA-6337-1 and SCENARIO-INFRA-6337-2: evidence stays exact."""

    report = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert mod.validate_report(report) == []
    assert report["queued_v545_task_ids"] == list(mod.EXPECTED_QUEUED_V545_TASK_IDS)
    assert set(report["terminal_v545_artifacts_by_task"]) == set(
        mod.EXPECTED_QUEUED_V545_TASK_IDS[1:]
    )
    missing = report["missing_artifacts_by_task"]
    assert list(missing) == ["exp6323-v545-terminal-transition"]
    assert missing["exp6323-v545-terminal-transition"]["present"] is False
    assert missing["exp6323-v545-terminal-transition"]["honest_verdict_from_artifact"] is None

    terminal = report["terminal_v545_artifacts_by_task"]
    assert (
        terminal["exp6324-v545-post-marker-source-scope-freeze"]["honest_verdict_raw"]
    ).startswith("complete_null:")
    assert (
        terminal["exp6325-gatemate-dated-receipt-single-detect"]["honest_verdict_raw"]
    ).startswith("blocked_detect_failed:")
    assert terminal["exp6326-restricted-policy-contract-compiler"]["terminal_class"] == "ready"
    assert terminal["exp6328-blind-guard-integrity-audit"]["terminal_class"] == "ready"

    receipts = report["exp6323_failure_receipts"]
    assert receipts["count"] == 3
    assert receipts["hard_cap_seconds"] == [4802, 4802, 4803]
    assert all(row["status"] == "FAIL" for row in receipts["rows"])
    assert receipts["invented_honest_verdict"] is None


def test_scenarios_6337_proposal_only_ghost_ids_are_not_queued() -> None:
    """SCENARIO-INFRA-6337-3: Exp6330-Exp6336 stay proposal-only."""

    receipt = mod.proposal_only_exp6330_through_exp6336_receipt(REPO)

    assert receipt["ids"] == list(mod.PROPOSAL_ONLY_EXP_IDS)
    assert receipt["active_v546_id_reuse_count"] == 0
    assert receipt["conductor_task_row_count"] == 0
    assert receipt["v545_queue_size_receipt"]["queued_count"] == 7
    assert receipt["proposal_mentions_count"] >= 2
    assert receipt["old_transition_contract_mentions_count"] >= 7
    assert receipt["proposal_only"] is True


def test_scenarios_6337_v546_contracts_and_dirty_failures(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6337-4 and SCENARIO-INFRA-6337-5: V546 checks fail closed."""

    data, identity = mod.load_v546_roadmap(REPO)
    clean = mod.validate_v546_roadmap_data(
        data,
        retired_exp_ids=mod.load_retired_exp_ids(REPO / mod.EXCLUSION_MANIFEST_RELATIVE_PATH),
    )

    assert identity["milestone"] == mod.MILESTONE_V546
    assert identity["task_count"] == 13
    assert clean["schema_validation"]["ok"] is True
    assert clean["v546_id_collision_check"]["ok"] is True
    assert clean["v546_deliverable_checks"]["ok"] is True
    assert clean["v546_dependency_checks"]["ok"] is True
    assert clean["v546_structured_gate_checks"]["ok"] is True
    assert clean["v546_prior_failure_checks"]["ok"] is True
    assert clean["v546_llm_model_policy_checks"]["ok"] is True
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
    tasks[6]["prior_failures"] = []
    tasks[7]["agent_type"] = "codex"
    tasks[7]["model"] = "opus"
    tasks[8]["prompt"] = tasks[8]["prompt"].replace("MODEL_SPECS, ", "")
    tasks[8]["prompt"] = tasks[8]["prompt"].replace(
        "unsloth/Qwen3.6-35B-A3B-GGUF", "Bad/Unexpected-GGUF"
    )
    tasks[9]["prompt"] = tasks[9]["prompt"].replace(
        "Do NOT push. Do NOT modify scripts/research_conductor.py.",
        "Do NOT push.",
    )

    dirty_result = mod.validate_v546_roadmap_data(dirty, retired_exp_ids={9999})

    assert dirty_result["schema_validation"]["ok"] is False
    assert dirty_result["v546_id_collision_check"]["ok"] is False
    assert dirty_result["v546_deliverable_checks"]["ok"] is False
    assert dirty_result["v546_dependency_checks"]["ok"] is False
    assert dirty_result["v546_structured_gate_checks"]["ok"] is False
    assert dirty_result["v546_prior_failure_checks"]["ok"] is False
    assert dirty_result["v546_llm_model_policy_checks"]["ok"] is False
    assert dirty_result["prompt_contract_checks"]["ok"] is False
    assert any(
        row["reason"] == "empty_prior_failures"
        for row in dirty_result["v546_prior_failure_checks"]["failures"]
    )
    assert any(
        row["reason"] == "non_mandated_gguf_id"
        for row in dirty_result["v546_llm_model_policy_checks"]["model_policy_failures"]
    )

    block = mod._required_artifact_fields_block(
        "REQUIRED ARTIFACT FIELDS: status,\nMODEL_SPECS,\n\nCONCRETE STEPS"
    )
    assert "MODEL_SPECS" in block

    log_root = REPO / "does-not-exist"
    tmp_root = tmp_path / "log_edge"
    (tmp_root / "ops").mkdir(parents=True, exist_ok=True)
    (tmp_root / mod.CONDUCTOR_LOG_RELATIVE_PATH).write_text(
        "| 2026-08-12 00:00 UTC | Exact terminal-boundary handoff from V544 into V54 | FAIL | no hard cap |\n",
        encoding="utf-8",
    )
    assert mod.exp6323_failure_receipts(tmp_root)["count"] == 0
    assert not log_root.exists()


def test_report_schema_write_and_validation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6337-6: output is validated, checksummed, and atomic."""

    report = mod.build_report(
        REPO,
        date="20260812",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        started_at=0.0,
    )

    assert report["status"] == "complete_with_missing"
    assert report["llm_call_count"] == 0
    assert report["verifier_is_oracle"] is False
    assert report["repository_validator_checks"]["ok"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert report["protected_files_unchanged"]["unchanged"] is True
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)
    assert report["honest_verdict"].startswith("complete_with_missing:")

    bad = copy.deepcopy(report)
    bad["llm_call_count"] = {"value": 0}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "llm_call_count must be bare 0" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be false" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert any("missing field_principles entry" in err for err in mod.validate_report(bad))

    bad = copy.deepcopy(report)
    bad["missing_artifacts_by_task"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6323 missing artifact must be recorded" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["proposal_only_exp6330_through_exp6336_receipt"]["proposal_only"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6330-Exp6336 must be proposal-only" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["honest_verdict"] = "ok"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_report(bad)

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


def test_run_paths_and_external_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-INFRA-6337: CLI helpers preserve receipts and write only the result."""

    receipt_path = tmp_path / "receipts.json"
    monkeypatch.setattr(mod, "EXTERNAL_TEST_RECEIPT_PATH", receipt_path)
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    receipt_path.write_text(json.dumps({"focused": 0, "broad": 3}), encoding="utf-8")
    assert mod.read_external_test_receipts() == [
        {"command": "focused", "exit_code": 0},
        {"command": "broad", "exit_code": 3},
    ]

    receipt_path.write_text("{bad", encoding="utf-8")
    assert mod.read_external_test_receipts() == [{"command": mod.RUN_COMMAND, "exit_code": 0}]

    writes: list[dict[str, object]] = []

    def fake_write_report(
        report: dict[str, object], root: Path = REPO, *, env: object = None
    ) -> Path:
        writes.append(report)
        return tmp_path / mod.RESULT_RELATIVE_PATH.name

    original_write_report = mod.write_report
    monkeypatch.setattr(mod, "write_report", fake_write_report)
    run_report = mod.run(
        date="20260812",
        root=REPO,
        write=True,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert writes and run_report["status"] == "complete_with_missing"

    no_write_report = mod.run(
        date="20260812",
        root=REPO,
        write=False,
        command_receipts=[{"command": "focused", "exit_code": 0}],
    )
    assert no_write_report["status"] == "complete_with_missing"

    with pytest.raises(ValueError, match="invalid Exp6337 report"):
        original_write_report({"status": "complete"}, REPO)
