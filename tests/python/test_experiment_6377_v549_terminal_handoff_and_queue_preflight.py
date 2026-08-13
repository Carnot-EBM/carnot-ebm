"""Tests for Exp6377 V549 terminal handoff.

Spec refs: REQ-INFRA-6377, SCENARIO-INFRA-6377-1,
SCENARIO-INFRA-6377-2, SCENARIO-INFRA-6377-3,
SCENARIO-INFRA-6377-4, SCENARIO-INFRA-6377-5.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6377_v549_terminal_handoff_and_queue_preflight as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _report() -> dict[str, object]:
    return mod.build_report(
        REPO,
        date="20260813",
        command_receipts=[{"command": "focused", "exit_code": 0}],
        before_hashes=mod.protected_hashes(REPO),
        duration_s=1.0,
    )


def test_req_infra_6377_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6377: OpenSpec records the V549 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6377") : text.index("REQ-INFRA-6365")]

    for marker in (
        "SCENARIO-INFRA-6377-1",
        "SCENARIO-INFRA-6377-2",
        "SCENARIO-INFRA-6377-3",
        "SCENARIO-INFRA-6377-4",
        "SCENARIO-INFRA-6377-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6377_v548_boundary_and_proposal_only_ids() -> None:
    """SCENARIO-INFRA-6377-1: V548 denominator and states stay exact."""

    report = _report()

    assert mod.validate_report(report) == []
    assert report["v548_active_task_ids"] == list(mod.ACTIVE_V548_TASK_IDS)

    states = report["v548_blocked_null_clean_and_proposal_only_states"]
    assert states["blocked"] == ["exp6363-v548-terminal-handoff-and-queue-preflight"]
    assert states["null"] == ["exp6366-repaired-live-factor-proposal-authenticity"]
    assert states["clean"] == [
        "exp6364-v548-post-marker-source-scope-freeze",
        "exp6365-gguf-child-failure-forensics-runtime-contract",
    ]
    assert states["missing_active"] == []
    assert states["proposal_only"]["task_ids"] == list(mod.PROPOSAL_ONLY_V548_IDS)
    assert states["proposal_only"]["executed_count"] == 0
    assert states["proposal_only"]["counted_as_missing"] is False

    rows = report["v548_terminal_artifacts_by_task"]
    assert set(rows) == set(mod.ACTIVE_V548_TASK_IDS)
    for task_id, row in rows.items():
        assert row["summary_receipt"]["invoked_before_field_import"] is True
        assert row["summary_receipt"]["exit_code"] == 0
        assert row["task_id"] == task_id

    outcomes = report["v548_conductor_outcomes"]
    assert outcomes["exp6363-v548-terminal-handoff-and-queue-preflight"]["FAIL"] == 1
    assert outcomes["exp6363-v548-terminal-handoff-and-queue-preflight"]["OK"] == 1
    assert outcomes["exp6364-v548-post-marker-source-scope-freeze"]["OK"] == 1
    assert outcomes["exp6365-gguf-child-failure-forensics-runtime-contract"]["OK"] == 1
    assert outcomes["exp6366-repaired-live-factor-proposal-authenticity"]["OK"] == 1


def test_scenario_infra_6377_runtime_and_transport_boundaries() -> None:
    """SCENARIO-INFRA-6377-2: runtime success does not become transport success."""

    report = _report()
    runtime = report["exp6365_runtime_boundary"]
    transport = report["exp6366_transport_failure_boundary"]

    assert runtime["gguf_runtime_observability_ready_score"] == 1.0
    assert runtime["all_three_mandated_models_used"] is True
    assert runtime["all_child_contracts_ok"] is True
    assert runtime["all_vram_rise_and_release_proved"] is True
    assert runtime["proposal_quality_claimed"] is False

    assert transport["repaired_live_factor_proposal_authenticity_ready_score"] == 0.0
    assert transport["total_raw_output_count"] == 3
    assert transport["total_raw_output_byte_count"] == 1991
    assert transport["parse_valid_count"] == 0
    assert transport["parse_invalid_count"] == 3
    assert transport["exact_checker_call_count"] == 0
    assert transport["hidden_state_access_count"] == 0
    assert transport["transport_ready"] is False


def test_scenario_infra_6377_v549_queue_preflight_passes() -> None:
    """SCENARIO-INFRA-6377-3 and 4: V549 IDs, gates, prompts, and models pass."""

    report = _report()

    assert report["v549_task_ids"] == list(mod.EXPECTED_V549_TASK_IDS)
    assert report["v549_id_and_deliverable_checks"]["ok"] is True
    assert report["v549_id_and_deliverable_checks"]["execution_order_ok"] is True
    assert report["v549_dependency_and_gate_checks"]["ok"] is True
    assert report["v549_dependency_and_gate_checks"]["gate_count"] == 11
    assert report["v549_gate_field_cross_reference_checks"]["ok"] is True
    assert report["v549_prior_failure_checks"]["ok"] is True
    assert report["v549_exclusion_manifest_checks"]["ok"] is True

    policy = report["v549_agent_model_and_llm_policy_checks"]
    assert policy["ok"] is True
    assert set(policy["llm_task_ids"]) == {
        "exp6379-canonical-factor-edit-transport-contract",
        "exp6380-three-family-canonical-factor-transport-canary",
        "exp6381-verified-frontier-live-factor-proposal-ab",
        "exp6382-chronological-verified-factor-self-learning",
        "exp6384-default-off-certified-factor-consumer-ab",
        "exp6388-arc-goal-evidence-response-calibration",
        "exp6389-arc-default-off-active-goal-shadow",
    }

    prompt_checks = report["prompt_contract_checks"]
    assert prompt_checks["ok"] is True
    assert prompt_checks["checked_task_count"] == 14
    assert prompt_checks["rendered_prompt_count"] == 14
    assert prompt_checks["raw_placeholder_contract_ok"] is True
    assert prompt_checks["failures"] == []


def test_scenario_infra_6377_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6377-5: output is annotated, checksummed, and atomic."""

    report = _report()

    assert report["status"] == "complete_v549_queue_preflight_passed"
    assert report["honest_verdict"].startswith("complete_v549_queue_preflight_passed:")
    assert report["active_roadmap_modified"] is False
    assert report["conductor_modified"] is False
    assert report["protected_files_unchanged"]["ok"] is True
    assert report["verifier_is_oracle"] is False
    assert report["random_seed"] is None
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    for expression in report["v549_dependency_and_gate_checks"]["structured_gate_expressions"]:
        assert expression in report["field_principles"]
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    bad = copy.deepcopy(report)
    del bad["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing required field: status" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "verifier_is_oracle must be false" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v548_blocked_null_clean_and_proposal_only_states"]["blocked"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6363 blocked state must be preserved" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v548_blocked_null_clean_and_proposal_only_states"]["null"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6366 null state must be preserved" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_principles must be a mapping" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    del bad["field_principles"]["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: status" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    gate_expression = bad["v549_dependency_and_gate_checks"]["structured_gate_expressions"][0]
    del bad["field_principles"][gate_expression]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert f"missing field_principles entry: {gate_expression}" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_provenance"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must be a mapping" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["random_seed"] = 6377
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "random_seed must be null" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v548_blocked_null_clean_and_proposal_only_states"]["proposal_only"][
        "executed_count"
    ] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "proposal-only V548 IDs must not be counted as executed" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["honest_verdict"] = "ok"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict lacks terminal prefix" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["prompt_contract_checks"]["ok"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "complete report has failed V549 checks" in mod.validate_report(bad)

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
    assert mod.main(["--date", "20260813"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_infra_6377_helper_edges_and_dirty_queue_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6377: helper edges fail closed without fabricated evidence."""

    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_mapping(bad_json)[1]["error"].startswith("json_error:")
    not_mapping = tmp_path / "array.json"
    not_mapping.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(not_mapping)[1]["error"] == "json_not_mapping"

    assert mod._summarize_artifact(tmp_path, "results/missing.json")["reason"] == "artifact_absent"
    assert mod._conductor_rows(tmp_path, mod.ACTIVE_V548_TASK_IDS[0]) == []
    log_dir = tmp_path / "ops"
    log_dir.mkdir()
    (log_dir / "conductor-log.md").write_text(
        "V547 terminal evidence handoff and V548 queue pref malformed\n", encoding="utf-8"
    )
    assert mod._conductor_rows(tmp_path, mod.ACTIVE_V548_TASK_IDS[0]) == []
    assert mod._risk_rows([object()])[0]["repr"].startswith("<object object")
    assert mod.v548_state_receipt({"task": {"terminal_class": "missing"}})["missing_active"] == [
        "task"
    ]
    (tmp_path / "research-roadmap.yaml").write_text("milestone: active\ntasks: []\n", encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text("milestone: next\ntasks: []\n", encoding="utf-8")
    _queue, identity = mod.load_v549_queue(tmp_path)
    assert identity["audited_queue"]["path"] == "research-roadmap-next.yaml"

    assert mod._tasks({"tasks": "not-list"}) == []
    rendered, receipt = mod.render_prompt("x {project_root} {date}", REPO, "20260813")
    assert REPO.as_posix() in rendered
    assert receipt["format_ok"] is True
    rendered, receipt = mod.render_prompt("x {broken} {project_root}", REPO, "20260813")
    assert "{broken}" in rendered and REPO.as_posix() in rendered
    assert receipt["format_ok"] is False

    data, _identity = mod.load_v549_queue(REPO)
    dirty = copy.deepcopy(data)
    tasks = dirty["tasks"]
    tasks[1]["id"] = tasks[0]["id"]
    tasks[2]["deliverable"] = "not-results.txt"
    tasks[3]["gated_on"] = [
        {
            "upstream": "exp2091-retired-upstream",
            "artifact_field": "not_declared",
            "op": "==",
            "value": 1.0,
        }
    ]
    tasks[4]["requires"] = [tasks[4]["id"], "exp2091-retired-upstream"]
    tasks[5]["prior_failures"] = [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
    tasks[6]["prior_failures"] = []
    tasks[6]["agent_type"] = "codex"
    tasks[6]["model"] = "opus"
    tasks[7]["agent_type"] = "gemini"
    tasks[8]["model"] = "haiku"
    tasks[2]["requires_gpu"] = True
    tasks[2]["prompt"] = (
        "CONTEXT\n"
        "{project_root}\n"
        "20260813\n"
        "TASK\n"
        "Run command: x\n"
        "MODEL_SPECS must include Bad/Unexpected-GGUF.\n"
        "AutoTokenizer.from_pretrained('Bad/Unexpected-GGUF')\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    tasks[3]["prompt"] = tasks[3]["prompt"].replace(
        "Do NOT push. Do NOT modify scripts/research_conductor.py.",
        "Do NOT push.",
    )
    tasks[3]["prompt"] = tasks[3]["prompt"].replace("{date}", "date")
    tasks[9]["requires_gpu"] = True
    tasks[9]["prompt"] = (
        "CONTEXT\n"
        "{project_root}\n"
        "{date}\n"
        "TASK\n"
        "CONCRETE STEPS\n"
        "Run command: x\n"
        "Models are declared without the required spec block.\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )

    checks = mod.validate_v549_queue_data(dirty, REPO, "20260813", retired_exp_ids={2091})
    assert checks["schema_validation"]["ok"] is False
    assert checks["v549_id_and_deliverable_checks"]["ok"] is False
    assert checks["v549_dependency_and_gate_checks"]["ok"] is False
    assert checks["v549_gate_field_cross_reference_checks"]["ok"] is False
    assert checks["v549_prior_failure_checks"]["ok"] is False
    assert checks["v549_exclusion_manifest_checks"]["ok"] is False
    assert checks["v549_agent_model_and_llm_policy_checks"]["ok"] is False
    assert checks["prompt_contract_checks"]["ok"] is False
    assert mod._test_rows(None)[0]["source"] == "declared"

    receipt = tmp_path / "receipts.json"
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text("{}", encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == []
    receipt.write_text('[{"command": "ok", "exit_code": 0}, "skip"]', encoding="utf-8")
    assert mod.read_external_test_receipts(receipt) == [{"command": "ok", "exit_code": 0}]

    writes: list[dict[str, object]] = []

    def fake_build_report(
        root: Path,
        *,
        date: str,
        command_receipts: list[dict[str, object]],
        before_hashes: dict[str, str],
        duration_s: float,
    ) -> dict[str, object]:
        return {
            "date": date,
            "command_receipts": command_receipts,
            "before_hashes": before_hashes,
            "duration_s": duration_s,
            "reproducibility_checksum": "sha256:fake",
        }

    monkeypatch.setattr(mod, "protected_hashes", lambda root: {"x": "sha256:x"})
    monkeypatch.setattr(mod, "read_external_test_receipts", lambda: [{"command": "external"}])
    monkeypatch.setattr(mod, "build_report", fake_build_report)
    monkeypatch.setattr(mod, "validate_report", lambda report: [])
    monkeypatch.setattr(mod, "write_report", lambda report, root: writes.append(report))

    report = mod.run(date="20260813", root=REPO, write=True)
    assert report["command_receipts"] == [{"command": "external"}]
    assert writes == [report]

    monkeypatch.setattr(mod, "validate_report", lambda report: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.run(date="20260813", root=REPO, write=False, command_receipts=[{"command": "c"}])
