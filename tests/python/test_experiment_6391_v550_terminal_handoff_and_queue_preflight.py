"""Tests for Exp6391 V550 terminal handoff.

Spec refs: REQ-INFRA-6391, SCENARIO-INFRA-6391-1,
SCENARIO-INFRA-6391-2, SCENARIO-INFRA-6391-3,
SCENARIO-INFRA-6391-4, SCENARIO-INFRA-6391-5,
SCENARIO-INFRA-6391-6.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from carnot import experiment_6391_v550_terminal_handoff_and_queue_preflight as mod
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


def test_req_infra_6391_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6391: OpenSpec owns the V550 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6391") : text.index("REQ-INFRA-6379")]

    for marker in (
        "SCENARIO-INFRA-6391-1",
        "SCENARIO-INFRA-6391-2",
        "SCENARIO-INFRA-6391-3",
        "SCENARIO-INFRA-6391-4",
        "SCENARIO-INFRA-6391-5",
        "SCENARIO-INFRA-6391-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6391_v549_evidence_stays_separate() -> None:
    """SCENARIO-INFRA-6391-1: V549 terminal classes are not promoted."""

    report = _report()

    assert mod.validate_report(report) == []
    assert report["v549_task_ids"] == list(mod.EXPECTED_V549_TASK_IDS)

    rows = report["v549_terminal_artifacts_by_task"]
    assert rows["exp6377-v549-terminal-handoff-and-queue-preflight"]["terminal_class"] == "flagged"
    assert (
        rows["exp6380-three-family-canonical-factor-transport-canary"]["terminal_class"] == "null"
    )
    assert rows["exp6381-verified-frontier-live-factor-proposal-ab"]["terminal_class"] == "blocked"
    assert rows["exp6382-chronological-verified-factor-self-learning"]["terminal_class"] == "absent"
    assert (
        rows["exp6385-live-factor-learning-and-rollback-safety-audit"]["terminal_class"]
        == "flagged"
    )

    verdicts = report["v549_artifact_verdicts"]
    assert verdicts["exp6380-three-family-canonical-factor-transport-canary"].startswith(
        "complete_null:"
    )
    assert verdicts["exp6382-chronological-verified-factor-self-learning"] is None
    assert verdicts["exp6389-arc-default-off-active-goal-shadow"] == "blocked_gate_check_failed"

    conductor = report["v549_conductor_outcomes"]
    assert conductor["exp6377-v549-terminal-handoff-and-queue-preflight"]["FLAGGED"] == 1
    assert conductor["exp6385-live-factor-learning-and-rollback-safety-audit"]["FLAGGED"] == 1
    assert conductor["exp6380-three-family-canonical-factor-transport-canary"]["OK"] == 1

    flags = report["v549_adversarial_flags"]
    assert (
        flags["exp6377-v549-terminal-handoff-and-queue-preflight"]["stamped_flagged_adversarial"]
        is True
    )
    assert flags["exp6377-v549-terminal-handoff-and-queue-preflight"]["live_has_critical"] is True
    assert (
        flags["exp6385-live-factor-learning-and-rollback-safety-audit"]["live_has_critical"] is True
    )
    assert flags["exp6382-chronological-verified-factor-self-learning"]["live_verdict"] == "absent"

    durations = report["v549_duration_receipts_by_task"]
    assert durations["exp6380-three-family-canonical-factor-transport-canary"][
        "duration_s"
    ] == pytest.approx(225.21800398197956)
    assert (
        durations["exp6382-chronological-verified-factor-self-learning"]["source"]
        == "artifact_absent"
    )
    assert durations["exp6390-v549-adversarial-capstone"]["source"] == "artifact.duration_s"


def test_scenario_infra_6391_factor_and_arc_boundaries() -> None:
    """SCENARIO-INFRA-6391-2 and 3: factor and ARC boundaries stay exact."""

    report = _report()
    factor = report["v549_factor_boundary"]
    arc = report["v549_arc_boundary"]

    assert factor["exp6379_ready"] is True
    assert factor["exp6380_global_null"] is True
    assert factor["qualified_gemma_observation_count"] == 2
    assert factor["qualified_gemma_models"] == [
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    assert factor["qwen_invalid"] is True
    assert factor["exp6381_blocked"] is True
    assert factor["exp6382_blocked_or_absent"] is True
    assert factor["exp6383_positive_control"] is True
    assert factor["exp6384_blocked"] is True
    assert factor["global_transport_promoted"] is False

    assert arc["exp6386_ready"] is True
    assert arc["exp6387_ready"] is True
    assert arc["exp6388_ready"] is True
    assert arc["exp6388_delta_admission_precision_shape"] == "dict"
    assert arc["exp6388_delta_admission_precision_pooled_unrounded"] == pytest.approx(0.75)
    assert arc["exp6389_honest_verdict"] == "blocked_gate_check_failed"
    assert arc["exp6389_blocked_gate_check_failed"] is True


def test_scenario_infra_6391_v550_queue_prompt_and_model_policy() -> None:
    """SCENARIO-INFRA-6391-4 and 5: V550 queue and prompt policy pass."""

    report = _report()

    assert report["v550_task_ids"] == list(mod.EXPECTED_V550_TASK_IDS)
    assert report["v550_id_and_deliverable_checks"]["ok"] is True
    assert report["v550_id_and_deliverable_checks"]["execution_order_ok"] is True
    assert report["v550_dependency_and_gate_checks"]["ok"] is True
    assert report["v550_dependency_and_gate_checks"]["gate_count"] == 14
    assert report["v550_gate_field_cross_reference_checks"]["ok"] is True
    assert report["v550_prior_failure_checks"]["ok"] is True
    assert report["v550_exclusion_manifest_checks"]["ok"] is True

    policy = report["v550_agent_model_and_llm_policy_checks"]
    assert policy["ok"] is True
    assert policy["all_tasks_codex_gpt55"] is True
    assert policy["llm_task_ids"] == list(mod.EXPECTED_V550_LLM_TASK_IDS)
    assert policy["model_policy_failures"] == []

    prompts = report["prompt_contract_checks"]
    assert prompts["ok"] is True
    assert prompts["checked_task_count"] == 13
    assert prompts["raw_placeholder_contract_ok"] is True
    assert prompts["failures"] == []


def test_scenario_infra_6391_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6391-6: artifact schema is stable and atomic."""

    report = _report()

    assert report["status"] == "complete_v550_queue_preflight_passed"
    assert report["honest_verdict"].startswith("complete_v550_queue_preflight_passed:")
    assert report["active_roadmap_modified"] is False
    assert report["conductor_modified"] is False
    assert report["protected_files_unchanged"]["ok"] is True
    assert report["verifier_is_oracle"] is False
    assert report["random_seed"] is None
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert set(report["field_provenance"].values()) <= {
        "measured",
        "derived",
        "constant",
        "upstream",
    }
    for expression in report["v550_dependency_and_gate_checks"]["structured_gate_expressions"]:
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
    bad["v549_factor_boundary"]["exp6380_global_null"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6380 global null boundary must be preserved" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v549_arc_boundary"]["exp6388_delta_admission_precision_shape"] = "float"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6388 nested ARC metric boundary must be preserved" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v549_factor_boundary"]["qualified_gemma_observation_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "two Gemma qualified observations must be preserved" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v549_arc_boundary"]["exp6389_blocked_gate_check_failed"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "Exp6389 blocked gate verdict must be preserved" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["protected_files_unchanged"]["ok"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "protected files changed" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_principles must be a mapping" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    del bad["field_principles"]["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: status" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    gate_expression = bad["v550_dependency_and_gate_checks"]["structured_gate_expressions"][0]
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
    bad["field_provenance"] = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, "bad_kind")
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance has invalid classification" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["random_seed"] = 6391
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "random_seed must be null" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["prompt_contract_checks"]["ok"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "complete report has failed V550 checks" in mod.validate_report(bad)

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
    assert mod.main(["--date", "20260813"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_infra_6391_helper_edges_and_dirty_queue_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6391: malformed inputs fail closed without fabricated evidence."""

    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod.read_json_mapping(bad_json)[1]["error"].startswith("json_error:")
    not_mapping = tmp_path / "array.json"
    not_mapping.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(not_mapping)[1]["error"] == "json_not_mapping"

    assert (
        mod._summarize_artifact(tmp_path, Path("results/missing.json"))["reason"]
        == "artifact_absent"
    )
    assert mod._conductor_rows(tmp_path, "unknown-task") == []
    log_dir = tmp_path / "ops"
    log_dir.mkdir()
    (log_dir / "conductor-log.md").write_text("malformed\n", encoding="utf-8")
    assert mod._conductor_rows(tmp_path, mod.EXPECTED_V549_TASK_IDS[0]) == []
    (log_dir / "conductor-log.md").write_text(
        "V548 terminal evidence handoff and V549 queue pref malformed\n", encoding="utf-8"
    )
    assert mod._conductor_rows(tmp_path, mod.EXPECTED_V549_TASK_IDS[0]) == []

    assert mod._base_terminal_class({}, {"error": "bad"}) == "malformed"
    assert (
        mod._base_terminal_class({"honest_verdict": "blocked_gate_check_failed"}, {"error": None})
        == "blocked"
    )
    assert mod._base_terminal_class({"status": "unknown"}, {"error": None}) == "unknown"

    payloads = {task_id: {"duration_s": 1.0} for task_id in mod.EXPECTED_V549_TASK_IDS}
    metas = {task_id: {"error": None} for task_id in mod.EXPECTED_V549_TASK_IDS}
    payloads[mod.EXPECTED_V549_TASK_IDS[0]] = {"duration_s": "bad"}
    assert (
        mod._duration_receipts(payloads, metas)[mod.EXPECTED_V549_TASK_IDS[0]]["source"]
        == "duration_missing_or_non_numeric"
    )

    assert mod._model_arm_counts({}, "missing", "model") == {}
    assert mod._model_arm_counts({"counts": []}, "counts", "model") == {}
    assert mod._model_arm_counts({"counts": {}}, "counts", "model") == {}
    assert (
        mod._model_arm_counts({"counts": {"by_model_and_arm": {"model": []}}}, "counts", "model")
        == {}
    )

    @dataclass
    class Risk:
        severity: str

    risk_rows = mod._risk_rows([Risk("HARD"), object()])
    assert risk_rows[0] == {"severity": "HARD"}
    assert risk_rows[1]["repr"].startswith("<object object")

    rendered, receipt = mod.render_prompt("x {project_root} {date}", REPO, "20260813")
    assert REPO.as_posix() in rendered
    assert receipt["format_ok"] is True
    rendered, receipt = mod.render_prompt("x {broken} {project_root}", REPO, "20260813")
    assert "{broken}" in rendered and REPO.as_posix() in rendered
    assert receipt["format_ok"] is False

    data, identity = mod.load_v550_queue(REPO)
    assert identity["audited_queue"]["path"] == "research-roadmap.yaml"
    (tmp_path / "research-roadmap.yaml").write_text(
        "milestone: active\ntasks: []\n", encoding="utf-8"
    )
    (tmp_path / "research-roadmap-next.yaml").write_text(
        'milestone: "2026.08.550"\ntasks: []\n', encoding="utf-8"
    )
    _next_data, next_identity = mod.load_v550_queue(tmp_path)
    assert next_identity["audited_queue"]["path"] == "research-roadmap-next.yaml"
    assert mod._tasks({"tasks": "not-list"}) == []

    dirty = copy.deepcopy(data)
    tasks = dirty["tasks"]
    tasks[1]["id"] = tasks[0]["id"]
    tasks[2]["deliverable"] = "not-results.txt"
    tasks[3]["gated_on"] = [
        {
            "upstream": "exp2091-retired-upstream",
            "artifact_field": "not_declared",
            "op": "??",
            "value": 1.0,
        }
    ]
    tasks[4]["requires"] = [tasks[4]["id"], "exp2091-retired-upstream"]
    tasks[5]["prior_failures"] = [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
    tasks[6]["prior_failures"] = []
    tasks[6]["agent_type"] = "gemini"
    tasks[7]["model"] = "opus"
    tasks[8]["requires_gpu"] = True
    tasks[8]["prompt"] = (
        "CONTEXT\n"
        "{project_root}\n"
        "{date}\n"
        "TASK\n"
        "CONCRETE STEPS\n"
        "Run command: x\n"
        "MODEL_SPECS must include Bad/Unexpected-GGUF from cached_sota_pair(). "
        "Use embedded tokenizers and no AutoTokenizer.\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    tasks[9]["requires_gpu"] = True
    tasks[9]["prompt"] = (
        "CONTEXT\n"
        "{project_root}\n"
        "{date}\n"
        "TASK\n"
        "CONCRETE STEPS\n"
        "Run command: x\n"
        "AutoTokenizer.from_pretrained appears in a legacy headline cell.\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    tasks[10]["prompt"] = tasks[10]["prompt"].replace(
        "Do NOT push. Do NOT modify scripts/research_conductor.py.",
        "Do NOT push.",
    )
    tasks[10]["prompt"] = tasks[10]["prompt"].replace("{date}", "date")

    checks = mod.validate_v550_queue_data(dirty, REPO, "20260813", retired_exp_ids={2091, 6391})
    assert checks["schema_validation"]["ok"] is False
    assert checks["v550_id_and_deliverable_checks"]["ok"] is False
    assert checks["v550_dependency_and_gate_checks"]["ok"] is False
    assert checks["v550_gate_field_cross_reference_checks"]["ok"] is False
    assert checks["v550_prior_failure_checks"]["ok"] is False
    assert checks["v550_exclusion_manifest_checks"]["ok"] is False
    assert checks["v550_agent_model_and_llm_policy_checks"]["ok"] is False
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
