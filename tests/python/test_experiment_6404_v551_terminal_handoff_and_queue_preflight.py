"""Tests for Exp6404 V551 terminal handoff.

Spec refs: REQ-INFRA-6404, SCENARIO-INFRA-6404-1,
SCENARIO-INFRA-6404-2, SCENARIO-INFRA-6404-3,
SCENARIO-INFRA-6404-4, SCENARIO-INFRA-6404-5,
SCENARIO-INFRA-6404-6.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from carnot import experiment_6404_v551_terminal_handoff_and_queue_preflight as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
_REPORT_CACHE: dict[str, object] | None = None


def _report() -> dict[str, object]:
    global _REPORT_CACHE
    if _REPORT_CACHE is None:
        _REPORT_CACHE = mod.build_report(
            REPO,
            date="20260813",
            command_receipts=[{"command": "focused", "exit_code": 0}],
            before_hashes=mod.protected_hashes(REPO),
            duration_s=1.0,
        )
    return copy.deepcopy(_REPORT_CACHE)


def test_req_infra_6404_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6404: OpenSpec owns the V551 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6404") : text.index("REQ-INFRA-6379")]

    for marker in (
        "SCENARIO-INFRA-6404-1",
        "SCENARIO-INFRA-6404-2",
        "SCENARIO-INFRA-6404-3",
        "SCENARIO-INFRA-6404-4",
        "SCENARIO-INFRA-6404-5",
        "SCENARIO-INFRA-6404-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6404_v550_evidence_stays_separate() -> None:
    """SCENARIO-INFRA-6404-1: V550 determinations are not collapsed."""

    report = _report()

    assert mod.validate_report(report) == []
    assert report["v550_task_ids"] == list(mod.EXPECTED_V550_TASK_IDS)

    rows = report["v550_terminal_artifacts_by_task"]
    assert rows["exp6395-held-factor-transport-license-matrix"]["terminal_class"] == "positive"
    assert rows["exp6399-capability-learning-safety-audit"]["terminal_class"] == "null"
    assert rows["exp6403-v550-adversarial-capstone"]["terminal_class"] == "clean"
    assert rows["terminal_class_counts"]["absent"] == 0
    assert rows["terminal_class_counts"]["retired"] == 0

    verdicts = report["v550_artifact_verdicts"]
    assert verdicts["exp6395-held-factor-transport-license-matrix"].startswith(
        "complete_positive:"
    )
    assert verdicts["exp6399-capability-learning-safety-audit"].startswith("complete_null:")
    assert verdicts["exp6403-v550-adversarial-capstone"].startswith("complete:")

    conductor = report["v550_conductor_outcomes"]
    assert conductor["exp6391-v550-terminal-handoff-and-queue-preflight"]["FAIL"] == 1
    assert conductor["exp6391-v550-terminal-handoff-and-queue-preflight"]["OK"] == 1
    assert conductor["exp6403-v550-adversarial-capstone"]["FAIL"] == 1
    assert conductor["exp6403-v550-adversarial-capstone"]["OK"] == 1
    assert conductor["exp6395-held-factor-transport-license-matrix"]["OK"] == 1

    findings = report["v550_adversarial_findings"]
    for task_id in mod.EXPECTED_V550_TASK_IDS:
        assert findings[task_id]["summary_receipt"]["invoked_before_field_import"] is True
        assert "live_verdict" in findings[task_id]
    assert findings["exp6399-capability-learning-safety-audit"]["public_claim_eligible"] is False
    assert (
        findings["exp6403-v550-adversarial-capstone"]["public_claim_eligible"] is False
    )

    durations = report["v550_duration_receipts_by_task"]
    assert durations["exp6395-held-factor-transport-license-matrix"]["duration_s"] == pytest.approx(
        53.7328749478329
    )
    assert durations["exp6403-v550-adversarial-capstone"]["duration_s"] > 0
    assert all(row["source"] == "artifact.duration_s" for row in durations.values())


def test_scenario_infra_6404_factor_and_arc_boundaries() -> None:
    """SCENARIO-INFRA-6404-2 and 3: V550 factor and ARC boundaries stay exact."""

    report = _report()
    factor = report["v550_factor_boundary"]
    arc = report["v550_arc_boundary"]

    assert factor["exp6395_licensed_cell_count"] == 4
    assert factor["exp6395_qwen_abstention_count"] == 3
    assert factor["exp6395_rejected_gemma_cell_count"] == 2
    assert factor["universal_support_claimed"] is False
    assert factor["licensed_cells"] == [
        {
            "model_id": "unsloth/gemma-4-31B-it-GGUF",
            "constraint_family": "threshold_guard",
        },
        {"model_id": "unsloth/gemma-4-31B-it-GGUF", "constraint_family": "route_guard"},
        {"model_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "constraint_family": "route_guard"},
        {
            "model_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "constraint_family": "conservation_guard",
        },
    ]
    assert factor["positive_internal_factor_results"][
        "exp6396_delta_verified_future_exact_yield"
    ] == pytest.approx(0.166666666667)
    assert factor["positive_internal_factor_results"]["exp6397_delta_future_exact_yield"] == 0.25
    assert factor["positive_internal_factor_results"]["exp6398_delta_exact_yield"] == 0.25
    assert factor["exp6399_public_block"]["public_factor_claim_eligibility"] is False
    assert factor["public_factor_utility_promotion_count"] == 0

    assert arc["exp6400_shadow_readiness"]["ready_score"] == 1.0
    assert arc["exp6401_causal_progress"]["delta_exact_progress_proxy"] == pytest.approx(3.5)
    assert arc["exp6401_causal_progress"]["delta_false_accept_count"] == -9
    assert arc["exp6401_internal_route_eligible"] is True
    assert arc["exp6402_public_arc_eligibility"] is False
    assert arc["actual_route_promotion_count"] == 0
    assert arc["solve_claim_count"] == 0
    assert arc["solve_registry_modified"] is False


def test_scenario_infra_6404_v551_queue_fails_closed() -> None:
    """SCENARIO-INFRA-6404-4 and 5: active V551 has six of twelve tasks."""

    report = _report()

    assert report["status"] == "complete_blocked_v551_queue_incomplete"
    assert report["honest_verdict"].startswith("complete_blocked_v551_queue_incomplete:")
    assert report["v551_task_ids"] == list(mod.EXPECTED_V551_TASK_IDS[:6])

    identity = report["v551_milestone_doc_and_queue_hashes"]
    assert identity["audited_queue"]["path"] == "research-roadmap.yaml"
    assert identity["requested_next_roadmap"]["present"] is False
    assert identity["milestone_doc"]["proposal_task_count"] == 12

    ids = report["v551_id_and_deliverable_checks"]
    assert ids["ok"] is False
    assert ids["task_count"] == 6
    assert ids["expected_task_count"] == 12
    assert ids["missing_expected_task_ids"] == list(mod.EXPECTED_V551_TASK_IDS[6:])
    assert ids["execution_order_ok"] is True
    assert ids["unique_deliverables"] is True

    assert report["v551_dependency_and_gate_checks"]["ok"] is True
    assert report["v551_gate_field_cross_reference_checks"]["ok"] is True
    assert report["v551_prior_failure_checks"]["ok"] is True
    assert report["v551_exclusion_manifest_checks"]["ok"] is True

    policy = report["v551_agent_model_and_llm_policy_checks"]
    assert policy["all_tasks_codex_gpt55"] is True
    assert policy["llm_task_ids"] == [
        "exp6408-powered-write-time-factor-admission-ab",
        "exp6409-graph-local-multisession-continuous-learning",
    ]
    assert {
        (failure["task_id"], failure["reason"]) for failure in policy["model_policy_failures"]
    } == {
        (
            "exp6408-powered-write-time-factor-admission-ab",
            "missing_no_legacy_headline_cell_rule",
        ),
        (
            "exp6409-graph-local-multisession-continuous-learning",
            "missing_no_legacy_headline_cell_rule",
        ),
    }
    assert policy["ok"] is False

    prompts = report["prompt_contract_checks"]
    assert prompts["ok"] is True
    assert prompts["checked_task_count"] == 6
    assert prompts["raw_placeholder_contract_ok"] is True


def test_scenario_infra_6404_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6404-6: artifact schema is stable and atomic."""

    report = _report()

    assert report["active_roadmap_modified"] is False
    assert report["conductor_modified"] is False
    assert report["solve_registry_modified"] is False
    assert report["claims_ledger_modified"] is False
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
    for expression in report["v551_dependency_and_gate_checks"]["structured_gate_expressions"]:
        assert expression in report["field_principles"]
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    validations = [
        ("delete", "status", "missing required field: status"),
        ("set", ("verifier_is_oracle", True), "verifier_is_oracle must be false"),
        ("set", ("random_seed", 6404), "random_seed must be null"),
        ("set", ("v550_factor_boundary.exp6395_licensed_cell_count", 3), "four Exp6395 licenses"),
        ("set", ("v550_factor_boundary.exp6395_qwen_abstention_count", 2), "Qwen abstention"),
        ("set", ("v550_factor_boundary.exp6395_rejected_gemma_cell_count", 1), "two rejected Gemma"),
        ("set", ("v550_factor_boundary.universal_support_claimed", True), "no universal support"),
        ("set", ("v550_factor_boundary.exp6399_public_block.public_factor_claim_eligibility", True), "public factor block"),
        ("set", ("v550_arc_boundary.actual_route_promotion_count", 1), "zero route promotion"),
        ("set", ("v550_arc_boundary.solve_claim_count", 1), "no ARC solve"),
        ("set", ("v550_arc_boundary.exp6402_public_arc_eligibility", True), "public ARC ineligibility"),
        ("set", ("active_roadmap_modified", True), "active roadmap changed"),
        ("set", ("conductor_modified", True), "conductor changed"),
        ("set", ("solve_registry_modified", True), "solve registry changed"),
        ("set", ("claims_ledger_modified", True), "claims ledger changed"),
        ("set", ("protected_files_unchanged.ok", False), "protected files changed"),
        ("set", ("honest_verdict", "ok"), "honest_verdict lacks terminal prefix"),
        ("set", ("reproducibility_checksum", "sha256:bad"), "reproducibility_checksum mismatch"),
    ]
    for mode, spec, expected in validations:
        bad = copy.deepcopy(report)
        if mode == "delete":
            del bad[spec]
        else:
            dotted, value = spec
            target = bad
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        if expected != "reproducibility_checksum mismatch":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert any(expected in error for error in mod.validate_report(bad))

    bad = copy.deepcopy(report)
    bad["v550_factor_boundary"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "v550_factor_boundary must be a mapping" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["v550_arc_boundary"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "v550_arc_boundary must be a mapping" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    del bad["field_principles"]["status"]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "missing field_principles entry: status" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["field_principles"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_principles must be a mapping" in mod.validate_report(bad)

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
    bad["status"] = "complete_v551_queue_preflight_passed"
    bad["v551_id_and_deliverable_checks"]["ok"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "passed report has failed V551 checks" in mod.validate_report(bad)

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


def test_req_infra_6404_helper_edges_and_dirty_queue_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6404: malformed inputs fail closed without fabricated evidence."""

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
    assert mod._live_adversarial(tmp_path, Path("results/missing.json"), False)[
        "verdict"
    ] == "absent"
    assert mod._terminal_class({}, {"error": "missing"}, {"verdict": "absent"}) == "absent"
    assert mod._terminal_class({}, {"error": "bad"}, {"verdict": "clean"}) == "malformed"
    assert (
        mod._terminal_class({"status": "complete"}, {"error": None}, {"critical_count": 1})
        == "flagged"
    )
    assert (
        mod._terminal_class({"honest_verdict": "retired_scope"}, {"error": None}, {"verdict": "clean"})
        == "retired"
    )
    assert mod._terminal_class({"status": "complete_partial"}, {"error": None}, {"verdict": "clean"}) == "partial"
    assert mod._terminal_class({"status": "skipped"}, {"error": None}, {"verdict": "clean"}) == "skipped"
    assert mod._terminal_class({"status": "blocked"}, {"error": None}, {"verdict": "clean"}) == "blocked"
    assert mod._terminal_class({"status": "unknown"}, {"error": None}, {"verdict": "clean"}) == "unknown"
    assert mod._duration_receipt("task", "file.json", {"duration_s": "bad"}, {"error": None})[
        "source"
    ] == "duration_missing_or_non_numeric"
    assert mod._duration_receipt("task", "file.json", {}, {"error": "missing"})["source"] == "artifact_absent"
    assert mod._conductor_rows(tmp_path, mod.EXPECTED_V550_TASK_IDS[0]) == []
    log_dir = tmp_path / "ops"
    log_dir.mkdir(exist_ok=True)
    (log_dir / "conductor-log.md").write_text(
        "V549 terminal evidence handoff and V550 queue pref malformed\n",
        encoding="utf-8",
    )
    assert mod._conductor_rows(tmp_path, mod.EXPECTED_V550_TASK_IDS[0]) == []

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

    assert mod._task_number("exp6404-demo") == 6404
    assert mod._task_number("not-exp") is None
    assert mod._proposal_exp_numbers(tmp_path) == []
    proposal = tmp_path / mod.MILESTONE_DOC_RELATIVE_PATH
    proposal.parent.mkdir(parents=True)
    proposal.write_text("### Exp6404\n### Exp6405\n", encoding="utf-8")
    assert mod._proposal_exp_numbers(tmp_path) == [6404, 6405]

    data, identity = mod.load_v551_queue(REPO)
    assert identity["audited_queue"]["path"] == "research-roadmap.yaml"
    assert mod._tasks({"tasks": "not-list"}) == []
    (tmp_path / "research-roadmap.yaml").write_text(
        'milestone: "2026.08.550"\ntasks: []\n', encoding="utf-8"
    )
    (tmp_path / "research-roadmap-next.yaml").write_text(
        'milestone: "2026.08.551"\ntasks: []\n', encoding="utf-8"
    )
    _next_data, next_identity = mod.load_v551_queue(tmp_path)
    assert next_identity["audited_queue"]["path"] == "research-roadmap-next.yaml"

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
    tasks[5]["agent_type"] = "gemini"
    tasks[5]["model"] = "opus"
    tasks[5]["requires_gpu"] = True
    tasks[5]["prompt"] = (
        "CONTEXT\n"
        "{project_root}\n"
        "{date}\n"
        "TASK\n"
        "CONCRETE STEPS\n"
        "Run command: x\n"
        "MODEL_SPECS Bad/Unexpected-GGUF cached_sota_pair() embedded tokenizer "
        "AutoTokenizer.from_pretrained legacy headline cell\n"
        "Do NOT push."
    )
    checks = mod.validate_v551_queue_data(dirty, REPO, "20260813", retired_exp_ids={2091, 6404})
    assert checks["schema_validation"]["ok"] is False
    assert checks["v551_id_and_deliverable_checks"]["ok"] is False
    assert checks["v551_dependency_and_gate_checks"]["ok"] is False
    assert checks["v551_gate_field_cross_reference_checks"]["ok"] is False
    assert checks["v551_prior_failure_checks"]["ok"] is False
    assert checks["v551_exclusion_manifest_checks"]["ok"] is False
    assert checks["v551_agent_model_and_llm_policy_checks"]["ok"] is False
    assert checks["prompt_contract_checks"]["ok"] is False

    minimal = copy.deepcopy(data)
    minimal["tasks"] = [copy.deepcopy(data["tasks"][0])]
    minimal["tasks"][0]["requires_gpu"] = True
    minimal["tasks"][0]["prior_failures"] = []
    minimal["tasks"][0]["agent_type"] = "gemini"
    minimal["tasks"][0]["model"] = "opus"
    minimal["tasks"][0]["prompt"] = (
        "CONTEXT\n"
        "{project_root}\n"
        "TASK\n"
        "CONCRETE STEPS\n"
        "Run command: x\n"
        "Do NOT push."
    )
    minimal_checks = mod.validate_v551_queue_data(minimal, REPO, "20260813", retired_exp_ids=set())
    minimal_reasons = {
        failure["reason"]
        for failure in minimal_checks["v551_agent_model_and_llm_policy_checks"][
            "model_policy_failures"
        ]
    }
    assert {
        "missing_model_specs",
        "missing_cached_sota_pair",
        "missing_mandated_gguf_id",
        "missing_embedded_tokenizer_rule",
    } <= minimal_reasons
    assert minimal_checks["v551_prior_failure_checks"]["ok"] is False
    assert minimal_checks["prompt_contract_checks"]["raw_placeholder_contract_ok"] is False

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
