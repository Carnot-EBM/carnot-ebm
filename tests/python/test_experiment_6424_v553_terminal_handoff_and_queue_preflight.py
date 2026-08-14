"""Tests for Exp6424 V553 terminal handoff.

Spec refs: REQ-INFRA-6424, SCENARIO-INFRA-6424-1,
SCENARIO-INFRA-6424-2, SCENARIO-INFRA-6424-3,
SCENARIO-INFRA-6424-4, SCENARIO-INFRA-6424-5,
SCENARIO-INFRA-6424-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6424_v553_terminal_handoff_and_queue_preflight as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
_REPORT_CACHE: dict[str, object] | None = None


def _report() -> dict[str, object]:
    global _REPORT_CACHE
    if _REPORT_CACHE is None:
        _REPORT_CACHE = mod.build_report(
            REPO,
            date="20260814",
            command_receipts=[{"command": "focused", "exit_code": 0}],
            before_hashes=mod.protected_hashes(REPO),
            duration_s=1.0,
        )
    return copy.deepcopy(_REPORT_CACHE)


def test_req_infra_6424_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6424: OpenSpec owns the V553 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6424") : text.index("REQ-INFRA-6404")]

    for marker in (
        "SCENARIO-INFRA-6424-1",
        "SCENARIO-INFRA-6424-2",
        "SCENARIO-INFRA-6424-3",
        "SCENARIO-INFRA-6424-4",
        "SCENARIO-INFRA-6424-5",
        "SCENARIO-INFRA-6424-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6424_v552_evidence_stays_separate() -> None:
    """SCENARIO-INFRA-6424-1: V552 determinations are not collapsed."""

    report = _report()

    assert mod.validate_report(report) == []
    assert report["v552_task_ids"] == list(mod.EXPECTED_V552_TASK_IDS)

    artifacts = report["v552_terminal_artifacts_and_sidecars_by_task"]
    assert (
        artifacts["exp6414-fresh-three-family-factor-event-corpus"]["terminal_class"]
        == "flagged"
    )
    assert (
        artifacts["exp6417-authentic-write-time-factor-admission-ab"]["terminal_class"]
        == "flagged"
    )
    assert artifacts["exp6420-csl-authenticity-safety-audit"]["terminal_class"] == "null"
    assert artifacts["terminal_class_counts"]["missing"] == 0
    assert artifacts["sidecar_counts_by_task"]["exp6412-v551-powered-claim-integrity-audit"] == 2
    assert artifacts["sidecar_counts_by_task"]["exp6413-authenticated-sota-gguf-execution-receipts"] == 1

    verdicts = report["v552_artifact_verdicts"]
    assert verdicts["exp6414-fresh-three-family-factor-event-corpus"].startswith(
        "complete: fresh three-family"
    )
    assert verdicts["exp6420-csl-authenticity-safety-audit"].startswith("complete_null:")

    conductor = report["v552_conductor_outcomes"]
    assert conductor["exp6414-fresh-three-family-factor-event-corpus"]["log_status_counts"][
        "FLAGGED"
    ] == 1
    assert (
        conductor["exp6414-fresh-three-family-factor-event-corpus"]["research_complete_result"]
        == "OK (conductor)"
    )

    adversarial = report["v552_current_adversarial_findings"]
    assert (
        adversarial["exp6414-fresh-three-family-factor-event-corpus"][
            "stamped_flagged_adversarial"
        ]
        is True
    )
    assert (
        adversarial["exp6414-fresh-three-family-factor-event-corpus"]["current_live_verdict"]
        == "critical"
    )
    assert adversarial["exp6413-authenticated-sota-gguf-execution-receipts"][
        "current_live_verdict"
    ] == "clean"
    assert adversarial["exp6413-authenticated-sota-gguf-execution-receipts"][
        "summary_receipt"
    ]["invoked_before_field_import"] is True


def test_scenario_infra_6424_boundary_facts_preserved() -> None:
    """SCENARIO-INFRA-6424-2: flags, CSL null, and ARC no-solve survive."""

    report = _report()
    eligibility = report["v552_scientific_claim_eligibility_by_task"]
    boundary = report["exp6414_6417_6420_6421_6422_boundary"]

    assert eligibility["exp6413-authenticated-sota-gguf-execution-receipts"][
        "authenticated_gguf_receipt_eligibility"
    ] is True
    assert eligibility["exp6414-fresh-three-family-factor-event-corpus"][
        "public_factor_claim_eligibility"
    ] is False
    assert eligibility["exp6417-authentic-write-time-factor-admission-ab"][
        "public_factor_claim_eligibility"
    ] is False
    assert eligibility["exp6418-execution-grounded-dual-path-csl"][
        "prospective_csl_claim_eligibility"
    ] is False
    assert eligibility["exp6419-held-shift-restart-csl-replication"][
        "prospective_csl_claim_eligibility"
    ] is False
    assert eligibility["exp6421-arc-opt-in-executed-policy-ab"][
        "internal_arc_policy_influence_eligibility"
    ] is True
    assert eligibility["exp6421-arc-opt-in-executed-policy-ab"][
        "public_arc_claim_eligibility"
    ] is False

    assert boundary["exp6414"]["duration_flag_preserved"] is True
    assert boundary["exp6414"]["claim_eligibility"] is False
    assert boundary["exp6417"]["duration_flag_preserved"] is True
    assert boundary["exp6417"]["claim_eligibility"] is False
    assert boundary["exp6420"]["csl_null_preserved"] is True
    assert boundary["exp6420"]["reported_metric_mismatch_count"] == 8
    assert boundary["exp6420"]["raw_output_reuse_preserved"] is True
    assert boundary["exp6420"]["cache_resurrection_preserved"] is True
    assert boundary["exp6420"]["underpowered_cell_count"] == 4
    for task_id in ("exp6421", "exp6422"):
        assert boundary[task_id]["level_solve_claimed"] is False
        assert boundary[task_id]["solve_registry_modified"] is False
        assert boundary[task_id]["public_arc_claim_eligibility"] is False


def test_scenario_infra_6424_v553_queue_validates() -> None:
    """SCENARIO-INFRA-6424-3 and 4: active V553 has twelve valid tasks."""

    report = _report()

    assert report["status"] == "complete_v553_queue_preflight_passed"
    assert report["blocked_reason"] is None
    assert report["honest_verdict"].startswith("complete_v553_queue_preflight_passed:")
    assert report["v553_task_ids"] == list(mod.EXPECTED_V553_TASK_IDS)

    identity = report["v553_milestone_doc_and_queue_hashes"]
    assert identity["audited_queue"]["path"] == "research-roadmap.yaml"
    assert identity["requested_next_roadmap"]["present"] is False
    assert identity["milestone_doc"]["proposal_task_count"] == 12

    ids = report["v553_id_and_deliverable_checks"]
    assert ids["ok"] is True
    assert ids["task_count"] == 12
    assert ids["unique_deliverables"] is True
    assert ids["execution_order_ok"] is True

    assert report["v553_dependency_and_gate_checks"]["ok"] is True
    assert report["v553_dependency_and_gate_checks"]["gate_count"] == 12
    assert report["v553_gate_field_cross_reference_checks"]["ok"] is True
    assert report["v553_prior_failure_checks"]["ok"] is True
    assert report["v553_exclusion_manifest_checks"]["ok"] is True
    assert report["v553_agent_model_and_llm_policy_checks"]["ok"] is True
    assert report["prompt_contract_checks"]["ok"] is True


def test_scenario_infra_6424_prompt_llm_and_arc_contracts() -> None:
    """SCENARIO-INFRA-6424-5: prompt, model, and ARC contracts validate."""

    report = _report()
    policy = report["v553_agent_model_and_llm_policy_checks"]
    arc = report["v553_arc_no_solve_checks"]

    assert policy["ok"] is True
    assert policy["local_gguf_task_ids"] == [
        "exp6426-task-scoped-runtime-receipt-contract",
        "exp6427-fresh-constraint-saturation-factor-corpus",
        "exp6430-prospective-write-once-memory-capacity-frontier",
        "exp6432-held-shift-process-restart-csl-replication",
    ]
    assert policy["model_policy_failures"] == []

    assert arc["ok"] is True
    assert arc["arc_task_ids"] == ["exp6434-arc-state-key-reachability-ab"]
    assert arc["solve_claim_failures"] == []
    assert arc["solve_registry_update_failures"] == []
    assert arc["canonical_live_path_failures"] == []
    assert arc["game_source_failures"] == []
    assert arc["exhaustive_ground_truth_failures"] == []
    assert arc["per_game_adapter_failures"] == []


def test_scenario_infra_6424_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6424-6: artifact schema is stable and atomic."""

    report = _report()

    assert report["active_roadmap_modified"] is False
    assert report["conductor_modified"] is False
    assert report["solve_registry_modified"] is False
    assert report["protected_files_unchanged"]["ok"] is True
    assert report["verifier_is_oracle"] is False
    assert report["random_seed"] is None
    assert report["preconditions_checked"]["system_state"]["research_compute_started"] is False
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(report["field_provenance"])
    assert set(report["field_provenance"].values()) <= {
        "measured",
        "derived",
        "constant",
        "upstream",
    }
    for expression in report["v553_dependency_and_gate_checks"]["structured_gate_expressions"]:
        assert expression in report["field_principles"]
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    validations = [
        ("delete", "status", "missing required field: status"),
        ("set", ("verifier_is_oracle", True), "verifier_is_oracle must be false"),
        ("set", ("random_seed", 6424), "random_seed must be null"),
        (
            "set",
            ("exp6414_6417_6420_6421_6422_boundary", []),
            "exp6414_6417_6420_6421_6422_boundary must be a mapping",
        ),
        (
            "set",
            ("exp6414_6417_6420_6421_6422_boundary.exp6414.duration_flag_preserved", False),
            "Exp6414 duration flag",
        ),
        (
            "set",
            ("exp6414_6417_6420_6421_6422_boundary.exp6417.duration_flag_preserved", False),
            "Exp6417 duration flag",
        ),
        (
            "set",
            ("exp6414_6417_6420_6421_6422_boundary.exp6420.csl_null_preserved", False),
            "Exp6420 CSL null",
        ),
        (
            "set",
            ("exp6414_6417_6420_6421_6422_boundary.exp6421.level_solve_claimed", True),
            "Exp6421 no-solve boundary",
        ),
        (
            "set",
            ("exp6414_6417_6420_6421_6422_boundary.exp6422.level_solve_claimed", True),
            "Exp6422 no-solve boundary",
        ),
        (
            "set",
            ("v552_scientific_claim_eligibility_by_task", []),
            "v552_scientific_claim_eligibility_by_task must be a mapping",
        ),
        (
            "set",
            (
                "v552_scientific_claim_eligibility_by_task.exp6414-fresh-three-family-factor-event-corpus.public_factor_claim_eligibility",
                True,
            ),
            "Exp6414 public factor eligibility",
        ),
        (
            "set",
            (
                "v552_scientific_claim_eligibility_by_task.exp6417-authentic-write-time-factor-admission-ab.public_factor_claim_eligibility",
                True,
            ),
            "Exp6417 public factor eligibility",
        ),
        (
            "set",
            (
                "v552_scientific_claim_eligibility_by_task.exp6420-csl-authenticity-safety-audit.prospective_csl_claim_eligibility",
                True,
            ),
            "Exp6420 prospective CSL eligibility",
        ),
        ("set", ("active_roadmap_modified", True), "active roadmap changed"),
        ("set", ("conductor_modified", True), "conductor changed"),
        ("set", ("solve_registry_modified", True), "solve registry changed"),
        ("set", ("protected_files_unchanged.ok", False), "protected files changed"),
        ("set", ("blocked_reason", "unexpected"), "passed report must not have blocked_reason"),
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
    bad["status"] = "complete_v553_queue_preflight_passed"
    bad["v553_id_and_deliverable_checks"]["ok"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "passed report has failed V553 checks" in mod.validate_report(bad)

    bad = copy.deepcopy(report)
    bad["status"] = "complete_blocked_v553_queue_preflight_failed"
    bad["blocked_reason"] = None
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "blocked report must name blocked_reason" in mod.validate_report(bad)

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    path = mod.write_report(report, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
    assert path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text(encoding="utf-8")) == report

    monkeypatch.setattr(
        mod,
        "run",
        lambda *, date, root=REPO, write=True, command_receipts=None: {
            "status": f"complete-{date}",
            "honest_verdict": "complete: patched",
        },
    )
    assert mod.main(["--date", "20260814"]) == 0
    assert mod.RESULT_RELATIVE_PATH.name in capsys.readouterr().out


def test_req_infra_6424_helper_edges_and_dirty_queue_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6424: malformed inputs fail closed without fabricated evidence."""

    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    assert mod._terminal_class({}, {"error": "missing"}, {}) == "missing"
    assert mod._terminal_class({}, {"error": "bad"}, {}) == "malformed"
    assert mod._terminal_class({"status": "complete_null"}, {"error": None}, {}) == "null"
    assert mod._terminal_class({"status": "complete_ready"}, {"error": None}, {}) == "ready"
    assert mod._terminal_class({"status": "complete_positive"}, {"error": None}, {}) == "positive"
    assert mod._terminal_class({"status": "complete"}, {"error": None}, {}) == "complete"
    assert mod._terminal_class({"status": "blocked"}, {"error": None}, {}) == "blocked"
    assert mod._terminal_class({"status": "weird"}, {"error": None}, {}) == "unknown"
    assert mod._conductor_log_rows(tmp_path, "missing-task") == []
    assert mod._research_complete_result(tmp_path, "missing-task") is None
    complete = tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.write_text("milestones: []\n", encoding="utf-8")
    assert mod._research_complete_result(tmp_path, "missing-task") is None

    payloads = {
        task_id: {"status": "complete", "honest_verdict": "complete: synthetic"}
        for task_id in mod.EXPECTED_V552_TASK_IDS
    }
    payloads["exp6413-authenticated-sota-gguf-execution-receipts"][
        "authenticated_receipt_contract_ready_score"
    ] = 1.0
    payloads["exp6420-csl-authenticity-safety-audit"][
        "prospective_csl_claim_eligibility"
    ] = ["not-a-mapping"]
    for task_id in (
        "exp6421-arc-opt-in-executed-policy-ab",
        "exp6422-arc-held-family-policy-safety-audit",
    ):
        payloads[task_id]["level_solve_claimed"] = False
        payloads[task_id]["solve_registry_modified"] = False
    adversarial = {
        task_id: {"current_live_has_critical": False, "current_live_flags": []}
        for task_id in mod.EXPECTED_V552_TASK_IDS
    }
    eligibility = mod._v552_scientific_claim_eligibility(payloads, adversarial)
    assert eligibility["exp6420-csl-authenticity-safety-audit"][
        "prospective_csl_blockers"
    ] == ["exp6420_null"]

    data, identity = mod.load_v553_queue(REPO)
    assert identity["audited_queue"]["path"] == "research-roadmap.yaml"
    assert mod._first_blocked_reason(
        {"v553_id_and_deliverable_checks": {"ok": True}}
    ) == "unknown_queue_contract_failure"
    assert mod._first_blocked_reason(
        {
            "v553_id_and_deliverable_checks": {
                "ok": False,
                "deliverable_failures": [
                    {
                        "task_id": "exp6424-x",
                        "reason": "bad_deliverable",
                        "deliverable": "x.txt",
                    }
                ],
            }
        }
    ) == "v553_id_and_deliverable_checks: exp6424-x.bad_deliverable:x.txt"
    assert mod._first_blocked_reason({"prompt_contract_checks": {"ok": False, "failures": []}}).startswith(
        "prompt_contract_checks"
    )
    assert {failure["reason"] for failure in mod._local_gguf_policy_failures(
        {"id": "exp6424-test"},
        "Qwen/Qwen2.5-7B-Instruct-GGUF raw output no AutoTokenizer",
    )} >= {
        "missing_model_specs",
        "missing_cached_sota_pair",
        "missing_embedded_tokenizer_rule",
    }
    (tmp_path / "research-roadmap.yaml").write_text(
        'milestone: "2026.08.552"\ntasks: []\n', encoding="utf-8"
    )
    (tmp_path / "research-roadmap-next.yaml").write_text(
        'milestone: "2026.08.553"\ntasks: []\n', encoding="utf-8"
    )
    _next_data, next_identity = mod.load_v553_queue(tmp_path)
    assert next_identity["audited_queue"]["path"] == "research-roadmap-next.yaml"

    dirty = copy.deepcopy(data)
    tasks = dirty["tasks"]
    tasks[1]["id"] = tasks[0]["id"]
    tasks[2]["deliverable"] = "not-results.txt"
    tasks[4]["gated_on"] = [
        {
            "upstream": "exp2091-retired-upstream",
            "artifact_field": None,
            "op": "??",
            "value": 1.0,
        },
        "bad-gate",
    ]
    tasks[5]["requires"] = [tasks[5]["id"], "exp2091-retired-upstream"]
    tasks[6]["prior_failures"] = []
    tasks[7]["agent_type"] = "gemini"
    tasks[8]["requires_gpu"] = True
    tasks[8]["prompt"] = (
        "CONTEXT\n"
        "{project_root}\n"
        "TASK\n"
        "CONCRETE STEPS\n"
        "Run command: x\n"
        "MODEL_SPECS Bad/Unexpected-GGUF cached_sota_pair() embedded tokenizer "
        "AutoTokenizer.from_pretrained legacy headline model\n"
        "Do NOT push."
    )
    tasks[9]["prior_failures"] = [{"experiment_id": "", "verdict": "", "addressed_by": ""}]
    tasks[10]["prompt"] = (
        "CONTEXT\n{project_root}\n{date}\nTASK\nCONCRETE STEPS\n"
        "Run command: x\nClaim a level solve and update the solve registry "
        "with a per-game adapter, game source, and exhaustive ground-truth search.\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    checks = mod.validate_v553_queue_data(dirty, REPO, "20260814", retired_exp_ids={2091, 6424})
    assert checks["schema_validation"]["ok"] is False
    assert checks["v553_id_and_deliverable_checks"]["ok"] is False
    assert checks["v553_dependency_and_gate_checks"]["ok"] is False
    assert checks["v553_gate_field_cross_reference_checks"]["ok"] is False
    assert checks["v553_prior_failure_checks"]["ok"] is False
    assert checks["v553_exclusion_manifest_checks"]["ok"] is False
    assert checks["v553_agent_model_and_llm_policy_checks"]["ok"] is False
    assert checks["v553_arc_no_solve_checks"]["ok"] is False
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
        before_hashes: dict[str, str | None],
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

    report = mod.run(date="20260814", root=REPO, write=True)
    assert report["command_receipts"] == [{"command": "external"}]
    assert writes == [report]

    monkeypatch.setattr(mod, "validate_report", lambda report: ["bad"])
    with pytest.raises(ValueError, match="bad"):
        mod.run(date="20260814", root=REPO, write=False, command_receipts=[{"command": "c"}])
