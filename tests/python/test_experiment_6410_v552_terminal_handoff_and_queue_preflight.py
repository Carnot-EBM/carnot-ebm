"""Tests for Exp6410 V552 terminal handoff.

Spec refs: REQ-INFRA-6410, SCENARIO-INFRA-6410-1,
SCENARIO-INFRA-6410-2, SCENARIO-INFRA-6410-3,
SCENARIO-INFRA-6410-4, SCENARIO-INFRA-6410-5,
SCENARIO-INFRA-6410-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6410_v552_terminal_handoff_and_queue_preflight as mod
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


def test_req_infra_6410_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6410: OpenSpec owns the V552 handoff contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6410") : text.index("REQ-INFRA-6404")]

    for marker in (
        "SCENARIO-INFRA-6410-1",
        "SCENARIO-INFRA-6410-2",
        "SCENARIO-INFRA-6410-3",
        "SCENARIO-INFRA-6410-4",
        "SCENARIO-INFRA-6410-5",
        "SCENARIO-INFRA-6410-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6410_v551_evidence_stays_separate() -> None:
    """SCENARIO-INFRA-6410-1: V551 determinations are not collapsed."""

    report = _report()

    assert mod.validate_report(report) == []
    assert report["v551_task_ids"] == list(mod.EXPECTED_V551_TASK_IDS)

    artifacts = report["v551_terminal_artifacts_and_sidecars_by_task"]
    assert (
        artifacts["exp6404-v551-terminal-handoff-and-queue-preflight"]["terminal_class"]
        == "blocked"
    )
    assert (
        artifacts["exp6407-provenance-tiered-factor-memory-protocol"]["terminal_class"] == "flagged"
    )
    assert (
        artifacts["exp6408-powered-write-time-factor-admission-ab"]["artifact_status_raw"]
        == "complete_positive"
    )
    assert artifacts["terminal_class_counts"]["missing"] == 0
    assert (
        artifacts["sidecar_counts_by_task"]["exp6407-provenance-tiered-factor-memory-protocol"] == 5
    )

    verdicts = report["v551_artifact_verdicts"]
    assert verdicts["exp6404-v551-terminal-handoff-and-queue-preflight"].startswith(
        "complete_blocked_v551_queue_incomplete:"
    )
    assert verdicts["exp6408-powered-write-time-factor-admission-ab"].startswith(
        "complete: powered write-time admission"
    )

    conductor = report["v551_conductor_outcomes"]
    assert (
        conductor["exp6407-provenance-tiered-factor-memory-protocol"]["log_status_counts"][
            "FLAGGED"
        ]
        == 1
    )
    assert (
        conductor["exp6407-provenance-tiered-factor-memory-protocol"]["research_complete_result"]
        == "OK (conductor)"
    )

    adversarial = report["v551_adversarial_findings"]
    assert (
        adversarial["exp6407-provenance-tiered-factor-memory-protocol"][
            "stamped_flagged_adversarial"
        ]
        is True
    )
    assert (
        adversarial["exp6407-provenance-tiered-factor-memory-protocol"]["current_live_verdict"]
        == "critical"
    )
    assert (
        adversarial["exp6408-powered-write-time-factor-admission-ab"]["current_live_verdict"]
        == "clean"
    )


def test_scenario_infra_6410_powered_positives_are_hypotheses() -> None:
    """SCENARIO-INFRA-6410-2: source findings bound Exp6408 and Exp6409."""

    report = _report()
    sources = report["v551_source_execution_findings"]
    eligibility = report["v551_scientific_claim_eligibility_by_task"]
    correction = report["exp6407_6408_6409_claim_correction"]

    exp6408 = sources["exp6408-powered-write-time-factor-admission-ab"]
    assert exp6408["declared_gguf_generator_invoked"] is False
    assert exp6408["authenticated_generation_receipt_present"] is False
    assert exp6408["runtime_receipt_kind"] == "derived_from_host_snapshot_and_constants"
    assert exp6408["constant_model_duration_values_s"] == [0.25, 0.3, 0.35]
    assert exp6408["derived_peak_memory_formula_present"] is True

    exp6409 = sources["exp6409-graph-local-multisession-continuous-learning"]
    assert exp6409["inherits_exp6408_runtime_surface"] is True
    assert exp6409["prospective_csl_receipt_surface"] == "inherited_deterministic_replay"

    assert (
        eligibility["exp6407-provenance-tiered-factor-memory-protocol"]["open_adversarial_flag"]
        is True
    )
    assert (
        eligibility["exp6408-powered-write-time-factor-admission-ab"][
            "powered_gguf_claim_eligibility"
        ]
        is False
    )
    assert (
        eligibility["exp6408-powered-write-time-factor-admission-ab"][
            "deterministic_replay_claim_eligibility"
        ]
        is True
    )
    assert (
        eligibility["exp6409-graph-local-multisession-continuous-learning"][
            "prospective_csl_claim_eligibility"
        ]
        is False
    )

    assert correction["historical_artifact_verdicts_preserved"] is True
    assert correction["exp6407_open_adversarial_flag_preserved"] is True
    assert correction["exp6408_powered_positive_requires_audit"] is True
    assert correction["exp6409_prospective_csl_positive_requires_audit"] is True


def test_scenario_infra_6410_v552_queue_validates() -> None:
    """SCENARIO-INFRA-6410-3 and 4: active V552 has fourteen valid tasks."""

    report = _report()

    assert report["status"] == "complete_v552_queue_preflight_passed"
    assert report["honest_verdict"].startswith("complete_v552_queue_preflight_passed:")
    assert report["v552_task_ids"] == list(mod.EXPECTED_V552_TASK_IDS)

    identity = report["v552_milestone_doc_and_queue_hashes"]
    assert identity["audited_queue"]["path"] == "research-roadmap.yaml"
    assert identity["requested_next_roadmap"]["present"] is False
    assert identity["milestone_doc"]["proposal_task_count"] == 14

    ids = report["v552_id_and_deliverable_checks"]
    assert ids["ok"] is True
    assert ids["task_count"] == 14
    assert ids["unique_deliverables"] is True
    assert ids["execution_order_ok"] is True

    assert report["v552_dependency_and_gate_checks"]["ok"] is True
    assert report["v552_dependency_and_gate_checks"]["gate_count"] == 12
    assert report["v552_gate_field_cross_reference_checks"]["ok"] is True
    assert report["v552_prior_failure_checks"]["ok"] is True
    assert report["v552_exclusion_manifest_checks"]["ok"] is True
    assert report["v552_agent_model_and_llm_policy_checks"]["ok"] is True
    assert report["prompt_contract_checks"]["ok"] is True


def test_scenario_infra_6410_arc_no_solve_contracts() -> None:
    """SCENARIO-INFRA-6410-5: ARC tasks stay on the live no-solve path."""

    report = _report()
    arc = report["v552_arc_no_solve_checks"]

    assert arc["ok"] is True
    assert arc["arc_task_ids"] == [
        "exp6421-arc-opt-in-executed-policy-ab",
        "exp6422-arc-held-family-policy-safety-audit",
    ]
    assert arc["solve_claim_failures"] == []
    assert arc["solve_registry_update_failures"] == []
    assert arc["canonical_live_agent_failures"] == []
    assert arc["outer_loop_or_adapter_failures"] == []


def test_scenario_infra_6410_schema_write_and_validation_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-INFRA-6410-6: artifact schema is stable and atomic."""

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
    for expression in report["v552_dependency_and_gate_checks"]["structured_gate_expressions"]:
        assert expression in report["field_principles"]
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    validations = [
        ("delete", "status", "missing required field: status"),
        ("set", ("verifier_is_oracle", True), "verifier_is_oracle must be false"),
        ("set", ("random_seed", 6410), "random_seed must be null"),
        (
            "set",
            ("exp6407_6408_6409_claim_correction", []),
            "exp6407_6408_6409_claim_correction must be a mapping",
        ),
        (
            "set",
            ("exp6407_6408_6409_claim_correction.exp6407_open_adversarial_flag_preserved", False),
            "Exp6407 open flag",
        ),
        (
            "set",
            ("exp6407_6408_6409_claim_correction.exp6408_powered_positive_requires_audit", False),
            "Exp6408 powered correction",
        ),
        (
            "set",
            (
                "exp6407_6408_6409_claim_correction.exp6409_prospective_csl_positive_requires_audit",
                False,
            ),
            "Exp6409 prospective CSL correction",
        ),
        (
            "set",
            ("v551_scientific_claim_eligibility_by_task", []),
            "v551_scientific_claim_eligibility_by_task must be a mapping",
        ),
        (
            "set",
            (
                "v551_scientific_claim_eligibility_by_task.exp6408-powered-write-time-factor-admission-ab.powered_gguf_claim_eligibility",
                True,
            ),
            "Exp6408 powered eligibility",
        ),
        (
            "set",
            (
                "v551_scientific_claim_eligibility_by_task.exp6409-graph-local-multisession-continuous-learning.prospective_csl_claim_eligibility",
                True,
            ),
            "Exp6409 prospective CSL eligibility",
        ),
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
    bad["status"] = "complete_v552_queue_preflight_passed"
    bad["v552_id_and_deliverable_checks"]["ok"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "passed report has failed V552 checks" in mod.validate_report(bad)

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


def test_req_infra_6410_helper_edges_and_dirty_queue_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6410: malformed inputs fail closed without fabricated evidence."""

    assert mod.path_receipt(tmp_path / "missing.json")["present"] is False
    assert mod.read_json_mapping(tmp_path / "missing.json")[1]["error"] == "missing"
    assert mod._terminal_class({}, {"error": "missing"}, {}) == "missing"
    assert mod._terminal_class({}, {"error": "bad"}, {}) == "malformed"
    assert mod._terminal_class({"status": "complete_unproven"}, {"error": None}, {}) == "unproven"
    assert mod._terminal_class({"status": "weird"}, {"error": None}, {}) == "unknown"
    assert (
        mod._conductor_log_rows(tmp_path, "exp6404-v551-terminal-handoff-and-queue-preflight") == []
    )
    assert mod._research_complete_result(tmp_path, "missing-task") is None
    complete = tmp_path / mod.RESEARCH_COMPLETE_RELATIVE_PATH
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.write_text("milestones: []\n", encoding="utf-8")
    assert mod._research_complete_result(tmp_path, "missing-task") is None

    data, identity = mod.load_v552_queue(REPO)
    assert identity["audited_queue"]["path"] == "research-roadmap.yaml"
    assert mod._tasks({"tasks": "not-list"}) == []
    (tmp_path / "research-roadmap.yaml").write_text(
        'milestone: "2026.08.551"\ntasks: []\n', encoding="utf-8"
    )
    (tmp_path / "research-roadmap-next.yaml").write_text(
        'milestone: "2026.08.552"\ntasks: []\n', encoding="utf-8"
    )
    _next_data, next_identity = mod.load_v552_queue(tmp_path)
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
    tasks[11]["prompt"] = (
        "CONTEXT\n{project_root}\n{date}\nTASK\nCONCRETE STEPS\n"
        "Run command: x\nClaim a level solve and update the solve registry "
        "with an outer-loop solver.\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py."
    )
    checks = mod.validate_v552_queue_data(dirty, REPO, "20260814", retired_exp_ids={2091, 6410})
    assert checks["schema_validation"]["ok"] is False
    assert checks["v552_id_and_deliverable_checks"]["ok"] is False
    assert checks["v552_dependency_and_gate_checks"]["ok"] is False
    assert checks["v552_gate_field_cross_reference_checks"]["ok"] is False
    assert checks["v552_prior_failure_checks"]["ok"] is False
    assert checks["v552_exclusion_manifest_checks"]["ok"] is False
    assert checks["v552_agent_model_and_llm_policy_checks"]["ok"] is False
    assert checks["v552_arc_no_solve_checks"]["ok"] is False
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
