"""Focused tests for the V605 document-YAML and evidence contract.

Spec refs: REQ-REPORT-6911 and SCENARIO-REPORT-6911-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from carnot import experiment_6911_v605_document_yaml_evidence_contract as mod


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_TEXT = (ROOT / "scripts/roadmap_schema.py").read_text(encoding="utf-8")
AUDIT_TEXT = (ROOT / "scripts/audit_roadmap_gates.py").read_text(encoding="utf-8")


def _prompt(number: int, deliverable: str, produced_fields: set[str]) -> str:
    required_fields = mod.COMMON_PROMPT_FIELDS | produced_fields
    if number == 6920:
        required_fields = required_fields | {"model_specs"}
    fields = "; ".join(sorted(required_fields))
    model_rule = ""
    if number == 6920:
        model_rule = (
            "\nUse MODEL_SPECS: "
            + ", ".join(mod.REQUIRED_MODEL_IDS)
            + ". Use llama.cpp's native tokenizer. Never pass a GGUF repository to "
            "AutoTokenizer.from_pretrained().\n"
        )
    return (
        "CONTEXT:\nA deterministic contract fixture.\n\n"
        "EXISTING CODE TO READ FIRST:\n- CODEX.md\n\n"
        "TASK:\nCheck the fixture.\n\n"
        "CONCRETE STEPS:\n1. Check it.\n"
        f"{model_rule}\n"
        "REQUIRED ARTIFACT FIELDS: field_principles with one principle per required field; "
        f"{fields}.\n\n"
        "Run command: cd {project_root} && .venv/bin/python "
        f"scripts/experiments/{Path(deliverable).stem}.py --date {{date}}\n"
        f"{mod.FINAL_SENTENCE}\n"
    )


def _valid_inputs() -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    produced: dict[int, set[str]] = {number: set() for number in mod.EXPECTED_NUMBERS}
    for task in mod.EXPECTED_TASKS:
        for gate in task["gates"]:
            produced[mod.experiment_number(gate["upstream"])].add(gate["artifact_field"])

    tasks: list[dict[str, Any]] = []
    for expected in mod.EXPECTED_TASKS:
        number = expected["number"]
        task = {
            "id": expected["task_id"],
            "title": expected["title"],
            "milestone": mod.V605_MILESTONE,
            "deliverable": expected["deliverable"],
            "agent_type": "codex",
            "model": "gpt-5.6-sol",
            "gated_on": deepcopy(expected["gates"]),
            "prior_failures": [],
        }
        task["prompt"] = _prompt(number, expected["deliverable"], produced[number])
        tasks.append(task)
    prior_id = "exp6898-v604-evidence-admissibility-contract"
    prior_verdict = "complete_blocked_v604_evidence_admissibility_contract"
    tasks[0]["prior_failures"] = [
        {
            "experiment_id": prior_id,
            "verdict": prior_verdict,
            "addressed_by": "The fixture supplies the complete 12-task contract.",
            "retire_if_same_verdict": True,
        }
    ]
    roadmap = {
        "milestone": mod.V605_MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": mod.DESIGN_PATH.as_posix(),
        "tasks": tasks,
    }
    lines = [
        "# V605 fixture",
        "",
        f"**Milestone:** `{mod.V605_MILESTONE}`",
        "",
        "## Exact Execution Order",
        "",
        "| Order | Task ID | Title | Phase | Structured gate |",
        "|---:|---|---|---|---|",
    ]
    for expected in mod.EXPECTED_TASKS:
        lines.append(
            f"| {expected['order']} | `{expected['task_id']}` | {expected['title']} | A | "
            f"{expected['design_gate']} |"
        )
    for expected in mod.EXPECTED_TASKS:
        lines.extend(
            [
                "",
                f"### Exp{expected['number']} — {expected['title']}",
                "",
                f"**Deliverable:** `{expected['deliverable']}`",
            ]
        )
    return "\n".join(lines) + "\n", roadmap, {prior_id: {"honest_verdict": prior_verdict}}


def _evaluate(
    design: str,
    roadmap: dict[str, Any],
    priors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return mod.evaluate_contract(design, roadmap, priors, SCHEMA_TEXT, AUDIT_TEXT)


def _v604_inputs() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], str]:
    artifacts = {
        6898: {
            "status": "complete_blocked",
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_v604_evidence_admissibility_contract",
            "source_artifact_hashes": {"roadmap": {"sha256": "sha256:a"}},
        },
        6899: {
            "status": "complete",
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_live_relation_acquisition_canary_ready",
            "source_artifact_hashes": {"fixture": {"sha256": "sha256:b"}},
        },
        6900: {
            "status": "complete",
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_authentic_anchored_relation_corpus",
            "source_artifact_hashes": {"fixture": {"sha256": "sha256:c"}},
        },
        6901: {
            "status": "blocked",
            "verdict_class": "blocked",
            "honest_verdict": "complete_blocked_independent_model_relation_qualification",
            "model_relation_qualification_ready_score": 0,
            "source_artifact_hashes": {"exp6900": {"path": "results/experiment_6900_x.json"}},
        },
    }
    rechecks = {
        6898: {"loaded": True, "flags": []},
        6899: {"loaded": True, "flags": []},
        6900: {
            "loaded": True,
            "flags": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "alias"}],
        },
        6901: {"loaded": True, "flags": []},
    }
    log = "\n".join(
        f"| time | {mod.V604_SKIP_TITLES[number]} | GATE_BLOCK | Pre-emptive skip: upstream retired |"
        for number in (6902, 6903, 6904)
    )
    return artifacts, rechecks, log


def test_req_report_6911_spec_owns_fields_and_scenarios() -> None:
    """REQ-REPORT-6911 declares the contract before implementation."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6911") :]
    for name in ("TASK-IDENTITY", "PROMPT", "GATE", "PRIOR", "MODEL", "ROOTS", "V604"):
        assert f"SCENARIO-REPORT-6911-{name}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field == "field_principles"


def test_req_report_6911_valid_contract_passes() -> None:
    """REQ-REPORT-6911 accepts the exact 12-task contract."""

    design, roadmap, priors = _valid_inputs()
    result = _evaluate(design, roadmap, priors)

    assert result["passed"] is True
    assert len(result["document_yaml_parity_rows"]) == 12
    assert len(result["model_contract_rows"]) == 12
    assert all(row["passed"] for field in mod.CONTRACT_ROW_FIELDS for row in result[field])


@pytest.mark.parametrize("mismatch", ["count", "order", "id", "deliverable", "milestone"])
def test_scenario_report_6911_task_identity_fails_closed(mismatch: str) -> None:
    """SCENARIO-REPORT-6911-TASK-IDENTITY covers all identity parts."""

    design, roadmap, priors = _valid_inputs()
    if mismatch == "count":
        roadmap["tasks"].pop()
    elif mismatch == "order":
        roadmap["tasks"][2], roadmap["tasks"][3] = roadmap["tasks"][3], roadmap["tasks"][2]
    elif mismatch == "id":
        roadmap["tasks"][2]["id"] = "exp6999-wrong"
    elif mismatch == "deliverable":
        roadmap["tasks"][2]["deliverable"] = "results/experiment_6913_wrong.json"
    else:
        roadmap["tasks"][2]["milestone"] = "2026.09.604"

    result = _evaluate(design, roadmap, priors)

    assert result["passed"] is False
    assert any(not row["passed"] for row in result["document_yaml_parity_rows"])


@pytest.mark.parametrize("mismatch", ["section", "extra_run", "run_command", "ending"])
def test_scenario_report_6911_prompt_contract_is_exact(mismatch: str) -> None:
    """SCENARIO-REPORT-6911-PROMPT rejects section and ending drift."""

    design, roadmap, priors = _valid_inputs()
    prompt = roadmap["tasks"][0]["prompt"]
    if mismatch == "section":
        prompt = prompt.replace("TASK:\n", "WORK:\n")
    elif mismatch == "extra_run":
        prompt = prompt.replace("CONCRETE STEPS:\n", "CONCRETE STEPS:\nRun command: extra\n")
    elif mismatch == "run_command":
        prompt = prompt.replace("experiment_6911_", "experiment_9999_", 1)
    else:
        prompt = prompt.replace("Do NOT push.", "Do not push.")
    roadmap["tasks"][0]["prompt"] = prompt

    result = _evaluate(design, roadmap, priors)

    assert result["prompt_ending_rows"][0]["passed"] is False
    assert result["passed"] is False


@pytest.mark.parametrize("mismatch", ["field", "operator", "value", "missing", "later"])
def test_scenario_report_6911_gate_contract_is_exact(mismatch: str) -> None:
    """SCENARIO-REPORT-6911-GATE covers spelling, validity, and producer order."""

    design, roadmap, priors = _valid_inputs()
    gate = roadmap["tasks"][2]["gated_on"][0]
    if mismatch == "field":
        gate["artifact_field"] = "clean_relation_corpus_ready_typo"
    elif mismatch == "operator":
        gate["op"] = "==>"
    elif mismatch == "value":
        gate["value"] = 0
    elif mismatch == "missing":
        gate["upstream"] = "exp6999-missing"
    else:
        gate["upstream"] = roadmap["tasks"][4]["id"]

    result = _evaluate(design, roadmap, priors)
    row = result["gate_contract_rows"][0]

    assert row["passed"] is False
    assert result["passed"] is False


def test_scenario_report_6911_gate_field_must_be_declared_by_producer() -> None:
    """SCENARIO-REPORT-6911-GATE rejects an undeclared upstream field."""

    design, roadmap, priors = _valid_inputs()
    roadmap["tasks"][1]["prompt"] = roadmap["tasks"][1]["prompt"].replace(
        "clean_relation_corpus_ready_score; ", ""
    )

    result = _evaluate(design, roadmap, priors)

    assert result["gate_contract_rows"][0]["checks"]["producer_field_declared"] is False
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("experiment_id", None), ("verdict", "wrong"), ("addressed_by", ""), ("retire_if_same_verdict", False)],
)
def test_scenario_report_6911_prior_failure_shape(field: str, replacement: Any) -> None:
    """SCENARIO-REPORT-6911-PRIOR requires all four complete fields."""

    design, roadmap, priors = _valid_inputs()
    prior = roadmap["tasks"][0]["prior_failures"][0]
    if replacement is None:
        prior.pop(field)
    else:
        prior[field] = replacement

    result = _evaluate(design, roadmap, priors)

    assert field in result["prior_failure_contract_rows"][0]["failed_subfields"]
    assert result["passed"] is False


@pytest.mark.parametrize(
    "mismatch",
    ["agent_model", "forbidden_agent", "opencode", "model_id", "native_tokenizer", "autotokenizer"],
)
def test_scenario_report_6911_current_model_rules(mismatch: str) -> None:
    """SCENARIO-REPORT-6911-MODEL enforces routing and GGUF prompt rules."""

    design, roadmap, priors = _valid_inputs()
    if mismatch == "agent_model":
        roadmap["tasks"][1]["model"] = "gpt-5.5"
        index = 1
    elif mismatch == "forbidden_agent":
        roadmap["tasks"][1]["agent_type"] = "gemini"
        index = 1
    elif mismatch == "opencode":
        roadmap["tasks"][1]["agent_type"] = "opencode"
        roadmap["tasks"][1]["model"] = ""
        index = 1
    else:
        index = 9
        prompt = roadmap["tasks"][index]["prompt"]
        if mismatch == "model_id":
            prompt = prompt.replace(mod.REQUIRED_MODEL_IDS[0], "missing/model")
        elif mismatch == "native_tokenizer":
            prompt = prompt.replace("llama.cpp's native tokenizer", "a tokenizer")
        else:
            prompt = prompt.replace("Never pass a GGUF repository", "Pass a GGUF repository")
        roadmap["tasks"][index]["prompt"] = prompt

    result = _evaluate(design, roadmap, priors)

    assert result["model_contract_rows"][index]["passed"] is False
    assert result["passed"] is False


@pytest.mark.parametrize("mismatch", ["consumer", "root", "tail"])
def test_scenario_report_6911_roots_and_tail_are_ungated(mismatch: str) -> None:
    """SCENARIO-REPORT-6911-ROOTS isolates the advisory task and capstone."""

    design, roadmap, priors = _valid_inputs()
    gate = {
        "upstream": roadmap["tasks"][0]["id"],
        "artifact_field": "v605_execution_contract_ready_score",
        "op": "==",
        "value": 1,
    }
    if mismatch == "consumer":
        roadmap["tasks"][2]["gated_on"] = [gate]
    elif mismatch == "root":
        roadmap["tasks"][8]["gated_on"] = [gate]
    else:
        roadmap["tasks"][-1]["gated_on"] = [gate]

    result = _evaluate(design, roadmap, priors)

    assert result["passed"] is False
    assert any(
        not row["passed"]
        for row in result["independent_root_rows"] + result["ungated_tail_rows"]
    )


def test_scenario_report_6911_v604_states_stay_separate() -> None:
    """SCENARIO-REPORT-6911-V604 preserves flags, blocks, skips, and taint."""

    artifacts, rechecks, log = _v604_inputs()
    result = mod.evaluate_v604_evidence(artifacts, rechecks, log)
    by_id = {row["experiment_id"]: row for row in result["v604_artifact_rows"]}

    assert by_id[6899]["evidence_state"] == "admissible"
    assert by_id[6900]["evidence_state"] == "flagged"
    assert by_id[6901]["evidence_state"] == "blocked"
    assert [by_id[number]["evidence_state"] for number in (6902, 6903, 6904)] == [
        "skipped",
        "skipped",
        "skipped",
    ]
    assert all(by_id[number]["science_success"] is False for number in range(6898, 6905))
    assert result["flags"][0]["kind"] == "TAUTOLOGY"
    assert [row["experiment_id"] for row in result["skips"]] == [6902, 6903, 6904]
    assert result["dependency_taint_rows"][0]["source_experiment_id"] == 6900


def test_req_report_6911_missing_recheck_and_skip_record_fail_closed() -> None:
    """REQ-REPORT-6911 keeps missing verifier and conductor evidence explicit."""

    artifacts, rechecks, _log = _v604_inputs()
    rechecks[6899] = {"loaded": False, "flags": []}
    result = mod.evaluate_v604_evidence(artifacts, rechecks, "")
    by_id = {row["experiment_id"]: row for row in result["v604_artifact_rows"]}

    assert by_id[6899]["evidence_state"] == "unverified"
    assert by_id[6902]["evidence_state"] == "missing_skip_record"
    assert result["warnings"]


def test_req_report_6911_warning_flags_and_nonmapping_sources_are_safe() -> None:
    """REQ-REPORT-6911 keeps warnings separate and ignores malformed source maps."""

    artifacts, rechecks, log = _v604_inputs()
    artifacts[6901]["source_artifact_hashes"] = []
    rechecks[6899] = {
        "loaded": True,
        "flags": [{"kind": "METHOD_NOTE", "severity": "warn", "detail": "fixture"}],
    }

    result = mod.evaluate_v604_evidence(artifacts, rechecks, log)

    assert result["warnings"][0]["kind"] == "METHOD_NOTE"
    assert result["dependency_taint_rows"] == []


def test_req_report_6911_repo_sources_emit_valid_blocked_receipt() -> None:
    """REQ-REPORT-6911 audits the active copy but preserves missing next YAML."""

    artifact = mod.build_artifact(ROOT, "20260903")

    assert artifact["task_count"] == 12
    assert artifact["experiment_range"] == [6911, 6922]
    assert len(artifact["document_task_rows"]) == 12
    assert len(artifact["yaml_task_rows"]) == 12
    assert artifact["preconditions_checked"]["passed"] is False
    assert artifact["v605_execution_contract_ready_score"] == 0
    assert artifact["status"] == "complete_blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert any(not row["passed"] for row in artifact["model_contract_rows"])
    assert mod.validate_artifact(artifact) == []


def test_req_report_6911_missing_sources_still_emit_complete_block(tmp_path: Path) -> None:
    """REQ-REPORT-6911 emits all required fields when preconditions fail."""

    artifact = mod.build_artifact(tmp_path, "20260903", verifier=lambda _path: {})

    assert artifact["gate_check_summary"]["failed_check"] == "preconditions"
    assert artifact["gate_check_summary"]["expected"] == "all required sources readable"
    assert artifact["gate_check_summary"]["observed"]
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert mod.validate_artifact(artifact) == []


def test_req_report_6911_validation_and_checksum_are_cold() -> None:
    """REQ-REPORT-6911 recomputes required fields and advisory readiness."""

    artifact = mod.build_artifact(ROOT, "20260903")
    broken = deepcopy(artifact)
    broken.pop("rows")
    broken["field_principles"].pop("task_count")
    broken["inference_substrate"] = "llm"
    broken["verifier_is_oracle"] = True
    broken["verdict_class"] = "unknown"
    broken["honest_verdict"] = "blocked"
    broken["v605_execution_contract_ready_score"] = 1
    broken["task_count"] = 13
    broken["science_claim_approved"] = True
    broken["reproducibility_checksum"] = "sha256:bad"

    errors = mod.validate_artifact(broken)

    assert "missing_required_fields:rows" in errors
    assert "field_principles_missing" in errors
    assert "invalid_inference_substrate" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "invalid_verdict_class" in errors
    assert "honest_verdict_not_terminal" in errors
    assert "readiness_recomputation_mismatch" in errors
    assert "task_identity_summary_mismatch" in errors
    assert "advisory_contract_cannot_approve_science" in errors
    assert "reproducibility_checksum_mismatch" in errors

    ready_shaped_wrong = deepcopy(artifact)
    for check in ready_shaped_wrong["gate_check_summary"]["checks"]:
        check["passed"] = True
    ready_shaped_wrong["v605_execution_contract_ready_score"] = 1
    ready_shaped_wrong["reproducibility_checksum"] = mod.reproducibility_checksum(
        ready_shaped_wrong
    )
    assert "ready_terminal_shape_mismatch" in mod.validate_artifact(ready_shaped_wrong)


def test_req_report_6911_parsers_and_checksum_helpers_fail_closed() -> None:
    """REQ-REPORT-6911 rejects malformed source shapes deterministically."""

    design, roadmap, _priors = _valid_inputs()
    assert mod.parse_design(design)["milestone"] == mod.V605_MILESTONE
    assert mod.parse_roadmap(roadmap)["milestone"] == mod.V605_MILESTONE
    ignored = design.replace(
        "| 1 |",
        "| 99 | `exp9999-ignored` | Ignored | A | none |\n| 1 |",
        1,
    )
    assert len(mod.parse_design(ignored)["tasks"]) == 12
    with pytest.raises(ValueError):
        mod.parse_design("missing")
    with pytest.raises(ValueError):
        mod.parse_design(f"**Milestone:** `{mod.V605_MILESTONE}`\n")
    with pytest.raises(ValueError):
        mod.parse_roadmap([])
    with pytest.raises(ValueError):
        mod.parse_roadmap({})
    with pytest.raises(ValueError):
        mod.parse_roadmap({"tasks": ["bad"]})
    with pytest.raises(ValueError):
        mod.parse_roadmap({"tasks": [{"gated_on": "bad"}]})
    assert mod.current_model_rule(AUDIT_TEXT) == ("gpt-5.6-sol", "gemini")
    with pytest.raises(ValueError):
        mod.current_model_rule("def audit(): pass")
    with pytest.raises(ValueError):
        mod.current_model_rule('agent_type == "codex" and model != "gpt-5.6-sol"')
    first = {"duration_s": 1.0, "reproducibility_checksum": "", "value": 2}
    second = {"duration_s": 9.0, "reproducibility_checksum": "bad", "value": 2}
    assert mod.reproducibility_checksum(first) == mod.reproducibility_checksum(second)


def test_req_report_6911_private_load_failures_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6911 reports malformed JSON, prior rows, schema, and verifier exits."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod._read_artifact_summary(malformed)[1].startswith("invalid_json:exit_")
    evidence, rows = mod._load_prior_evidence(
        tmp_path, [{"prior_failures": [None]}], ""
    )
    assert evidence == {}
    assert rows == []

    roadmap_path = tmp_path / "roadmap.yaml"
    roadmap_path.write_text("milestone: bad\ntasks: []\n", encoding="utf-8")
    checks = mod._external_validation_checks(
        tmp_path, roadmap_path, {"milestone": "bad", "tasks": []}
    )
    assert checks[0]["passed"] is False

    class Result:
        returncode = 2
        stdout = ""
        stderr = "failed"

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError):
        mod._default_verifier(malformed)


def test_req_report_6911_verifier_exception_is_a_precondition(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6911 records a verifier exception instead of losing the task row."""

    path = tmp_path / mod.V604_ARTIFACT_PATHS[6898]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "complete_blocked",
                "verdict_class": "blocked",
                "honest_verdict": "complete_blocked_v604_evidence_admissibility_contract",
            }
        ),
        encoding="utf-8",
    )

    def failing_verifier(_path: Path) -> dict[str, Any]:
        raise RuntimeError("fixture verifier failure")

    artifact = mod.build_artifact(tmp_path, "20260903", verifier=failing_verifier)

    assert any(
        row.get("error") == "verifier_failed:RuntimeError"
        for row in artifact["preconditions_checked"]["observed"]
    )


def test_req_report_6911_writer_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-6911 writes only the requested append-only receipt path."""

    output = tmp_path / "receipt.json"
    mod.write_json_atomic(output, {"z": 1, "a": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 2, "z": 1}
    assert mod.main(["--date", "bad", "--output", str(output)]) == 2
    assert mod.main(["--date", "20260903", "--output", str(output)]) == 0
    assert mod.validate_artifact(json.loads(output.read_text(encoding="utf-8"))) == []
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["invalid"])
    assert mod.main(["--date", "20260903", "--output", str(output)]) == 1


def test_req_report_6911_script_entry_point_runs_real_audit(tmp_path: Path) -> None:
    """REQ-REPORT-6911 supports the exact required command through its wrapper."""

    output = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/experiment_6911_v605_document_yaml_evidence_contract.py",
            "--date",
            "20260903",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    by_id = {row["experiment_id"]: row for row in artifact["v604_artifact_rows"]}
    assert by_id[6900]["evidence_state"] == "flagged"
    assert by_id[6901]["evidence_state"] == "blocked"
    assert [by_id[number]["evidence_state"] for number in (6902, 6903, 6904)] == [
        "skipped",
        "skipped",
        "skipped",
    ]


def test_req_report_6911_yaml_round_trip_is_structural() -> None:
    """REQ-REPORT-6911 evaluates YAML value types after a real round trip."""

    design, roadmap, priors = _valid_inputs()
    loaded = yaml.safe_load(yaml.safe_dump(roadmap, sort_keys=False))
    assert _evaluate(design, loaded, priors)["passed"] is True
