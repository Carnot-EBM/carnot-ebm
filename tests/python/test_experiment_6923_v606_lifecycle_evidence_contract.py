"""Focused tests for the V606 lifecycle and evidence contract.

Spec refs: REQ-REPORT-6923 and SCENARIO-REPORT-6923-*.
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

from carnot import experiment_6923_v606_lifecycle_evidence_contract as mod


ROOT = Path(__file__).resolve().parents[2]
DESIGN_TEXT = (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
SCHEMA_TEXT = (ROOT / mod.SCHEMA_PATH).read_text(encoding="utf-8")
AUDIT_TEXT = (ROOT / mod.GATE_AUDIT_PATH).read_text(encoding="utf-8")


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _prompt(number: int, deliverable: str, produced_fields: set[str]) -> str:
    fields = mod.COMMON_PROMPT_FIELDS | produced_fields
    model_text = ""
    if number in mod.MODEL_BEARING_NUMBERS:
        model_ids = mod.REQUIRED_MODEL_IDS if number != 6935 else mod.REQUIRED_MODEL_IDS[:1]
        production_model = " plus Qwen3.8-27B" if number == 6935 else ""
        model_text = (
            "\nDeclare MODEL_SPECS with "
            + ", ".join(model_ids)
            + production_model
            + ". Use llama.cpp's embedded tokenizer. Never pass a GGUF repository to "
            "AutoTokenizer.from_pretrained().\n"
        )
        fields.add("model_specs")
    declared = "; ".join(sorted(fields))
    return (
        "CONTEXT:\nA deterministic contract fixture.\n\n"
        "EXISTING CODE TO READ FIRST:\n- CODEX.md\n\n"
        "TASK:\nAudit one fixture.\n\n"
        "CONCRETE STEPS:\n0. PRECONDITIONS: require fixture inputs.\n"
        f"{model_text}\n"
        "REQUIRED ARTIFACT FIELDS: field_principles with one principle per required field; "
        f"{declared}.\n\n"
        "Run command: cd {project_root} && .venv/bin/python "
        f"scripts/experiments/{Path(deliverable).stem}.py --date {{date}}\n"
        f"{mod.FINAL_SENTENCE}\n"
    )


def _valid_roadmap() -> dict[str, Any]:
    produced = {number: set() for number in mod.EXPECTED_NUMBERS}
    for expected in mod.EXPECTED_TASKS:
        for gate in expected["gates"]:
            produced[mod.experiment_number(gate["upstream"])].add(gate["artifact_field"])
    tasks = []
    for expected in mod.EXPECTED_TASKS:
        number = expected["number"]
        task = {
            "id": expected["task_id"],
            "title": expected["title"],
            "track": "infrastructure" if number in mod.INFRASTRUCTURE_NUMBERS else "research",
            "priority": "critical",
            "agent_type": "codex",
            "model": "gpt-5.6-sol",
            "requires_gpu": False,
            "milestone": mod.V606_MILESTONE,
            "deliverable": expected["deliverable"],
            "gated_on": deepcopy(expected["gates"]),
            "prior_failures": [],
            "prompt": _prompt(number, expected["deliverable"], produced[number]),
        }
        tasks.append(task)
    return {
        "milestone": mod.V606_MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": mod.DESIGN_PATH.as_posix(),
        "tasks": tasks,
    }


def _evaluate(roadmap: dict[str, Any]) -> dict[str, Any]:
    return mod.evaluate_contract(DESIGN_TEXT, roadmap, SCHEMA_TEXT, AUDIT_TEXT, {})


def test_req_report_6923_spec_precedes_implementation() -> None:
    """REQ-REPORT-6923 owns all focused lifecycle scenarios."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6923") :]
    for name in (
        "LIFECYCLE-PRESTAGED",
        "LIFECYCLE-ACTIVATED",
        "LIFECYCLE-MISSING",
        "LIFECYCLE-MISMATCHED",
        "LIFECYCLE-AMBIGUOUS",
        "PARITY",
        "GATES",
        "PRIORS",
        "PROMPTS",
        "ROUTING",
        "SLOTS-ROOTS",
    ):
        assert f"SCENARIO-REPORT-6923-{name}" in section


def test_scenario_report_6923_lifecycle_prestaged(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6923-LIFECYCLE-PRESTAGED selects only next."""

    old = {"milestone": "2026.09.605", "tasks": [{"id": "old"}]}
    _write_yaml(tmp_path / mod.ACTIVE_ROADMAP_PATH, old)
    _write_yaml(tmp_path / mod.NEXT_ROADMAP_PATH, _valid_roadmap())

    result = mod.resolve_roadmap_lifecycle(tmp_path)

    assert result["state"] == "pre_staged"
    assert result["selected_path"] == mod.NEXT_ROADMAP_PATH
    assert result["roadmap"]["tasks"][0]["id"].startswith("exp6923-")
    assert sum(row["selected"] for row in result["rows"]) == 1


def test_scenario_report_6923_lifecycle_activated(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6923-LIFECYCLE-ACTIVATED accepts the active copy."""

    _write_yaml(tmp_path / mod.ACTIVE_ROADMAP_PATH, _valid_roadmap())

    result = mod.resolve_roadmap_lifecycle(tmp_path)

    assert result["state"] == "activated"
    assert result["selected_path"] == mod.ACTIVE_ROADMAP_PATH
    assert result["passed"] is True


@pytest.mark.parametrize(
    ("state", "active", "next_"),
    [
        ("missing", None, None),
        ("mismatched", {"milestone": "2026.09.605", "tasks": []}, None),
        ("ambiguous", _valid_roadmap(), _valid_roadmap()),
    ],
)
def test_scenario_report_6923_invalid_lifecycle_states(
    tmp_path: Path,
    state: str,
    active: dict[str, Any] | None,
    next_: dict[str, Any] | None,
) -> None:
    """SCENARIO-REPORT-6923-LIFECYCLE-* blocks missing, drift, and ambiguity."""

    if active is not None:
        _write_yaml(tmp_path / mod.ACTIVE_ROADMAP_PATH, active)
    if next_ is not None:
        _write_yaml(tmp_path / mod.NEXT_ROADMAP_PATH, next_)

    result = mod.resolve_roadmap_lifecycle(tmp_path)

    assert result["state"] == state
    assert result["selected_path"] is None
    assert result["roadmap"] is None
    assert result["passed"] is False


def test_req_report_6923_valid_contract_passes() -> None:
    """REQ-REPORT-6923 accepts the exact documented 14-task contract."""

    result = _evaluate(_valid_roadmap())

    assert result["passed"] is True
    assert len(result["document_yaml_parity_rows"]) == 14
    assert len(result["prompt_ending_rows"]) == 14
    assert len(result["model_contract_rows"]) == 14
    assert len(result["infrastructure_slot_rows"]) == 2
    assert len(result["sota_ingestion_slot_rows"]) == 1
    assert len(result["ungated_tail_rows"]) == 1
    assert all(row["passed"] for row in result["checks"])


@pytest.mark.parametrize(
    "mismatch",
    ["id", "title", "order", "deliverable", "gate", "milestone", "infrastructure", "sota"],
)
def test_scenario_report_6923_document_yaml_parity_is_exact(mismatch: str) -> None:
    """SCENARIO-REPORT-6923-PARITY rejects identity and role drift."""

    roadmap = _valid_roadmap()
    if mismatch == "id":
        roadmap["tasks"][4]["id"] = "exp6927-alias"
    elif mismatch == "title":
        roadmap["tasks"][4]["title"] = "Changed title"
    elif mismatch == "order":
        roadmap["tasks"][4], roadmap["tasks"][5] = roadmap["tasks"][5], roadmap["tasks"][4]
    elif mismatch == "deliverable":
        roadmap["tasks"][4]["deliverable"] = "results/changed.json"
    elif mismatch == "gate":
        roadmap["tasks"][4]["gated_on"][0]["value"] = 0
    elif mismatch == "milestone":
        roadmap["tasks"][4]["milestone"] = "2026.09.605"
    elif mismatch == "infrastructure":
        roadmap["tasks"][1]["track"] = "research"
    else:
        roadmap["tasks"][2]["title"] = "Ordinary research map"

    result = _evaluate(roadmap)

    assert result["passed"] is False
    assert any(not row["passed"] for row in result["checks"])


@pytest.mark.parametrize("mismatch", ["section", "duplicate_run", "run", "ending"])
def test_scenario_report_6923_prompt_contract_is_exact(mismatch: str) -> None:
    """SCENARIO-REPORT-6923-PROMPTS rejects structure and ending drift."""

    roadmap = _valid_roadmap()
    prompt = roadmap["tasks"][0]["prompt"]
    if mismatch == "section":
        prompt = prompt.replace("TASK:", "WORK:")
    elif mismatch == "duplicate_run":
        prompt = prompt.replace("Run command:", "Run command:\nRun command:")
    elif mismatch == "run":
        prompt = prompt.replace("experiment_6923_", "experiment_9999_")
    else:
        prompt = prompt.replace(mod.FINAL_SENTENCE, "Do not push.")
    roadmap["tasks"][0]["prompt"] = prompt

    row = _evaluate(roadmap)["prompt_ending_rows"][0]

    assert row["passed"] is False


@pytest.mark.parametrize("mismatch", ["alias", "missing", "later", "field", "operator"])
def test_scenario_report_6923_gate_contract_is_exact(mismatch: str) -> None:
    """SCENARIO-REPORT-6923-GATES rejects every unsafe gate form."""

    roadmap = _valid_roadmap()
    gate = roadmap["tasks"][4]["gated_on"][0]
    if mismatch == "alias":
        gate["upstream"] = "exp6926"
    elif mismatch == "missing":
        gate["upstream"] = "exp9999-missing"
    elif mismatch == "later":
        gate["upstream"] = roadmap["tasks"][5]["id"]
    elif mismatch == "field":
        gate["artifact_field"] = "fixture_ready"
    else:
        gate["op"] = "==="

    row = _evaluate(roadmap)["gate_contract_rows"][4]

    assert row["passed"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", None),
        ("verdict", None),
        ("addressed_by", ""),
        ("retire_if_same_verdict", False),
    ],
)
def test_scenario_report_6923_prior_failure_shape(field: str, value: Any) -> None:
    """SCENARIO-REPORT-6923-PRIORS requires complete retirement rows."""

    roadmap = _valid_roadmap()
    prior = {
        "experiment_id": "exp6911-v605-document-yaml-evidence-contract",
        "verdict": "complete_blocked_v605_document_yaml_evidence_contract",
        "addressed_by": "The lifecycle resolver accepts the activated file.",
        "retire_if_same_verdict": True,
    }
    if value is None:
        prior.pop(field)
    else:
        prior[field] = value
    roadmap["tasks"][0]["prior_failures"] = [prior]

    row = _evaluate(roadmap)["prior_failure_contract_rows"][0]

    assert field in row["failed_subfields"]
    assert row["passed"] is False


@pytest.mark.parametrize(
    "mismatch", ["codex", "gemini", "claude", "model_id", "tokenizer", "loader"]
)
def test_scenario_report_6923_current_model_contract(mismatch: str) -> None:
    """SCENARIO-REPORT-6923-ROUTING enforces model and vendor rules."""

    roadmap = _valid_roadmap()
    index = 4
    if mismatch == "codex":
        roadmap["tasks"][index]["model"] = "gpt-5.5"
    elif mismatch == "gemini":
        roadmap["tasks"][index]["agent_type"] = "gemini"
    elif mismatch == "claude":
        roadmap["tasks"][index]["agent_type"] = "claude"
        roadmap["tasks"][index]["model"] = "gpt-5.6-sol"
    else:
        prompt = roadmap["tasks"][index]["prompt"]
        if mismatch == "model_id":
            prompt = prompt.replace(mod.REQUIRED_MODEL_IDS[0], "missing/model")
        elif mismatch == "tokenizer":
            prompt = prompt.replace("llama.cpp's embedded tokenizer", "another tokenizer")
        else:
            prompt = prompt.replace("Never pass a GGUF repository", "Pass a GGUF repository")
        roadmap["tasks"][index]["prompt"] = prompt

    row = _evaluate(roadmap)["model_contract_rows"][index]

    assert row["passed"] is False
    assert mod.current_model_rule(AUDIT_TEXT) == ("gpt-5.6-sol", "gemini")


@pytest.mark.parametrize("mismatch", ["root", "contract_consumer", "sota_consumer", "tail"])
def test_scenario_report_6923_roots_and_tail_are_ungated(mismatch: str) -> None:
    """SCENARIO-REPORT-6923-SLOTS-ROOTS preserves independent roots."""

    roadmap = _valid_roadmap()
    gate = {
        "upstream": roadmap["tasks"][0]["id"],
        "artifact_field": "v606_execution_contract_ready_score",
        "op": "==",
        "value": 1,
    }
    if mismatch == "root":
        roadmap["tasks"][3]["gated_on"] = [gate]
    elif mismatch == "contract_consumer":
        roadmap["tasks"][4]["gated_on"] = [gate]
    elif mismatch == "sota_consumer":
        gate["upstream"] = roadmap["tasks"][2]["id"]
        gate["artifact_field"] = "v606_sota_ingestion_complete_score"
        roadmap["tasks"][4]["gated_on"] = [gate]
    else:
        roadmap["tasks"][-1]["gated_on"] = [gate]

    result = _evaluate(roadmap)

    assert result["passed"] is False
    assert any(
        not row["passed"] for row in result["independent_root_rows"] + result["ungated_tail_rows"]
    )


def test_req_report_6923_repo_sources_emit_valid_blocked_receipt() -> None:
    """REQ-REPORT-6923 catches the activated V606 task-count mismatch."""

    before = (ROOT / mod.ACTIVE_ROADMAP_PATH).read_bytes()
    artifact = mod.build_artifact(ROOT, "20260903")

    assert mod.validate_artifact(artifact) == []
    assert artifact["lifecycle_resolution_rows"][0]["lifecycle_state"] == "activated"
    assert artifact["task_count"] == 4
    assert artifact["experiment_range"] == [6923, 6926]
    assert artifact["v606_execution_contract_ready_score"] == 0
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["science_claim_approved"] is False
    assert (ROOT / mod.ACTIVE_ROADMAP_PATH).read_bytes() == before


def test_req_report_6923_missing_inputs_emit_complete_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6923-LIFECYCLE-MISSING still emits the full schema."""

    artifact = mod.build_artifact(tmp_path, "20260903")

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["task_count"] == 0
    assert artifact["experiment_range"] == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "preconditions"
    assert mod.validate_artifact(artifact) == []


def test_req_report_6923_validation_recomputes_advisory_fields() -> None:
    """REQ-REPORT-6923 validates terminal fields, principles, and checksum."""

    artifact = mod.build_artifact(ROOT, "20260903")
    mutations = {
        "field_principles": {},
        "inference_substrate": "llm",
        "verifier_is_oracle": True,
        "verdict_class": "success",
        "honest_verdict": "blocked_without_terminal_prefix",
        "task_count": 99,
        "science_claim_approved": True,
        "v606_execution_contract_ready_score": 1,
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed.pop("rows")
    changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
    assert "missing_required_fields:rows" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(changed)


def test_req_report_6923_parser_and_cli_edges(tmp_path: Path) -> None:
    """REQ-REPORT-6923 rejects malformed sources and supports its wrapper."""

    with pytest.raises(ValueError):
        mod.parse_design("missing")
    with pytest.raises(ValueError):
        mod.parse_roadmap([])
    with pytest.raises(ValueError):
        mod.parse_roadmap({"tasks": ["bad"]})
    with pytest.raises(ValueError):
        mod.current_model_rule("def audit(): pass")
    assert mod.main(["--date", "bad"]) == 2

    output = tmp_path / "artifact.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/experiment_6923_v606_lifecycle_evidence_contract.py",
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
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == mod.BLOCKED_VERDICT
    assert mod.validate_artifact(written) == []


def test_req_report_6923_defensive_source_edges(tmp_path: Path) -> None:
    """REQ-REPORT-6923 keeps malformed lifecycle and source states explicit."""

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text("tasks: [", encoding="utf-8")
    assert mod.resolve_roadmap_lifecycle(tmp_path)["state"] == "unresolved"
    assert mod._parse_document_gate("not a gate", {}) == [{"unparsed_document_gate": "not a gate"}]
    assert mod._parse_document_gate("Exp6926 `field == word`", {})[0]["value"] == "word"
    with pytest.raises(ValueError):
        mod.parse_design(
            "**Milestone:** `2026.09.606`\n"
            "| 1 | `exp1-outside` | Outside | `results/outside.json` | None; root |\n"
        )
    with pytest.raises(ValueError):
        mod.parse_roadmap({})
    with pytest.raises(ValueError):
        mod.parse_roadmap({"tasks": [{"gated_on": "bad"}]})
    assert mod._routing_ok("opencode", "model", "gpt-5.6-sol", "gemini") is True
    assert mod._routing_ok("unknown", "model", "gpt-5.6-sol", "gemini") is False
    assert mod._date_argument("20260903") == "20260903"


def test_req_report_6923_prior_and_validator_failure_edges(tmp_path: Path) -> None:
    """REQ-REPORT-6923 records malformed prior evidence and schema failures."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_1_invalid.json").write_text("[", encoding="utf-8")
    evidence, rows = mod._load_prior_evidence(
        tmp_path,
        [
            {"prior_failures": ["bad"]},
            {"prior_failures": [{"experiment_id": "exp1-invalid"}]},
        ],
    )
    assert evidence == {}
    assert rows[0]["error"].startswith("invalid_json:")

    roadmap_path = tmp_path / "invalid-roadmap.yaml"
    _write_yaml(roadmap_path, {"milestone": mod.V606_MILESTONE, "tasks": []})
    checks = mod._external_validation_checks(
        ROOT,
        roadmap_path,
        {"milestone": mod.V606_MILESTONE, "tasks": []},
    )
    assert checks[0]["passed"] is False


def test_req_report_6923_ready_validation_and_direct_main_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6923 validates ready receipts and both direct CLI outcomes."""

    artifact = mod.build_artifact(ROOT, "20260903")
    for check in artifact["gate_check_summary"]["checks"]:
        check["passed"] = True
    artifact["v606_execution_contract_ready_score"] = 1
    artifact["status"] = "complete"
    artifact["verdict_class"] = "null"
    artifact["honest_verdict"] = mod.READY_VERDICT
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    artifact["status"] = "bad"
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    assert "ready_terminal_shape_mismatch" in mod.validate_artifact(artifact)

    minimal = {
        "v606_execution_contract_ready_score": 0,
        "honest_verdict": mod.BLOCKED_VERDICT,
    }
    written: list[tuple[Path, dict[str, Any]]] = []
    monkeypatch.setattr(mod, "build_artifact", lambda _root, _date: minimal)
    monkeypatch.setattr(mod, "write_json_atomic", lambda path, value: written.append((path, value)))
    monkeypatch.setattr(mod, "validate_artifact", lambda _value: [])
    output = tmp_path / "direct.json"
    assert mod.main(["--date", "20260903", "--output", str(output)]) == 0
    assert written == [(output, minimal)]
    monkeypatch.setattr(mod, "validate_artifact", lambda _value: ["invalid"])
    assert mod.main(["--date", "20260903", "--output", str(output)]) == 1
