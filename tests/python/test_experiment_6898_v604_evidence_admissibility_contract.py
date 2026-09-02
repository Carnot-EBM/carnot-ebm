"""Focused tests for the V604 manifest and evidence-admission receipt.

Spec refs: REQ-REPORT-6898 and SCENARIO-REPORT-6898-*.
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

from carnot import experiment_6898_v604_evidence_admissibility_contract as mod


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_TEXT = (ROOT / "scripts/roadmap_schema.py").read_text(encoding="utf-8")
GATE_TEXT = (ROOT / "scripts/conductor_gates.py").read_text(encoding="utf-8")


def _prompt(deliverable: str, produced_fields: set[str]) -> str:
    stem = Path(deliverable).stem
    fields = "; ".join(sorted(produced_fields | {"honest_verdict"}))
    return (
        "CONTEXT:\nA deterministic fixture.\n\n"
        "EXISTING CODE TO READ FIRST:\n- CODEX.md\n\n"
        "TASK:\nValidate the fixture.\n\n"
        "CONCRETE STEPS:\n1. Validate it.\n\n"
        f"REQUIRED ARTIFACT FIELDS: {fields}\n\n"
        f"Run command: cd {{project_root}} && .venv/bin/python scripts/experiments/{stem}.py --date {{date}}\n"
        f"{mod.FINAL_SENTENCE}\n"
    )


def _design(roadmap: dict[str, Any]) -> str:
    sections = [
        "# Carnot Research Roadmap V604 Fixture",
        "",
        f"**Milestone:** `{roadmap['milestone']}`",
        "",
    ]
    for task in roadmap["tasks"]:
        number = mod.experiment_number(task["id"])
        sections.extend(
            [
                f"### Exp{number}: {task['title']}",
                "",
                f"**Deliverable:** `{task['deliverable']}`",
                "",
            ]
        )
    sections.extend(["## Dependency Graph", "", "| Downstream task | Upstream field | Condition |"])
    sections.append("|---|---|---|")
    for task in roadmap["tasks"]:
        for gate in task.get("gated_on", []):
            sections.append(
                f"| Exp{mod.experiment_number(task['id'])} | "
                f"`exp{mod.experiment_number(gate['upstream'])}.{gate['artifact_field']}` | "
                f"`{gate['op']} {gate['value']}` |"
            )
    return "\n".join(sections) + "\n"


def _valid_inputs() -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    gates_by_number = {
        6900: [
            {
                "upstream": "exp6899-live-relation-acquisition-canary",
                "artifact_field": "relation_canary_ready_score",
                "op": "==",
                "value": 1,
            }
        ],
        6907: [
            {
                "upstream": "exp6906-arc-trace-root",
                "artifact_field": "arc_trace_ready_score",
                "op": ">=",
                "value": 1,
            }
        ],
    }
    produced = {
        "exp6899-live-relation-acquisition-canary": {"relation_canary_ready_score"},
        "exp6906-arc-trace-root": {"arc_trace_ready_score"},
    }
    tasks: list[dict[str, Any]] = []
    for number in mod.EXPECTED_NUMBERS:
        slug = {
            6898: "v604-evidence-admissibility-contract",
            6899: "live-relation-acquisition-canary",
            6905: "independent-science-root",
            6906: "arc-trace-root",
            6910: "independent-v604-capstone",
        }.get(number, f"fixture-task-{number}")
        task_id = f"exp{number}-{slug}"
        deliverable = f"results/experiment_{number}_{slug.replace('-', '_')}.json"
        task = {
            "id": task_id,
            "title": f"Fixture task {number}",
            "milestone": mod.V604_MILESTONE,
            "deliverable": deliverable,
            "gated_on": deepcopy(gates_by_number.get(number, [])),
            "prior_failures": [],
        }
        task["prompt"] = _prompt(deliverable, produced.get(task_id, set()))
        tasks.append(task)
    tasks[0]["prior_failures"] = [
        {
            "experiment_id": "exp6885-v603-executable-manifest-branch-contract",
            "verdict": "complete_blocked_v603_executable_manifest_branch_contract",
            "addressed_by": "The V604 fixture has the complete task graph.",
            "retire_if_same_verdict": True,
        }
    ]
    roadmap = {
        "milestone": mod.V604_MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": mod.DESIGN_PATH.as_posix(),
        "tasks": tasks,
    }
    priors = {
        "exp6885-v603-executable-manifest-branch-contract": {
            "honest_verdict": "complete_blocked_v603_executable_manifest_branch_contract"
        }
    }
    return _design(roadmap), roadmap, priors


def _evaluate(
    design: str,
    roadmap: dict[str, Any],
    priors: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return mod.evaluate_manifest_contract(
        design,
        roadmap,
        priors or {},
        SCHEMA_TEXT,
        GATE_TEXT,
    )


def _artifact_inputs() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    artifacts = {
        6885: {
            "experiment_id": 6885,
            "status": "complete_blocked",
            "honest_verdict": "complete_blocked_v603_executable_manifest_branch_contract",
            "source_artifact_hashes": {"roadmap": {"sha256": "sha256:a"}},
        },
        6887: {
            "experiment_id": 6887,
            "status": "complete",
            "honest_verdict": "complete_positive_three_family_relation_corpus",
            "source_artifact_hashes": {"fixture": {"sha256": "sha256:b"}},
        },
        6888: {
            "experiment_id": 6888,
            "status": "complete",
            "honest_verdict": "complete_circular_positive_independent_relation_qualification",
            "source_artifact_hashes": {"exp6887": {"sha256": "sha256:c"}},
            "eligible_arm_rows": [
                {"arm": "rule:anchored_lexical_v1", "passed": True},
                {"arm": "gguf:one", "passed": False},
            ],
        },
    }
    reports = {
        6885: {"loaded": True, "flag_count": 0, "max_severity": -1, "flags": []},
        6887: {
            "loaded": True,
            "flag_count": 1,
            "max_severity": 2,
            "flags": [{"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "too fast"}],
        },
        6888: {
            "loaded": True,
            "flag_count": 1,
            "max_severity": 1,
            "flags": [{"kind": "SUBSTRATE_NO_LLM_BY_NAME", "severity": "warn"}],
        },
    }
    return artifacts, reports


def _write_complete_fixture_root(root: Path) -> dict[int, dict[str, Any]]:
    """Write the smallest complete primary-source tree under a temporary root."""

    design, roadmap, _priors = _valid_inputs()
    artifacts, _reports = _artifact_inputs()
    paths = {
        mod.DESIGN_PATH: design,
        mod.ROADMAP_PATH: yaml.safe_dump(roadmap, sort_keys=False),
        mod.EXCLUSION_PATH: "{}\n",
        mod.SCHEMA_PATH: SCHEMA_TEXT,
        mod.GATE_PATH: GATE_TEXT,
        mod.PRIOR_VALIDATOR_PATH: "# deterministic prior validator fixture\n",
        mod.GATE_AUDIT_PATH: "# deterministic gate audit fixture\n",
        mod.ADVERSARIAL_PATH: "# deterministic adversarial fixture\n",
    }
    for path, text in paths.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    for number, path in mod.V603_ARTIFACT_PATHS.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(artifacts[number]), encoding="utf-8")
    return artifacts


def test_req_report_6898_spec_owns_fields_and_scenarios() -> None:
    """REQ-REPORT-6898 declares the full contract before implementation."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6898") :]
    for scenario in (
        "SCENARIO-REPORT-6898-COUNT-ORDER",
        "SCENARIO-REPORT-6898-IDENTITY",
        "SCENARIO-REPORT-6898-PROMPT-END",
        "SCENARIO-REPORT-6898-GATE-FIELD",
        "SCENARIO-REPORT-6898-MISSING-UPSTREAM",
        "SCENARIO-REPORT-6898-PRIOR-FAILURE",
        "SCENARIO-REPORT-6898-FLAGGED-SOURCE",
        "SCENARIO-REPORT-6898-DEPENDENCY-TAINT",
        "SCENARIO-REPORT-6898-UNGATED-TAIL",
    ):
        assert scenario in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section or field == "field_principles"


def test_req_report_6898_valid_manifest_contract_passes() -> None:
    """REQ-REPORT-6898 accepts the exact 13-task isolated graph."""

    design, roadmap, priors = _valid_inputs()
    result = _evaluate(design, roadmap, priors)

    assert result["passed"] is True
    assert len(result["document_yaml_parity_rows"]) == 13
    assert all(row["passed"] for row in result["document_yaml_parity_rows"])
    assert all(row["passed"] for row in result["gate_contract_rows"])
    assert all(row["passed"] for row in result["prior_failure_contract_rows"])
    assert all(row["passed"] for row in result["prompt_ending_rows"])
    assert all(row["passed"] for row in result["branch_root_rows"])
    assert result["ungated_tail_rows"][0]["passed"] is True


@pytest.mark.parametrize("mismatch", ["count", "order", "id", "deliverable", "milestone"])
def test_scenario_report_6898_task_contract_mismatches(mismatch: str) -> None:
    """SCENARIO-REPORT-6898-COUNT-ORDER and -IDENTITY fail exact fields."""

    design, roadmap, priors = _valid_inputs()
    tasks = roadmap["tasks"]
    if mismatch == "count":
        tasks.pop()
    elif mismatch == "order":
        tasks[2], tasks[3] = tasks[3], tasks[2]
    elif mismatch == "id":
        tasks[3]["id"] = "exp6901-wrong-id"
    elif mismatch == "deliverable":
        tasks[3]["deliverable"] = "results/experiment_6901_wrong.json"
    else:
        tasks[3]["milestone"] = "2026.09.603"

    result = _evaluate(design, roadmap, priors)

    assert result["passed"] is False
    assert any(not row["passed"] for row in result["document_yaml_parity_rows"])


@pytest.mark.parametrize("mismatch", ["missing_section", "run_command", "final_sentence"])
def test_scenario_report_6898_prompt_endings_are_exact(mismatch: str) -> None:
    """SCENARIO-REPORT-6898-PROMPT-END rejects every prompt mismatch."""

    design, roadmap, priors = _valid_inputs()
    prompt = roadmap["tasks"][0]["prompt"]
    if mismatch == "missing_section":
        prompt = prompt.replace("TASK:\n", "WORK:\n")
    elif mismatch == "run_command":
        prompt = prompt.replace("experiment_6898_", "experiment_9999_", 1)
    else:
        prompt = prompt.replace("Do NOT push.", "Do not push.")
    roadmap["tasks"][0]["prompt"] = prompt

    result = _evaluate(design, roadmap, priors)

    assert result["prompt_ending_rows"][0]["passed"] is False
    assert result["passed"] is False


@pytest.mark.parametrize("mismatch", ["field", "operator", "value", "missing", "later"])
def test_scenario_report_6898_gate_contract_fails_closed(mismatch: str) -> None:
    """SCENARIO-REPORT-6898-GATE-FIELD and -MISSING-UPSTREAM cover gate parts."""

    design, roadmap, priors = _valid_inputs()
    gate = roadmap["tasks"][2]["gated_on"][0]
    if mismatch == "field":
        gate["artifact_field"] = "relation_canary_ready_typo"
    elif mismatch == "operator":
        gate["op"] = "==>"
    elif mismatch == "value":
        gate["value"] = 0
    elif mismatch == "missing":
        gate["upstream"] = "exp6999-missing-upstream"
    else:
        gate["upstream"] = roadmap["tasks"][4]["id"]

    result = _evaluate(design, roadmap, priors)
    row = next(row for row in result["gate_contract_rows"] if row["downstream_number"] == 6900)

    assert row["passed"] is False
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("experiment_id", None),
        ("verdict", None),
        ("addressed_by", ""),
        ("retire_if_same_verdict", False),
    ],
)
def test_scenario_report_6898_prior_failure_fields(field: str, replacement: Any) -> None:
    """SCENARIO-REPORT-6898-PRIOR-FAILURE requires all four exact fields."""

    design, roadmap, priors = _valid_inputs()
    prior = roadmap["tasks"][0]["prior_failures"][0]
    if replacement is None:
        prior.pop(field)
    else:
        prior[field] = replacement

    result = _evaluate(design, roadmap, priors)

    assert field in result["prior_failure_contract_rows"][0]["failed_subfields"]
    assert result["passed"] is False


def test_scenario_report_6898_prior_verdict_uses_primary_artifact() -> None:
    """SCENARIO-REPORT-6898-PRIOR-FAILURE rejects changed verdict text."""

    design, roadmap, priors = _valid_inputs()
    roadmap["tasks"][0]["prior_failures"][0]["verdict"] = "wrong"

    result = _evaluate(design, roadmap, priors)

    assert result["prior_failure_contract_rows"][0]["failed_subfields"] == ["verdict"]


@pytest.mark.parametrize("mismatch", ["advisory_consumer", "root_gated", "tail_gated"])
def test_scenario_report_6898_roots_and_tail_are_ungated(mismatch: str) -> None:
    """SCENARIO-REPORT-6898-UNGATED-TAIL preserves independent execution."""

    design, roadmap, priors = _valid_inputs()
    gate = {
        "upstream": roadmap["tasks"][0]["id"],
        "artifact_field": "v604_manifest_contract_ready_score",
        "op": "==",
        "value": 1,
    }
    if mismatch == "advisory_consumer":
        roadmap["tasks"][2]["gated_on"] = [gate]
    elif mismatch == "root_gated":
        roadmap["tasks"][7]["gated_on"] = [gate]
    else:
        roadmap["tasks"][-1]["gated_on"] = [gate]

    result = _evaluate(design, roadmap, priors)

    assert result["passed"] is False
    assert any(
        not row["passed"] for row in result["branch_root_rows"] + result["ungated_tail_rows"]
    )


def test_scenario_report_6898_flagged_source_and_dependency_taint() -> None:
    """SCENARIO-REPORT-6898-FLAGGED-SOURCE and -DEPENDENCY-TAINT quarantine claims."""

    artifacts, reports = _artifact_inputs()
    result = mod.evaluate_evidence_admission(artifacts, reports)

    by_id = {row["experiment_id"]: row for row in result["artifact_admissibility_rows"]}
    assert by_id[6885]["science_source"] is False
    assert by_id[6887]["admissible"] is False
    assert by_id[6887]["critical_flag_kinds"] == ["DURATION_TOO_SHORT"]
    assert by_id[6888]["admissible"] is False
    assert by_id[6888]["headline_eligible"] is False
    assert by_id[6888]["locally_qualified_arms"] == ["rule:anchored_lexical_v1"]
    assert result["quarantined_artifact_ids"] == [6887, 6888]
    assert result["v603_admissible_science_source_count"] == 0
    assert result["dependency_taint_rows"][0]["source_experiment_id"] == 6887
    assert result["warnings"][0]["kind"] == "SUBSTRATE_NO_LLM_BY_NAME"
    assert result["hard_failures"][0]["kind"] == "DURATION_TOO_SHORT"


def test_req_report_6898_unflagged_traceable_sources_are_counted() -> None:
    """REQ-REPORT-6898 counts only clean sources with independent hashes."""

    artifacts, reports = _artifact_inputs()
    reports[6887] = {"loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}
    reports[6888] = {"loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}

    result = mod.evaluate_evidence_admission(artifacts, reports)

    assert result["quarantined_artifact_ids"] == []
    assert result["dependency_taint_rows"] == []
    assert result["v603_admissible_science_source_count"] == 2


def test_req_report_6898_missing_recheck_fails_admission() -> None:
    """REQ-REPORT-6898 does not admit evidence when the verifier did not load it."""

    artifacts, reports = _artifact_inputs()
    reports[6887] = {"loaded": False, "flag_count": 0, "max_severity": -1, "flags": []}

    result = mod.evaluate_evidence_admission(artifacts, reports)

    row = next(row for row in result["artifact_admissibility_rows"] if row["experiment_id"] == 6887)
    assert row["admissible"] is False
    assert row["admission_failures"] == ["adversarial_recheck_not_loaded"]


def test_req_report_6898_missing_traceability_and_nonmapping_sources_fail() -> None:
    """REQ-REPORT-6898 requires structured source hashes for science admission."""

    artifacts, reports = _artifact_inputs()
    artifacts[6887].pop("source_artifact_hashes")
    artifacts[6888]["source_artifact_hashes"] = []
    reports[6887] = {"loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}
    reports[6888] = {"loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}

    result = mod.evaluate_evidence_admission(artifacts, reports)

    by_id = {row["experiment_id"]: row for row in result["artifact_admissibility_rows"]}
    assert by_id[6887]["admission_failures"] == ["independent_traceability_missing"]
    assert by_id[6888]["dependency_ids"] == []


def test_req_report_6898_active_primary_files_emit_complete_block() -> None:
    """REQ-REPORT-6898 preserves the current stale-design and short-roadmap failure."""

    artifact = mod.build_artifact(ROOT, "20260902")

    assert artifact["task_count"] == 13
    assert artifact["experiment_range"] == [6898, 6910]
    assert len(artifact["document_task_rows"]) == 0
    assert len(artifact["yaml_task_rows"]) == 7
    assert artifact["v604_manifest_contract_ready_score"] == 0
    assert artifact["v603_admissible_science_source_count"] == 0
    assert artifact["quarantined_artifact_ids"] == [6887, 6888]
    assert artifact["status"] == "complete_blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] is not None
    assert mod.validate_artifact(artifact) == []


def test_req_report_6898_missing_preconditions_still_emit_complete_block(tmp_path: Path) -> None:
    """REQ-REPORT-6898 emits all collections and exact source failures."""

    artifact = mod.build_artifact(tmp_path, "20260902", verifier=lambda _path: {})

    assert artifact["preconditions_checked"]["passed"] is False
    assert artifact["gate_check_summary"]["failed_check"] == "preconditions"
    assert artifact["gate_check_summary"]["expected"] == "all required sources readable"
    assert artifact["gate_check_summary"]["observed"]
    for field in set(mod.ROW_COLLECTION_FIELDS) - {"rows"}:
        assert artifact[field] == []
    assert all(row["row_type"] == "contract_check" for row in artifact["rows"])
    assert mod.validate_artifact(artifact) == []


def test_req_report_6898_validation_and_checksum_failures_are_explicit() -> None:
    """REQ-REPORT-6898 validates fields, principles, enums, scores, and checksum."""

    artifact = mod.build_artifact(ROOT, "20260902")
    broken = deepcopy(artifact)
    broken.pop("rows")
    broken["field_principles"].pop("task_count")
    broken["inference_substrate"] = "live_llm_inference"
    broken["verifier_is_oracle"] = True
    broken["verdict_class"] = "unknown"
    broken["honest_verdict"] = "blocked"
    broken["v604_manifest_contract_ready_score"] = 1
    broken["v603_admissible_science_source_count"] = 99
    broken["reproducibility_checksum"] = "sha256:bad"

    errors = mod.validate_artifact(broken)

    assert "missing_required_fields:rows" in errors
    assert "field_principles_missing" in errors
    assert "invalid_inference_substrate" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "invalid_verdict_class" in errors
    assert "honest_verdict_not_terminal" in errors
    assert "readiness_recomputation_mismatch" in errors
    assert "science_source_count_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_req_report_6898_ready_terminal_shape_is_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6898 rejects a ready score with a blocked terminal shape."""

    artifact = mod.build_artifact(ROOT, "20260902")
    for check in artifact["gate_check_summary"]["checks"]:
        check["passed"] = True
    artifact["v604_manifest_contract_ready_score"] = 1
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    monkeypatch.setattr(mod, "BLOCKED_VERDICT", artifact["honest_verdict"])

    assert "ready_terminal_shape_mismatch" in mod.validate_artifact(artifact)


def test_req_report_6898_parsers_and_loaders_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6898 rejects malformed designs, roadmaps, YAML, and JSON."""

    with pytest.raises(ValueError):
        mod.parse_design("not a roadmap")
    with pytest.raises(ValueError):
        mod.parse_design(
            "**Milestone:** `2026.09.604`\n\n### Exp6898: Missing deliverable\n\nBody\n"
        )
    with pytest.raises(ValueError):
        mod.parse_roadmap([])
    with pytest.raises(ValueError):
        mod.parse_roadmap({"tasks": ["bad"]})
    with pytest.raises(ValueError):
        mod.parse_roadmap({})
    with pytest.raises(ValueError):
        mod.parse_roadmap({"tasks": [{"gated_on": "bad"}]})
    with pytest.raises(ValueError):
        mod.parse_roadmap({"tasks": [{"gated_on": ["bad"]}]})
    assert mod.experiment_number(6898) == 6898
    assert mod.experiment_number(True) is None
    assert mod._parse_condition("contains 1") == ("contains 1", None)
    assert mod._prior_artifact_path(tmp_path, "bad") is None
    yaml_path = tmp_path / "bad.yaml"
    yaml_path.write_text("[", encoding="utf-8")
    assert mod.read_yaml_mapping(yaml_path)[1].startswith("invalid_yaml:")
    yaml_path.write_text("- row\n", encoding="utf-8")
    assert mod.read_yaml_mapping(yaml_path) == (None, "yaml_mapping_required")
    json_path = tmp_path / "bad.json"
    json_path.write_text("[", encoding="utf-8")
    assert mod.read_json_mapping(json_path)[1] == "invalid_json:JSONDecodeError"
    json_path.write_text("[]", encoding="utf-8")
    assert mod.read_json_mapping(json_path) == (None, "json_object_required")
    assert mod.read_yaml_mapping(tmp_path / "missing.yaml") == (None, "missing")
    assert mod.read_json_mapping(tmp_path / "missing.json") == (None, "missing")


def test_req_report_6898_operator_extractors_fail_closed() -> None:
    """REQ-REPORT-6898 reads the current schema and gate evaluator."""

    assert {"==", ">=", "exists"}.issubset(mod.schema_gate_operators(SCHEMA_TEXT))
    assert {"==", ">=", "not_contains"}.issubset(mod.conductor_gate_operators(GATE_TEXT))
    with pytest.raises(ValueError):
        mod.schema_gate_operators("GateOp = str")
    with pytest.raises(ValueError):
        mod.conductor_gate_operators("def other(): pass")
    with pytest.raises(ValueError):
        mod.conductor_gate_operators("def _eval_op(op): return op == object()")


def test_req_report_6898_verifier_failures_and_empty_code_are_preconditions(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6898 records verifier exceptions, unloaded rows, and empty code."""

    _write_complete_fixture_root(tmp_path)
    (tmp_path / mod.PRIOR_VALIDATOR_PATH).write_text("", encoding="utf-8")

    def verifier(path: Path) -> dict[str, Any]:
        number = mod.experiment_number(path.name)
        if number == 6885:
            raise RuntimeError("fixture verifier failure")
        if number == 6887:
            return {"loaded": False, "flag_count": 0, "flags": []}
        return {"loaded": True, "flag_count": 0, "max_severity": -1, "flags": []}

    artifact = mod.build_artifact(tmp_path, "20260902", verifier=verifier)

    observed = artifact["preconditions_checked"]["observed"]
    assert {"source": "prior_failure_validator", "error": "empty"} in observed
    assert any(row.get("error") == "verifier_failed:RuntimeError" for row in observed)
    assert any(row.get("error") == "adversarial_recheck_not_loaded" for row in observed)
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert mod.validate_artifact(artifact) == []


def test_req_report_6898_checksum_ignores_runtime_fields() -> None:
    """REQ-REPORT-6898 keeps content identity stable across elapsed time."""

    first = {"duration_s": 1.0, "reproducibility_checksum": "", "value": 2}
    second = {"duration_s": 9.0, "reproducibility_checksum": "bad", "value": 2}
    assert mod.reproducibility_checksum(first) == mod.reproducibility_checksum(second)


def test_req_report_6898_writer_and_cli_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6898 writes only the requested append-only receipt path."""

    output = tmp_path / "receipt.json"
    mod.write_json_atomic(output, {"z": 1, "a": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 2, "z": 1}
    assert mod.main(["--date", "bad", "--output", str(output)]) == 2
    assert mod.main(["--date", "20260902", "--output", str(output)]) == 0
    assert mod.validate_artifact(json.loads(output.read_text(encoding="utf-8"))) == []
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: {"value": 1})
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["invalid"])
    assert mod.main(["--date", "20260902", "--output", str(output)]) == 1


def test_req_report_6898_script_cli_rechecks_with_repo_imports(tmp_path: Path) -> None:
    """REQ-REPORT-6898 makes the exact script entry point run the real verifier."""

    output = tmp_path / "receipt.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/experiment_6898_v604_evidence_admissibility_contract.py",
            "--date",
            "20260902",
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
    assert artifact["quarantined_artifact_ids"] == [6887, 6888]
    rechecks = {row["experiment_id"]: row for row in artifact["adversarial_recheck_rows"]}
    assert rechecks[6887]["max_severity"] == 2
    assert rechecks[6887]["flags"][0]["severity"] == "critical"
    assert artifact["dependency_taint_rows"] == [
        {
            "claim_experiment_id": 6888,
            "source_experiment_id": 6887,
            "all_tainted_sources": [6887],
            "taint_kind": "transitive_source_quarantine",
            "headline_eligible": False,
        }
    ]


def test_req_report_6898_yaml_round_trip_fixture_is_structural() -> None:
    """REQ-REPORT-6898 uses the active roadmap's YAML value types."""

    design, roadmap, priors = _valid_inputs()
    loaded = yaml.safe_load(yaml.safe_dump(roadmap, sort_keys=False))
    assert _evaluate(design, loaded, priors)["passed"] is True
