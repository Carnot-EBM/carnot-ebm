"""Focused tests for the V608 executable contract preflight.

Spec refs: REQ-REPORT-6942 and SCENARIO-REPORT-6942-*.
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

from carnot import experiment_6942_v608_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]


def _gate_text(gates: list[dict[str, Any]]) -> str:
    if not gates:
        return "none"
    gate = gates[0]
    return f"{gate['upstream']}.{gate['artifact_field']} {gate['op']} {gate['value']}"


def _valid_design() -> str:
    rows = [
        "# V608 fixture",
        "",
        "**Milestone:** 2026.09.608",
        "",
        "## Exact Task Contract",
        "",
        "| Order | Task ID | Title | Deliverable | Structured gate |",
        "|---:|---|---|---|---|",
    ]
    for expected in mod.EXPECTED_TASKS:
        rows.append(
            f"| {expected['order']} | {expected['task_id']} | {expected['title']} | "
            f"{expected['deliverable']} | {_gate_text(expected['gates'])} |"
        )
    rows.extend(
        [
            "",
            "## Model Contract",
            "",
            "Every task that invokes an LLM declares `MODEL_SPECS`.",
            "",
            "## Hardware Requirements",
            "",
            "GPU tasks checkpoint each output or event.",
        ]
    )
    return "\n".join(rows)


def _prompt(number: int, deliverable: str, produced_fields: set[str]) -> str:
    fields = set(mod.COMMON_PROMPT_FIELDS) | produced_fields
    model_text = ""
    scope_text = ""
    if number in mod.MODEL_REQUIREMENTS:
        required_models = ", ".join(mod.MODEL_REQUIREMENTS[number])
        model_text = (
            f"2. Define MODEL_SPECS with {required_models}. Legacy small models may run "
            "smoke tests only.\n"
        )
        scope_text = "3. Run exactly 12 rows. After each output, checkpoint that row.\n"
        fields.update({"model_specs", "checkpoint_rows"})
    declared = "; ".join(sorted(fields))
    return (
        "CONTEXT:\nA deterministic contract fixture.\n\n"
        "EXISTING CODE TO READ FIRST:\n- CODEX.md\n\n"
        "TASK:\nAudit one fixture.\n\n"
        "CONCRETE STEPS:\n"
        "0. PRECONDITIONS: require fixture inputs. On failure, write "
        "blocked_fixture_contract with gate_check_summary.\n"
        f"{model_text}{scope_text}"
        "4. REQUIRED ARTIFACT FIELDS: field_principles with one principle per field; "
        f"{declared}.\n\n"
        "Run command: cd {project_root} && .venv/bin/python "
        f"scripts/experiments/{Path(deliverable).stem}.py --date {{date}}\n"
        f"{mod.FINAL_PROHIBITIONS}\n"
    )


def _valid_roadmap() -> dict[str, Any]:
    produced = {number: set() for number in mod.EXPECTED_NUMBERS}
    for expected in mod.EXPECTED_TASKS:
        for gate in expected["gates"]:
            upstream = mod.experiment_number(gate["upstream"])
            produced[upstream].add(gate["artifact_field"])
    tasks = []
    for expected in mod.EXPECTED_TASKS:
        number = expected["number"]
        task = {
            "id": expected["task_id"],
            "title": expected["title"],
            "milestone": mod.MILESTONE,
            "deliverable": expected["deliverable"],
            "requires_gpu": number in mod.MODEL_REQUIREMENTS,
            "estimated_wall_time_min": 60,
            "per_unit_rows": True,
            "gated_on": deepcopy(expected["gates"]),
            "prior_failures": [],
            "prompt": _prompt(number, expected["deliverable"], produced[number]),
        }
        tasks.append(task)
    tasks[0]["prior_failures"] = [
        {
            "experiment_id": "exp6000-old-contract",
            "verdict": "blocked_old_contract",
            "addressed_by": "The fixture checks the new V608 contract.",
            "retire_if_same_verdict": True,
        }
    ]
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": mod.DESIGN_PATH.as_posix(),
        "tasks": tasks,
    }


def _evaluate(roadmap: dict[str, Any]) -> dict[str, Any]:
    return mod.evaluate_contract(_valid_design(), roadmap, retired_ids=set())


def test_req_report_6942_spec_precedes_implementation() -> None:
    """REQ-REPORT-6942 owns each required contract scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6942") :]
    for name in (
        "PREFLIGHT",
        "PARITY",
        "GATES",
        "PROMPTS",
        "MODELS",
        "PRIORS",
        "BOUNDS",
        "MUTATIONS",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-6942-{name}" in section


def test_scenario_report_6942_parity_accepts_exact_fixture() -> None:
    """SCENARIO-REPORT-6942-PARITY accepts the locked twelve-task order."""

    result = _evaluate(_valid_roadmap())

    assert result["passed"] is True
    assert len(result["document_task_rows"]) == 12
    assert len(result["yaml_task_rows"]) == 12
    assert all(row["passed"] for row in result["task_parity_rows"])
    assert all(row["passed"] for row in result["gate_contract_rows"])
    assert all(row["passed"] for row in result["producer_field_rows"])


@pytest.mark.parametrize(
    "mutation",
    [
        "task_count",
        "id_order",
        "title",
        "deliverable",
        "gate_field",
        "model_specs",
        "prior_failure_field",
        "prompt_ending",
    ],
)
def test_scenario_report_6942_required_mutations_fail(mutation: str) -> None:
    """SCENARIO-REPORT-6942-MUTATIONS rejects every required mutation."""

    roadmap = _valid_roadmap()
    if mutation == "task_count":
        roadmap["tasks"].pop()
    elif mutation == "id_order":
        roadmap["tasks"][0], roadmap["tasks"][1] = roadmap["tasks"][1], roadmap["tasks"][0]
    elif mutation == "title":
        roadmap["tasks"][2]["title"] = "Changed title"
    elif mutation == "deliverable":
        roadmap["tasks"][3]["deliverable"] = "results/changed.json"
    elif mutation == "gate_field":
        roadmap["tasks"][2]["gated_on"][0]["artifact_field"] = "missing_ready_score"
    elif mutation == "model_specs":
        prompt = roadmap["tasks"][3]["prompt"]
        roadmap["tasks"][3]["prompt"] = prompt.replace("MODEL_SPECS", "MODEL PLAN")
    elif mutation == "prior_failure_field":
        roadmap["tasks"][0]["prior_failures"][0].pop("addressed_by")
    else:
        roadmap["tasks"][0]["prompt"] = roadmap["tasks"][0]["prompt"].replace(
            mod.FINAL_PROHIBITIONS, "Do NOT push."
        )

    assert _evaluate(roadmap)["passed"] is False


def test_scenario_report_6942_mutation_receipts_cover_all_dimensions() -> None:
    """SCENARIO-REPORT-6942-MUTATIONS records eight effective mutations."""

    rows = mod.build_mutation_rows(_valid_design(), _valid_roadmap(), retired_ids=set())

    assert [row["mutation"] for row in rows] == list(mod.REQUIRED_MUTATIONS)
    assert all(row["failed_as_expected"] for row in rows)


def test_scenario_report_6942_gate_and_model_edges_fail() -> None:
    """SCENARIO-REPORT-6942-GATES and MODELS reject unsafe contracts."""

    missing = _valid_roadmap()
    missing["tasks"][2]["gated_on"][0]["upstream"] = "exp9999-missing"
    assert _evaluate(missing)["gate_contract_rows"][0]["passed"] is False

    retired = _valid_roadmap()
    upstream = retired["tasks"][2]["gated_on"][0]["upstream"]
    result = mod.evaluate_contract(_valid_design(), retired, retired_ids={upstream})
    assert result["gate_contract_rows"][0]["passed"] is False

    tokenizer = _valid_roadmap()
    tokenizer["tasks"][3]["prompt"] += (
        '\nAutoTokenizer.from_pretrained("unsloth/Qwen3.6-35B-A3B-GGUF")\n'
    )
    assert _evaluate(tokenizer)["model_contract_rows"][3]["passed"] is False

    legacy = _valid_roadmap()
    for model_id in mod.MODEL_REQUIREMENTS[6944]:
        legacy["tasks"][3]["prompt"] = legacy["tasks"][3]["prompt"].replace(model_id, "")
    assert _evaluate(legacy)["model_contract_rows"][3]["passed"] is False


@pytest.mark.parametrize("defect", ["checkpoint", "row_ceiling", "wall_time", "blocked_artifact"])
def test_scenario_report_6942_scope_edges_fail(defect: str) -> None:
    """SCENARIO-REPORT-6942-BOUNDS rejects each unbounded task shape."""

    roadmap = _valid_roadmap()
    task = roadmap["tasks"][3]
    if defect == "checkpoint":
        task["prompt"] = task["prompt"].replace("After each output, checkpoint that row.", "")
    elif defect == "row_ceiling":
        task["prompt"] = task["prompt"].replace("exactly 12 rows", "at least 12 rows")
    elif defect == "wall_time":
        task["estimated_wall_time_min"] = 721
    else:
        task["prompt"] = task["prompt"].replace(
            "On failure, write blocked_fixture_contract with gate_check_summary.", ""
        )

    assert _evaluate(roadmap)["bounded_scope_rows"][3]["passed"] is False


def test_scenario_report_6942_parser_edges_are_fail_closed() -> None:
    """SCENARIO-REPORT-6942-PARITY rejects malformed independent inputs."""

    with pytest.raises(ValueError, match="milestone"):
        mod.parse_design("no milestone")
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design("**Milestone:** 2026.09.608")
    with pytest.raises(ValueError, match="roadmap mapping"):
        mod.parse_roadmap([])
    with pytest.raises(ValueError, match="tasks list"):
        mod.parse_roadmap({})
    with pytest.raises(ValueError, match="task at order"):
        mod.parse_roadmap({"milestone": mod.MILESTONE, "tasks": ["bad"]})
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_design(_valid_design().replace("none |", "not-a-gate |", 1))
    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_design_gate("exp6941-v608-source-delta.ready == {}")
    malformed_row = _valid_design().replace(
        "| 1 | exp6941-v608-source-delta |",
        "| 1 | extra | exp6941-v608-source-delta |",
    )
    with pytest.raises(ValueError, match="malformed design task row"):
        mod.parse_design(malformed_row)
    empty_table = (
        "**Milestone:** 2026.09.608\n\n"
        "## Exact Task Contract\n\n"
        "| Order | Task ID | Title | Deliverable | Structured gate |\n"
        "\n## Next\n"
    )
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design(empty_table)
    with pytest.raises(ValueError, match="task lists"):
        mod.parse_roadmap({"tasks": [{"gated_on": "bad"}]})

    no_fields = _valid_roadmap()
    no_fields["tasks"][0]["prompt"] = ""
    assert mod.parse_roadmap(no_fields)["tasks"][0]["required_fields_block"] == ""
    assert mod.retired_experiment_ids([]) == set()
    assert mod.retired_experiment_ids({"retired": "bad"}) == set()


def test_scenario_report_6942_malformed_prior_and_parse_failure_are_visible(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6942-PRIORS retains malformed rows and parse errors."""

    roadmap = _valid_roadmap()
    roadmap["tasks"][0]["prior_failures"] = ["bad"]
    assert _evaluate(roadmap)["prior_failure_rows"][0]["passed"] is False

    for path in mod.PRECONDITION_PATHS.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("invalid", encoding="utf-8")
    artifact = mod.build_artifact(tmp_path, "20260903")
    assert artifact["gate_check_summary"]["failed_check"] == "contract_parse"
    assert "ValueError" in artifact["gate_check_summary"]["observed"]
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6942_preflight_missing_inputs_writes_complete_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6942-PREFLIGHT retains the complete blocked schema."""

    artifact = mod.build_artifact(tmp_path, "20260903")

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["v608_execution_contract_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "preconditions"
    assert mod.validate_artifact(artifact) == []


def test_req_report_6942_active_contract_emits_honest_blocked_receipt() -> None:
    """REQ-REPORT-6942 preserves current lint and scope failures as blocked."""

    roadmap_before = (ROOT / mod.ACTIVE_ROADMAP_PATH).read_bytes()
    artifact = mod.build_artifact(ROOT, "20260903")

    assert mod.validate_artifact(artifact) == []
    assert artifact["v608_execution_contract_ready_score"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert any(not row["passed"] for row in artifact["lint_command_rows"])
    assert all(row["failed_as_expected"] for row in artifact["mutation_rows"])
    assert (ROOT / mod.ACTIVE_ROADMAP_PATH).read_bytes() == roadmap_before


def test_scenario_report_6942_artifact_validator_recomputes_binary_verdict() -> None:
    """SCENARIO-REPORT-6942-ARTIFACT rejects forged terminal fields."""

    artifact = mod.build_artifact(ROOT, "20260903")
    mutations = {
        "field_principles": {},
        "inference_substrate": "llm",
        "v608_execution_contract_ready_score": 1,
        "verdict_class": "partial",
        "honest_verdict": "complete_ready",
        "verifier_is_oracle": False,
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert mod.validate_artifact(changed)

    missing = deepcopy(artifact)
    missing.pop("rows")
    missing["reproducibility_checksum"] = mod.reproducibility_checksum(missing)
    assert "missing_required_fields:rows" in mod.validate_artifact(missing)

    bad_hash = deepcopy(artifact)
    bad_hash["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(bad_hash)

    invalid_class = deepcopy(artifact)
    invalid_class["verdict_class"] = "unknown"
    invalid_class["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_class)
    assert "verdict_class_invalid" in mod.validate_artifact(invalid_class)

    wrong_mutations = deepcopy(artifact)
    wrong_mutations["mutation_rows"] = [{"mutation": "other"}]
    wrong_mutations["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_mutations)
    assert "mutation_rows_mismatch" in mod.validate_artifact(wrong_mutations)

    ready = deepcopy(artifact)
    for check in ready["gate_check_summary"]["checks"]:
        check["passed"] = True
    ready["v608_execution_contract_ready_score"] = 1
    ready["reproducibility_checksum"] = mod.reproducibility_checksum(ready)
    assert "ready_terminal_shape_mismatch" in mod.validate_artifact(ready)


def test_req_report_6942_command_receipts_and_cli_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6942 records command errors and supports the required wrapper."""

    def raise_timeout(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["test"], 1, output="partial", stderr="late")

    monkeypatch.setattr(mod.subprocess, "run", raise_timeout)
    timed = mod._run_command(ROOT, "timeout", [sys.executable, "-V"])
    assert timed["exit_code"] == 124
    assert timed["passed"] is False

    monkeypatch.undo()
    assert mod.main(["--date", "bad"]) == 2

    output = tmp_path / "artifact.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/experiment_6942_v608_contract_preflight.py",
            "--date",
            "20260903",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["honest_verdict"] == mod.BLOCKED_VERDICT
    assert mod.validate_artifact(written) == []


def test_req_report_6942_direct_main_covers_terminal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6942 writes valid output and refuses an invalid artifact."""

    minimal = {"honest_verdict": mod.BLOCKED_VERDICT}
    monkeypatch.setattr(mod, "build_artifact", lambda _root, _date: minimal)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    output = tmp_path / "direct.json"
    assert mod.main(["--date", "20260903", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == minimal
    assert mod._date_argument("20260903") == "20260903"

    written: list[Path] = []
    monkeypatch.setattr(mod, "_write_json_atomic", lambda path, _value: written.append(path))
    assert mod.main(["--date", "20260903", "--output", "relative.json"]) == 0
    assert written == [mod.REPO_ROOT / "relative.json"]

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["invalid"])
    assert mod.main(["--date", "20260903", "--output", str(output)]) == 1


def test_req_report_6942_yaml_round_trip_preserves_gate_types(tmp_path: Path) -> None:
    """REQ-REPORT-6942 reads executable YAML values after serialization."""

    path = tmp_path / "roadmap.yaml"
    path.write_text(yaml.safe_dump(_valid_roadmap(), sort_keys=False), encoding="utf-8")
    parsed = mod.parse_roadmap(yaml.safe_load(path.read_text(encoding="utf-8")))

    assert parsed["milestone"] == mod.MILESTONE
    assert parsed["tasks"][2]["gates"][0]["value"] == 1
