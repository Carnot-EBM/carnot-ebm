"""Focused contract and mutation tests for REQ-REPORT-6965."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from carnot import experiment_6965_v610_contract_advisory as mod


ROOT = Path(__file__).resolve().parents[2]


def _gate_text(gates: list[dict[str, Any]]) -> str:
    """Render the design-table spelling independently from the YAML fixture."""

    if not gates:
        return "none"
    return "; ".join(
        f"exp{mod.experiment_number(gate['upstream'])} "
        f"`{gate['artifact_field']} {gate['op']} {gate['value']}`"
        for gate in gates
    )


def _valid_design() -> str:
    """Build a complete Markdown source without calling the production parser."""

    lines = [
        "# V610 fixture",
        "",
        f"**Milestone:** {mod.MILESTONE}  ",
        "",
        "## Exact Task Contract",
        "",
        "| Order | ID | Title | Deliverable | Structured gate |",
        "|---:|---|---|---|---|",
    ]
    for task in mod.EXPECTED_TASKS:
        lines.append(
            f"| {task['order']} | {task['task_id']} | {task['title']} | "
            f"`{task['deliverable']}` | {_gate_text(task['gates'])} |"
        )
    lines.extend(["", "## Runtime", "", "The contract is bounded."])
    return "\n".join(lines)


def _prompt(task: dict[str, Any], produced_fields: set[str]) -> str:
    """Create one fully bounded prompt with task-owned evidence fields."""

    number = int(task["number"])
    fields = set(mod.COMMON_PROMPT_FIELDS) | produced_fields
    model_text = ""
    if number in mod.LLM_TASK_NUMBERS:
        fields.add("MODEL_SPECS")
        model_text = (
            f"2. Define MODEL_SPECS with {mod.MANDATED_GGUF_MODELS[0]}. "
            "Legacy models are smoke-only and cannot enter headline rows.\n"
        )
    arc_text = ""
    if number in mod.ARC_TASK_NUMBERS:
        fields.update(mod.ARC_FALSE_FIELDS)
        arc_text = (
            "3. Emit solve_claimed=false, level_claimed=false, "
            "registry_updated=false, and submitted_to_leaderboard=false. "
            "Do not solve or re-solve a game, calibrate with outer-loop ground truth, "
            "update the solve registry, or submit to the leaderboard.\n"
        )
    declared = "; ".join(
        f"{field} (false)" if field in mod.ARC_FALSE_FIELDS else field for field in sorted(fields)
    )
    return (
        "CONTEXT:\nA deterministic V610 contract fixture.\n\n"
        "EXISTING CODE TO READ FIRST:\n- CODEX.md\n\n"
        "TASK:\nAudit one bounded fixture.\n\n"
        "CONCRETE STEPS:\n"
        "0. PRECONDITIONS: require immutable fixture inputs. On failure, write "
        "blocked_fixture_contract with gate_check_summary. Missing science inputs are "
        "blocked, not partial.\n"
        "1. Run exactly 14 rows, checkpoint after each row, and preserve task-owned "
        "execution evidence.\n"
        f"{model_text}{arc_text}"
        "4. Require verdict_class (positive | circular_positive | null | blocked | "
        "disqualified | partial) and honest_verdict with a terminal prefix consistent "
        "with verdict_class.\n"
        "5. REQUIRED ARTIFACT FIELDS: field_principles with one scientific principle "
        f"per required field; {declared}.\n\n"
        "Run command: cd {project_root} && .venv/bin/python scripts/experiments/"
        f"{Path(task['deliverable']).stem}.py --date {{date}}\n"
        f"{mod.FINAL_PROHIBITIONS}\n"
    )


def _valid_roadmap() -> dict[str, Any]:
    """Build a valid 14-task executable source for focused mutation tests."""

    produced = {number: set() for number in mod.EXPECTED_NUMBERS}
    for task in mod.EXPECTED_TASKS:
        for gate in task["gates"]:
            produced[mod.experiment_number(gate["upstream"])].add(gate["artifact_field"])
    tasks = []
    for expected in mod.EXPECTED_TASKS:
        number = int(expected["number"])
        tasks.append(
            {
                "id": expected["task_id"],
                "title": expected["title"],
                "track": "arc" if number in mod.ARC_TASK_NUMBERS else "verification",
                "agent_type": "codex",
                "model": "gpt-5.6-sol",
                "milestone": mod.MILESTONE,
                "deliverable": expected["deliverable"],
                "requires_gpu": number in mod.LLM_TASK_NUMBERS,
                "max_turns": 50,
                "estimated_wall_time_min": 120,
                "per_unit_rows": True,
                "gated_on": deepcopy(expected["gates"]),
                "prior_failures": [
                    {
                        "experiment_id": f"exp{number - 14}-prior",
                        "verdict": "blocked_prior_scope",
                        "addressed_by": "The V610 fixture uses a materially different bounded mechanism.",
                        "retire_if_same_verdict": True,
                    }
                ],
                "prompt": _prompt(expected, produced[number]),
            }
        )
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": mod.DESIGN_PATH.as_posix(),
        "tasks": tasks,
    }


def _evaluate(roadmap: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an in-memory fixture with no retired experiment IDs."""

    return mod.evaluate_contract(_valid_design(), roadmap, retired_ids=set())


def test_req_report_6965_spec_precedes_implementation() -> None:
    """REQ-REPORT-6965 owns each required V610 audit behavior."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6965") :]
    for name in (
        "PREFLIGHT",
        "PARITY",
        "GATES",
        "PROMPTS",
        "MODELS",
        "PRIORS",
        "ARC",
        "BOUNDS",
        "MUTATIONS",
        "ADVISORY",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-6965-{name}" in section


def test_scenario_report_6965_parity_accepts_complete_multigate_fixture() -> None:
    """SCENARIO-REPORT-6965-PARITY accepts all 14 locked task contracts."""

    result = _evaluate(_valid_roadmap())

    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 14
    assert len(result["gate_rows"]) == 16
    assert all(row["passed"] for row in result["task_contract_rows"])
    assert all(row["passed"] for row in result["gate_rows"])
    assert all(row["passed"] for row in result["producer_field_rows"])


@pytest.mark.parametrize("mutation", mod.REQUIRED_MUTATIONS)
def test_scenario_report_6965_required_mutations_fail(mutation: str) -> None:
    """SCENARIO-REPORT-6965-MUTATIONS detects every requested mutation."""

    roadmap = _valid_roadmap()
    mod.apply_mutation(roadmap, mutation)

    assert _evaluate(roadmap)["passed"] is False


def test_scenario_report_6965_mutation_receipts_are_effective() -> None:
    """SCENARIO-REPORT-6965-MUTATIONS records changed and rejected inputs."""

    rows = mod.build_mutation_rows(_valid_design(), _valid_roadmap(), retired_ids=set())

    assert [row["mutation"] for row in rows] == list(mod.REQUIRED_MUTATIONS)
    assert all(row["input_changed"] for row in rows)
    assert all(row["failed_as_expected"] for row in rows)


def test_scenario_report_6965_gate_and_advisory_edges_fail() -> None:
    """SCENARIO-REPORT-6965-GATES rejects missing, retired, and advisory producers."""

    missing = _valid_roadmap()
    missing["tasks"][4]["gated_on"][0]["upstream"] = "exp9999-missing"
    assert any(not row["passed"] for row in _evaluate(missing)["gate_rows"])

    retired = _valid_roadmap()
    upstream = retired["tasks"][4]["gated_on"][0]["upstream"]
    evaluated = mod.evaluate_contract(_valid_design(), retired, retired_ids={upstream})
    assert any(row["upstream_retired"] for row in evaluated["gate_rows"])
    assert evaluated["passed"] is False

    advisory = _valid_roadmap()
    advisory["tasks"][4]["gated_on"][0] = {
        "upstream": "exp6965-v610-contract-advisory",
        "artifact_field": "contract_conforms_score",
        "op": "==",
        "value": 1,
    }
    assert any(row["advisory_dependency"] for row in _evaluate(advisory)["gate_rows"])


def test_scenario_report_6965_model_and_arc_edges_fail() -> None:
    """SCENARIO-REPORT-6965-MODELS and ARC reject unsafe model and solve plans."""

    tokenizer = _valid_roadmap()
    tokenizer["tasks"][1]["prompt"] += (
        '\nAutoTokenizer.from_pretrained("unsloth/Qwen3.6-35B-A3B-GGUF")\n'
    )
    assert _evaluate(tokenizer)["model_contract_rows"][1]["passed"] is False

    legacy = _valid_roadmap()
    legacy["tasks"][1]["prompt"] = legacy["tasks"][1]["prompt"].replace(
        mod.MANDATED_GGUF_MODELS[0], "Qwen3.5-0.8B"
    )
    model_row = _evaluate(legacy)["model_contract_rows"][1]
    assert model_row["legacy_only_headline"] is True
    assert model_row["passed"] is False

    solve = _valid_roadmap()
    solve["tasks"][3]["prompt"] = (
        solve["tasks"][3]["prompt"]
        .replace("solve_claimed=false", "solve_claimed=true")
        .replace("solve_claimed (false)", "solve_claimed (true)")
    )
    assert _evaluate(solve)["arc_solve_rule_rows"][0]["passed"] is False


@pytest.mark.parametrize(
    "defect",
    ["sections", "command", "ending", "wall", "turns", "rows", "checkpoint", "blocked"],
)
def test_scenario_report_6965_prompt_and_bound_edges_fail(defect: str) -> None:
    """SCENARIO-REPORT-6965-PROMPTS and BOUNDS reject unbounded prompt shapes."""

    roadmap = _valid_roadmap()
    task = roadmap["tasks"][1]
    if defect == "sections":
        task["prompt"] = task["prompt"].replace("TASK:", "WORK:")
    elif defect == "command":
        task["prompt"] = task["prompt"].replace("Run command:", "Run command:\nRun command:")
    elif defect == "ending":
        task["prompt"] += "trailing content\n"
    elif defect == "wall":
        task["estimated_wall_time_min"] = 721
    elif defect == "turns":
        task["max_turns"] = 0
    elif defect == "rows":
        task["per_unit_rows"] = False
    elif defect == "checkpoint":
        task["prompt"] = task["prompt"].replace("checkpoint after each row", "record each row")
    else:
        task["prompt"] = task["prompt"].replace(
            "On failure, write blocked_fixture_contract with gate_check_summary.", ""
        )

    assert _evaluate(roadmap)["passed"] is False


def test_scenario_report_6965_parser_and_prior_edges_fail_closed() -> None:
    """SCENARIO-REPORT-6965-PRIORS rejects malformed sources and subfields."""

    assert len(mod.parse_design(_valid_design())["tasks"]) == 14
    assert len(mod.parse_roadmap(_valid_roadmap())["tasks"]) == 14
    with pytest.raises(ValueError, match="milestone"):
        mod.parse_design("no milestone")
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design(f"**Milestone:** {mod.MILESTONE}")
    with pytest.raises(ValueError, match="roadmap mapping"):
        mod.parse_roadmap([])
    with pytest.raises(ValueError, match="tasks list"):
        mod.parse_roadmap({})
    with pytest.raises(ValueError, match="task at order"):
        mod.parse_roadmap({"tasks": ["bad"]})
    with pytest.raises(ValueError, match="task lists"):
        mod.parse_roadmap({"tasks": [{"gated_on": "bad"}]})
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_design(_valid_design().replace("none |", "not-a-gate |", 1))
    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_design_gates("exp6966 `ready == {}`")
    malformed_row = _valid_design().replace(
        "| 1 | exp6965-v610-contract-advisory |",
        "| 1 | extra | exp6965-v610-contract-advisory |",
    )
    with pytest.raises(ValueError, match="malformed design task row"):
        mod.parse_design(malformed_row)
    empty_table = (
        f"**Milestone:** {mod.MILESTONE}\n\n## Exact Task Contract\n\n"
        "| Order | ID | Title | Deliverable | Structured gate |\n\n## Next\n"
    )
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design(empty_table)

    no_fields = _valid_roadmap()
    no_fields["tasks"][0]["prompt"] = ""
    assert mod.parse_roadmap(no_fields)["tasks"][0]["required_fields_block"] == ""

    malformed = _valid_roadmap()
    malformed["tasks"][0]["prior_failures"] = ["bad"]
    assert _evaluate(malformed)["prior_failure_rows"][0]["passed"] is False
    wrong_bool = _valid_roadmap()
    wrong_bool["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"] = "true"
    assert _evaluate(wrong_bool)["prior_failure_rows"][0]["passed"] is False


def test_req_report_6965_manifest_parser_honors_unretirement() -> None:
    """REQ-REPORT-6965 reads numeric, string, nested, and un-retired manifest IDs."""

    manifest = {
        "retired": [{"experiment_id": 1}],
        "retired_experiments": [{"experiment_id": "exp2-x"}],
        "retired_extras": [
            {"experiment_ids": ["exp3-x", 4], "un_retired_experiment_ids": ["exp3-x"]}
        ],
    }

    assert mod.retired_experiment_ids(manifest) == {"exp1", "exp2", "exp4"}
    assert mod.retired_experiment_ids([]) == set()
    assert mod.retired_experiment_ids({"retired": "bad", "retired_experiments": ["bad"]}) == set()


def test_req_report_6965_unknown_and_parse_error_mutations_are_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6965 keeps programming errors and parser failures explicit."""

    with pytest.raises(ValueError, match="unknown mutation"):
        mod.apply_mutation(_valid_roadmap(), "unknown")

    original = mod.evaluate_contract
    calls = 0

    def fail_after_baseline(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("mutated parse failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "evaluate_contract", fail_after_baseline)
    rows = mod.build_mutation_rows(_valid_design(), _valid_roadmap(), set())
    assert all(row["error"] == "ValueError: mutated parse failure" for row in rows)
    assert all(row["failed_as_expected"] for row in rows)


def _write_precondition_tree(
    root: Path, design: str, roadmap: dict[str, Any], manifest: str
) -> None:
    """Materialize only the precondition inputs beneath an isolated root."""

    for path in mod.PRECONDITION_PATHS.values():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder\n", encoding="utf-8")
    (root / mod.DESIGN_PATH).write_text(design, encoding="utf-8")
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap, sort_keys=False), encoding="utf-8"
    )
    (root / mod.EXCLUSION_PATH).write_text(manifest, encoding="utf-8")


def test_scenario_report_6965_malformed_manifest_and_contract_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6965-PREFLIGHT blocks malformed complete-looking inputs."""

    _write_precondition_tree(tmp_path, _valid_design(), _valid_roadmap(), "{}\n")
    shape = mod.build_artifact(tmp_path, "20260904")
    assert shape["gate_check_summary"]["failed_check"] == "preconditions"

    (tmp_path / mod.EXCLUSION_PATH).write_text("[\n", encoding="utf-8")
    syntax = mod.build_artifact(tmp_path, "20260904")
    manifest_row = next(
        row
        for row in syntax["preconditions_checked"]
        if row["resource"] == "full_exclusion_manifest"
    )
    assert "ParserError" in manifest_row["parse_error"]

    _write_precondition_tree(
        tmp_path,
        "not a design",
        _valid_roadmap(),
        "retired: []\nretired_experiments: []\nretired_extras: []\n",
    )
    contract = mod.build_artifact(tmp_path, "20260904")
    assert contract["gate_check_summary"]["failed_check"] == "contract_parse"
    assert "ValueError" in contract["gate_check_summary"]["observed"]


def test_scenario_report_6965_preflight_and_active_defects_are_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6965-PREFLIGHT blocks missing inputs but completes active defects."""

    missing = mod.build_artifact(tmp_path, "20260904")
    assert mod.REQUIRED_ARTIFACT_FIELDS <= missing.keys()
    assert missing["v610_contract_audit_complete_score"] == 0
    assert missing["contract_conforms_score"] == 0
    assert missing["verdict_class"] == "blocked"
    assert missing["honest_verdict"] == mod.BLOCKED_VERDICT
    assert missing["gate_check_summary"]["failed_check"] == "preconditions"
    assert mod.validate_artifact(missing) == []

    active = mod.build_artifact(ROOT, "20260904")
    assert mod.validate_artifact(active) == []
    assert active["v610_contract_audit_complete_score"] == 1
    assert active["contract_conforms_score"] == 0
    assert active["verdict_class"] == "null"
    assert active["honest_verdict"] == mod.DEFECT_VERDICT
    assert len(active["task_contract_rows"]) == 14
    assert sum(row["yaml_present"] for row in active["task_contract_rows"]) == 7
    assert [row["name"] for row in active["command_receipt_rows"]] == list(mod.COMMAND_NAMES)
    assert all(row["terminal"] for row in active["command_receipt_rows"])


def test_scenario_report_6965_artifact_validator_recomputes_terminal_state() -> None:
    """SCENARIO-REPORT-6965-ARTIFACT rejects forged required fields and scores."""

    artifact = mod.build_artifact(ROOT, "20260904")
    for field, value in {
        "field_principles": {},
        "inference_substrate": "llm",
        "v610_contract_audit_complete_score": 0,
        "contract_conforms_score": 1,
        "verdict_class": "partial",
        "honest_verdict": "complete_ready",
        "verifier_is_oracle": False,
    }.items():
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

    invalid_summary = deepcopy(artifact)
    invalid_summary["gate_check_summary"] = []
    invalid_summary["reproducibility_checksum"] = mod.reproducibility_checksum(invalid_summary)
    assert "gate_check_summary_invalid" in mod.validate_artifact(invalid_summary)

    wrong_summary = deepcopy(artifact)
    wrong_summary["gate_check_summary"]["audit_complete"] = False
    wrong_summary["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_summary)
    assert "gate_check_summary_score_mismatch" in mod.validate_artifact(wrong_summary)

    blocked_shape = deepcopy(artifact)
    for row in blocked_shape["command_receipt_rows"]:
        row["terminal"] = False
    blocked_shape["status"] = "complete"
    blocked_shape["verdict_class"] = "blocked"
    blocked_shape["honest_verdict"] = mod.BLOCKED_VERDICT
    blocked_shape["v610_contract_audit_complete_score"] = 0
    blocked_shape["contract_conforms_score"] = 0
    blocked_shape["gate_check_summary"]["audit_complete"] = False
    blocked_shape["gate_check_summary"]["passed"] = False
    blocked_shape["reproducibility_checksum"] = mod.reproducibility_checksum(blocked_shape)
    assert "blocked_terminal_shape_mismatch" in mod.validate_artifact(blocked_shape)


def test_scenario_report_6965_conforming_artifact_and_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6965-ADVISORY keeps a clean contract circular and advisory."""

    _write_precondition_tree(
        tmp_path,
        _valid_design(),
        _valid_roadmap(),
        "retired: []\nretired_experiments: []\nretired_extras: []\n",
    )

    def passing_commands(_root: Path, _evaluation: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "command": name,
                "exit_code": 0,
                "stdout": "clean",
                "stderr": "",
                "outcome": "pass",
                "terminal": True,
                "passed": True,
            }
            for name in mod.COMMAND_NAMES
        ]

    monkeypatch.setattr(mod, "run_lint_commands", passing_commands)
    artifact = mod.build_artifact(tmp_path, "20260904")
    assert artifact["contract_conforms_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert mod.validate_artifact(artifact) == []

    forged = deepcopy(artifact)
    forged["honest_verdict"] = "complete_wrong"
    forged["reproducibility_checksum"] = mod.reproducibility_checksum(forged)
    assert "conforming_terminal_shape_mismatch" in mod.validate_artifact(forged)


def test_req_report_6965_command_receipts_distinguish_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6965 separates terminal contract findings from tool failures."""

    completed = subprocess.CompletedProcess(["lint"], 1, "defect", "")
    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: completed)
    defect = mod._run_command(ROOT, "lint", [sys.executable, "lint.py"])
    assert defect["outcome"] == "contract_defect"
    assert defect["terminal"] is True

    def raise_timeout(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["test"], 1, output="partial", stderr="late")

    monkeypatch.setattr(mod.subprocess, "run", raise_timeout)
    timed = mod._run_command(ROOT, "timeout", [sys.executable, "-V"])
    assert timed["exit_code"] == 124
    assert timed["outcome"] == "tool_failure"
    assert timed["terminal"] is False

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    failed = mod._run_command(ROOT, "missing", ["missing"])
    assert failed["outcome"] == "tool_failure"
    assert "OSError" in failed["stderr"]


def test_req_report_6965_cli_writes_valid_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-6965 exposes the required wrapper and rejects ambiguous dates."""

    assert mod.main(["--date", "bad"]) == 2
    output = tmp_path / "artifact.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/experiment_6965_v610_contract_advisory.py",
            "--date",
            "20260904",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    written = json.loads(output.read_text(encoding="utf-8"))
    assert mod.validate_artifact(written) == []
    assert mod._date_argument("20260904") == "20260904"


def test_req_report_6965_direct_main_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-6965 validates before writing relative and absolute outputs."""

    artifact = {"honest_verdict": mod.BLOCKED_VERDICT}
    monkeypatch.setattr(mod, "build_artifact", lambda _root, _date: artifact)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    written: list[Path] = []
    monkeypatch.setattr(mod, "_write_json_atomic", lambda path, _value: written.append(path))

    absolute = tmp_path / "absolute.json"
    assert mod.main(["--date", "20260904", "--output", str(absolute)]) == 0
    assert mod.main(["--date", "20260904", "--output", "relative.json"]) == 0
    assert written == [absolute, mod.REPO_ROOT / "relative.json"]

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["invalid"])
    assert mod.main(["--date", "20260904", "--output", str(absolute)]) == 1


def test_req_report_6965_atomic_writer(tmp_path: Path) -> None:
    """REQ-REPORT-6965 publishes only a complete JSON object."""

    output = tmp_path / "nested" / "artifact.json"
    mod._write_json_atomic(output, {"terminal": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"terminal": True}
    assert not output.with_suffix(".json.tmp").exists()


def test_req_report_6965_yaml_round_trip_preserves_gate_scalars(tmp_path: Path) -> None:
    """REQ-REPORT-6965 keeps numeric gate values after YAML serialization."""

    path = tmp_path / "roadmap.yaml"
    path.write_text(yaml.safe_dump(_valid_roadmap(), sort_keys=False), encoding="utf-8")
    parsed = mod.parse_roadmap(yaml.safe_load(path.read_text(encoding="utf-8")))

    assert parsed["milestone"] == mod.MILESTONE
    assert parsed["tasks"][9]["gates"][1]["value"] == 6
