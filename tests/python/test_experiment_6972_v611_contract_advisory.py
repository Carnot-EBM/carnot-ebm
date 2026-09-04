"""Focused contract and mutation tests for REQ-REPORT-6972."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

from carnot import experiment_6972_v611_contract_advisory as mod


ROOT = Path(__file__).resolve().parents[2]


def _design_text() -> str:
    """Read the design contract without using the production parser."""

    return (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")


def _roadmap() -> dict[str, Any]:
    """Read the activated YAML without using the production parser."""

    value = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _evaluate(roadmap: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one in-memory contract with no retired IDs."""

    return mod.evaluate_contract(_design_text(), roadmap or _roadmap(), retired_ids=set())


def test_req_report_6972_spec_precedes_implementation() -> None:
    """REQ-REPORT-6972 owns every focused audit scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6972") :]
    for name in (
        "PREFLIGHT",
        "PARITY",
        "GATES",
        "PROMPTS",
        "MODELS",
        "PRIORS",
        "ARC",
        "MUTATIONS",
        "ADVISORY",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-6972-{name}" in section


def test_scenario_report_6972_parity_accepts_the_active_contract() -> None:
    """SCENARIO-REPORT-6972-PARITY accepts all 12 activated task rows."""

    result = _evaluate()

    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_rows"]) == 13
    assert all(row["passed"] for row in result["task_contract_rows"])
    assert all(row["passed"] for row in result["producer_field_rows"])


@pytest.mark.parametrize("mutation", mod.REQUIRED_MUTATIONS)
def test_scenario_report_6972_required_mutations_fail(mutation: str) -> None:
    """SCENARIO-REPORT-6972-MUTATIONS detects each required mutation."""

    roadmap = _roadmap()
    mod.apply_mutation(roadmap, mutation)

    assert _evaluate(roadmap)["passed"] is False


def test_scenario_report_6972_mutation_receipts_are_effective() -> None:
    """SCENARIO-REPORT-6972-MUTATIONS records changed, failed inputs."""

    rows = mod.build_mutation_rows(_design_text(), _roadmap(), retired_ids=set())

    assert [row["mutation"] for row in rows] == list(mod.REQUIRED_MUTATIONS)
    assert all(row["input_changed"] for row in rows)
    assert all(row["failed_as_expected"] for row in rows)


def test_scenario_report_6972_gate_edges_fail() -> None:
    """SCENARIO-REPORT-6972-GATES rejects missing, later, retired, and advisory producers."""

    missing = _roadmap()
    missing["tasks"][3]["gated_on"][0]["upstream"] = "exp9999-missing"
    assert any(not row["passed"] for row in _evaluate(missing)["gate_rows"])

    later = _roadmap()
    later["tasks"][3]["gated_on"][0]["upstream"] = later["tasks"][4]["id"]
    assert any(not row["upstream_precedes_consumer"] for row in _evaluate(later)["gate_rows"])

    retired = _roadmap()
    upstream = retired["tasks"][3]["gated_on"][0]["upstream"]
    checked = mod.evaluate_contract(_design_text(), retired, retired_ids={upstream})
    assert any(row["upstream_retired"] for row in checked["gate_rows"])

    advisory = _roadmap()
    advisory["tasks"][3]["gated_on"][0] = {
        "upstream": "exp6972-v611-contract-advisory",
        "artifact_field": "contract_conforms_score",
        "op": "==",
        "value": 1,
    }
    assert any(row["advisory_dependency"] for row in _evaluate(advisory)["gate_rows"])


def test_scenario_report_6972_model_edges_fail() -> None:
    """SCENARIO-REPORT-6972-MODELS rejects legacy-only and GGUF tokenizer plans."""

    tokenizer = _roadmap()
    tokenizer["tasks"][1]["prompt"] += (
        '\nAutoTokenizer.from_pretrained("unsloth/Qwen3.6-35B-A3B-GGUF")\n'
    )
    assert _evaluate(tokenizer)["model_contract_rows"][1]["passed"] is False

    legacy = _roadmap()
    for model_id in mod.MANDATED_GGUF_MODELS:
        legacy["tasks"][1]["prompt"] = legacy["tasks"][1]["prompt"].replace(
            model_id, "Qwen3.5-0.8B"
        )
    row = _evaluate(legacy)["model_contract_rows"][1]
    assert row["legacy_only_headline"] is True
    assert row["passed"] is False

    incoherent = _roadmap()
    incoherent["tasks"][4]["model"] = "opus"
    assert _evaluate(incoherent)["model_contract_rows"][4]["passed"] is False


def test_scenario_report_6972_arc_edges_fail() -> None:
    """SCENARIO-REPORT-6972-ARC rejects solve claims and omitted prohibitions."""

    solve = _roadmap()
    arc = solve["tasks"][9]
    arc["prompt"] = arc["prompt"].replace("solve_claimed (false)", "solve_claimed (true)")
    assert _evaluate(solve)["arc_solve_rule_rows"][0]["passed"] is False

    registry = _roadmap()
    registry["tasks"][9]["prompt"] = registry["tasks"][9]["prompt"].replace(
        "Do not modify", "Modify"
    )
    assert _evaluate(registry)["arc_solve_rule_rows"][0]["passed"] is False


@pytest.mark.parametrize("defect", ["section", "command", "ending", "blocked", "partial"])
def test_scenario_report_6972_prompt_edges_fail(defect: str) -> None:
    """SCENARIO-REPORT-6972-PROMPTS rejects malformed prompt contracts."""

    roadmap = _roadmap()
    prompt = roadmap["tasks"][0]["prompt"]
    if defect == "section":
        prompt = prompt.replace("TASK:", "WORK:")
    elif defect == "command":
        prompt = prompt.replace("Run command:", "Run command:\nRun command:")
    elif defect == "ending":
        prompt += "\ntrailing content"
    elif defect == "blocked":
        prompt = prompt.replace("blocked_v611_contract_advisory", "v611_contract_advisory")
    else:
        prompt = prompt.replace("blocked, not partial", "partial")
    roadmap["tasks"][0]["prompt"] = prompt

    assert _evaluate(roadmap)["prompt_ending_rows"][0]["passed"] is False


def test_scenario_report_6972_prior_fields_fail_closed() -> None:
    """SCENARIO-REPORT-6972-PRIORS rejects malformed and mistyped prior rows."""

    malformed = _roadmap()
    malformed["tasks"][0]["prior_failures"] = ["bad"]
    assert _evaluate(malformed)["prior_failure_rows"][0]["passed"] is False

    wrong_bool = _roadmap()
    wrong_bool["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"] = "true"
    assert _evaluate(wrong_bool)["prior_failure_rows"][0]["passed"] is False

    empty = _roadmap()
    empty["tasks"][0]["prior_failures"][0]["addressed_by"] = ""
    assert _evaluate(empty)["prior_failure_rows"][0]["passed"] is False


def test_req_report_6972_parsers_fail_closed() -> None:
    """REQ-REPORT-6972 rejects malformed contract and manifest shapes."""

    assert len(mod.parse_design(_design_text())["tasks"]) == 12
    assert len(mod.parse_roadmap(_roadmap())["tasks"]) == 12
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
    with pytest.raises(ValueError, match="structured prerequisite"):
        mod.parse_design(_design_text().replace("| none |", "| malformed |", 1))
    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_scalar("{}")
    malformed_row = _design_text().replace(
        "| 1 | exp6972-v611-contract-advisory |",
        "| 1 | extra | exp6972-v611-contract-advisory |",
    )
    with pytest.raises(ValueError, match="malformed design task row"):
        mod.parse_design(malformed_row)
    empty_table = (
        f"**Milestone:** {mod.MILESTONE}\n\n## Exact task contract\n\n"
        "| Order | ID | Title | Deliverable | Structured prerequisites |\n\n## Next\n"
    )
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design(empty_table)

    no_fields = _roadmap()
    no_fields["tasks"][0]["prompt"] = ""
    assert mod.parse_roadmap(no_fields)["tasks"][0]["required_fields_block"] == ""

    manifest = {
        "retired": [{"experiment_id": 1}],
        "retired_experiments": [{"experiment_id": "exp2-x"}],
        "retired_extras": [
            {"experiment_ids": ["exp3-x", 4], "un_retired_experiment_ids": ["exp3-x"]}
        ],
    }
    assert mod.retired_experiment_ids(manifest) == {"exp1", "exp2", "exp4"}
    assert mod.retired_experiment_ids([]) == set()
    malformed_manifest = {
        "retired": "bad",
        "retired_experiments": ["bad"],
        "retired_extras": [
            {"experiment_ids": "bad", "un_retired_experiment_ids": "bad"}
        ],
    }
    assert mod.retired_experiment_ids(malformed_manifest) == set()


def _write_preconditions(root: Path, manifest: str | None = None) -> None:
    """Create isolated readable inputs for preflight and artifact tests."""

    for path in mod.PRECONDITION_PATHS.values():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder\n", encoding="utf-8")
    (root / mod.DESIGN_PATH).write_text(_design_text(), encoding="utf-8")
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(_roadmap(), sort_keys=False), encoding="utf-8"
    )
    (root / mod.EXCLUSION_PATH).write_text(
        manifest
        or "retired: []\nretired_experiments: []\nretired_extras: []\n",
        encoding="utf-8",
    )


def _passing_receipts() -> list[dict[str, Any]]:
    """Return one terminal passing receipt for each required command."""

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


def test_scenario_report_6972_preflight_blocks_missing_and_malformed_inputs(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6972-PREFLIGHT emits complete blocked shapes."""

    missing = mod.build_artifact(tmp_path, "20260904")
    assert mod.REQUIRED_ARTIFACT_FIELDS <= missing.keys()
    assert missing["v611_contract_audit_complete_score"] == 0
    assert missing["contract_conforms_score"] == 0
    assert missing["verdict_class"] == "blocked"
    assert missing["honest_verdict"] == mod.BLOCKED_VERDICT
    assert missing["gate_check_summary"]["failed_check"] == "preconditions"
    assert mod.validate_artifact(missing) == []

    _write_preconditions(tmp_path, manifest="[\n")
    malformed = mod.build_artifact(tmp_path, "20260904")
    row = next(
        item
        for item in malformed["preconditions_checked"]
        if item["resource"] == "full_exclusion_manifest"
    )
    assert row["parse_error"]
    assert malformed["gate_check_summary"]["failed_check"] == "preconditions"

    _write_preconditions(tmp_path)
    (tmp_path / mod.DESIGN_PATH).write_text("not a design", encoding="utf-8")
    bad_contract = mod.build_artifact(tmp_path, "20260904")
    assert bad_contract["gate_check_summary"]["failed_check"] == "contract_parse"


def test_scenario_report_6972_clean_artifact_is_advisory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6972-ADVISORY keeps a clean audit circular and ungated."""

    _write_preconditions(tmp_path)
    monkeypatch.setattr(mod, "run_lint_commands", lambda *_args: _passing_receipts())

    artifact = mod.build_artifact(tmp_path, "20260904")

    assert artifact["v611_contract_audit_complete_score"] == 1
    assert artifact["contract_conforms_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["honest_verdict"] == mod.CONFORMS_VERDICT
    assert mod.validate_artifact(artifact) == []
    assert all(not row["advisory_dependency"] for row in artifact["gate_rows"])


def test_scenario_report_6972_artifact_validator_recomputes_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6972-ARTIFACT rejects forged fields and scores."""

    _write_preconditions(tmp_path)
    monkeypatch.setattr(mod, "run_lint_commands", lambda *_args: _passing_receipts())
    artifact = mod.build_artifact(tmp_path, "20260904")

    for field, value in {
        "field_principles": {},
        "inference_substrate": "llm",
        "v611_contract_audit_complete_score": 0,
        "contract_conforms_score": 0,
        "verdict_class": "partial",
        "honest_verdict": "complete_wrong",
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

    blocked = mod.build_artifact(Path("/missing-root"), "20260904")
    blocked["gate_check_summary"]["failed_check"] = None
    blocked["reproducibility_checksum"] = mod.reproducibility_checksum(blocked)
    assert "blocked_terminal_shape_mismatch" in mod.validate_artifact(blocked)

    wrong_clean = deepcopy(artifact)
    wrong_clean["honest_verdict"] = "complete_wrong"
    wrong_clean["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_clean)
    assert "conforming_terminal_shape_mismatch" in mod.validate_artifact(wrong_clean)

    wrong_defect = deepcopy(artifact)
    wrong_defect["contract_conforms_score"] = 0
    wrong_defect["gate_check_summary"]["passed"] = False
    wrong_defect["verdict_class"] = "positive"
    wrong_defect["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_defect)
    assert "defect_terminal_shape_mismatch" in mod.validate_artifact(wrong_defect)

    nonterminal = deepcopy(artifact)
    nonterminal["v611_contract_audit_complete_score"] = 0
    nonterminal["contract_conforms_score"] = 0
    nonterminal["gate_check_summary"]["audit_complete"] = False
    nonterminal["gate_check_summary"]["passed"] = False
    nonterminal["verdict_class"] = "positive"
    nonterminal["reproducibility_checksum"] = mod.reproducibility_checksum(nonterminal)
    assert "nonterminal_artifact_shape" in mod.validate_artifact(nonterminal)


def test_req_report_6972_command_receipts_distinguish_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6972 separates lint findings from tool failures."""

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


def test_req_report_6972_named_command_and_internal_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6972 runs every named command and reports internal defects."""

    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda _root, name, argv: {
            "name": name,
            "command": " ".join(argv),
            "exit_code": 0,
            "stdout": "clean",
            "stderr": "",
            "outcome": "pass",
            "terminal": True,
            "passed": True,
        },
    )
    evaluation = _evaluate()
    evaluation["retired_scope_rows"][0]["passed"] = False
    receipts = mod.run_lint_commands(ROOT, evaluation)

    assert [row["name"] for row in receipts] == list(mod.COMMAND_NAMES)
    assert receipts[5]["outcome"] == "contract_defect"
    assert receipts[7]["outcome"] == "pass"


def test_scenario_report_6972_tool_failure_and_contract_defect_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6972-ARTIFACT separates blocked tools from audit defects."""

    _write_preconditions(tmp_path)
    tool_failure = _passing_receipts()
    tool_failure[0].update({"terminal": False, "passed": False, "outcome": "tool_failure"})
    monkeypatch.setattr(mod, "run_lint_commands", lambda *_args: tool_failure)
    blocked = mod.build_artifact(tmp_path, "20260904")
    assert blocked["verdict_class"] == "blocked"
    assert mod.validate_artifact(blocked) == []

    roadmap = _roadmap()
    roadmap["tasks"][0]["title"] += " changed"
    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(roadmap, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(mod, "run_lint_commands", lambda *_args: _passing_receipts())
    defect = mod.build_artifact(tmp_path, "20260904")
    assert defect["verdict_class"] == "null"
    assert defect["honest_verdict"] == mod.DEFECT_VERDICT
    assert mod.validate_artifact(defect) == []


def test_req_report_6972_cli_writes_a_valid_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-6972 exposes the required wrapper and strict date parsing."""

    assert mod.main(["--date", "bad"]) == 2
    output = tmp_path / "artifact.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/experiment_6972_v611_contract_advisory.py",
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


def test_req_report_6972_unknown_mutation_is_an_error() -> None:
    """REQ-REPORT-6972 keeps unknown mutation names explicit."""

    with pytest.raises(ValueError, match="unknown mutation"):
        mod.apply_mutation(_roadmap(), "unknown")


def test_req_report_6972_mutation_parse_errors_are_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6972 records evaluator exceptions as failed mutation receipts."""

    original = mod.evaluate_contract
    calls = 0

    def fail_after_baseline(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("mutated parse failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "evaluate_contract", fail_after_baseline)
    rows = mod.build_mutation_rows(_design_text(), _roadmap(), set())
    assert all(row["error"] == "ValueError: mutated parse failure" for row in rows)
    assert all(row["failed_as_expected"] for row in rows)


def test_req_report_6972_direct_main_and_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6972 validates dates and writes only complete JSON objects."""

    assert mod._date_argument("20260904") == "20260904"
    output = tmp_path / "nested" / "artifact.json"
    mod._write_json_atomic(output, {"terminal": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"terminal": True}
    assert not output.with_suffix(".json.tmp").exists()

    artifact = {"honest_verdict": mod.BLOCKED_VERDICT, "verdict_class": "blocked"}
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
