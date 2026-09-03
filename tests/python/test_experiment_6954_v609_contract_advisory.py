"""Focused tests for the V609 advisory contract audit.

Spec refs: REQ-REPORT-6954 and SCENARIO-REPORT-6954-*.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "python/carnot/experiment_6954_v609_contract_advisory.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "experiment_6954_v609_contract_advisory", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
mod = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(mod)


def _gate_text(gates: list[dict[str, Any]]) -> str:
    """Render every fixture gate in the same compact form as the design table."""

    if not gates:
        return "none"
    return "; ".join(
        f"{gate['upstream'].split('-', 1)[0]} `"
        f"{gate['artifact_field']} {gate['op']} {gate['value']}`"
        for gate in gates
    )


def _valid_design() -> str:
    """Build a design source without reading the executable YAML fixture."""

    rows = [
        "# V609 fixture",
        "",
        "**Milestone:** 2026.09.609",
        "",
        "## Exact Task Contract",
        "",
        "| Order | ID | Title | Deliverable | Structured gate |",
        "|---:|---|---|---|---|",
    ]
    for expected in mod.EXPECTED_TASKS:
        rows.append(
            f"| {expected['order']} | {expected['task_id']} | {expected['title']} | "
            f"{expected['deliverable']} | {_gate_text(expected['gates'])} |"
        )
    rows.extend(["", "## Runtime", "", "All local-model tasks own their receipts."])
    return "\n".join(rows)


def _prompt(number: int, deliverable: str, produced_fields: set[str]) -> str:
    """Build one complete prompt with the fields needed by downstream gates."""

    fields = set(mod.COMMON_PROMPT_FIELDS) | produced_fields
    model_text = ""
    scope_text = ""
    if number in mod.MODEL_REQUIREMENTS:
        required_models = ", ".join(mod.MODEL_REQUIREMENTS[number])
        model_text = (
            f"2. Define MODEL_SPECS with {required_models}. Legacy small models may run "
            "CPU smoke tests only and cannot enter headline rows.\n"
        )
        scope_text = (
            "3. Run exactly 12 attempted outputs. After each output, atomically checkpoint "
            "the row and its task runtime receipt.\n"
        )
        fields.update({"model_specs", "checkpoint_rows", "task_runtime_receipt"})
    declared = "; ".join(sorted(fields))
    return (
        "CONTEXT:\nA deterministic advisory fixture.\n\n"
        "EXISTING CODE TO READ FIRST:\n- CODEX.md\n\n"
        "TASK:\nAudit one fixture.\n\n"
        "CONCRETE STEPS:\n"
        "0. PRECONDITIONS: require fixture inputs. On failure, write "
        "blocked_fixture_contract with gate_check_summary.\n"
        f"{model_text}{scope_text}"
        "4. Require verdict_class (positive | circular_positive | null | blocked | "
        "disqualified | partial) and honest_verdict with a terminal prefix consistent "
        "with verdict_class.\n"
        "5. REQUIRED ARTIFACT FIELDS: field_principles with one principle per field; "
        f"{declared}.\n\n"
        "Run command: cd {project_root} && .venv/bin/python "
        f"scripts/experiments/{Path(deliverable).stem}.py --date {{date}}\n"
        f"{mod.FINAL_PROHIBITIONS}\n"
    )


def _valid_roadmap() -> dict[str, Any]:
    """Build the valid V609 YAML shape used by mutation tests."""

    produced = {number: set() for number in mod.EXPECTED_NUMBERS}
    for expected in mod.EXPECTED_TASKS:
        for gate in expected["gates"]:
            produced[mod.experiment_number(gate["upstream"])].add(gate["artifact_field"])
    tasks = []
    for expected in mod.EXPECTED_TASKS:
        number = expected["number"]
        tasks.append(
            {
                "id": expected["task_id"],
                "title": expected["title"],
                "milestone": mod.MILESTONE,
                "deliverable": expected["deliverable"],
                "requires_gpu": number in mod.MODEL_REQUIREMENTS,
                "estimated_wall_time_min": 720 if number in mod.MODEL_REQUIREMENTS else 60,
                "per_unit_rows": True,
                "gated_on": deepcopy(expected["gates"]),
                "prior_failures": [
                    {
                        "experiment_id": f"exp{number - 12}-prior",
                        "verdict": "blocked_prior_scope",
                        "addressed_by": "The V609 fixture uses a bounded replacement.",
                        "retire_if_same_verdict": True,
                    }
                ],
                "prompt": _prompt(number, expected["deliverable"], produced[number]),
            }
        )
    return {
        "milestone": mod.MILESTONE,
        "milestone_title": "fixture",
        "milestone_doc": mod.DESIGN_PATH.as_posix(),
        "tasks": tasks,
    }


def _evaluate(roadmap: dict[str, Any]) -> dict[str, Any]:
    """Run the in-memory evaluator with no retired fixture IDs."""

    return mod.evaluate_contract(_valid_design(), roadmap, retired_ids=set())


def test_req_report_6954_spec_precedes_implementation() -> None:
    """REQ-REPORT-6954 owns every required advisory contract scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6954") :]
    for name in (
        "PREFLIGHT",
        "PARITY",
        "GATES",
        "PROMPTS",
        "MODELS",
        "PRIORS",
        "BOUNDS",
        "MUTATIONS",
        "ADVISORY",
        "ARTIFACT",
    ):
        assert f"SCENARIO-REPORT-6954-{name}" in section


def test_scenario_report_6954_parity_accepts_exact_multigate_fixture() -> None:
    """SCENARIO-REPORT-6954-PARITY preserves all gates and the locked order."""

    result = _evaluate(_valid_roadmap())

    assert result["passed"] is True
    assert len(result["document_task_rows"]) == 12
    assert len(result["yaml_task_rows"]) == 12
    assert len(result["task_parity_rows"][6]["yaml"]["gates"]) == 2
    assert result["document_task_rows"][6]["gates"] == mod.EXPECTED_TASKS[6]["gates"]
    assert all(row["passed"] for row in result["task_parity_rows"])
    assert all(row["passed"] for row in result["gate_contract_rows"])
    assert all(row["passed"] for row in result["producer_field_rows"])


@pytest.mark.parametrize("mutation", mod.REQUIRED_MUTATIONS)
def test_scenario_report_6954_required_mutations_fail(mutation: str) -> None:
    """SCENARIO-REPORT-6954-MUTATIONS rejects every required changed input."""

    roadmap = _valid_roadmap()
    mod.apply_mutation(roadmap, mutation)

    assert _evaluate(roadmap)["passed"] is False


def test_scenario_report_6954_mutation_receipts_cover_all_dimensions() -> None:
    """SCENARIO-REPORT-6954-MUTATIONS records effective and detected changes."""

    rows = mod.build_mutation_rows(_valid_design(), _valid_roadmap(), retired_ids=set())

    assert [row["mutation"] for row in rows] == list(mod.REQUIRED_MUTATIONS)
    assert all(row["input_changed"] for row in rows)
    assert all(row["failed_as_expected"] for row in rows)


def test_scenario_report_6954_gate_and_cascade_edges_fail() -> None:
    """SCENARIO-REPORT-6954-GATES rejects absent, retired, and advisory gates."""

    missing = _valid_roadmap()
    missing["tasks"][3]["gated_on"][0]["upstream"] = "exp9999-missing"
    assert _evaluate(missing)["gate_contract_rows"][0]["passed"] is False

    retired = _valid_roadmap()
    upstream = retired["tasks"][3]["gated_on"][0]["upstream"]
    result = mod.evaluate_contract(_valid_design(), retired, retired_ids={upstream})
    assert result["gate_contract_rows"][0]["passed"] is False

    advisory = _valid_roadmap()
    advisory["tasks"][3]["gated_on"][0] = {
        "upstream": "exp6954-v609-contract-advisory",
        "artifact_field": "v609_contract_conforms_score",
        "op": "==",
        "value": 1,
    }
    evaluated = _evaluate(advisory)
    assert any(not row["passed"] for row in evaluated["cascade_risk_rows"])
    assert evaluated["passed"] is False


def test_scenario_report_6954_model_edges_fail() -> None:
    """SCENARIO-REPORT-6954-MODELS rejects legacy plans and GGUF tokenizer calls."""

    tokenizer = _valid_roadmap()
    tokenizer["tasks"][3]["prompt"] += (
        '\nAutoTokenizer.from_pretrained("unsloth/Qwen3.6-35B-A3B-GGUF")\n'
    )
    assert _evaluate(tokenizer)["model_contract_rows"][3]["passed"] is False

    legacy = _valid_roadmap()
    for model_id in mod.MODEL_REQUIREMENTS[6956]:
        legacy["tasks"][3]["prompt"] = legacy["tasks"][3]["prompt"].replace(model_id, "")
    assert _evaluate(legacy)["model_contract_rows"][3]["legacy_only_headline"] is True
    assert _evaluate(legacy)["model_contract_rows"][3]["passed"] is False


@pytest.mark.parametrize(
    "defect",
    [
        "checkpoint",
        "unit_ceiling",
        "runtime_receipt",
        "per_unit_rows",
        "wall_time_high",
        "wall_time_bool",
        "blocked_artifact",
        "verdict_contract",
    ],
)
def test_scenario_report_6954_scope_edges_fail(defect: str) -> None:
    """SCENARIO-REPORT-6954-BOUNDS rejects each unbounded task shape."""

    roadmap = _valid_roadmap()
    task = roadmap["tasks"][3]
    if defect == "checkpoint":
        task["prompt"] = task["prompt"].replace("atomically checkpoint", "record")
    elif defect == "unit_ceiling":
        task["prompt"] = task["prompt"].replace("exactly 12 attempted outputs", "many outputs")
    elif defect == "runtime_receipt":
        task["prompt"] = task["prompt"].replace("task_runtime_receipt", "runtime_log")
        task["prompt"] = task["prompt"].replace("task runtime receipt", "runtime log")
    elif defect == "per_unit_rows":
        task["per_unit_rows"] = False
    elif defect == "wall_time_high":
        task["estimated_wall_time_min"] = 721
    elif defect == "wall_time_bool":
        task["estimated_wall_time_min"] = True
    elif defect == "blocked_artifact":
        task["prompt"] = task["prompt"].replace(
            "On failure, write blocked_fixture_contract with gate_check_summary.", ""
        )
    else:
        task["prompt"] = task["prompt"].replace(
            "honest_verdict with a terminal prefix consistent with verdict_class", "honest_verdict"
        )

    assert _evaluate(roadmap)["bounded_scope_rows"][3]["passed"] is False


def test_scenario_report_6954_long_per_event_checkpoint_is_bounded() -> None:
    """SCENARIO-REPORT-6954-BOUNDS accepts a detailed per-event checkpoint receipt."""

    prompt = (
        "Use a sealed sequence and match total attempted calls. Atomically checkpoint raw output, "
        "retrieved IDs, prompt hash, exact outcome, write decision, queue state, store hash, tokens, "
        "latency, and runtime receipt after each event."
    )

    checkpoint, ceiling, boundary = mod._has_checkpoint_ceiling(prompt)
    assert checkpoint is True
    assert ceiling is True
    assert boundary == "sealed_upstream_roster_and_fixed_call_total"


def test_scenario_report_6954_parser_edges_fail_closed() -> None:
    """SCENARIO-REPORT-6954-PARITY rejects malformed independent inputs."""

    parsed = mod.parse_design(_valid_design())
    assert len(parsed["tasks"][6]["gates"]) == 2
    with pytest.raises(ValueError, match="milestone"):
        mod.parse_design("no milestone")
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design("**Milestone:** 2026.09.609")
    with pytest.raises(ValueError, match="roadmap mapping"):
        mod.parse_roadmap([])
    with pytest.raises(ValueError, match="tasks list"):
        mod.parse_roadmap({})
    with pytest.raises(ValueError, match="task at order"):
        mod.parse_roadmap({"milestone": mod.MILESTONE, "tasks": ["bad"]})
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_design(_valid_design().replace("none |", "not-a-gate |", 1))
    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_design_gates("exp6955-reformulation-fixture.ready == {}")
    malformed_row = _valid_design().replace(
        "| 1 | exp6953-v609-source-delta |",
        "| 1 | extra | exp6953-v609-source-delta |",
    )
    with pytest.raises(ValueError, match="malformed design task row"):
        mod.parse_design(malformed_row)
    empty_table = (
        "**Milestone:** 2026.09.609\n\n"
        "## Exact Task Contract\n\n"
        "| Order | ID | Title | Deliverable | Structured gate |\n"
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


def test_scenario_report_6954_priors_check_types_and_all_subfields() -> None:
    """SCENARIO-REPORT-6954-PRIORS rejects malformed and mistyped prior rows."""

    malformed = _valid_roadmap()
    malformed["tasks"][0]["prior_failures"] = ["bad"]
    assert _evaluate(malformed)["prior_failure_rows"][0]["passed"] is False

    wrong_bool = _valid_roadmap()
    wrong_bool["tasks"][0]["prior_failures"][0]["retire_if_same_verdict"] = "true"
    assert _evaluate(wrong_bool)["prior_failure_rows"][0]["passed"] is False


def test_scenario_report_6954_preflight_and_parse_failures_write_blocked_schema(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6954-PREFLIGHT retains complete blocked receipts."""

    missing = mod.build_artifact(tmp_path, "20260903")
    assert mod.REQUIRED_ARTIFACT_FIELDS <= missing.keys()
    assert missing["v609_contract_audit_complete_score"] == 0
    assert missing["v609_contract_conforms_score"] == 0
    assert missing["verdict_class"] == "blocked"
    assert missing["honest_verdict"] == mod.BLOCKED_VERDICT
    assert missing["gate_check_summary"]["failed_check"] == "preconditions"
    assert mod.validate_artifact(missing) == []

    for path in mod.PRECONDITION_PATHS.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("invalid", encoding="utf-8")
    malformed = mod.build_artifact(tmp_path, "20260903")
    assert malformed["gate_check_summary"]["failed_check"] == "contract_parse"
    assert "ValueError" in malformed["gate_check_summary"]["observed"]
    assert mod.validate_artifact(malformed) == []


def test_scenario_report_6954_active_contract_is_terminal_and_advisory() -> None:
    """SCENARIO-REPORT-6954-ADVISORY completes even when contract guards find defects."""

    roadmap_before = (ROOT / mod.ACTIVE_ROADMAP_PATH).read_bytes()
    artifact = mod.build_artifact(ROOT, "20260903")

    assert mod.validate_artifact(artifact) == []
    assert artifact["v609_contract_audit_complete_score"] == 1
    assert artifact["v609_contract_conforms_score"] in (0, 1)
    assert artifact["verdict_class"] in ("null", "circular_positive")
    assert all(row["failed_as_expected"] for row in artifact["mutation_rows"])
    assert all(row["outcome"] != "tool_failure" for row in artifact["lint_command_rows"])
    assert (ROOT / mod.ACTIVE_ROADMAP_PATH).read_bytes() == roadmap_before


def test_scenario_report_6954_artifact_validator_recomputes_both_scores() -> None:
    """SCENARIO-REPORT-6954-ARTIFACT rejects forged terminal fields."""

    artifact = mod.build_artifact(ROOT, "20260903")
    mutations = {
        "field_principles": {},
        "inference_substrate": "llm",
        "v609_contract_audit_complete_score": 0,
        "v609_contract_conforms_score": 1 - artifact["v609_contract_conforms_score"],
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


def test_req_report_6954_command_receipts_classify_contract_and_tool_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6954 distinguishes a finding from a command execution failure."""

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

    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")))
    failed = mod._run_command(ROOT, "missing", ["missing"])
    assert failed["outcome"] == "tool_failure"
    assert "OSError" in failed["stderr"]


def test_req_report_6954_cli_writes_valid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6954 supports the required wrapper and direct terminal paths."""

    assert mod.main(["--date", "bad"]) == 2
    assert mod._date_argument("20260903") == "20260903"

    output = tmp_path / "artifact.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/experiment_6954_v609_contract_advisory.py",
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
    assert mod.validate_artifact(written) == []

    minimal = {"honest_verdict": mod.BLOCKED_VERDICT}
    monkeypatch.setattr(mod, "build_artifact", lambda _root, _date: minimal)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    direct = tmp_path / "direct.json"
    assert mod.main(["--date", "20260903", "--output", str(direct)]) == 0
    assert json.loads(direct.read_text(encoding="utf-8")) == minimal

    written_paths: list[Path] = []
    monkeypatch.setattr(mod, "_write_json_atomic", lambda path, _value: written_paths.append(path))
    assert mod.main(["--date", "20260903", "--output", "relative.json"]) == 0
    assert written_paths == [mod.REPO_ROOT / "relative.json"]

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["invalid"])
    assert mod.main(["--date", "20260903", "--output", str(direct)]) == 1


def test_req_report_6954_yaml_round_trip_preserves_gate_types(tmp_path: Path) -> None:
    """REQ-REPORT-6954 reads executable gate scalars after YAML serialization."""

    path = tmp_path / "roadmap.yaml"
    path.write_text(yaml.safe_dump(_valid_roadmap(), sort_keys=False), encoding="utf-8")
    parsed = mod.parse_roadmap(yaml.safe_load(path.read_text(encoding="utf-8")))

    assert parsed["milestone"] == mod.MILESTONE
    assert parsed["tasks"][6]["gates"][1]["value"] == 1
