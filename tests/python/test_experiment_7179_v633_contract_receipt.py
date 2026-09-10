"""Focused tests for REQ-REPORT-7179 and its V633 contract scenarios."""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import time
from typing import Any

import pytest
import yaml

from carnot import experiment_7179_v633_contract_receipt as mod


ROOT = Path(__file__).resolve().parents[2]
IDS = (
    "exp7179-contract-receipt",
    "exp7180-symbolic-edit-fixture",
    "exp7181-qwen38-symbolic-traces",
    "exp7182-grounding-energy-audit",
    "exp7183-supersession-stream",
    "exp7184-revocable-template-csl",
    "exp7185-memory-cold-audit",
    "exp7186-arc-withheld-transfer",
    "exp7187-slice-sampler",
    "exp7188-quantized-transition-audit",
    "exp7189-rust-slice-parity",
    "exp7190-board-placement-receipt",
    "exp7191-capstone",
)
SUBSTRATES = (
    "aggregation",
    "cpu_exact_solver_or_simulator",
    "model_full_generation",
    "no_model_load",
    "cpu_exact_solver_or_simulator",
    "cpu_exact_solver_or_simulator",
    "no_model_load",
    "model_full_generation",
    "cpu_exact_solver_or_simulator",
    "cpu_exact_solver_or_simulator",
    "cpu_exact_solver_or_simulator",
    "aggregation",
    "aggregation",
)


def _markdown(*, milestone: str = mod.MILESTONE) -> str:
    """Build an independent SCENARIO-REPORT-7179-PARITY source."""

    rows = []
    for index, task_id in enumerate(IDS, 1):
        gate = "none"
        if index == 3:
            gate = f"`{IDS[1]}.ready_score == 1`"
        rows.append(
            f"| {index} | `{task_id}` | Task {7178 + index} | "
            f"`results/experiment_{7178 + index}.json` | phase-{index} | "
            f"`{SUBSTRATES[index - 1]}` | {gate} |"
        )
    return "\n".join(
        (
            "# V633 fixture",
            "",
            f"**Milestone:** `{milestone}`",
            "",
            "## Exact Task Contract",
            "",
            "| Order | Task ID | Exact title | Deliverable | Phase | Substrate class | Structured gate |",
            "|---:|---|---|---|---|---|---|",
            *rows,
        )
    )


def _prompt(index: int) -> str:
    """Create a producer declaration for SCENARIO-REPORT-7179-GATES."""

    producer = '- ready_score: principle: "An earlier result is complete."\n' if index == 2 else ""
    return (
        "CONCRETE STEPS:\n"
        "1. REQUIRED ARTIFACT FIELDS:\n"
        '- rows: principle: "Keep every unit."\n' + producer + "Run command: fixture\n"
    )


def _roadmap(*, milestone: str = mod.MILESTONE) -> dict[str, Any]:
    """Build an independent SCENARIO-REPORT-7179-PARITY YAML source."""

    tasks: list[dict[str, Any]] = []
    for index, task_id in enumerate(IDS, 1):
        task: dict[str, Any] = {
            "id": task_id,
            "title": f"Task {7178 + index}",
            "track": f"phase-{index}",
            "milestone": milestone,
            "deliverable": f"results/experiment_{7178 + index}.json",
            "prompt": _prompt(index),
        }
        if index == 3:
            task["gated_on"] = [
                {
                    "upstream": IDS[1],
                    "artifact_field": "ready_score",
                    "op": "==",
                    "value": 1,
                }
            ]
        task["prompt"] = task["prompt"].replace(
            "REQUIRED ARTIFACT FIELDS:\n",
            "REQUIRED ARTIFACT FIELDS:\n"
            f'- inference_substrate_class: principle: "Use {SUBSTRATES[index - 1]}."\n',
        )
        tasks.append(task)
    return {
        "milestone": milestone,
        "milestone_title": "fixture",
        "milestone_doc": str(mod.DESIGN_PATH),
        "tasks": tasks,
    }


def _prior_receipt() -> dict[str, Any]:
    """Keep the exact V632 negative result available as history."""

    return {
        "honest_verdict": "complete_disqualified_v632_markdown_yaml_contract_mismatch",
        "v632_task_contract_conforms_score": 0,
        "expected_task_count": 13,
        "observed_task_count": 4,
        "expected_id_order": [f"exp{number}-expected" for number in range(7166, 7179)],
        "observed_id_order": [f"exp{number}-observed" for number in range(7166, 7170)],
    }


def _write_inputs(
    root: Path,
    *,
    active: dict[str, Any] | None = None,
    next_roadmap: dict[str, Any] | None = None,
    markdown: str | None = None,
) -> tuple[Path, Path, Path]:
    """Write isolated SCENARIO-REPORT-7179-PREFLIGHT inputs."""

    files: dict[Path, str] = {
        mod.DESIGN_PATH: markdown or _markdown(),
        mod.SPEC_PATH: "REQ-REPORT-7179\nSCENARIO-REPORT-7179-PARITY\n",
        mod.PRIOR_ARTIFACT_PATH: json.dumps(_prior_receipt()),
        mod.EXCLUSION_PATH: "retired: []\n",
        Path("CLAUDE.md"): "instructions\n",
        Path("CODEX.md"): "instructions\n",
        Path("research-program.md"): "program\n",
        Path("research-references.md"): "references\n",
    }
    for relative in mod.TOOL_PATHS + (mod.MODULE_PATH, mod.WRAPPER_PATH, mod.TEST_PATH):
        files[relative] = "source\n"
    if active is not None:
        files[mod.ACTIVE_ROADMAP_PATH] = yaml.safe_dump(active, sort_keys=False)
    if next_roadmap is not None:
        files[mod.NEXT_ROADMAP_PATH] = yaml.safe_dump(next_roadmap, sort_keys=False)
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    output = root / mod.DEFAULT_OUTPUT_PATH
    raw_dir = root / mod.RAW_DIR
    checkpoint = root / mod.CHECKPOINT_PATH
    return output, raw_dir, checkpoint


def test_req_report_7179_spec_names_fields_and_scenarios() -> None:
    """REQ-REPORT-7179 names the result fields and focused scenarios."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7179") :]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for name in ("PARITY", "GATES", "PREFLIGHT", "RAW", "HISTORY", "COMMANDS", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7179-{name}" in section


def test_scenario_report_7179_parity_uses_independent_rows() -> None:
    """SCENARIO-REPORT-7179-PARITY accepts all 13 exact independent rows."""

    result = mod.evaluate_contract(_markdown(), _roadmap())
    assert result["passed"] is True
    assert result["expected_id_order"] == list(IDS)
    assert result["markdown_task_rows"] is not result["yaml_task_rows"]
    assert result["task_contract_rows"][2]["checks"] == {
        "id": True,
        "title": True,
        "deliverable": True,
        "phase": True,
        "substrate": True,
        "gates": True,
    }


@pytest.mark.parametrize(
    ("name", "mutation"),
    (
        ("missing", lambda value: value["tasks"].pop()),
        (
            "reordered",
            lambda value: value["tasks"].__setitem__(
                slice(2, 4), [value["tasks"][3], value["tasks"][2]]
            ),
        ),
        ("title", lambda value: value["tasks"][0].__setitem__("title", "changed")),
        (
            "deliverable",
            lambda value: value["tasks"][0].__setitem__("deliverable", "results/changed.json"),
        ),
        ("phase", lambda value: value["tasks"][0].__setitem__("track", "changed")),
        (
            "substrate",
            lambda value: value["tasks"][0].__setitem__(
                "prompt", value["tasks"][0]["prompt"].replace("aggregation", "no_model_load")
            ),
        ),
        ("operator", lambda value: value["tasks"][2]["gated_on"][0].__setitem__("op", "!=")),
        ("value", lambda value: value["tasks"][2]["gated_on"][0].__setitem__("value", 0)),
    ),
)
def test_req_report_7179_contract_mutations_fail(name: str, mutation: Any) -> None:
    """REQ-REPORT-7179 rejects changes to every compared dimension."""

    roadmap = _roadmap()
    mutation(roadmap)
    assert mod.evaluate_contract(_markdown(), roadmap)["passed"] is False, name


def test_scenario_report_7179_gates_fail_closed() -> None:
    """SCENARIO-REPORT-7179-GATES rejects late, absent, and receipt producers."""

    cases = []
    late = _roadmap()
    late["tasks"][2]["gated_on"][0]["upstream"] = IDS[5]
    cases.append(late)
    absent = _roadmap()
    absent["tasks"][1]["prompt"] = absent["tasks"][1]["prompt"].replace("ready_score", "other")
    cases.append(absent)
    receipt = _roadmap()
    receipt["tasks"][2]["gated_on"][0]["upstream"] = IDS[0]
    cases.append(receipt)
    for roadmap in cases:
        assert mod.evaluate_contract(_markdown(), roadmap)["passed"] is False


def test_scenario_report_7179_preflight_selects_by_milestone(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7179-PREFLIGHT prefers matching active, then matching next."""

    stale = _roadmap(milestone="2026.09.632")
    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=stale, next_roadmap=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    assert artifact["yaml_authority_path"] == str(mod.NEXT_ROADMAP_PATH)
    assert artifact["verdict_class"] == "positive"
    assert output.is_file() and checkpoint.is_file()
    assert mod.validate_artifact(artifact) == []

    active_path = tmp_path / mod.ACTIVE_ROADMAP_PATH
    active_path.write_text(yaml.safe_dump(_roadmap(), sort_keys=False), encoding="utf-8")
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    assert artifact["yaml_authority_path"] == str(mod.ACTIVE_ROADMAP_PATH)


def test_scenario_report_7179_blocked_contract_source_is_complete(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7179-PREFLIGHT blocks when neither milestone matches."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap(milestone="2026.09.632"))
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_contract_source"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["field"] == "milestone"
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7179_raw_and_history_are_exact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7179-RAW and HISTORY retain exact source evidence."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    assert (raw_dir / mod.RAW_MARKDOWN_NAME).read_bytes() == (
        tmp_path / mod.DESIGN_PATH
    ).read_bytes()
    assert (raw_dir / mod.RAW_YAML_NAME).read_bytes() == (
        tmp_path / mod.ACTIVE_ROADMAP_PATH
    ).read_bytes()
    assert all(row["hash_matches"] for row in artifact["raw_source_rows"])
    assert artifact["v632_mismatch_history"]["observed_task_count"] == 4
    assert artifact["v632_mismatch_history"]["historical_only"] is True


def test_scenario_report_7179_real_stale_markdown_is_disqualified(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7179-PARITY keeps a readable prior milestone as mismatch."""

    output, raw_dir, checkpoint = _write_inputs(
        tmp_path, active=_roadmap(), markdown=_markdown(milestone="2026.09.632")
    )
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    assert artifact["contract_complete_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["gate_check_summary"]["failed_check"] == "markdown_milestone"
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7179_commands_stream_and_timeout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-REPORT-7179-COMMANDS streams output and keeps real timeouts."""

    ok = mod._run_streaming_command(
        [sys.executable, "-u", "-c", "print('child-line', flush=True)"],
        cwd=tmp_path,
        timeout_s=2.0,
        heartbeat_s=0.01,
        operation="fixture-ok",
    )
    assert ok["exit_code"] == 0 and "child-line" in ok["output"]
    timeout = mod._run_streaming_command(
        [sys.executable, "-u", "-c", "import time; time.sleep(1)"],
        cwd=tmp_path,
        timeout_s=0.05,
        heartbeat_s=0.01,
        operation="fixture-timeout",
    )
    assert timeout["exit_code"] == 124 and timeout["timed_out"] is True
    assert "heartbeat" in capsys.readouterr().out


def test_scenario_report_7179_command_drains_post_exit_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7179-COMMANDS keeps output buffered until child exit."""

    class Process:
        stdout = io.StringIO("tail-without-newline")
        returncode = 0

        @staticmethod
        def poll() -> int:
            return 0

    class Selector:
        @staticmethod
        def register(*_args: object) -> None:
            return None

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(mod.selectors, "DefaultSelector", Selector)
    receipt = mod._run_streaming_command(
        ["fixture"],
        cwd=tmp_path,
        timeout_s=1,
        heartbeat_s=1,
        operation="post-exit-drain",
    )
    assert receipt["output"] == "tail-without-newline"


def test_scenario_report_7179_artifact_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7179-ARTIFACT recomputes state and checksum."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    mutations = (
        lambda value: value.__setitem__("status", "running"),
        lambda value: value.__setitem__("contract_complete_score", 0),
        lambda value: value.__setitem__("verdict_class", "null"),
        lambda value: value.__setitem__("honest_verdict", "forged"),
        lambda value: value.__setitem__("rows", []),
        lambda value: value["task_contract_rows"][0].__setitem__("passed", False),
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:forged"),
    )
    for mutation in mutations:
        changed = deepcopy(artifact)
        mutation(changed)
        assert mod.validate_artifact(changed)


def test_req_report_7179_date_and_validation_commands() -> None:
    """REQ-REPORT-7179 fixes the run date and uses the shipped checks."""

    assert mod._date_argument("20260910") == "20260910"
    with pytest.raises(ValueError):
        mod._date_argument("20260909x")
    names = [
        name
        for name, _argv in mod._validation_commands(
            ROOT, ROOT / mod.DEFAULT_OUTPUT_PATH, mod.ACTIVE_ROADMAP_PATH
        )
    ]
    assert names[:5] == [
        "roadmap_schema",
        "prior_failure",
        "exclusion_manifest",
        "arc_generalization",
        "gate_audit",
    ]


def test_req_report_7179_parsers_reject_malformed_sources() -> None:
    """REQ-REPORT-7179 rejects malformed Markdown, YAML, fields, and gates."""

    with pytest.raises(ValueError):
        mod._load_yaml_bytes(b"\xff", Path("bad.yaml"))
    with pytest.raises(ValueError):
        mod._load_yaml_bytes(b"[]", Path("bad.yaml"))
    assert mod._required_block("ordinary prose") == ""
    assert mod._substrate_class("REQUIRED ARTIFACT FIELDS:\n- rows: x") is None
    with pytest.raises(ValueError):
        mod._parse_scalar("[1]")
    with pytest.raises(ValueError):
        mod._parse_markdown_gate("not a gate", {})
    with pytest.raises(ValueError):
        mod.parse_markdown_contract("no milestone")
    with pytest.raises(ValueError):
        mod.parse_markdown_contract(f"**Milestone:** `{mod.MILESTONE}`")
    with pytest.raises(ValueError):
        mod.parse_markdown_contract(
            f"**Milestone:** `{mod.MILESTONE}`\n## Exact Task Contract\n| Order | Task ID |"
        )
    with pytest.raises(ValueError):
        mod.parse_markdown_contract(
            f"**Milestone:** `{mod.MILESTONE}`\n## Exact Task Contract\n"
            "| Order | Task ID | Exact title | Deliverable | Structured gate |\n"
            "|---|---|---|---|---|\n| x | bad | | | none |"
        )
    with pytest.raises(ValueError):
        mod.parse_yaml_contract(None)
    with pytest.raises(ValueError):
        mod.parse_yaml_contract({"tasks": ["bad"]})
    with pytest.raises(ValueError):
        mod.parse_yaml_contract({"tasks": [{}]})
    malformed = _roadmap()
    malformed["tasks"][0]["gated_on"] = "bad"
    with pytest.raises(ValueError):
        mod.parse_yaml_contract(malformed)


def test_req_report_7179_precondition_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7179 records unavailable hashes, specifications, and output paths."""

    hashes = mod._source_hashes(tmp_path, None)
    assert hashes and all(value is None for value in hashes.values())

    def fail_write(*_args: object, **_kwargs: object) -> object:
        raise OSError("read-only")

    monkeypatch.setattr(mod.tempfile, "NamedTemporaryFile", fail_write)
    assert mod._check_writable_directory(tmp_path) == (False, "OSError: read-only")
    monkeypatch.undo()

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    (tmp_path / mod.SPEC_PATH).unlink()
    rows, authority, roadmap, yaml_bytes = mod._preconditions(tmp_path, output, raw_dir, checkpoint)
    assert authority == mod.ACTIVE_ROADMAP_PATH and roadmap and yaml_bytes
    assert next(row for row in rows if row["check"] == "driving_requirement")["available"] is False


def test_scenario_report_7179_failure_summaries_name_each_contract_class(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7179-ARTIFACT keeps a precise first failure."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    cases: list[tuple[str, dict[str, Any]]] = []
    for expected, mutation in (
        ("yaml_milestone", lambda value: value.__setitem__("yaml_milestone", "stale")),
        ("markdown_task_count", lambda value: value["markdown_task_rows"].pop()),
        ("yaml_task_count", lambda value: value["yaml_task_rows"].pop()),
        ("markdown_id_order", lambda value: value["markdown_id_order"].reverse()),
        ("yaml_id_order", lambda value: value["observed_id_order"].reverse()),
        (
            "gate_producer_contract",
            lambda value: value["gate_producer_rows"][0].__setitem__("passed", False),
        ),
        (
            "receipt_has_downstream_gate",
            lambda value: value["receipt_dependency_rows"][0].update(
                {"passed": False, "observed_consumers": [IDS[1]]}
            ),
        ),
        (
            "task_contract_order_1",
            lambda value: value["task_contract_rows"][0].__setitem__("passed", False),
        ),
        ("stored_contract_evidence", lambda value: value["task_contract_rows"].clear()),
    ):
        changed = deepcopy(artifact)
        mutation(changed)
        cases.append((expected, changed))
    for expected, changed in cases:
        assert mod._failure_summary(changed)["failed_check"] == expected


def test_req_report_7179_score_rejects_incomplete_evidence(tmp_path: Path) -> None:
    """REQ-REPORT-7179 computes one only from exact complete stored rows."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    mutations = (
        lambda value: value.__setitem__("markdown_task_rows", None),
        lambda value: value.__setitem__("markdown_milestone", "stale"),
        lambda value: value["gate_contract_rows"].pop(),
        lambda value: value["markdown_task_rows"][0].__setitem__("id", "wrong"),
        lambda value: value["yaml_task_rows"][0].__setitem__("id", "wrong"),
        lambda value: value["gate_contract_rows"][0].__setitem__("passed", False),
    )
    for mutation in mutations:
        changed = deepcopy(artifact)
        mutation(changed)
        assert mod._score_from_artifact(changed) == 0


def test_scenario_report_7179_validator_rejects_all_field_classes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7179-ARTIFACT checks every required evidence class."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    assert mod.validate_artifact(None) == ["artifact_mapping_required"]
    missing = deepcopy(artifact)
    del missing["status"]
    assert mod.validate_artifact(missing)[0] == "missing_required_field:status"
    mutations: tuple[tuple[str, Any], ...] = (
        ("field_principles_invalid", lambda value: value.__setitem__("field_principles", {})),
        ("run_date_invalid", lambda value: value.__setitem__("run_date", "20260909")),
        (
            "inference_substrate_invalid",
            lambda value: value.__setitem__("inference_substrate", "model"),
        ),
        ("execution_venue_invalid", lambda value: value.__setitem__("execution_venue", "gpu")),
        ("host_identity_invalid", lambda value: value.__setitem__("host_identity", "")),
        ("duration_s_invalid", lambda value: value.__setitem__("duration_s", True)),
        (
            "preconditions_checked_invalid",
            lambda value: value.__setitem__("preconditions_checked", "bad"),
        ),
        (
            "source_artifact_hashes_invalid",
            lambda value: value.__setitem__("source_artifact_hashes", {}),
        ),
        ("expected_task_count_invalid", lambda value: value.__setitem__("expected_task_count", 12)),
        ("expected_id_order_invalid", lambda value: value.__setitem__("expected_id_order", [])),
        ("random_seed_invalid", lambda value: value.__setitem__("random_seed", 1)),
        ("verifier_is_oracle_invalid", lambda value: value.__setitem__("verifier_is_oracle", True)),
        (
            "raw_source_rows_invalid",
            lambda value: value["raw_source_rows"][0].__setitem__("hash_matches", False),
        ),
        (
            "v632_mismatch_history_invalid",
            lambda value: value["v632_mismatch_history"].__setitem__("historical_only", False),
        ),
        (
            "validation_command_rows_invalid",
            lambda value: value.__setitem__("validation_command_rows", [{"name": "wrong"}]),
        ),
        (
            "inference_substrate_class_invalid",
            lambda value: value.__setitem__("inference_substrate_class", "blocked_no_run"),
        ),
    )
    for expected, mutation in mutations:
        changed = deepcopy(artifact)
        mutation(changed)
        assert expected in mod.validate_artifact(changed)


def test_scenario_report_7179_parse_failure_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7179-PARITY disqualifies a readable malformed table."""

    output, raw_dir, checkpoint = _write_inputs(
        tmp_path,
        active=_roadmap(),
        markdown=f"**Milestone:** `{mod.MILESTONE}`\n## Exact Task Contract\nnot a table\n",
    )
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    assert artifact["gate_check_summary"]["failed_check"] == "contract_parse"
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7179_validation_runner_records_real_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7179-COMMANDS persists each unchanged result."""

    output, raw_dir, checkpoint = _write_inputs(tmp_path, active=_roadmap())
    artifact = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=False,
    )
    monkeypatch.setattr(
        mod,
        "_validation_commands",
        lambda *_args: (
            ("roadmap_schema", [sys.executable, "-c", "print('one')"]),
            ("prior_failure", [sys.executable, "-c", "print('two')"]),
        ),
    )
    rows = mod.run_validation_commands(tmp_path, checkpoint, artifact, time.monotonic())
    assert [row["name"] for row in rows] == ["roadmap_schema", "prior_failure"]
    assert all(row["passed"] for row in rows)
    rebuilt = mod.build_artifact(
        tmp_path,
        "20260910",
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
        run_commands=True,
    )
    assert len(rebuilt["validation_command_rows"]) == 2


def test_req_report_7179_main_paths(monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """REQ-REPORT-7179 exposes one strict-date terminal CLI."""

    with pytest.raises(mod.argparse.ArgumentTypeError):
        mod._date_argument("20260909")
    built: dict[str, Any] = {"honest_verdict": "done", "contract_complete_score": 0}
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: built)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    assert mod.main(["--date", "20260910"]) == 0
    assert "complete verdict=done" in capsys.readouterr().out
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["bad"])
    assert mod.main(["--date", "20260910"]) == 1
