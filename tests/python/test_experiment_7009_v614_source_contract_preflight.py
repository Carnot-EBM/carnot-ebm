"""Tests for REQ-REPORT-7009 and its V614 contract scenarios."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
import yaml

from carnot import experiment_7009_v614_source_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]


def _design_text() -> str:
    """Read the Markdown contract without using the implementation parser."""

    return (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")


def _active_roadmap() -> dict[str, Any]:
    """Read the activated YAML without using the implementation parser."""

    value = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _valid_prompt(number: int, produced_field: str) -> str:
    """Build one complete prompt used only to test the contract evaluator."""

    substrate = (
        "live_llm_inference"
        if number in mod.LLM_TASK_NUMBERS
        else f"deterministic_exp{number}_no_llm"
    )
    model_text = ""
    if number == 7013:
        model_text = "MODEL_SPECS: " + ", ".join(mod.REQUIRED_EXP7013_MODELS) + "; "
    elif number == 7021:
        model_text = f"MODEL_SPECS: {mod.QWEN36_MODEL}; "
    return (
        "TASK: Compare bounded rows.\n"
        "REQUIRED ARTIFACT FIELDS: field_principles; "
        f"inference_substrate ({substrate}); duration_s; {model_text}{produced_field}; "
        "rows; random_seed; reproducibility_checksum; gate_check_summary with failed check, "
        "expected value, and observed value; verifier_is_oracle (false); verdict_class "
        "(positive | circular_positive | null | blocked | disqualified | partial); "
        "honest_verdict.\n"
        "Run command: test only\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py.\n"
    )


def _conforming_roadmap() -> dict[str, Any]:
    """Build a complete YAML fixture from independently parsed Markdown facts."""

    design = mod.parse_design(_design_text())
    tasks = []
    for row in design["tasks"]:
        task = {
            "id": row["id"],
            "title": row["title"],
            "deliverable": row["deliverable"],
            "milestone": mod.MILESTONE,
            "requires_gpu": row["number"] in mod.LLM_TASK_NUMBERS,
            "per_unit_rows": True,
            "gated_on": deepcopy(row["gates"]),
            "prompt": _valid_prompt(row["number"], row["produced_field"]),
        }
        if row["number"] == 7009:
            task["prior_failures"] = [
                {
                    "experiment_id": "exp6942-v608-contract-preflight",
                    "verdict": "blocked_v608_contract_preflight",
                    "addressed_by": "The new audit reads one frozen V614 contract.",
                    "retire_if_same_verdict": True,
                }
            ]
        tasks.append(task)
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _source_rows(*, changed: bool = False) -> list[dict[str, Any]]:
    """Make one terminal receipt for each required public route."""

    rows = []
    for index, route in enumerate(mod.REQUIRED_SOURCE_ROUTES):
        category = "semantic" if route.startswith("semantic_") else "primary"
        if route.startswith("huggingface_"):
            category = "secondary"
        is_change = changed and index == 0
        rows.append(
            {
                "route": route,
                "category": category,
                "url": f"https://example.test/{route}",
                "query": route.replace("_", " "),
                "accessed_on": "2026-09-05",
                "outcome": "ok",
                "http_status": 200,
                "terminal": True,
                "published_or_changed_at": "2026-09-05T01:00:00Z" if is_change else None,
                "post_marker_change_proven": is_change,
                "relevant": is_change,
                "finding": "new relevant primary source" if is_change else "no post-marker change",
                "response_sha256": "sha256:" + hashlib.sha256(route.encode()).hexdigest(),
            }
        )
    return rows


def _passing_receipts() -> list[dict[str, Any]]:
    """Make terminal passing receipts for all required validation commands."""

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
        for name in mod.VALIDATION_COMMAND_NAMES
    ]


def _write_preconditions(root: Path, roadmap: dict[str, Any] | None = None) -> None:
    """Create isolated readable inputs for precondition tests."""

    for source in (
        mod.DESIGN_PATH,
        mod.CAPSTONE_PATH,
        mod.EXCLUSION_PATH,
        mod.REFERENCE_PATH,
        mod.SPEC_PATH,
    ):
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
    target = root / mod.ACTIVE_ROADMAP_PATH
    target.write_text(yaml.safe_dump(roadmap or _conforming_roadmap()), encoding="utf-8")


def test_req_report_7009_spec_precedes_implementation() -> None:
    """REQ-REPORT-7009 owns every required audit scenario and field."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7009") :]
    for name in ("PREFLIGHT", "PARITY", "GATES", "DISCIPLINE", "SOURCES", "NOCHANGE", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7009-{name}" in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_7009_parity_parses_fourteen_markdown_rows() -> None:
    """SCENARIO-REPORT-7009-PARITY reads both Markdown tables independently."""

    design = mod.parse_design(_design_text())

    assert design["milestone"] == mod.MILESTONE
    assert [row["number"] for row in design["tasks"]] == list(mod.EXPECTED_NUMBERS)
    assert len(design["tasks"]) == 14
    assert sum(len(row["gates"]) for row in design["tasks"]) == 10
    assert all(row["produced_field"] for row in design["tasks"])


def test_scenario_report_7009_current_yaml_mismatch_stays_explicit() -> None:
    """SCENARIO-REPORT-7009-PARITY preserves the frozen seven-row YAML defect."""

    result = mod.evaluate_contract(_design_text(), _active_roadmap(), set())

    assert result["passed"] is False
    assert len(result["markdown_task_rows"]) == 14
    assert len(result["yaml_task_rows"]) == 7
    assert len(result["task_contract_rows"]) == 14
    assert [row["number"] for row in result["yaml_task_rows"]] == list(range(7009, 7016))
    assert [
        row["number"] for row in result["task_contract_rows"] if not row["yaml_present"]
    ] == list(range(7016, 7023))


def test_scenario_report_7009_conforming_fixture_checks_all_contract_rows() -> None:
    """SCENARIO-REPORT-7009-PARITY accepts all exact rows and producer fields."""

    result = mod.evaluate_contract(_design_text(), _conforming_roadmap(), set())

    assert result["passed"] is True
    assert len(result["task_contract_rows"]) == 14
    assert len(result["gate_producer_rows"]) == 10
    for key in (
        "task_contract_rows",
        "title_parity_rows",
        "deliverable_parity_rows",
        "gate_contract_rows",
        "gate_producer_rows",
        "prior_failure_rows",
        "retired_id_rows",
        "model_compliance_rows",
        "artifact_field_rows",
        "prompt_tail_rows",
    ):
        assert all(row["passed"] for row in result[key]), key
    assert next(row for row in result["model_compliance_rows"] if row["number"] == 7013)[
        "required_three_models_present"
    ]
    assert next(row for row in result["model_compliance_rows"] if row["number"] == 7021)[
        "qwen36_present"
    ]
    assert next(row for row in result["artifact_field_rows"] if row["number"] == 7022)[
        "capstone_ungated"
    ]


@pytest.mark.parametrize("mutation", mod.REQUIRED_MUTATIONS)
def test_scenario_report_7009_each_required_mutation_fails(mutation: str) -> None:
    """SCENARIO-REPORT-7009-DISCIPLINE detects every named mutation."""

    roadmap = _conforming_roadmap()
    mod.apply_mutation(roadmap, mutation)

    assert mod.evaluate_contract(_design_text(), roadmap, set())["passed"] is False


def test_scenario_report_7009_mutation_receipts_are_effective() -> None:
    """SCENARIO-REPORT-7009-DISCIPLINE proves each mutation changed valid input."""

    rows = mod.build_mutation_rows(_design_text(), _conforming_roadmap(), set())

    assert [row["mutation"] for row in rows] == list(mod.REQUIRED_MUTATIONS)
    assert all(row["baseline_passed"] for row in rows)
    assert all(row["input_changed"] and row["failed_as_expected"] for row in rows)


def test_scenario_report_7009_gate_failures_cover_all_boundaries() -> None:
    """SCENARIO-REPORT-7009-GATES rejects missing, late, cross-milestone, non-bare, and retired producers."""

    cases = []
    missing = _conforming_roadmap()
    missing["tasks"][4]["gated_on"][0]["upstream"] = "exp9999-missing"
    cases.append((missing, "producer_present"))
    late = _conforming_roadmap()
    late["tasks"][4]["gated_on"][0]["upstream"] = late["tasks"][5]["id"]
    cases.append((late, "producer_precedes_consumer"))
    cross = _conforming_roadmap()
    cross["tasks"][3]["milestone"] = "2026.09.613"
    cases.append((cross, "same_milestone"))
    nonbare = _conforming_roadmap()
    nonbare["tasks"][4]["gated_on"][0]["artifact_field"] = "nested.ready"
    cases.append((nonbare, "bare_field"))
    typo = _conforming_roadmap()
    typo["tasks"][4]["gated_on"][0]["artifact_field"] = "ready_scor"
    cases.append((typo, "field_declared"))
    for roadmap, field in cases:
        assert any(
            not row[field]
            for row in mod.evaluate_contract(_design_text(), roadmap, set())["gate_producer_rows"]
        )

    upstream = _conforming_roadmap()["tasks"][4]["gated_on"][0]["upstream"]
    assert any(
        row["upstream_retired"]
        for row in mod.evaluate_contract(_design_text(), _conforming_roadmap(), {upstream})[
            "gate_producer_rows"
        ]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_id", ""),
        ("verdict", ""),
        ("addressed_by", ""),
        ("retire_if_same_verdict", "true"),
    ],
)
def test_scenario_report_7009_prior_failure_schema_is_exact(field: str, value: object) -> None:
    """SCENARIO-REPORT-7009-DISCIPLINE rejects each malformed prior field."""

    roadmap = _conforming_roadmap()
    roadmap["tasks"][0]["prior_failures"][0][field] = value

    assert any(
        not row["passed"]
        for row in mod.evaluate_contract(_design_text(), roadmap, set())["prior_failure_rows"]
    )


def test_scenario_report_7009_model_and_legacy_rules_fail_closed() -> None:
    """SCENARIO-REPORT-7009-DISCIPLINE checks both live tasks and smoke-only legacy models."""

    missing_family = _conforming_roadmap()
    missing_family["tasks"][4]["prompt"] = missing_family["tasks"][4]["prompt"].replace(
        mod.REQUIRED_EXP7013_MODELS[0], "unsloth/not-current-GGUF"
    )
    assert not mod.evaluate_contract(_design_text(), missing_family, set())[
        "model_compliance_rows"
    ][4]["passed"]

    missing_qwen = _conforming_roadmap()
    missing_qwen["tasks"][12]["prompt"] = missing_qwen["tasks"][12]["prompt"].replace(
        mod.QWEN36_MODEL, "unsloth/not-current-GGUF"
    )
    assert not mod.evaluate_contract(_design_text(), missing_qwen, set())["model_compliance_rows"][
        12
    ]["passed"]

    legacy = _conforming_roadmap()
    legacy["tasks"][4]["prompt"] += "Use Qwen3.5-0.8B as a headline model."
    assert not mod.evaluate_contract(_design_text(), legacy, set())["model_compliance_rows"][4][
        "passed"
    ]


def test_req_report_7009_parsers_resolution_and_manifest_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7009 rejects malformed contracts and resolves activated YAML."""

    assert len(mod.parse_roadmap(_conforming_roadmap())["tasks"]) == 14
    with pytest.raises(ValueError, match="milestone"):
        mod.parse_design("no milestone")
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design(f"**Milestone:** `{mod.MILESTONE}`")
    with pytest.raises(ValueError, match="gate table"):
        mod.parse_design(
            _design_text().replace("## Dependency and Gate Contract", "## Missing Gate Contract")
        )
    with pytest.raises(ValueError, match="roadmap mapping"):
        mod.parse_roadmap([])
    with pytest.raises(ValueError, match="tasks list"):
        mod.parse_roadmap({})
    with pytest.raises(ValueError, match="task at order"):
        mod.parse_roadmap({"tasks": ["bad"]})
    with pytest.raises(ValueError, match="task lists"):
        mod.parse_roadmap({"tasks": [{"gated_on": "bad"}]})
    with pytest.raises(ValueError, match="task lists"):
        mod.parse_roadmap({"tasks": [{"gated_on": ["bad"]}]})

    active = tmp_path / mod.ACTIVE_ROADMAP_PATH
    active.write_text(yaml.safe_dump(_conforming_roadmap()), encoding="utf-8")
    assert mod.resolve_v614_roadmap(tmp_path) == active
    draft = tmp_path / mod.DRAFT_ROADMAP_PATH
    draft.write_text(yaml.safe_dump(_conforming_roadmap()), encoding="utf-8")
    assert mod.resolve_v614_roadmap(tmp_path) == draft
    draft.write_text("milestone: wrong\ntasks: []\n", encoding="utf-8")
    assert mod.resolve_v614_roadmap(tmp_path) == active
    active.unlink()
    with pytest.raises(FileNotFoundError, match="V614 YAML"):
        mod.resolve_v614_roadmap(tmp_path)

    malformed = tmp_path / "bad.yaml"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        mod.load_yaml(malformed)

    manifest = {
        "retired": [{"experiment_id": 1}],
        "retired_experiments": [{"experiment_id": "exp2-old"}],
        "retired_extras": [
            {"experiment_ids": ["exp3-old", 4], "un_retired_experiment_ids": ["exp3-old"]}
        ],
    }
    assert mod.retired_experiment_ids(manifest) == {"exp1", "exp2", "exp4"}
    assert mod.retired_experiment_ids([]) == set()


def test_scenario_report_7009_sources_are_terminal_and_no_change_is_explicit() -> None:
    """SCENARIO-REPORT-7009-SOURCES and NOCHANGE keep source classes separate."""

    evidence = mod.build_source_evidence(_source_rows())

    assert mod.source_delta_complete_score(evidence["source_query_rows"]) == 1
    assert len(evidence["primary_source_rows"]) == 7
    assert len(evidence["secondary_source_rows"]) == 2
    assert len(evidence["semantic_scholar_rows"]) == 2
    assert evidence["reference_append_rows"][0]["action"] == "no_change"
    assert mod.source_delta_complete_score(_source_rows()[:-1]) == 0
    changed = mod.build_source_evidence(_source_rows(changed=True))
    assert changed["reference_append_rows"][0]["action"] == "append_required"
    assert changed["reference_append_rows"][0]["appended"] is False


def test_req_report_7009_source_fetcher_records_all_terminal_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7009 preserves successful and failed access as terminal evidence."""

    class Response:
        status = 200
        headers = {"Last-Modified": "Sat, 05 Sep 2026 01:00:00 GMT"}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return b'{"updated_at":"2026-09-05T01:00:00Z"}'

    monkeypatch.setattr(mod, "urlopen", lambda *_args, **_kwargs: Response())
    success = mod._fetch_url("https://example.test", timeout_s=1.0)
    assert success["outcome"] == "ok" and success["last_modified"]

    for error, outcome in (
        (HTTPError("https://example.test", 429, "limited", {}, None), "http_error"),
        (URLError("offline"), "network_error"),
        (OSError("broken"), "tool_error"),
    ):
        monkeypatch.setattr(
            mod, "urlopen", lambda *_args, _error=error, **_kwargs: (_ for _ in ()).throw(_error)
        )
        assert mod._fetch_url("https://example.test")["outcome"] == outcome

    rows = mod.collect_source_query_rows(
        "20260905",
        fetch=lambda _url: {
            "outcome": "ok",
            "http_status": 200,
            "terminal": True,
            "response_sha256": "sha256:test",
            "last_modified": None,
            "response_excerpt": "result",
        },
    )
    assert [row["route"] for row in rows] == list(mod.REQUIRED_SOURCE_ROUTES)
    assert all(row["accessed_on"] == "2026-09-05" for row in rows)


def test_scenario_report_7009_preflight_blocks_external_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7009-PREFLIGHT emits a complete blocked artifact."""

    missing = mod.build_artifact(
        tmp_path,
        "20260905",
        output_path=tmp_path / "artifact.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
    )
    assert mod.REQUIRED_ARTIFACT_FIELDS <= missing.keys()
    assert missing["verdict_class"] == "blocked"
    assert missing["honest_verdict"] == mod.BLOCKED_VERDICT
    assert missing["gate_check_summary"]["failed_check"] == "preconditions"
    assert mod.validate_artifact(missing) == []

    _write_preconditions(tmp_path)
    (tmp_path / mod.CAPSTONE_PATH).write_text("{", encoding="utf-8")
    malformed = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "artifact.json", network_available=True
    )
    assert malformed["verdict_class"] == "blocked"

    _write_preconditions(tmp_path)
    offline = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "artifact.json", network_available=False
    )
    assert offline["gate_check_summary"]["observed_value"] == "network_unavailable"

    monkeypatch.setattr(mod, "_artifact_path_writable", lambda _path: False)
    unwritable = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "artifact.json", network_available=True
    )
    assert any(
        row["resource"] == "writable_artifact_path" and not row["available"]
        for row in unwritable["preconditions_checked"]
    )


def test_scenario_report_7009_artifacts_derive_positive_and_disqualified(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7009-ARTIFACT derives verdicts from independent rows."""

    positive = mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "positive.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
        roadmap_override=_conforming_roadmap(),
    )
    assert positive["expected_task_count"] == positive["observed_task_count"] == 14
    assert (
        positive["expected_id_order"] == positive["observed_id_order"] == list(mod.EXPECTED_NUMBERS)
    )
    assert positive["v614_source_delta_complete_score"] == 1
    assert positive["v614_task_contract_conforms_score"] == 1
    assert positive["verdict_class"] == "positive"
    assert mod.validate_artifact(positive) == []

    actual = mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "actual.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
    )
    assert actual["observed_task_count"] == 7
    assert actual["v614_task_contract_conforms_score"] == 0
    assert actual["verdict_class"] == "disqualified"
    assert actual["honest_verdict"] == mod.DISQUALIFIED_VERDICT
    assert mod.validate_artifact(actual) == []

    nonterminal = deepcopy(_passing_receipts())
    nonterminal[0]["terminal"] = False
    blocked = mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "blocked.json",
        source_rows=_source_rows(),
        command_rows=nonterminal,
        network_available=True,
    )
    assert blocked["gate_check_summary"]["failed_check"] == "validation_tools"


def test_scenario_report_7009_validator_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7009-ARTIFACT rejects changed fields, scores, and checksum."""

    artifact = mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "artifact.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
        roadmap_override=_conforming_roadmap(),
    )
    for field, value in {
        "field_principles": {},
        "inference_substrate": "wrong",
        "v614_source_delta_complete_score": 0,
        "v614_task_contract_conforms_score": 0,
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_wrong",
    }.items():
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert mod.validate_artifact(changed), field

    missing = deepcopy(artifact)
    missing.pop("rows")
    assert "missing_required_fields:rows" in mod.validate_artifact(missing)
    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(bad_checksum)


def test_req_report_7009_commands_dates_and_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7009 records command failures and writes only its artifact."""

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["check"], 1, "finding", ""),
    )
    defect = mod._run_command(ROOT, "check", [sys.executable, "-V"])
    assert defect["outcome"] == "contract_defect" and defect["terminal"]

    def timeout(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["check"], 1, output="partial", stderr="late")

    monkeypatch.setattr(mod.subprocess, "run", timeout)
    assert mod._run_command(ROOT, "check", [sys.executable, "-V"])["outcome"] == "tool_failure"
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing"))
    )
    assert mod._run_command(ROOT, "check", ["missing"])["terminal"] is False

    calls: list[str] = []
    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda _root, name, _argv: calls.append(name) or _passing_receipts()[0] | {"name": name},
    )
    assert len(mod.run_validation_commands(ROOT, tmp_path / "artifact.json")) == len(
        mod.EXTERNAL_COMMAND_NAMES
    )
    assert calls == list(mod.EXTERNAL_COMMAND_NAMES)

    assert mod._date_argument("20260905") == "20260905"
    assert mod.main(["--date", "bad"]) == 2
    target = tmp_path / "nested" / "artifact.json"
    mod._write_json_atomic(target, {"terminal": True})
    assert json.loads(target.read_text()) == {"terminal": True}

    complete = {field: [] for field in mod.REQUIRED_ARTIFACT_FIELDS}
    complete["source_query_rows"] = _source_rows()
    complete["verdict_class"] = "disqualified"
    complete["honest_verdict"] = mod.DISQUALIFIED_VERDICT
    builds: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mod, "build_artifact", lambda *_args, **kwargs: builds.append(kwargs) or deepcopy(complete)
    )
    monkeypatch.setattr(mod, "run_validation_commands", lambda *_args: [])
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    writes: list[Path] = []
    monkeypatch.setattr(mod, "_write_json_atomic", lambda path, _value: writes.append(path))
    assert mod.main(["--date", "20260905", "--output", str(target)]) == 0
    assert len(builds) == 2 and writes == [target, target]


def test_req_report_7009_helpers_cover_parse_and_write_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7009 keeps low-level parser and write failures explicit."""

    with pytest.raises(ValueError, match="unknown mutation"):
        mod.apply_mutation(_conforming_roadmap(), "unknown")
    assert mod._required_block("no required marker") == ""
    assert mod._normal_exp_id("bad") is None
    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod._artifact_path_writable(tmp_path / "artifact.json") is False


def test_req_report_7009_markdown_parser_reports_each_malformed_shape() -> None:
    """REQ-REPORT-7009 gives a terminal parser error for each malformed table shape."""

    text = _design_text()
    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_scalar("{}")
    with pytest.raises(ValueError, match="malformed design task row"):
        mod.parse_design(text.replace("| 1 | `exp7009", "| 1 | extra | `exp7009", 1))
    with pytest.raises(ValueError, match="malformed design task row"):
        mod.parse_design(text.replace("| 1 | `exp7009", "| 1 | `invalid", 1))
    empty_tasks = f"**Milestone:** `{mod.MILESTONE}`\n## Exact Task Contract\n| no rows |\n## Dependency and Gate Contract\n| no rows |"
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design(empty_tasks)
    with pytest.raises(ValueError, match="malformed design gate row"):
        mod.parse_design(text.replace("| Exp7009 |", "| Exp7009 | extra |", 1))
    with pytest.raises(ValueError, match="malformed design gate row"):
        mod.parse_design(
            text.replace(
                "| Exp7009 | none; advisory | `v614_task_contract_conforms_score` |",
                "| Exp7009 | none; advisory | `bad.field` |",
                1,
            )
        )
    with pytest.raises(ValueError, match="malformed structured gate"):
        mod.parse_design(text.replace("Exp7012 ready `== 1`", "Exp7012 ready", 1))
    gate_section = text.split("## Dependency and Gate Contract", 1)[1].split("\n## ", 1)[0]
    without_gate_rows = text.replace(gate_section, "\n| no gate rows |\n")
    with pytest.raises(ValueError, match="gate table"):
        mod.parse_design(without_gate_rows)
    missing_one = "\n".join(
        line for line in text.splitlines() if not line.startswith("| Exp7009 |")
    )
    with pytest.raises(ValueError, match="gate row is missing"):
        mod.parse_design(missing_one)


def test_req_report_7009_yaml_and_manifest_defensive_branches(tmp_path: Path) -> None:
    """REQ-REPORT-7009 normalizes null lists and ignores irrelevant manifest shapes."""

    normalized = mod.parse_roadmap({"tasks": [{"gated_on": None, "prior_failures": None}]})
    assert normalized["tasks"][0]["gates"] == []
    active = tmp_path / mod.ACTIVE_ROADMAP_PATH
    active.write_text(yaml.safe_dump(_conforming_roadmap()), encoding="utf-8")
    draft = tmp_path / mod.DRAFT_ROADMAP_PATH
    draft.write_text("[\n", encoding="utf-8")
    assert mod.resolve_v614_roadmap(tmp_path) == active
    assert (
        mod.retired_experiment_ids(
            {
                "not_retired": [{"experiment_id": 9}],
                "retired": ["bad", {"experiment_ids": "bad", "un_retired_experiment_ids": "bad"}],
            }
        )
        == set()
    )


def test_req_report_7009_source_dates_and_precondition_error_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7009 preserves GitHub dates and reports marker, probe, and read failures."""

    assert (
        mod._observed_source_date(
            {"route": "github_extropic", "known_date": None},
            {"response_excerpt": '{"updated_at":"2026-09-05T01:00:00Z"}'},
        )
        == "2026-09-05T01:00:00Z"
    )
    _write_preconditions(tmp_path)
    reference = tmp_path / mod.REFERENCE_PATH
    reference.write_text("no marker\n", encoding="utf-8")
    rows, _, _ = mod._preconditions(tmp_path, tmp_path / "artifact.json", True)
    assert next(row for row in rows if row["resource"] == "reference_file")["parse_error"]

    reference.write_bytes((ROOT / mod.REFERENCE_PATH).read_bytes())
    monkeypatch.setattr(mod, "_fetch_url", lambda *_args, **_kwargs: {"outcome": "http_error"})
    rows, _, _ = mod._preconditions(tmp_path, tmp_path / "artifact.json", None)
    assert next(row for row in rows if row["resource"] == "network_access")["available"]

    original_read = Path.read_text

    def fail_reference(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == reference:
            raise OSError("unreadable reference")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_reference)
    rows, _, _ = mod._preconditions(tmp_path, tmp_path / "artifact.json", True)
    assert (
        "OSError" in next(row for row in rows if row["resource"] == "reference_file")["parse_error"]
    )


def test_req_report_7009_artifact_covers_parse_source_and_validator_failures(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7009 derives blocked states and validates malformed diagnostics."""

    parse_blocked = mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "parse.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
        roadmap_override=[],
    )
    assert parse_blocked["gate_check_summary"]["failed_check"] == "contract_parse"

    source_blocked = mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "source.json",
        source_rows=_source_rows()[:-1],
        command_rows=_passing_receipts(),
        network_available=True,
        roadmap_override=_conforming_roadmap(),
    )
    assert source_blocked["gate_check_summary"]["failed_check"] == "source_routes"

    positive = mod.build_artifact(
        ROOT,
        "20260905",
        output_path=tmp_path / "positive.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
        roadmap_override=_conforming_roadmap(),
    )
    bad_class = deepcopy(positive)
    bad_class["verdict_class"] = "unknown"
    bad_class["reproducibility_checksum"] = mod.reproducibility_checksum(bad_class)
    assert "verdict_class_invalid" in mod.validate_artifact(bad_class)
    bad_summary = deepcopy(positive)
    bad_summary["gate_check_summary"] = []
    bad_summary["reproducibility_checksum"] = mod.reproducibility_checksum(bad_summary)
    assert "gate_check_summary_invalid" in mod.validate_artifact(bad_summary)


def test_req_report_7009_warning_receipt_and_main_terminal_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7009 covers advisory verifier output and all CLI exits."""

    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda _root, name, _argv: {
            "name": name,
            "command": name,
            "exit_code": 1 if name == "adversarial_verification" else 0,
            "stdout": "[WARN] no-LLM review",
            "stderr": "",
            "outcome": "contract_defect" if name == "adversarial_verification" else "pass",
            "terminal": True,
            "passed": name != "adversarial_verification",
        },
    )
    warning = next(
        row
        for row in mod.run_validation_commands(ROOT, tmp_path / "artifact.json")
        if row["name"] == "adversarial_verification"
    )
    assert warning["outcome"] == "warning" and warning["passed"]

    blocked = {"verdict_class": "blocked", "honest_verdict": mod.BLOCKED_VERDICT}
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: deepcopy(blocked))
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    monkeypatch.setattr(mod, "_write_json_atomic", lambda *_args: None)
    assert mod.main(["--date", "20260905", "--output", "relative.json"]) == 0

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["invalid"])
    assert mod.main(["--date", "20260905", "--output", str(tmp_path / "bad.json")]) == 1

    complete = {
        "verdict_class": "disqualified",
        "honest_verdict": mod.DISQUALIFIED_VERDICT,
        "source_query_rows": _source_rows(),
    }
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: deepcopy(complete))
    validations = iter([[], ["bad final"]])
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: next(validations))
    monkeypatch.setattr(mod, "run_validation_commands", lambda *_args: [])
    assert mod.main(["--date", "20260905", "--output", str(tmp_path / "final.json")]) == 1
