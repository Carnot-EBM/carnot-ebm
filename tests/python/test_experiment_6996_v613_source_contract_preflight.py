"""Focused tests for REQ-REPORT-6996 and its source-contract scenarios."""

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

from carnot import experiment_6996_v613_source_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]


def _design_text() -> str:
    """Read the Markdown source without using the experiment parser."""

    return (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")


def _roadmap() -> dict[str, Any]:
    """Read the activated YAML source without using the experiment parser."""

    value = yaml.safe_load((ROOT / mod.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _source_rows(*, new_source: bool = False) -> list[dict[str, Any]]:
    """Make one terminal access receipt for every required source route."""

    rows = []
    for index, route in enumerate(mod.REQUIRED_SOURCE_ROUTES):
        category = "semantic" if route.startswith("semantic_") else "primary"
        if route.startswith("huggingface_"):
            category = "secondary"
        rows.append(
            {
                "route": route,
                "category": category,
                "url": f"https://example.test/{route}",
                "query": route.replace("_", " "),
                "accessed_on": "2026-09-04",
                "outcome": "ok",
                "http_status": 200,
                "terminal": True,
                "published_or_changed_at": (
                    "2026-09-04T23:00:00Z" if new_source and index == 0 else None
                ),
                "post_marker_change_proven": new_source and index == 0,
                "relevant": new_source and index == 0,
                "finding": "new relevant primary source" if new_source and index == 0 else "no change",
                "response_sha256": "sha256:" + hashlib.sha256(route.encode()).hexdigest(),
            }
        )
    return rows


def _passing_receipts() -> list[dict[str, Any]]:
    """Make terminal passing rows for each required validation command."""

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


def _evaluate(roadmap: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one V613 fixture against the independent Markdown contract."""

    return mod.evaluate_contract(
        _design_text(), roadmap or _roadmap(), retired_ids=set()
    )


def test_req_report_6996_spec_precedes_implementation() -> None:
    """REQ-REPORT-6996 owns each required source and contract scenario."""

    text = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6996") :]
    for name in ("PREFLIGHT", "PARITY", "GATES", "DISCIPLINE", "SOURCES", "NOCHANGE", "ARTIFACT"):
        assert f"SCENARIO-REPORT-6996-{name}" in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_6996_parity_accepts_all_thirteen_rows() -> None:
    """SCENARIO-REPORT-6996-PARITY matches the active contract exactly."""

    result = _evaluate()

    assert result["passed"] is True
    assert [row["number"] for row in result["markdown_task_rows"]] == list(
        mod.EXPECTED_NUMBERS
    )
    assert [row["number"] for row in result["yaml_task_rows"]] == list(
        mod.EXPECTED_NUMBERS
    )
    assert len(result["task_contract_rows"]) == 13
    assert len(result["gate_producer_rows"]) == 11
    for key in (
        "task_contract_rows",
        "title_parity_rows",
        "deliverable_parity_rows",
        "gate_contract_rows",
        "gate_producer_rows",
        "prior_failure_rows",
        "retired_id_rows",
        "model_compliance_rows",
        "substrate_name_rows",
        "artifact_field_rows",
        "prompt_tail_rows",
    ):
        assert all(row["passed"] for row in result[key]), key

    exp6998 = next(
        row for row in result["model_compliance_rows"] if row["number"] == 6998
    )
    assert exp6998["model_bearing"] is True
    assert exp6998["required_three_models_present"] is True


@pytest.mark.parametrize("mutation", mod.REQUIRED_MUTATIONS)
def test_scenario_report_6996_required_mutations_fail(mutation: str) -> None:
    """SCENARIO-REPORT-6996-DISCIPLINE detects every requested mutation."""

    roadmap = _roadmap()
    mod.apply_mutation(roadmap, mutation)

    assert _evaluate(roadmap)["passed"] is False


def test_scenario_report_6996_mutation_rows_prove_the_attack_changed_input() -> None:
    """SCENARIO-REPORT-6996-DISCIPLINE records effective mutation attacks."""

    rows = mod.build_mutation_rows(_design_text(), _roadmap(), set())

    assert [row["mutation"] for row in rows] == list(mod.REQUIRED_MUTATIONS)
    assert all(row["input_changed"] for row in rows)
    assert all(row["failed_as_expected"] for row in rows)


def test_scenario_report_6996_gate_producers_fail_closed() -> None:
    """SCENARIO-REPORT-6996-GATES rejects missing, later, cross-milestone, and misspelled producers."""

    missing = _roadmap()
    missing["tasks"][2]["gated_on"][0]["upstream"] = "exp9999-missing"
    assert any(not row["producer_present"] for row in _evaluate(missing)["gate_producer_rows"])

    later = _roadmap()
    later["tasks"][2]["gated_on"][0]["upstream"] = later["tasks"][3]["id"]
    assert any(
        not row["producer_precedes_consumer"]
        for row in _evaluate(later)["gate_producer_rows"]
    )

    cross = _roadmap()
    cross["tasks"][1]["milestone"] = "2026.09.612"
    assert any(
        not row["same_milestone"] for row in _evaluate(cross)["gate_producer_rows"]
    )

    typo = _roadmap()
    typo["tasks"][2]["gated_on"][0]["artifact_field"] = "blinded_learner_view_ready_scor"
    assert any(not row["field_declared"] for row in _evaluate(typo)["gate_producer_rows"])

    nonbare = _roadmap()
    nonbare["tasks"][2]["gated_on"][0]["artifact_field"] = "result.ready_score"
    assert any(not row["bare_field"] for row in _evaluate(nonbare)["gate_producer_rows"])

    upstream = _roadmap()["tasks"][2]["gated_on"][0]["upstream"]
    assert any(
        row["upstream_retired"]
        for row in mod.evaluate_contract(_design_text(), _roadmap(), {upstream})[
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
def test_scenario_report_6996_prior_failure_schema_is_exact(
    field: str, value: object
) -> None:
    """SCENARIO-REPORT-6996-DISCIPLINE rejects each malformed prior-failure field."""

    roadmap = _roadmap()
    roadmap["tasks"][0]["prior_failures"][0][field] = value

    assert any(not row["passed"] for row in _evaluate(roadmap)["prior_failure_rows"])


def test_scenario_report_6996_model_and_smoke_rules_fail_closed() -> None:
    """SCENARIO-REPORT-6996-DISCIPLINE requires all Exp6998 GGUFs and smoke-only legacy models."""

    missing_model = _roadmap()
    missing_model["tasks"][2]["prompt"] = missing_model["tasks"][2]["prompt"].replace(
        mod.REQUIRED_EXP6998_MODELS[0], "unsloth/not-current-GGUF"
    )
    row = _evaluate(missing_model)["model_compliance_rows"][2]
    assert row["required_three_models_present"] is False
    assert row["passed"] is False

    missing_specs = _roadmap()
    missing_specs["tasks"][2]["prompt"] = missing_specs["tasks"][2]["prompt"].replace(
        "MODEL_SPECS", "MODEL_PLAN"
    )
    assert _evaluate(missing_specs)["model_compliance_rows"][2]["passed"] is False

    legacy_headline = _roadmap()
    legacy_headline["tasks"][2]["prompt"] += "\nUse Qwen3.5-0.8B as a headline model."
    assert _evaluate(legacy_headline)["model_compliance_rows"][2]["passed"] is False


def test_scenario_report_6996_verdict_rows_substrates_and_tail_fail_closed() -> None:
    """SCENARIO-REPORT-6996-DISCIPLINE checks verdict, rows, no-LLM suffixes, and prompt tails."""

    no_verdict = _roadmap()
    no_verdict["tasks"][0]["prompt"] = no_verdict["tasks"][0]["prompt"].replace(
        "; verdict_class\n", "; result_class\n", 1
    )
    assert _evaluate(no_verdict)["artifact_field_rows"][0]["passed"] is False

    no_diagnostics = _roadmap()
    no_diagnostics["tasks"][0]["prompt"] = no_diagnostics["tasks"][0]["prompt"].replace(
        "gate_check_summary", "gate_summary"
    )
    assert _evaluate(no_diagnostics)["artifact_field_rows"][0]["passed"] is False

    no_rows = _roadmap()
    no_rows["tasks"][2]["per_unit_rows"] = False
    assert _evaluate(no_rows)["artifact_field_rows"][2]["passed"] is False

    bad_substrate = _roadmap()
    bad_substrate["tasks"][0]["prompt"] = bad_substrate["tasks"][0]["prompt"].replace(
        mod.INFERENCE_SUBSTRATE, "deterministic_source_and_contract_audit"
    )
    assert _evaluate(bad_substrate)["substrate_name_rows"][0]["passed"] is False

    bad_tail = _roadmap()
    bad_tail["tasks"][0]["prompt"] += "\nTrailing sentence."
    assert _evaluate(bad_tail)["prompt_tail_rows"][0]["passed"] is False


def test_req_report_6996_parsers_and_roadmap_resolution_fail_closed(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6996 parses only well-formed independent contracts and resolves activated YAML."""

    assert len(mod.parse_design(_design_text())["tasks"]) == 13
    assert len(mod.parse_roadmap(_roadmap())["tasks"]) == 13
    with pytest.raises(ValueError, match="milestone"):
        mod.parse_design("no milestone")
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design(f"**Milestone:** `{mod.MILESTONE}`")
    with pytest.raises(ValueError, match="roadmap mapping"):
        mod.parse_roadmap([])
    with pytest.raises(ValueError, match="tasks list"):
        mod.parse_roadmap({})
    with pytest.raises(ValueError, match="task at order"):
        mod.parse_roadmap({"tasks": ["bad"]})
    with pytest.raises(ValueError, match="task lists"):
        mod.parse_roadmap({"tasks": [{"gated_on": "bad"}]})
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_design(_design_text().replace("| None |", "| malformed |", 1))
    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_scalar("{}")
    malformed_row = _design_text().replace(
        "| 1 | `exp6996-v613-source-contract-preflight` |",
        "| 1 | extra | `exp6996-v613-source-contract-preflight` |",
    )
    with pytest.raises(ValueError, match="malformed design task row"):
        mod.parse_design(malformed_row)

    assert mod.resolve_v613_roadmap(ROOT) == ROOT / mod.ACTIVE_ROADMAP_PATH
    active = tmp_path / mod.ACTIVE_ROADMAP_PATH
    active.write_text(yaml.safe_dump(_roadmap()), encoding="utf-8")
    assert mod.resolve_v613_roadmap(tmp_path) == active
    draft = tmp_path / mod.DRAFT_ROADMAP_PATH
    draft.write_text(yaml.safe_dump(_roadmap()), encoding="utf-8")
    assert mod.resolve_v613_roadmap(tmp_path) == draft
    draft.write_text("milestone: wrong\ntasks: []\n", encoding="utf-8")
    assert mod.resolve_v613_roadmap(tmp_path) == active
    active.unlink()
    with pytest.raises(FileNotFoundError, match="V613 YAML"):
        mod.resolve_v613_roadmap(tmp_path)

    malformed_yaml = tmp_path / "malformed.yaml"
    malformed_yaml.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        mod.load_yaml(malformed_yaml)

    malformed_design = _design_text().replace(
        "`exp6996-v613-source-contract-preflight`",
        "`not-an-experiment`",
        1,
    )
    with pytest.raises(ValueError, match="malformed design task row"):
        mod.parse_design(malformed_design)
    empty_table = _design_text().replace(
        "| 1 | `exp6996-v613-source-contract-preflight` |",
        "| one | `exp6996-v613-source-contract-preflight` |",
        1,
    )
    empty_table = "\n".join(
        line for line in empty_table.splitlines() if not line.startswith("| ")
    )
    with pytest.raises(ValueError, match="task table"):
        mod.parse_design(empty_table)
    assert mod._required_block("no required marker") == ""

    normalized = mod.parse_roadmap(
        {"tasks": [{"gated_on": None, "prior_failures": None}]}
    )
    assert normalized["tasks"][0]["gates"] == []
    with pytest.raises(ValueError, match="task lists"):
        mod.parse_roadmap({"tasks": [{"gated_on": ["bad"]}]})

    active.write_text(yaml.safe_dump(_roadmap()), encoding="utf-8")
    draft.write_text("[\n", encoding="utf-8")
    assert mod.resolve_v613_roadmap(tmp_path) == active


def test_req_report_6996_manifest_parser_handles_all_retirement_shapes() -> None:
    """REQ-REPORT-6996 normalizes retired IDs and explicit restorations."""

    manifest = {
        "retired": [{"experiment_id": 1}],
        "retired_experiments": [{"experiment_id": "exp2-old"}],
        "retired_extras": [
            {"experiment_ids": ["exp3-old", 4], "un_retired_experiment_ids": ["exp3-old"]}
        ],
    }
    assert mod.retired_experiment_ids(manifest) == {"exp1", "exp2", "exp4"}
    assert mod.retired_experiment_ids([]) == set()
    assert mod.retired_experiment_ids(
        {"retired": "bad", "retired_experiments": ["bad"], "retired_extras": [{}]}
    ) == set()


def test_scenario_report_6996_sources_are_terminal_and_no_change_is_explicit() -> None:
    """SCENARIO-REPORT-6996-SOURCES and NOCHANGE keep route classes and history separate."""

    evidence = mod.build_source_evidence(_source_rows())

    assert mod.source_delta_complete_score(evidence["source_query_rows"]) == 1
    assert len(evidence["primary_source_rows"]) == 6
    assert len(evidence["secondary_source_rows"]) == 2
    assert len(evidence["semantic_scholar_rows"]) == 2
    assert all(row["terminal"] for row in evidence["post_marker_delta_rows"])
    assert evidence["reference_append_rows"] == [
        {
            "action": "no_change",
            "appended": False,
            "reason": "No primary or first-party source proved a relevant change after the V613 planner marker.",
            "terminal": True,
        }
    ]

    incomplete = _source_rows()[:-1]
    assert mod.source_delta_complete_score(incomplete) == 0
    new_evidence = mod.build_source_evidence(_source_rows(new_source=True))
    assert new_evidence["reference_append_rows"][0]["action"] == "append_required"
    assert new_evidence["reference_append_rows"][0]["appended"] is False


def test_req_report_6996_source_fetcher_records_success_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6996 preserves HTTP, network, and tool access outcomes as terminal rows."""

    class Response:
        status = 200
        headers = {"Last-Modified": "Fri, 04 Sep 2026 20:00:00 GMT"}

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return b"primary response"

    monkeypatch.setattr(mod, "urlopen", lambda *_args, **_kwargs: Response())
    success = mod._fetch_url("https://example.test", timeout_s=1.0)
    assert success["outcome"] == "ok"
    assert success["http_status"] == 200
    assert success["last_modified"]

    http_error = HTTPError("https://example.test", 429, "limited", {}, None)
    monkeypatch.setattr(
        mod, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(http_error)
    )
    assert mod._fetch_url("https://example.test")["outcome"] == "http_error"

    monkeypatch.setattr(
        mod,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    assert mod._fetch_url("https://example.test")["outcome"] == "network_error"

    monkeypatch.setattr(
        mod, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("broken"))
    )
    assert mod._fetch_url("https://example.test")["outcome"] == "tool_error"

    rows = mod.collect_source_query_rows(
        "20260904",
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
    assert all(row["accessed_on"] == "2026-09-04" for row in rows)


def _write_preconditions(root: Path) -> None:
    """Create isolated readable inputs for artifact precondition tests."""

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
    (root / mod.ACTIVE_ROADMAP_PATH).write_text(
        yaml.safe_dump(_roadmap(), sort_keys=False), encoding="utf-8"
    )


def test_scenario_report_6996_preflight_blocks_missing_and_malformed_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6996-PREFLIGHT returns schema-complete blocked artifacts."""

    missing = mod.build_artifact(
        tmp_path,
        "20260904",
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
    (tmp_path / mod.EXCLUSION_PATH).write_text("[\n", encoding="utf-8")
    malformed = mod.build_artifact(
        tmp_path,
        "20260904",
        output_path=tmp_path / "artifact.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
    )
    assert malformed["verdict_class"] == "blocked"
    assert any(row.get("parse_error") for row in malformed["preconditions_checked"])

    _write_preconditions(tmp_path)
    network_blocked = mod.build_artifact(
        tmp_path,
        "20260904",
        output_path=tmp_path / "artifact.json",
        network_available=False,
    )
    assert network_blocked["gate_check_summary"]["observed_value"] == "network_unavailable"

    monkeypatch.setattr(mod, "_artifact_path_writable", lambda _path: False)
    write_blocked = mod.build_artifact(
        tmp_path,
        "20260904",
        output_path=tmp_path / "artifact.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
    )
    assert any(
        row["resource"] == "writable_artifact_path" and not row["available"]
        for row in write_blocked["preconditions_checked"]
    )


def test_req_report_6996_preflight_covers_probe_and_parse_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6996 reports local write, capstone, network, and contract failures."""

    _write_preconditions(tmp_path)
    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod._artifact_path_writable(tmp_path / "artifact.json") is False
    monkeypatch.undo()

    _write_preconditions(tmp_path)
    (tmp_path / mod.CAPSTONE_PATH).write_text("{", encoding="utf-8")
    bad_capstone = mod.build_artifact(
        tmp_path,
        "20260904",
        output_path=tmp_path / "artifact.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
    )
    assert bad_capstone["verdict_class"] == "blocked"

    _write_preconditions(tmp_path)
    monkeypatch.setattr(
        mod,
        "_fetch_url",
        lambda *_args, **_kwargs: {"outcome": "http_error"},
    )
    probed = mod.build_artifact(
        tmp_path,
        "20260904",
        output_path=tmp_path / "artifact.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
    )
    assert probed["preconditions_checked"][-2]["available"] is True

    parse_blocked = mod.build_artifact(
        tmp_path,
        "20260904",
        output_path=tmp_path / "artifact.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
        roadmap_override=[],
    )
    assert parse_blocked["gate_check_summary"]["failed_check"] == "contract_parse"


def test_scenario_report_6996_positive_and_disqualified_artifacts_recompute(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6996-ARTIFACT derives positive and disqualified states from rows."""

    positive = mod.build_artifact(
        ROOT,
        "20260904",
        output_path=tmp_path / "positive.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
    )
    assert positive["expected_task_count"] == 13
    assert positive["observed_task_count"] == 13
    assert positive["expected_id_order"] == list(mod.EXPECTED_NUMBERS)
    assert positive["observed_id_order"] == list(mod.EXPECTED_NUMBERS)
    assert positive["v613_source_delta_complete_score"] == 1
    assert positive["v613_task_contract_conforms_score"] == 1
    assert positive["verifier_is_oracle"] is False
    assert positive["verdict_class"] == "positive"
    assert positive["honest_verdict"] == mod.CONFORMS_VERDICT
    assert mod.validate_artifact(positive) == []

    changed = _roadmap()
    changed["tasks"][0]["title"] += " changed"
    disqualified = mod.build_artifact(
        ROOT,
        "20260904",
        output_path=tmp_path / "disqualified.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
        roadmap_override=changed,
    )
    assert disqualified["v613_source_delta_complete_score"] == 1
    assert disqualified["v613_task_contract_conforms_score"] == 0
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["honest_verdict"] == mod.DISQUALIFIED_VERDICT
    assert mod.validate_artifact(disqualified) == []

    nonterminal = deepcopy(_passing_receipts())
    nonterminal[0]["terminal"] = False
    nonterminal_artifact = mod.build_artifact(
        ROOT,
        "20260904",
        output_path=tmp_path / "nonterminal.json",
        source_rows=_source_rows(),
        command_rows=nonterminal,
        network_available=True,
    )
    assert nonterminal_artifact["gate_check_summary"]["failed_check"] == "validation_tools"

    incomplete_source = mod.build_artifact(
        ROOT,
        "20260904",
        output_path=tmp_path / "incomplete.json",
        source_rows=_source_rows()[:-1],
        command_rows=_passing_receipts(),
        network_available=True,
    )
    assert incomplete_source["gate_check_summary"]["failed_check"] == "source_routes"


def test_scenario_report_6996_validator_rejects_forged_artifacts(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6996-ARTIFACT rejects missing fields, scores, classes, prefixes, and hashes."""

    artifact = mod.build_artifact(
        ROOT,
        "20260904",
        output_path=tmp_path / "artifact.json",
        source_rows=_source_rows(),
        command_rows=_passing_receipts(),
        network_available=True,
    )
    mutations = {
        "field_principles": {},
        "inference_substrate": "wrong",
        "v613_source_delta_complete_score": 0,
        "v613_task_contract_conforms_score": 0,
        "verifier_is_oracle": True,
        "verdict_class": "partial",
        "honest_verdict": "partial_wrong",
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert mod.validate_artifact(changed), field

    missing = deepcopy(artifact)
    missing.pop("rows")
    missing["reproducibility_checksum"] = mod.reproducibility_checksum(missing)
    assert "missing_required_fields:rows" in mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(bad_checksum)

    bad_summary = deepcopy(artifact)
    bad_summary["gate_check_summary"] = []
    bad_summary["reproducibility_checksum"] = mod.reproducibility_checksum(bad_summary)
    assert "gate_check_summary_invalid" in mod.validate_artifact(bad_summary)

    unknown = deepcopy(artifact)
    unknown["verdict_class"] = "unknown"
    unknown["reproducibility_checksum"] = mod.reproducibility_checksum(unknown)
    assert "verdict_class_invalid" in mod.validate_artifact(unknown)


def test_req_report_6996_command_receipts_separate_defects_from_tool_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6996 records completed lint findings separately from unavailable tools."""

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(["check"], 1, "finding", ""),
    )
    defect = mod._run_command(ROOT, "check", [sys.executable, "-V"])
    assert defect["outcome"] == "contract_defect"
    assert defect["terminal"] is True

    def timeout(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(["check"], 1, output="partial", stderr="late")

    monkeypatch.setattr(mod.subprocess, "run", timeout)
    timed = mod._run_command(ROOT, "check", [sys.executable, "-V"])
    assert timed["outcome"] == "tool_failure"
    assert timed["terminal"] is False

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    failed = mod._run_command(ROOT, "check", ["missing"])
    assert failed["outcome"] == "tool_failure"
    assert "OSError" in failed["stderr"]

    calls: list[str] = []
    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda _root, name, _argv: calls.append(name)
        or {
            "name": name,
            "command": name,
            "exit_code": 0,
            "stdout": "clean",
            "stderr": "",
            "outcome": "pass",
            "terminal": True,
            "passed": True,
        },
    )
    receipts = mod.run_validation_commands(ROOT, tmp_path / "artifact.json")
    assert calls == list(mod.EXTERNAL_COMMAND_NAMES)
    assert [row["name"] for row in receipts] == list(mod.EXTERNAL_COMMAND_NAMES)

    monkeypatch.setattr(
        mod,
        "_run_command",
        lambda _root, name, _argv: {
            "name": name,
            "command": name,
            "exit_code": 1 if name == "adversarial_verification" else 0,
            "stdout": "[WARN] reviewed no-LLM substrate",
            "stderr": "",
            "outcome": "contract_defect" if name == "adversarial_verification" else "pass",
            "terminal": True,
            "passed": name != "adversarial_verification",
        },
    )
    warning_receipts = mod.run_validation_commands(ROOT, tmp_path / "artifact.json")
    warning = next(row for row in warning_receipts if row["name"] == "adversarial_verification")
    assert warning["outcome"] == "warning"
    assert warning["passed"] is True


def test_req_report_6996_date_main_and_atomic_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6996 validates dates and writes only the requested complete artifact."""

    assert mod._date_argument("20260904") == "20260904"
    assert mod.main(["--date", "bad"]) == 2
    target = tmp_path / "nested" / "artifact.json"
    mod._write_json_atomic(target, {"terminal": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"terminal": True}
    assert not target.with_suffix(".json.tmp").exists()

    artifact = {
        "verdict_class": "blocked",
        "honest_verdict": mod.BLOCKED_VERDICT,
    }
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: artifact)
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    written: list[Path] = []
    monkeypatch.setattr(mod, "_write_json_atomic", lambda path, _value: written.append(path))
    assert mod.main(["--date", "20260904", "--output", str(target)]) == 0
    assert mod.main(["--date", "20260904", "--output", "relative.json"]) == 0
    assert written == [target, mod.REPO_ROOT / "relative.json"]

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["invalid"])
    assert mod.main(["--date", "20260904", "--output", str(target)]) == 1


def test_req_report_6996_main_runs_terminal_validation_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6996 runs command receipts before it freezes a complete CLI artifact."""

    complete = {field: [] for field in mod.REQUIRED_ARTIFACT_FIELDS}
    complete["source_query_rows"] = _source_rows()
    complete["verdict_class"] = "positive"
    complete["honest_verdict"] = mod.CONFORMS_VERDICT
    builds: list[dict[str, Any]] = []

    def build(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        builds.append(kwargs)
        return deepcopy(complete)

    monkeypatch.setattr(mod, "build_artifact", build)
    monkeypatch.setattr(mod, "run_validation_commands", lambda *_args: [])
    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: [])
    writes: list[Path] = []
    monkeypatch.setattr(mod, "_write_json_atomic", lambda path, _value: writes.append(path))
    target = tmp_path / "complete.json"
    assert mod.main(["--date", "20260904", "--output", str(target)]) == 0
    assert len(builds) == 2
    assert writes == [target, target]

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["bad final"])
    assert mod.main(["--date", "20260904", "--output", str(target)]) == 1


def test_req_report_6996_unknown_mutation_is_explicit() -> None:
    """REQ-REPORT-6996 refuses an unknown adversarial mutation name."""

    with pytest.raises(ValueError, match="unknown mutation"):
        mod.apply_mutation(_roadmap(), "unknown")
