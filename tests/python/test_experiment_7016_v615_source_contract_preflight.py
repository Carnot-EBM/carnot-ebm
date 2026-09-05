"""Tests for REQ-REPORT-7016 and the V615 contract scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any, Callable

import pytest
import yaml

from carnot import experiment_7016_v615_source_contract_preflight as mod


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_ROWS = (
    (
        "exp7016-v615-source-contract-preflight",
        "V615 source delta and exact task-contract preflight",
        "results/experiment_7016_v615_source_contract_preflight.json",
    ),
    (
        "exp7017-task-linked-compute-receipts",
        "Task-linked phase, GPU, and runner receipt contract",
        "results/experiment_7017_task_linked_compute_receipts.json",
    ),
    (
        "exp7018-v615-sota-ingestion",
        "V615 recent-source ingestion and architecture map",
        "results/experiment_7018_v615_sota_ingestion.json",
    ),
    (
        "exp7019-arc-belief-stream-fixture",
        "Immutable ARC chronological belief-stream fixture",
        "results/experiment_7019_arc_belief_stream_fixture.json",
    ),
    (
        "exp7020-counterexample-belief-ledger",
        "Counterexample-updated ARC belief ledger",
        "results/experiment_7020_counterexample_belief_ledger.json",
    ),
    (
        "exp7021-prospective-belief-utility",
        "Prospective held-future belief utility comparison",
        "results/experiment_7021_prospective_belief_utility.json",
    ),
    (
        "exp7022-belief-ledger-cold-audit",
        "Fresh-process belief isolation, retention, and poison audit",
        "results/experiment_7022_belief_ledger_cold_audit.json",
    ),
    (
        "exp7023-belief-query-api",
        "Bounded belief-query API for the ARC policy",
        "results/experiment_7023_belief_query_api.json",
    ),
    (
        "exp7024-belief-aware-e3-selector",
        "Default-off belief-aware E3 selector wiring",
        "results/experiment_7024_belief_aware_e3_selector.json",
    ),
    (
        "exp7025-belief-shadow-live-trace",
        "Provenance-complete live belief shadow trace",
        "results/experiment_7025_belief_shadow_live_trace.json",
    ),
    (
        "exp7026-held-mechanic-belief-ab",
        "Held-mechanic live belief and simulation A/B",
        "results/experiment_7026_held_mechanic_belief_ab.json",
    ),
    (
        "exp7027-v615-capstone",
        "V615 independent evidence capstone and V616 handoff",
        "results/experiment_7027_v615_capstone.json",
    ),
)

GATES: dict[int, list[dict[str, Any]]] = {
    7020: [
        {
            "upstream": EXPECTED_ROWS[3][0],
            "artifact_field": "arc_belief_stream_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    7021: [
        {
            "upstream": EXPECTED_ROWS[4][0],
            "artifact_field": "belief_ledger_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    7022: [
        {
            "upstream": EXPECTED_ROWS[4][0],
            "artifact_field": "belief_ledger_ready_score",
            "op": "==",
            "value": 1,
        },
        {
            "upstream": EXPECTED_ROWS[5][0],
            "artifact_field": "belief_utility_comparison_complete_score",
            "op": "==",
            "value": 1,
        },
    ],
    7023: [
        {
            "upstream": EXPECTED_ROWS[6][0],
            "artifact_field": "belief_shadow_safe_score",
            "op": "==",
            "value": 1,
        }
    ],
    7024: [
        {
            "upstream": EXPECTED_ROWS[7][0],
            "artifact_field": "belief_query_api_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    7025: [
        {
            "upstream": EXPECTED_ROWS[1][0],
            "artifact_field": "task_compute_receipt_ready_score",
            "op": "==",
            "value": 1,
        },
        {
            "upstream": EXPECTED_ROWS[8][0],
            "artifact_field": "belief_selector_live_path_ready_score",
            "op": "==",
            "value": 1,
        },
    ],
    7026: [
        {
            "upstream": EXPECTED_ROWS[1][0],
            "artifact_field": "task_compute_receipt_ready_score",
            "op": "==",
            "value": 1,
        },
        {
            "upstream": EXPECTED_ROWS[9][0],
            "artifact_field": "belief_shadow_trace_ready_score",
            "op": "==",
            "value": 1,
        },
    ],
}

PRODUCED_FIELDS = {
    7016: "v615_task_contract_conforms_score",
    7017: "task_compute_receipt_ready_score",
    7018: "v615_sota_ingestion_complete_score",
    7019: "arc_belief_stream_ready_score",
    7020: "belief_ledger_ready_score",
    7021: "belief_utility_comparison_complete_score",
    7022: "belief_shadow_safe_score",
    7023: "belief_query_api_ready_score",
    7024: "belief_selector_live_path_ready_score",
    7025: "belief_shadow_trace_ready_score",
    7026: "held_mechanic_ab_complete_score",
    7027: "v616_handoff",
}

PRIORS = {
    7016: [("exp7009-v614-source-contract-preflight", "disqualified_v614_contract")],
    7020: [("exp6978-transactional-constraint-self-learning", "complete_null_prior")],
    7021: [
        ("exp6873-prospective-sealed-self-learning-audit", "complete_null_prior"),
        ("exp6978-transactional-constraint-self-learning", "complete_null_prior"),
    ],
    7022: [("exp6873-prospective-sealed-self-learning-audit", "complete_null_prior")],
    7026: [("exp7005-arc-live-envelope-audit", "complete_null_prior")],
}


def _prompt(number: int) -> str:
    """Build a complete prompt without using the production parser."""

    live = number in mod.LLM_TASK_NUMBERS
    substrate = f"live_llm_exp{number}" if live else f"deterministic_exp{number}_no_llm"
    model = f"MODEL_SPECS: [{mod.MANDATED_SOTA_GGUFS[0]}]; " if live else ""
    return (
        "TASK: Compare independent rows and controls.\n"
        "REQUIRED ARTIFACT FIELDS: field_principles; preconditions_checked; "
        f"inference_substrate ({substrate}); duration_s; {model}rows; "
        f"{PRODUCED_FIELDS[number]}; random_seed; reproducibility_checksum; "
        "gate_check_summary with failed check, expected value, and observed value; "
        "verifier_is_oracle (false); verdict_class "
        "(positive | circular_positive | null | blocked | disqualified | partial); "
        "honest_verdict.\n"
        "Run command: fixture only\n"
        "Do NOT push. Do NOT modify scripts/research_conductor.py.\n"
    )


def _roadmap() -> dict[str, Any]:
    """Build the YAML side from hard-coded facts, independent of Markdown."""

    tasks = []
    for order, (task_id, title, deliverable) in enumerate(EXPECTED_ROWS, start=1):
        number = 7015 + order
        priors = [
            {
                "experiment_id": experiment_id,
                "verdict": verdict,
                "addressed_by": "This task changes the tested mechanism and outcome.",
                "retire_if_same_verdict": True,
            }
            for experiment_id, verdict in PRIORS.get(number, [])
        ]
        tasks.append(
            {
                "id": task_id,
                "title": title,
                "deliverable": deliverable,
                "milestone": mod.MILESTONE,
                "requires_gpu": number in mod.LLM_TASK_NUMBERS,
                "per_unit_rows": True,
                "gated_on": deepcopy(GATES.get(number, [])),
                "prior_failures": priors,
                "prompt": _prompt(number),
            }
        )
    return {"milestone": mod.MILESTONE, "tasks": tasks}


def _copy_input(root: Path, relative: Path) -> None:
    """Copy one repository input into an isolated precondition fixture."""

    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / relative).read_bytes())


def _write_preconditions(root: Path, roadmap: dict[str, Any] | None = None) -> None:
    """Create every external input required by the preflight."""

    for relative in (
        mod.DESIGN_PATH,
        mod.V614_EVIDENCE_PATH,
        mod.EXCLUSION_PATH,
        mod.SPEC_PATH,
    ):
        _copy_input(root, relative)
    target = root / mod.DRAFT_ROADMAP_PATH
    target.write_text(yaml.safe_dump(roadmap or _roadmap()), encoding="utf-8")


def test_req_report_7016_spec_and_markdown_define_exact_contract() -> None:
    """REQ-REPORT-7016 and SCENARIO-REPORT-7016-PARITY define all 12 rows."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-7016") :]
    for scenario in ("PREFLIGHT", "PARITY", "GATES", "DISCIPLINE", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7016-{scenario}" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section

    parsed = mod.parse_markdown_contract((ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"))
    assert parsed["milestone"] == mod.MILESTONE
    assert [(row["id"], row["title"], row["deliverable"]) for row in parsed["tasks"]] == list(
        EXPECTED_ROWS
    )
    assert sum(len(row["gates"]) for row in parsed["tasks"]) == 10


def test_scenario_report_7016_conforming_fixture_checks_every_row() -> None:
    """SCENARIO-REPORT-7016-PARITY checks exact rows from independent parsers."""

    result = mod.evaluate_contract(
        (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"), _roadmap(), set()
    )

    assert result["passed"] is True
    assert result["observed_id_order"] == [row[0] for row in EXPECTED_ROWS]
    assert len(result["task_contract_rows"]) == 12
    assert len(result["gate_producer_rows"]) == 10
    for name in mod.CONTRACT_ROW_FIELDS:
        assert all(row["passed"] for row in result[name]), name


def _mutate_task_count(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"].pop()


def _mutate_id_order(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0], roadmap["tasks"][1] = roadmap["tasks"][1], roadmap["tasks"][0]


def _mutate_title(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["title"] += " changed"


def _mutate_deliverable(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["deliverable"] = "results/wrong.json"


def _mutate_gate(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][4]["gated_on"][0]["op"] = ">="


def _mutate_producer_spelling(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][3]["prompt"] = roadmap["tasks"][3]["prompt"].replace(
        "arc_belief_stream_ready_score", "arc_belief_stream_ready_scor"
    )


def _mutate_producer_missing(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][4]["gated_on"][0]["upstream"] = "exp9999-missing"


def _mutate_prior_field(roadmap: dict[str, Any], field: str) -> None:
    roadmap["tasks"][0]["prior_failures"][0][field] = (
        False if field == "retire_if_same_verdict" else ""
    )


def _mutate_model(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][9]["prompt"] = roadmap["tasks"][9]["prompt"].replace(
        mod.MANDATED_SOTA_GGUFS[0], "old/model-GGUF"
    )


def _mutate_artifact_field(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["prompt"] = roadmap["tasks"][0]["prompt"].replace(
        "verdict_class", "result_class"
    )


def _mutate_prompt_tail(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][0]["prompt"] += "Trailing text.\n"


def _mutate_comparison_rows(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][5]["per_unit_rows"] = False


def _mutate_capstone_gate(roadmap: dict[str, Any]) -> None:
    roadmap["tasks"][-1]["gated_on"] = deepcopy(GATES[7026][:1])


MUTATIONS: tuple[tuple[str, Callable[[dict[str, Any]], None]], ...] = (
    ("task_count", _mutate_task_count),
    ("id_order", _mutate_id_order),
    ("title", _mutate_title),
    ("deliverable", _mutate_deliverable),
    ("gate", _mutate_gate),
    ("producer_spelling", _mutate_producer_spelling),
    ("producer_missing", _mutate_producer_missing),
    ("prior_experiment_id", lambda value: _mutate_prior_field(value, "experiment_id")),
    ("prior_verdict", lambda value: _mutate_prior_field(value, "verdict")),
    ("prior_addressed_by", lambda value: _mutate_prior_field(value, "addressed_by")),
    ("prior_retire", lambda value: _mutate_prior_field(value, "retire_if_same_verdict")),
    ("model", _mutate_model),
    ("artifact_field", _mutate_artifact_field),
    ("prompt_tail", _mutate_prompt_tail),
    ("comparison_rows", _mutate_comparison_rows),
    ("capstone_gate", _mutate_capstone_gate),
)


@pytest.mark.parametrize(("name", "mutation"), MUTATIONS)
def test_scenario_report_7016_required_mutations_fail(
    name: str, mutation: Callable[[dict[str, Any]], None]
) -> None:
    """SCENARIO-REPORT-7016-DISCIPLINE rejects each contract mutation."""

    roadmap = _roadmap()
    mutation(roadmap)
    result = mod.evaluate_contract(
        (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"), roadmap, set()
    )

    assert result["passed"] is False, name


@pytest.mark.parametrize(
    ("mutation", "retired"),
    (
        (lambda roadmap: roadmap["tasks"][3].__setitem__("milestone", "2026.09.999"), set()),
        (lambda _roadmap: None, {"exp7019"}),
    ),
)
def test_scenario_report_7016_gate_producers_fail_closed(
    mutation: Callable[[dict[str, Any]], None], retired: set[str]
) -> None:
    """SCENARIO-REPORT-7016-GATES rejects cross-milestone and retired producers."""

    roadmap = _roadmap()
    mutation(roadmap)
    result = mod.evaluate_contract(
        (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"), roadmap, retired
    )

    assert result["passed"] is False
    assert any(not row["passed"] for row in result["gate_producer_rows"])


def test_scenario_report_7016_preflight_blocks_missing_yaml(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7016-PREFLIGHT emits the complete blocked shape."""

    artifact = mod.build_artifact(ROOT, "20260905", output_path=tmp_path / "blocked.json")

    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["gate_check_summary"]["failed_check"] == "v615_yaml_readable"
    assert artifact["v615_task_contract_conforms_score"] == 0
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == artifact["field_principles"].keys()
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_7016_artifact_derives_positive_and_disqualified(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7016-ARTIFACT derives both contract terminal states."""

    _write_preconditions(tmp_path)
    positive = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/positive.json"
    )
    assert positive["verdict_class"] == "positive"
    assert positive["v615_task_contract_conforms_score"] == 1
    assert mod.validate_artifact(positive) == []

    changed = _roadmap()
    changed["tasks"][0]["title"] += " changed"
    (tmp_path / mod.DRAFT_ROADMAP_PATH).write_text(yaml.safe_dump(changed), encoding="utf-8")
    disqualified = mod.build_artifact(
        tmp_path, "20260905", output_path=tmp_path / "results/disqualified.json"
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["observed_task_count"] == 12
    assert disqualified["title_parity_rows"][0]["observed"].endswith("changed")
    assert mod.validate_artifact(disqualified) == []


def test_scenario_report_7016_validator_rejects_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7016-ARTIFACT rejects forged score and checksum values."""

    _write_preconditions(tmp_path)
    artifact = mod.build_artifact(tmp_path, "20260905", output_path=tmp_path / "result.json")
    artifact["v615_task_contract_conforms_score"] = 0
    assert "v615_task_contract_conforms_score_invalid" in mod.validate_artifact(artifact)

    artifact["v615_task_contract_conforms_score"] = 1
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    artifact["honest_verdict"] = mod.BLOCKED_VERDICT
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    assert "honest_verdict_not_derived" in mod.validate_artifact(artifact)


def test_req_report_7016_cli_writes_current_blocked_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-7016 writes the terminal block when the planned YAML is absent."""

    output = tmp_path / "experiment_7016.json"
    assert mod.main(["--date", "20260905", "--root", str(ROOT), "--output", str(output)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert mod.validate_artifact(artifact) == []


def test_req_report_7016_parsers_reject_malformed_contracts(tmp_path: Path) -> None:
    """REQ-REPORT-7016 reports malformed independent sources without inference."""

    markdown = (ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="must be scalar"):
        mod._parse_scalar("{}")
    with pytest.raises(ValueError, match="milestone"):
        mod.parse_markdown_contract("no milestone")
    with pytest.raises(ValueError, match="section"):
        mod.parse_markdown_contract(f"**Milestone:** `{mod.MILESTONE}`")
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(markdown.replace("| 1 | `exp7016", "| 1 | extra | `exp7016", 1))
    with pytest.raises(ValueError, match="task row"):
        mod.parse_markdown_contract(markdown.replace("`exp7016-v615", "`invalid-v615", 1))
    empty = (
        f"**Milestone:** `{mod.MILESTONE}`\n"
        "## 7. Phases and Exact Task Contract\n"
        "| no task rows |\n"
        "### Phase I"
    )
    with pytest.raises(ValueError, match="task table"):
        mod.parse_markdown_contract(empty)
    with pytest.raises(ValueError, match="structured gate"):
        mod.parse_markdown_contract(
            markdown.replace("Exp7019 `arc_belief_stream_ready_score == 1`", "Exp7019 bad gate", 1)
        )
    assert mod._required_block("no declarations") == ""

    for document, message in (
        ([], "mapping"),
        ({}, "tasks list"),
        ({"tasks": [1]}, "must be a mapping"),
        ({"tasks": [{"gated_on": "bad"}]}, "malformed list"),
        ({"tasks": [{"gated_on": ["bad"]}]}, "malformed gate"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.parse_yaml_contract(document)
    path = tmp_path / "list.yaml"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping required"):
        mod.load_yaml(path)


def test_req_report_7016_manifest_and_precondition_defensive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7016 preserves missing inputs and manifest restorations."""

    assert mod._normal_exp_id("bad") is None
    assert mod.retired_experiment_ids([]) == set()
    manifest = {
        "not_retired": [{"experiment_id": 7016}],
        "retired_rows": [
            "bad",
            {"experiment_id": 7016},
            {
                "experiment_ids": ["exp7017-task", "bad"],
                "un_retired_experiment_ids": [7016],
            },
        ],
    }
    assert mod.retired_experiment_ids(manifest) == {"exp7017"}

    rows, roadmap, exclusion = mod._preconditions(tmp_path, tmp_path / "result.json")
    assert roadmap is None and exclusion is None
    assert {row["check"] for row in rows if not row["available"]} >= {
        "v615_markdown_readable",
        "v615_yaml_readable",
        "v614_evidence_readable",
        "exclusion_manifest_readable",
        "reporting_spec_readable",
    }

    monkeypatch.setattr(
        mod.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("read only")),
    )
    assert mod._artifact_path_writable(tmp_path / "blocked.json")[0] is False


def test_req_report_7016_parse_and_command_blocks_are_terminal(tmp_path: Path) -> None:
    """REQ-REPORT-7016 distinguishes malformed contracts from tool blocks."""

    _write_preconditions(tmp_path)
    (tmp_path / mod.DESIGN_PATH).write_text("readable but malformed\n", encoding="utf-8")
    malformed = mod.build_artifact(tmp_path, "20260905", output_path=tmp_path / "malformed.json")
    assert malformed["verdict_class"] == "disqualified"
    assert malformed["gate_check_summary"]["failed_check"] == "contract_parse"
    assert mod.validate_artifact(malformed) == []

    _copy_input(tmp_path, mod.DESIGN_PATH)
    failed_receipt = {
        "name": "focused_tests",
        "outcome": "failed",
        "terminal": True,
        "passed": False,
    }
    blocked = mod.build_artifact(
        tmp_path,
        "20260905",
        output_path=tmp_path / "command.json",
        command_rows=[failed_receipt],
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "focused_tests"
    assert mod.validate_artifact(blocked) == []


def test_req_report_7016_validator_rejects_each_common_forgery(tmp_path: Path) -> None:
    """REQ-REPORT-7016 validates common fields before trusting row-derived state."""

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260905", output_path=tmp_path / "good.json")
    cases = (
        ("field_principles", {}, "field_principles_invalid"),
        ("inference_substrate", "wrong", "inference_substrate_invalid"),
        ("verifier_is_oracle", True, "verifier_is_oracle_invalid"),
        ("expected_task_count", 13, "expected_task_count_invalid"),
        ("expected_id_order", [], "expected_id_order_invalid"),
        ("random_seed", 0, "random_seed_invalid"),
        ("duration_s", -1, "duration_s_invalid"),
        ("observed_task_count", 0, "observed_task_count_invalid"),
        ("observed_id_order", [], "observed_id_order_invalid"),
        ("verdict_class", "blocked", "verdict_class_not_derived"),
        ("gate_check_summary", [], "gate_check_summary_invalid"),
    )
    for field, value, expected_error in cases:
        changed = deepcopy(good)
        changed[field] = value
        changed["reproducibility_checksum"] = mod.reproducibility_checksum(changed)
        assert expected_error in mod.validate_artifact(changed), field

    missing = dict(good)
    missing.pop("rows")
    assert mod.validate_artifact(missing) == ["missing_required_field:rows"]

    row_forgery = deepcopy(good)
    row_forgery["artifact_field_rows"][0]["passed"] = False
    row_forgery["reproducibility_checksum"] = mod.reproducibility_checksum(row_forgery)
    assert "v615_task_contract_conforms_score_invalid" in mod.validate_artifact(row_forgery)

    command_forgery = deepcopy(good)
    command_forgery["command_receipt_rows"] = [{}]
    command_forgery["reproducibility_checksum"] = mod.reproducibility_checksum(command_forgery)
    assert "verdict_class_not_derived" in mod.validate_artifact(command_forgery)

    summary_forgery = deepcopy(good)
    summary_forgery["gate_check_summary"]["passed"] = False
    summary_forgery["reproducibility_checksum"] = mod.reproducibility_checksum(summary_forgery)
    assert "gate_check_summary_not_derived" in mod.validate_artifact(summary_forgery)


def test_req_report_7016_command_receipts_cover_all_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7016 records pass, failure, timeout, and unavailable tools."""

    passed = mod._run_command(tmp_path, "pass", ["/bin/sh", "-c", "printf clean"])
    failed = mod._run_command(tmp_path, "fail", ["/bin/sh", "-c", "exit 3"])
    assert passed["passed"] is True and passed["stdout"] == "clean"
    assert failed["passed"] is False and failed["exit_code"] == 3

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("slow", 1, output="partial", stderr="late")
        ),
    )
    assert mod._run_command(tmp_path, "slow", ["slow"])["outcome"] == "timeout"
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    assert mod._run_command(tmp_path, "missing", ["missing"])["outcome"] == "tool_error"

    names: list[str] = []

    def record(_root: Path, name: str, _argv: list[str]) -> dict[str, Any]:
        names.append(name)
        return {"name": name, "terminal": True, "passed": True}

    monkeypatch.setattr(mod, "_run_command", record)
    rows = mod.run_validation_commands(ROOT, tmp_path / "artifact.json")
    assert len(rows) == 9
    assert names == [row["name"] for row in rows]


def test_req_report_7016_cli_validation_and_positive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7016 exposes deterministic validation and generation exits."""

    assert mod._date_argument("20260905") == "20260905"
    with pytest.raises(Exception, match="YYYYMMDD"):
        mod._date_argument("2026-09-05")
    with pytest.raises(SystemExit):
        mod.main([])

    unreadable = tmp_path / "unreadable.json"
    assert mod.main(["--validate-artifact", str(unreadable)]) == 1
    list_path = tmp_path / "list.json"
    list_path.write_text("[]\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(list_path)]) == 1

    _write_preconditions(tmp_path)
    good = mod.build_artifact(tmp_path, "20260905", output_path=tmp_path / "good.json")
    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps(good), encoding="utf-8")
    assert mod.main(["--validate-artifact", str(good_path)]) == 0
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate-artifact", str(bad_path)]) == 1

    monkeypatch.setattr(mod, "run_validation_commands", lambda *_args: [])
    output = tmp_path / "nested/positive.json"
    assert mod.main(
        ["--date", "20260905", "--root", str(tmp_path), "--output", str(output)]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["verdict_class"] == "positive"

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced failure"])
    assert mod.main(
        ["--date", "20260905", "--root", str(ROOT), "--output", str(tmp_path / "bad-final.json")]
    ) == 1
