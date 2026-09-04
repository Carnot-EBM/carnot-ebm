"""Focused tests for REQ-REPORT-6995 and its V612 capstone scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6995_v612_capstone as mod


REPO_ROOT = Path(__file__).resolve().parents[2]


def _payload(
    verdict_class: str = "positive",
    honest_verdict: str = "complete_positive_fixture",
    **updates: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "complete",
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "rows": [{"terminal": True}],
    }
    payload.update(updates)
    return payload


@pytest.fixture(scope="module")
def actual_artifact() -> dict[str, object]:
    """Build once because the feature-bank source artifact is intentionally large."""

    return mod.build_artifact(REPO_ROOT, "20260904", run_commands=False)


def test_req_report_6995_spec_precedes_implementation() -> None:
    """REQ-REPORT-6995 names the complete schema and each required scenario."""

    text = (REPO_ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("### REQ-REPORT-6995", 1)[1]
    assert {
        "SCENARIO-REPORT-6995-CONTRACT",
        "SCENARIO-REPORT-6995-ARTIFACTS",
        "SCENARIO-REPORT-6995-ROWS",
        "SCENARIO-REPORT-6995-VERDICTS",
        "SCENARIO-REPORT-6995-BLOCKED",
        "SCENARIO-REPORT-6995-PUBLICATION",
        "SCENARIO-REPORT-6995-HANDOFF",
        "SCENARIO-REPORT-6995-ARTIFACT",
    } <= set(mod.spec_anchors(section))
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section


def test_scenario_report_6995_contract_exact_parity() -> None:
    """SCENARIO-REPORT-6995-CONTRACT compares both primary contracts independently."""

    document = mod.parse_document_contract(
        (REPO_ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    )
    roadmap = mod.load_yaml(REPO_ROOT / mod.ROADMAP_PATH)
    yaml_rows = mod.parse_yaml_contract(roadmap)
    task_rows = mod.build_task_contract_rows(document, yaml_rows)
    gate_rows = mod.build_gate_contract_rows(document, yaml_rows)

    assert [row["task_id"] for row in document] == mod.EXPECTED_TASK_IDS
    assert [row["task_id"] for row in yaml_rows] == mod.EXPECTED_TASK_IDS
    assert len(task_rows) == 12
    assert len(gate_rows) == 16
    assert all(row["passed"] for row in task_rows + gate_rows)
    assert mod.contract_conforms(document, yaml_rows, task_rows, gate_rows)

    changed = deepcopy(roadmap)
    changed["tasks"][0]["title"] = "Wrong title"
    changed["tasks"][1]["deliverable"] = "results/wrong.json"
    changed["tasks"][2]["gated_on"][0]["artifact_field"] = "wrong_field"
    changed["tasks"].append(changed["tasks"].pop(3))
    bad_yaml = mod.parse_yaml_contract(changed)
    bad_tasks = mod.build_task_contract_rows(document, bad_yaml)
    bad_gates = mod.build_gate_contract_rows(document, bad_yaml)
    assert not mod.contract_conforms(document, bad_yaml, bad_tasks, bad_gates)
    assert {row["check"] for row in bad_tasks if not row["passed"]} >= {
        "title",
        "deliverable",
        "order",
    }
    assert any(not row["passed"] for row in bad_gates)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, "blocked"),
        (
            {
                "status": "blocked",
                "honest_verdict": "blocked_gate_check_failed",
                "blocked_at_layer": "conductor_pre_gate",
                "gate_check_summary": "gate-unsat",
                "failed_field": "ready_score",
                "failed_expected": 1,
                "failed_observed": 0,
            },
            "blocked",
        ),
        (_payload(), "positive"),
        (_payload(verifier_is_oracle=True), "positive"),
        (_payload("circular_positive", "complete_circular_positive_fixture"), "circular_positive"),
        (_payload("null", "complete_null_fixture"), "null"),
        (_payload("blocked", "blocked_fixture"), "blocked"),
        (_payload("disqualified", "complete_disqualified_fixture"), "disqualified"),
        (_payload("partial", "partial_fixture"), "partial"),
        ({"status": "complete", "honest_verdict": "unfinished"}, "partial"),
    ],
)
def test_scenario_report_6995_verdict_class_propagation(
    payload: dict[str, object] | None,
    expected: str,
) -> None:
    """SCENARIO-REPORT-6995-VERDICTS preserves the closed verdict vocabulary."""

    row = mod.classify_task(mod.EXPECTED_TASKS[0], payload)
    assert row["verdict_class"] == expected
    assert row["terminal"] is True
    if payload is None:
        assert row["verdict_class"] != "partial"


def test_scenario_report_6995_blocked_diagnostics_preserve_source_summary() -> None:
    """SCENARIO-REPORT-6995-BLOCKED keeps the exact failed check values."""

    payload = {
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "gate_check_summary": "gate-unsat(final)",
        "failed_upstream": "exp6987-contrast-feature-audit",
        "failed_field": "contrast_feature_bank_ready_score",
        "failed_expected": 1,
        "failed_observed": 0,
    }
    row = mod.classify_task(mod.EXPECTED_TASKS[4], payload)
    diagnostic = mod.blocked_diagnostic(mod.EXPECTED_TASKS[4], payload, row)
    assert diagnostic == {
        "number": 6988,
        "task_id": "exp6988-certified-pwa-kan-ranker",
        "failed_check": "exp6987-contrast-feature-audit.contrast_feature_bank_ready_score",
        "expected_value": 1,
        "observed_value": 0,
        "gate_check_summary": "gate-unsat(final)",
        "terminal": True,
    }

    absent = mod.blocked_diagnostic(mod.EXPECTED_TASKS[5], None, {"verdict_class": "blocked"})
    assert absent["failed_check"] == "expected_deliverable_readable"
    assert absent["expected_value"] is True
    assert absent["observed_value"] is False


def test_scenario_report_6995_rows_recompute_available_primary_evidence(
    actual_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6995-ROWS derives counts and the failed shortcut from rows."""

    contrast = actual_artifact["contrast_evidence_rows"]
    recomputed = {
        row["evidence"]: row["recomputed"] for row in contrast if "recomputed" in row
    }
    assert {
        "contrast_pair_count": 36,
        "contrast_candidate_count": 72,
        "chronological_event_count": 24,
        "chronological_candidate_count": 48,
        "feature_candidate_count": 138,
        "feature_model_row_count": 414,
        "feature_raw_token_count": 201485,
    }.items() <= recomputed.items()
    shortcut = next(
        row
        for row in contrast
        if row["evidence"] == "shortcut_probe:mutation_metadata"
    )
    assert shortcut["recomputed_passed"] is False
    assert shortcut["upper_ci"] == pytest.approx(1.0)
    assert actual_artifact["pwa_energy_evidence_rows"][0]["state"] == "blocked"
    assert actual_artifact["selection_evidence_rows"][0]["state"] == "blocked"
    assert actual_artifact["continuous_learning_evidence_rows"][0]["state"] == "blocked"
    assert actual_artifact["support_and_forgetting_rows"][0]["state"] == "blocked"


def test_scenario_report_6995_arc_recomputes_hashes_and_route_influence(
    actual_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6995-ROWS recomputes ARC reachability without solve credit."""

    rows = actual_artifact["arc_producer_evidence_rows"]
    summary = next(row for row in rows if row["evidence"] == "producer_reachability")
    assert summary["producer_complete"] is True
    assert summary["cold_audit_confirmed"] is True
    assert summary["route_influence"] is True
    assert summary["hash_checks_passed"] is True
    assert summary["solve_claimed"] is False
    assert summary["model_quality_claimed"] is False
    assert summary["registry_updated"] is False
    assert summary["science_positive"] is False


@pytest.mark.parametrize(
    ("selection", "selection_audit", "learning", "learning_audit", "expected"),
    [
        (1, True, 0, False, 1),
        (1, False, 0, False, 0),
        (0, False, 1, True, 1),
        (0, False, 1, False, 0),
        (0, False, 0, False, 0),
    ],
)
def test_scenario_report_6995_science_score_requires_cold_audit(
    selection: int,
    selection_audit: bool,
    learning: int,
    learning_audit: bool,
    expected: int,
) -> None:
    """SCENARIO-REPORT-6995-VERDICTS admits only audited selection or learning."""

    assert (
        mod.science_positive_score(selection, selection_audit, learning, learning_audit)
        == expected
    )


def test_scenario_report_6995_publication_fields_are_copied() -> None:
    """SCENARIO-REPORT-6995-PUBLICATION never derives stable gates from task rows."""

    source = {
        "paper_ready": False,
        "gates": {
            "G1": {"pass": True, "detail": "one"},
            "G2": {"pass": False, "detail": "two"},
            "G3": {"pass": True, "detail": "three"},
            "G4": {"pass": True, "detail": "four"},
        },
        "unmet_gates": ["G2"],
    }
    state = mod.publication_state(source)
    assert state["g1"] is True
    assert state["g2"] is False
    assert state["g3"] is True
    assert state["g4"] is True
    assert state["paper_ready"] is False
    assert state["unmet_gates"] == ["G2"]
    assert [row["gate"] for row in state["publication_gate_rows"]] == [
        "G1",
        "G2",
        "G3",
        "G4",
    ]


def test_scenario_report_6995_actual_capstone_is_terminal_and_evidence_led(
    actual_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6995-ARTIFACTS closes every slot despite missing science."""

    assert actual_artifact["v612_capstone_complete_score"] == 1
    assert actual_artifact["v612_task_contract_conforms_score"] == 1
    assert actual_artifact["v612_science_positive_score"] == 0
    assert actual_artifact["verdict_class"] == "null"
    assert actual_artifact["honest_verdict"].startswith("complete_null_")
    assert len(actual_artifact["rows"]) == 12
    assert [row["number"] for row in actual_artifact["rows"]] == list(range(6984, 6996))
    assert [row["number"] for row in actual_artifact["missing_task_rows"]] == [
        6989,
        6990,
        6991,
        6992,
    ]
    assert all(row["verdict_class"] == "blocked" for row in actual_artifact["missing_task_rows"])
    assert len(actual_artifact["per_branch_results"]) == 3
    assert all(row["terminal"] for row in actual_artifact["per_branch_results"])
    handoff = actual_artifact["v613_handoff_rows"]
    assert len(handoff) == 1
    assert handoff[0]["earliest_causal_boundary"] == "exp6987_mutation_metadata_shortcut"
    forbidden = " ".join(handoff[0]["forbidden_revivals"])
    assert "spilled_energy" in forbidden
    assert "public_arc_resolve" in forbidden


def test_scenario_report_6995_blocked_contract_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6995-BLOCKED writes a terminal contract-precondition result."""

    blocked = mod.build_artifact(tmp_path, "20260904", run_commands=False)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == mod.BLOCKED_VERDICT
    assert blocked["gate_check_summary"]["failed_check"] == "contract_preconditions"
    assert blocked["gate_check_summary"]["expected_value"] == "readable activated V612 contracts"
    assert blocked["v612_capstone_complete_score"] == 0


def test_scenario_report_6995_validator_rejects_forgery(
    actual_artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-6995-ARTIFACT rejects scores, fields, gates, prefixes, and checksum."""

    artifact = deepcopy(actual_artifact)
    artifact["command_receipt_rows"] = [
        {
            "name": "fixture",
            "command": "fixture",
            "exit_code": 0,
            "finding": "ok",
            "terminal": True,
        }
    ]
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    forged = deepcopy(artifact)
    del forged["selection_evidence_rows"]
    forged["v612_capstone_complete_score"] = 0
    forged["v612_science_positive_score"] = 1
    forged["g1"] = not forged["publication_gate_rows"][0]["passed"]
    forged["paper_ready"] = not all(
        row["passed"] for row in forged["publication_gate_rows"]
    )
    forged["unmet_gates"] = ["forged"]
    forged["honest_verdict"] = "partial_wrong"
    forged["reproducibility_checksum"] = "sha256:forged"
    errors = mod.validate_artifact(forged)
    assert any("missing required field" in error for error in errors)
    assert any("complete score" in error for error in errors)
    assert any("science score" in error for error in errors)
    assert any("publication gate" in error for error in errors)
    assert any("paper_ready" in error for error in errors)
    assert any("unmet_gates" in error for error in errors)
    assert any("prefix" in error for error in errors)
    assert any("checksum" in error for error in errors)


def test_command_receipts_and_cli_write_only_requested_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    actual_artifact: dict[str, object],
) -> None:
    """REQ-REPORT-6995 records command findings and validates the requested output."""

    class Result:
        returncode = 7
        stdout = "finding"
        stderr = "warning"

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Result())
    receipt = mod.run_command(REPO_ROOT, "fixture", ["tool", "--check"])
    assert receipt["exit_code"] == 7
    assert receipt["finding"] == "finding\nwarning"
    assert receipt["terminal"] is True

    output = tmp_path / "capstone.json"
    monkeypatch.setattr(mod, "build_artifact", lambda *args, **kwargs: deepcopy(actual_artifact))
    assert mod.main(["--repo-root", str(REPO_ROOT), "--date", "20260904", "--output", str(output), "--no-commands"]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["experiment_id"] == 6995
    assert mod.main(["--validate", str(output)]) == 0

    written["reproducibility_checksum"] = "bad"
    output.write_text(json.dumps(written), encoding="utf-8")
    assert mod.main(["--validate", str(output)]) == 1


def test_parser_and_loader_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-6995 closes malformed contract and JSON inputs."""

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mod.load_yaml(bad_yaml)
    assert mod.parse_document_contract("no contract table") == []
    assert mod.parse_yaml_contract({"tasks": "wrong"}) == []
    rows = mod.parse_yaml_contract({"milestone": mod.MILESTONE, "tasks": [None]})
    assert rows[0]["number"] is None

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._load_json(bad_json)[0] is None
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._load_json(list_json) == (None, "JSON root is not an object")


def test_reducers_cover_available_science_and_absent_arc() -> None:
    """SCENARIO-REPORT-6995-ROWS recomputes available branch rows."""

    task_rows = {
        number: {"verdict_class": "positive"}
        for number in range(6988, 6993)
    }
    payloads = {
        6988: {"rows": [{"terminal": True}], "pwa_model_ready_score": 1},
        6989: {"pwa_certificate_confirmed_score": 1},
        6990: {
            "rows": [
                {"selected_exact": True, "baseline_exact": False},
                {"selected_exact": False, "baseline_exact": False},
            ],
            "selection_positive_score": 1,
        },
        6991: {
            "rows": [{"learning_gain": 0.2}, {"learning_gain": 0.4}, {"learning_gain": None}],
            "learning_positive_score": 1,
        },
        6992: {
            "rows": [{"terminal": True}],
            "future_support_confirmed_score": 1,
            "forgetting_gate_passed": True,
            "self_learning_confirmed_score": 1,
        },
    }
    pwa, selection, learning, support = mod.recompute_pwa_selection_learning(
        payloads, task_rows
    )
    assert pwa[0]["certificate_confirmed"] is True
    assert selection[0]["selection_delta"] == 1
    assert selection[0]["cold_audit_confirmed"] is True
    assert learning[0]["mean_learning_gain"] == pytest.approx(0.3)
    assert support[0]["future_support_confirmed"] is True
    assert support[0]["forgetting_within_budget"] is True
    assert mod.recompute_arc(None, None)[0]["state"] == "blocked"
    assert all(
        row["evidence"] != "shortcut_probe:bad"
        for row in mod.recompute_contrast(
            {6987: {"shortcut_interval_rows": [None, {"probe_name": "bad"}]}}
        )
        if row.get("recomputed_passed") is True
    )
    retirement_rows = mod._retirement_rows(
        [
            {
                "number": 6984,
                "task_id": "fixture",
                "prior_failures": [None, {"verdict": "same", "retire_if_same_verdict": True}],
            }
        ],
        {6984: {"honest_verdict": "same"}},
    )
    assert retirement_rows[0]["retirement_required"] is True
    assert retirement_rows[0]["disposition"] == "retire"


def test_operational_command_list_and_publication_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6995 runs all reporting commands and handles bad gate JSON."""

    calls: list[str] = []

    def fake_command(root: Path, name: str, argv: list[str]) -> dict[str, object]:
        calls.append(name)
        stdout = (
            json.dumps(
                {
                    "paper_ready": True,
                    "gates": {name: {"pass": True} for name in ("G1", "G2", "G3", "G4")},
                    "unmet_gates": [],
                }
            )
            if name == "publication_gate"
            else "ok"
        )
        return {
            "name": name,
            "command": " ".join(argv),
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "finding": stdout,
            "terminal": True,
        }

    monkeypatch.setattr(mod, "run_command", fake_command)
    receipts, publication = mod.run_operational_commands(REPO_ROOT, [])
    assert len(receipts) == 10
    assert calls[-1] == "publication_gate"
    assert publication["paper_ready"] is True

    def bad_publication(root: Path, name: str, argv: list[str]) -> dict[str, object]:
        row = fake_command(root, name, argv)
        if name == "publication_gate":
            row["stdout"] = "not json"
        return row

    monkeypatch.setattr(mod, "run_command", bad_publication)
    _, fallback = mod.run_operational_commands(REPO_ROOT, [])
    assert fallback["unmet_gates"] == ["G1", "G2", "G3", "G4"]


def test_build_handles_unreadable_inputs_and_command_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6995-BLOCKED distinguishes contract and artifact failures."""

    design = tmp_path / mod.DESIGN_PATH
    roadmap = tmp_path / mod.ROADMAP_PATH
    design.parent.mkdir(parents=True)
    design.write_text((REPO_ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    roadmap.write_text((REPO_ROOT / mod.ROADMAP_PATH).read_text(encoding="utf-8"), encoding="utf-8")
    bad_result = tmp_path / mod.EXPECTED_TASKS[0]["deliverable"]
    bad_result.parent.mkdir(parents=True)
    bad_result.write_text("{", encoding="utf-8")

    required_files = [
        mod.V611_CAPSTONE_PATH,
        Path("scripts/publication_gate.py"),
        Path("scripts/adversarial_verify.py"),
        Path("scripts/verdict_row_consistency_lint.py"),
        Path("scripts/recurring_blocker_ledger.py"),
        Path("scripts/roadmap_schema.py"),
        Path("scripts/audit_roadmap_gates.py"),
        Path("scripts/exclusion_manifest_lint.py"),
        Path("scripts/check_spec_coverage.py"),
        Path("scripts/root_clutter_sweep.py"),
    ]
    for relative in required_files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
    publication = mod.publication_state(
        {"paper_ready": False, "gates": {}, "unmet_gates": ["G1", "G2", "G3", "G4"]}
    )
    monkeypatch.setattr(
        mod,
        "run_operational_commands",
        lambda *args: ([{"terminal": True}], publication),
    )
    built = mod.build_artifact(tmp_path, "20260904", run_commands=True)
    assert built["rows"][0]["outcome"] == "unreadable"
    assert built["command_receipt_rows"] == [{"terminal": True}]

    broken_root = tmp_path / "broken"
    (broken_root / mod.DESIGN_PATH).mkdir(parents=True)
    (broken_root / mod.ROADMAP_PATH).parent.mkdir(parents=True, exist_ok=True)
    (broken_root / mod.ROADMAP_PATH).write_text("tasks: []\n", encoding="utf-8")
    (broken_root / "results").mkdir(parents=True, exist_ok=True)
    for relative in required_files:
        target = broken_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
    blocked = mod.build_artifact(broken_root, "20260904", run_commands=False)
    assert blocked["gate_check_summary"]["failed_check"] == "contract_readability"


def test_validator_covers_all_structural_errors_and_main_build_failure(
    actual_artifact: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6995-ARTIFACT reports each independent schema failure."""

    broken = deepcopy(actual_artifact)
    broken["field_principles"] = []
    broken["inference_substrate"] = "wrong"
    broken["verifier_is_oracle"] = False
    broken["verdict_class"] = "unknown"
    broken["v612_task_contract_conforms_score"] = 0
    broken["reproducibility_checksum"] = "bad"
    errors = mod.validate_artifact(broken)
    assert "field_principles must be a mapping" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "verdict_class is not closed" in errors
    assert "contract score does not match exact rows" in errors

    blocked = mod.build_artifact(tmp_path, "20260904", run_commands=False)
    assert mod.validate_artifact(blocked) == []

    monkeypatch.setattr(mod, "build_artifact", lambda *args, **kwargs: broken)
    assert mod.main(["--repo-root", str(tmp_path), "--output", "bad.json", "--no-commands"]) == 1
