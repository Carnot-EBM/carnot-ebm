"""Tests for REQ-CAPSTONE-7027 and its named scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from carnot import experiment_7027_v615_capstone as mod


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


def test_req_capstone_7027_spec_precedes_implementation() -> None:
    """REQ-CAPSTONE-7027 defines the full capstone contract first."""

    text = (REPO_ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-CAPSTONE-7027", 1)[1]
    for anchor in (
        "SCENARIO-CAPSTONE-7027-CONTRACT",
        "SCENARIO-CAPSTONE-7027-UPSTREAM-STATES",
        "SCENARIO-CAPSTONE-7027-ROWS",
        "SCENARIO-CAPSTONE-7027-ARC",
        "SCENARIO-CAPSTONE-7027-TERMINAL",
        "SCENARIO-CAPSTONE-7027-HANDOFF",
        "SCENARIO-CAPSTONE-7027-INTEGRITY",
    ):
        assert anchor in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert mod.REQUIRED_ARTIFACT_FIELDS <= set(mod.FIELD_PRINCIPLES)


def test_scenario_capstone_7027_contract_exact_order_and_parity() -> None:
    """SCENARIO-CAPSTONE-7027-CONTRACT checks all 12 ordered tasks."""

    document = mod.parse_document_contract(
        (REPO_ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    )
    roadmap = mod.parse_yaml_contract(
        yaml.safe_load((REPO_ROOT / mod.ROADMAP_PATH).read_text(encoding="utf-8"))
    )
    rows = mod.build_contract_rows(document, roadmap)
    assert [row["task_id"] for row in document] == mod.EXPECTED_ID_ORDER
    assert [row["task_id"] for row in roadmap] == mod.EXPECTED_ID_ORDER
    assert len(rows) == 12
    assert all(row["passed"] for row in rows)
    assert all(row["producer_fields_present"] for row in rows)

    changed = deepcopy(roadmap)
    changed[0]["title"] = "wrong"
    changed[1]["deliverable"] = "results/wrong.json"
    changed[4]["gates"][0]["artifact_field"] = "wrong_field"
    changed.append(changed.pop(5))
    bad = mod.build_contract_rows(document, changed)
    assert not mod.contract_conforms(document, changed, bad)
    assert {name for row in bad for name in row["failed_checks"]} >= {
        "title",
        "deliverable",
        "gates",
        "order",
    }


@pytest.mark.parametrize(
    ("payload", "expected_class", "expected_state", "eligible"),
    [
        (None, "blocked", "missing", False),
        (_payload(), "positive", "positive", True),
        (_payload("null", "complete_null_fixture"), "null", "null", False),
        (_payload("blocked", "blocked_fixture"), "blocked", "blocked", False),
        (
            _payload("disqualified", "complete_disqualified_fixture"),
            "disqualified",
            "disqualified",
            False,
        ),
        (
            _payload("circular_positive", "complete_circular_positive_fixture"),
            "circular_positive",
            "circular_positive",
            False,
        ),
        (_payload("partial", "partial_fixture"), "partial", "partial", False),
    ],
)
def test_scenario_capstone_7027_closed_enum_propagates(
    payload: dict[str, object] | None,
    expected_class: str,
    expected_state: str,
    eligible: bool,
) -> None:
    """SCENARIO-CAPSTONE-7027-UPSTREAM-STATES preserves closed classes."""

    row = mod.classify_task(mod.EXPECTED_TASKS[0], payload)
    assert row["verdict_class"] == expected_class
    assert row["state"] == expected_state
    assert row["eligible_for_promotion"] is eligible
    if payload is None:
        assert row["verdict_class"] != "partial"


def test_scenario_capstone_7027_blocked_and_oracle_boundaries() -> None:
    """SCENARIO-CAPSTONE-7027-UPSTREAM-STATES keeps gate and oracle limits."""

    blocked = _payload(
        "blocked",
        "blocked_gate_check_failed",
        failed_upstream="exp7025-belief-shadow-live-trace",
        failed_field="belief_shadow_trace_ready_score",
        failed_expected=1,
        failed_observed=0,
        blocked_at_layer="conductor_pre_gate",
    )
    row = mod.classify_task(mod.EXPECTED_TASKS[10], blocked)
    assert row["verdict_class"] == "blocked"
    assert row["state"] == "blocked"
    assert row["blocked_at_layer"] == "conductor_pre_gate"

    oracle = mod.classify_task(mod.EXPECTED_TASKS[5], _payload(verifier_is_oracle=True))
    assert oracle["verdict_class"] == "circular_positive"
    assert oracle["eligible_for_promotion"] is False


def test_scenario_capstone_7027_recomputes_prospective_rows() -> None:
    """SCENARIO-CAPSTONE-7027-ROWS derives comparisons from unit rows."""

    rows = []
    values = {
        "u1": {"frozen": 0.0, "recency_only": 1.0, "counterexample_belief": 1.0},
        "u2": {"frozen": 0.0, "recency_only": 1.0, "counterexample_belief": 0.0},
        "u3": {"frozen": 0.5, "recency_only": 0.5, "counterexample_belief": 1.0},
    }
    for unit, arms in values.items():
        for arm, score in arms.items():
            rows.append(
                {
                    "unit_id": unit,
                    "arm": arm,
                    "support_class": "supported",
                    "action_ranking_accuracy": score,
                    "selected_action_id": f"{unit}-{arm}",
                    "terminal": True,
                }
            )
    rows.extend(
        {
            "unit_id": "u4",
            "arm": arm,
            "support_class": "no_headroom",
            "action_ranking_accuracy": None,
            "selected_action_id": None,
            "terminal": True,
        }
        for arm in mod.PROSPECTIVE_ARMS
    )
    payload = {
        "per_decision_results": rows,
        "paired_delta_rows": [{"comparison": "counterexample_belief_minus_recency_only"}],
        "confidence_intervals": [{"arm": "counterexample_belief", "lower": 0.0}],
        "query_cost_rows": rows,
    }
    result = mod.recompute_prospective_utility(payload)
    assert result["unit_count"] == 4
    assert result["no_headroom_unit_count"] == 1
    assert result["missing_cell_count"] == 0
    assert result["comparisons"]["counterexample_belief_minus_frozen"] == {
        "wins": 2,
        "losses": 0,
        "ties": 1,
        "comparable_units": 3,
    }
    assert result["comparisons"]["counterexample_belief_minus_recency_only"] == {
        "wins": 1,
        "losses": 1,
        "ties": 1,
        "comparable_units": 3,
    }
    assert result["selected_action_count"] == 9
    assert result["model_call_count"] == 0
    assert result["intervals"] == payload["confidence_intervals"]


def test_scenario_capstone_7027_live_block_has_no_pooled_substitute() -> None:
    """SCENARIO-CAPSTONE-7027-ROWS keeps an absent live table absent."""

    blocked = {
        "verdict_class": "blocked",
        "honest_verdict": "blocked_gate_check_failed",
        "failed_field": "belief_shadow_trace_ready_score",
        "failed_expected": 1,
        "failed_observed": 0,
    }
    result = mod.recompute_live_ab(blocked)
    assert result["blocked"] is True
    assert result["per_game_row_count"] == 0
    assert result["wins"] == result["losses"] == result["ties"] == 0
    assert result["pooled_value_used"] is False
    assert result["missing_cells"] == "all_planned_live_cells"


def test_scenario_capstone_7027_solve_provenance_and_duplicates() -> None:
    """SCENARIO-CAPSTONE-7027-ARC excludes proxies and registry duplicates."""

    registry = {"games": {"g1": {"levels_reproduced": 2}}}
    payloads = {
        1: {
            "per_game_results": [
                {
                    "game": "g1",
                    "level_after": 2,
                    "solved": True,
                    "solve_provenance": "live_agent_self_discovery",
                },
                {
                    "game": "g2",
                    "level_after": 1,
                    "solved": True,
                    "solve_provenance": "development_proxy",
                },
                {
                    "game": "g3",
                    "level_after": 1,
                    "solved": True,
                    "solve_provenance": "live_agent_self_discovery",
                },
            ]
        }
    }
    provenance, registry_rows = mod.build_solve_rows(payloads, registry)
    assert [row["eligible_for_live_headline"] for row in provenance] == [False, False, True]
    assert provenance[0]["duplicate"] is True
    assert registry_rows[1]["registered_level"] == 2


def test_scenario_capstone_7027_actual_artifact_hashes_and_blocked_handoff() -> None:
    """SCENARIO-CAPSTONE-7027-INTEGRITY hashes sources and stops on block."""

    artifact = mod.build_artifact(REPO_ROOT, "20260905", run_commands=False)
    assert mod.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert set(artifact["field_principles"]) == mod.REQUIRED_ARTIFACT_FIELDS
    assert artifact["expected_task_count"] == 12
    assert artifact["observed_task_count"] == 12
    assert artifact["expected_id_order"] == mod.EXPECTED_ID_ORDER
    assert artifact["observed_id_order"] == mod.EXPECTED_ID_ORDER
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["docs_reconciled"] is False
    assert artifact["v616_handoff"]["action"] == "repair_one_named_evidence_defect"
    assert "model_filename" in artifact["v616_handoff"]["named_defect"]
    assert artifact["promoted_claims"] == []
    for citation in artifact["cited_upstream_artifacts"]:
        source = REPO_ROOT / citation["artifact_path"]
        assert citation["sha256"] == mod.sha256_path(source)
        assert citation["summarizer_ran_before_import"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_7027_checksum_rejects_evidence_change() -> None:
    """SCENARIO-CAPSTONE-7027-INTEGRITY binds the evidence rows."""

    artifact = mod.build_artifact(REPO_ROOT, "20260905", run_commands=False)
    changed = deepcopy(artifact)
    changed["blockers"] = [*changed["blockers"], {"unexpected": True}]
    assert changed["reproducibility_checksum"] != mod.reproducibility_checksum(changed)
    assert "reproducibility checksum mismatch" in mod.validate_artifact(changed)


def test_scenario_capstone_7027_main_writes_only_requested_output(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-7027-INTEGRITY supports an isolated output path."""

    output = tmp_path / "capstone.json"
    assert (
        mod.main(
            [
                "--root",
                str(REPO_ROOT),
                "--date",
                "20260905",
                "--output",
                str(output),
                "--no-commands",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["verdict_class"] == "blocked"
    assert payload["gate_check_summary"]["failed_check"] == "required_live_upstream_complete"


def test_contract_parsers_and_classifier_reject_malformed_inputs() -> None:
    """REQ-CAPSTONE-7027 fails closed for malformed contract and verdict data."""

    assert mod.parse_document_contract("no contract") == []
    assert mod.parse_yaml_contract({"tasks": "wrong"}) == []
    assert mod._document_gates("Exp9999 `field >= 2`") == [
        {"upstream": "exp9999", "artifact_field": "field", "op": ">=", "value": 2}
    ]
    task = mod.EXPECTED_TASKS[0]
    assert (
        mod.classify_task(task, {"honest_verdict": "complete_disqualified_x"})["verdict_class"]
        == "disqualified"
    )
    assert mod.classify_task(task, {"honest_verdict": "complete_null_x"})["verdict_class"] == "null"
    assert mod.classify_task(task, {"honest_verdict": "unknown"})["verdict_class"] == "partial"
    flagged = mod.classify_task(task, _payload(flagged_adversarial=True))
    assert flagged["state"] == "flagged"


def test_source_loader_preserves_missing_unreadable_and_non_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-7027-UPSTREAM-STATES records bad source shapes."""

    tasks = (
        {"number": 1, "task_id": "exp1-a", "deliverable": "results/a.json"},
        {"number": 2, "task_id": "exp2-b", "deliverable": "results/b.json"},
        {"number": 3, "task_id": "exp3-c", "deliverable": "results/c.json"},
        {"number": 4, "task_id": "exp4-self", "deliverable": "results/self.json"},
    )
    (tmp_path / "results").mkdir()
    (tmp_path / "results/a.json").write_text("{", encoding="utf-8")
    (tmp_path / "results/b.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(mod, "EXPECTED_TASKS", tasks)
    monkeypatch.setattr(mod, "IMPORTED_FIELDS", {1: [], 2: [], 3: []})
    monkeypatch.setattr(
        mod,
        "_summarize_before_import",
        lambda root, path: {"exit_code": 0, "status": "clean"},
    )
    payloads, citations, hashes = mod._load_sources(tmp_path)
    assert payloads == {}
    assert len(citations) == 1
    assert citations[0]["read_error"].startswith("JSONDecodeError")
    assert set(hashes) == {"results/a.json"}


def test_verifier_rows_skip_unreadable_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-7027 retains verifier results for readable sources only."""

    from scripts import adversarial_verify, verdict_row_consistency_lint

    citations = [
        {"experiment_id": 1, "artifact_path": "a.json"},
        {"experiment_id": 2, "artifact_path": "b.json", "read_error": "bad"},
    ]
    monkeypatch.setattr(
        adversarial_verify,
        "verify_artifact",
        lambda path: {
            "max_severity": 2,
            "flags": [{"kind": "X", "severity": "critical"}],
        },
    )
    monkeypatch.setattr(
        verdict_row_consistency_lint,
        "check_artifact",
        lambda path: ("findings", ["bad row"]),
    )
    adversarial = mod._adversarial_rows(Path("/tmp"), citations)
    consistency = mod._row_consistency_rows(Path("/tmp"), citations)
    assert len(adversarial) == len(consistency) == 1
    assert adversarial[0]["critical"] is True
    assert consistency[0]["critical"] is True


def test_recomputation_covers_missing_cells_and_live_rows() -> None:
    """SCENARIO-CAPSTONE-7027-ROWS counts only real cells and receipts."""

    prospective = mod.recompute_prospective_utility(
        {
            "per_decision_results": [
                {
                    "unit_id": "u",
                    "arm": "counterexample_belief",
                    "action_ranking_accuracy": 1.0,
                    "support_class": "supported",
                    "selected_action_id": "a",
                    "model_calls": 2,
                },
                {"unit_id": "", "arm": "frozen", "action_ranking_accuracy": 0.0},
            ]
        }
    )
    assert prospective["missing_cell_count"] == 2
    assert prospective["model_call_count"] == 2
    live = mod.recompute_live_ab(
        {
            "status": "complete",
            "per_game_results": [
                {
                    "win": True,
                    "loss": False,
                    "tie": False,
                    "no_headroom": True,
                    "model_calls": 3,
                    "actions": 7,
                    "compute_receipt": {"ok": True},
                },
                {"loss": True, "tie": True, "action_count": 2},
            ],
            "confidence_intervals": [{"lower": 0.0}],
            "paired_delta_rows": [{"point_estimate": 0.1}],
        }
    )
    assert (live["wins"], live["losses"], live["ties"]) == (1, 1, 1)
    assert live["no_headroom_unit_count"] == 1
    assert live["model_call_count"] == 3
    assert live["action_count"] == 9
    assert live["compute_receipt_count"] == 1


def test_solve_rows_ignore_unsolved_rows_and_support_list_registry() -> None:
    """SCENARIO-CAPSTONE-7027-ARC accepts the production registry shape."""

    payloads = {
        1: {
            "per_game_results": [
                {"game": "g", "solved": False},
                {
                    "game": "g",
                    "new_levels_banked": 1,
                    "reproduced_levels": 2,
                    "solve_provenance": "live_agent_self_discovery",
                },
            ]
        }
    }
    provenance, registry = mod.build_solve_rows(
        payloads, {"games": [{"game": "g", "levels_reproduced": 1}]}
    )
    assert len(provenance) == 1
    assert provenance[0]["eligible_for_live_headline"] is True
    assert registry[1]["registered_level"] == 1


def _branch_payloads(kind: str) -> dict[int, dict[str, object]]:
    """Return complete synthetic upstreams for terminal-class branch tests."""

    payloads = {number: _payload() for number in range(7016, 7027)}
    payloads[7021].update(
        verdict_class="null",
        honest_verdict="complete_null_fixture",
        belief_future_utility_positive_score=int(kind == "positive"),
    )
    payloads[7022]["belief_shadow_safe_score"] = int(kind != "fallback")
    payloads[7025].update(
        belief_shadow_trace_ready_score=int(kind == "positive"),
        gate_check_summary="not a mapping",
    )
    payloads[7026].update(
        belief_live_value_positive_score=int(kind == "positive"),
        per_game_results=[{"win": True}] if kind == "positive" else [],
        budget_rows=[{"passed": True}] if kind == "positive" else [],
        phase_receipt_rows=[{"phase": "inference"}] if kind == "positive" else [],
        gpu_sample_rows=[{"gpu": 0}] if kind == "positive" else [],
    )
    if kind == "disqualified":
        payloads[7018].update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_fixture",
        )
    return payloads


@pytest.mark.parametrize(
    ("kind", "expected_class"),
    [
        ("disqualified", "disqualified"),
        ("null", "null"),
        ("positive", "positive"),
        ("fallback", "null"),
    ],
)
def test_build_artifact_terminal_class_branches(
    kind: str, expected_class: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-7027-TERMINAL exercises each completed outcome."""

    payloads = _branch_payloads(kind)
    monkeypatch.setattr(mod, "_load_sources", lambda root: (payloads, [], {}))
    monkeypatch.setattr(mod, "_adversarial_rows", lambda root, citations: [])
    monkeypatch.setattr(mod, "_row_consistency_rows", lambda root, citations: [])
    artifact = mod.build_artifact(REPO_ROOT, "20260905", run_commands=False)
    assert artifact["verdict_class"] == expected_class
    if kind == "positive":
        assert artifact["promoted_claims"]
        assert artifact["v616_handoff"]["action"] == "promote_belief_to_larger_held_roster"
    if kind in {"null", "fallback"}:
        assert artifact["retirements"]
        assert artifact["v616_handoff"]["action"] == "retire_explicit_belief_after_complete_null"


def test_build_artifact_precondition_and_parse_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CAPSTONE-7027-INTEGRITY names local execution failures."""

    artifact = mod.build_artifact(tmp_path, "20260905", run_commands=False)
    assert artifact["gate_check_summary"]["failed_check"] == "v615_markdown_contract"

    monkeypatch.setattr(
        mod,
        "_preconditions",
        lambda root, output: ([{"check": "all", "passed": True}], True),
    )
    parse_block = mod.build_artifact(REPO_ROOT, "20260905", run_commands=False)
    assert parse_block["gate_check_summary"]["failed_check"] != "capstone_source_parse"
    monkeypatch.setattr(mod.yaml, "safe_load", lambda text: [])
    parse_block = mod.build_artifact(REPO_ROOT, "20260905", run_commands=False)
    assert parse_block["gate_check_summary"]["failed_check"] == "capstone_source_parse"
    active = yaml.load(
        (REPO_ROOT / mod.ROADMAP_PATH).read_text(encoding="utf-8"),
        Loader=yaml.SafeLoader,
    )
    document = mod.parse_document_contract(
        (REPO_ROOT / mod.DESIGN_PATH).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(mod, "parse_document_contract", lambda text: document)
    values = iter((active, []))
    monkeypatch.setattr(mod.yaml, "safe_load", lambda text: next(values))
    registry_block = mod.build_artifact(REPO_ROOT, "20260905", run_commands=False)
    assert registry_block["gate_check_summary"]["failed_check"] == "capstone_source_parse"


def test_validator_reports_all_structural_failures() -> None:
    """REQ-CAPSTONE-7027 artifact validation fails closed."""

    bad = {
        "field_principles": {},
        "inference_substrate": "wrong",
        "expected_task_count": 13,
        "expected_id_order": [],
        "observed_task_count": 1,
        "observed_id_order": [],
        "verdict_class": "blocked",
        "honest_verdict": "wrong",
        "gate_check_summary": {},
        "docs_reconciled": True,
        "reproducibility_checksum": "bad",
    }
    errors = mod.validate_artifact(bad)
    assert "field principles do not cover every required field" in errors
    assert "inference substrate mismatch" in errors
    assert "expected task count mismatch" in errors
    assert "expected task order mismatch" in errors
    assert "observed task order mismatch" in errors
    assert "honest verdict prefix does not match verdict class" in errors
    assert "blocked verdict lacks exact gate diagnostic" in errors
    assert "conductor-owned document reconciliation must remain deferred" in errors
    assert "reproducibility checksum mismatch" in errors


def test_command_helpers_and_main_validation_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-7027 records checks and validates isolated artifacts."""

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    assert mod._run_command(tmp_path, "check", ["true"])["exit_code"] == 0
    assert len(mod.run_validation_commands(tmp_path, tmp_path / "out.json")) == 9

    good = mod.build_artifact(REPO_ROOT, "20260905", run_commands=False)
    good_path = tmp_path / "good.json"
    good_path.write_text(json.dumps(good), encoding="utf-8")
    assert mod.main(["--validate", str(good_path)]) == 0
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{", encoding="utf-8")
    assert mod.main(["--validate", str(broken_path)]) == 1


def test_main_default_output_and_command_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CAPSTONE-7027 main can attach validation receipts."""

    artifact = mod.build_artifact(REPO_ROOT, "20260905", run_commands=False)
    writes: list[Path] = []
    monkeypatch.setattr(mod, "build_artifact", lambda root, date, run_commands: deepcopy(artifact))
    monkeypatch.setattr(mod, "_write", lambda path, payload: writes.append(path))
    monkeypatch.setattr(
        mod,
        "run_validation_commands",
        lambda root, output: [{"name": "all", "exit_code": 0}],
    )
    assert mod.main(["--root", str(tmp_path)]) == 0
    assert writes == [tmp_path / mod.OUTPUT_PATH, tmp_path / mod.OUTPUT_PATH]
    writes.clear()
    assert mod.main(["--root", str(tmp_path), "--output", "relative.json", "--no-commands"]) == 0
    assert writes == [tmp_path / "relative.json"]
