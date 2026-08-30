"""Focused tests for the V589 branch-disposition capstone.

Spec refs: REQ-REPORT-6767, SCENARIO-REPORT-6767-PRECONDITIONS,
SCENARIO-REPORT-6767-ROW-RECOMPUTATION,
SCENARIO-REPORT-6767-BRANCH-ISOLATION, and
SCENARIO-REPORT-6767-VALIDATORS.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import yaml


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "python/carnot/experiment_6767_v589_branch_disposition.py"
SPEC = importlib.util.spec_from_file_location("exp6767_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
exp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = exp
SPEC.loader.exec_module(exp)


def _present(payload: dict, path: str = "results/synthetic.json") -> dict:
    return {
        "artifact_state": "present",
        "valid_json": True,
        "payload": payload,
        "path": path,
        "sha256": "sha256:synthetic",
        "error": None,
    }


@pytest.fixture(scope="module")
def current_inputs() -> tuple[list[dict], dict[str, dict]]:
    planned = exp.load_planned_tasks(REPO)
    sources = exp.load_source_artifacts(REPO, planned)
    return planned, sources


def test_req_report_6767_spec_precedes_implementation() -> None:
    """REQ-REPORT-6767: the reporting spec owns this capstone contract."""

    text = (REPO / exp.REPORT_SPEC_PATH).read_text(encoding="utf-8")
    section = text.split("REQ-REPORT-6767", 1)[1]
    anchors = set(exp.spec_anchors(section))

    assert {
        "REQ-REPORT-6767",
        "SCENARIO-REPORT-6767-PRECONDITIONS",
        "SCENARIO-REPORT-6767-ROW-RECOMPUTATION",
        "SCENARIO-REPORT-6767-BRANCH-ISOLATION",
        "SCENARIO-REPORT-6767-VALIDATORS",
    } <= anchors
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    assert exp.INFERENCE_SUBSTRATE in section
    assert exp.RESULT_PATH.as_posix() in section


def test_scenario_report_6767_preconditions_list_all_tasks_and_missing_artifacts(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6767-PRECONDITIONS: expected IDs and deliverables stay visible."""

    planned, sources = current_inputs
    matrix = exp.build_artifact_matrix(REPO, planned, sources)

    assert [row["task_id"] for row in planned] == list(exp.EXPECTED_TASK_IDS)
    assert [row["manifest_task_id"] for row in planned] == list(exp.FULL_TASK_IDS)
    assert len(matrix) == 13
    assert {row["task_id"] for row in matrix if row["artifact_state"] == "missing"} == {
        "exp6757",
        "exp6759",
    }
    assert sum(row["artifact_state"] == "present" for row in matrix) == 10
    assert matrix[-1]["artifact_state"] == "current_synthesis"
    assert all(row["path"] for row in matrix)


def test_scenario_report_6767_recomputes_current_branch_headlines(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6767-ROW-RECOMPUTATION: current branch numbers replay from rows."""

    _planned, sources = current_inputs
    headlines = exp.recompute_headlines(sources)

    proof = headlines["proof_transport"]
    assert proof["lossless_reparse_rows"] == {"numerator": 216, "denominator": 216, "rate": 1.0}
    assert proof["pre_reparse_exact_valid"] == {"numerator": 0, "denominator": 216, "rate": 0.0}
    assert proof["post_reparse_exact_valid"] == {
        "numerator": 11,
        "denominator": 216,
        "rate": pytest.approx(11 / 216),
    }
    assert proof["exact_valid_delta_vs_pre_reparse"] == {
        "numerator": 11,
        "denominator": 216,
        "rate": pytest.approx(11 / 216),
    }
    assert proof["environment_grammar_targetable_rows"] == 21
    assert proof["comparative_ab_rows"] == exp.empty_metric("Exp6757 missing")
    assert proof["proof_transport_audit_ready"] is False

    repair = headlines["repair"]
    assert repair["heldout_reasoning_error_auroc"] is None
    assert repair["oracle_leakage_detected"] is None
    assert repair["diagnostic_panel_ready"] is False
    assert repair["repair_interval"] is None
    assert repair["harmful_flips"] == {"numerator": 0, "denominator": 0, "rate": None}

    memory = headlines["continuous_memory"]
    assert memory["stream_orders"] == 6
    assert memory["stream_accept_opportunities"] == 72
    assert memory["stream_reject_opportunities"] == 72
    assert memory["transaction_activity"] == {"commits": 0, "rejects": 0, "rollbacks": 0}
    assert memory["prospective_rows"] == 0
    assert memory["procedural_over_no_memory_order_lcb"] == 0.0
    assert memory["procedural_over_trace_order_lcb"] == 0.0
    assert memory["cold_audit_completed"] is False

    arc = headlines["arc"]
    assert arc["preflight_ready"] is True
    assert arc["ab_science_rows"] == {"numerator": 0, "denominator": 120, "rate": 0.0}
    assert arc["live_quality_rows"] == {"numerator": 0, "denominator": 120, "rate": 0.0}
    assert arc["mean_prompt_token_savings"] is None
    assert arc["change_fidelity_delta"] is None
    assert arc["noninferiority_passed"] is False
    assert arc["solve_claim"] is False

    stochastic = headlines["stochastic_portability"]
    assert stochastic["exact_rows"] == {"numerator": 192, "denominator": 192, "rate": 1.0}
    assert stochastic["mean_trajectory_tv_by_method"]["independent_factor"]["value"] == pytest.approx(
        0.31752594692265373
    )
    assert stochastic["mean_trajectory_tv_by_method"]["context_matched"]["value"] == pytest.approx(
        0.2594914781375938
    )
    assert stochastic["paired_all_depth_deltas"]["context_matched"]["ci95_low"] == pytest.approx(
        0.0396871197471805
    )
    assert stochastic["evaluator_distinct"] is True
    assert stochastic["verifier_is_oracle"] is True


def test_scenario_report_6767_branch_classes_are_isolated(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6767-BRANCH-ISOLATION: no branch votes for another branch."""

    planned, sources = current_inputs
    rows = exp.build_rows(REPO, planned, sources, summary_findings=[])
    headlines = exp.recompute_headlines(sources)
    branch_rows = exp.build_branch_rows(rows, headlines, [], [], [])

    assert {row["branch"]: row["verdict_class"] for row in branch_rows} == {
        "proof_transport": "partial",
        "repair": "blocked",
        "continuous_memory": "blocked",
        "arc": "blocked",
        "stochastic_portability": "circular_positive",
        "infrastructure": "partial",
    }
    assert all("next_action" in row and row["next_action"] for row in branch_rows)
    assert headlines["pooled_milestone_success_score"] is None
    assert headlines["pooled_success_claim_emitted"] is False


def test_scenario_report_6767_complete_artifact_preserves_required_fields(
    current_inputs: tuple[list[dict], dict[str, dict]],
) -> None:
    """SCENARIO-REPORT-6767-BRANCH-ISOLATION: the terminal artifact has no pooled claim."""

    planned, sources = current_inputs
    artifact = exp.build_artifact(
        REPO,
        duration_s=0.25,
        planned=planned,
        sources=sources,
        summary_findings=[],
        adversarial_findings=[],
        row_consistency_findings=[],
        recurring_blockers=exp.recurring_blocker_placeholder(),
    )

    assert exp.validate_artifact(artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact) == set(artifact["field_principles"])
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["expected_task_ids"] == list(exp.FULL_TASK_IDS)
    assert [row["branch"] for row in artifact["branch_rows"]] == list(exp.BRANCH_ORDER)
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("complete_partial:")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["docs_reconciled"]["reconciled"] is False
    assert artifact["row_recomputed_headlines"]["pooled_success_claim_emitted"] is False
    assert artifact["fr11_disposition"]["positive"] is False
    assert artifact["fr12_disposition"]["transport"]["disposition"] == "partial"
    assert artifact["live_hardware_disposition"]["arc"]["disposition"] == "blocked"
    assert {row["disposition"] for row in artifact["prd_gap_disposition"]} == {
        "narrowed",
        "blocked",
    }
    assert any(
        row["same_verdict_condition_fired"] and row["prior_experiment_id"] == "exp6754-v588-branch-disposition"
        for row in artifact["prior_verdict_recurrences"]
    )
    assert artifact["retirement_recommendations"]
    protected = {row["path"]: row for row in artifact["protected_files_unchanged"]}
    assert protected["scripts/research_conductor.py"]["unchanged"] is True
    assert protected["research-roadmap.yaml"]["unchanged"] is True


def test_scenario_report_6767_validator_findings_are_preserved(
    current_inputs: tuple[list[dict], dict[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6767-VALIDATORS: warnings and row blocks stay explicit."""

    _planned, sources = current_inputs
    calls: list[list[str]] = []

    def fake_run(args: list[str], _root: Path) -> tuple[int, str]:
        calls.append(args)
        joined = " ".join(args)
        if "summarize_artifact.py" in joined:
            return 1, "summary warning"
        if "adversarial_verify.py" in joined:
            return (
                1,
                json.dumps(
                    {
                        "reports": [
                            {
                                "artifact": args[-1],
                                "flags": [
                                    {
                                        "kind": "SUBSTRATE_HAS_NO_DURATION_FLOOR",
                                        "severity": "warn",
                                        "detail": "fixture",
                                    }
                                ],
                            }
                        ]
                    }
                ),
            )
        if "verdict_row_consistency_lint.py" in joined:
            return 1, "  [BLOCK] ALL_ROWS_NULL: fixture\n  [warn ] NO_HEADROOM_MAJORITY: fixture"
        if "recurring_blocker_ledger.py" in joined:
            return 0, "  RECURRING (2+ times):\n    x27  blocked_gate_check_failed"
        raise AssertionError(args)

    monkeypatch.setattr(exp, "_run_command", fake_run)
    one_source = {task: exp.missing_source_record(task, "missing") for task in exp.EXPECTED_TASK_IDS}
    one_source["exp6755"] = sources["exp6755"]
    one_source[exp.CAPSTONE_TASK_ID] = sources[exp.CAPSTONE_TASK_ID]

    summaries = exp.run_summarizers(REPO, one_source)
    adversarial = exp.run_adversarial_findings(REPO, one_source)
    row_lint = exp.run_row_consistency_findings(REPO, one_source)
    blockers = exp.run_recurring_blockers(REPO)

    assert summaries[0]["exit_code"] == 1
    assert adversarial[0]["findings"][0]["kind"] == "SUBSTRATE_HAS_NO_DURATION_FLOOR"
    assert row_lint[0]["blocking_count"] == 1
    assert row_lint[0]["warning_count"] == 1
    assert blockers["recurring"] == ["x27  blocked_gate_check_failed"]
    assert len(calls) == 4


def test_scenario_report_6767_validator_parsers_ignore_nonfinding_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6767-VALIDATORS: parser noise is not promoted to findings."""

    assert exp._parse_row_lint_findings("plain line\n[BLOCK] real\ntrailer") == ["[BLOCK] real"]

    def fake_ledger(_args: list[str], _root: Path) -> tuple[int, str]:
        return 0, "  RECURRING (2+ times):\n    not-a-count\n    x2  blocked_gate"

    monkeypatch.setattr(exp, "_run_command", fake_ledger)
    assert exp.run_recurring_blockers(REPO)["recurring"] == ["x2  blocked_gate"]

    wrapper_path = REPO / "scripts/experiments/experiment_6767_v589_branch_disposition.py"
    saved_path = list(sys.path)
    saved_module = sys.modules.get("carnot.experiment_6767_v589_branch_disposition")
    sys.modules["carnot.experiment_6767_v589_branch_disposition"] = exp
    for path in (REPO, REPO / "python"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("exp6767_wrapper_present", wrapper_path)
    assert spec and spec.loader
    wrapper = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(wrapper)
    finally:
        sys.path = saved_path
        if saved_module is None:
            sys.modules.pop("carnot.experiment_6767_v589_branch_disposition", None)
        else:
            sys.modules["carnot.experiment_6767_v589_branch_disposition"] = saved_module
    assert wrapper.main(["--validate", "--output", str(tmp_path / "missing.json")]) == 1


def test_req_report_6767_missing_or_mutated_inputs_fail_closed(
    current_inputs: tuple[list[dict], dict[str, dict]],
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6767: invalid inputs stay blocked, missing, or rejected."""

    planned, sources = current_inputs
    artifact = exp.build_artifact(
        REPO,
        duration_s=0.25,
        planned=planned,
        sources=sources,
        summary_findings=[],
        adversarial_findings=[],
        row_consistency_findings=[],
        recurring_blockers=exp.recurring_blocker_placeholder(),
    )

    for field, bad in (
        ("verdict_class", "positive"),
        ("verifier_is_oracle", True),
        ("expected_task_ids", []),
        ("branch_rows", []),
        ("rows", []),
        ("field_principles", {}),
        ("reproducibility_checksum", "sha256:bad"),
    ):
        changed = deepcopy(artifact)
        changed[field] = bad
        if field != "reproducibility_checksum":
            changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert exp.validate_artifact(changed)

    changed_sources = deepcopy(sources)
    changed_sources["exp6766"]["payload"]["paired_trajectory_deltas"] = []
    changed_headlines = exp.recompute_headlines(changed_sources)
    assert changed_headlines["stochastic_portability"]["paired_all_depth_deltas"] == {}

    assert exp.short_task_id("exp6767-v589-branch-disposition") == "exp6767"
    with pytest.raises(ValueError, match="invalid V589 task id"):
        exp.short_task_id("bad-task")
    with pytest.raises(ValueError, match="deliverable missing"):
        exp._next_deliverable(["### Exp 6767: Missing"], 0)
    with pytest.raises(ValueError, match="deliverable missing"):
        exp._next_deliverable(["### Exp 6766: Missing", "### Exp 6767: Next"], 0)
    with pytest.raises(ValueError, match="milestone missing"):
        exp.parse_design_tasks("# no milestone")
    assert exp.sha256_file(tmp_path / "missing.json") is None
    assert exp._get_path({"a": {"b": 1}}, "a.b") == 1
    assert exp._get_path({"a": {}}, "a.c") is None
    assert exp.gate_failures("exp1", 99) == []
    assert exp.gate_failures("exp1", "blocked") == [
        {
            "task_id": "exp1",
            "check": "gate_check_summary",
            "expected": "gate predicate passes",
            "observed": "blocked",
            "passed": False,
            "reason": "blocked_gate_text",
        }
    ]
    assert exp.gate_failures(
        "exp2",
        {"failed_check": "lease", "expected": True, "observed": False},
    ) == [
        {
            "task_id": "exp2",
            "check": "lease",
            "expected": True,
            "observed": False,
            "passed": False,
            "reason": "failed_check_summary",
        }
    ]


def test_req_report_6767_defensive_classification_paths() -> None:
    """REQ-REPORT-6767: fallback classifiers remain closed and auditable."""

    assert exp._record_class({"artifact_state": "invalid"}) == "disqualified"
    for text, expected in (
        ("circular simulator win", "circular_positive"),
        ("disqualified retired scope", "disqualified"),
        ("partial handoff", "partial"),
        ("positive ready", "positive"),
        ("complete null", "null"),
        ("unknown", "disqualified"),
    ):
        assert exp._record_class({"artifact_state": "present", "payload": {"status": text}}) == expected

    assert exp._model_ids(
        {
            "models_used": [{"model_id": "a"}, {"hf_id": "b"}, {"id": "c"}, "d"],
            "rows": [
                {"model": {"hf_id": "e"}},
                {"model": {"model_id": "f"}},
                {"model_id": "g"},
            ],
        }
    ) == ["a", "b", "c", "d", "e", "f", "g"]
    assert exp._sum_mapping_numbers(None) == 0
    assert (
        exp.build_prior_verdict_recurrences(
            [{"task_id": exp.CAPSTONE_TASK_ID, "prior_failures": ["not-a-mapping"]}],
            "complete_partial: unchanged",
        )
        == []
    )

    headlines = {
        "proof_transport": {"transport_reparse_ready": True},
        "continuous_memory": {
            "prospective_csl_completed": True,
            "cold_audit_completed": True,
            "procedural_over_no_memory_order_lcb": 0.1,
            "procedural_over_trace_order_lcb": 0.2,
        },
        "arc": {"object_table_ab_completed": True, "adoption_gate_passed": False},
        "stochastic_portability": {
            "verifier_is_oracle": False,
            "independent_trajectory_audit_completed": True,
            "evaluator_distinct": True,
        },
    }
    with pytest.raises(ValueError, match="unknown branch"):
        exp._class_for_branch("missing", [], headlines, [])
    assert exp._class_for_branch("proof_transport", ["disqualified"], headlines, []) == "disqualified"
    assert exp._class_for_branch("continuous_memory", ["positive"], headlines, []) == "positive"
    changed = deepcopy(headlines)
    changed["continuous_memory"]["procedural_over_trace_order_lcb"] = 0.0
    assert exp._class_for_branch("continuous_memory", ["positive"], changed, []) == "null"
    assert exp._class_for_branch("arc", ["positive"], headlines, []) == "null"
    changed = deepcopy(headlines)
    changed["arc"]["adoption_gate_passed"] = True
    assert exp._class_for_branch("arc", ["positive"], changed, []) == "positive"
    assert exp._class_for_branch("stochastic_portability", ["positive"], headlines, []) == "positive"
    changed = deepcopy(headlines)
    changed["stochastic_portability"]["independent_trajectory_audit_completed"] = False
    assert exp._class_for_branch("stochastic_portability", ["blocked"], changed, []) == "blocked"
    assert exp._class_for_branch("stochastic_portability", ["null"], changed, []) == "null"


def test_req_report_6767_synthetic_nonblocked_row_recomputes() -> None:
    """SCENARIO-REPORT-6767-ROW-RECOMPUTATION: nonblocked rows compute directly."""

    arc = exp.recompute_arc(
        {
            "exp6764": _present(
                {
                    "arc_exclusive_load_ready": True,
                    "rows": [{"model": {"hf_id": "qwen"}}],
                    "vram_recovery_receipts": [{"passed": True}],
                }
            ),
            "exp6765": _present(
                {
                    "object_table_ab_completed": True,
                    "adoption_gate_passed": True,
                    "change_fidelity_interval": {"lower": -0.01, "upper": 0.05},
                    "noninferiority_margin": 0.02,
                    "solve_claim": True,
                    "rows": [
                        {
                            "row_kind": "science",
                            "arm": "table_inline",
                            "prompt_tokens": 100,
                            "change_fidelity": 0.90,
                            "failure_class": None,
                            "live_model_invoked": True,
                        },
                        {
                            "row_kind": "science",
                            "arm": "fetch_on_demand",
                            "prompt_tokens": 80,
                            "change_fidelity": 0.92,
                            "failure_class": None,
                            "live_model_invoked": True,
                        },
                    ],
                }
            ),
        }
    )

    assert arc["ab_science_rows"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert arc["live_quality_rows"] == {"numerator": 2, "denominator": 2, "rate": 1.0}
    assert arc["mean_prompt_token_savings"] == 20.0
    assert arc["change_fidelity_by_arm"]["fetch_on_demand"]["value"] == 0.92
    assert arc["noninferiority_passed"] is True
    assert arc["solve_claim"] is True

    paired = exp._paired_trajectory_deltas(
        [{"evaluator_path": "direct_sampler", "method": "independent_factor", "trajectory_tv": 1.0}]
    )
    assert paired["context_matched"]["pair_count"] == 0


def test_req_report_6767_sparse_row_shapes_stay_explicit() -> None:
    """REQ-REPORT-6767: sparse rows stay local instead of being repaired by prose."""

    assert exp._model_ids({"models_used": [{}, None, ""], "rows": [{"model": {}}]}) == []

    arc = exp.recompute_arc(
        {
            "exp6764": _present({"arc_exclusive_load_ready": True}),
            "exp6765": _present(
                {
                    "object_table_ab_completed": True,
                    "rows": [
                        {
                            "row_kind": "science",
                            "arm": "table_inline",
                            "prompt_tokens": "100",
                            "change_fidelity": "0.90",
                            "failure_class": None,
                            "live_model_invoked": False,
                        }
                    ],
                }
            ),
        }
    )
    assert arc["ab_science_rows"] == {"numerator": 1, "denominator": 1, "rate": 1.0}
    assert arc["mean_prompt_token_savings"] is None
    assert arc["change_fidelity_by_arm"] == {}

    paired = exp._paired_trajectory_deltas(
        [
            {
                "evaluator_path": "exact_enumerator",
                "method": "independent_factor",
                "trajectory_tv": "unknown",
            },
            {
                "evaluator_path": "exact_enumerator",
                "method": "independent_factor",
                "trajectory_tv": 0.4,
                "factor_id": "f1",
            },
            {
                "evaluator_path": "exact_enumerator",
                "method": "context_matched",
                "trajectory_tv": 0.3,
                "factor_id": "other",
            },
        ]
    )
    assert paired["context_matched"]["pair_count"] == 0

    stochastic = exp.recompute_stochastic(
        {
            "exp6766": _present(
                {
                    "rows": [
                        {
                            "evaluator_path": "exact_enumerator",
                            "method": "independent_factor",
                            "trajectory_tv": "unknown",
                            "conditional_kl": 0.2,
                        },
                        {
                            "evaluator_path": "exact_enumerator",
                            "method": "context_matched",
                            "trajectory_tv": 0.1,
                            "conditional_kl": "unknown",
                        },
                    ]
                }
            )
        }
    )
    assert stochastic["mean_trajectory_tv_by_method"]["context_matched"]["value"] == 0.1
    assert stochastic["mean_conditional_kl_by_method"]["independent_factor"]["value"] == 0.2


def test_scenario_report_6767_validator_and_validation_edges(
    current_inputs: tuple[list[dict], dict[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6767-VALIDATORS: parser and schema failures are retained."""

    planned, sources = current_inputs
    artifact = exp.build_artifact(
        REPO,
        duration_s=0.25,
        planned=planned,
        sources=sources,
        summary_findings=[],
        adversarial_findings=[],
        row_consistency_findings=[],
        recurring_blockers=exp.recurring_blocker_placeholder(),
    )
    bad_branch = deepcopy(artifact)
    bad_branch["branch_rows"][0]["verdict_class"] = "mystery"
    bad_branch["reproducibility_checksum"] = exp.reproducibility_checksum(bad_branch)
    assert "branch_rows_closed_class" in exp.validate_artifact(bad_branch)

    bad_pool = deepcopy(artifact)
    bad_pool["row_recomputed_headlines"]["pooled_success_claim_emitted"] = True
    bad_pool["row_recomputed_headlines"]["pooled_milestone_success_score"] = 0.5
    bad_pool["reproducibility_checksum"] = exp.reproducibility_checksum(bad_pool)
    errors = exp.validate_artifact(bad_pool)
    assert "pooled_success_claim_emitted" in errors
    assert "pooled_milestone_success_score" in errors

    def bad_json_run(_args: list[str], _root: Path) -> tuple[int, str]:
        return 1, "not json"

    monkeypatch.setattr(exp, "_run_command", bad_json_run)
    one_source = {task: exp.missing_source_record(task, "missing") for task in exp.EXPECTED_TASK_IDS}
    one_source["exp6755"] = sources["exp6755"]
    assert exp.run_adversarial_findings(REPO, one_source)[0]["flag_count"] == 0

    monkeypatch.setattr(exp, "run_summarizers", lambda *_args: [])
    monkeypatch.setattr(exp, "run_adversarial_findings", lambda *_args: [])
    monkeypatch.setattr(exp, "run_row_consistency_findings", lambda *_args: [])
    monkeypatch.setattr(exp, "run_recurring_blockers", lambda *_args: exp.recurring_blocker_placeholder())
    assert exp.main(["--repo-root", str(REPO), "--output", str(tmp_path / "full.json")]) == 0

    monkeypatch.setattr(exp, "build_artifact", lambda *_args, **_kwargs: {"bad": "payload"})
    assert (
        exp.main(
            [
                "--repo-root",
                str(REPO),
                "--output",
                str(tmp_path / "invalid.json"),
                "--skip-external-checks",
            ]
        )
        == 1
    )


def _design_text(
    ids: tuple[str, ...] = exp.EXPECTED_TASK_IDS,
    milestone: str = exp.MILESTONE,
) -> str:
    lines = [f"**Milestone:** `{milestone}`"]
    for task_id in ids:
        lines.extend(
            [
                f"### Exp {task_id.removeprefix('exp')}: Synthetic {task_id}",
                f"**Deliverable:** `{exp.TASK_PATHS[task_id]}`",
            ]
        )
    return "\n".join(lines)


def _manifest(deliverable_override: tuple[str, str] | None = None) -> dict[str, object]:
    tasks = []
    for full_id in exp.FULL_TASK_IDS:
        task_id = exp.short_task_id(full_id)
        deliverable = exp.TASK_PATHS[task_id]
        if deliverable_override and deliverable_override[0] == task_id:
            deliverable = deliverable_override[1]
        tasks.append(
            {
                "id": full_id,
                "title": f"Synthetic {task_id}",
                "deliverable": deliverable,
                "prior_failures": [],
            }
        )
    return {"milestone": exp.MILESTONE, "tasks": tasks}


def _write_plan_root(root: Path, manifest: object, design: str) -> None:
    (root / exp.ACTIVE_ROADMAP_PATH).write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (root / exp.DESIGN_PATH.parent).mkdir(parents=True, exist_ok=True)
    (root / exp.DESIGN_PATH).write_text(design, encoding="utf-8")


def test_req_report_6767_plan_and_write_edges(tmp_path: Path) -> None:
    """REQ-REPORT-6767: malformed local plans and writes fail closed."""

    _write_plan_root(tmp_path, [], _design_text())
    with pytest.raises(ValueError, match="mapping with tasks"):
        exp.load_planned_tasks(tmp_path)
    _write_plan_root(tmp_path, _manifest(), _design_text(milestone="2026.08.999"))
    with pytest.raises(ValueError, match="expected V589 design"):
        exp.load_planned_tasks(tmp_path)
    _write_plan_root(tmp_path, _manifest(), _design_text(exp.EXPECTED_TASK_IDS[:-1]))
    with pytest.raises(ValueError, match="Exp6755 through Exp6767"):
        exp.load_planned_tasks(tmp_path)
    bad_manifest = _manifest()
    bad_manifest["tasks"] = bad_manifest["tasks"][:-1]
    _write_plan_root(tmp_path, bad_manifest, _design_text())
    with pytest.raises(ValueError, match="exact V589 task list"):
        exp.load_planned_tasks(tmp_path)
    _write_plan_root(tmp_path, _manifest(("exp6755", "results/wrong.json")), _design_text())
    with pytest.raises(ValueError, match="deliverable mismatch"):
        exp.load_planned_tasks(tmp_path)

    planned = [
        {"task_id": "exp6755", "path": "missing.json"},
        {"task_id": "exp6756", "path": "bad.json"},
        {"task_id": exp.CAPSTONE_TASK_ID, "path": exp.RESULT_PATH.as_posix()},
    ]
    (tmp_path / "bad.json").write_text("[]", encoding="utf-8")
    sources = exp.load_source_artifacts(tmp_path, planned)
    assert sources["exp6755"]["artifact_state"] == "missing"
    assert sources["exp6756"]["artifact_state"] == "invalid"
    assert sources[exp.CAPSTONE_TASK_ID]["artifact_state"] == "current_synthesis"

    target = tmp_path / "nested" / "artifact.json"
    artifact = exp.build_artifact(
        REPO,
        duration_s=0.25,
        summary_findings=[],
        adversarial_findings=[],
        row_consistency_findings=[],
        recurring_blockers=exp.recurring_blocker_placeholder(),
    )
    exp.write_json_atomic(target, artifact)
    assert exp.validate_artifact(json.loads(target.read_text(encoding="utf-8"))) == []
    target.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        exp._load_artifact(target)

    bad = deepcopy(artifact)
    bad["verdict_class"] = "positive"
    bad["reproducibility_checksum"] = exp.reproducibility_checksum(bad)
    with pytest.raises(ValueError, match="invalid Exp6767 artifact"):
        exp.write_json_atomic(tmp_path / "bad-artifact.json", bad)

    real_replace = exp.os.replace
    try:
        exp.os.replace = lambda *_args: (_ for _ in ()).throw(OSError("replace failed"))
        with pytest.raises(OSError, match="replace failed"):
            exp.write_json_atomic(tmp_path / "replace-fails.json", artifact)
    finally:
        exp.os.replace = real_replace
    assert not list(tmp_path.glob("*.tmp"))


def test_req_report_6767_cli_and_wrapper(tmp_path: Path) -> None:
    """REQ-REPORT-6767: CLI and script wrapper publish valid JSON."""

    target = tmp_path / "artifact.json"
    assert (
        exp.main(
            [
                "--repo-root",
                str(REPO),
                "--output",
                str(target),
                "--skip-external-checks",
            ]
        )
        == 0
    )
    assert exp.main(["--validate", "--output", str(target)]) == 0
    target.write_text(json.dumps({"bad": "payload"}), encoding="utf-8")
    assert exp.main(["--validate", "--output", str(target)]) == 1
    assert exp.main(["--validate", "--output", str(tmp_path / "missing.json")]) == 1

    wrapper_path = REPO / "scripts/experiments/experiment_6767_v589_branch_disposition.py"
    saved_path = list(sys.path)
    saved_carnot = sys.modules.get("carnot")
    saved_module = sys.modules.get("carnot.experiment_6767_v589_branch_disposition")
    fake_carnot = types.ModuleType("carnot")
    fake_carnot.__path__ = []
    fake_module = types.ModuleType("carnot.experiment_6767_v589_branch_disposition")
    fake_module.main = exp.main
    sys.modules["carnot"] = fake_carnot
    sys.modules["carnot.experiment_6767_v589_branch_disposition"] = fake_module
    for path in (REPO, REPO / "python"):
        while str(path) in sys.path:
            sys.path.remove(str(path))
    spec = importlib.util.spec_from_file_location("exp6767_wrapper", wrapper_path)
    assert spec and spec.loader
    wrapper = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(wrapper)
    finally:
        sys.path = saved_path
        if saved_carnot is None:
            sys.modules.pop("carnot", None)
        else:
            sys.modules["carnot"] = saved_carnot
        if saved_module is None:
            sys.modules.pop("carnot.experiment_6767_v589_branch_disposition", None)
        else:
            sys.modules["carnot.experiment_6767_v589_branch_disposition"] = saved_module
    assert (
        wrapper.main(
            [
                "--repo-root",
                str(REPO),
                "--output",
                str(tmp_path / "wrapper.json"),
                "--skip-external-checks",
            ]
        )
        == 0
    )
