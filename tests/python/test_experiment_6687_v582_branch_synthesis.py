"""Tests for the V582 branch synthesis.

Spec: REQ-REPORT-6687 and SCENARIO-REPORT-6687-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from carnot import experiment_6687_v582_branch_synthesis as mod


# The archived V582 task manifest (commit a8e0d917e5, milestone 2026.08.582).
# Pinned here because the live research-roadmap.yaml moves on after V582
# closes; before this pin, all evidence-based tests errored at setup once the
# next milestone activated (the 2026-08-27 stale-capstone incident shape).
V582_TASKS = (
    (
        "exp6674-v582-manifest-parity-contract",
        "V582 document-to-manifest parity contract",
        "results/experiment_6674_v582_manifest_parity_contract.json",
    ),
    (
        "exp6675-triggered-tail-scope-receipt",
        "Triggered-tail task-owned verification receipt",
        "results/experiment_6675_triggered_tail_scope_receipt.json",
    ),
    (
        "exp6676-three-family-triggered-tail-ab",
        "Three-family delayed syntax-tail A/B",
        "results/experiment_6676_three_family_triggered_tail_ab.json",
    ),
    (
        "exp6677-triggered-tail-independent-audit",
        "Delayed syntax-tail blinded row audit",
        "results/experiment_6677_triggered_tail_independent_audit.json",
    ),
    (
        "exp6678-constraint-family-stream",
        "Independent constraint-family prequential stream",
        "results/experiment_6678_constraint_family_stream.json",
    ),
    (
        "exp6679-prequential-cross-family-csl-ab",
        "Prequential cross-family continuous self-learning A/B",
        "results/experiment_6679_prequential_cross_family_csl_ab.json",
    ),
    (
        "exp6680-csl-durability-audit",
        "Cross-family CSL chronological durability audit",
        "results/experiment_6680_csl_durability_audit.json",
    ),
    (
        "exp6681-arc-post-redirect-outcomes",
        "Canonical ARC post-redirect outcome transport",
        "results/experiment_6681_arc_post_redirect_outcomes.json",
    ),
    (
        "exp6682-arc-held-family-supervisor-ab",
        "Held-family ARC supervisor outcome A/B",
        "results/experiment_6682_arc_held_family_supervisor_ab.json",
    ),
    (
        "exp6683-ising-reference-scope-receipt",
        "Bounded-treewidth Ising task-owned receipt",
        "results/experiment_6683_ising_reference_scope_receipt.json",
    ),
    (
        "exp6684-torx-typed-factor-parity",
        "Torx energy-distribution conformance",
        "results/experiment_6684_torx_typed_factor_parity.json",
    ),
    (
        "exp6685-autocorrelation-schedule-ab",
        "Autocorrelation-aware stochastic schedule A/B",
        "results/experiment_6685_autocorrelation_schedule_ab.json",
    ),
    (
        "exp6686-stochastic-portability-audit",
        "Cold exact and raw-chain review",
        "results/experiment_6686_stochastic_portability_audit.json",
    ),
    (
        "exp6687-v582-branch-synthesis",
        "V582 five-branch disposition",
        "results/experiment_6687_v582_branch_synthesis.json",
    ),
)


@pytest.fixture(scope="module")
def evidence(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[dict], dict[str, dict | None], dict[str, dict]]:
    manifest_root = tmp_path_factory.mktemp("exp6687-v582-manifest")
    manifest = {
        "milestone": "2026.08.582",
        "tasks": [
            {"id": task_id, "title": title, "deliverable": deliverable}
            for task_id, title, deliverable in V582_TASKS
        ],
    }
    (manifest_root / mod.ACTIVE_ROADMAP_PATH.name).write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )
    planned = mod.load_planned_tasks(manifest_root)
    sources = mod.load_source_artifacts(mod.REPO_ROOT, planned)
    conductor = mod.load_conductor_states(mod.REPO_ROOT, planned)
    return planned, sources, conductor


def test_req_report_6687_spec_precedes_implementation() -> None:
    text = (mod.REPO_ROOT / mod.REPORT_SPEC_PATH).read_text(encoding="utf-8")
    anchors = set(mod.spec_anchors(text))
    assert {
        "REQ-REPORT-6687",
        "SCENARIO-REPORT-6687-TERMINAL-TASKS",
        "SCENARIO-REPORT-6687-ROW-RECOMPUTATION",
        "SCENARIO-REPORT-6687-BRANCH-CLASSIFICATION",
        "SCENARIO-REPORT-6687-VALIDATION",
        "SCENARIO-REPORT-6687-ATOMIC-PROTECTION",
    } <= anchors


def test_scenario_report_6687_terminal_tasks_preserve_missing_states(
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
) -> None:
    planned, sources, conductor = evidence
    rows = mod.build_terminal_task_rows(mod.REPO_ROOT, planned, sources, conductor)

    assert len(planned) == len(rows) == 14
    assert [row["order"] for row in rows] == list(range(1, 15))
    assert [row["task_id"] for row in rows] == mod.PLANNED_TASK_IDS
    missing = [row for row in rows if row["artifact_state"] == "missing"]
    assert [row["experiment_number"] for row in missing] == [6680, 6686]
    assert all(row["terminal_source"] == "conductor" for row in missing)
    assert all(row["verdict_class"] == "blocked" for row in missing)
    assert rows[-1]["terminal_source"] == "current_synthesis"
    assert rows[-1]["deliverable_hash"] is None


def test_scenario_report_6687_output_rows_do_not_coerce_missing_to_zero(
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
) -> None:
    _, sources, _ = evidence
    branch = mod.recompute_output_transport(sources)

    assert branch["verdict_class"] == "blocked"
    for arm in mod.OUTPUT_ARMS:
        assert branch["exact_success"][arm]["successes"] == 0
        assert branch["exact_success"][arm]["denominator"] == 0
        assert branch["exact_success"][arm]["value"] is None
        assert branch["exact_success"][arm]["state"] == "missing"
        assert branch["exact_success"][arm]["cause"]
        assert branch["parse_yield"][arm]["value"] is None
    assert branch["harmful_flips"]["denominator"] == 0
    assert branch["harmful_flips"]["value"] is None
    assert branch["audit"]["state"] == "blocked"


def test_req_report_6687_row_reducers_measure_available_evidence(
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
) -> None:
    _, sources, _ = evidence
    csl = mod.recompute_continuous_self_learning(sources)
    arc = mod.recompute_live_arc(sources)
    stochastic = mod.recompute_stochastic_portability(sources)

    assert csl["future_yield"]["value"] is None
    assert csl["order_level_intervals"]["value"] is None
    assert csl["restart"]["passed"] == csl["restart"]["denominator"] == 4
    assert csl["rollback"]["passed"] == csl["rollback"]["denominator"] == 4
    assert csl["audit"]["state"] == "missing"

    assert arc["transport"]["eligible_outcomes"] == 30
    assert arc["utility"]["denominator"] == 9
    assert arc["utility"]["delta"] == pytest.approx(-1 / 3)
    assert arc["false_intervention"]["count"] == 9
    assert arc["false_intervention"]["delta"] == pytest.approx(1.0)
    assert arc["forbidden_action"]["delta"] == 0.0
    assert arc["forbidden_action"]["no_headroom_rows"] == 9
    assert arc["solve_claim"] is False

    assert stochastic["exact_reference"]["state_count"] == 294
    assert stochastic["exact_reference"]["maximum_probability_error"] == pytest.approx(
        1.1102230246251565e-16
    )
    assert stochastic["torx_parity"]["state_count"] == 294
    assert stochastic["torx_parity"]["maximum_factor_energy_error"] == 0.0
    for name in ("likelihood_error", "acf", "iat", "ess"):
        assert stochastic[name]["value"] is None
        assert stochastic[name]["denominator"] == 0


def test_scenario_report_6687_branch_rows_are_independent(
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
) -> None:
    _, sources, _ = evidence
    branches = mod.build_branch_rows(sources)

    assert [row["branch"] for row in branches] == list(mod.BRANCH_ORDER)
    assert [row["verdict_class"] for row in branches] == [
        "null",
        "blocked",
        "blocked",
        "partial",
        "blocked",
    ]
    assert all(row["evidence"] and row["promotion_gate"] for row in branches)
    assert all(row["claim_boundary"] and row["exact_next_action"] for row in branches)
    assert not any(row.get("pooled_success") for row in branches)


def test_scenario_report_6687_validation_records_every_missing_input(
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
) -> None:
    planned, sources, conductor = evidence
    rows = mod.build_validation_rows(mod.REPO_ROOT, planned, sources, conductor)

    assert len(rows) == 13 * len(mod.VALIDATOR_NAMES)
    assert all(
        {"validator", "target", "exit", "finding", "severity", "hash"} <= set(row) for row in rows
    )
    missing = [row for row in rows if row["target_experiment"] in {6680, 6686}]
    assert len(missing) == 2 * len(mod.VALIDATOR_NAMES)
    assert all(row["exit"] is None and row["severity"] == "blocked" for row in missing)
    class_warnings = [
        row for row in rows if row["validator"] == "verdict_class_consistency" and row["exit"] == 1
    ]
    assert {row["target_experiment"] for row in class_warnings} == {6677, 6679, 6683, 6685}


def test_req_report_6687_complete_artifact_recomputes_from_rows(
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
) -> None:
    planned, sources, conductor = evidence
    before = mod.protected_hashes(mod.REPO_ROOT)
    artifact = mod.build_artifact(
        root=mod.REPO_ROOT,
        date=mod.RUN_DATE,
        duration_s=0.25,
        planned=planned,
        sources=sources,
        conductor=conductor,
        validation_rows=mod.build_validation_rows(mod.REPO_ROOT, planned, sources, conductor),
        tests_run=[{"command": "fixture", "exit": 0, "summary": "passed"}],
        protected_before=before,
    )

    assert mod.validate_artifact(artifact) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    # REQ-CONDUCTOR-VERDICT-3: the finished synthesis declares null, not
    # partial (its partial declaration re-ran the completed 2,983s task).
    assert artifact["status"] == "complete_terminal_null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["verdict_class"] == "null"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] == mod.VERIFIER_BY_BRANCH
    assert artifact["verifier_is_oracle"]["mode"] == "mixed_by_branch"
    assert len(artifact["planned_task_rows"]) == 14
    assert len(artifact["terminal_task_rows"]) == 14
    assert len(artifact["per_unit_rows"]) == 14
    assert len(artifact["missing_artifact_rows"]) == 2
    assert artifact["aggregate_row_recomputation"]["pooled_success_claim"] is False
    assert artifact["protected_files_unchanged"]["all_unchanged"] is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("verdict_class", "positive"),
        # REQ-CONDUCTOR-VERDICT-3 / SCENARIO-CONDUCTOR-VERDICT-5: partial is
        # the may-retry class and a finished synthesis may not declare it.
        ("verdict_class", "partial"),
        ("inference_substrate", "llm"),
        ("verifier_is_oracle", True),
        ("branch_rows", []),
        ("missing_artifact_rows", []),
        ("reproducibility_checksum", "sha256:bad"),
    ],
)
def test_scenario_report_6687_validator_fails_closed(
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
    field: str,
    bad: object,
) -> None:
    planned, sources, conductor = evidence
    artifact = mod.build_artifact(
        root=mod.REPO_ROOT,
        date=mod.RUN_DATE,
        duration_s=0.25,
        planned=planned,
        sources=sources,
        conductor=conductor,
        validation_rows=mod.build_validation_rows(mod.REPO_ROOT, planned, sources, conductor),
        tests_run=[{"command": "fixture", "exit": 0, "summary": "passed"}],
        protected_before=mod.protected_hashes(mod.REPO_ROOT),
    )
    changed = deepcopy(artifact)
    changed[field] = bad
    assert mod.validate_artifact(changed)


def test_scenario_report_6687_atomic_write_and_validation(
    tmp_path: Path,
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
) -> None:
    planned, sources, conductor = evidence
    artifact = mod.build_artifact(
        root=mod.REPO_ROOT,
        date=mod.RUN_DATE,
        duration_s=0.25,
        planned=planned,
        sources=sources,
        conductor=conductor,
        validation_rows=mod.build_validation_rows(mod.REPO_ROOT, planned, sources, conductor),
        tests_run=[{"command": "fixture", "exit": 0, "summary": "passed"}],
        protected_before=mod.protected_hashes(mod.REPO_ROOT),
    )
    path = tmp_path / "nested" / "artifact.json"
    mod.atomic_write_json(path, artifact)

    assert json.loads(path.read_text(encoding="utf-8")) == artifact
    assert not path.with_suffix(".json.tmp").exists()
    assert mod.validate_path(path) == []


def test_req_report_6687_helpers_fail_closed(tmp_path: Path) -> None:
    bad_json = tmp_path / "list.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        mod.load_json(bad_json)
    with pytest.raises(ValueError, match="invalid task id"):
        mod.experiment_number("task-6687")

    (tmp_path / mod.ACTIVE_ROADMAP_PATH).write_text("tasks: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ordered Exp6674-Exp6687"):
        mod.load_planned_tasks(tmp_path)

    assert mod._task_class(1, {}) == "disqualified"
    assert mod._diagnostic({"blocked_reason": "reason"}, {}) == "reason"
    assert (
        mod._owner_validation(
            6677,
            {"status": "blocked", "honest_verdict": "blocked", "blocked_reason": "gate"},
        )
        == []
    )
    assert mod._owner_validation(6677, {}) == ["generic terminal artifact schema"]
    assert mod._claim_issues(
        6681,
        {
            "verifier_is_oracle": True,
            "verdict_class": "positive",
            "honest_verdict": "ARC solve",
        },
    ) == [
        "positive oracle result must be circular_positive",
        "unsupported ARC solve claim",
    ]
    assert mod._claim_issues(6684, {"honest_verdict": "hardware parity"}) == [
        "unsupported stochastic hardware claim"
    ]


def test_scenario_report_6687_external_validation_receipts_are_retained(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
) -> None:
    planned, sources, conductor = evidence
    assert mod._owner_validation(6674, sources["exp6674"] or {}) == []
    monkeypatch.setattr(mod, "_run_shell", lambda command, root: (7, f"failed: {command}"))
    monkeypatch.setattr(mod, "_owner_validation", lambda number, payload: ["owner issue"])
    rows = mod.build_validation_rows(
        mod.REPO_ROOT,
        [planned[0], planned[-1]],
        sources,
        conductor,
        run_external=True,
    )

    by_name = {row["validator"]: row for row in rows}
    assert by_name["row_consistency"]["exit"] == 7
    assert by_name["adversarial_verification"]["exit"] == 7
    assert by_name["artifact_validation"]["exit"] == 1
    assert "owner issue" in by_name["artifact_validation"]["finding"]


def test_scenario_report_6687_validator_covers_terminal_inconsistencies(
    evidence: tuple[list[dict], dict[str, dict | None], dict[str, dict]],
    tmp_path: Path,
) -> None:
    planned, sources, conductor = evidence
    artifact = mod.build_artifact(
        root=mod.REPO_ROOT,
        date=mod.RUN_DATE,
        duration_s=0.25,
        planned=planned,
        sources=sources,
        conductor=conductor,
        validation_rows=mod.build_validation_rows(mod.REPO_ROOT, planned, sources, conductor),
        tests_run=[{"command": "fixture", "exit": 0, "summary": "passed"}],
        protected_before=mod.protected_hashes(mod.REPO_ROOT),
    )

    mutations = []
    missing = deepcopy(artifact)
    missing.pop("status")
    mutations.append(missing)
    for field, value in (
        ("status", "blocked"),
        ("honest_verdict", "bad"),
        ("planned_task_rows", []),
        ("per_unit_rows", []),
        ("field_provenance", {}),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        mutations.append(changed)
    changed = deepcopy(artifact)
    changed["branch_rows"][0]["verdict_class"] = "positive"
    mutations.append(changed)
    changed = deepcopy(artifact)
    changed["aggregate_row_recomputation"]["pooled_success_claim"] = True
    mutations.append(changed)
    changed = deepcopy(artifact)
    changed["protected_files_unchanged"]["all_unchanged"] = False
    mutations.append(changed)
    changed = deepcopy(artifact)
    changed["output_transport_branch"]["bad"] = {"denominator": 0, "value": 0}
    mutations.append(changed)

    assert all(mod.validate_artifact(changed) for changed in mutations)

    # REQ-CONDUCTOR-VERDICT-3 / SCENARIO-CONDUCTOR-VERDICT-5: a finished
    # synthesis may not declare the may-retry class. The checksum is
    # recomputed so this pins the class rule itself, not the checksum check.
    partial_declared = deepcopy(artifact)
    partial_declared["verdict_class"] = "partial"
    partial_declared["reproducibility_checksum"] = mod._checksum(partial_declared)
    assert "verdict_class" in mod.validate_artifact(partial_declared)
    with pytest.raises(ValueError, match="invalid Exp6687 artifact"):
        mod.atomic_write_json(tmp_path / "bad.json", mutations[0])
