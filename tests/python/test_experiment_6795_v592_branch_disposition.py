"""Tests for REQ-CAPSTONE-6795 and its named scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.experiment_6795_v592_branch_disposition as disposition_module
from carnot.experiment_6795_v592_branch_disposition import (
    EXPECTED_TASK_IDS,
    REQUIRED_FIELDS,
    build_disposition,
    main,
    mean_ci95,
    recompute_csl,
    recompute_fixed_point,
    recompute_temporal_exchange,
)


REPO = Path(__file__).resolve().parents[2]


def _fixed_row(arm: str, key: str, valid: int) -> dict:
    return {
        "arm": arm,
        "paired_key": key,
        "split": "held_topology_test",
        "topology_family": "cycle",
        "candidate_budget": 3,
        "parameter_count": 10,
        "optimizer_update_count": 6,
        "stop_reason": "iteration_cap",
        "runtime_s": 0.1 if arm == "grouped_fixed_point" else 0.05,
        "exact_outcomes": [
            {
                "candidate_hash": f"{arm}-{key}-{index}",
                "exact_valid": index < valid,
                "dependency_violation_count": 0 if index < valid else 1,
                "distance_to_nearest_valid": 0 if index < valid else 1,
            }
            for index in range(3)
        ],
        "exact_evaluation_receipt": {
            "evaluated_after_proposal": True,
            "model_feedback_applied": False,
            "candidate_hashes_before": ["a", "b", "c"],
            "candidate_hashes_after": ["a", "b", "c"],
        },
    }


def _csl_rows() -> list[dict]:
    rows = []
    for order in range(1, 6):
        order_id = f"order_{order}"
        for arm, utility, selected in (
            ("compositional_online", 1.0, "learned"),
            ("frozen_controller", 0.0, "baseline"),
            ("random_update_placebo", 0.2, "placebo"),
            ("retrieval_disabled_online", 0.1, "disabled"),
        ):
            rows.append(
                {
                    "arm": arm,
                    "order_id": order_id,
                    "held_future": True,
                    "route_utility": utility,
                    "memory_write_count": int(arm != "frozen_controller"),
                    "memory_read_count": int(
                        arm in {"compositional_online", "random_update_placebo"}
                    ),
                    "baseline_action": "baseline",
                    "selected_action": selected,
                    "transaction": {"committed": arm != "frozen_controller"},
                    "retrieved_factor_ids": ["factor"] if arm == "compositional_online" else [],
                }
            )
    return rows


def _temporal_rows(audit: bool = False) -> list[dict]:
    rows = []
    for seed in (1, 2, 3):
        for arm, efficiency, tv in (
            ("ordinary_gibbs", 0.5, 0.1),
            ("temporal_exchange", 0.4, 0.2),
            ("temporal_exchange_zero_coupling", 0.5, 0.1),
        ):
            row = {
                "arm": arm,
                "graph_id": "mixed_sparse_n8",
                "temperature": 0.75,
                "seed": seed,
                "target_total_variation": tv,
                "update_count": 100,
            }
            if audit:
                row.update(
                    row_kind="source_recomputation",
                    energy_effective_samples_per_attempted_update=efficiency,
                )
            else:
                row["effective_samples_per_update"] = efficiency
            rows.append(row)
    return rows


def test_mean_ci95_handles_singleton_and_variable_rows() -> None:
    """SCENARIO-CAPSTONE-6795-ROWS: measured bounds derive from row values."""
    assert mean_ci95([0.5]) == {"mean": 0.5, "lower": 0.5, "upper": 0.5, "n": 1}
    interval = mean_ci95([0.0, 1.0, 2.0])
    assert interval["mean"] == 1.0
    assert interval["lower"] < 1.0 < interval["upper"]
    assert mean_ci95([]) == {"mean": None, "lower": None, "upper": None, "n": 0}


def test_fixed_point_recomputation_requires_matched_non_oracle_rows() -> None:
    """REQ-CAPSTONE-6795: grouped fixed points must beat a matched control."""
    source_rows = []
    audit_rows = []
    for key in ("u1", "u2", "u3"):
        source_rows.extend(
            [
                _fixed_row("grouped_fixed_point", key, 3),
                _fixed_row("flat_recurrent_control", key, 1),
            ]
        )
        for row in source_rows[-2:]:
            audited = {**row, "row_type": "source_recompute"}
            audited.pop("exact_evaluation_receipt")
            audited["exact_checker_after_candidate_freeze"] = True
            audit_rows.append(audited)

    result = recompute_fixed_point(source_rows, audit_rows)

    assert result["evidence_authority"] == "independent_audit_rows"
    assert result["paired_exact_valid_delta"]["mean"] == 2 / 3
    assert result["paired_exact_valid_delta"]["lower"] > 0
    assert result["matched_parameter_counts"] is True
    assert result["matched_candidate_work"] is True
    assert result["oracle_leakage_free"] is True
    assert result["positive_gate"] is True

    source_result = recompute_fixed_point(
        source_rows
        + [_fixed_row("grouped_fixed_point", "orphan", 3)]
        + [{"arm": "irrelevant_control", "paired_key": "ignored"}],
        [],
    )
    assert source_result["evidence_authority"] == "source_rows"
    assert source_result["oracle_leakage_free"] is True


def test_csl_recomputation_stops_at_blocked_cold_audit() -> None:
    """SCENARIO-CAPSTONE-6795-BRANCHES: source lift cannot replace cold causality."""
    result = recompute_csl(
        _csl_rows(),
        {"verdict_class": "blocked", "csl_causal_audit_completed": False, "rows": []},
    )

    assert result["held_future_online_minus_frozen"]["lower"] > 0
    assert result["writes"] == 5
    assert result["later_reads"] == 5
    assert result["action_changes"] == 5
    assert result["prospective_causal_activity"] is True
    assert result["cold_causal_audit_passed"] is False
    assert result["promotion_gate"] is False

    incomplete_controls = recompute_csl(
        [
            {
                "arm": "compositional_online",
                "order_id": "online_only",
                "held_future": True,
                "route_utility": 1.0,
            }
        ],
        {},
    )
    assert incomplete_controls["order_effects"] == []
    assert incomplete_controls["held_future_online_minus_placebo"]["n"] == 0


def test_temporal_recomputation_uses_audit_and_rejects_failed_law() -> None:
    """SCENARIO-CAPSTONE-6795-BRANCHES: matched work also needs target-law fidelity."""
    result = recompute_temporal_exchange(_temporal_rows(), _temporal_rows(audit=True))

    assert result["evidence_authority"] == "independent_audit_rows"
    assert result["matched_attempted_updates"] is True
    assert result["efficiency_gate_passed"] is False
    assert result["target_law_gate_passed"] is False
    assert result["positive_gate"] is False

    incomplete = _temporal_rows() + [
        {
            "arm": "ordinary_gibbs",
            "graph_id": "orphan",
            "temperature": 1.0,
            "seed": 9,
            "effective_samples_per_update": 0.1,
        },
        {"arm": "irrelevant_control", "graph_id": "ignored"},
        {
            "arm": "temporal_exchange",
            "graph_id": "no_zero_control",
            "temperature": 1.0,
            "seed": 11,
            "effective_samples_per_update": 0.2,
            "update_count": 100,
        },
        {
            "arm": "ordinary_gibbs",
            "graph_id": "no_zero_control",
            "temperature": 1.0,
            "seed": 11,
            "effective_samples_per_update": 0.1,
            "update_count": 100,
        },
    ]
    assert recompute_temporal_exchange(incomplete, [])["evidence_authority"] == "source_rows"


def test_real_v592_build_has_complete_ungated_inventory() -> None:
    """REQ-CAPSTONE-6795: the real capstone reads every exact V592 task artifact."""
    artifact = build_disposition(REPO, "20260830")

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["expected_task_ids"] == list(EXPECTED_TASK_IDS)
    assert len(artifact["artifact_inventory"]) == len(EXPECTED_TASK_IDS)
    assert {row["artifact_state"] for row in artifact["artifact_inventory"]} == {"present"}
    assert artifact["fixed_point_disposition"]["verdict_class"] == "positive"
    assert artifact["csl_disposition"]["verdict_class"] == "partial"
    assert artifact["temporal_exchange_disposition"]["verdict_class"] == "null"
    assert artifact["pooled_score_computed"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["honest_verdict"].startswith("complete_partial:")
    assert artifact["prior_verdict_recurrences"][0]["repeated_outcome"] == "complete_partial"
    assert artifact["retirement_recommendations"][0]["proposed_exclusion_manifest_entry"]


def test_unreadable_design_emits_complete_blocked_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6795-PRECONDITIONS: unreadable design blocks honestly."""
    (tmp_path / "openspec/change-proposals").mkdir(parents=True)
    (tmp_path / "openspec/change-proposals/research-roadmap-vNEXT.md").write_text("")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.08.592\ntasks: []\n")

    artifact = build_disposition(tmp_path, "20260830")

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_v592_disposition"
    assert artifact["gate_check_summary"]["failed_checks"][0]["check"] == "v592_design_nonempty"


def test_invalid_roadmap_and_unreadable_sources_remain_data(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-6795-PRECONDITIONS: only design and active roadmap block."""
    design = tmp_path / "openspec/change-proposals/research-roadmap-vNEXT.md"
    design.parent.mkdir(parents=True)
    design.write_text("V592 design")
    (tmp_path / "research-roadmap.yaml").write_text("milestone: wrong\ntasks: []\n")
    blocked = build_disposition(tmp_path, "20260830")
    assert blocked["gate_check_summary"]["failed_checks"][0]["check"] == "v592_roadmap_mapping"

    (tmp_path / "research-roadmap.yaml").write_text(
        "milestone: 2026.08.592\ntasks:\n  - ignored\n  - id: noop\n"
        "    prior_failures:\n      - null\n      - retire_if_same_verdict: false\n"
    )
    first = tmp_path / disposition_module.TASK_PATHS[EXPECTED_TASK_IDS[0]]
    second = tmp_path / disposition_module.TASK_PATHS[EXPECTED_TASK_IDS[1]]
    first.parent.mkdir(parents=True)
    first.write_text("[]")
    second.write_text("{bad json")
    artifact = build_disposition(tmp_path, "20260830")
    assert artifact["artifact_inventory"][0]["artifact_state"] == "unreadable"
    assert artifact["artifact_inventory"][1]["artifact_state"] == "unreadable"
    assert artifact["artifact_inventory"][2]["artifact_state"] == "missing"
    assert disposition_module.sha256_file(tmp_path / "absent") is None


def test_disposition_and_validation_failure_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-CAPSTONE-6795: closed classes and atomic writes fail closed."""
    assert (
        disposition_module._csl_disposition({"promotion_gate": True})["verdict_class"] == "positive"
    )
    assert disposition_module._csl_disposition({})["verdict_class"] == "blocked"

    findings = disposition_module.validate_artifact({})
    assert set(findings) == {
        "required_fields",
        "verdict_class",
        "terminal_prefix",
        "pooled_score_computed",
        "verifier_is_oracle",
        "reproducibility_checksum",
    }

    target = tmp_path / "failed.json"
    monkeypatch.setattr(
        disposition_module.os, "replace", lambda *_args: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    with pytest.raises(RuntimeError, match="stop"):
        disposition_module.atomic_write_json(target, {"ok": True})
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(disposition_module, "build_disposition", lambda *_args: {})
    with pytest.raises(ValueError, match="invalid V592 disposition"):
        disposition_module.main(["--date", "20260830", "--repo", str(tmp_path)])


def test_main_writes_stable_json_to_requested_repo(tmp_path: Path) -> None:
    """REQ-CAPSTONE-6795: the CLI atomically writes the declared deliverable."""
    assert main(["--date", "20260830", "--repo", str(tmp_path)]) == 0
    output = tmp_path / "results/experiment_6795_v592_branch_disposition.json"
    artifact = json.loads(output.read_text())
    assert artifact["honest_verdict"] == "complete_blocked_v592_disposition"
    assert artifact["reproducibility_checksum"].startswith("sha256:")
