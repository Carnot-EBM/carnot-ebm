"""Tests for the Exp 2948 milestone .277 capstone.

Spec refs: REQ-REPORT-2948, SCENARIO-REPORT-2948.

The capstone synthesizes the three Deep Think Corrigenda outcomes
(exp2938 MMD, exp2939 same-schedule speedup, exp2940 code-corpus AUPRC),
the Paper-v6 Narrowing Discipline audit (exp2944), the Phase-4 VFE
firewall verification (exp2945), and the hardware-continuity outcomes
(exp2941 PolarFire, exp2942 KV260 n-scaling). These tests exercise the
synthesis logic over a synthetic temp-dir mirror of the real upstream
artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v277_2948 as exp2948


REQUIRED_FIELDS = {
    "honest_verdict",
    "milestone",
    "inference_substrate",
    "paper_ready",
    "clean_artifacts",
    "flagged_artifacts",
    "blocked_artifacts",
    "missing_artifacts",
    "artifact_classification_counts",
    "deep_think_corrigenda_outcomes",
    "paper_v6_safe_claims",
    "paper_v6_forbidden_claims",
    "narrowing_discipline_compliance_audit",
    "top_3_next_actions",
    "gaps_for_278",
    "cited_upstream_artifacts",
    "source_artifact_status",
    "field_principles",
    "no_new_llm_call",
    "no_new_hardware_run",
    "duration_s",
    "run_date",
}


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp2937_payload() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: archive_ready=true; archived_milestone=2026.05.276; "
            "activated_milestone=2026.05.277"
        ),
        "archive_ready": True,
    }


def _exp2938_payload(*, distinguishable: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_mmd_vs_cpu_sequential_gibbs_recorded",
        "distributions_distinguishable": distinguishable,
        "per_seed_mmd_pvalue": [0.000999, 0.000999, 0.000999],
        "paper_v6_recommendation": (
            "retract: distributions distinguishable at p<0.01; paper-v6 must "
            "retract the 'exact sampling on FPGA' claim and frame KV260 outputs "
            "as fixed-schedule heuristic samples."
        ),
    }


def _exp2939_payload(*, speedup: float = 0.98225) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_slower_than_same_schedule_cpu_at_n64",
        "kv260_speedup_vs_same_schedule_cpu": {
            "value": speedup,
            "unit": "cpu_synchronous_parallel_us_median / kv260_us_per_sample",
        },
        "cpu_synchronous_parallel_per_sample_us_median": 23.574,
        "kv260_per_sample_us_cited": 24.0,
        "paper_v6_recommendation": (
            "retract: KV260 is slower than the same-schedule CPU baseline at n=64; "
            "paper-v6 must retract the current speedup claim."
        ),
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "experiment_id == random_seed == 2939 (known false positive)",
            }
        ],
    }


def _exp2940_payload(*, auprc: float = 0.888889, recommendation: str = "retain") -> dict[str, Any]:
    return {
        "honest_verdict": "complete: verifier provides meaningful information on code corpora",
        "code_corpus_auprc": auprc,
        "code_corpus_baseline_random_auprc": {"value": 0.075},
        "paper_v6_recommendation": {"value": recommendation},
    }


def _exp2941_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: polarfire_500_clause_constraint_scorer_hash_verified",
    }


def _exp2942_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_fixed_n64_latency_profile_recorded",
    }


def _exp2943_payload(*, ready: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: matrix_v11_ready=true",
        "matrix_v11_ready": ready,
    }


def _exp2944_payload(*, all_resolved: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: paper_v6_narrowing_audit_recorded_matches",
        "per_file_hits": [
            {
                "file": "docs/arxiv-paper/main.tex",
                "line": 897,
                "matched_phrase": "Extropic Z1 as a near-term Phase-2 production target",
                "retracted_claim_id": "#7",
            },
            {
                "file": "docs/technical-report.md",
                "line": 69,
                "matched_phrase": "Hardware sovereignty",
                "retracted_claim_id": "#9",
            },
            {
                "file": "docs/technical-report.html",
                "line": 65,
                "matched_phrase": "Hardware sovereignty",
                "retracted_claim_id": "#9",
            },
        ],
        "audit_resolution_by_operator": [
            {
                "file": "docs/arxiv-paper/main.tex",
                "retracted_claim_id": "#7",
                "resolution": (
                    "false_positive_already_retracted_in_context"
                    if all_resolved
                    else "pending"
                ),
                "resolved_at": "2026-05-23",
                "operator_authorized": True,
            },
            {
                "file": "docs/technical-report.md",
                "retracted_claim_id": "#9",
                "resolution": (
                    "resolved_by_operator_narrowing_edit" if all_resolved else "pending"
                ),
                "resolved_at": "2026-05-23",
                "operator_authorized": True,
            },
            {
                "file": "docs/technical-report.html",
                "retracted_claim_id": "#9",
                "resolution": (
                    "resolved_by_operator_narrowing_edit" if all_resolved else "pending"
                ),
                "resolved_at": "2026-05-23",
                "operator_authorized": True,
            },
        ],
    }


def _exp2945_payload(*, n_violations: int = 0) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: phase4_vfe_firewall_no_violations",
        "n_violations": n_violations,
        "firewall_violations": [],
    }


def _exp2946_payload() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: retain continuation executed with n_tasks=50, "
            "pass@1=0.0600, pass@k=0.1600"
        ),
    }


def _exp2947_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: nonuniform_continuation_replay_curriculum_piloted",
    }


def _write_default_inputs(root: Path) -> None:
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2937"].path, _exp2937_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2938"].path, _exp2938_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2939"].path, _exp2939_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2940"].path, _exp2940_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2941"].path, _exp2941_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2942"].path, _exp2942_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2943"].path, _exp2943_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2944"].path, _exp2944_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2945"].path, _exp2945_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2946"].path, _exp2946_payload())
    _write_json(root, exp2948.EXPECTED_ARTIFACTS["exp2947"].path, _exp2947_payload())


def test_build_artifact_returns_all_required_fields(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2948: required schema fields are present."""

    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    missing = REQUIRED_FIELDS - set(artifact.keys())
    assert not missing, f"missing fields: {missing}"


def test_build_artifact_paper_ready_true_on_happy_path(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2948: paper_ready=true when the three corrigenda
    confirm the Deep Think narrowing AND the audit hits are resolved."""

    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    assert artifact["paper_ready"] is True
    assert artifact["milestone"] == "2026.05.277"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"].startswith("complete:")


def test_deep_think_corrigenda_outcomes_shape(tmp_path: Path) -> None:
    """Required shape: {mmd_distinguishable: bool, same_schedule_speedup: float,
    code_auprc_recommendation: str}."""

    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    outcomes = artifact["deep_think_corrigenda_outcomes"]
    assert outcomes["mmd_distinguishable"] is True
    assert pytest.approx(outcomes["same_schedule_speedup"], rel=1e-6) == 0.98225
    assert outcomes["code_auprc_recommendation"] == "retain"
    assert outcomes["headline_outcome"] == "narrow"


def test_paper_v6_safe_and_forbidden_claims_populated(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    safe = artifact["paper_v6_safe_claims"]
    forbidden = artifact["paper_v6_forbidden_claims"]
    assert isinstance(safe, list) and len(safe) >= 6
    assert isinstance(forbidden, list) and len(forbidden) >= 6
    # The retracted-claim numbers from CLAUDE.md Paper-v6 Narrowing Discipline
    # MUST be present so paper-v6 LaTeX-side narrowing can grep for them.
    forbidden_blob = "\n".join(forbidden)
    for marker in ("#2", "#3", "#7", "#8", "#9", "#10"):
        assert marker in forbidden_blob


def test_narrowing_audit_collapsed_per_file(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    audit = artifact["narrowing_discipline_compliance_audit"]
    assert isinstance(audit, list)
    files = sorted(row["file"] for row in audit)
    assert files == sorted(
        [
            "docs/arxiv-paper/main.tex",
            "docs/technical-report.html",
            "docs/technical-report.md",
        ]
    )
    for row in audit:
        assert row["hits"] == 1
        assert isinstance(row["fixes_applied"], list) and row["fixes_applied"]


def test_exp2939_tautology_false_positive_kept_clean(tmp_path: Path) -> None:
    """The exp2939 adversarial flag is a known TAUTOLOGY false-positive
    (experiment_id == random_seed by team convention). The capstone MUST
    classify it clean despite the linter flag."""

    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    assert "exp2939" in artifact["clean_artifacts"]
    assert "exp2939" not in artifact["flagged_artifacts"]


def test_paper_ready_false_when_matrix_v11_missing(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    _write_json(
        tmp_path,
        exp2948.EXPECTED_ARTIFACTS["exp2943"].path,
        _exp2943_payload(ready=False),
    )
    artifact = exp2948.build_artifact(tmp_path)
    assert artifact["paper_ready"] is False


def test_paper_ready_false_when_firewall_violations(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    _write_json(
        tmp_path,
        exp2948.EXPECTED_ARTIFACTS["exp2945"].path,
        _exp2945_payload(n_violations=2),
    )
    artifact = exp2948.build_artifact(tmp_path)
    assert artifact["paper_ready"] is False


def test_paper_ready_false_when_audit_still_pending(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    _write_json(
        tmp_path,
        exp2948.EXPECTED_ARTIFACTS["exp2944"].path,
        _exp2944_payload(all_resolved=False),
    )
    artifact = exp2948.build_artifact(tmp_path)
    assert artifact["paper_ready"] is False
    gaps = artifact["gaps_for_278"]
    assert any("pending operator resolution" in g for g in gaps)


def test_paper_ready_false_when_mmd_indistinguishable_breaks_narrow(tmp_path: Path) -> None:
    """If MMD is NOT distinguishable AND speedup < 1.0 AND AUPRC retain, the
    headline_outcome is 'additional_rounds_needed' (one corrigendum disagrees
    with the Deep Think narrowing). paper_ready must be False."""

    _write_default_inputs(tmp_path)
    _write_json(
        tmp_path,
        exp2948.EXPECTED_ARTIFACTS["exp2938"].path,
        _exp2938_payload(distinguishable=False),
    )
    artifact = exp2948.build_artifact(tmp_path)
    assert artifact["paper_ready"] is False
    assert artifact["deep_think_corrigenda_outcomes"]["headline_outcome"] == "additional_rounds_needed"


def test_missing_artifact_listed(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    # Delete one of the upstream artifacts.
    missing_path = tmp_path / exp2948.EXPECTED_ARTIFACTS["exp2941"].path
    missing_path.unlink()
    artifact = exp2948.build_artifact(tmp_path)
    assert "exp2941" in artifact["missing_artifacts"]
    assert "exp2941" not in artifact["clean_artifacts"]


def test_blocked_verdict_classified_blocked(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    blocked = dict(_exp2941_payload())
    blocked["honest_verdict"] = "blocked_polarfire_ssh_unreachable"
    _write_json(tmp_path, exp2948.EXPECTED_ARTIFACTS["exp2941"].path, blocked)
    artifact = exp2948.build_artifact(tmp_path)
    assert "exp2941" in artifact["blocked_artifacts"]


def test_flagged_when_not_in_false_positive_overrides(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    flagged = dict(_exp2942_payload())
    flagged["flagged_adversarial"] = True
    flagged["corrigendum_pending"] = [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}]
    _write_json(tmp_path, exp2948.EXPECTED_ARTIFACTS["exp2942"].path, flagged)
    artifact = exp2948.build_artifact(tmp_path)
    assert "exp2942" in artifact["flagged_artifacts"]


def test_flagged_via_audit_rerun_findings(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    flagged = dict(_exp2942_payload())
    flagged["adversarial_audit_rerun"] = {"flagged": True, "findings": ["x"]}
    _write_json(tmp_path, exp2948.EXPECTED_ARTIFACTS["exp2942"].path, flagged)
    artifact = exp2948.build_artifact(tmp_path)
    assert "exp2942" in artifact["flagged_artifacts"]


def test_flagged_via_adversarial_verify_passed_false(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    flagged = dict(_exp2942_payload())
    flagged["adversarial_verify_passed"] = False
    _write_json(tmp_path, exp2948.EXPECTED_ARTIFACTS["exp2942"].path, flagged)
    artifact = exp2948.build_artifact(tmp_path)
    assert "exp2942" in artifact["flagged_artifacts"]


def test_flagged_via_adversarial_verify_summary_count(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    flagged = dict(_exp2942_payload())
    flagged["adversarial_verify_summary"] = {"flag_count": 3}
    _write_json(tmp_path, exp2948.EXPECTED_ARTIFACTS["exp2942"].path, flagged)
    artifact = exp2948.build_artifact(tmp_path)
    assert "exp2942" in artifact["flagged_artifacts"]


def test_resolution_pending_is_non_terminal() -> None:
    assert exp2948._resolution_is_terminal({"resolution": "pending"}) is False
    assert exp2948._resolution_is_terminal({"resolution": "", "operator_authorized": True}) is False
    assert exp2948._resolution_is_terminal({}) is False
    assert exp2948._resolution_is_terminal("not a dict") is False  # type: ignore[arg-type]
    assert (
        exp2948._resolution_is_terminal(
            {"resolution": "applied_by_operator_authorized_outer_loop", "operator_authorized": True}
        )
        is True
    )
    assert (
        exp2948._resolution_is_terminal(
            {"resolution": "applied_by_operator_authorized_outer_loop", "operator_authorized": False}
        )
        is False
    )
    assert (
        exp2948._resolution_is_terminal(
            {"resolution": "unknown_random_string", "operator_authorized": True}
        )
        is False
    )


def test_cited_upstream_artifacts_includes_sha256(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    cited = artifact["cited_upstream_artifacts"]
    assert isinstance(cited, list) and len(cited) == len(exp2948.EXPECTED_ARTIFACTS)
    for row in cited:
        assert row["present"] is True
        assert isinstance(row["sha256"], str) and len(row["sha256"]) == 64


def test_cited_upstream_artifact_sha256_none_when_missing(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    (tmp_path / exp2948.EXPECTED_ARTIFACTS["exp2941"].path).unlink()
    artifact = exp2948.build_artifact(tmp_path)
    by_id = {row["experiment_id"]: row for row in artifact["cited_upstream_artifacts"]}
    assert by_id["exp2941"]["present"] is False
    assert by_id["exp2941"]["sha256"] is None


def test_write_artifact_persists_to_disk(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    output_path = exp2948.write_artifact(tmp_path)
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["artifact"] == "experiment_2948_capstone_v277"
    assert payload["milestone"] == "2026.05.277"


def test_write_artifact_with_absolute_output_path(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    custom_out = tmp_path / "custom_dir" / "custom_capstone.json"
    output_path = exp2948.write_artifact(tmp_path, output_path=custom_out)
    assert output_path == custom_out
    assert custom_out.exists()


def test_read_json_mapping_handles_bad_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert exp2948.read_json_mapping(bad) == {}


def test_read_json_mapping_handles_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    assert exp2948.read_json_mapping(missing) == {}


def test_read_json_mapping_rejects_non_dict(tmp_path: Path) -> None:
    list_file = tmp_path / "list.json"
    list_file.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp2948.read_json_mapping(list_file) == {}


def test_artifact_classification_counts_consistent(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    counts = artifact["artifact_classification_counts"]
    total_listed = (
        len(artifact["clean_artifacts"])
        + len(artifact["flagged_artifacts"])
        + len(artifact["blocked_artifacts"])
        + len(artifact["missing_artifacts"])
    )
    assert total_listed == len(exp2948.EXPECTED_ARTIFACTS)
    assert (
        counts["clean"] + counts["flagged"] + counts["blocked"] + counts["missing"]
        == len(exp2948.EXPECTED_ARTIFACTS)
    )


def test_gaps_for_278_flags_exp2946_low_pass_at_1(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    gaps = artifact["gaps_for_278"]
    # exp2946 pass@1=0.06 is the one weak signal the .277 close should
    # surface for .278's planner.
    assert any("pass@1" in g.lower() or "candidate-generation" in g for g in gaps)


def test_gaps_for_278_default_message_when_no_gaps(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    # Strip the exp2946 low-pass marker so no other gap triggers.
    _write_json(
        tmp_path,
        exp2948.EXPECTED_ARTIFACTS["exp2946"].path,
        {"honest_verdict": "complete: continuation_executed_pass@1=0.85"},
    )
    artifact = exp2948.build_artifact(tmp_path)
    gaps = artifact["gaps_for_278"]
    assert any("operator-curated narrowing-edit" in g for g in gaps)


def test_top_3_next_actions_length(tmp_path: Path) -> None:
    _write_default_inputs(tmp_path)
    artifact = exp2948.build_artifact(tmp_path)
    actions = artifact["top_3_next_actions"]
    assert isinstance(actions, list) and len(actions) == 3
    for action in actions:
        assert isinstance(action, str) and action.strip()


def test_headline_outcome_rescue(tmp_path: Path) -> None:
    """If MMD finds distributions indistinguishable AND speedup >= 1.0 AND
    AUPRC retain, the corrigenda 'rescued' the original draft (no narrowing
    needed)."""

    _write_default_inputs(tmp_path)
    _write_json(
        tmp_path,
        exp2948.EXPECTED_ARTIFACTS["exp2938"].path,
        _exp2938_payload(distinguishable=False),
    )
    _write_json(
        tmp_path,
        exp2948.EXPECTED_ARTIFACTS["exp2939"].path,
        _exp2939_payload(speedup=1.25),
    )
    artifact = exp2948.build_artifact(tmp_path)
    assert artifact["deep_think_corrigenda_outcomes"]["headline_outcome"] == "rescue"


def test_coerce_float_handles_various_types() -> None:
    assert exp2948._coerce_float(1.5) == 1.5
    assert exp2948._coerce_float(2) == 2.0
    assert exp2948._coerce_float("3.14") == 3.14
    assert exp2948._coerce_float("not a number") is None
    assert exp2948._coerce_float(None) is None
    assert exp2948._coerce_float(True) is None  # bool is rejected explicitly


def test_max_pvalue_with_non_list() -> None:
    assert exp2948._max_pvalue("not a list") is None
    assert exp2948._max_pvalue(None) is None
    assert exp2948._max_pvalue([]) is None
    assert exp2948._max_pvalue([0.1, 0.5, 0.001]) == 0.5


def test_baseline_auprc_handles_dict_and_scalar() -> None:
    assert exp2948._baseline_auprc({"code_corpus_baseline_random_auprc": {"value": 0.075}}) == 0.075
    assert exp2948._baseline_auprc({"code_corpus_baseline_random_auprc": 0.5}) == 0.5
    assert exp2948._baseline_auprc({}) is None


def test_real_inputs_produce_paper_ready_capstone() -> None:
    """Smoke test against the actual repository artifacts (not a tmp_path
    mirror). This pins the deliverable's behavior to today's real data and
    is the test that the conductor's archive/activate step relies on."""

    real_root = Path(__file__).resolve().parents[2]
    expected = real_root / exp2948.EXPECTED_ARTIFACTS["exp2938"].path
    if not expected.is_file():
        pytest.skip("real .277 artifacts not present in working tree")
    artifact = exp2948.build_artifact(real_root)
    assert artifact["milestone"] == "2026.05.277"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "exp2938" in artifact["clean_artifacts"]
    assert "exp2939" in artifact["clean_artifacts"]
    assert "exp2940" in artifact["clean_artifacts"]
    outcomes = artifact["deep_think_corrigenda_outcomes"]
    assert outcomes["mmd_distinguishable"] is True
    assert outcomes["same_schedule_speedup"] is not None
    assert outcomes["same_schedule_speedup"] < 1.0
    assert outcomes["code_auprc_recommendation"] == "retain"
    assert outcomes["headline_outcome"] == "narrow"


def test_module_constants() -> None:
    """Sanity: the module's exported constants describe milestone .277."""

    assert exp2948.MILESTONE == "2026.05.277"
    assert exp2948.SCHEMA == "carnot.milestone_capstone.v277"
    assert exp2948.ARTIFACT == "experiment_2948_capstone_v277"
    assert exp2948.INFERENCE_SUBSTRATE == "aggregation_from_upstream_artifacts"
    assert exp2948.RUN_DATE == "20260523"
    assert exp2948.OUTPUT_REL_PATH == Path("results/experiment_2948_capstone_v277.json")
    assert set(exp2948.ROW_CLASSES) == {"clean", "flagged", "blocked", "missing"}
