"""Tests for Exp 3739 Thesis-A kill-gate part-(b) verdict.

Spec refs: REQ-EBT-3739, SCENARIO-EBT-3739-WIN,
SCENARIO-EBT-3739-BOUNDED, SCENARIO-EBT-3739-INVALID,
SCENARIO-EBT-3739-NOT-RUN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_3739_kill_gate_part_b_verdict as exp3739


SPEC_PATH = Path("openspec/capabilities/ebt-nrgpt/spec.md")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _part_a(*, green: bool = True, conclusion: str | None = None) -> dict[str, object]:
    if conclusion is None:
        conclusion = (
            "GREEN-LIGHT: real run completed."
            if green
            else "UNTESTED: training did not complete -- part-(a) remains untested."
        )
    return {
        "schema": "carnot.experiment_3736_real_kill_gate_part_a_verdict.v1",
        "experiment": 3736,
        "honest_verdict": (
            exp3739.PART_A_PASS_VERDICT
            if green
            else "complete: real_kill_gate_part_a_untested_training_did_not_complete"
        ),
        "green_light_342": green,
        "ebt_trained_stably": green,
        "training_actually_ran": True,
        "kill_gate_conclusion": conclusion,
        "real_run_diagnostics": {"training_actually_ran": True},
        "random_seed": 3736,
        "reproducibility_checksum": "3" * 64,
        "duration_s": 0.5,
    }


def _exp3738(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "carnot.experiment_3738_matched_compute_comparison.v1",
        "experiment": 3738,
        "honest_verdict": "complete: exp3738_matched_compute_heldout_comparison",
        "accuracy_delta": 0.04,
        "flops_matched_within_tolerance": True,
        "n_heldout": 120,
        "ebt_accuracy": 0.62,
        "ar_accuracy": 0.58,
        "random_seed": 3738,
        "reproducibility_checksum": "8" * 64,
        "duration_s": 10.0,
    }
    payload.update(overrides)
    return payload


def _seed_root(
    root: Path,
    *,
    part_a: dict[str, object] | None = None,
    exp3738_payload: dict[str, object] | None = None,
) -> None:
    if part_a is not None:
        _write_json(root / exp3739.EXP3736_REL_PATH, part_a)
    if exp3738_payload is not None:
        _write_json(root / exp3739.EXP3738_REL_PATH, exp3738_payload)


def test_req_ebt_3739_spec_anchor_exists() -> None:
    """REQ-EBT-3739: OpenSpec declares the part-(b) verdict contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-EBT-3739" in spec
    assert "SCENARIO-EBT-3739-WIN" in spec
    assert "SCENARIO-EBT-3739-BOUNDED" in spec
    assert "SCENARIO-EBT-3739-INVALID" in spec
    assert "SCENARIO-EBT-3739-NOT-RUN" in spec
    assert exp3739.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_ebt_3739_part_a_not_green_lit_is_not_run(tmp_path: Path) -> None:
    """SCENARIO-EBT-3739-NOT-RUN: no Exp 3738 read can become a negative."""

    _seed_root(tmp_path, part_a=_part_a(green=False), exp3738_payload=None)

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=10.25,
        adversarial_verify_report={"flags": []},
    )

    exp3739.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3739.NOT_RUN_PART_A_VERDICT
    assert artifact["inference_substrate"] == exp3739.INFERENCE_SUBSTRATE
    assert artifact["thesis_a_outcome"] == "part_b_not_run"
    assert artifact["ebt_beats_ar_at_matched_compute"] is False
    assert artifact["accuracy_delta_cited"] is None
    assert artifact["flops_matched_cited"] is None
    assert artifact["n_heldout_cited"] is None
    assert "part-(a) did not green-light" in artifact["part_b_not_run_reason"]
    assert "training did not complete" in artifact["part_b_not_run_reason"]
    assert "bounded_at_small_scale" not in artifact["thesis_a_outcome"]
    assert artifact["cited_upstream_artifacts"][0]["experiment_id"] == 3736
    assert artifact["reproducibility_checksum"] == exp3739.payload_checksum(artifact)


def test_scenario_ebt_3739_exp3738_absent_after_green_light_is_not_run(tmp_path: Path) -> None:
    """SCENARIO-EBT-3739-NOT-RUN: absent part-(b) artifact remains not-run."""

    _seed_root(tmp_path, part_a=_part_a(green=True), exp3738_payload=None)

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3739.NOT_RUN_EXP3738_VERDICT
    assert artifact["thesis_a_outcome"] == "part_b_not_run"
    assert artifact["part_b_not_run_reason"] == "Exp 3738 artifact absent or unreadable"
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == {3736}


def test_scenario_ebt_3739_blocked_exp3738_is_not_run(tmp_path: Path) -> None:
    """SCENARIO-EBT-3739-NOT-RUN: blocked comparison is not a bounded result."""

    _seed_root(
        tmp_path,
        part_a=_part_a(green=True),
        exp3738_payload=_exp3738(
            honest_verdict="blocked_cuda_gpu_unavailable",
            accuracy_delta=None,
            flops_matched_within_tolerance=None,
            n_heldout=None,
        ),
    )

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=1.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3739.NOT_RUN_EXP3738_VERDICT
    assert artifact["thesis_a_outcome"] == "part_b_not_run"
    assert "blocked" in artifact["part_b_not_run_reason"]
    assert artifact["accuracy_delta_cited"] is None
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == {3736, 3738}


def test_scenario_ebt_3739_win_requires_positive_delta_matched_flops_and_n100(
    tmp_path: Path,
) -> None:
    """SCENARIO-EBT-3739-WIN: all three thesis criteria make the claim true."""

    _seed_root(
        tmp_path,
        part_a=_part_a(green=True),
        exp3738_payload=_exp3738(accuracy_delta=0.03125, n_heldout=160),
    )

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=2.5,
        adversarial_verify_report={"flags": []},
    )

    exp3739.validate_artifact(artifact)
    assert artifact["thesis_a_outcome"] == "ebt_beats_ar_at_matched_compute"
    assert artifact["ebt_beats_ar_at_matched_compute"] is True
    assert artifact["accuracy_delta_cited"] == pytest.approx(0.03125)
    assert artifact["flops_matched_cited"] is True
    assert artifact["n_heldout_cited"] == 160
    assert artifact["honest_verdict"].startswith(
        "complete: kill_gate_part_b_ebt_BEATS_ar_at_matched_compute_delta_0.03125_n160"
    )
    assert "bounded scale-up" in artifact["next_step_recommendation"].lower()
    assert {item["experiment_id"] for item in artifact["cited_upstream_artifacts"]} == {3736, 3738}


def test_scenario_ebt_3739_bounded_with_narrowing_gap_recommends_one_2x_attempt(
    tmp_path: Path,
) -> None:
    """SCENARIO-EBT-3739-BOUNDED: narrowing non-win gets one bounded retry."""

    _seed_root(
        tmp_path,
        part_a=_part_a(green=True),
        exp3738_payload=_exp3738(accuracy_delta=-0.01, gap_narrowing=True),
    )

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=3.0,
        now_s=3.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3739.BOUNDED_VERDICT
    assert artifact["thesis_a_outcome"] == "bounded_at_small_scale"
    assert artifact["ebt_beats_ar_at_matched_compute"] is False
    assert artifact["accuracy_delta_cited"] == pytest.approx(-0.01)
    assert "does NOT beat AR at equal compute" in artifact["decision_basis"]
    assert "one 2x-training attempt" in artifact["next_step_recommendation"]
    assert artifact["gap_narrowing_cited"] is True


def test_scenario_ebt_3739_bounded_flat_gap_retires_small_scale_route(tmp_path: Path) -> None:
    """SCENARIO-EBT-3739-BOUNDED: flat or negative gap retires the route."""

    _seed_root(
        tmp_path,
        part_a=_part_a(green=True),
        exp3738_payload=_exp3738(accuracy_delta=-0.08, gap_trend="flat_negative"),
    )

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=4.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["thesis_a_outcome"] == "bounded_at_small_scale"
    assert artifact["gap_narrowing_cited"] is False
    assert "Retire the route at small scale" in artifact["next_step_recommendation"]


def test_scenario_ebt_3739_invalid_flop_mismatch_calls_no_winner(tmp_path: Path) -> None:
    """SCENARIO-EBT-3739-INVALID: compute confound voids the comparison."""

    _seed_root(
        tmp_path,
        part_a=_part_a(green=True),
        exp3738_payload=_exp3738(
            accuracy_delta=0.2,
            flops_matched_within_tolerance=False,
            n_heldout=200,
        ),
    )

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=5.0,
        now_s=5.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["honest_verdict"] == exp3739.INVALID_VERDICT
    assert artifact["thesis_a_outcome"] == "comparison_invalid"
    assert artifact["ebt_beats_ar_at_matched_compute"] is False
    assert artifact["accuracy_delta_cited"] == pytest.approx(0.2)
    assert artifact["flops_matched_cited"] is False
    assert "do not call a winner" in artifact["decision_basis"]
    assert "tighter budget match" in artifact["next_step_recommendation"]


def test_req_ebt_3739_positive_delta_below_n100_is_invalid(tmp_path: Path) -> None:
    """REQ-EBT-3739: a positive delta below n=100 cannot prove Thesis-A."""

    _seed_root(
        tmp_path,
        part_a=_part_a(green=True),
        exp3738_payload=_exp3738(accuracy_delta=0.02, n_heldout=99),
    )

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=6.0,
        now_s=6.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["thesis_a_outcome"] == "comparison_invalid"
    assert artifact["ebt_beats_ar_at_matched_compute"] is False
    assert "n_heldout=99" in artifact["decision_basis"]


def test_req_ebt_3739_nested_exp3738_metrics_are_extracted(tmp_path: Path) -> None:
    """REQ-EBT-3739: Exp 3738 nested matched-compute reports are accepted."""

    nested = _exp3738(
        accuracy_delta=None,
        flops_matched_within_tolerance=None,
        n_heldout=None,
        matched_compute_report={
            "ebt_accuracy": 0.55,
            "ar_accuracy": 0.5,
            "n_heldout": 140,
            "budget_match": {"within_tolerance": True},
        },
    )
    _seed_root(tmp_path, part_a=_part_a(green=True), exp3738_payload=nested)

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=7.0,
        now_s=7.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["thesis_a_outcome"] == "ebt_beats_ar_at_matched_compute"
    assert artifact["accuracy_delta_cited"] == pytest.approx(0.05)
    assert artifact["flops_matched_cited"] is True
    assert artifact["n_heldout_cited"] == 140


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("honest_verdict"), "missing required"),
        (lambda p: p.update(honest_verdict="complete: wrong"), "terminal verdict"),
        (lambda p: p.update(inference_substrate="live_llm_inference"), "inference_substrate"),
        (lambda p: p.update(thesis_a_outcome="wrong"), "thesis_a_outcome"),
        (lambda p: p.update(ebt_beats_ar_at_matched_compute="yes"), "bare bool"),
        (lambda p: p.update(accuracy_delta_cited="0.1"), "accuracy_delta_cited"),
        (lambda p: p.update(flops_matched_cited="true"), "flops_matched_cited"),
        (lambda p: p.update(n_heldout_cited="100"), "n_heldout_cited"),
        (lambda p: p.update(next_step_recommendation=""), "next_step_recommendation"),
        (lambda p: p.update(cited_upstream_artifacts=[]), "cited_upstream_artifacts"),
        (lambda p: p.update(random_seed=3738), "random_seed"),
        (lambda p: p.update(duration_s=0.0), "duration_s"),
        (lambda p: p.update(field_principles=[]), "field_principles"),
        (lambda p: p["field_principles"].pop("honest_verdict"), "field principles"),
        (lambda p: p.update(model_specs={}), "live-model markers"),
        (lambda p: p.update(adversarial_verify_report={"flags": [{"severity": "critical"}]}), "critical"),
        (lambda p: p.update(reproducibility_checksum="bad"), "sha256"),
        (lambda p: p.update(reproducibility_checksum="0" * 64), "does not match"),
    ],
)
def test_req_ebt_3739_validate_rejects_schema_regressions(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-EBT-3739: validation blocks dishonest verdict artifacts."""

    _seed_root(tmp_path, part_a=_part_a(green=True), exp3738_payload=_exp3738())
    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=8.0,
        now_s=8.5,
        adversarial_verify_report={"flags": []},
    )
    broken = json.loads(json.dumps(artifact))
    mutate(broken)

    with pytest.raises(ValueError, match=message):
        exp3739.validate_artifact(broken)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(cited_upstream_artifacts="bad"), "cited_upstream_artifacts"),
        (lambda p: p["cited_upstream_artifacts"].append(123), "object"),
        (lambda p: p["cited_upstream_artifacts"][0].update(fields_imported=[]), "fields_imported"),
        (lambda p: p["cited_upstream_artifacts"][0].update(sha256="bad"), "sha256"),
        (lambda p: p["cited_upstream_artifacts"][1].update(experiment_id=3737), "Exp 3738"),
    ],
)
def test_req_ebt_3739_validate_rejects_bad_citations(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    """REQ-EBT-3739: provenance citations stay traceable to upstream data."""

    _seed_root(tmp_path, part_a=_part_a(green=True), exp3738_payload=_exp3738())
    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=9.0,
        now_s=9.5,
        adversarial_verify_report={"flags": []},
    )
    broken = json.loads(json.dumps(artifact))
    mutate(broken)

    with pytest.raises(ValueError, match=message):
        exp3739.validate_artifact(broken)


def test_req_ebt_3739_validate_rejects_missing_exp3736_citation(tmp_path: Path) -> None:
    """REQ-EBT-3739: Exp 3736 part-(a) provenance is always required."""

    _seed_root(tmp_path, part_a=_part_a(green=True), exp3738_payload=_exp3738())
    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=9.0,
        now_s=9.5,
        adversarial_verify_report={"flags": []},
    )
    broken = json.loads(json.dumps(artifact))
    broken["cited_upstream_artifacts"][0]["experiment_id"] = 3737
    broken["reproducibility_checksum"] = exp3739.payload_checksum(broken)

    with pytest.raises(ValueError, match="Exp 3736"):
        exp3739.validate_artifact(broken)


@pytest.mark.parametrize(
    ("base", "mutate", "message"),
    [
        (
            "win",
            lambda p: p.update(ebt_beats_ar_at_matched_compute=False),
            "win outcome must set",
        ),
        (
            "win",
            lambda p: p.update(accuracy_delta_cited=-0.1),
            "positive delta",
        ),
        (
            "win",
            lambda p: p.update(n_heldout_cited=99),
            "n_heldout>=100",
        ),
        (
            "bounded",
            lambda p: p.update(ebt_beats_ar_at_matched_compute=True),
            "non-win",
        ),
        (
            "bounded",
            lambda p: p.update(honest_verdict=exp3739.INVALID_VERDICT),
            "bounded terminal",
        ),
        (
            "bounded",
            lambda p: p.update(accuracy_delta_cited=0.1),
            "non-positive delta",
        ),
        (
            "invalid",
            lambda p: p.update(honest_verdict=exp3739.BOUNDED_VERDICT),
            "invalid terminal",
        ),
        (
            "not_run",
            lambda p: p.update(honest_verdict=exp3739.INVALID_VERDICT),
            "not-run outcome",
        ),
        (
            "not_run",
            lambda p: p.update(part_b_not_run_reason=""),
            "fallback reason",
        ),
    ],
)
def test_req_ebt_3739_validate_rejects_decision_semantic_regressions(
    tmp_path: Path,
    base: str,
    mutate,
    message: str,
) -> None:
    """REQ-EBT-3739: decision-specific invariants are enforced."""

    if base == "win":
        _seed_root(tmp_path, part_a=_part_a(green=True), exp3738_payload=_exp3738())
    elif base == "bounded":
        _seed_root(
            tmp_path,
            part_a=_part_a(green=True),
            exp3738_payload=_exp3738(accuracy_delta=-0.01),
        )
    elif base == "invalid":
        _seed_root(
            tmp_path,
            part_a=_part_a(green=True),
            exp3738_payload=_exp3738(flops_matched_within_tolerance=False),
        )
    else:
        _seed_root(tmp_path, part_a=_part_a(green=False), exp3738_payload=None)
    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=9.0,
        now_s=9.5,
        adversarial_verify_report={"flags": []},
    )
    broken = json.loads(json.dumps(artifact))
    mutate(broken)
    broken["reproducibility_checksum"] = exp3739.payload_checksum(broken)

    with pytest.raises(ValueError, match=message):
        exp3739.validate_artifact(broken)


def test_req_ebt_3739_missing_accuracy_delta_is_invalid(tmp_path: Path) -> None:
    """REQ-EBT-3739: present Exp 3738 without a delta is invalid, not a crash."""

    no_delta = {
        "honest_verdict": "complete: exp3738_missing_delta",
        "flops_matched_within_tolerance": True,
        "n_heldout": 120,
        "random_seed": 3738,
        "reproducibility_checksum": "8" * 64,
        "duration_s": 10.0,
    }
    _seed_root(tmp_path, part_a=_part_a(green=True), exp3738_payload=no_delta)

    artifact = exp3739.build_artifact(
        tmp_path,
        started_s=9.0,
        now_s=9.5,
        adversarial_verify_report={"flags": []},
    )

    assert artifact["thesis_a_outcome"] == "comparison_invalid"
    assert "accuracy_delta is missing" in artifact["decision_basis"]


def test_scenario_ebt_3739_write_runs_adversarial_verify(tmp_path: Path) -> None:
    """REQ-EBT-3739: writing the artifact records verifier no-critical status."""

    _seed_root(tmp_path, part_a=_part_a(green=False), exp3738_payload=None)

    output = exp3739.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / exp3739.OUTPUT_REL_PATH
    assert payload["adversarial_verify_clean"] is True
    assert payload["adversarial_verify_report"]["critical_flag_count"] == 0
    assert payload["reproducibility_checksum"] == exp3739.payload_checksum(payload)
    exp3739.validate_artifact(payload)


def test_req_ebt_3739_main_writes_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-EBT-3739: CLI writes and prints the terminal verdict."""

    _seed_root(tmp_path, part_a=_part_a(green=False), exp3738_payload=None)
    monkeypatch.setattr(exp3739, "REPO_ROOT", tmp_path)

    assert exp3739.main([]) == 0

    payload = json.loads((tmp_path / exp3739.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert payload["honest_verdict"] == exp3739.NOT_RUN_PART_A_VERDICT
    assert exp3739.NOT_RUN_PART_A_VERDICT in capsys.readouterr().out


def test_req_ebt_3739_malformed_present_artifact_fails_explicitly(tmp_path: Path) -> None:
    """REQ-EBT-3739: malformed JSON objects fail without None crashes."""

    _write_json(tmp_path / exp3739.EXP3736_REL_PATH, _part_a(green=True))
    bad = tmp_path / exp3739.EXP3738_REL_PATH
    bad.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        exp3739.build_artifact(
            tmp_path,
            started_s=10.0,
            now_s=10.5,
            adversarial_verify_report={"flags": []},
        )


def test_req_ebt_3739_helper_edge_cases(tmp_path: Path) -> None:
    """REQ-EBT-3739: helper edge cases remain deterministic."""

    globbed = tmp_path / "results" / "experiment_3738_alt_name.json"
    _write_json(globbed, _exp3738())
    assert exp3739._extract_bool({"x": "true"}, ["x"]) is True
    assert exp3739._extract_bool({"x": "false"}, ["x"]) is False
    assert exp3739._extract_bool({"x": 1}, ["x"]) is None
    assert exp3739._extract_accuracy_delta({}) is None
    assert exp3739._extract_n_heldout({}) is None
    assert exp3739._read_json_object(tmp_path / "missing.json", required=False) is None
    with pytest.raises(FileNotFoundError):
        exp3739._read_json_object(tmp_path / "missing.json", required=True)
    assert exp3739._safe_int(object()) is None
    assert exp3739._format_delta(None) == "unknown"
    assert exp3739._has_live_model_markers({"target_model": "forbidden"}) is True
    assert exp3739._critical_flag_count({"critical_flag_count": 2, "flags": []}) == 2
    assert exp3739._critical_flag_count({"flags": "bad"}) == 0
    assert exp3739._critical_flag_count(None) == 0
    assert exp3739._is_sha256("z" * 64) is False
    assert exp3739._gap_narrowing({"gap_trend": "narrowing"}, -0.01) is True
    assert exp3739._gap_narrowing({"gap_narrowing": "yes"}, -0.01) is False
    assert exp3739._find_exp3738(tmp_path)[0] == globbed
    empty = tmp_path / "empty"
    empty.mkdir()
    assert exp3739._find_exp3738(empty) == (None, None)
