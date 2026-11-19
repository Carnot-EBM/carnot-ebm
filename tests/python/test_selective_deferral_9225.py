"""REQ-REPORT-9225-V785 / SCENARIO-REPORT-9225-V785: selective-deferral masks and spline-support diagnostics."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import selective_deferral_9225 as d
from carnot.reporting import selective_deferral_runner_9225 as c


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-9225-V785: authenticate custody and measure selective-deferral and spline controls once."""
    return c.measure(d.ROOT, tmp_path_factory.mktemp("deferral9225"), "20261119")


def test_source_custody_and_stream_deferral_masks(work):
    """REQ-REPORT-9225-V785: 900 authenticated responses, 10 five-qid streams, 150 seeded masks and controls."""
    inspected = work["inspection"]
    assert all(r["passed"] for r in inspected["checks"])
    rows = inspected["rows"]
    assert len(rows) == 150
    assert len({r["qid"] for r in rows}) == 50
    assert len({r["stream_id"] for r in rows}) == 10
    assert {r["sibling"] for r in rows} == set(d.GENERATORS)
    for r in rows:
        expected_draw = d.audit_draw(r["stream_id"], r["qid"], r["sibling"])
        assert r["seed_sha256"] == expected_draw["seed_sha256"]
        assert math.isclose(r["audit_u"], expected_draw["audit_u"], abs_tol=1e-15)
        assert 0.0 <= r["audit_u"] < 1.0
        assert r["audit_alpha_0_10"] <= r["audit_alpha_0_25"]
        assert r["full_release_non_escalated_pi"] == 1.0
        assert r["escalation_only_non_escalated_pi"] == 0.0
    ctrl = inspected["controls"]
    assert all(r["passed"] for r in ctrl["estimator_control_rows"])
    assert all(r["passed"] for r in ctrl["spline_control_rows"])
    assert ctrl["support_counts"]["ALL"]["unique_responses"] == 900
    assert ctrl["support_counts"]["ALL"]["independent_qids"] == 300
    assert ctrl["support_counts"]["initial100"]["unique_responses"] == 300
    assert ctrl["support_counts"]["stream50"]["unique_responses"] == 150
    assert ctrl["support_counts"]["stream50"]["independent_qids"] == 50
    assert ctrl["support_counts"]["future40"]["unique_responses"] == 120
    assert ctrl["support_counts"]["guard50"]["unique_responses"] == 150
    assert ctrl["support_counts"]["retention100"]["repeated_stream_appearances"] == 3000


def test_observation_regimes_horvitz_thompson_and_unidentifiability():
    """SCENARIO-REPORT-9225-V785: IPW unbiasedness for alpha>0, naive selection bias, and alpha=0 unidentifiability."""
    assert d.typed_decision(0.1) == "accept"
    assert d.typed_decision(0.5) == "escalate"
    assert d.typed_decision(0.9) == "reject"
    assert d.observation_propensity(0.05, "full_release") == 1.0
    assert d.observation_propensity(0.05, "escalation_only") == 0.0
    assert d.observation_propensity(0.50, "escalation_only") == 1.0
    assert d.observation_propensity(0.05, "audited_deferral", 0.10) == 0.10
    assert d.observation_propensity(0.95, "audited_deferral", 0.25) == 0.25
    assert d.observation_propensity(0.50, "audited_deferral", 0.10) == 1.0

    obs_hit = d.observe_feedback(0.05, "audited_deferral", alpha=0.25, audit_u=0.10)
    assert obs_hit["observed"] is True and math.isclose(
        obs_hit["inverse_propensity_weight"], 4.0, abs_tol=1e-15
    )
    obs_miss = d.observe_feedback(0.05, "audited_deferral", alpha=0.10, audit_u=0.20)
    assert obs_miss["observed"] is False and obs_miss["inverse_propensity_weight"] == 0.0
    obs_esc_miss = d.observe_feedback(0.05, "escalation_only")
    assert obs_esc_miss["observed"] is False and obs_esc_miss["inverse_propensity_weight"] is None

    # Synthetic calibrated stream with severe accepted-region drift vs clean accepted region.
    panel = []
    for idx in range(60):
        if idx < 20:
            p, y = 0.05, int(idx % 20 == 0)
        elif idx < 40:
            p, y = 0.50, int(idx % 2 == 0)
        else:
            p, y = 0.95, int(idx % 20 != 0)
        panel.append(
            dict(
                unit_id=f"u_{idx:02d}",
                stream_id=f"stream_{idx // 6:02d}",
                qid=f"q_{idx // 3:02d}",
                sibling=d.GENERATORS[idx % 3],
                p=p,
                y=y,
                weight=1.0 / 60.0,
            )
        )
    full = d.estimate_selective_risk(panel, "full_release")
    assert full["identifiability_status"] == "identifiable"
    assert math.isclose(full["ipw_estimate"], full["true_fixture_risk"], abs_tol=1e-15)
    assert math.isclose(full["naive_estimate"], full["true_fixture_risk"], abs_tol=1e-15)
    assert math.isclose(
        full["ipw_accepted_error_conditional"],
        full["true_accepted_error_conditional"],
        abs_tol=1e-15,
    )

    esc = d.estimate_selective_risk(panel, "escalation_only")
    assert esc["identifiability_status"] == "unidentifiable"
    assert esc["accepted_drift_identifiable"] is False
    assert esc["ipw_estimate"] is None
    assert esc["ipw_accepted_error_conditional"] is None
    assert esc["naive_accepted_error_conditional"] is None
    assert esc["naive_estimate"] > full["true_fixture_risk"] + 0.08

    # Even if 100% of accepted items drift to error y=1, escalation_only observes identical labels.
    drifted_panel = [dict(row, y=1 if row["p"] <= 0.1 else row["y"]) for row in panel]
    esc_drift = d.estimate_selective_risk(drifted_panel, "escalation_only")
    assert esc_drift["identifiability_status"] == "unidentifiable"
    assert esc_drift["ipw_accepted_error_conditional"] is None
    assert esc_drift["naive_accepted_error_conditional"] is None
    assert math.isclose(esc_drift["naive_estimate"], esc["naive_estimate"], abs_tol=1e-15)

    for alpha in d.ALPHA_GRID:
        aud = d.estimate_selective_risk(panel, "audited_deferral", alpha=alpha)
        assert aud["identifiability_status"] == "identifiable"
        assert aud["accepted_drift_identifiable"] is True
        assert math.isclose(
            aud["exact_ipw_brier_expectation"], aud["true_fixture_risk"], abs_tol=1e-15
        )
        assert math.isclose(
            aud["exact_ipw_accepted_error_expectation"],
            aud["true_accepted_error_conditional"],
            abs_tol=1e-15,
        )
        mc = d.monte_carlo_audit_summary(panel, alpha=alpha, trials=400, seed=785225)
        assert abs(mc["mc_ipw_brier_mean"] - aud["true_fixture_risk"]) < 0.006
        assert (
            abs(mc["mc_ipw_accepted_error_mean"] - aud["true_accepted_error_conditional"]) < 0.015
        )
        assert mc["mc_naive_brier_mean"] - aud["true_fixture_risk"] > 0.05

    all_esc = [dict(unit_id="e0", p=0.5, y=1, weight=1.0)]
    all_esc_res = d.estimate_selective_risk(all_esc, "escalation_only")
    assert all_esc_res["identifiability_status"] == "identifiable"
    assert all_esc_res["true_accepted_error_conditional"] is None
    assert all_esc_res["ipw_accepted_error_conditional"] is None


def test_spline_support_density_classifier_and_rejections():
    """SCENARIO-REPORT-9225-V785: 57-column cubic B-spline sum-of-unity, clipping, constants, monotone transforms and label-free rejection."""
    base_init = np.linspace(-2.0, 2.0, 300)[:, None] * np.arange(1, 8, dtype=float)[None, :]
    fitted = d.fit_feature_transform(base_init)

    # Interior coordinates in [-0.5, 0.5] after scaling.
    interior_raw = np.zeros((4, 7), dtype=float)
    interior = d.classify_spline_support(interior_raw, fitted)
    assert interior["basis_columns"] == 57
    assert interior["interior_dense_count"] == 4
    assert interior["boundary_sparse_or_clipped_count"] == 0
    assert interior["max_partition_of_unity_error"] < 1e-14
    for row in interior["rows"]:
        assert row["clipped_coordinate_count"] == 0
        assert row["out_of_range_coordinate_count"] == 0
        assert math.isclose(row["boundary_basis_mass_fraction"], 0.0, abs_tol=1e-15)
        assert math.isclose(row["basis_sum"], 7.0, abs_tol=1e-14)
        assert row["stratum"] == "interior_dense"

    # Boundary-clipped, sparse boundary, and out-of-range test inputs.
    extreme_raw = np.vstack(
        [
            base_init[0],
            base_init[-1],
            base_init[-1] * 3.0,
            base_init[-1] * 0.75,
        ]
    )
    extreme = d.classify_spline_support(extreme_raw, fitted)
    assert extreme["boundary_sparse_or_clipped_count"] == 4
    assert extreme["max_partition_of_unity_error"] < 1e-14
    assert extreme["rows"][0]["clipped_coordinate_count"] == 7
    assert extreme["rows"][1]["clipped_coordinate_count"] == 7
    assert extreme["rows"][2]["clipped_coordinate_count"] == 7
    assert extreme["rows"][2]["out_of_range_coordinate_count"] == 7
    assert extreme["rows"][3]["clipped_coordinate_count"] == 0
    assert math.isclose(extreme["rows"][3]["boundary_basis_mass_fraction"], 0.125, abs_tol=1e-12)
    assert all(r["stratum"] == "boundary_sparse_or_clipped" for r in extreme["rows"])

    # Constant-feature zero mapping and NaN imputation to initialization median.
    const_init = base_init.copy()
    const_init[:, 2] = 5.0
    const_fitted = d.fit_feature_transform(const_init)
    assert const_fitted["constant_coordinates"] == [2]
    const_test = np.zeros((2, 7), dtype=float)
    const_test[0, 2] = 99.0
    const_test[1, 0] = float("nan")
    const_res = d.classify_spline_support(const_test, const_fitted)
    assert const_res["max_partition_of_unity_error"] < 1e-14
    assert all(r["stratum"] == "interior_dense" for r in const_res["rows"])
    assert all(math.isclose(r["scaled_features"][2], 0.0, abs_tol=1e-15) for r in const_res["rows"])

    # Positive affine transformation and odd monotone cubic transformation preserve sum-of-unity & strata.
    scales = np.array([1.5, 2.0, 0.5, 3.0, 4.0, 1.25, 2.5])
    shifts = np.array([-1.0, 0.5, 2.0, -3.0, 1.0, 0.0, -0.5])
    affine_fitted = d.fit_feature_transform(base_init * scales + shifts)
    mixed_raw = np.vstack([interior_raw[0], extreme_raw[0], extreme_raw[2], extreme_raw[3]])
    orig_mixed = d.classify_spline_support(mixed_raw, fitted)
    affine_mixed = d.classify_spline_support(mixed_raw * scales + shifts, affine_fitted)
    assert [r["stratum"] for r in orig_mixed["rows"]] == [
        r["stratum"] for r in affine_mixed["rows"]
    ]
    for r_o, r_a in zip(orig_mixed["rows"], affine_mixed["rows"], strict=True):
        assert math.isclose(
            r_o["boundary_basis_mass_fraction"],
            r_a["boundary_basis_mass_fraction"],
            abs_tol=1e-12,
        )
    monotone_res = d.classify_spline_support(
        mixed_raw, fitted, monotone_transform="cubic_preserving_endpoints"
    )
    assert monotone_res["max_partition_of_unity_error"] < 1e-14
    assert [r["stratum"] for r in monotone_res["rows"]] == [
        r["stratum"] for r in orig_mixed["rows"]
    ]

    # Reject malformed inputs and any label leakage.
    for bad_init in [
        np.zeros((10,)),
        np.zeros((0, 7)),
        np.zeros((10, 6)),
        np.full((10, 7), np.nan),
        np.full((10, 7), np.inf),
    ]:
        with pytest.raises(ValueError):
            d.fit_feature_transform(bad_init)
    with pytest.raises(ValueError, match="label_leakage_refused"):
        d.classify_spline_support(
            interior_raw,
            fitted,
            unit_records=[dict(unit_id="u0", annotator1=1)] * 4,
        )
    with pytest.raises(ValueError, match="unit_records_length"):
        d.classify_spline_support(
            interior_raw,
            fitted,
            unit_records=[dict(unit_id="u0")],
        )
    with pytest.raises(ValueError, match="unknown_monotone_transform"):
        d.classify_spline_support(interior_raw, fitted, monotone_transform="bad")
    with pytest.raises(ValueError):
        d.classify_spline_support(np.zeros((0, 7)), fitted)


def test_deferral_and_builder_rejection_controls():
    """SCENARIO-REPORT-9225-V785-CONTROLS: reject bad seeds, probabilities, alphas, regimes, weights and rosters."""
    with pytest.raises(ValueError, match="invalid_seed_identifiers"):
        d.audit_draw("", "q0", "llama3-8b")
    for bad_p in [-0.1, 1.1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="invalid_probability"):
            d.typed_decision(bad_p)
    with pytest.raises(ValueError, match="unknown_regime"):
        d.observation_propensity(0.5, "unknown")
    with pytest.raises(ValueError, match="invalid_alpha"):
        d.observation_propensity(0.5, "escalation_only", alpha=0.1)
    with pytest.raises(ValueError, match="invalid_alpha"):
        d.observation_propensity(0.5, "audited_deferral", alpha=0.0)
    with pytest.raises(ValueError, match="invalid_audit_u"):
        d.observe_feedback(0.5, "audited_deferral", alpha=0.1, audit_u=1.0)
    for bad_obs, regime, alpha in [
        ([], "full_release", 0.0),
        ([dict(unit_id="u0", p=0.5, y=2)], "full_release", 0.0),
        ([dict(unit_id="u0", p=0.5, y=1, weight=0.0)], "full_release", 0.0),
        ([dict(unit_id="u0", p=-0.1, y=1)], "full_release", 0.0),
        ([dict(unit_id="u0", p=0.5, y=1)], "bad_regime", 0.0),
    ]:
        with pytest.raises(ValueError):
            d.estimate_selective_risk(bad_obs, regime, alpha=alpha)
    with pytest.raises(ValueError, match="invalid_monte_carlo_operands"):
        d.monte_carlo_audit_summary([], alpha=0.1, trials=10)
    with pytest.raises(ValueError, match="invalid_stream_roster"):
        d.build_stream_deferral_masks([], {}, {})
    base_labels = {
        f"q{q}__{gen}": dict(
            unit_id=f"q{q}__{gen}",
            qid=f"q{q}",
            group_id=f"g{q}",
            role="TRAIN",
            annotator1=0,
            annotator2=-1,
        )
        for q in range(50)
        for gen in d.GENERATORS
    }
    base_master = {
        f"q{q}__{gen}": dict(
            item_id=f"q{q}__{gen}",
            qid=f"q{q}",
            model=gen,
            question="Q?",
            answer_text="A.",
            annotator1_label="0",
            annotator2_label="-1",
            original_row=idx + 2,
        )
        for idx, (q, gen) in enumerate((q, g) for q in range(50) for g in d.GENERATORS)
    }
    base_streams = [[f"q{5 * s + j}" for j in range(5)] for s in range(10)]
    built = d.build_stream_deferral_masks(base_streams, base_labels, base_master)
    assert len(built["rows"]) == 150
    dup_streams = deepcopy(base_streams)
    dup_streams[1][0] = dup_streams[0][0]
    with pytest.raises(ValueError, match="duplicate_stream_qid"):
        d.build_stream_deferral_masks(dup_streams, base_labels, base_master)
    bad_role_labels = deepcopy(base_labels)
    bad_role_labels["q0__llama3-8b"]["role"] = "TEST"
    with pytest.raises(ValueError, match="stream_member_invalid"):
        d.build_stream_deferral_masks(base_streams, bad_role_labels, base_master)


def test_authentication_failures_and_operands(tmp_path, monkeypatch):
    """REQ-REPORT-9225-V785: missing files, missing fields and observed zero stay distinct."""
    absent = d.inspect(tmp_path / "absent")
    assert absent["checks"][0]["observed"] == "missing_file"
    assert absent["rows"] == []
    original_ref = d.ref
    first_rel = next(iter(d.PINS))
    target_file = tmp_path / first_rel
    target_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads((d.ROOT / first_rel).read_bytes())
    for rel, _ in d.PINS.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((d.ROOT / rel).read_bytes())
    for operand in ["missing_field", 0, False]:
        changed = deepcopy(payload)
        if operand == "missing_field":
            changed.pop("source_ready_score")
        else:
            changed["source_ready_score"] = operand
        target_file.write_text(json.dumps(changed))
        monkeypatch.setitem(d.PINS, first_rel, original_ref(target_file)["sha256"])
        res = d.inspect(tmp_path)
        gate = next(r for r in res["checks"] if r["artifact_field"] == "source_ready_score")
        assert gate["observed"] == operand and not gate["passed"] and not res["rows"]
    target_file.write_text("{bad json")
    monkeypatch.setitem(d.PINS, first_rel, original_ref(target_file)["sha256"])
    res = d.inspect(tmp_path)
    assert not res["rows"] and any(not r["passed"] for r in res["checks"])


def child(tmp_path, *args):
    """Execute actual standalone CLI in a foreign directory without PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(d.ROOT / d.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9225] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180
    )
    print(
        "[test9225] child_after",
        result.returncode,
        result.stdout[-240:],
        result.stderr[-240:],
        flush=True,
    )
    return result


def test_private_cli_tamper_and_deadlines(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9225-V785-CONTROLS: private CLI success, refusal, tamper, cold replay and deadlines."""
    source, output = tmp_path / "work.json", tmp_path / (d.NAME + ".json")
    c.atomic_json(source, work)
    res = child(tmp_path, "--from-work", source, "--output", output)
    assert res.returncode == 0, res.stdout + res.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert (
        value["selective_deferral_ready_score"] == 0
        and value["spline_support_diagnostic_ready_score"] == 0
    )
    value["estimator_control_rows"][0]["passed"] = False
    value["reproducibility_checksum"] = c.checksum(value)
    c.atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    assert child(tmp_path, "--date", "bad").returncode == 1
    assert child(tmp_path, "--from-work", source).returncode == 1
    assert (
        child(
            tmp_path, "--checks-only", "--root", tmp_path / "absent", "--output", output
        ).returncode
        == 0
    )

    assert c.build(work, {})["verdict_class"] == "disqualified"
    monkeypatch.setattr(c, "receipts_pass", lambda *_: True)
    built = c.build(work, {})
    assert built["verdict_class"] == "null"
    assert built["selective_deferral_ready_score"] == 1
    assert built["spline_support_diagnostic_ready_score"] == 1
    assert built["scientific_effect_measured"] is False
    assert built["sample_size_budget"]["effective_independent_groups"] == 50

    blocked_work = c.measure(tmp_path / "absent", tmp_path / "blocked", "20261119")
    blocked_val = c.build(blocked_work, {})
    assert blocked_val["verdict_class"] == "blocked"
    assert blocked_val["inference_substrate_class"] == "blocked_no_run"
    assert blocked_val["sample_size_budget"]["effective_independent_groups"] == 0

    monkeypatch.setattr(
        c.lifecycle,
        "child",
        lambda name, argv, *_, **__: dict(
            name=name, argv=argv, returncode=0, passed=True, log_sha256="sha256:1"
        ),
    )
    monkeypatch.setattr(
        c.lifecycle,
        "measured",
        lambda *_: dict(complete=True, percent=100.0, files={}),
    )
    monkeypatch.setattr(c.lifecycle, "terminal", lambda *_: dict(passed=True))
    qual = c.qualify(work)
    assert len(qual["checks"]) == len(work["scope"]["commands"])
    assert c.terminal(output, tmp_path)["passed"] is True

    monkeypatch.setattr(c, "measure", lambda *_: deepcopy(work))
    monkeypatch.setattr(c, "qualify", lambda _: {})
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + 1)
    monkeypatch.setattr(c, "receipts_pass", lambda *_: False)
    assert c.main(["--output", str(output)]) == 1
    monkeypatch.setattr(c, "receipts_pass", lambda *_: True)
    monkeypatch.setattr(c, "terminal", lambda *_: dict(passed=True))
    assert c.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["selective_deferral_ready_score"] == 1
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + int(3301e9))
    assert c.main(["--output", str(output)]) == 1
    times = iter([work["start_ns"] + 1, work["start_ns"] + int(4801e9)])
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: next(times))
    assert c.main(["--output", str(output)]) == 1


def test_replay_sidecar_and_custody_drift(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9225-V785-CONTROLS: replay refuses forged sidecars, masks, protocols and receipts."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    for key in [
        "current_work_reference",
        "selective_deferral_protocol_reference",
        "spline_support_protocol_reference",
        "selective_deferral_masks_reference",
    ]:
        ref = changed[key]
        p = tmp_path / Path(ref["path"]).name
        p.write_bytes(Path(ref["path"]).read_bytes())
        changed[key] = dict(ref, path=str(p))
    output = tmp_path / (d.NAME + ".json")

    def save():
        c.atomic_json(tmp_path / "work.json", changed)
        val = c.build(changed, {})
        c.atomic_json(output, val)
        return val

    val = save()
    assert c.replay(output)["passed"]
    val["work_reference"]["sha256"] = "sha256:forged"
    c.atomic_json(output, val)
    with pytest.raises(ValueError, match="work_hash"):
        c.replay(output)

    base = deepcopy(changed)
    for key, match in [
        ("current_work_reference", "operand_hash"),
        ("selective_deferral_protocol_reference", "protocol_hash"),
        ("spline_support_protocol_reference", "spline_protocol_hash"),
        ("selective_deferral_masks_reference", "mask_hash"),
    ]:
        changed = deepcopy(base)
        changed[key]["sha256"] = "sha256:forged"
        save()
        with pytest.raises(ValueError, match=match):
            c.replay(output)

    changed = deepcopy(base)
    proto_path = Path(changed["selective_deferral_protocol_reference"]["path"])
    c.atomic_json(proto_path, dict(d.selective_deferral_protocol(), altered=True))
    changed["selective_deferral_protocol_reference"] = dict(
        changed["selective_deferral_protocol_reference"], **d.ref(proto_path)
    )
    save()
    with pytest.raises(ValueError, match="protocol_drift"):
        c.replay(output)
    c.atomic_json(proto_path, d.selective_deferral_protocol())

    changed = deepcopy(base)
    spline_path = Path(changed["spline_support_protocol_reference"]["path"])
    c.atomic_json(spline_path, dict(d.spline_support_protocol(), altered=True))
    changed["spline_support_protocol_reference"] = dict(
        changed["spline_support_protocol_reference"], **d.ref(spline_path)
    )
    save()
    with pytest.raises(ValueError, match="spline_protocol_drift"):
        c.replay(output)
    c.atomic_json(spline_path, d.spline_support_protocol())

    changed = deepcopy(base)
    mask_path = Path(changed["selective_deferral_masks_reference"]["path"])
    original_masks = json.loads(mask_path.read_bytes())
    c.atomic_json(mask_path, dict(rows=[]))
    changed["selective_deferral_masks_reference"] = dict(
        changed["selective_deferral_masks_reference"], **d.ref(mask_path)
    )
    save()
    with pytest.raises(ValueError, match="mask_drift"):
        c.replay(output)
    c.atomic_json(mask_path, original_masks)

    changed = deepcopy(base)
    changed["inspection"]["rows"][0]["audit_alpha_0_10"] ^= 1
    save()
    with pytest.raises(ValueError, match="source_drift"):
        c.replay(output)

    changed = deepcopy(base)
    save()
    monkeypatch.setattr(c, "validate_current_work_receipt", lambda *_: ["forged"])
    with pytest.raises(ValueError, match="current_work_drift"):
        c.replay(output)
