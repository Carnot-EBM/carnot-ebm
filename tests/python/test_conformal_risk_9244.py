"""REQ-REPORT-9244-V787 / SCENARIO-REPORT-9244-V787: cluster-valid conformal risk, CA-OCRC tracking and Mycelium provenance."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import conformal_risk_9244 as d
from carnot.reporting import conformal_risk_runner_9244 as c


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-9244-V787: authenticate custody and measure conformal, CA-OCRC and Mycelium controls once."""
    return c.measure(d.ROOT, tmp_path_factory.mktemp("conformal9244"), "20261120")


def test_source_custody_mycelium_ledger_and_ca_ocrc_masks(work):
    """REQ-REPORT-9244-V787: 900 authenticated responses, 300 qids, Mycelium provenance ledger and 150 CA-OCRC masks."""
    inspected = work["inspection"]
    assert all(r["passed"] for r in inspected["checks"])
    rows = inspected["rows"]
    assert len(rows) == 900
    assert len({r["qid"] for r in rows}) == 300
    assert {r["sibling"] for r in rows} == set(d.GENERATORS)
    assert all(r["answer_level_granularity"] == "present" for r in rows)
    assert all(r["span_level_granularity"] == "absent" for r in rows)
    summary = inspected["provenance_summary"]
    assert summary["total_responses"] == 900
    assert summary["total_qids"] == 300
    assert summary["granularity_counts"] == {
        "answer_level_present": 900,
        "span_level_present": 0,
        "span_level_absent": 900,
    }
    assert summary["role_counts"]["TRAIN150"]["responses"] == 450
    assert summary["role_counts"]["TRAIN150"]["qids"] == 150
    assert summary["role_counts"]["CAL50"]["responses"] == 150
    assert summary["role_counts"]["CAL50"]["qids"] == 50
    assert summary["role_counts"]["TEST100"]["responses"] == 300
    assert summary["role_counts"]["TEST100"]["qids"] == 100
    assert sum(summary["dataset_counts"].values()) == 900
    assert summary["generator_counts"] == {g: 300 for g in d.GENERATORS}
    assert (
        summary["typology_counts"]["factual_correct"]
        + summary["typology_counts"]["factual_error_unverified_claim"]
        == 900
    )
    assert summary["typology_counts"]["fine_grained_subtypology_present"] == 0

    mask_rows = inspected["ca_ocrc_mask_rows"]
    assert len(mask_rows) == 150
    assert len({r["qid"] for r in mask_rows}) == 50
    assert len({r["stream_id"] for r in mask_rows}) == 10
    for r in mask_rows:
        expected_draw = d.ca_ocrc_draw(r["stream_id"], r["qid"], r["sibling"])
        assert r["seed_sha256"] == expected_draw["seed_sha256"]
        assert math.isclose(r["tracking_u"], expected_draw["tracking_u"], abs_tol=1e-15)
        assert r["track_c1_d1"] <= r["track_c2_d1"] <= r["track_c5_d1"] == 1

    ctrl = inspected["controls"]
    assert all(r["passed"] for r in ctrl["coverage_control_rows"])
    assert all(r["passed"] for r in ctrl["ca_ocrc_control_rows"])


def test_cluster_conformal_risk_bounds_and_coverage():
    """SCENARIO-REPORT-9244-V787: cluster Clopper-Pearson and betting e-process bounds, edge cases and rho>=0.7 coverage."""
    assert d.clopper_pearson_upper(0, 0) is None
    cp_zero = d.clopper_pearson_upper(0, 50, delta=0.05)
    assert cp_zero is not None and math.isclose(
        cp_zero, 1.0 - (0.05 ** (1.0 / 50.0)), abs_tol=1e-12
    )
    assert d.clopper_pearson_upper(50, 50, delta=0.05) == 1.0

    assert d.betting_e_process_upper([], delta=0.05) is None
    e_zero = d.betting_e_process_upper([0.0] * 50, delta=0.05)
    assert e_zero is not None and 0.0 < e_zero < 0.15
    assert d.betting_e_process_upper([1.0] * 50, delta=0.05) == 1.0

    # 0 accepted qids -> unavailable, never zero-filled.
    no_accept_obs = [
        dict(qid=f"q{q}", sibling=g, p=0.50, y=0) for q in range(10) for g in d.GENERATORS
    ]
    no_acc = d.cluster_conformal_risk_bounds(no_accept_obs, threshold=0.10)
    assert no_acc["status"] == "unavailable"
    assert no_acc["accepted_qids"] == 0
    assert no_acc["empirical_cluster_error_mean"] is None
    assert no_acc["clopper_pearson_cluster_upper"] is None
    assert no_acc["betting_e_process_cluster_upper"] is None
    assert all(
        g["status"] == "unavailable" and g["risk_slack"] is None for g in no_acc["alpha_grid_rows"]
    )

    # 0 errors across 50 accepted qids.
    zero_err_obs = [
        dict(qid=f"q{q:02d}", sibling=g, p=0.02 + 0.01 * (q % 5), y=0)
        for q in range(50)
        for g in d.GENERATORS
    ]
    zero_res = d.cluster_conformal_risk_bounds(zero_err_obs, threshold=0.10)
    assert zero_res["status"] == "ok"
    assert zero_res["accepted_qids"] == 50
    assert zero_res["empirical_cluster_error_mean"] == 0.0
    assert zero_res["clopper_pearson_cluster_upper"] < 0.06
    assert zero_res["betting_e_process_cluster_upper"] < 0.12
    slacks = [g["clopper_pearson_slack"] for g in zero_res["alpha_grid_rows"]]
    cal_taus = [g["calibrated_threshold"] for g in zero_res["alpha_grid_rows"]]
    assert slacks[0] < slacks[1] < slacks[2]
    assert cal_taus[0] <= cal_taus[1] <= cal_taus[2]

    # 100 percent errors across 20 accepted qids.
    all_err_obs = [
        dict(qid=f"q{q:02d}", sibling=g, p=0.05, y=1) for q in range(20) for g in d.GENERATORS
    ]
    all_err_res = d.cluster_conformal_risk_bounds(all_err_obs, threshold=0.10)
    assert all_err_res["empirical_cluster_error_mean"] == 1.0
    assert all_err_res["clopper_pearson_cluster_upper"] == 1.0
    assert all_err_res["betting_e_process_cluster_upper"] == 1.0
    assert all(not g["clopper_pearson_admissible"] for g in all_err_res["alpha_grid_rows"])

    # Synthetic correlated 3-sibling Monte Carlo coverage across 2000 trials (seed 787244).
    for rho in (0.0, 0.7, 0.85, 1.0):
        cov_rows = d.simulate_cluster_coverage(rho, trials=2000, seed=d.SEED)
        assert len(cov_rows) == 2
        for row in cov_rows:
            assert row["cluster_valid_coverage"] >= 0.95
            if rho >= 0.7:
                assert row["naive_sibling_coverage"] < 0.93
            assert row["passed"] is True


def test_ca_ocrc_delayed_feedback_tracking_and_corrupted_bounds():
    """SCENARIO-REPORT-9244-V787: CA-OCRC IPW unbiasedness vs drop-on-full bias for C in [1,2,5] and epsilon in [0.0,0.05]."""
    panel = []
    for s_idx in range(10):
        stream_id = f"stream_{s_idx:02d}"
        for q_idx in range(5):
            qid = f"q{5 * s_idx + q_idx:02d}"
            # Later queries in each stream/delay window have higher true risk so drop-on-full is biased.
            high_slot = q_idx >= 2
            p = 0.08 if high_slot else 0.02
            y = 1 if (high_slot and (s_idx + q_idx) % 2 == 0) else 0
            for sib in d.GENERATORS:
                draw = d.ca_ocrc_draw(stream_id, qid, sib)
                panel.append(
                    dict(
                        unit_id=f"{qid}__{sib}",
                        stream_id=stream_id,
                        stream_index=s_idx,
                        query_index=q_idx,
                        qid=qid,
                        sibling=sib,
                        p=p,
                        y=y,
                        tracking_u=draw["tracking_u"],
                    )
                )

    for cap in d.CA_OCRC_CAPACITIES:
        for delay in d.CA_OCRC_DELAYS:
            for eps in d.CORRUPTED_EPSILONS:
                res = d.ca_ocrc_track_stream(
                    panel,
                    capacity=cap,
                    delay=delay,
                    corrupted_epsilon=eps,
                    seed=d.CA_OCRC_SEED,
                )
                assert math.isclose(res["ipw_risk_bias"], 0.0, abs_tol=1e-15)
                assert abs(res["mc_ipw_risk_bias"]) < 0.015
                if cap < 5 * delay:
                    assert abs(res["naive_risk_bias"]) > 0.05
                else:
                    assert math.isclose(res["naive_risk_bias"], 0.0, abs_tol=1e-15)
                assert (
                    res["corrupted_risk_lower"] - 1e-12
                    <= res["true_risk"]
                    <= res["corrupted_risk_upper"] + 1e-12
                )
                assert math.isclose(
                    res["debiased_corrupted_ipw_risk"], res["true_risk"], abs_tol=1e-15
                )
                assert res["passed"] is True


def test_rejection_and_tamper_controls():
    """SCENARIO-REPORT-9244-V787-CONTROLS: reject bad probabilities, deltas, correlations, capacities, delays and invented spans."""
    for bad_e, bad_n, bad_d in [
        (-1, 5, 0.05),
        (6, 5, 0.05),
        (1.5, 5, 0.05),
        (1, 5, 0.0),
        (1, 5, 1.0),
        (1, 5, float("nan")),
    ]:
        with pytest.raises(ValueError):
            d.clopper_pearson_upper(bad_e, bad_n, bad_d)

    for bad_vals, bad_d, bad_bet in [
        ([-0.1, 0.5], 0.05, 0.5),
        ([0.2, 1.1], 0.05, 0.5),
        ([float("nan")], 0.05, 0.5),
        ([0.2], 0.0, 0.5),
        ([0.2], 0.05, 0.0),
        ([0.2], 0.05, 1.0),
    ]:
        with pytest.raises(ValueError):
            d.betting_e_process_upper(bad_vals, delta=bad_d, bet_fraction=bad_bet)

    valid_obs = [dict(qid="q0", sibling="llama3-8b", p=0.05, y=0)]
    for bad_obs, tau, delta, grid in [
        ([], 0.10, 0.05, (0.05, 0.10)),
        ([dict(qid="", sibling="llama3-8b", p=0.05, y=0)], 0.10, 0.05, (0.05, 0.10)),
        ([dict(qid="q0", sibling="llama3-8b", p=1.2, y=0)], 0.10, 0.05, (0.05, 0.10)),
        ([dict(qid="q0", sibling="llama3-8b", p=0.05, y=2)], 0.10, 0.05, (0.05, 0.10)),
        (valid_obs, -0.1, 0.05, (0.05, 0.10)),
        (valid_obs, 0.10, 0.0, (0.05, 0.10)),
        (valid_obs, 0.10, 0.05, ()),
        (valid_obs, 0.10, 0.05, (0.10, 0.05)),
    ]:
        with pytest.raises(ValueError):
            d.cluster_conformal_risk_bounds(
                bad_obs, threshold=tau, delta=delta, alpha_risk_grid=grid
            )

    for bad_rho, bad_q, bad_s, bad_r, bad_t in [
        (-0.1, 10, 3, 0.1, 10),
        (1.1, 10, 3, 0.1, 10),
        (0.5, 0, 3, 0.1, 10),
        (0.5, 10, 0, 0.1, 10),
        (0.5, 10, 3, 0.0, 10),
        (0.5, 10, 3, 0.1, 0),
    ]:
        with pytest.raises(ValueError):
            d.simulate_cluster_coverage(
                bad_rho, qids=bad_q, siblings=bad_s, true_risk=bad_r, trials=bad_t
            )

    with pytest.raises(ValueError, match="invalid_seed_identifiers"):
        d.ca_ocrc_draw("", "q0", "llama3-8b")

    for bad_obs, cap, delay, eps in [
        ([], 1, 1, 0.0),
        ([dict(qid="q0", p=0.05, y=0, tracking_u=1.5)], 1, 1, 0.0),
        ([dict(qid="q0", p=-0.1, y=0)], 1, 1, 0.0),
        ([dict(qid="q0", p=0.05, y=2)], 1, 1, 0.0),
        ([dict(qid="q0", p=0.05, y=0)], 0, 1, 0.0),
        ([dict(qid="q0", p=0.05, y=0)], 1, 0, 0.0),
        ([dict(qid="q0", p=0.05, y=0)], 1, 1, -0.01),
        ([dict(qid="q0", p=0.05, y=0)], 1, 1, 0.5),
    ]:
        with pytest.raises(ValueError):
            d.ca_ocrc_track_stream(bad_obs, capacity=cap, delay=delay, corrupted_epsilon=eps)

    base_labels = [
        dict(
            unit_id=f"q{q}__{gen}",
            qid=f"q{q}",
            group_id=f"g{q}",
            role="TRAIN" if q < 150 else "CAL" if q < 200 else "TEST",
            annotator1=q % 2,
            annotator2=-1 if q % 3 == 0 else (q % 2),
        )
        for q in range(300)
        for gen in d.GENERATORS
    ]
    base_master = {
        f"q{q}__{gen}": dict(
            item_id=f"q{q}__{gen}",
            dataset="nq" if q % 2 == 0 else "triviaqa",
            qid=f"q{q}",
            model=gen,
            question="Q?",
            answer_text="A.",
            annotator1_label=str(q % 2),
            annotator2_label=str(-1 if q % 3 == 0 else (q % 2)),
            original_row=idx + 2,
        )
        for idx, (q, gen) in enumerate((q, g) for q in range(300) for g in d.GENERATORS)
    }
    base_streams = [[f"q{5 * s + j}" for j in range(5)] for s in range(10)]
    built = d.build_provenance_ledger(base_labels, base_master, base_streams)
    assert len(built["rows"]) == 900
    assert len(built["ca_ocrc_mask_rows"]) == 150

    with pytest.raises(ValueError, match="invented_span_or_model_judge_refused"):
        d.build_provenance_ledger(base_labels, base_master, base_streams, allow_invented_spans=True)
    with pytest.raises(ValueError, match="invented_span_or_model_judge_refused"):
        d.build_provenance_ledger(base_labels, base_master, base_streams, allow_model_judge=True)
    with pytest.raises(ValueError, match="invalid_stream_roster"):
        d.build_provenance_ledger(base_labels, base_master, [])
    dup_streams = deepcopy(base_streams)
    dup_streams[1][0] = dup_streams[0][0]
    with pytest.raises(ValueError, match="duplicate_stream_qid"):
        d.build_provenance_ledger(base_labels, base_master, dup_streams)
    bad_stream_role = deepcopy(base_streams)
    bad_stream_role[0][0] = "q250"
    with pytest.raises(ValueError, match="stream_member_invalid"):
        d.build_provenance_ledger(base_labels, base_master, bad_stream_role)
    with pytest.raises(ValueError, match="whole_qid_count"):
        d.build_provenance_ledger(base_labels[:3], base_master, base_streams)
    dup_labels = deepcopy(base_labels)
    dup_labels[1] = deepcopy(dup_labels[0])
    with pytest.raises(ValueError, match="duplicate_or_label_encoding"):
        d.build_provenance_ledger(dup_labels, base_master, base_streams)
    bad_part = deepcopy(base_labels)
    bad_part[0]["role"] = "CAL"
    with pytest.raises(ValueError, match="whole_qid_partition"):
        d.build_provenance_ledger(bad_part, base_master, base_streams)
    bad_ds_master = deepcopy(base_master)
    bad_ds_master["q0__llama3-8b"]["dataset"] = ""
    with pytest.raises(ValueError, match="invalid_master_provenance"):
        d.build_provenance_ledger(base_labels, bad_ds_master, base_streams)
    bad_span_master = deepcopy(base_master)
    bad_span_master["q0__llama3-8b"]["span_annotations"] = ["fake"]
    with pytest.raises(ValueError, match="invented_span_or_model_judge_refused"):
        d.build_provenance_ledger(base_labels, bad_span_master, base_streams)
    bad_role_count = deepcopy(base_labels)
    for rec in bad_role_count[:3]:
        rec["role"] = "CAL"
    with pytest.raises(ValueError, match="role_cohort_counts"):
        d.build_provenance_ledger(
            bad_role_count,
            base_master,
            [[f"q{5 * s + j + 1}" for j in range(5)] for s in range(10)],
        )


def test_authentication_failures_and_operands(tmp_path, monkeypatch):
    """REQ-REPORT-9244-V787: missing files, missing fields and observed zero stay distinct."""
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
    print("[test9244] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180
    )
    print(
        "[test9244] child_after",
        result.returncode,
        result.stdout[-240:],
        result.stderr[-240:],
        flush=True,
    )
    return result


def test_private_cli_tamper_and_deadlines(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9244-V787-CONTROLS: private CLI success, refusal, tamper, cold replay and deadlines."""
    source, output = tmp_path / "work.json", tmp_path / (d.NAME + ".json")
    c.atomic_json(source, work)
    res = child(tmp_path, "--from-work", source, "--output", output)
    assert res.returncode == 0, res.stdout + res.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert (
        value["conformal_risk_ready_score"] == 0
        and value["ca_ocrc_ready_score"] == 0
        and value["provenance_audit_ready_score"] == 0
    )
    value["coverage_control_rows"][0]["passed"] = False
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
    assert built["conformal_risk_ready_score"] == 1
    assert built["ca_ocrc_ready_score"] == 1
    assert built["provenance_audit_ready_score"] == 1
    assert built["scientific_effect_measured"] is False
    assert built["sample_size_budget"]["effective_independent_groups"] == 300

    blocked_work = c.measure(tmp_path / "absent", tmp_path / "blocked", "20261120")
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
    assert json.loads(output.read_bytes())["conformal_risk_ready_score"] == 1
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + int(3301e9))
    assert c.main(["--output", str(output)]) == 1
    times = iter([work["start_ns"] + 1, work["start_ns"] + int(4801e9)])
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: next(times))
    assert c.main(["--output", str(output)]) == 1


def test_replay_sidecar_and_custody_drift(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9244-V787-CONTROLS: replay refuses forged sidecars, masks, protocols, ledgers and receipts."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    for key in [
        "current_work_reference",
        "conformal_risk_protocol_reference",
        "ca_ocrc_protocol_reference",
        "provenance_ledger_reference",
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
        ("conformal_risk_protocol_reference", "conformal_protocol_hash"),
        ("ca_ocrc_protocol_reference", "ca_ocrc_protocol_hash"),
        ("provenance_ledger_reference", "provenance_ledger_hash"),
    ]:
        changed = deepcopy(base)
        changed[key]["sha256"] = "sha256:forged"
        save()
        with pytest.raises(ValueError, match=match):
            c.replay(output)

    changed = deepcopy(base)
    proto_path = Path(changed["conformal_risk_protocol_reference"]["path"])
    c.atomic_json(proto_path, dict(d.conformal_risk_protocol(), altered=True))
    changed["conformal_risk_protocol_reference"] = dict(
        changed["conformal_risk_protocol_reference"], **d.ref(proto_path)
    )
    save()
    with pytest.raises(ValueError, match="conformal_protocol_drift"):
        c.replay(output)
    c.atomic_json(proto_path, d.conformal_risk_protocol())

    changed = deepcopy(base)
    ca_path = Path(changed["ca_ocrc_protocol_reference"]["path"])
    original_ca = json.loads(ca_path.read_bytes())
    c.atomic_json(ca_path, dict(original_ca, altered=True))
    changed["ca_ocrc_protocol_reference"] = dict(
        changed["ca_ocrc_protocol_reference"], **d.ref(ca_path)
    )
    save()
    with pytest.raises(ValueError, match="ca_ocrc_protocol_drift"):
        c.replay(output)
    c.atomic_json(ca_path, original_ca)

    changed = deepcopy(base)
    ledger_path = Path(changed["provenance_ledger_reference"]["path"])
    original_ledger = json.loads(ledger_path.read_bytes())
    c.atomic_json(ledger_path, dict(rows=[]))
    changed["provenance_ledger_reference"] = dict(
        changed["provenance_ledger_reference"], **d.ref(ledger_path)
    )
    save()
    with pytest.raises(ValueError, match="provenance_ledger_drift"):
        c.replay(output)
    c.atomic_json(ledger_path, original_ledger)

    changed = deepcopy(base)
    changed["inspection"]["rows"][0]["numerator"] ^= 1
    save()
    with pytest.raises(ValueError, match="source_drift"):
        c.replay(output)

    changed = deepcopy(base)
    save()
    monkeypatch.setattr(c, "validate_current_work_receipt", lambda *_: ["forged"])
    with pytest.raises(ValueError, match="current_work_drift"):
        c.replay(output)
