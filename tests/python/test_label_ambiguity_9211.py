"""REQ-REPORT-9211-V784 / SCENARIO-REPORT-9211-V784: target identifiability and label ambiguity bounds."""

from copy import deepcopy
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import label_ambiguity_9211 as d
from carnot.reporting import label_ambiguity_runner_9211 as c


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-9211-V784: measure authenticated human source and controls once without model load."""
    return c.measure(d.ROOT, tmp_path_factory.mktemp("ambiguity9211"), "20261118")


def test_source_custody_targets_and_masks(work):
    """REQ-REPORT-9211-V784: 900 responses, A2=-1 missingness, target audit and both masks."""
    inspected = work["inspection"]
    assert all(r["passed"] for r in inspected["checks"])
    rows = inspected["rows"]
    assert len(rows) == 900
    assert len({r["qid"] for r in rows}) == 300
    assert {role: len({r["qid"] for r in rows if r["role"] == role}) for role in d.ROLES} == d.ROLES
    assert sum(r["annotator2"] == -1 for r in rows) == 600
    assert sum(r["a2_status"] == "observed_agreement" for r in rows) == 271
    assert sum(r["a2_status"] == "observed_disagreement" for r in rows) == 29
    assert sum(r["a2_status"] == "missing_a2" for r in rows) == 600
    assert sum(len(r["observed_disagreement_admissible"]) == 2 for r in rows) == 29
    assert sum(len(r["missing_annotation_envelope_admissible"]) == 2 for r in rows) == 629
    trows = inspected["target_identifiability_rows"]
    assert len(trows) == 2700
    by_target = {t: [r for r in trows if r["target"] == t] for t in d.TARGETS}
    assert all(
        r["status"] == "present" and r["identifiable"] is True for r in by_target["factual_error"]
    )
    assert all(
        r["status"] == "absent" and r["identifiable"] is False
        for r in by_target["expressed_uncertainty"]
    )
    assert all(
        r["status"] == "absent" and r["identifiable"] is False for r in by_target["reelicitation"]
    )
    assert all(r["si_cf_identifiable"] is False for r in trows)
    assert inspected["controls"]["target_summary"]["identifiable_si_cf_rows"] == 0
    assert (
        inspected["controls"]["support_counts"]["ALL"]["observed_disagreement_variable_responses"]
        == 29
    )
    assert (
        inspected["controls"]["support_counts"]["ALL"][
            "missing_annotation_envelope_variable_responses"
        ]
        == 629
    )
    assert (
        inspected["controls"]["support_counts"]["retention100"]["repeated_stream_appearances"]
        == 3000
    )


def test_target_identifiability_rejection_controls():
    """SCENARIO-REPORT-9211-V784: missing provenance, -1 conflation, synthetic and cross-model rejection."""
    rec = dict(
        unit_id="q__llama3-8b",
        qid="q",
        sibling="llama3-8b",
        role="TEST",
        annotator1=1,
        annotator2=-1,
    )
    assert d.evaluate_target_evidence(rec, "factual_error")["decision"] == "source_evidence_present"
    assert (
        d.evaluate_target_evidence(rec, "expressed_uncertainty")["decision"]
        == "source_evidence_absent"
    )
    assert d.evaluate_target_evidence(rec, "reelicitation")["decision"] == "source_evidence_absent"
    assert (
        d.evaluate_target_evidence(rec, "factual_error", provenance_valid=False)["decision"]
        == "source_evidence_absent"
    )
    assert (
        d.evaluate_target_evidence(rec, "expressed_uncertainty", synthetic_category=True)[
            "decision"
        ]
        == "source_evidence_absent"
    )
    assert (
        d.evaluate_target_evidence(rec, "reelicitation", cross_generator_substitution=True)[
            "decision"
        ]
        == "source_evidence_absent"
    )
    assert (
        d.evaluate_target_evidence(rec, "factual_error", conflicting_criterion=True)["decision"]
        == "source_evidence_ambiguous"
    )
    sentinel = dict(rec, annotator1=-1)
    assert (
        d.evaluate_target_evidence(sentinel, "factual_error")["decision"]
        == "source_evidence_ambiguous"
    )
    assert (
        d.evaluate_target_evidence(sentinel, "expressed_uncertainty")["decision"]
        == "source_evidence_ambiguous"
    )
    absent_a1 = dict(rec, annotator1=None)
    assert (
        d.evaluate_target_evidence(absent_a1, "factual_error")["decision"]
        == "source_evidence_absent"
    )
    unverified_re = d.evaluate_target_evidence(rec, "reelicitation", same_generator_prompts=3)
    assert (
        unverified_re["decision"] == "source_evidence_ambiguous"
        and not unverified_re["si_cf_identifiable"]
    )
    verified_re = d.evaluate_target_evidence(
        dict(rec, reelicitation_human_verified=True), "reelicitation", same_generator_prompts=3
    )
    assert (
        verified_re["decision"] == "source_evidence_present" and verified_re["si_cf_identifiable"]
    )
    with pytest.raises(ValueError, match="unknown_target"):
        d.evaluate_target_evidence(rec, "unknown")


def test_exhaustive_endpoint_reducer_and_repeated_retention():
    """SCENARIO-REPORT-9211-V784-CONTROLS: exhaustive binary panel, repeated retention and validation errors."""
    masks = [
        dict(
            unit_id="u0",
            qid="q0",
            sibling="llama3-8b",
            role="TEST",
            annotator1=0,
            annotator2=0,
            a2_status="observed_agreement",
            observed_disagreement_admissible=[0],
            missing_annotation_envelope_admissible=[0],
        ),
        dict(
            unit_id="u1",
            qid="q0",
            sibling="gemma-2-9b",
            role="TEST",
            annotator1=1,
            annotator2=0,
            a2_status="observed_disagreement",
            observed_disagreement_admissible=[0, 1],
            missing_annotation_envelope_admissible=[0, 1],
        ),
        dict(
            unit_id="u2",
            qid="q0",
            sibling="mistral-7b",
            role="TEST",
            annotator1=0,
            annotator2=-1,
            a2_status="missing_a2",
            observed_disagreement_admissible=[0],
            missing_annotation_envelope_admissible=[0, 1],
        ),
    ]
    observations = [
        dict(unit_id="u0", p_control=0.7, p_treatment=0.2, weight=0.25),
        dict(unit_id="u1", p_control=0.8, p_treatment=0.3, weight=0.25),
        dict(unit_id="u1", p_control=0.1, p_treatment=0.9, weight=0.25),
        dict(unit_id="u2", p_control=0.0, p_treatment=1.0, weight=0.25),
    ]
    both = d.reduce_both_scenarios(observations, masks)
    for scenario, key in [
        ("observed_disagreement", "observed_disagreement_admissible"),
        ("missing_annotation_envelope", "missing_annotation_envelope_admissible"),
    ]:
        choices = [m[key] for m in masks]
        totals = []
        for y0, y1, y2 in itertools.product(*choices):
            y_map = {"u0": y0, "u1": y1, "u2": y2}
            val = sum(
                o["weight"]
                * (
                    (o["p_control"] - y_map[o["unit_id"]]) ** 2
                    - (o["p_treatment"] - y_map[o["unit_id"]]) ** 2
                )
                for o in observations
            )
            totals.append(val)
        assert math.isclose(both[scenario]["min_contrast"], min(totals), abs_tol=1e-15)
        assert math.isclose(both[scenario]["max_contrast"], max(totals), abs_tol=1e-15)
    # Per-observation independent label selection on u1 would falsely give a wider bound.
    naive_min = sum(
        min(
            o["weight"] * ((o["p_control"] - y) ** 2 - (o["p_treatment"] - y) ** 2)
            for y in dict(u0=[0], u1=[0, 1], u2=[0])[o["unit_id"]]
        )
        for o in observations
    )
    assert naive_min < both["observed_disagreement"]["min_contrast"] - 0.1
    permuted = d.reduce_both_scenarios(
        list(reversed(observations)), {m["unit_id"]: m for m in masks}
    )
    assert math.isclose(
        permuted["observed_disagreement"]["min_contrast"],
        both["observed_disagreement"]["min_contrast"],
        abs_tol=1e-15,
    )
    duplicated = d.reduce_both_scenarios(
        [dict(o, weight=o["weight"] / 2) for o in observations * 2], masks
    )
    assert math.isclose(
        duplicated["missing_annotation_envelope"]["max_contrast"],
        both["missing_annotation_envelope"]["max_contrast"],
        abs_tol=1e-15,
    )
    swapped = d.reduce_both_scenarios(
        [dict(o, p_control=o["p_treatment"], p_treatment=o["p_control"]) for o in observations],
        masks,
    )
    assert math.isclose(
        swapped["observed_disagreement"]["min_contrast"],
        -both["observed_disagreement"]["max_contrast"],
        abs_tol=1e-15,
    )
    assert math.isclose(
        swapped["observed_disagreement"]["max_contrast"],
        -both["observed_disagreement"]["min_contrast"],
        abs_tol=1e-15,
    )
    constant = d.reduce_both_scenarios(
        [dict(o, p_treatment=o["p_control"]) for o in observations], masks
    )
    assert (
        constant["missing_annotation_envelope"]["min_contrast"]
        == constant["missing_annotation_envelope"]["max_contrast"]
        == 0.0
    )


def test_reducer_and_builder_rejections():
    """SCENARIO-REPORT-9211-V784-CONTROLS: reject nonfinite probabilities, bad weights and malformed masks."""
    mask = [
        dict(
            unit_id="u0",
            qid="q0",
            sibling="llama3-8b",
            role="TEST",
            annotator1=0,
            annotator2=0,
            a2_status="observed_agreement",
            observed_disagreement_admissible=[0],
            missing_annotation_envelope_admissible=[0],
        )
    ]
    assert (
        d.reduce_contrast_endpoints([dict(unit_id="u0", p_control=0.4, p_treatment=0.4)], mask)[
            "width"
        ]
        == 0.0
    )
    for bad_obs, bad_mask, scenario in [
        ([], mask, "observed_disagreement"),
        ([dict(unit_id="u0", p_control=0.5, p_treatment=0.5)], mask, "bad_scenario"),
        ([dict(unit_id="missing", p_control=0.5, p_treatment=0.5)], mask, "observed_disagreement"),
        ([dict(unit_id="u0", p_control=-0.1, p_treatment=0.5)], mask, "observed_disagreement"),
        ([dict(unit_id="u0", p_control=0.5, p_treatment=1.2)], mask, "observed_disagreement"),
        (
            [dict(unit_id="u0", p_control=float("nan"), p_treatment=0.5)],
            mask,
            "observed_disagreement",
        ),
        (
            [dict(unit_id="u0", p_control=0.5, p_treatment=0.5, weight=0.0)],
            mask,
            "observed_disagreement",
        ),
        (
            [dict(unit_id="u0", p_control=0.5, p_treatment=0.5)],
            [dict(mask[0], annotator1=2)],
            "observed_disagreement",
        ),
        (
            [dict(unit_id="u0", p_control=0.5, p_treatment=0.5)],
            [dict(mask[0], observed_disagreement_admissible=[])],
            "observed_disagreement",
        ),
    ]:
        with pytest.raises(ValueError):
            d.reduce_contrast_endpoints(bad_obs, bad_mask, scenario=scenario)
    base_labels = [
        dict(
            unit_id=f"q0__{gen}",
            qid="q0",
            group_id="g0",
            role="TEST",
            annotator1=0,
            annotator2=0,
        )
        for gen in d.GENERATORS
    ]
    master_rows = {
        f"q0__{gen}": dict(
            item_id=f"q0__{gen}",
            qid="q0",
            model=gen,
            question="Q?",
            answer_text="A.",
            annotator1_label="0",
            annotator2_label="0",
            original_row=i + 2,
        )
        for i, gen in enumerate(d.GENERATORS)
    }
    for mutation in [
        "duplicate",
        "bad_a1",
        "bad_a2",
        "missing_master",
        "bad_sibling_count",
        "bad_role",
        "dup_group",
    ]:
        changed_labels = deepcopy(base_labels)
        changed_master = deepcopy(master_rows)
        if mutation == "duplicate":
            changed_labels[1]["unit_id"] = changed_labels[0]["unit_id"]
        elif mutation == "bad_a1":
            changed_labels[0]["annotator1"] = 2
        elif mutation == "bad_a2":
            changed_labels[0]["annotator2"] = 2
        elif mutation == "missing_master":
            changed_master.pop(changed_labels[0]["unit_id"])
        elif mutation == "bad_sibling_count":
            changed_labels.pop()
        elif mutation == "bad_role":
            changed_labels[0]["role"] = "OTHER"
        else:
            changed_labels.extend(
                dict(
                    unit_id=f"q1__{gen}",
                    qid="q1",
                    group_id="g0",
                    role="TEST",
                    annotator1=0,
                    annotator2=0,
                )
                for gen in d.GENERATORS
            )
            for i, gen in enumerate(d.GENERATORS):
                changed_master[f"q1__{gen}"] = dict(
                    item_id=f"q1__{gen}",
                    qid="q1",
                    model=gen,
                    question="Q1?",
                    answer_text="A1.",
                    annotator1_label="0",
                    annotator2_label="0",
                    original_row=i + 5,
                )
        with pytest.raises(ValueError):
            d.build_masks_and_targets(
                changed_labels,
                changed_master,
                dict(path="/tmp/m.csv", sha256="sha256:1"),
                dict(path="/tmp/d.csv", sha256="sha256:2"),
                expected_qids=None,
            )
    with pytest.raises(ValueError, match="whole_qid_count"):
        d.build_masks_and_targets(
            base_labels,
            master_rows,
            dict(path="/tmp/m.csv", sha256="sha256:1"),
            dict(path="/tmp/d.csv", sha256="sha256:2"),
        )


def test_authentication_failures_and_operands(tmp_path, monkeypatch):
    """REQ-REPORT-9211-V784: missing files, missing fields and observed zero stay distinct."""
    absent = d.inspect(tmp_path / "absent")
    assert absent["checks"][0]["observed"] == "missing_file"
    assert absent["rows"] == []
    original_ref = d.ref
    first_rel = next(iter(d.PINS))
    target_file = tmp_path / first_rel
    target_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads((d.ROOT / first_rel).read_bytes())
    for Rel, Pin in d.PINS.items():
        p = tmp_path / Rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((d.ROOT / Rel).read_bytes())
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
    print("[test9211] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180
    )
    print(
        "[test9211] child_after",
        result.returncode,
        result.stdout[-240:],
        result.stderr[-240:],
        flush=True,
    )
    return result


def test_private_cli_tamper_and_deadlines(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9211-V784-CONTROLS: private CLI success, refusal, tamper, cold replay and deadlines."""
    source, output = tmp_path / "work.json", tmp_path / (d.NAME + ".json")
    c.atomic_json(source, work)
    res = child(tmp_path, "--from-work", source, "--output", output)
    assert res.returncode == 0, res.stdout + res.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert (
        value["label_ambiguity_ready_score"] == 0
        and value["target_identifiability_ready_score"] == 0
    )
    value["endpoint_control_rows"][0]["passed"] = False
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
    assert built["label_ambiguity_ready_score"] == 1
    assert built["target_identifiability_ready_score"] == 1
    assert built["scientific_effect_measured"] is False
    assert built["label_exposure_scope"] == "exposed_development_only"
    assert built["sample_size_budget"]["effective_independent_groups"] == 300

    blocked_work = c.measure(tmp_path / "absent", tmp_path / "blocked", "20261118")
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
    assert json.loads(output.read_bytes())["label_ambiguity_ready_score"] == 1
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + int(3301e9))
    assert c.main(["--output", str(output)]) == 1
    times = iter([work["start_ns"] + 1, work["start_ns"] + int(4801e9)])
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: next(times))
    assert c.main(["--output", str(output)]) == 1


def test_replay_sidecar_and_custody_drift(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9211-V784-CONTROLS: replay refuses forged sidecars, masks, rubrics and receipts."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    for key in [
        "current_work_reference",
        "label_ambiguity_protocol_reference",
        "target_rubric_reference",
        "ambiguity_masks_reference",
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
        ("label_ambiguity_protocol_reference", "operand_hash"),
        ("target_rubric_reference", "rubric_hash"),
        ("ambiguity_masks_reference", "mask_hash"),
    ]:
        changed = deepcopy(base)
        changed[key]["sha256"] = "sha256:forged"
        save()
        with pytest.raises(ValueError, match=match):
            c.replay(output)

    changed = deepcopy(base)
    proto_path = Path(changed["label_ambiguity_protocol_reference"]["path"])
    c.atomic_json(proto_path, dict(d.protocol(), altered=True))
    changed["label_ambiguity_protocol_reference"] = d.ref(proto_path)
    save()
    with pytest.raises(ValueError, match="protocol_drift"):
        c.replay(output)
    c.atomic_json(proto_path, d.protocol())

    changed = deepcopy(base)
    rubric_path = Path(changed["target_rubric_reference"]["path"])
    c.atomic_json(rubric_path, dict(d.rubric(), altered=True))
    changed["target_rubric_reference"] = dict(
        changed["target_rubric_reference"], **d.ref(rubric_path)
    )
    save()
    with pytest.raises(ValueError, match="rubric_drift"):
        c.replay(output)
    c.atomic_json(rubric_path, d.rubric())

    changed = deepcopy(base)
    mask_path = Path(changed["ambiguity_masks_reference"]["path"])
    original_masks = json.loads(mask_path.read_bytes())
    c.atomic_json(mask_path, dict(rows=[]))
    changed["ambiguity_masks_reference"] = dict(
        changed["ambiguity_masks_reference"], **d.ref(mask_path)
    )
    save()
    with pytest.raises(ValueError, match="mask_drift"):
        c.replay(output)
    c.atomic_json(mask_path, original_masks)

    changed = deepcopy(base)
    changed["inspection"]["rows"][0]["annotator1"] ^= 1
    save()
    with pytest.raises(ValueError, match="source_drift"):
        c.replay(output)

    changed = deepcopy(base)
    save()
    monkeypatch.setattr(c, "validate_current_work_receipt", lambda *_: ["forged"])
    with pytest.raises(ValueError, match="current_work_drift"):
        c.replay(output)
