"""REQ-REPORT-9166-V781 / SCENARIO-REPORT-9166-V781: immutable binding controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from carnot.reporting import v781_contract as c
from carnot.reporting import v781_contract_evidence as e


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """Freeze real inputs once; a fixture cannot establish scientific benefit."""
    return c.measure(e.ROOT, tmp_path_factory.mktemp("contract9166"), "20261117")


def test_original_authority(work):
    """REQ-REPORT-9166-V781: reuse complete source/method authority without new science."""
    i = work["inspection"]
    assert all(r["passed"] for r in i["checks"]), [r for r in i["checks"] if not r["passed"]]
    assert i["source_ready"] and i["static_ready"] and i["online_ready"]
    assert i["source"]["roles"] == dict(TRAIN=150, CAL=50, TEST=100)
    assert [len(r["qids"]) for r in i["execution_rosters"]] == [100] * 3
    assert [len(r["rows"]) for r in i["execution_rosters"]] == [300] * 3
    assert len(i["static_protocol"]["arms"]) == 5
    assert len(i["static_protocol"]["bounds"]) == 2
    assert len(i["online_protocol"]["arms"]) == 6
    assert len(i["online_protocol"]["benefit_bounds"]["bounds"]) == 7
    assert len(i["previous_outcome_rows"]) == 14
    obligations = i["validation_obligations"]
    assert len(obligations["historical_rows"]) == 674
    assert obligations["historical_counts"] == dict(completed=32, censored=642)
    assert obligations["current_qualification_credit"] == 0
    assert len(obligations["original_node_ids"]) == 621
    assert len(obligations["v779_addition_ids"]) == 53
    assert obligations["assertion_references"] and obligations["affected_current_paths"]
    assert len(obligations["node_bindings"]) == 674
    assert all(r["assertion_reference"]["sha256"] for r in obligations["node_bindings"])
    assert obligations["affected_v780_node_assertions"]
    assert any(
        r["node_id"].startswith(e.TEST + "::") for r in obligations["affected_node_assertions"]
    )
    moment = i["diagnostic_protocol"]["brier_moments"]
    assert moment["identity_tolerance"] == 1e-12 and moment["selection"] == "none"
    assert i["diagnostic_protocol"]["scale_ranking"]["one_class_auroc"] == "unavailable"


def test_mutations_and_absence(tmp_path):
    """SCENARIO-REPORT-9166-V781: prompt, metadata, gates, order and count all bind."""
    for name in [e.DESIGN, e.ACTIVE]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((e.ROOT / name).read_bytes())
    assert e.authority(tmp_path)["ready"]
    original = yaml.safe_load((tmp_path / e.ACTIVE).read_bytes())
    for key in [
        "prompt",
        "MODEL_SPECS",
        "inference_substrate_class",
        "deliverable",
        "gated_on",
        "order",
        "count",
    ]:
        q = deepcopy(original)
        if key == "order":
            q["tasks"].reverse()
        elif key == "count":
            q["tasks"].pop()
        else:
            q["tasks"][1][key] = "forged"
        (tmp_path / e.ACTIVE).write_text(yaml.safe_dump(q))
        assert not e.authority(tmp_path)["ready"]
    (tmp_path / e.ACTIVE).unlink()
    assert e.authority(tmp_path)["checks"][-1]["observed"] == "missing_file"
    (tmp_path / e.ACTIVE).write_text("{}")
    assert e.authority(tmp_path)["checks"][-1]["observed"] == "missing_field"
    for observed in ["missing_file", "missing_field", 0, False, 1]:
        assert e.check(tmp_path, "score", 1, observed, "gate")["passed"] == (
            type(observed) is int and observed == 1
        )


def test_moment_controls():
    """REQ-REPORT-9166-V781: population moments equal direct Brier, even at zero variance."""
    for p, y in [
        ([0.1, 0.8, 0.6], [0, 1, 0]),
        ([0.5] * 3, [0, 1, 0]),
        ([0, 1], [1, 1]),
        ([0], [0]),
    ]:
        v = e.brier_moments(p, y)
        assert abs(v["signed_residual"]) <= 1e-12
        assert v["brier"] == pytest.approx(sum((a - b) ** 2 for a, b in zip(p, y)) / len(p))
    assert e.brier_moments([0.5], [1])["correlation"] is None
    assert e.brier_moments([], [])["status"] == "unavailable"
    for p, y in [([0], []), ([float("nan")], [1]), ([2], [0]), ([0.5], [-1])]:
        with pytest.raises(ValueError, match="moment_support"):
            e.brier_moments(p, y)


def test_build_readiness_and_block(work, tmp_path, monkeypatch):
    """REQ-REPORT-9166-V781: administrative success never promotes censored science."""
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    before = deepcopy(work)
    v = c.build(work, {})
    assert work == before
    assert v["experiment_id"] == 9166 and v["verdict_class"] == "null"
    assert [
        v[k]
        for k in [
            "contract_ready_score",
            "source_ready_score",
            "static_methods_ready_score",
            "online_methods_ready_score",
        ]
    ] == [1] * 4
    assert v["MODEL_SPECS"] == []
    assert v["current_llm_invocations"] == v["generated_tokens"] == 0
    assert v["claim_scope"]["independent_generalization_score"] == 0
    assert all(
        "canonical_task_hash" in t and "previous_outcome_rows" in t for t in v["full_task_objects"]
    )
    assert v["preconditions_checked"] and v["model_invocation_status"] == "not_requested"
    assert all("principle" in gate for gate in v["acceptance_gates"].values())
    changed = deepcopy(work)
    changed["inspection"]["source_ready"] = False
    assert c.build(changed, {})["source_ready_score"] == 0
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: False)
    assert c.build(work, {})["verdict_class"] == "disqualified"
    missing = c.measure(tmp_path / "absent", tmp_path / "raw", "20261117")
    v = c.build(missing, {})
    assert v["verdict_class"] == "blocked" and v["censored_count"] == 14
    assert v["inference_substrate_class"] == "blocked_no_run"
    assert [r["unit_id"] for r in v["rows"]] == [f"exp{n}" for n in range(9166, 9180)]
    monkeypatch.setattr(c.legacy, "qualify", lambda _: dict(delegated=True))
    assert c.qualify(work) == dict(delegated=True)


def test_authenticate_before_labels(monkeypatch):
    """SCENARIO-REPORT-9166-V781: failed producer bytes prevent dependent label reads."""
    original = e.ref

    def absent(path):
        r = original(path)
        if path.name == "experiment_9152_v780_contract.json":
            r["sha256"] = None
        return r

    def forbidden(_):
        raise AssertionError("labels accessed before authentication")

    monkeypatch.setattr(e, "ref", absent)
    monkeypatch.setattr(e.prior, "inspect", forbidden)
    assert not e.inspect(e.ROOT)["source_ready"]


def child(tmp_path, *args):
    """Use a real external process without PYTHONPATH, retaining actual child coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("[test9166] child_before", args, flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=180
    )
    print(
        "[test9166] child_after",
        result.returncode,
        result.stdout[-300:],
        result.stderr[-300:],
        flush=True,
    )
    return result


def test_private_cli(work, tmp_path):
    """SCENARIO-REPORT-9166-V781: E2E-018/021 private success/refusal/tamper/cold replay."""
    source, output = tmp_path / "work.json", tmp_path / (e.NAME + ".json")
    c.atomic_json(source, work)
    result = child(tmp_path, "--from-work", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    v = json.loads(output.read_bytes())
    v["source_ready_score"] = 99
    v["reproducibility_checksum"] = c.legacy.checksum(v)
    c.atomic_json(output, v)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    assert child(tmp_path, "--date", "bad").returncode == 1
    assert child(tmp_path, "--from-work", source).returncode == 1
    assert (
        child(
            tmp_path, "--checks-only", "--root", tmp_path / "absent", "--output", output
        ).returncode
        == 0
    )


def test_sidecar_tamper(work, tmp_path):
    """SCENARIO-REPORT-9166-V781: a fresh hash cannot bless invented diagnostic bytes."""
    changed = deepcopy(work)
    changed["raw"] = str(tmp_path)
    path = tmp_path / "forged.json"
    c.atomic_json(path, dict(invented=True))
    changed["protocol_sidecars"]["diagnostic_protocol_reference"] = e.ref(path)
    c.atomic_json(tmp_path / "work.json", changed)
    output = tmp_path / (e.NAME + ".json")
    c.atomic_json(output, c.build(changed, {}))
    with pytest.raises(ValueError, match="protocol_sidecar_drift"):
        c.replay(output)


def test_main_validation(work, tmp_path, monkeypatch):
    """REQ-REPORT-9166-V781: real main branches require owned checks and bounded time."""
    monkeypatch.setattr(c, "measure", lambda *_: deepcopy(work))
    monkeypatch.setattr(c, "qualify", lambda _: {})
    output = tmp_path / (e.NAME + ".json")
    assert c.main(["--output", str(output)]) == 1
    monkeypatch.setattr(c.legacy, "receipts_pass", lambda *_: True)
    monkeypatch.setattr(c.legacy, "terminal", lambda *_: dict(passed=True))
    assert c.main(["--output", str(output)]) == 0
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: work["start_ns"] + 3400 * 10**9)
    assert c.main(["--output", str(output)]) == 1
    clock = iter([work["start_ns"], work["start_ns"] + 4900 * 10**9])
    monkeypatch.setattr(c.time, "monotonic_ns", lambda: next(clock))
    assert c.main(["--output", str(output)]) == 1
