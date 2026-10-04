"""REQ-VERIFY-8116 / REQ-REPORT-8116: private causal memory and terminal routes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import independent_online_memory_8116 as e
from carnot.reporting import independent_memory_execution_8116 as cli


def reference(path, value):
    atomic_json(path, value)
    return dict(path=str(path), sha256=sha256_file(path))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Keep oracle labels private; original missing slots still consume time."""
    root = tmp_path_factory.mktemp("memory8116")
    features, labels = {}, {}
    for role, total, missing in [("stream", 256, 17), ("retention", 64, 3)]:
        rows, targets = [], []
        for slot in range(1, total + 1):
            values = [(-1) ** slot * 1.5, *[float((slot * j) % 19) for j in range(1, 9)]]
            rows.append(
                dict(
                    slot=slot,
                    unit_id=f"{role}-{slot}",
                    source_id=f"{role}-{slot}",
                    source_cluster_id=canonical_hash([role, slot]),
                    values=None if slot > total - missing else values,
                    status="excluded" if slot > total - missing else "completed",
                    exclusion_reason="capture_excluded" if slot > total - missing else None,
                )
            )
            targets.append(
                dict(
                    slot=slot,
                    unit_id=f"{role}-{slot}",
                    source_cluster_id=canonical_hash([role, slot]),
                    y=None if slot == 27 else slot % 2,
                    status="excluded" if slot == 27 else "completed",
                    exclusion_reason="human_unknown" if slot == 27 else None,
                )
            )
        features[role] = reference(root / f"inputs/{role}_features.json", dict(rows=rows))
        labels[role] = reference(root / f"inputs/{role}_labels.json", dict(rows=targets))
    value = dict(
        experiment_id=8111,
        methods_ready_score=1,
        stream_input_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        stream_feature_manifest=features["stream"],
        retention_feature_manifest=features["retention"],
        evaluator_label_manifests=labels,
        raw_shard_hashes=[features["stream"]],
        code_config_hashes={
            "python/carnot/verify/radial_memory_8085.py": sha256_file(Path(e.radial.__file__))
        },
    )
    path = root / e.UPSTREAM
    side = root / "inputs/terminal.json"
    value["terminal_validation_sidecar_path"] = str(side)
    atomic_json(path, value)
    report = reference(
        path.parent / "raw" / path.stem / "validators/report.json",
        dict(primary_sha256=sha256_file(path), report=dict(passed=True)),
    )
    atomic_json(
        side,
        dict(
            normal_process_exit=True,
            publication=dict(primary_sha256=sha256_file(path), sidecar_path=report["path"]),
        ),
    )
    return root


def public(world, role="stream"):
    return json.loads((world / f"inputs/{role}_features.json").read_text())["rows"]


def vault(world, role="stream"):
    return e.LabelVault(world / f"inputs/{role}_labels.json", public(world, role))


def test_zero_genesis_and_geometry(world):
    """SCENARIO-VERIFY-8116-CAUSAL: future public values cannot steer genesis."""
    rows = public(world)
    state = e.genesis(rows, 101)
    changed = deepcopy(rows)
    changed[64]["values"] = [999.0] * 9
    modified = e.genesis(changed, 101)
    assert modified["geometry"] == state["geometry"]
    assert modified["arms"] == state["arms"]
    assert all(a["intercept"] == 0 and a["weights"] == [0.0] * 16 for a in state["arms"].values())
    assert len({canonical_hash(a["centers"]) for a in state["arms"].values()}) == 1
    assert e.probability(
        state["arms"]["frozen"], state["geometry"], rows[0]["values"]
    ) == pytest.approx(1 / (1 + np.exp(-1.5 * -1)))
    with pytest.raises(ValueError, match="original_slots"):
        e.genesis(rows[:-1], 101)
    changed[0]["y"] = 1
    with pytest.raises(ValueError, match="public_label"):
        e.genesis(changed, 101)
    with pytest.raises(ValueError, match="warmup"):
        e.genesis([dict(r, values=None) if r["slot"] <= 64 else r for r in rows], 101)


def test_delay_growth_shared_labels_and_tail(world):
    """REQ-VERIFY-8116: every update has four steps and precedes no label."""
    state = e.genesis(public(world), 101)
    access = vault(world)
    e.execute(state, public(world), access)
    assert state["arms"]["frozen"]["weights"] == [0.0] * 16
    assert state["arms"]["frozen"]["intercept"] == 0
    training = {}
    for update in state["updates"]:
        assert update["label_slot"] + 8 == update["clock_slot"]
        assert update["label_slot"] % 4 != 0
        assert update["steps"] == 4
        assert max(map(abs, update["after_weights"])) <= 4
        assert max(update["gradient_norms"]) <= 1 + 1e-12
        training.setdefault(update["arm"], []).append(update["label_slot"])
    assert len(training) == 3 and len({tuple(v) for v in training.values()}) == 1
    assert all(r["label_slot"] <= 248 for r in state["updates"])
    assert state["cursor"] == 257
    assert {a: len(v["centers"]) for a, v in state["arms"].items()}["error"] == len(
        state["arms"]["random"]["centers"]
    )
    assert all(r["label_slot"] % 4 == 0 for r in state["admissions"])
    assert len({r["label_slot"] for r in state["admissions"]}) == len(state["admissions"])
    assert all(len(a["centers"]) <= 28 for a in state["arms"].values())
    assert any(r["reason"] == "human_unknown" for r in state["rejected_labels"])
    assert any(r["kind"] == "missed_growth" for r in state["proposals"])
    assert len(state["issued"]) == 256


@pytest.mark.parametrize("boundary", ["issued", "candidate_commit", "admission"])
def test_restart_exact_boundaries(world, boundary):
    """SCENARIO-VERIFY-8116-RECOVERY: JSON recovery preserves future predictions."""
    rows = public(world)
    reference_state = e.genesis(rows, 104)
    e.execute(reference_state, rows, vault(world))
    state = e.genesis(rows, 104)
    e.execute(state, rows, vault(world), stop_event=boundary)
    assert state["cursor"] <= 256
    restored = e.restore(json.loads(json.dumps(state)), rows)
    e.execute(restored, rows, vault(world))
    assert restored == reference_state
    restored["baseline_hash"] = "changed"
    with pytest.raises(ValueError, match="baseline_hash"):
        e.restore(restored, rows)


def test_future_label_and_admission_guards(world):
    """REQ-VERIFY-8116: future access and growth capacity mismatch fail closed."""
    access = vault(world)
    with pytest.raises(ValueError, match="future_label"):
        access.release(10, 11, sealed=False)
    with pytest.raises(ValueError, match="prediction_seal"):
        access.release(1, 9, sealed=False)
    assert access.release(1, 9, sealed=True)["y"] == 1
    state = e.genesis(public(world), 101)
    rows = public(world)
    e.execute(state, rows, vault(world), stop_event="candidate_commit")
    assert set(state["candidates"]) == {"random", "error"}
    assert len(state["candidates"]["error"]["head"]["centers"]) == 20
    assert len(state["candidates"]["random"]["head"]["centers"]) == 20
    assert len(state["proposals"][-1]["pool"]) == 32
    for candidate in state["candidates"].values():
        candidate["scores"] = [1.0] * 8
        candidate["current_scores"] = [0.0] * 8
        candidate["frozen_scores"] = [0.0] * 8
    e.admit(state, 120)
    assert state["decisions"][-1]["accepted"] is False
    assert all(len(state["arms"][a]["centers"]) == 16 for a in ["error", "random"])
    state["candidates"] = {}
    e.admit(state, 121)
    assert len(state["decisions"]) == 1


def test_measure_build_and_replay(world, tmp_path):
    """REQ-REPORT-8116: raw reductions bind exact bytes and source denominators."""
    raw = tmp_path / "raw"
    work = e.measure(world, raw, fixture=True, seeds=[101])
    assert work["execution_ready"] == 1
    assert len(work["retention_predictions"]) == 64
    assert work["resume_checks"] and all(r["passed"] for r in work["resume_checks"])
    receipts = [dict(name="private", passed=True)]
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    value = e.build(work, raw, receipts, fixture=True)
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert value["learning_trajectory_ready_score"] == 1
    assert value["independent_count"] <= 320
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    value["completed_count"] += 1
    atomic_json(path, value)
    assert not e.replay(path)
    value = e.build(work, raw, [dict(passed=False)], fixture=True)
    assert (
        value["verdict_class"] == "disqualified" and value["learning_trajectory_ready_score"] == 0
    )
    assert not e.replay(tmp_path / "absent.json")


def test_external_block(world, tmp_path):
    """REQ-REPORT-8116: actual missing upstream operand closes terminal blocked."""
    work = e.measure(tmp_path / "missing", tmp_path / "raw", fixture=True, seeds=[101])
    value = e.build(work, tmp_path / "raw", [dict(passed=True)], fixture=True)
    assert value["verdict_class"] == "blocked"
    failed = next(r for r in value["gate_check_summary"] if not r["passed"])
    assert failed["observed"] is False
    assert str(tmp_path / "missing" / e.UPSTREAM) == failed["path"]
    assert value["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert value["completed_count"] == 0


def test_private_cli_routes(world, tmp_path):
    """SCENARIO-REPORT-8116-TERMINAL: real script works outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if os.environ.get("COVERAGE_RCFILE"):
        env["COVERAGE_PROCESS_START"] = env["COVERAGE_RCFILE"]
    prefix = [sys.executable, "-u", str(e.ROOT / e.CLI)]

    def run(args):
        result = subprocess.run(
            prefix + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        print(result.stdout, result.stderr, flush=True)
        return result

    for route in ["success", "blocked", "mutation"]:
        output = tmp_path / route / (e.NAME + ".json")
        args = [
            "--fixture-output",
            str(output),
            "--root",
            str(world if route != "blocked" else tmp_path / "absent"),
        ]
        if route == "mutation":
            args += ["--mutation", "future-label"]
        assert run(args).returncode == 0
        value = json.loads(output.read_text())
        assert (
            value["verdict_class"]
            == {"success": "circular_positive", "blocked": "blocked", "mutation": "disqualified"}[
                route
            ]
        )
        assert run(["--cold-replay", str(output)]).returncode == 0
        value["rows"][0]["numerator"] = 987
        atomic_json(output, value)
        assert run(["--cold-replay", str(output)]).returncode == 1
    assert run(["--date", "wrong"]).returncode == 2
    assert run(["--fixture-output", str(e.ROOT / "results/forbidden.json")]).returncode == 2


def test_manifest_frozen(tmp_path):
    """REQ-REPORT-8116: validation tools and code coverage are fixed before work."""
    commands = cli.manifest(tmp_path, tmp_path / "candidate.json")
    assert commands["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    assert any(
        "test_primary_publication_7928.py" in " ".join(c["argv"]) for c in commands["commands"]
    )
    assert cli.OWNED == [e.MODULE, e.RUNNER, e.CLI]


def test_replay_rejects_rehashed_causal_mutation(world, tmp_path):
    """SCENARIO-REPORT-8116-TERMINAL: consistent hashes cannot certify wrong SGD."""
    raw = tmp_path / "raw"
    work = e.measure(world, raw, fixture=True, seeds=[101])
    receipts = [dict(passed=True)]
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    work["update_rows"][0]["steps"] = 5
    causal = json.loads((raw / "causal_evidence.json").read_text())
    causal["update_rows"] = work["update_rows"]
    atomic_json(raw / "causal_evidence.json", causal)
    for ref in work["raw_shard_hashes"]:
        if ref["path"].endswith("/causal_evidence.json"):
            ref["sha256"] = sha256_file(raw / "causal_evidence.json")
    atomic_json(raw / "measurement.json", work)
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, e.build(work, raw, receipts, fixture=True))
    assert not e.replay(path)


def test_guard_operands(world, tmp_path):
    """REQ-VERIFY-8116: malformed evaluator rows and empty growth fail explicitly."""
    rows = public(world)
    state = e.genesis(rows, 101)
    assert not e.propose(state, 96)
    assert state["proposals"][-1]["reason"] == "empty_pool_or_capacity"
    bad = tmp_path / "bad.json"
    atomic_json(bad, dict(rows=[]))
    with pytest.raises(ValueError, match="evaluator_slots"):
        e.LabelVault(bad, rows)
    targets = json.loads((world / "inputs/stream_labels.json").read_text())
    targets["rows"][0]["unit_id"] = "changed"
    atomic_json(bad, targets)
    with pytest.raises(ValueError, match="evaluator_identity"):
        e.LabelVault(bad, rows).release(1, 9, sealed=True)


def test_owned_cli_supervision(world, tmp_path, monkeypatch):
    """REQ-REPORT-8116: owned child receipts control readiness and preserve failures."""
    real_measure = e.measure
    monkeypatch.setattr(
        e, "measure", lambda root, raw: real_measure(root, raw, fixture=True, seeds=[101])
    )
    rejected = [False]
    seen = []

    def child(root, spec, private, durable, heartbeat_s):
        seen.append(spec["name"])
        if spec["name"] == "measurement":
            assert cli.main(spec["argv"][3:]) == 0
        return dict(
            name=spec["name"],
            passed=not (rejected[0] and spec["name"] == "strict_row_lint"),
            argv=spec["argv"],
            actual_exit=0,
            normal_exit=True,
        )

    monkeypatch.setattr(cli, "run_check", child)
    for fail in [False, True]:
        rejected[0] = fail
        output = tmp_path / str(fail) / (e.NAME + ".json")
        assert cli.main(["--root", str(world), "--output", str(output)]) == 0
        value = json.loads(output.read_text())
        assert value["verdict_class"] == ("disqualified" if fail else "null")
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        assert (raw / "failed_terminal_candidate.json").exists() is fail
        assert value["global_health"]["passed"]
    assert "repository_full_suite" in seen and "cold_replay" in seen

    def conflict(*args):
        raise ValueError("conflicting_identity")

    monkeypatch.setattr(cli, "publish_primary", conflict)
    with pytest.raises(ValueError, match="conflicting_identity"):
        cli.publish(value, output, tmp_path, raw, [], True)


def test_cold_replay_guard_branches(world, tmp_path, monkeypatch):
    """REQ-REPORT-8116: independently reject each changed operand and reduction."""
    raw = tmp_path / "raw"
    work = e.measure(world, raw, fixture=True, seeds=[101])
    receipts = [dict(passed=True)]
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    path = tmp_path / (e.NAME + ".json")

    def save(candidate):
        atomic_json(raw / "measurement.json", candidate)
        atomic_json(path, e.build(candidate, raw, receipts, fixture=True))

    save(work)
    altered = deepcopy(work)
    altered["source_artifact_hashes"][0]["sha256"] = "changed"
    save(altered)
    assert not e.replay(path)
    altered = deepcopy(work)
    altered["reductions"] = {}
    save(altered)
    assert not e.replay(path)
    altered = deepcopy(work)
    altered["issued_state_rows"][0]["prediction_hash"] = "changed"
    save(altered)
    assert not e.replay(path)
    altered = deepcopy(work)
    altered["rows"][0]["numerator"] = 99
    primitive = dict(rows=altered["rows"])
    atomic_json(raw / "primitive_rows.json", primitive)
    for ref in altered["raw_shard_hashes"]:
        if ref["path"].endswith("/primitive_rows.json"):
            ref["sha256"] = sha256_file(raw / "primitive_rows.json")
    save(altered)
    assert not e.replay(path)
    assert not e.verify_trajectory(dict(work, final_state_hashes={"101": "changed"}))
    assert not e.verify_trajectory(dict(work, retention_predictions=[]))
    assert e.verify_trajectory(dict(work, execution_ready=0))
    atomic_json(raw / "primitive_rows.json", dict(rows=work["rows"]))
    save(work)
    receipts.append(dict(passed=True, log_path=str(path), log_sha256="changed"))
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    save(work)
    assert not e.replay(path)


def test_failed_recovery_zeroes_readiness(world, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8116-RECOVERY: failed owned recovery is disqualified."""
    real_execute = e.execute

    def execute(state, rows, vault, **kwargs):
        real_execute(state, rows, vault, **kwargs)
        if kwargs.get("stop_event"):
            state["seed"] = -1

    monkeypatch.setattr(e, "execute", execute)
    work = e.measure(world, tmp_path / "raw", fixture=True, seeds=[101])
    assert work["owned_failure"] and not work["execution_ready"]


def test_zero_weight_candidates_and_matched_acceptance(world):
    """REQ-VERIFY-8116: growth adds no extra training and preserves capacity parity."""
    state = e.genesis(public(world), 101)
    e.execute(state, public(world), vault(world), stop_event="candidate_commit")
    for arm, candidate in state["candidates"].items():
        assert candidate["head"]["weights"][-4:] == [0.0] * 4
        assert candidate["head"]["optimizer_step"] == state["arms"][arm]["optimizer_step"]
        candidate["scores"] = [0.0] * 8
        candidate["current_scores"] = [0.0] * 8
        candidate["frozen_scores"] = [0.0] * 8
    assert e.admit(state, 136)
    assert len(state["arms"]["error"]["centers"]) == len(state["arms"]["random"]["centers"]) == 20


def test_resume_receipts_are_durable(world, tmp_path):
    """SCENARIO-VERIFY-8116-RECOVERY: restart evidence binds durable checkpoint bytes."""
    work = e.measure(world, tmp_path / "raw", fixture=True, seeds=[101])
    for receipt in work["resume_checks"]:
        assert sha256_file(Path(receipt["checkpoint_path"])) == receipt["checkpoint_sha256"]
