"""REQ-VERIFY-8111 / REQ-REPORT-8111: private immutable custody and CLI routes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import methods_stream_custody_8111 as e
from carnot.verify import development_methods_8098 as m
from carnot.verify import qwen_learning_stream_capture_8102 as c
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def ref(path, value):
    atomic_json(path, value)
    return dict(path=str(path), sha256=sha256_file(path))


def primary(path, value):
    side = path.parent / "raw" / path.stem / "terminal.json"
    value.update(
        terminal_validation_sidecar_path=str(side),
        required_checks_passed=True,
        flagged_adversarial=False,
        verdict_class="null",
        code_config_hashes=value.get("code_config_hashes", {}),
    )
    atomic_json(path, value)
    report = path.parent / "raw" / path.stem / "validators/report.json"
    ref(report, dict(primary_sha256=sha256_file(path), report=dict(passed=True, checks=[])))
    ref(side, dict(publication=dict(primary_sha256=sha256_file(path), sidecar_path=str(report))))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Full private source slots exercise selection without borrowing real labels."""
    root = tmp_path_factory.mktemp("custody8111")
    data = root / "data/ragtruth"
    data.mkdir(parents=True)
    sources, responses = [], []
    for i in range(640):
        sources.append(
            dict(
                source_id=str(i),
                task_type="Summary",
                source_info=f"source{i} fact{i} evidence{i} item{i} token{i}.",
            )
        )
        responses.append(
            dict(
                id=str(i),
                source_id=str(i),
                split="train",
                model="fixture",
                response="Answer.",
                quality="good",
                labels=[],
            )
        )
    for name, rows in [("source_info", sources), ("response", responses)]:
        (data / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    cohort = m.seal(root, root / "evidence/cohort")
    primary(
        root / e.COHORT,
        dict(
            cohort,
            experiment_id=8098,
            code_config_hashes={
                "python/carnot/verify/development_methods_8098.py": sha256_file(Path(m.__file__))
            },
            source_artifact_hashes=[
                dict(path=str(data / (n + ".jsonl")), sha256=sha256_file(data / (n + ".jsonl")))
                for n in ["source_info", "response"]
            ],
        ),
    )
    views = {r: json.loads(Path(cohort["role_manifests"][r]["path"]).read_text()) for r in c.ROLES}
    frozen = c.freeze(views)
    raw = dict(
        model=c.risk.MODEL,
        choices=[
            dict(
                message=dict(
                    content=json.dumps(dict(unsupported_probability=0.25, source_sentence_id=None))
                ),
                finish_reason="stop",
            )
        ],
        usage=dict(prompt_tokens=10, completion_tokens=10),
    )
    rows = [
        dict(
            s,
            raw_response=raw,
            parsed=c.risk.transport.parse_response(raw, s["visible_ids"]),
            started=True,
            status="completed",
            exclusion_reason=None,
        )
        for s in frozen
    ]
    reduced = c.reduce(rows)
    directory = root / "evidence/stream"
    refs = [
        ref(directory / "primitive_rows.json", dict(rows=rows)),
        ref(directory / "capture_manifest.json", dict(rows=frozen)),
        ref(directory / "stream_features.json", dict(rows=reduced["stream_features"])),
        ref(directory / "retention_features.json", dict(rows=reduced["retention_features"])),
    ]
    primary(
        root / e.STREAM,
        dict(
            reduced,
            experiment_id=8102,
            raw_shard_hashes=refs,
            stream_feature_manifest=refs[2],
            retention_feature_manifest=refs[3],
            raw_completion_hashes=[canonical_hash(raw) for _ in rows],
        ),
    )
    return root


def test_sealed_protocol_and_source_level_rows(world, tmp_path):
    """SCENARIO-VERIFY-8111: cached stream is independent of failed fit capture."""
    work = e.measure(world, tmp_path / "raw", fixture=True)
    assert work["methods_ready_score"] == work["stream_input_ready_score"] == 1
    assert len(work["rows"]) == 640
    p = work["protocol"]
    assert p["delayed_memory"]["fitted_head_required"] is False
    assert p["delayed_memory"]["maximum_centers"] == 28
    assert p["delayed_memory"]["sgd_steps_per_update"] == 4
    assert p["statistical_plan"]["one_sided_confidence"] == 0.975
    value = e.build(work, tmp_path / "raw", [dict(passed=True)], fixture=True)
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == value["call_ledger"] == value["trained_head_specs"] == []
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["completed_count"] == 640
    assert value["inference_substrate_class"] == "no_model_load"
    assert e.reduce_rows(value["rows"])["independent_count"] == 640
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert value["memory_genesis"]["residual_coefficients"] == [0.0] * 17


def test_missing_stream_preserves_methods(world, tmp_path):
    """SCENARIO-VERIFY-8111: external missing evidence is terminal blocked."""
    work = e.measure(world, tmp_path / "raw", fixture=True, stream_path=tmp_path / "absent.json")
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["methods_ready_score"] == 1 and value["stream_input_ready_score"] == 0
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert any(
        r["path"] == str(tmp_path / "absent.json") and r["observed"] is False
        for r in value["gate_check_summary"]
    )
    bad = e.build(work, tmp_path / "raw", [dict(passed=False)])
    assert bad["verdict_class"] == "disqualified" and bad["methods_ready_score"] == 0


@pytest.mark.parametrize("mutation", ["labels", "roles", "slots", "source"])
def test_public_custody_mutations(world, tmp_path, mutation):
    """SCENARIO-VERIFY-8111: preserve E2E-015/019 immutable-source regressions."""
    work = e.measure(world, tmp_path / "raw", fixture=True, mutation=mutation)
    assert work["owned_failure"] and work["methods_ready_score"] == 0
    assert e.build(work, tmp_path / "raw", [dict(passed=True)])["verdict_class"] == "disqualified"


def test_protocol_rejects_incomplete_and_changed_contract(tmp_path):
    """REQ-VERIFY-8111: numerical choices cannot drift silently."""
    bad = tmp_path / "design.md"
    bad.write_text("missing protocol")
    with pytest.raises(ValueError):
        e.protocol(bad)
    bad.write_text(
        e.DESIGN.read_text().replace('"sgd_steps_per_update":4', '"sgd_steps_per_update":5')
    )
    with pytest.raises(ValueError, match="protocol_contract"):
        e.protocol(bad)
    with pytest.raises(ValueError, match="source_denominator"):
        e.reduce_rows([dict(unit_id="same", source_cluster_id="same", denominator=2)])


def cli(argv, cwd):
    """A real external-CWD child also records CLI statements under coverage."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [sys.executable]
    if env.get("COVERAGE_RCFILE"):
        command += ["-m", "coverage", "run", "--rcfile=" + env["COVERAGE_RCFILE"]]
    command += [str(e.ROOT / e.CLI), *argv]
    print("before private CLI subprocess", flush=True)
    done = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    print("after private CLI subprocess exit=" + str(done.returncode), flush=True)
    return done


def test_cli_success_block_mutation_and_cold_replay(world, tmp_path):
    """SCENARIO-REPORT-8111: all outputs are private; real CLI runs outside checkout."""
    output = tmp_path / (e.NAME + ".json")
    base = ["--date", "20261004", "--root", str(world), "--fixture-output", str(output)]
    done = cli(base, tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    value = json.loads(output.read_text())
    assert value["methods_ready_score"] == value["stream_input_ready_score"] == 1
    done = cli(["--cold-replay", str(output)], tmp_path)
    assert done.returncode == 0 and "replay_passed" in done.stdout
    assert e.replay(output)
    for key in ["rows", "method_config", "validation_receipts", "code_config_hashes"]:
        forged = deepcopy(value)
        if key == "rows":
            forged[key][0]["numerator"] = 0
        else:
            forged[key] = {}
        atomic_json(output, forged)
        assert not e.replay(output)
    atomic_json(output, value)
    snapshot = Path(value["source_artifact_hashes"][0]["snapshot_path"])
    old = snapshot.read_bytes()
    snapshot.write_bytes(b"changed bytes")
    assert not e.replay(output)
    snapshot.write_bytes(old)
    for extra, verdict in [
        (["--stream-path", str(tmp_path / "absent")], "blocked"),
        (["--mutation", "slots"], "disqualified"),
    ]:
        destination = tmp_path / verdict / (e.NAME + ".json")
        done = cli(base[:-1] + [str(destination)] + extra, tmp_path)
        assert done.returncode == 0, done.stdout + done.stderr
        assert json.loads(destination.read_text())["verdict_class"] == verdict
    atomic_json(output, dict(value, completed_count=0))
    assert cli(["--cold-replay", str(output)], tmp_path).returncode == 1
    assert cli(["--date", "20261005"], tmp_path).returncode == 2


def test_empty_external_cohort_and_source_hash(world, tmp_path):
    """REQ-REPORT-8111: missing original operand is preserved, never replaced."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    assert work["methods_ready_score"] == work["stream_input_ready_score"] == 0
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["intended_count"] == value["excluded_count"] == 640
    assert len(value["rows"]) == 640 and value["verdict_class"] == "blocked"
    assert not e.replay(tmp_path / "nonexistent")


def test_normal_supervisor_worker_and_validator_paths(world, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8111: normal child supervision and validator reduction."""
    from carnot.reporting import methods_stream_execution_8111 as runner

    output = tmp_path / "supervised" / (e.NAME + ".json")
    logs = tmp_path / "test-log"
    logs.write_text("private stubbed validation supervisor\n")

    def check(root, spec, private, durable, heartbeat_s):
        if spec["name"] == "measurement":
            target = Path(spec["argv"][-1])
            e.measure(world, target.parent, fixture=True)
        return dict(
            name=spec["name"],
            argv=spec["argv"],
            passed=True,
            exit_code=0,
            normal_exit=True,
            duration_s=0,
            log_path=str(logs),
            log_sha256=sha256_file(logs),
        )

    monkeypatch.setattr(runner, "run_check", check)
    assert runner.main(["--root", str(world), "--output", str(output)]) == 0
    assert e.replay(output)
    logs.write_text("changed log")
    assert not e.replay(output)
    worker = tmp_path / "worker/measurement.json"
    done = cli(["--root", str(tmp_path / "empty"), "--worker-output", str(worker)], tmp_path)
    assert done.returncode == 0 and worker.is_file()
    assert (
        cli(["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))], tmp_path).returncode
        == 2
    )


@pytest.mark.parametrize("change", ["completion", "label", "mask"])
def test_historical_stream_reduction_mutations(world, tmp_path, change):
    """REQ-VERIFY-8111: authentic hashes still require independently valid semantics."""
    path = world / e.STREAM
    original = json.loads(path.read_text())
    candidate = deepcopy(original)
    raw = tmp_path / "primitives.json"
    rows = json.loads(Path(original["raw_shard_hashes"][0]["path"]).read_text())["rows"]
    if change == "completion":
        rows[0]["raw_response"]["choices"][0]["message"]["content"] = "invalid"
    if change == "label":
        rows[0]["human_target"] = 1
    if change == "mask":
        candidate["original_slot_mask"]["stream"][0] = False
    candidate["raw_shard_hashes"][0] = ref(raw, dict(rows=rows))
    candidate["raw_shard_hashes"][0]["path"] = str(raw.rename(tmp_path / "primitive_rows.json"))
    try:
        primary(path, candidate)
        work = e.measure(world, tmp_path / "raw", fixture=True)
        assert work["methods_ready_score"] == 1 and work["stream_input_ready_score"] == 0
        assert any(not r["passed"] for r in work["gate_check_summary"])
    finally:
        primary(path, original)


def test_frozen_measurement_argv_and_terminal_failure(world, tmp_path, monkeypatch):
    """REQ-REPORT-8111: terminal owned failure publishes disqualified zero readiness."""
    from carnot.reporting import methods_stream_execution_8111 as runner

    work = e.measure(world, tmp_path / "raw", fixture=True)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    atomic_json(tmp_path / "raw/validation_receipts.json", dict(rows=[dict(passed=True)]))
    output = tmp_path / "published" / (e.NAME + ".json")
    failure_log = tmp_path / "failed.log"
    failure_log.write_text("owned terminal verifier failure")
    monkeypatch.setattr(
        runner,
        "run_check",
        lambda *a, **k: dict(
            name="owned_terminal",
            passed=False,
            exit_code=1,
            normal_exit=True,
            log_path=str(failure_log),
            log_sha256=sha256_file(failure_log),
        ),
    )
    runner.publish(value, output, tmp_path, tmp_path / "raw", [dict(name="owned_terminal")], False)
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert json.loads(output.read_text())["methods_ready_score"] == 0
    with pytest.raises(ValueError, match="primary_name"):
        runner.publish(value, tmp_path / "invalid.json", tmp_path, tmp_path / "raw", [], False)
