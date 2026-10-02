"""REQ-REPORT-8006: original custody and issued states determine recovery."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import independent_replay_8006 as replay
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def fixture():
    """Private labels test the reader without supplying historical evidence."""
    issue = dict(
        family_id="a",
        source_cluster_id="source",
        arm="interleaved",
        delay=20,
        issue_slot=1,
        due_slot=21,
        phase=0,
        prediction_set=[0],
        issue_alpha=0.1,
        eligibility=True,
        status="completed",
        seed=1,
    )
    unknown = dict(
        issue,
        family_id="unknown",
        source_cluster_id="unknown-source",
        issue_slot=2,
        due_slot=22,
        phase=1,
        prediction_set=[],
        eligibility=False,
        status="excluded",
    )
    feedback = [
        dict(issue, y=0, error=0, release_slot=21, base_alpha=0.1, alpha_after=0.101),
        dict(unknown, y=None, error=None, release_slot=22, base_alpha=0.1, alpha_after=0.1),
    ]
    roles = {
        "stream": [dict(source_cluster_id="source")],
        "calibration": [dict(source_cluster_id="other")],
    }
    static = dict(
        prior_exposure_receipt=dict(predictions_sealed_before_stream_label_access=False),
        role_hashes=replay.disjoint(roles),
        label_access_events=[
            dict(role="stream", sha256="target-hash", preceded_by=dict(sha256="seal-hash"))
        ],
        rows=[
            dict(
                family_id="a",
                source_cluster_id="source",
                arm="scalar",
                seed=1,
                probability=0.1,
                y=0,
                actual_cost=0,
                action="accept",
                status="completed",
            )
        ],
    )
    return dict(
        static=static,
        confidence=dict(issued_state_rows=[issue, unknown], rows=feedback),
        bundle=dict(targets={"a": 0, "unknown": None}),
        roles=roles,
        public=dict(stream=[dict(family_id="a", source_cluster_id="source")]),
        predictions=[dict(family_id="a", arm="scalar", seed=1, probability=0.1)],
        stream_seal_sha256="seal-hash",
        target_sha256="target-hash",
        original_stream_seal_present=False,
        producer_hashes={"7997": "static", "8000": "confidence"},
    )


def test_independent_readers_and_unknown_minimal_row():
    """REQ-REPORT-8006: excluded null errors explain the original failed assertion."""
    source = fixture()
    value = replay.reduce(source)
    assert value["static_reader_ready_score"] == 1
    assert value["historical_static_recovered_score"] == 0
    assert value["confidence_recovered_score"] == 1
    assert value["issued_error_rows"][1]["recomputed_error"] is None
    failure = value["failed_operand_rows"][0]
    assert failure["original_assertion"] == "issued_error_drift"
    assert failure["observed"] is None and failure["old_reader_expected"] == 1
    assert value["rows"][0]["original_exposure_label"] == "known_early_stream_target_access"
    clean = deepcopy(source)
    clean["original_stream_seal_present"] = True
    clean["static"]["prior_exposure_receipt"]["predictions_sealed_before_stream_label_access"] = (
        True
    )
    assert replay.reduce(clean)["historical_static_recovered_score"] == 1
    assert (
        replay.reduce(source)["independent_reduction_rows"] == value["independent_reduction_rows"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("error", 1),
        ("y", 1),
        ("prediction_set", [1]),
        ("release_slot", 20),
        ("base_alpha", 0.4),
        ("alpha_after", 0.4),
    ],
)
def test_corrupted_feedback_rejected(field, value):
    """SCENARIO-REPORT-8006-REPLAY: changed issued outcomes cannot qualify."""
    source = fixture()
    source["confidence"]["rows"][0][field] = value
    with pytest.raises(ValueError):
        replay.reduce(source)


def test_phase_join_duplicate_and_static_custody_mutations():
    """REQ-REPORT-8006: path, role and issue joins are separate from early access."""
    for mutate in (
        lambda s: s["confidence"]["issued_state_rows"][0].update(phase=1),
        lambda s: s["confidence"]["issued_state_rows"].append(
            s["confidence"]["issued_state_rows"][0]
        ),
        lambda s: s["confidence"]["rows"].reverse(),
        lambda s: s["roles"]["calibration"][0].update(source_cluster_id="source"),
        lambda s: s.update(stream_seal_sha256="wrong"),
        lambda s: s.update(target_sha256="wrong"),
        lambda s: s["predictions"][0].update(probability=0.2),
        lambda s: s["static"]["rows"][0].update(family_id="missing"),
        lambda s: s["static"].update(role_hashes={}),
    ):
        source = fixture()
        mutate(source)
        with pytest.raises(ValueError):
            replay.reduce(source)
    source = fixture()
    source["confidence"]["rows"] = []
    result = replay.reduce(source)
    assert all(r["censor_reason"] == "pending_at_stream_end" for r in result["issued_error_rows"])
    del source["static"]["prior_exposure_receipt"]
    with pytest.raises(KeyError):
        replay.reduce(source)


def test_private_cli_and_cold_mutations(tmp_path):
    """SCENARIO-REPORT-8006-REPLAY: real CLI final bytes and durable checkpoints are checked."""
    source = tmp_path / "fixture.json"
    atomic_json(source, fixture())
    output = tmp_path / (replay.NAME + ".json")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    base = [str(replay.ROOT / ".venv/bin/python"), "-u", str(replay.ROOT / replay.OWNED[1])]
    if env.get("CARNOT_8006_COVERAGE"):
        base = [
            str(replay.ROOT / ".venv/bin/coverage"),
            "run",
            "--parallel-mode",
            "--data-file=" + env["CARNOT_8006_COVERAGE"],
            "--include=" + ",".join(str(replay.ROOT / p) for p in replay.OWNED),
            base[-1],
        ]

    def run(argv):
        print("[test8006] subprocess_before", flush=True)
        result = subprocess.run(
            base + argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        print(f"[test8006] subprocess_after exit={result.returncode}", flush=True)
        return result

    args = ["--fixture-input", str(source), "--validation-worker", "--output", str(output)]
    result = run(args)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["replay_reader_ready_score"] == 0
    assert run(["--cold-replay", str(output)]).returncode == 0
    original = output.read_bytes()
    atomic_json(output, {**value, "confidence_recovered_score": 0})
    assert run(["--cold-replay", str(output)]).returncode == 1
    output.write_bytes(original)
    checkpoint = Path(value["checkpoint_references"][0]["path"])
    saved = checkpoint.read_bytes()
    checkpoint.write_text("changed")
    assert run(["--cold-replay", str(output)]).returncode == 1
    checkpoint.write_bytes(saved)
    assert run(["--date", "bad"]).returncode == 2
    assert run(["--fixture-input", str(source), "--validation-worker"]).returncode == 2
    bad = fixture()
    bad["confidence"]["rows"][0]["error"] = 1
    atomic_json(source, bad)
    assert run(args).returncode == 1
    assert (
        run(
            ["--root", str(tmp_path / "absent"), "--validation-worker", "--output", str(output)]
        ).returncode
        == 0
    )
    blocked = json.loads(output.read_text())
    assert blocked["verdict_class"] == "blocked" and blocked["replay_reader_ready_score"] == 0
    assert run(["--cold-replay", str(output)]).returncode == 0
    assert canonical_hash(json.loads(checkpoint.read_text()))


def historical_fixture(root, original_seal=False):
    """Private producer-shaped records exercise original request and log custody."""
    source = fixture()
    evidence = root / "original"

    def save(name, value):
        path = evidence / (name + ".json")
        atomic_json(path, value)
        return replay.reference(path)

    checkpoints = {
        "roles": save("roles", source["roles"]),
        "public": save("public", source["public"]),
        "stream_predictions": save("stream_predictions", dict(rows=source["predictions"])),
    }
    for name in ("heads", "validation_manifest"):
        save(name, {})
    prior = dict(
        source["static"]["prior_exposure_receipt"],
        original_task_configuration=save("task_configuration", {}),
        original_calibration_predictions=save("calibration_predictions", {}),
        missing_original_stream_predictions=str(evidence / "original_stream.json"),
    )
    if original_seal:
        save("original_stream", dict(rows=source["predictions"]))
    prior_ref = save("prior", prior)
    target_ref = save(
        "targets",
        dict(rows=[dict(family_id=k, y=v) for k, v in source["bundle"]["targets"].items()]),
    )
    static = dict(
        source["static"],
        checkpoints=checkpoints,
        prior_exposure_receipt=prior,
        prior_exposure_reference=prior_ref,
        evaluator_targets=dict(stream=target_ref),
        honest_verdict="complete_disqualified_original",
        verdict_class="disqualified",
    )
    static["label_access_events"][0].update(
        sha256=target_ref["sha256"], preceded_by=checkpoints["stream_predictions"]
    )
    log = evidence / "failure.log"
    log.write_text("stream_label_custody\n")
    code = evidence / "producer.py"
    code.write_text("print('original producer')\n")
    static["code_config_hashes"] = [replay.reference(code)]
    static["validation_receipts"] = [
        dict(
            name="stream_target_exposure_order",
            log_path=str(log),
            log_sha256=sha256_file(log),
            command_argv=["original", str(evidence / "prior.json")],
        )
    ]
    confidence = dict(
        source["confidence"],
        primitive_bundle=save("bundle", source["bundle"]),
        point_prediction_seal=save("points", {}),
        restart_state_checkpoint=save("restart", {}),
        honest_verdict="complete_null_original",
        verdict_class="null",
        code_config_hashes={str(code): sha256_file(code), "config": "unused"},
    )
    capstone = dict(
        honest_verdict="complete_blocked_original",
        verdict_class="blocked",
        independent_reduction_rows=[
            dict(task_id="exp8000-delayed-confidence", reduction_error="issued_error_drift")
        ],
    )
    for eid, value in ((7997, static), (8000, confidence), (8004, capstone)):
        atomic_json(root / "results" / (replay.SOURCES[eid] + ".json"), value)
    return source


def test_source_snapshot_and_original_seal_paths(tmp_path):
    """REQ-REPORT-8006: durable copies bind requests, original failures and targets."""
    root = tmp_path / "root"
    historical_fixture(root)
    source, refs, cites, gates = replay.load_sources(root, tmp_path / "saved")
    assert len(cites) == 3 and len(gates) == 2
    assert replay.reduce(source)["confidence_recovered_score"] == 1
    assert all(Path(r["path"]).is_relative_to(tmp_path / "saved") for r in refs)
    assert (tmp_path / "saved/role_budget_freeze.json").is_file()
    assert any(r["original_path"].endswith("producer.py") for r in refs)
    historical_fixture(root, original_seal=True)
    assert len(replay.load_sources(root, tmp_path / "sealed")[3]) == 1
    primary = root / "results" / (replay.SOURCES[8000] + ".json")
    value = json.loads(primary.read_text())
    bundle = Path(value["primitive_bundle"]["path"])
    atomic_json(bundle, dict(targets={"changed": 1}))
    value["primitive_bundle"] = replay.reference(bundle)
    atomic_json(primary, value)
    with pytest.raises(ValueError, match="original_target_drift"):
        replay.load_sources(root, tmp_path / "bad-target")


def test_owned_dispatch_and_readiness(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8006-REPLAY: coverage and owned failures control consumer readiness."""
    root = tmp_path / "root"
    historical_fixture(root)
    output = tmp_path / (replay.NAME + ".json")
    real_execute = replay.validation.execute
    should_pass = [True]

    def execute(manifest, raw):
        if manifest["commands"][0]["name"] == "cold_reduce":
            return real_execute(manifest, raw)
        report = Path(manifest["environment"]["CARNOT_8006_COVERAGE"]).parent / "coverage.json"
        atomic_json(
            report,
            dict(
                files={
                    p: dict(summary=dict(num_statements=1, missing_lines=0)) for p in replay.OWNED
                }
            ),
        )
        return [
            dict(
                required=True,
                passed=should_pass[0],
                name="private_control",
                exit_code=int(not should_pass[0]),
            )
        ]

    monkeypatch.setattr(replay.validation, "execute", execute)
    args = ["--root", str(root), "--output", str(output)]
    assert replay.main(args) == 0
    assert json.loads(output.read_text())["replay_reader_ready_score"] == 1
    assert replay.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    atomic_json(output, {**value, "duration_s": -1})
    assert replay.main(["--cold-replay", str(output)]) == 1
    should_pass[0] = False
    assert replay.main(args) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(replay, "reader_receipt", lambda *a, **k: dict(passed=False))
    assert replay.main(args) == 1
