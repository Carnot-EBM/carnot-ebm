"""REQ-REPORT-8007: preserve roles, threshold ties, custody and cold reduction."""

import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from carnot import experiment_8007_v694_conditioning_diagnosis as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.multivariate_validation_7982 import fixture_data
from carnot.verify import sparse_energy_7996 as sparse


def fixture():
    """Use disjoint artificial identities so the test claims stay circular."""
    data = {k: v for k, v in fixture_data().items() if k in ("fit", "tune")}
    for role, rows in data.items():
        rows.extend(
            dict(rows[i], family_id=f"{role}-extra-{i}", source_cluster_id=f"{role}-extra-{i}")
            for i in range(2)
        )
    scaler = sparse.fit_scaler(sparse.inputs(data["fit"]))
    heads = {
        "spline": [
            dict(
                arm="spline",
                seed=17,
                parameters=[0.0] * 109,
                decay_scale=1.0,
                temperature=1.0,
                scaler=scaler,
                initial_loss=1.0,
                final_loss=1.0,
            )
        ]
    }
    return dict(
        data=data,
        heads=heads,
        slots=[],
        role_manifests={},
        references=[],
        original_exposure_rows=[],
        upstream_checks=[],
    )


def private_primaries(root):
    """Mirror producer contracts with artificial rows, never natural outcome claims."""
    raw = root / "evidence"
    bundle = fixture()
    inputs = raw / "inputs.json"
    heads = raw / "heads.json"
    atomic_json(inputs, dict(data=bundle["data"]))
    atomic_json(heads, dict(heads=bundle["heads"]))
    public, evaluator, slots = {}, {}, []
    for role in ("calibration", "stream", "retention"):
        path = raw / (role + ".json")
        atomic_json(path, dict(features=[dict(family_id=role, source_normalized_hash=role)]))
        public[role] = e.reference(path)
        evaluator[role] = e.reference(path)
        row = dict(
            family_id=role,
            role=role,
            status="generated",
            started=True,
            source_cluster_id=role,
            exclusion_reason=None,
            public_eligible=True,
            raw_response=dict(
                choices=[
                    dict(
                        finish_reason="stop",
                        message=dict(
                            content=json.dumps(
                                dict(unsupported_probability=0.5, source_sentence_id=None)
                            )
                        ),
                    )
                ]
            ),
            visible_ids=[0],
        )
        row["parsed"] = e.capture_helper.risk.transport.parse_response(
            row["raw_response"], row["visible_ids"]
        )
        slots.append(row)
    shardrefs = []
    for i, row in enumerate(slots):
        path = raw / f"capture-{i}.json"
        atomic_json(path, row)
        shardrefs.append(e.reference(path))
    values = {
        7994: dict(
            public_role_manifests=public, evaluator_role_manifests=evaluator, known_exposure_rows=[]
        ),
        7995: dict(
            rows=slots,
            raw_response_shards=shardrefs,
            role_completion_counts=e.capture_helper.reduce(slots)["role_completion_counts"],
        ),
        7996: dict(checkpoints=dict(inputs=e.reference(inputs), heads=e.reference(heads))),
        7999: dict(
            checkpoints=dict(bundle=e.reference(inputs)),
            headroom_diagnostics=dict(selectable_changes=0),
        ),
    }
    for eid, (suffix, ready) in e.INPUTS.items():
        atomic_json(
            root / "results" / f"experiment_{eid}_v693_{suffix}.json",
            dict(
                values[eid],
                experiment_id=eid,
                task_id=f"exp{eid}-fixture",
                honest_verdict="complete_null_fixture",
                verdict_class="null",
                flagged_adversarial=False,
                **{ready: 1},
            ),
        )
    atomic_json(
        root / "results/raw" / e.NAME / "literature_review.json",
        dict(
            method_map=[dict(method=str(i), scope="circular_method_map_fixture") for i in range(5)],
            citations=[dict(evidence=[e.reference(inputs)])],
        ),
    )
    return bundle


def test_custody_contracts(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8007-ISOLATION: verify copies and reject a missing gate field."""
    private_primaries(tmp_path)
    bundle, failures = e.materialize(tmp_path, tmp_path / "owned")
    assert not failures
    assert len(bundle["slots"]) == 3
    bundle["heads"]["ignored_equivalent"] = bundle["heads"]["spline"]
    bundle["heads"]["mlp"] = [dict(bundle["heads"]["spline"][0], arm="mlp", parameters=[0.0] * 89)]
    assert e.diagnose(bundle)["conditioning_rows"]
    capture = next((tmp_path / "results").glob("experiment_7995_*.json"))
    value = json.loads(capture.read_text())
    value["rows"][0]["status"] = "drift"
    atomic_json(capture, value)
    with pytest.raises(ValueError, match="capture_custody"):
        e.materialize(tmp_path, tmp_path / "drift")
    primary = next((tmp_path / "results").glob("experiment_7994_*.json"))
    value = json.loads(primary.read_text())
    value.pop("cohort_ready_score")
    atomic_json(primary, value)
    with pytest.raises(ValueError, match="upstream_contract"):
        e.materialize(tmp_path, tmp_path / "second")
    monkeypatch.setattr(e.shutil, "copyfile", lambda source, target: target.write_text("drift"))
    with pytest.raises(ValueError, match="copy_hash"):
        e.copy_evidence(e.reference(primary), tmp_path / "third")


def test_validation_and_rejection_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8007-PUBLICATION: exercise owned pass/fail orchestration privately."""
    private_primaries(tmp_path)
    fake_pass = [True]

    def supervised(root, commands, **kwargs):
        for command in commands:
            if command.name == "coverage_json":
                atomic_json(
                    Path(command.argv[-1]),
                    dict(
                        files={
                            p: dict(summary=dict(missing_lines=0, num_statements=1))
                            for p in e.OWNED
                        }
                    ),
                )
        return [dict(scope=c.scope, passed=fake_pass[0], name=c.name) for c in commands]

    monkeypatch.setattr(e, "run_commands", supervised)
    assert e.terminal(tmp_path / "evidence/inputs.json")["passed"]
    for passed in (True, False):
        fake_pass[0] = passed
        output = tmp_path / str(passed) / (e.NAME + ".json")
        monkeypatch.setattr(e, "terminal", e.replay)
        assert e.main(["--root", str(tmp_path), "--output", str(output)]) == 0
        value = json.loads(output.read_text())
        assert value["methods_ready_score"] == int(passed)
        value["role_manifests"] = {}
        atomic_json(output, value)
        with pytest.raises(ValueError, match="role_manifests"):
            e.replay(output)
    blocked = e.base([])
    blocked["methods_ready_score"] = 1
    path = tmp_path / "unsafe.json"
    atomic_json(path, blocked)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(path)
    literature = tmp_path / "results/raw" / e.NAME / "literature_review.json"
    atomic_json(literature, dict(method_map=[], citations=[]))
    output = tmp_path / "missingmap" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["gate_check_summary"][0]["artifact_field"] == "method_map"
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **k: dict(passed=False, gate_sha256=None))
    src = tmp_path / "fixture.json"
    atomic_json(src, fixture())
    assert (
        e.main(
            [
                "--fixture-input",
                str(src),
                "--output",
                str(tmp_path / "badreader" / (e.NAME + ".json")),
            ]
        )
        == 1
    )


def test_thresholds_and_response():
    """SCENARIO-REPORT-8007-THRESHOLDS: strict ties and independent synthetic response."""
    assert [e.policy(p) for p in (None, 0, 0.099, 0.1, 0.5, 0.501, 1)] == [
        "escalate",
        "accept",
        "accept",
        "escalate",
        "escalate",
        "reject",
        "reject",
    ]
    assert e.loss("accept", 1) == 5
    assert e.loss("reject", 0) == 1
    assert e.loss("escalate", 1) == 0.5
    bundle = fixture()
    result = e.diagnose(bundle)
    assert result["positive_control_results"]["passed"]
    assert abs(result["intercept_checkpoint"]["gradient_norm"]) < 1e-8
    assert result["statistical_plan"]["draws"] == 10000
    assert result["diagnosis"]["deployment_benefit_established"] is False
    assert all(r["role"] in ("fit", "tune") for r in result["conditioning_rows"])


def test_isolation_and_missing_values():
    """SCENARIO-REPORT-8007-ISOLATION: evaluator changes cannot select optimizer state."""
    bundle = fixture()
    before = e.diagnose(bundle)
    bundle["slots"] = [dict(role="stream", y=1, family_id="private-label")]
    after = e.diagnose(bundle)
    assert before == after
    bundle["data"]["stream"] = []
    with pytest.raises(ValueError, match="role_roster"):
        e.diagnose(bundle)
    bundle = fixture()
    bundle["data"]["fit"][0]["q"] = None
    bundle["data"]["tune"][0]["y"] = None
    result = e.diagnose(bundle)
    assert any(r["status"] == "excluded" and r["denominator"] == 0 for r in result["rows"])
    bad = copy.deepcopy(bundle)
    bad["data"]["tune"][1]["source_cluster_id"] = bad["data"]["fit"][1]["source_cluster_id"]
    with pytest.raises(ValueError, match="cross_role"):
        e.diagnose(bad)


def test_copy_and_reduction(tmp_path):
    """SCENARIO-REPORT-8007-PUBLICATION: byte drift and edited reduction must fail."""
    source = tmp_path / "source.json"
    atomic_json(source, fixture())
    ref = e.reference(source)
    copied = e.copy_evidence(ref, tmp_path / "raw")
    assert e.checked(copied).read_bytes() == source.read_bytes()
    source.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        e.copy_evidence(ref, tmp_path / "raw")
    assert np.isfinite(e.distribution(np.array([0.0, 1.0]))["mean"])


def test_real_cli(tmp_path):
    """SCENARIO-REPORT-8007-PUBLICATION: success, blocked, invalid and cold routes."""
    bundle = tmp_path / "fixture.json"
    atomic_json(bundle, fixture())
    output = tmp_path / "success" / (e.NAME + ".json")
    cli = str(e.ROOT / e.OWNED[-1])
    prefix = [str(e.ROOT / ".venv/bin/python"), "-u", cli]
    if os.environ.get("CARNOT_8007_COVERAGE_FILE"):
        prefix = [
            str(e.ROOT / ".venv/bin/coverage"),
            "run",
            "--parallel-mode",
            "--data-file=" + os.environ["CARNOT_8007_COVERAGE_FILE"],
            "--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED),
            cli,
        ]
    run = subprocess.run(
        prefix + ["--fixture-input", str(bundle), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert e.replay(output)["passed"]
    assert all(Path(r["path"]).is_relative_to(output.parent) for r in value["raw_shard_hashes"])
    cold = subprocess.run(
        prefix + ["--cold-replay", str(output)], capture_output=True, text=True, timeout=120
    )
    assert cold.returncode == 0, cold.stdout + cold.stderr
    value["conditioning_rows"][0]["gradient_norm"] += 1
    atomic_json(output, value)
    assert e.main(["--cold-replay", str(output)]) == 1
    assert e.main(["--fixture-input", str(tmp_path / "missing")]) == 1
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(blocked)]) == 0
    assert json.loads(blocked.read_text())["methods_ready_score"] == 0
    with pytest.raises(SystemExit):
        e.main(["--date", "20261001"])
