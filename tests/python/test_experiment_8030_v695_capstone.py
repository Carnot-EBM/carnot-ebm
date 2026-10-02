"""REQ-REPORT-8030: private controls cannot repair natural scientific failures."""

import copy
import json
from pathlib import Path
import runpy
import sys
import subprocess
import os

import numpy as np
import pytest

from carnot.reporting import v695_capstone as cap
from carnot.reporting import v695_capstone_reduction as red
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def reference(path, value):
    atomic_json(path, value)
    return dict(path=str(path), sha256=sha256_file(path))


def static_fixture(tmp_path, gain=True):
    predictions, targets, tune = [], [], []
    for i in range(200):
        y = int(i < 30)
        targets.append(dict(family_id=str(i), eligible_y=y))
        for arm, p in (("conditioned_energy", 0.9 if y else 0.01), ("linear", 0.01)):
            if not gain:
                p = 0.01
            row = dict(
                family_id=str(i),
                source_cluster_id=str(i),
                arm=arm,
                condition="posthoc",
                seed=17,
                p=p,
                y=None,
                role="stream",
            )
            predictions.append(row)
            tune.append(dict(row, role="tune", y=y))
    measurement = reference(tmp_path / "measurement.json", dict(primitive_rows=predictions + tune))
    seal = reference(
        tmp_path / "seal.json",
        dict(
            original_seal=canonical_hash(predictions),
            measurement=measurement,
            comparator="linear/posthoc",
        ),
    )
    access = reference(
        tmp_path / "access.json",
        dict(
            prediction_receipt=seal,
            stream_access_monotonic_ns=2,
            retention_labels_opened=False,
            target_reference=reference(tmp_path / "targets.json", dict(rows=targets)),
        ),
    )
    seal_value = json.loads(Path(seal["path"]).read_text())
    seal_value["invocation_seal_checked_monotonic_ns"] = 1
    seal = reference(Path(seal["path"]), seal_value)
    access_value = json.loads(Path(access["path"]).read_text())
    access_value["prediction_receipt"] = seal
    access = reference(Path(access["path"]), access_value)
    return dict(
        rows=[{}],
        checkpoint_references=[
            reference(
                tmp_path / "inputs.json",
                dict(predictions=predictions, targets=targets, comparator="linear/posthoc"),
            )
        ],
        prediction_seal=seal,
        label_access_receipt=access,
    )


def test_static_raw_seal_and_null(tmp_path):
    """SCENARIO-REPORT-8030-REDUCTION: exact costs precede producer summaries."""
    data = static_fixture(tmp_path)
    result = red.independent(data, 8021)
    assert result["primary"]["gain"] == pytest.approx(0.75)
    assert result["diagnostic"]["acceptance_gate_results"]["support"]
    assert red.independent(static_fixture(tmp_path / "null", False), 8021)["primary"]["gain"] == 0
    data.pop("prediction_seal")
    with pytest.raises(KeyError):
        red.independent(data, 8021)


def test_source_missing_and_full_tokens(monkeypatch, tmp_path):
    """SCENARIO-REPORT-8030-REDUCTION: absent source evidence is never a win."""
    assert red.independent({}, 8024)["primary"]["raw_p"] == 1
    monkeypatch.setattr(
        red.source,
        "reduce",
        lambda panel, rows: dict(
            passed=True,
            source_feature_rows=[
                dict(family_id="a", status="completed", source_cluster_id="source-a")
            ],
        ),
    )
    monkeypatch.setattr(red.source, "predict", lambda head, rows: np.array([head["p"]]))
    bundle = dict(
        panel=dict(rows=[dict(family_id="a", q=0.0)]),
        token_rows=[{}],
        targets=[dict(family_id="a", eligible_y=1)],
        treatment_head=dict(p=0.9),
        comparator_head=dict(p=0.1),
        prediction_seal=canonical_hash(
            dict(
                features=[
                    dict(family_id="a", status="completed", q=0.0, source_cluster_id="source-a")
                ],
                treatment=[0.9],
                comparator=[0.1],
            )
        ),
    )
    data = dict(rows=[{}], checkpoint_references=[reference(tmp_path / "source.json", bundle)])
    assert red.independent(data, 8024)["primary"]["gain"] == pytest.approx(0.8)
    bundle["prediction_seal"] = "bad"
    data["checkpoint_references"] = [reference(tmp_path / "source.json", bundle)]
    with pytest.raises(ValueError, match="source_prediction_seal"):
        red.independent(data, 8024)
    assert red.independent({}, 0) == dict(measurement_available=False)


def test_learning_seed_mean_and_censor(monkeypatch, tmp_path):
    """SCENARIO-REPORT-8030-REDUCTION: repeat seeds do not enlarge independent n."""
    rows = [
        dict(arm=arm, seed=s, slot=i, cost=c, eligibility=i != 37)
        for s in (17, 29)
        for i in range(36, 40)
        for arm, c in (("uniform", 1.0), ("decision_loss", 0.5))
    ]
    monkeypatch.setattr(
        red.learning,
        "reduce",
        lambda b: dict(
            independent_issue_rows=rows,
            retention_support=dict(passed=True),
            benefit_rows=[],
            retention_drift_rows=[],
        ),
    )
    data = dict(rows=[{}], audit_bundle=reference(tmp_path / "bundle.json", {}))
    result = red.independent(data, 8026)
    assert result["primary"]["gain"] == 0.5
    assert result["primary"]["independent"] == 1
    assert result["primary"]["eligible_slots"] == 3
    assert result["primary"]["slot_count"] == 4


@pytest.mark.parametrize("diff", [[], [float("nan")], [0, 0], [-1, -1], [1, 1], [0, 1, 2]])
def test_bootstrap_inversion(diff):
    """SCENARIO-REPORT-8030-REDUCTION: Holm uses actual bootstrap test inversion."""
    result = red.bootstrap(diff, 32)
    assert 0 <= result["raw_p"] <= 1
    family = red.family([result, result, red.bootstrap([])], [True, False, False])
    assert len(family) == 3
    assert family[1]["family_p"] == family[2]["family_p"] == 1
    assert not family[1]["positive_claim"]


def test_build_cli_and_cold_fixture(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8030-SEAL: private CLI terminalizes once and rejects drift."""
    monkeypatch.setattr(
        cap.authority.authority, "reader_rows", lambda raw, digest: [dict(passed=True)] * 13
    )
    root = tmp_path / "fixture"
    cap.prepare_fixture(root)
    output = root / "results/experiment_8030_v695_capstone.json"
    argv = ["--date", "20261002", "--root", str(root), "--output", str(output), "--fixture-e2e"]
    assert cap.main(argv) == 0
    value = json.loads(output.read_text())
    assert len(value["task_dispositions"]) == 13
    assert value["sample_size_budget"]["completed"] == 13
    assert value["verdict_class"] == "null"
    assert cap.main(["--cold-replay", str(output)]) == 0
    altered_manifest = reference(
        root / "altered_commands.json", dict(dependency_hashes={cap.OWNED[0]: "bad"})
    )
    changed = copy.deepcopy(value)
    changed["checkpoint_references"].append(dict(altered_manifest, role="command_freeze"))
    assert "code_configuration_changed" in cap.cold_replay(changed)
    changed = copy.deepcopy(value)
    changed["primary_hypothesis_results"][0]["gain"] = 999
    atomic_json(output, changed)
    assert cap.main(["--cold-replay", str(output)]) == 1
    atomic_json(output, value)
    value["science_ready"] = True
    assert cap.cold_replay(value)
    value = json.loads(output.read_text())
    value["verdict_class"] = "blocked"
    assert cap.cold_replay(value)
    assert cap.main(["--cold-replay", str(root / "absent")]) == 1
    with pytest.raises(ValueError, match="date"):
        cap.main(["--date", "20261003", "--fixture-e2e", "--root", str(root)])
    monkeypatch.setattr(sys, "argv", [cap.CLI, "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(cap.ROOT / cap.CLI), run_name="__main__")
    assert exited.value.code == 0


def test_custody_failure_paths_and_owned_validation(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8030-BOUNDARIES: absent, corrupted and failed inputs stay terminal."""
    monkeypatch.setattr(
        cap.authority.authority, "reader_rows", lambda raw, digest: [dict(passed=True)] * 13
    )
    cap.prepare_fixture(tmp_path)
    tasks = cap.yaml.safe_load((tmp_path / "active.yaml").read_bytes())["tasks"]
    path = tmp_path / tasks[3]["deliverable"]
    data = cap.read(path)
    data["validation_receipts"] = [dict(name="owned", passed=False)]
    data["rows"] = [{}]
    atomic_json(path, data)
    rows, refs, failed, audits = cap.collect(tmp_path, tasks)
    assert any(r["artifact_field"] == "validation_receipts.owned.passed" for r in failed)
    assert audits[3]["measurement_available"] is False
    path.write_text("[]")
    assert cap.collect(tmp_path, tasks)[0][3]["verdict_class"] == "disqualified"
    path.unlink()
    assert cap.collect(tmp_path, tasks)[0][3]["producer_status"] == "missing"
    skip = tmp_path / "results/experiment_8021_typed_decision_test.json"
    atomic_json(skip, dict(honest_verdict="blocked_gate_check_failed", status="blocked"))
    assert cap.collect(tmp_path, tasks)[0][3]["producer_status"] == "conductor_skip_receipt"
    prior = tmp_path / "prior.json"
    side = tmp_path / "prior-sidecar.json"
    atomic_json(
        prior,
        dict(
            honest_verdict="complete_blocked_authority", terminal_validation_sidecar_path=str(side)
        ),
    )
    atomic_json(side, dict(primary_sha256=sha256_file(prior)))
    (tmp_path / "research-complete.yaml").write_text(
        cap.yaml.safe_dump(
            dict(
                milestones=[
                    dict(tasks=[dict(id="exp7992-contract-methods", deliverable="prior.json")])
                ]
            )
        )
    )
    log = tmp_path / "old_failure.log"
    log.write_text("preserved failed assertion")
    atomic_json(
        tmp_path / "results/experiment_8017_v694_capstone.json",
        dict(
            validation_receipts=[
                dict(
                    name="old_failed", passed=False, log_path=str(log), log_sha256=sha256_file(log)
                ),
                dict(name="old_passed", passed=True),
            ]
        ),
    )
    value = cap.build(tmp_path, "20261002", tmp_path / "raw")
    counts = {p: dict(num_statements=2, covered_lines=2) for p in cap.OWNED}
    cap.apply_checks(value, [dict(passed=True)], counts)
    assert value["verdict_class"] == "blocked" and value["capstone_execution_ready_score"] == 0
    cap.apply_checks(value, [dict(passed=False)], counts)
    assert value["verdict_class"] == "disqualified"
    with pytest.raises(ValueError, match="date"):
        cap.build(tmp_path, "wrong", tmp_path / "raw")
    atomic_json(tmp_path / "raw/claim_seal.json", cap.claim_payload(value))
    value["checkpoint_references"].append(
        cap.reference(tmp_path / "raw/claim_seal.json", "claim_seal")
    )
    value["canonical_tasks_sha256"] = "bad"
    assert "authority_drift" in cap.cold_replay(value)
    skip.write_text("{}")
    assert cap.cold_replay(value) == ["source_bytes_changed"]


def test_reducer_negative_contracts(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8030-REDUCTION: changed seals and support fail at the original field."""
    data = static_fixture(tmp_path)
    access = json.loads(Path(data["label_access_receipt"]["path"]).read_text())
    access["stream_access_monotonic_ns"] = 0
    data["label_access_receipt"] = reference(Path(data["label_access_receipt"]["path"]), access)
    with pytest.raises(ValueError, match="access_contract"):
        red.independent(data, 8021)
    access["target_reference"] = reference(tmp_path / "bad_target.json", dict(rows=[]))
    data["label_access_receipt"] = reference(Path(data["label_access_receipt"]["path"]), access)
    with pytest.raises(ValueError, match="original_targets"):
        red.independent(data, 8021)
    monkeypatch.setattr(red.source, "reduce", lambda *a: dict(passed=False))
    source = dict(
        rows=[{}],
        checkpoint_references=[reference(tmp_path / "source.json", dict(panel={}, token_rows=[]))],
    )
    with pytest.raises(ValueError, match="complete_answer"):
        red.independent(source, 8024)
    monkeypatch.setattr(
        red.targets_reader,
        "reduce",
        lambda b: dict(rows=[{}], support_by_role={}, sample_size_budget={}),
    )
    data = dict(
        rows=[{}],
        support_by_role={},
        checkpoint_references=[reference(tmp_path / "eligible.json", {})],
    )
    assert red.independent(data, 8019)["target_orientation"].startswith("one")
    data["support_by_role"] = {"changed": True}
    with pytest.raises(ValueError, match="annotation_drift"):
        red.independent(data, 8019)


def test_main_owned_plan_and_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8030-SEAL: exact-byte publication and command failures are tested privately."""
    monkeypatch.setattr(
        cap.authority.authority, "reader_rows", lambda raw, digest: [dict(passed=True)] * 13
    )
    cap.prepare_fixture(tmp_path)
    publication = dict(
        paper_ready=False,
        unmet_gates=["G2"],
        gates={f"G{i}": dict(pass_=False) for i in range(1, 5)},
    )
    publication["gates"] = {f"G{i}": {"pass": i != 2} for i in range(1, 5)}

    def run(root, spec, private, durable):
        log = durable / (spec["name"] + ".log")
        atomic_json(log, publication if spec["name"] == "publication_gate" else {})
        atomic_json(
            private / "coverage.json",
            dict(
                files={p: dict(summary=dict(num_statements=2, covered_lines=2)) for p in cap.OWNED}
            ),
        )
        return dict(
            spec, passed=True, log_path=str(log), log_sha256=sha256_file(log), actual_exit=0
        )

    monkeypatch.setattr(cap, "run_check", run)
    output = tmp_path / "results/experiment_8030_v695_capstone.json"
    assert cap.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    value = cap.read(output)
    assert value["g1"] and not value["g2"] and not value["science_ready"]
    assert cap.read(Path(value["terminal_validation_sidecar_path"]))["passed"]
    monkeypatch.setattr(cap, "reader_receipt", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="reader_drift"):
        cap.publish(value, output, tmp_path / "private", tmp_path / "durable")
    monkeypatch.setattr(
        cap, "run_check", lambda *a, **k: dict(passed=False, log_path=str(tmp_path / "missing"))
    )
    with pytest.raises(ValueError, match="terminal_validation"):
        cap.publish(value, output, tmp_path / "private", tmp_path / "durable")


def test_nested_and_directory_terminal_custody(tmp_path):
    """SCENARIO-REPORT-8030-SEAL: existing publication reports authenticate original primary bytes."""
    cap.prepare_fixture(tmp_path)
    tasks = cap.yaml.safe_load((tmp_path / "active.yaml").read_bytes())["tasks"]
    for i in (7, 8):
        path = tmp_path / tasks[i]["deliverable"]
        data = cap.read(path)
        raw = path.parent / "raw" / path.stem
        terminal = raw / "terminal.json" if i == 7 else raw / "validators"
        data["terminal_validation_sidecar_path"] = str(terminal)
        data["validation_receipts"] = [dict(name="health", passed=False, scope="repository_health")]
        atomic_json(path, data)
        bound = raw / "validators" / (sha256_file(path)[7:] + ".json")
        atomic_json(
            bound,
            dict(
                primary_path=str(path), primary_sha256=sha256_file(path), report=dict(passed=True)
            ),
        )
        if i == 7:
            atomic_json(terminal, dict(publication=dict(sidecar_path=str(bound))))
        assert cap.collect(tmp_path, tasks)[0][i]["eligible"]
        atomic_json(
            bound, dict(primary_path=str(path), primary_sha256="bad", report=dict(passed=True))
        )
        assert not cap.collect(tmp_path, tasks)[0][i]["eligible"]


def test_duplicate_predictions(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8030-REDUCTION: duplicate seeds cannot silently change the denominator."""
    data = static_fixture(tmp_path)
    row = dict(arm="linear", condition="posthoc", seed=17, family_id="a")
    monkeypatch.setattr(red.static, "reduce", lambda *a: dict(rows=[row, row]))
    with pytest.raises(ValueError, match="duplicate_prediction"):
        red.independent(data, 8021)


def test_real_source_learning_cold_cli(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8030-SEAL: private real token and checkpoint reducers cold-replay outside the checkout."""
    from test_learning_retention_audit_8026 import bundle as learning_fixture

    monkeypatch.setattr(
        cap.authority.authority, "reader_rows", lambda raw, digest: [dict(passed=True)] * 13
    )
    root = tmp_path / "fixture"
    cap.prepare_fixture(root)
    tasks = cap.yaml.safe_load((root / "active.yaml").read_bytes())["tasks"]
    item = dict(
        family_id="a",
        source_bytes="61",
        answer_bytes="62",
        source_normalized_hash="source-a",
        role="evaluation",
        q=0.0,
        target_tokens=[1],
        response_token_offsets=[0],
        views={arm: dict(arm=arm) for arm in red.source.s.ARMS},
    )
    token_rows = [
        dict(
            id="a:" + arm,
            view_sha256=canonical_hash(item["views"][arm]),
            source_sha256=canonical_hash(item["source_bytes"]),
            answer_bytes="62",
            generated_tokens=0,
            status="completed",
            denominator=1,
            numerator=nll,
            mean_nll=nll,
            token_rows=[dict(token_id=1, offset=0, log_probability=-nll)],
        )
        for arm, nll in zip(red.source.s.ARMS, (1.0, 1.0, 2.0, 1.5), strict=True)
    ]
    a = dict(
        arm="conditional_linear_energy", mean=[0] * 4, scale=[1] * 4, columns=[], parameters=[2.0]
    )
    b = dict(a, arm="full_nll_qwen", parameters=[-2.0])
    panel = dict(rows=[item])
    features = [dict(r, q=0.0) for r in red.source.reduce(panel, token_rows)["source_feature_rows"]]
    source_bundle = dict(
        panel=panel,
        token_rows=token_rows,
        targets=[dict(family_id="a", eligible_y=1)],
        treatment_head=a,
        comparator_head=b,
        prediction_seal=canonical_hash(
            dict(
                features=features,
                treatment=red.source.predict(a, features).tolist(),
                comparator=red.source.predict(b, features).tolist(),
            )
        ),
    )
    learning_bundle = learning_fixture(tmp_path / "learning")
    for index, payload in [
        (6, dict(checkpoint_references=[reference(root / "raw/source.json", source_bundle)])),
        (8, dict(audit_bundle=reference(root / "raw/learning.json", learning_bundle))),
    ]:
        path = root / tasks[index]["deliverable"]
        data = dict(cap.read(path), rows=[{}], **payload)
        atomic_json(path, data)
        atomic_json(
            Path(data["terminal_validation_sidecar_path"]),
            dict(passed=True, primary_sha256=sha256_file(path)),
        )
    output = root / "results/experiment_8030_v695_capstone.json"
    assert cap.main(["--root", str(root), "--fixture-e2e"]) == 0
    result = cap.read(output)
    assert result["independent_reduction_rows"][6]["measurement_available"]
    assert result["independent_reduction_rows"][8]["measurement_available"]
    assert result["primary_hypothesis_results"][1]["qualified"] is False
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [
        str(cap.ROOT / ".venv/bin/python"),
        "-u",
        str(cap.ROOT / cap.CLI),
        "--cold-replay",
        str(output),
    ]
    with (tmp_path / "cold-cli.log").open("w") as stream:
        child = subprocess.run(
            command, cwd=tmp_path, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=60
        )
    assert child.returncode == 0, (tmp_path / "cold-cli.log").read_text()
