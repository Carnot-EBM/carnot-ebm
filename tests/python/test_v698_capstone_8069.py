"""REQ-REPORT-8069; SCENARIO-REPORT-8069-REDUCTION/TERMINAL."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting import v698_capstone as c
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def fixture(root: Path) -> None:
    """A private activated contract supplies custody without scientific credit."""
    tasks, authority = c.authorities(c.ROOT)
    for role, ref in authority["authority_snapshots"].items():
        if ref["exists"]:
            p = root / role
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(Path(ref["snapshot_path"]).read_bytes())
            ref["snapshot_path"] = str(p)
    atomic_json(root / c.INPUT, authority)
    (root / "ops").mkdir(exist_ok=True)
    (root / "ops/conductor-log.md").write_text("")
    for t in tasks[:-1]:
        p = root / t["deliverable"]
        value = dict(
            experiment_id=int(t["id"][3:7]),
            task_id=t["id"],
            honest_verdict="complete_null_fixture",
            verdict_class="null",
            flagged_adversarial=False,
            required_checks_passed=True,
            verifier_is_oracle=True,
            rows=[],
            raw_shard_hashes=[],
        )
        if t == tasks[0]:
            value.update(
                authority_snapshots=authority["authority_snapshots"],
                canonical_tasks_sha256=authority["canonical_tasks_sha256"],
            )
        receipt = c.publish_primary(p, value, lambda _: dict(passed=True))
        side = p.parent / "raw" / p.stem / "terminal.json"
        value["terminal_validation_sidecar_path"] = str(side)
        receipt = c.publish_primary(p, value, lambda _: dict(passed=True))
        atomic_json(side, dict(publication=receipt))


def test_authority_full_prompts_and_digest(tmp_path):
    """REQ-REPORT-8069: a title match cannot hide prompt mutation."""
    fixture(tmp_path)
    tasks, authority = c.authorities(tmp_path)
    assert len(tasks) == 13
    p = Path(authority["authority_snapshots"]["active"]["snapshot_path"])
    p.write_text(p.read_text() + "\n# changed bytes\n")
    with pytest.raises(ValueError):
        c.authorities(tmp_path)


def test_missing_and_forged_skip(tmp_path):
    """REQ-REPORT-8069: absence and actual conductor receipts remain distinct."""
    fixture(tmp_path)
    tasks, _ = c.authorities(tmp_path)
    target = tmp_path / tasks[3]["deliverable"]
    target.unlink()
    skip = c.skip_path(tmp_path, tasks[3])
    atomic_json(skip, dict(experiment=9999, schema="blocked_gate_check_v1"))
    with pytest.raises(ValueError, match="skip_identity"):
        c.collect(tmp_path, tasks)
    atomic_json(
        skip,
        dict(
            experiment=8060,
            schema="blocked_gate_check_v1",
            honest_verdict="blocked_gate_check_failed",
            failed_upstream=tasks[2]["id"],
            failed_field="fit_capture_ready_score",
            failed_expected=1,
            failed_observed=0,
            failed_evidence_path=str(tmp_path / tasks[2]["deliverable"]),
            failed_evidence_sha256=sha256_file(tmp_path / tasks[2]["deliverable"]),
        ),
    )
    rows, refs, failures = c.collect(tmp_path, tasks)
    assert rows[3]["verdict_class"] == "blocked"
    assert rows[3]["primary_present"] is False
    assert rows[3]["producer_status"] == "conductor_skip_receipt"
    assert failures and refs
    target = tmp_path / tasks[4]["deliverable"]
    target.unlink()
    rows, _, _ = c.collect(tmp_path, tasks)
    assert rows[4]["producer_honest_verdict"] is None


def test_holm_safety_and_fixture():
    """REQ-REPORT-8069: unsafe raw tests remain in the family with p=1."""
    h = dict(
        hypothesis="H3",
        tests=[dict(raw_p=0.001, margin=0.02)],
        support_passed=True,
        safety_passed=False,
    )
    result = c.holm(h, valid=True)
    assert [r["family_p"] for r in result] == [1, 1, 1]
    h["safety_passed"] = True
    assert c.holm(h, valid=True)[2]["holm_adjusted_p"] == 0.003
    assert c.holm(h, valid=False)[2]["family_p"] == 1


def test_dispositions_do_not_promote_fixture(tmp_path):
    """REQ-REPORT-8069: a finished oracle fixture cannot close a PRD gap."""
    fixture(tmp_path)
    value = c.build(tmp_path, "20261003", tmp_path / "durable")
    assert len(value["task_dispositions"]) == 13
    assert len(value["gap_decisions"]) == 3
    assert value["science_ready"] is False
    assert value["generalized_learning_benefit_score"] == 0
    assert c.cold_replay(value) == []
    value["science_ready"] = True
    assert c.cold_replay(value)


def test_real_cli_success_blocked_and_mutation(tmp_path):
    """SCENARIO-REPORT-8069-TERMINAL: actual CLI exits outside the checkout."""
    fixture(tmp_path / "input")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    cli = str(c.ROOT / c.CLI)
    output = tmp_path / "results/experiment_8069_v698_capstone.json"
    command = [str(c.ROOT / ".venv/bin/python"), "-u", cli]
    p = subprocess.run(
        command + ["--fixture-e2e", "--root", str(tmp_path / "input"), "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    value = json.loads(output.read_text())
    assert value["verifier_is_oracle"] and not value["science_ready"]
    for changed, expected in [(False, 0), (True, 1)]:
        if changed:
            value["primary_hypothesis_results"][0]["family_p"] = 0.001
            atomic_json(output, value)
        p = subprocess.run(
            command + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert p.returncode == expected, p.stdout + p.stderr
    p = subprocess.run(
        command
        + [
            "--fixture-e2e",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(tmp_path / "blocked/experiment_8069_v698_capstone.json"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    blocked = json.loads((tmp_path / "blocked/experiment_8069_v698_capstone.json").read_text())
    assert blocked["verdict_class"] == "blocked" and blocked["gate_check_summary"]


def test_current_primitive_science_and_mutation():
    """REQ-REPORT-8069: reconstruct actual source, learning and deployment operands."""
    from carnot.reporting import v698_capstone_reduction as r

    for n in (8059, 8063, 8065, 8066, 8068):
        path = next((c.ROOT / "results").glob(f"experiment_{n}_*.json"))
        data = json.loads(path.read_text())
        observed = r.independent(data, n)
        if n == 8059:
            assert not observed["pilot_passed"]
            assert len(observed["feature_rows"]) == 7
        if n == 8063:
            assert observed["missed_numerator"] == 30
            assert observed["missed_denominator"] == 37
        if n == 8065:
            h = observed["primary_hypothesis_results"][0]
            assert h["support_count"] == 88 and h["retention_support_count"] == 63
            assert not h["safety_passed"] and h["beneficial_changed_sources"] == 0
            assert h["tests"][0]["gain"] == pytest.approx(-0.011363636363636364)
            data["per_seed_false_accept_rows"][0]["numerator"] += 1
            with pytest.raises(ValueError):
                r.independent(data, n)
        if n == 8066:
            assert observed["censored_count"] == 784
        if n == 8068:
            assert observed["hardware_custody_ready_score"] == 1
            assert observed["acceleration_bounds"] == []


def test_custody_rejections(tmp_path):
    """REQ-REPORT-8069: reject a replaced primary and missing sidecar independently."""
    fixture(tmp_path)
    tasks, _ = c.authorities(tmp_path)
    target = tmp_path / tasks[2]["deliverable"]
    value = json.loads(target.read_text())
    value["task_id"] = "exp9999-forged"
    atomic_json(target, value)
    with pytest.raises(ValueError, match="primary_identity"):
        c.collect(tmp_path, tasks)
    value.update(
        task_id=tasks[2]["id"],
        required_checks_passed=False,
        flagged_adversarial=True,
        raw_shard_hashes={"/missing-primitive": "sha256:wrong"},
        gate_check_summary=[dict(field="support", expected=True, observed=False)],
    )
    atomic_json(target, value)
    rows, _, errors = c.collect(tmp_path, tasks)
    assert not rows[2]["eligible"]
    assert any(e["field"] == "primitive_sha256" for e in errors)
    value["required_checks_passed"] = True
    value["flagged_adversarial"] = False
    assert any(e["field"] == "terminal_primary_binding" for e in errors)


def test_complete_replay_and_owned_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8069: changed observations and failed coverage cannot qualify."""
    fixture(tmp_path)
    value = c.build(tmp_path, "20261003", tmp_path / "durable")
    counts = {p: dict(num_statements=1, covered_lines=1) for p in c.OWNED}
    c.complete(value, [dict(passed=True)], counts)
    assert value["capstone_execution_ready_score"] == 1
    assert value["completed_count"] == 13
    c.complete(value, [dict(passed=False)], counts)
    assert value["verdict_class"] == "disqualified"
    old = c.reduction.independent
    monkeypatch.setattr(c.reduction, "independent", lambda *a: dict(measurement_available=True))
    assert c.cold_replay(value) == ["independent_reduction_drift"]
    monkeypatch.setattr(
        c.reduction, "independent", lambda *a: (_ for _ in ()).throw(ValueError("primitive"))
    )
    changed = c.build(tmp_path, "20261003", tmp_path / "bad-reduction")
    assert any(e["field"] == "independent_reduction" for e in changed["gate_check_summary"])
    assert c.cold_replay(changed) == []
    monkeypatch.setattr(c.reduction, "independent", old)
    Path(value["source_artifact_hashes"][0]["path"]).write_text("tampered")
    assert c.cold_replay(value) == ["source_bytes_changed"]


def test_cli_dispatch_and_full_parent(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8069-TERMINAL: parent waits for actual reduction child exit."""
    import runpy
    import sys

    fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", [c.CLI, "--cold-replay", str(tmp_path / "missing")])
    with pytest.raises(SystemExit) as exit_status:
        runpy.run_path(str(c.ROOT / c.CLI), run_name="__main__")
    assert exit_status.value.code == 1
    with pytest.raises(ValueError, match="date"):
        c.main(["--date", "20260101"])
    assert (
        c.main(
            [
                "--worker",
                "--root",
                str(tmp_path),
                "--durable",
                str(tmp_path / "worker"),
                "--output",
                str(tmp_path / "worker.json"),
            ]
        )
        == 0
    )
    assert c.main(["--cold-replay", str(tmp_path / "worker.json")]) == 0
    frozen = c.manifest(tmp_path / "scratch")
    assert all(s["name"] != "full_suite" for s in frozen["commands"])

    def small_manifest(private):
        atomic_json(
            private / "coverage.json",
            dict(files={p: dict(summary=dict(num_statements=1, covered_lines=1)) for p in c.OWNED}),
        )
        return dict(commands=[frozen["commands"][0]])

    monkeypatch.setattr(c, "manifest", small_manifest)
    output = tmp_path / "published/experiment_8069_v698_capstone.json"
    assert c.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["capstone_execution_ready_score"] == 1
    monkeypatch.setattr(c, "run_check", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="reduction_child_failed"):
        c.main(["--root", str(tmp_path), "--output", str(output)])


def test_terminal_rejects_flags_and_reader_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8069: a flagged artifact or wrong consumer cannot be published."""
    fixture(tmp_path)
    value = c.build(tmp_path, "20261003", tmp_path / "durable")
    log = tmp_path / "log.json"
    atomic_json(log, dict(flagged_count=1))
    monkeypatch.setattr(c, "run_check", lambda *a: dict(passed=True, log_path=str(log)))
    output = tmp_path / "pub/experiment_8069_v698_capstone.json"
    with pytest.raises(ValueError, match="terminal_validation_failed"):
        c.publish(value, output, tmp_path / "scratch", tmp_path / "durable", fixture=False)
    atomic_json(log, dict(flagged_count=0))
    monkeypatch.setattr(c, "reader_receipt", lambda *a, **k: dict(passed=False))
    with pytest.raises(ValueError, match="published_reader_drift"):
        c.publish(value, output, tmp_path / "scratch", tmp_path / "durable", fixture=False)


def test_remaining_negative_operands(tmp_path):
    """REQ-REPORT-8069: exact negative branches cover authority, skips and seals."""
    fixture(tmp_path)
    tasks, _ = c.authorities(tmp_path)
    authority_path = tmp_path / c.INPUT
    original = json.loads(authority_path.read_text())
    changed = dict(original, canonical_tasks_sha256="wrong")
    atomic_json(authority_path, changed)
    with pytest.raises(ValueError, match="immutable_authority"):
        c.authorities(tmp_path)
    atomic_json(authority_path, original)
    target = tmp_path / tasks[2]["deliverable"]
    side = json.loads(
        Path(json.loads(target.read_text())["terminal_validation_sidecar_path"]).read_text()
    )
    bound = Path(side["publication"]["sidecar_path"])
    report = json.loads(bound.read_text())
    report["report"]["passed"] = False
    atomic_json(bound, report)
    assert any("terminal_report_failed" in str(g) for g in c.collect(tmp_path, tasks)[2])
    skip = c.skip_path(tmp_path, tasks[3])
    (tmp_path / tasks[3]["deliverable"]).unlink()
    atomic_json(
        skip,
        dict(
            experiment=8060,
            schema="blocked_gate_check_v1",
            failed_evidence_path=str(target),
            failed_evidence_sha256="wrong",
        ),
    )
    with pytest.raises(ValueError, match="skip_upstream_hash"):
        c.collect(tmp_path, tasks)
    blocked = c.build(tmp_path / "absent", "20261003", tmp_path / "blocked-raw")
    assert blocked["gate_check_summary"][0]["field"] == "immutable_authority"
    value = c.build(tmp_path / "absent", "20261003", tmp_path / "private-raw")
    c.previous.seal(value, tmp_path / "seal")
    value["capstone_execution_ready_score"] = 1
    assert c.cold_replay(value) == ["claim_seal_drift"]


def test_inprocess_fixture_parent(tmp_path):
    """REQ-REPORT-8069: fixture publication explicitly withholds owned readiness."""
    fixture(tmp_path)
    output = tmp_path / "private-output/experiment_8069_v698_capstone.json"
    assert c.main(["--fixture-e2e", "--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verifier_is_oracle"] is True


def test_durable_token_copy_and_shard_mutation(tmp_path):
    """REQ-REPORT-8069: cold equations read copies and authenticate every shard byte."""
    from carnot.reporting import v698_capstone_reduction as r
    import shutil

    fixture(tmp_path)
    value = c.build(tmp_path, "20261003", tmp_path / "durable")
    part = tmp_path / "part.bin"
    part.write_bytes(b"primitive")
    manifest = tmp_path / "example.shards.json"
    atomic_json(
        manifest,
        dict(
            original_sha256=sha256_file(part),
            shards=[dict(path=str(part), sha256=sha256_file(part))],
        ),
    )
    value["raw_shard_hashes"].append(dict(path=str(manifest), sha256=sha256_file(manifest)))
    assert c.cold_replay(value) == []
    original_manifest = json.loads(manifest.read_text())
    atomic_json(manifest, dict(original_manifest, original_sha256="wrong"))
    value["raw_shard_hashes"][-1]["sha256"] = sha256_file(manifest)
    assert c.cold_replay(value) == ["source_bytes_changed"]
    atomic_json(manifest, original_manifest)
    value["raw_shard_hashes"][-1]["sha256"] = sha256_file(manifest)
    part.write_bytes(b"forged")
    assert c.cold_replay(value) == ["source_bytes_changed"]
    data = json.loads((c.ROOT / "results/experiment_8059_v698_fit_source_scoring.json").read_text())
    original = Path(data["terminal_validation_sidecar_path"]).parent
    target = tmp_path / "source-copy"
    target.mkdir()
    shutil.copyfile(original / "runtime.json", target / "runtime.json")
    shutil.copytree(original / "forwards", target / "forwards")
    mapping = {str(p): str(target / p.relative_to(original)) for p in original.rglob("*.json")}
    assert r.independent(data, 8059, mapping)["scored_tokens"] == data["scored_tokens"]
    (target / "forwards/pass-000.json").unlink()
    with pytest.raises(ValueError):
        r.independent(data, 8059, mapping)
