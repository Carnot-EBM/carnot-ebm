"""REQ-REPORT-8032: sealed custody does not create unseen development labels."""

import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from carnot import experiment_8032_v696_sealed_methods as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def public_row(role="fit", slot=0):
    """Private sources give reproducible admission without using repository labels."""
    return dict(
        family_id=f"{role}-{slot}",
        role=role,
        slot=slot,
        source_bytes=f"Source {role} {slot}.".encode().hex(),
        answer_bytes=b"Answer.".hex(),
        features=[0.0] * 8,
        q=0.5,
        public_eligible=True,
        exclusion_reason=None,
    )


def test_frozen_methods_and_selection():
    """SCENARIO-REPORT-8032-METHODS: equal budgets do not imply independent seeds."""
    assert e.METHODS["learning"]["seeds"] == list(range(101, 121))
    assert e.METHODS["learning"]["maximum_updates"] == 64
    rows = [dict(family_id=str(i), slot=i) for i in range(100)]
    for arm in e.LEARNING_ARMS[:-1]:
        picked = e.select(rows, arm, 101, 7)
        assert len(picked) == len({r["family_id"] for r in picked}) == 4
        if arm == "recent64":
            assert min(r["slot"] for r in picked) >= 36
        if arm == "newest16":
            assert min(r["slot"] for r in picked) >= 84
    assert e.select(rows, "frozen_no_write", 101, 7) == []
    assert e.select(rows, "recent64", 101, 7) == e.select(rows, "recent64", 101, 7)
    with pytest.raises(ValueError, match="arm"):
        e.select(rows, "oracle", 101, 7)
    with pytest.raises(ValueError, match="support"):
        e.select(rows[:3], "recent64", 101, 7)
    for p, expected in ((0.09, "accept"), (0.1, "escalate"), (0.5, "escalate"), (0.51, "reject")):
        assert e.action(p) == expected


def test_worker_sentinel_and_public_projection():
    """SCENARIO-REPORT-8032-ACCESS: no target-bearing object reaches a scorer."""
    row = public_row()
    assert e.public_worker([row])["completed"] == 1
    for key in ("y", "label_path", "eligible_y", "annotation_rows"):
        with pytest.raises(ValueError, match="public_worker_fields"):
            e.public_worker([dict(row, **{key: "PRIVATE_SENTINEL"})])
    with pytest.raises(ValueError, match="incomplete"):
        e.public_worker([dict(row, source_bytes="")])


def test_vault_protocol_and_delayed_release(tmp_path):
    """SCENARIO-REPORT-8032-ACCESS: real private files cannot bypass chronology."""
    vault = tmp_path / "vault.json"
    atomic_json(vault, dict(role="stream", rows=[dict(family_id="stream-0", slot=0, eligible_y=1)]))
    protocol = tmp_path / "methods.json"
    with pytest.raises(ValueError, match="protocol_before_vault"):
        e.evaluator(protocol, vault, "support")
    e.immutable(protocol, dict(methods=e.METHODS, frozen=True))
    assert e.evaluator(protocol, vault, "support")["class_counts"] == {"0": 0, "1": 1}
    issue = tmp_path / "issue.json"
    atomic_json(issue, dict(family_id="stream-0", slot=0, durable=True))
    for now in (0, 19, 21):
        with pytest.raises(ValueError, match="release_slot"):
            e.evaluator(protocol, vault, "release", issue=issue, now=now)
    assert e.evaluator(protocol, vault, "release", issue=issue, now=20)["y"] == 1
    atomic_json(issue, dict(family_id="stream-0", slot=0, durable=False))
    with pytest.raises(ValueError, match="durable_issue"):
        e.evaluator(protocol, vault, "release", issue=issue, now=20)
    atomic_json(vault, dict(role="retention", rows=[]))
    with pytest.raises(ValueError, match="retention"):
        e.evaluator(protocol, vault, "release", issue=issue, now=20)
    atomic_json(protocol, dict(methods={}, frozen=True))
    with pytest.raises(ValueError, match="protocol_before_vault"):
        e.evaluator(protocol, vault, "support")


def test_unknown_target_and_immutable_bytes(tmp_path):
    """SCENARIO-REPORT-8032-CUSTODY: unknown is never supported; frozen bytes resist edits."""
    protocol, vault = tmp_path / "methods.json", tmp_path / "vault.json"
    e.immutable(protocol, dict(methods=e.METHODS, frozen=True))
    e.immutable(protocol, dict(methods=e.METHODS, frozen=True))
    with pytest.raises(ValueError, match="immutable"):
        e.immutable(protocol, dict(methods={}, frozen=True))
    atomic_json(vault, dict(role="fit", rows=[dict(eligible_y=None), dict(eligible_y=0)]))
    assert e.evaluator(protocol, vault, "support")["unknown_target"] == 1
    for row in ({}, dict(eligible_y=2), dict(eligible_y=True)):
        atomic_json(vault, dict(role="fit", rows=[row]))
        with pytest.raises(ValueError, match="target_contract"):
            e.evaluator(protocol, vault, "support")


def cli(tmp_path, *args):
    """Instrument the real entry point while the inherited artifact guard stays active."""
    prefix = [str(e.ROOT / ".venv/bin/python")]
    config = os.environ.get("CARNOT_8032_COVERAGE_CONFIG")
    if config:
        prefix += [
            "-m",
            "coverage",
            "run",
            "--data-file=" + str(Path(config).parent / ".coverage"),
            "--rcfile=" + config,
        ]
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        prefix + [str(e.ROOT / e.SCRIPT), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def fixture_root(tmp_path):
    """Build independent primary/sidecar bytes without touching production results."""
    root = tmp_path / "repo"
    results = root / "results"
    results.mkdir(parents=True)
    public, evaluator = {}, {}
    for role, count in e.SLOTS.items():
        pub, ev = results / f"public-{role}.json", results / f"vault-{role}.json"
        rows = [public_row(role, i) for i in range(count)]
        atomic_json(pub, dict(rows=rows, role=role))
        atomic_json(
            ev,
            dict(
                role=role,
                rows=[
                    dict(family_id=r["family_id"], slot=r["slot"], eligible_y=i % 2)
                    for i, r in enumerate(rows)
                ],
            ),
        )
        public[role] = dict(path=str(pub), sha256=sha256_file(pub))
        evaluator[role] = dict(path=str(ev), sha256=sha256_file(ev))
    head = results / "head.json"
    atomic_json(
        head,
        dict(
            arm="conditioned_energy",
            seed=17,
            converged=True,
            parameters=[0.0] * 110,
            geometry=dict(
                scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9), logit_center=0.0, logit_scale=1.0
            ),
        ),
    )
    calibration = results / "calibration.json"
    atomic_json(
        calibration,
        dict(maps={"conditioned_energy-17": dict(converged=True, parameters=[0.0, 1.0])}),
    )
    panel = []
    for role, n in e.METHODS["likelihood"]["roles"].items():
        for i in range(n):
            r = public_row(role, i)
            blob = r["source_bytes"]
            view = dict(source_bytes=blob, input_tokens=100, tokens=[0, 1], response_start=1)
            panel.append(
                dict(
                    r,
                    target_tokens=[1],
                    views=dict(
                        full=view,
                        duplicate=copy.deepcopy(view),
                        no_source=dict(view, source_bytes=""),
                    ),
                )
            )
    for n, name in e.UPSTREAM.items():
        p = results / (name + ".json")
        value = dict(
            experiment_id=n,
            task_id=f"exp{n}-fixture",
            run_date="20261002",
            verdict_class="null",
            honest_verdict="complete_null_fixture",
            flagged_adversarial=False,
            raw_shard_hashes=[],
            code_config_hashes=[],
        )
        if n == 8019:
            value.update(
                public_manifests=public,
                evaluator_manifests=evaluator,
                support_by_role={},
                exposure_rows=[],
                historical_failure_logs=[],
            )
        if n == 8020:
            value.update(
                energy_fit_ready_score=1,
                calibration_checkpoint=dict(path=str(calibration), sha256=sha256_file(calibration)),
                head_checkpoints=[dict(path=str(head), sha256=sha256_file(head))],
            )
        if n == 8022:
            value.update(
                public_panel_manifest=dict(
                    rows=panel,
                    counts=e.METHODS["likelihood"]["roles"],
                    complete=True,
                    exclusions=[],
                )
            )
        if n == 8025:
            value.update(
                learning_measurement_ready_score=1, cited_upstream_artifacts=[public["stream"]]
            )
        if n == 8026:
            value.update(
                verdict_class="disqualified",
                honest_verdict="complete_disqualified_fixture",
                prefreeze_retention_exposure=True,
                gate_check_summary=[
                    dict(
                        artifact_field="early_retention",
                        expected=False,
                        observed=True,
                        passed=False,
                    )
                ],
            )
        terminal = results / "raw" / name / "terminal_validation.json"
        value["terminal_validation_sidecar_path"] = str(terminal)
        atomic_json(p, value)
        sidecar = terminal.parent / "validators" / "bound.json"
        atomic_json(
            sidecar,
            dict(primary_path=str(p), primary_sha256=sha256_file(p), report=dict(passed=True)),
        )
        atomic_json(
            terminal,
            dict(primary_path=str(p), primary_sha256=sha256_file(p), sidecar_path=str(sidecar)),
        )
    return root


def test_seal_and_cold_reduction(tmp_path):
    """SCENARIO-REPORT-8032-CUSTODY: replay binds every frozen row and vault event."""
    root = fixture_root(tmp_path)
    raw = tmp_path / "raw"
    value = e.seal(root, raw)
    assert value["likelihood_panel_ready_score"] == value["learning_inputs_ready_score"] == 1
    assert all(
        x["protocol_sha256"] == value["methods_reference"]["sha256"]
        for x in value["evaluator_access_log"]
    )
    assert len(value["overlap_cutpoints"]) == 3
    assert value["historical_exposure"]["prior_verdicts"][-1]["verdict_class"] == "disqualified"
    assert not value["gate_check_summary"]
    artifact = tmp_path / (e.NAME + ".json")
    atomic_json(artifact, value)
    assert e.replay(artifact)["passed"]
    value["overlap_cutpoints"][0] += 1
    atomic_json(artifact, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(artifact)


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "stale",
        "head",
        "panel",
        "duplicate",
        "tokens",
        "roles",
        "source_overlap",
        "unknown",
        "producer_bytes",
        "stream_binding",
    ],
)
def test_external_block_operands(tmp_path, failure):
    """SCENARIO-REPORT-8032-CUSTODY: failures name a field rather than measuring zero."""
    root = fixture_root(tmp_path)
    p = (
        root
        / "results"
        / (
            e.UPSTREAM[
                8020
                if failure in {"missing", "stale", "head", "producer_bytes"}
                else 8025
                if failure == "stream_binding"
                else 8019
                if failure == "unknown"
                else 8022
            ]
            + ".json"
        )
    )
    d = json.loads(p.read_text())
    if failure == "missing":
        p.unlink()
    elif failure == "stale":
        atomic_json(p, dict(d, energy_fit_ready_score=0))
    else:
        if failure == "head":
            Path(d["head_checkpoints"][0]["path"]).write_text("{}")
        elif failure == "producer_bytes":
            d["code_config_hashes"] = [dict(path=str(root / "absent.py"), sha256="sha256:absent")]
        elif failure == "stream_binding":
            d["cited_upstream_artifacts"] = []
        elif failure == "unknown":
            vault = Path(d["evaluator_manifests"]["retention"]["path"])
            data = json.loads(vault.read_text())
            data["rows"] = [dict(family_id="retention-0", slot=0, eligible_y=None)] * 64
            atomic_json(vault, data)
            d["evaluator_manifests"]["retention"]["sha256"] = sha256_file(vault)
        else:
            panel = d["public_panel_manifest"]
            if failure == "panel":
                panel["rows"][0]["source_bytes"] = ""
            if failure == "duplicate":
                panel["rows"][0]["views"]["duplicate"]["source_bytes"] = "00"
            if failure == "tokens":
                panel["rows"][0]["views"]["full"]["input_tokens"] = 6001
            if failure == "roles":
                panel["rows"].pop()
            if failure == "source_overlap":
                panel["rows"][1]["source_bytes"] = panel["rows"][0]["source_bytes"]
        atomic_json(p, d)
        terminal = json.loads(Path(d["terminal_validation_sidecar_path"]).read_text())
        terminal["primary_sha256"] = sha256_file(p)
        sidecar = Path(terminal["sidecar_path"])
        atomic_json(
            sidecar,
            dict(primary_path=str(p), primary_sha256=sha256_file(p), report=dict(passed=True)),
        )
        atomic_json(Path(d["terminal_validation_sidecar_path"]), terminal)
    value = e.seal(root, tmp_path / "raw")
    assert value["gate_check_summary"]
    for gate in value["gate_check_summary"]:
        assert {
            "upstream_id",
            "path",
            "sha256",
            "artifact_field",
            "expected",
            "observed",
            "passed",
            "check_name",
        } <= set(gate)
        assert not gate["passed"]


def test_private_cli_and_consumer(tmp_path):
    """SCENARIO-REPORT-8032-CLI: actual script publishes, replays, and rejects tampering."""
    root = fixture_root(tmp_path)
    result = cli(tmp_path, "--fixture-root", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    out = Path(json.loads(result.stdout.splitlines()[-1])["primary_path"])
    assert out.is_relative_to(Path(os.environ["CARNOT_EXPERIMENT_ARTIFACT_ROOT"]))
    data = json.loads(out.read_text())
    assert data["verdict_class"] == "disqualified" and data["protocol_ready_score"] == 0
    assert cli(tmp_path, "--cold-replay", str(out)).returncode == 0
    data["rows"][0]["eligible"] = False
    atomic_json(out, data)
    assert cli(tmp_path, "--cold-replay", str(out)).returncode == 1
    assert cli(tmp_path, "--date", "20261003", "--fixture-root", str(root)).returncode == 1
    for worker in ("scoring", "fitter", "learner"):
        public = tmp_path / (worker + ".json")
        atomic_json(public, dict(rows=[public_row()]))
        assert (
            cli(tmp_path, "--public-worker", worker, "--public-input", str(public)).returncode == 0
        )
        atomic_json(public, dict(rows=[dict(public_row(), y="PRIVATE_SENTINEL")]))
        assert (
            cli(tmp_path, "--public-worker", worker, "--public-input", str(public)).returncode == 1
        )
    protocol, vault = tmp_path / "methods.json", tmp_path / "vault.json"
    e.immutable(protocol, dict(methods=e.METHODS, frozen=True))
    atomic_json(vault, dict(role="retention", rows=[dict(eligible_y=None)]))
    assert (
        cli(
            tmp_path,
            "--evaluator-worker",
            "support",
            "--protocol",
            str(protocol),
            "--vault",
            str(vault),
        ).returncode
        == 0
    )
    assert (
        cli(
            tmp_path,
            "--evaluator-worker",
            "support",
            "--protocol",
            str(tmp_path / "absent"),
            "--vault",
            str(vault),
        ).returncode
        == 1
    )


def test_validation_and_terminal_commands(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8032-CLI: explicit commands preserve coverage and exit receipts."""
    plan = e.validation_plan(tmp_path)
    assert e.TEST in plan[0].argv
    assert plan[-1].scope == "repository_health" and plan[-1].timeout_s == 180
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, dict(example=True))
    calls = []

    def run(root, commands, **kwargs):
        calls.extend(commands)
        return [dict(scope="terminal", passed=True)]

    monkeypatch.setattr(e, "run_commands", run)
    assert e.terminal(path)["passed"]
    assert len(calls) == 3
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=False)])
    assert not e.terminal(path)["passed"]


def test_finish_and_production_orchestration(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8032-CLI: owned failures disqualify while external failures block."""
    value = e.seal(fixture_root(tmp_path), tmp_path / "raw")
    counts = {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED}
    receipts = [dict(scope="owned", passed=True)]
    done = e.finish(copy.deepcopy(value), tmp_path, receipts, counts, 0.01)
    assert done["protocol_ready_score"] == 1 and done["verdict_class"] == "null"
    broken = copy.deepcopy(value)
    broken["gate_check_summary"] = [dict(passed=False)]
    assert e.finish(broken, tmp_path, receipts, counts, 0.01)["verdict_class"] == "blocked"
    assert e.finish(copy.deepcopy(value), tmp_path, [], {}, 0.01)["verdict_class"] == "disqualified"

    def plan(scratch):
        atomic_json(
            scratch / "coverage.json", dict(files={p: dict(summary=s) for p, s in counts.items()})
        )
        return []

    monkeypatch.setattr(e, "validation_plan", plan)
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: receipts)
    monkeypatch.setattr(e, "seal", lambda *a, **k: copy.deepcopy(value))
    monkeypatch.setattr(e, "terminal", lambda p: dict(passed=True))
    assert e.main([]) == 0
    monkeypatch.setattr(e, "terminal", lambda p: dict(passed=False))
    assert e.main([]) == 1


def test_replay_access_and_readiness_tamper(tmp_path):
    """SCENARIO-REPORT-8032-CUSTODY: a changed access operand cannot remain ready."""
    value = e.seal(fixture_root(tmp_path), tmp_path / "raw")
    artifact = tmp_path / (e.NAME + ".json")
    value.update(protocol_ready_score=1, verdict_class="blocked")
    atomic_json(artifact, value)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(artifact)
    value.update(protocol_ready_score=0)
    value["evaluator_access_log"][0]["protocol_frozen_before_access"] = False
    atomic_json(artifact, value)
    with pytest.raises(e.Contract, match="protocol_frozen_before_access"):
        e.replay(artifact)


def test_full_cold_reduction_rejects_masks_and_protocol(tmp_path):
    """SCENARIO-REPORT-8032-CUSTODY: a frozen mask must match actual fit activations."""
    value = e.seal(fixture_root(tmp_path), tmp_path / "raw")
    methods = Path(value["methods_reference"]["path"])
    protocol = json.loads(methods.read_text())
    protocol["overlap"]["masks"]["fit"][0]["columns"] = []
    atomic_json(methods, protocol)
    for ref in value["raw_shard_hashes"]:
        if ref["path"] == str(methods):
            ref["sha256"] = sha256_file(methods)
    value["methods_reference"]["sha256"] = sha256_file(methods)
    artifact = tmp_path / (e.NAME + ".json")
    atomic_json(artifact, value)
    with pytest.raises(ValueError, match="overlap_reduction"):
        e.replay(artifact)


def test_sidecar_missing_contract_and_history(tmp_path):
    """SCENARIO-REPORT-8032-CUSTODY: earlier contradictory verdicts stay durable."""
    root = fixture_root(tmp_path)
    p = root / "results" / (e.UPSTREAM[8019] + ".json")
    d = json.loads(p.read_text())
    historic = root / "results" / "raw" / e.UPSTREAM[8019] / "failure_logs" / "first-attempt.json"
    atomic_json(
        historic,
        dict(d, verdict_class="disqualified", honest_verdict="complete_disqualified_fixture"),
    )
    value = e.seal(root, tmp_path / "raw")
    assert any(r.get("previous_attempt") for r in value["historical_exposure"]["prior_verdicts"])
    p = root / "results" / (e.UPSTREAM[8020] + ".json")
    d = json.loads(p.read_text())
    del d["terminal_validation_sidecar_path"]
    atomic_json(p, d)
    value = e.seal(root, tmp_path / "blocked")
    assert any(
        r["artifact_field"] == "terminal_validation_sidecar_path"
        for r in value["gate_check_summary"]
    )


def test_calibration_binding_and_original_role_order(tmp_path):
    """SCENARIO-REPORT-8032-METHODS: qualified calibration is part of the imported head."""
    root = fixture_root(tmp_path)
    value = e.seal(root, tmp_path / "raw")
    protocol = json.loads(Path(value["methods_reference"]["path"]).read_text())
    assert protocol["original_role_order"] == list(e.SLOTS)
    assert protocol["head"]["calibration"] == [0.0, 1.0]
    p = root / "results" / (e.UPSTREAM[8020] + ".json")
    d = json.loads(p.read_text())
    calibration = Path(d["calibration_checkpoint"]["path"])
    calibration.write_text("{}")
    value = e.seal(root, tmp_path / "blocked")
    assert any(r["artifact_field"] == "sha256" for r in value["gate_check_summary"])


def test_authentication_capture_and_duplicate_copy(tmp_path):
    """SCENARIO-REPORT-8032-CUSTODY: capture and producer bytes are authenticated independently."""
    root = fixture_root(tmp_path)
    p = root / "results" / (e.UPSTREAM[8022] + ".json")
    d = json.loads(p.read_text())
    capture = root / "results" / "capture.json"
    atomic_json(capture, dict(private_scoring=True))
    ref = dict(path=str(capture), sha256=sha256_file(capture))
    d["raw_shard_hashes"] = [ref]
    atomic_json(p, d)
    terminal_path = Path(d["terminal_validation_sidecar_path"])
    terminal = json.loads(terminal_path.read_text())
    terminal["primary_sha256"] = sha256_file(p)
    atomic_json(
        Path(terminal["sidecar_path"]),
        dict(primary_path=str(p), primary_sha256=sha256_file(p), report=dict(passed=True)),
    )
    atomic_json(terminal_path, terminal)
    value = e.seal(root, tmp_path / "raw")
    assert any(r.get("original_path") == str(capture) for r in value["cited_upstream_artifacts"])
    copied = e.copy_bound(ref, tmp_path / "copy")
    assert e.copy_bound(ref, tmp_path / "copy") == copied
    Path(copied["path"]).chmod(0o600)
    Path(copied["path"]).write_text("{}")
    with pytest.raises(e.Contract, match="sha256"):
        e.copy_bound(ref, tmp_path / "copy")


def test_tokens_and_unopened_public_role(tmp_path):
    """SCENARIO-REPORT-8032-CUSTODY: admission errors preserve a terminal operand."""
    root = fixture_root(tmp_path)
    p = root / "results" / (e.UPSTREAM[8022] + ".json")
    panel = json.loads(p.read_text())["public_panel_manifest"]
    for arm in ("full", "duplicate"):
        panel["rows"][0]["views"][arm]["input_tokens"] = 6001
    with pytest.raises(ValueError, match="panel_token_admission"):
        e.panel_rows(panel)
    p = root / "results" / (e.UPSTREAM[8019] + ".json")
    d = json.loads(p.read_text())
    Path(d["public_manifests"]["fit"]["path"]).write_text("{}")
    value = e.seal(root, tmp_path / "raw")
    assert value["gate_check_summary"] and not value["evaluator_access_log"]


def test_published_validation_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8032-CLI: a failed published-byte assertion must fail process exit."""
    value = e.seal(fixture_root(tmp_path), tmp_path / "raw")
    monkeypatch.setattr(e, "validation_plan", lambda p: [])
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [])
    monkeypatch.setattr(e, "seal", lambda *a, **k: value)
    reports = iter([dict(passed=True), dict(passed=False)])
    monkeypatch.setattr(e, "terminal", lambda p: next(reports))
    assert e.main([]) == 1


def test_learning_is_independent_of_likelihood_capture(tmp_path):
    """SCENARIO-REPORT-8032-METHODS: a likelihood block cannot invalidate learning inputs."""
    root = fixture_root(tmp_path)
    (root / "results" / (e.UPSTREAM[8022] + ".json")).unlink()
    value = e.seal(root, tmp_path / "raw")
    assert value["likelihood_panel_ready_score"] == 0
    assert value["learning_inputs_ready_score"] == 1
    assert value["gate_check_summary"][0]["branch"] == "likelihood"
    root = fixture_root(tmp_path / "other")
    p = root / "results" / (e.UPSTREAM[8022] + ".json")
    data = json.loads(p.read_text())["public_panel_manifest"]
    data["rows"][0]["views"]["no_source"]["tokens"] = [2, 3]
    monkey = pytest.MonkeyPatch()
    original = e.primary

    def load(root, identity, raw, refs):
        value = original(root, identity, raw, refs)
        if identity == 8022:
            value["public_panel_manifest"] = data
        return value

    with monkey.context() as patch:
        patch.setattr(e, "primary", load)
        value = e.seal(root, tmp_path / "other-raw")
    assert value["learning_inputs_ready_score"] == 1
    assert any(g["branch"] == "likelihood" for g in value["gate_check_summary"])


def test_missing_public_contract_is_terminal(tmp_path):
    """SCENARIO-REPORT-8032-CUSTODY: missing feature fields are not numerical zeros."""
    root = fixture_root(tmp_path)
    original = e.primary
    vault = root / "results" / "public-missing-q.json"
    atomic_json(vault, dict(role="fit", rows=[{k: v for k, v in public_row().items() if k != "q"}]))
    with pytest.MonkeyPatch.context() as patch:

        def load(root, identity, raw, refs):
            value = original(root, identity, raw, refs)
            if identity == 8019:
                value["public_manifests"]["fit"] = dict(path=str(vault), sha256=sha256_file(vault))
            return value

        patch.setattr(e, "primary", load)
        value = e.seal(root, tmp_path / "raw")
    assert value["gate_check_summary"]
    assert not value["learning_inputs_ready_score"]
    artifact = tmp_path / (e.NAME + ".json")
    atomic_json(artifact, value)
    assert e.replay(artifact)["passed"]


def test_current_health_receipt_is_reused_once(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8032-CLI: repeated owned checks retain the first bounded health failure."""
    root = fixture_root(tmp_path)
    value = e.seal(root, tmp_path / "raw")
    output = Path(os.environ["CARNOT_EXPERIMENT_ARTIFACT_ROOT"]) / (e.NAME + ".json")
    log = tmp_path / "health.log"
    log.write_text("historical health failure")
    health = dict(
        scope="repository_health",
        passed=False,
        log_path=str(log),
        log_sha256=sha256_file(log),
        exit_code=1,
    )
    atomic_json(
        output,
        dict(experiment_id=8032, task_id="exp8032-sealed-methods", repository_health=[health]),
    )
    calls = []
    monkeypatch.setattr(
        e,
        "validation_plan",
        lambda p: [
            e.CommandSpec("owned", ("owned",), "owned"),
            e.CommandSpec("health", ("health",), "repository_health"),
        ],
    )

    def run(root, commands, **kwargs):
        calls.extend(commands)
        return []

    monkeypatch.setattr(e, "run_commands", run)
    monkeypatch.setattr(e, "seal", lambda *a, **k: copy.deepcopy(value))
    monkeypatch.setattr(e, "terminal", lambda p: dict(passed=True))
    assert e.main([]) == 0
    assert all(c.scope == "owned" for c in calls)
    result = json.loads(output.read_text())
    assert result["repository_health"][0]["imported_diagnostic"]
    assert result["historical_exposure"]["owned_prior_attempt"]["sha256"]
