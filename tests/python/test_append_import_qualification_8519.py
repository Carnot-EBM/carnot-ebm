"""REQ-REPORT-8519, REQ-VERIFY-8519: imports qualify readers, never speed."""

import ast
from pathlib import Path
import subprocess
import sys

from carnot.reporting import append_service_cost_8494 as old


def test_explicit_store_import():
    """SCENARIO-VERIFY-8519-IMPORT: the reader must own its Store import."""
    tree = ast.parse(Path(old.__file__).read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert any(
        n.module == "carnot.verify.append_store_8493" and any(a.name == "Store" for a in n.names)
        for n in imports
    )
    assert not any(
        isinstance(n, ast.Attribute) and ast.unparse(n) == "k.append.Store" for n in ast.walk(tree)
    )


def test_current_typed_reader():
    """SCENARIO-VERIFY-8519-IMPORT: keep strict export checking enabled."""
    proc = subprocess.run(
        [
            str(old.ROOT / ".venv/bin/mypy"),
            "--config-file=/dev/null",
            "--strict",
            "--follow-imports=skip",
            "--ignore-missing-imports",
            *old.OWNED,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_new_qualification_module():
    """SCENARIO-REPORT-8519-TERMINAL: qualification has current identity."""
    from carnot.reporting import append_import_qualification_8519 as q

    assert q.TASK == "exp8519-append-import-qualification"
    assert q.MODEL_SPECS == []


def test_historical_replay_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8519-VALIDATION: immutable operands defeat repaired hashes."""
    from carnot.reporting import append_import_qualification_8519 as q
    from carnot.reporting.current_work_receipt import atomic_json

    history = q.history(q.ROOT, tmp_path / "history")
    assert len(history["rows"]) == 40
    assert all(r["passed"] for r in history["rows"])
    assert history["historical_clock_eligibility"] == "historical_only_disqualified_8494"
    work = q.preflight(q.ROOT, tmp_path / "raw")
    work.update(history=history)
    receipts = [dict(name=n, passed=True) for n in q.REQUIRED]
    value = q.build(work, receipts, tmp_path / "output.json")
    assert value["append_runner_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert q.replay(candidate)
    value["append_runner_ready_score"] = 999
    value["reproducibility_checksum"] = q.checksum(value)
    atomic_json(candidate, value)
    assert not q.replay(candidate)


def test_missing_failure_and_cli(tmp_path):
    """SCENARIO-REPORT-8519-TERMINAL: private missing inputs are blocked."""
    from carnot.reporting import append_import_qualification_8519 as q
    from carnot.reporting.current_work_receipt import atomic_json

    work = q.preflight(tmp_path / "absent", tmp_path / "raw")
    assert work["gate_check_summary"]
    receipts = [dict(name=n, passed=True) for n in q.REQUIRED]
    value = q.build(work, receipts, tmp_path / "output.json")
    assert value["verdict_class"] == "blocked"
    assert value["append_runner_ready_score"] == 0
    value = q.build(work, [dict(name="owned", passed=False)], tmp_path / "output.json")
    assert value["verdict_class"] == "disqualified"
    candidate = tmp_path / "bad.json"
    atomic_json(candidate, {})
    assert not q.replay(candidate)
    assert not q.replay(tmp_path / "absent")
    for args, code in [
        (["--deliberate-error"], 7),
        (["--cold-replay", str(candidate)], 1),
        (["--date", "20261013"], 2),
        (["--history-root", str(tmp_path)], 2),
    ]:
        result = subprocess.run(
            [sys.executable, str(q.ROOT / q.CLI), *args], capture_output=True, timeout=60
        )
        assert result.returncode == code, result.stderr.decode()


def test_preflight_rejections(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8519-TERMINAL: observed fields and resource failures remain exact."""
    from copy import deepcopy
    from carnot.reporting import append_import_qualification_8519 as q

    real_authority, real_read = q.methods.authority, q.read
    for mode in ("task", "gate", "consumer", "activation", "tools"):
        with monkeypatch.context() as m:
            if mode == "tools":
                m.setattr(q.shutil, "which", lambda _: None)
            if mode in ("task", "activation"):

                def authority(root, raw):
                    value = real_authority(root, raw)
                    if mode == "task":
                        next(t for t in value["tasks"] if t["id"] == q.TASK)["deliverable"] = (
                            "wrong"
                        )
                    else:
                        value["activated"] = False
                        value["gate_check_summary"] = [q.gate(root, "activation", True, False)]
                    return value

                m.setattr(q.methods, "authority", authority)
            if mode in ("gate", "consumer"):

                def read(ref):
                    value = deepcopy(real_read(ref))
                    if "experiment_8512_" in ref["path"] and ref["path"].endswith(".json"):
                        if mode == "gate":
                            value["methods_ready_score"] = 0
                        else:
                            value["required_checks_passed"] = False
                    return value

                m.setattr(q, "read", read)
            work = q.preflight(q.ROOT, tmp_path / mode)
            assert work["gate_check_summary"], mode


def test_historical_operand_rejections(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8519-VALIDATION: each historical boundary rejects changed operands."""
    from copy import deepcopy
    import pytest
    from carnot.reporting import append_import_qualification_8519 as q

    real_read, real_hash = q.read, q.sha256_file
    for mode in (
        "authority",
        "date",
        "publication",
        "source",
        "contract",
        "tape",
        "bytes",
        "store",
        "disk",
    ):
        with monkeypatch.context() as m:

            def read(ref):
                value = deepcopy(real_read(ref))
                if mode == "date" and value.get("experiment_id") == 8493:
                    value["run_date"] = "20261014"
                if mode == "contract" and "measurement.json" in ref["path"]:
                    value["authority"]["tasks"] = []
                if (
                    mode == "tape"
                    and "frozen.json" in ref["path"]
                    and "experiment_8493" in ref["path"]
                ):
                    value["cell"]["batch"] = 999
                return value

            m.setattr(q, "read", read)
            if mode in ("authority", "source", "bytes"):

                def digest(path):
                    if (
                        (mode == "authority" and "v730-preserved" in str(path))
                        or (mode == "source" and str(path).endswith("verify/append_store_8493.py"))
                        or (mode == "bytes" and "operations-final.json" in str(path))
                    ):
                        return "changed"
                    return real_hash(path)

                m.setattr(q, "sha256_file", digest)
            if mode == "publication":
                m.setattr(q, "read_bound_sidecar", lambda *_: dict(report=dict(passed=False)))
            if mode == "store":
                m.setattr(q.store_reader, "reduce", lambda _: False)
            if mode == "disk":
                m.setattr(q.store_reader, "reduce", lambda _: True)
                m.setattr(q.Store, "read", lambda *_: {})
            with pytest.raises(ValueError):
                q.history(q.ROOT, tmp_path / mode)


def test_replay_rejection_fields(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8519-TERMINAL: recomputed aggregate digests cannot replace operands."""
    from copy import deepcopy
    from carnot.reporting import append_import_qualification_8519 as q
    from carnot.reporting.current_work_receipt import atomic_json

    work = q.preflight(tmp_path / "missing", tmp_path / "raw")
    value = q.build(work, [], tmp_path / "output.json")
    candidate = tmp_path / "candidate.json"
    for mode in ("identity", "checksum", "source", "history"):
        changed = deepcopy(value)
        if mode == "identity":
            changed["experiment_id"] = 0
        elif mode == "source":
            changed["code_config_hashes"][0]["sha256"] = "changed"
        elif mode == "history":
            work["history"] = dict(rows=[])
            changed = q.build(work, [], tmp_path / "output.json")
            monkeypatch.setattr(q, "history", lambda *_: dict(rows=[{}]))
        changed["reproducibility_checksum"] = (
            "invalid" if mode == "checksum" else q.checksum(changed)
        )
        atomic_json(candidate, changed)
        assert not q.replay(candidate)


def test_manifest_and_current_publication(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8519-VALIDATION: exact partitions and terminal failures control readiness."""
    import json
    from carnot.reporting import append_import_qualification_8519 as q
    from carnot.reporting import append_import_runner_8519 as runner
    from carnot.reporting.current_work_receipt import atomic_json
    from carnot.reporting.primary_publication import publish_primary

    monkeypatch.setattr(
        runner, "collect", lambda p, s: ([s + "::one", s + "::two"], dict(passed=True))
    )
    plan = runner.manifest(tmp_path / "plan")
    historical = [p for p in plan if p["name"].startswith("historical_") and p.get("nodes")]
    assert sum(len(p["nodes"]) for p in historical) == 10
    assert len({n for p in historical for n in p["nodes"]}) == 10
    union = next(p for p in plan if p["name"] == "historical_union")
    assert subprocess.run(union["argv"], capture_output=True, timeout=30).returncode == 0
    history = q.history(q.ROOT, tmp_path / "history")

    def child(name, argv, logs, **kwargs):
        stdout = logs / (name + ".stdout")
        atomic_json(stdout, dict(reports=[dict(flags=[])]))
        if name == "historical_replay":
            atomic_json(Path(argv[-1]), history)
        return dict(name=name, passed=True, duration_s=0.001, stdout_path=str(stdout))

    monkeypatch.setattr(runner, "child", child)

    def private_manifest(path):
        atomic_json(
            path / "coverage.json",
            dict(totals=dict(num_statements=1, covered_lines=1, missing_lines=0)),
        )
        atomic_json(path / "coverage_shards/coverage_fixture.json", dict(unit_fixture=True))
        atomic_json(path / "collection/collection_fixture.json", dict(unit_fixture=True))
        return []

    monkeypatch.setattr(runner, "manifest", private_manifest)
    monkeypatch.setattr(
        runner,
        "execute",
        lambda *_: [dict(name=n, passed=True, duration_s=0.001) for n in q.REQUIRED],
    )
    for mode in ("ready", "history_failed", "terminal_failed"):
        with monkeypatch.context() as m:
            if mode == "history_failed":

                def failed_child(name, argv, logs, **kwargs):
                    return dict(
                        child(name, argv, logs, **kwargs), passed=name != "historical_replay"
                    )

                m.setattr(runner, "child", failed_child)
            if mode == "terminal_failed":
                attempts = []

                def publish(output, value, validator):
                    if not attempts:
                        attempts.append(True)
                        validator(tmp_path / "candidate.json")
                        raise ValueError("candidate_rejected")
                    return publish_primary(output, value, validator)

                m.setattr(runner, "publish_primary", publish)
            output = tmp_path / mode / (q.NAME + ".json")
            assert runner.main(["--output", str(output)]) == 0
            value = json.loads(output.read_bytes())
            assert value["append_runner_ready_score"] == int(mode == "ready")
            assert value["verdict_class"] == (
                "circular_positive" if mode == "ready" else "disqualified"
            )


def test_real_private_cli(tmp_path):
    """SCENARIO-REPORT-8519-TERMINAL: standalone children publish and replay isolated blocked bytes."""
    import json
    from carnot.reporting import append_import_qualification_8519 as q

    output = tmp_path / (q.NAME + ".json")
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(q.ROOT / q.CLI),
            "--date",
            "20261014",
            "--private-control",
            "--root",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 40
    assert q.replay(output)
    assert (
        subprocess.run(
            [sys.executable, str(q.ROOT / q.CLI), "--private-control"],
            capture_output=True,
            timeout=60,
        ).returncode
        == 2
    )


def test_real_history_child(tmp_path):
    """SCENARIO-VERIFY-8519-VALIDATION: a fresh child reduces all original primitives."""
    import json
    from carnot.reporting import append_import_qualification_8519 as q

    output = tmp_path / "history.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            str(q.ROOT / q.CLI),
            "--history-root",
            str(q.ROOT),
            "--history-output",
            str(output),
        ],
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout.decode() + proc.stderr.decode()
    value = json.loads(output.read_bytes())
    assert len(value["rows"]) == 40 and all(r["passed"] for r in value["rows"])
    assert proc.stdout.decode().count("phase=historical_primitive") == 40


def test_repaired_reader_real_disk_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8519-IMPORT: the changed reader executes its direct Store path."""
    from copy import deepcopy
    from carnot.reporting.current_work_receipt import atomic_json
    from carnot.reporting import append_import_qualification_8519 as q

    producer = old.read(
        dict(
            path=str(old.ROOT / "results/experiment_8494_v730_append_service_cost.json"),
            sha256=q.PINS[8494],
        )
    )
    sealed = old.read(producer["work_reference"])
    work = old.measure(tmp_path / "missing", tmp_path / "reader")
    work.update(
        tapes=deepcopy(sealed["tapes"]), rows=deepcopy(sealed["rows"]), gate_check_summary=[]
    )
    monkeypatch.setattr(old, "admission", lambda *_: dict(tapes=sealed["tapes"]))
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    output = tmp_path / "reader_candidate.json"
    value = old.build(work, [], Path(work["raw"]), output)
    atomic_json(output, value)
    assert old.replay(output)


def test_missing_historical_precondition(tmp_path):
    """SCENARIO-REPORT-8519-TERMINAL: missing original producers block before child replay."""
    from carnot.reporting import append_import_qualification_8519 as q

    work = q.preflight(tmp_path / "missing", tmp_path / "raw")
    assert any(
        g.get("upstream") == q.store_reader.TASK
        and g.get("field") == "sha256"
        and g.get("expected") == q.PINS[8493]
        and g.get("observed") is None
        for g in work["gate_check_summary"]
    )


def test_original_authority_adapter(tmp_path):
    """SCENARIO-VERIFY-8519-VALIDATION: original gates use sealed V730 without editing V732."""
    from types import SimpleNamespace
    from carnot.reporting import append_import_history_8519 as adapter
    from carnot.reporting import append_import_qualification_8519 as q

    session = SimpleNamespace(
        config=SimpleNamespace(option=SimpleNamespace(basetemp=str(tmp_path)))
    )
    current = (q.ROOT / q.methods.ACTIVE).read_bytes()
    adapter.pytest_sessionstart(session)
    try:
        authority = q.store_reader.upstream.methods.authority(q.ROOT, tmp_path / "authority")
        assert authority["activated"]
        assert (
            q.store_reader.upstream.methods.tasks_digest(authority["tasks"])
            == q.store_reader.upstream.methods.TASKS_PIN
        )
        assert q.store_reader.admission(q.ROOT, tmp_path / "admission")["receipt"]["passed"]
    finally:
        adapter.pytest_sessionfinish(session, 0)
    adapter.pytest_sessionfinish(session, 0)
    assert (q.ROOT / q.methods.ACTIVE).read_bytes() == current


def test_original_authority_adapter_rejects_hash(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8519-VALIDATION: the private adapter cannot accept changed authority."""
    import pytest
    from types import SimpleNamespace
    from carnot.reporting import append_import_history_8519 as adapter

    protocol = adapter.methods.protocol()
    monkeypatch.setattr(adapter.methods, "protocol", lambda: protocol)
    monkeypatch.setattr(adapter.methods, "sha256_file", lambda _: "wrong")
    session = SimpleNamespace(
        config=SimpleNamespace(option=SimpleNamespace(basetemp=str(tmp_path)))
    )
    with pytest.raises(ValueError):
        adapter.pytest_sessionstart(session)
