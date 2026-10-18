"""REQ-VERIFY-8641: new qualification never repairs historical failures."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import learning_execution_custody_8641 as e
from carnot.reporting.current_work_receipt import atomic_json


def test_upstream_missing_zero_and_historical_independence(tmp_path, monkeypatch):
    """REQ-REPORT-8641: operands, not failed execution, control admission."""
    assert e.upstream(e.ROOT)["failures"] == []
    assert e.upstream(tmp_path)["failures"][0]["observed"] is None
    path = tmp_path / "results" / (e.METHODS + ".json")
    atomic_json(path, dict(methods_ready_score=0, learning_operands_ready_score=0))
    gates = e.upstream(tmp_path)["failures"]
    assert any(g["observed"] == 0 for g in gates)
    atomic_json(path, dict(methods_ready_score=1, learning_operands_ready_score=1))
    assert any(g["check"] == "upstream_publication" for g in e.upstream(tmp_path)["failures"])
    sidecar = tmp_path / "sidecar.json"
    atomic_json(sidecar, dict(publication=dict(sidecar_path="missing")))
    atomic_json(
        path,
        dict(
            methods_ready_score=1,
            learning_operands_ready_score=1,
            terminal_validation_sidecar_path=str(sidecar),
        ),
    )
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    assert any(g["check"] == "upstream_publication" for g in e.upstream(tmp_path)["failures"])


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Prepare once without executing the historical experiment controllers."""
    return e.prepare(tmp_path_factory.mktemp("custody8641") / "raw")


def test_prepared_schema_and_replay_rejections(prepared, tmp_path):
    """SCENARIO-VERIFY-8641-REPLAY: missing execution cannot become readiness."""
    value = e.build(prepared, tmp_path / (e.NAME + ".json"))
    assert len(prepared["source_refs"]) == 6
    assert value["experiment_id"] == 8641
    assert value["learning_execution_ready_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["model_invocation_counts"]["current_llm_calls"] == 0
    assert not e.qualified(prepared)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path) and not e.replay(tmp_path / "absent")
    for attack in [
        "score",
        "authority",
        "source",
        "fixture",
        "checkpoint",
        "pin",
        "fixture_rehashed",
        "projection",
        "completed",
        "failures",
        "manifest",
        "failed_evidence",
    ]:
        work = deepcopy(prepared)
        changed = deepcopy(value)
        if attack == "score":
            changed["learning_execution_ready_score"] = 1
        elif attack == "authority":
            work["original_authority"] = {}
        elif attack == "source":
            work["current_source_reference"][0]["sha256"] = "wrong"
        elif attack == "fixture":
            work["mechanics_fixture"]["sha256"] = "wrong"
        elif attack == "checkpoint":
            work["checkpoint_reference"] = dict(path="foreign", sha256="rehashed")
        elif attack == "pin":
            foreign = tmp_path / "foreign-upstream.json"
            foreign.write_bytes(Path(work["source_refs"][0]["path"]).read_bytes())
            work["source_refs"][0] = e.ref(foreign)
        elif attack == "fixture_rehashed":
            fixture = e.read(work["mechanics_fixture"])
            fixture["owned_ready"] = True
            atomic_json(tmp_path / "changed-fixture.json", fixture)
            work["mechanics_fixture"] = e.ref(tmp_path / "changed-fixture.json")
        elif attack == "projection":
            projection = Path(work["mechanics_fixture"]["path"]).parent / "stream_rows.json"
            saved = projection.read_bytes()
            atomic_json(projection, dict(rows=[]))
        elif attack == "completed":
            work["completed_execution"] = True
        elif attack == "manifest":
            work["child_shard_manifest"] = [dict(data=[])]
        elif attack == "failed_evidence":
            atomic_json(tmp_path / "failed-evidence.json", dict(error="test", last_receipt=dict()))
            work["execution_failure_reference"] = e.ref(tmp_path / "failed-evidence.json")
        else:
            work["failures"] = [dict(check="invented")]
        atomic_json(tmp_path / "work.json", work)
        if attack in {"checkpoint", "pin"}:
            changed = e.build(work, tmp_path / (e.NAME + ".json"))
        changed["primitive_reference"] = e.ref(tmp_path / "work.json")
        changed["reproducibility_checksum"] = e.base.digest(changed)
        atomic_json(path, changed)
        assert not e.replay(path), attack
        if attack == "projection":
            projection.write_bytes(saved)


def test_private_cli_and_error_paths(prepared, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8641-REPLAY: actual private publication and fresh rejection."""
    argv = [str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    for name, args, expected in [
        (
            "private",
            ["--private-output", str(output), "--work", str(Path(prepared["raw"]) / "work.json")],
            0,
        ),
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("date", ["--date", "20260230"], 2),
        ("unsafe", ["--private-output", str(e.ROOT / "bad")], 2),
        ("owned", ["--deliberate-error"], 7),
    ]:
        receipt = e.child(
            name,
            [sys.executable, "-u", *argv, *args],
            tmp_path / "logs",
            expected=expected,
            deadline=120,
        )
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert e.main(["--date", "bad"]) == 2
    assert e.main(["--date", "20260230"]) == 2
    assert e.main(["--private-output", str(e.ROOT / "bad")]) == 2
    assert e.main(["--deliberate-error"]) == 7
    assert e.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(e, "run", lambda *a, **k: 0)
    assert e.main(["--date", "20261018"]) == 0


def test_unresumed_dispatch(prepared, tmp_path, monkeypatch):
    """REQ-VERIFY-8641: normal dispatch invokes only the new execution adapter."""
    from carnot.reporting import learning_custody_validation_8641 as v

    work = deepcopy(prepared)
    work["raw"] = str(tmp_path / "raw")
    calls = []
    monkeypatch.setattr(e, "prepare", lambda raw: work)
    monkeypatch.setattr(v, "execute", lambda w: calls.append(w))
    assert e.run(tmp_path / (e.NAME + ".json"), "20261018") == 0
    assert len(calls) == 1
    failed = deepcopy(prepared)
    failed["raw"] = str(tmp_path / "failed")
    monkeypatch.setattr(e, "prepare", lambda raw: failed)

    def failure(candidate):
        receipt = e.child(
            "failed", [sys.executable, "-c", "raise SystemExit(7)"], Path(candidate["raw"]) / "logs"
        )
        candidate["receipts"].append(receipt)
        raise ValueError("failed_child")

    monkeypatch.setattr(v, "execute", failure)
    output = tmp_path / "failed" / (e.NAME + ".json")
    assert e.run(output, "20261018") == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_external_precondition_failure_blocks(prepared, tmp_path):
    """REQ-REPORT-8641: unavailable operands block; they never become a scientific null."""
    work = deepcopy(prepared)
    work["failures"] = e.upstream(tmp_path)["failures"]
    value = e.build(work, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    assert value["learning_execution_ready_score"] == 0
