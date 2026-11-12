"""REQ-REPORT-8990; SCENARIO-REPORT-8990-CUSTODY: private shard custody checks."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import graph_corpus_a_8990 as d
from carnot.reporting import graph_corpus_runner_8961 as runner
from carnot.reporting.current_work_receipt import atomic_json


def test_authentication_roster_and_tamper(tmp_path, monkeypatch):
    """REQ-REPORT-8990: only the immutable admitted source permits collection."""
    evidence = d.authenticate(d.ROOT)
    assert not evidence["failures"]
    units = d.ordered_units(evidence["protocol"])
    assert len(units) == len({u["graph_hash"] for u in units}) == 512
    assert all(u["shard"] == "A" for u in units)
    assert d.authenticate(tmp_path)["failures"][0]["observed"] == "missing_file"
    value = json.loads((d.ROOT / d.INPUTS[0][0]).read_bytes())
    for field, bad in [
        ("collection_ready_score", 0),
        ("collection_ready_score", True),
        ("collection_ready_score", None),
        ("required_checks_passed", False),
    ]:
        changed = deepcopy(value)
        changed[field] = bad
        path = tmp_path / d.INPUTS[0][0]
        atomic_json(path, changed)
        monkeypatch.setattr(
            d,
            "INPUTS",
            [
                (
                    d.INPUTS[0][0],
                    d.reference(path)["sha256"],
                    "collection_ready_score",
                    "exp8987-spline-development-admission",
                )
            ],
        )
        assert d.authenticate(tmp_path)["failures"]
    changed = deepcopy(value)
    changed["role_manifest"][0]["role"] = "test"
    atomic_json(path, changed)
    monkeypatch.setattr(
        d,
        "INPUTS",
        [
            (
                d.INPUTS[0][0],
                d.reference(path)["sha256"],
                "collection_ready_score",
                "exp8987-spline-development-admission",
            )
        ],
    )
    assert d.authenticate(tmp_path)["failures"]
    changed = deepcopy(value)
    changed["frozen_protocol"]["degree"] = 99
    atomic_json(path, changed)
    monkeypatch.setattr(
        d,
        "INPUTS",
        [
            (
                d.INPUTS[0][0],
                d.reference(path)["sha256"],
                "collection_ready_score",
                "exp8987-spline-development-admission",
            )
        ],
    )
    assert "spline_protocol_drift" in str(d.authenticate(tmp_path)["failures"])


def test_build_and_dispatch(tmp_path, monkeypatch):
    """REQ-REPORT-8990: qualify only owned evidence and keep fixture readiness zero."""
    work = runner.freeze(d.ROOT, tmp_path / "raw", True, producer=d)
    validation = dict(receipts=[], coverage={})
    atomic_json(Path(work["raw"]) / "work.json", work)
    atomic_json(Path(work["raw"]) / "validation.json", validation)
    value = d.build(work, validation, True)
    assert value["experiment_id"] == 8990 and value["run_date"] == "20261111"
    assert value["shard_ready_score"] == 0 and value["inference_mode"] == "no_model_load"
    assert len(value["first_response_rows"]) == 512 and value["completed_count"] == 0
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    assert not d.qualified(validation)
    monkeypatch.setattr(d, "qualified", lambda _: True)
    work["authentication"]["failures"] = [dict(check="collection_gate", observed=0, passed=False)]
    blocked = d.build(work, validation, False)
    assert blocked["honest_verdict"] == "complete_blocked_collection_gate"
    assert blocked["verdict_class"] == "blocked"


def test_private_cli_replay(tmp_path):
    """SCENARIO-REPORT-8990-CUSTODY: E2E foreign cwd, missing input and tamper."""
    output = tmp_path / (d.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(d.ROOT / d.CLI)]

    def child(args):
        print("before_private_child", args, flush=True)
        result = subprocess.run(
            command + args, cwd="/tmp", env=env, capture_output=True, text=True, timeout=90
        )
        print("after_private_child", result.returncode, result.stdout[-1500:], flush=True)
        return result

    result = child(["--date", "20261111", "--fixture-e2e", str(output)])
    assert result.returncode == 0, result.stdout + result.stderr
    assert child(["--cold-replay", str(output)]).returncode == 0
    value = json.loads(output.read_bytes())
    value["completed_count"] = 512
    value["reproducibility_checksum"] = d.shared.pilot.checksum(value)
    atomic_json(output, value)
    assert child(["--cold-replay", str(output)]).returncode == 1
    pristine = json.loads(output.read_bytes())
    raw = Path(pristine["raw_shard_hashes"][0]["path"]).parent
    raw_work = raw / "work.json"
    raw_work.write_bytes(raw_work.read_bytes() + b" ")
    assert child(["--cold-replay", str(output)]).returncode == 1
    env["CARNOT_8990_INPUT_ROOT"] = str(tmp_path / "absent")
    blocked = tmp_path / "blocked" / (d.NAME + ".json")
    assert child(["--date", "20261111", "--fixture-e2e", str(blocked)]).returncode == 0
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"
    assert child(["--cold-replay", str(blocked)]).returncode == 0
    assert child(["--date", "wrong"]).returncode == 2
    assert child(["--date", "20261111", "--fixture-e2e", str(d.ROOT / "bad.json")]).returncode == 2


def test_preflight_resources_cache_tokenizer(tmp_path, monkeypatch):
    """REQ-REPORT-8990: resource, cache and tokenizer failures stop before weight loading."""
    from carnot.reporting import graph_corpus_runner_8990 as r

    def resources(raw, evidence):
        return dict(private=True)

    monkeypatch.setattr(r.basis, "preflight", resources)
    monkeypatch.setattr(
        r, "_preflight", lambda raw: dict(model=dict(model_path="private.gguf", sha256=r.MODEL_PIN))
    )
    monkeypatch.setattr(r, "gguf_tokenizer_loadable", lambda path: (True, "private tokenizer"))
    result = r.preflight(tmp_path)
    assert result["embedded_tokenizer"]["passed"] and result["resources"]["private"]
    monkeypatch.setattr(r, "gguf_tokenizer_loadable", lambda path: (False, "private rejected"))
    with pytest.raises(RuntimeError, match="embedded_tokenizer"):
        r.preflight(tmp_path)
    monkeypatch.setattr(
        r,
        "_preflight",
        lambda raw: dict(model=dict(model_path="private.gguf", sha256="sha256:wrong")),
    )
    with pytest.raises(RuntimeError, match="model_cache_hash"):
        r.preflight(tmp_path)

    def missing(raw):
        raise ImportError("private dependency")

    monkeypatch.setattr(r, "_preflight", missing)
    with pytest.raises(RuntimeError, match="runtime_dependency"):
        r.preflight(tmp_path)

    def failed_resources(raw, evidence):
        evidence["failures"].append(dict(check="memory", observed=0))
        return {}

    monkeypatch.setattr(r.basis, "preflight", failed_resources)
    with pytest.raises(RuntimeError, match="resource_preconditions"):
        r.preflight(tmp_path)
