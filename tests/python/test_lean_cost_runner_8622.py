"""REQ-VERIFY-8622: bounded private publication and real child rejection paths."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import lean_cost_8622 as e
from carnot.reporting import lean_cost_runner_8622 as runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_private_cli_and_controls(tmp_path):
    """SCENARIO-VERIFY-8622-REPLAY: private missing inputs remain blocked, not successful."""
    output = tmp_path / (e.NAME + ".json")
    receipt = child(
        "private",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--date",
            "20261017",
            "--input-root",
            str(tmp_path),
            "--private-output",
            str(output),
            "--validation-receipts",
            str(tmp_path / "receipts.json"),
        ],
        tmp_path / "logs",
        deadline=90,
        expected=1,
    )
    assert receipt["passed"]
    atomic_json(tmp_path / "receipts.json", dict(receipts=[]))
    assert runner.main(["--date", "bad"]) == 1
    assert runner.main(["--input-root", str(tmp_path)]) == 1
    assert runner.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1
    assert (
        runner.main(
            [
                "--date",
                "20261017",
                "--input-root",
                str(tmp_path),
                "--private-output",
                str(output),
                "--validation-receipts",
                str(tmp_path / "receipts.json"),
            ]
        )
        == 0
    )
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert runner.main(["--cold-replay", str(output)]) == 0
    changed = json.loads(output.read_bytes())
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    assert runner.main(["--cold-replay", str(output)]) == 1


def test_session_deadline_and_worker_rejections(tmp_path):
    """REQ-VERIFY-8622: real worker failures cannot yield a clean timing receipt."""
    from carnot.reporting.lean_cost_session_8622 import session
    from carnot.reporting.lean_cost_worker_8622 import main

    with pytest.raises(TimeoutError):
        session({}, e.roster()[0], tmp_path, deadline_ns=0)
    assert main([]) == 1
    assert main(["missing", "wrong", "foreign"]) == 1
    result = child(
        "worker_missing",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.WORKER),
            str(tmp_path / "missing"),
            "wrong",
            "lean_python",
        ],
        tmp_path / "logs",
        expected=1,
        deadline=60,
    )
    assert result["passed"]


def test_owned_validation_failure(tmp_path):
    """REQ-REPORT-8622: owned-check failure disqualifies independently of upstreams."""
    work = e.measure(tmp_path, tmp_path / "raw", "20261017")
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, [dict(name="owned_tests", passed=False)], output)
    assert value["verdict_class"] == "disqualified"
    assert (
        value["lean_cost_ready_score"]
        == value["lean_service_benefit_score"]
        == value["nfr01_score"]
        == 0
    )
    atomic_json(output, value)
    e.replay(output)
    changed = deepcopy(value)
    changed["lean_cost_ready_score"] = 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    with pytest.raises(ValueError):
        e.replay(output)


def test_worker_bounded_loop_rejections(tmp_path):
    """REQ-VERIFY-8622: protocol controls cover bounds without numerical fixtures as science."""
    from types import SimpleNamespace
    from carnot.reporting import lean_cost_worker_8622 as worker

    count = 0
    kind = "limit"

    def receive(fd, timeout):
        nonlocal count
        count += 1
        return json.dumps(
            dict(
                config_hash="digest" if kind != "hash" else "wrong",
                request_id=str(count),
                deadline_ns=2**63,
                operation="infer",
                row=dict(rows=[{}] * (2 if kind == "batch" else 1)),
            )
        )

    wire = SimpleNamespace(
        configuration=lambda *a: ({"head": {}}, None, None),
        write_frame=lambda *a: None,
        read_frame=receive,
        batch=lambda *a: ([], []),
    )
    with (
        patch.object(worker.importlib, "import_module", return_value=wire),
        patch.object(worker.sys, "path", list(sys.path)),
    ):
        for kind in ["limit", "hash", "batch"]:
            assert worker.main([str(tmp_path), "digest", "lean_python"]) == 1


def test_validation_plan(tmp_path):
    """REQ-VERIFY-8622: collect original identities and compare their exact executed union."""
    from carnot.reporting import lean_cost_validation_8622 as validation

    node = "tests/python/test_private.py::test_owned"

    def fake(name, argv, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        stdout = folder / (name + ".stdout")
        stdout.write_text(node if name.startswith("collect") else "")
        if name.startswith("tests"):
            (folder / "tests0.xml").write_text("<testsuites/>")
        if name == "coverage_report":
            atomic_json(
                folder / "coverage.json",
                dict(
                    files={
                        p: dict(missing_lines=[], summary=dict(num_statements=1)) for p in e.OWNED
                    }
                ),
            )
        return dict(name=name, passed=True, stdout_path=str(stdout))

    with (
        patch.object(e, "TESTS", ["tests/python/test_private.py"]),
        patch.object(validation, "child", side_effect=fake),
        patch.object(validation, "xml_nodes", return_value=[node]),
    ):
        receipts = validation.validate(tmp_path)
    assert all(r["passed"] for r in receipts)
    assert next(r for r in receipts if r["name"] == "exact_node_union")["original_node_ids"] == [
        node
    ]


def test_session_rejection_cleanup(tmp_path):
    """REQ-VERIFY-8622: abnormal handshakes, responses and shutdowns fail and reap children."""
    from carnot.reporting import lean_cost_session_8622 as peer

    bundle = e.admit(e.ROOT)
    original = peer.read_frame
    for kind in ["ready", "response", "shutdown", "deadline"]:
        calls = 0

        def receive(fd, timeout):
            nonlocal calls
            calls += 1
            value = json.loads(original(fd, timeout))
            if (
                (kind == "ready" and calls == 1)
                or (kind == "response" and calls == 2)
                or (kind == "shutdown" and value.get("status") == "shutdown")
            ):
                value["status"] = "error"
            return json.dumps(value).encode()

        if kind == "deadline":
            clock = peer.time.monotonic_ns
            ticks = 0

            def bounded_clock():
                nonlocal ticks
                ticks += 1
                return 2**63 if ticks == 4 else clock()

            with (
                patch.object(peer.time, "monotonic_ns", side_effect=bounded_clock),
                pytest.raises(TimeoutError),
            ):
                peer.session(bundle, dict(e.roster()[2]), tmp_path / kind)
        else:
            with patch.object(peer, "read_frame", side_effect=receive), pytest.raises(ValueError):
                peer.session(bundle, dict(e.roster()[2]), tmp_path / kind)
        assert (tmp_path / kind / "failed_child.json").exists()


def test_private_validation_branch(tmp_path):
    """REQ-VERIFY-8622: the normal validation entry uses the owned validator."""
    with patch("carnot.reporting.lean_cost_validation_8622.validate", return_value=[]):
        assert (
            runner.main(
                [
                    "--input-root",
                    str(tmp_path),
                    "--private-output",
                    str(tmp_path / (e.NAME + ".json")),
                ]
            )
            == 0
        )


def test_actual_worker_rejects_bad_batch(tmp_path):
    """REQ-VERIFY-8622: a real child rejects an unsupported batch before returning a result."""
    import subprocess
    from carnot.reporting.resident_worker_8572 import envelope, read_frame, write_frame

    bundle = e.admit(e.ROOT)
    path = tmp_path / "configuration.json"
    atomic_json(path, dict(head=bundle["head"], extension=bundle["extension"]))
    digest = e.reference(path)["sha256"]
    with (tmp_path / "stderr").open("wb") as stderr:
        with subprocess.Popen(
            [sys.executable, "-u", str(e.ROOT / e.WORKER), str(path), digest, "lean_python"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        ) as process:
            assert process.stdin is not None and process.stdout is not None
            assert json.loads(read_frame(process.stdout.fileno(), 30))["status"] == "ready"
            write_frame(
                process.stdin.fileno(), envelope("bad", digest, dict(rows=bundle["requests"][:2]))
            )
            process.wait(timeout=30)
            assert process.returncode == 1
            assert read_frame(process.stdout.fileno(), 1) is None
    assert "batch_size" in (tmp_path / "stderr").read_text()
