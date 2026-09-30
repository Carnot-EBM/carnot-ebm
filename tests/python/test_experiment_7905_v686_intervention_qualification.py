"""REQ-REPORT-7905-V686: current model-free intervention qualification."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_scenario_report_7905_manifest(tmp_path: Path) -> None:
    """Freeze the dated CLI, affected closure, and private coverage locations."""
    from carnot import experiment_7905_v686_intervention_qualification as exp

    plan = exp.command_manifest(tmp_path)
    names = [row["name"] for row in plan["commands"]]
    assert len(names) == len(set(names))
    assert {
        "affected_pytest",
        "unit_coverage",
        "coverage_report",
        "e2e_016_fixture",
        "e2e_016_replay",
        "v686_cli_fixture",
        "v686_cli_replay",
        "v686_cli_failure",
    } <= set(names)
    assert exp.TEST in plan["affected_tests"]
    assert exp.MODULE in plan["affected_sources"]
    assert exp.CLI in plan["affected_sources"]
    assert all(row["timeout_s"] <= 300 for row in plan["commands"])
    for row in plan["commands"]:
        if row["name"] in {
            "e2e_016_fixture",
            "e2e_016_replay",
            "v686_cli_fixture",
            "v686_cli_replay",
            "v686_cli_failure",
        }:
            assert row["argv"][row["argv"].index("--date") + 1] == "20260930"
        assert all(not token.startswith("--basetemp=results") for token in row["argv"])


def test_scenario_report_7905_fixture_peer(tmp_path: Path) -> None:
    """A local HTTP peer returns raw scripted replies without model inference."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from urllib.request import Request, urlopen

    from carnot import experiment_7881_v684_intervention_protocol as old
    from carnot.verify import intervention_protocol_7868 as protocol

    class Peer(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            json.loads(body)["messages"]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(old.family(0)[1].encode())

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Peer)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        row, _reply, finish = old.family(0)
        from carnot.verify import context_sufficiency_7854 as context

        request = context.build_request(
            row,
            "full_source",
            None,
            None,
            context.freeze_protocol(seed=old.SEED),
            lambda text: len(text.split()),
        )
        wire = Request(
            f"http://127.0.0.1:{server.server_port}/",
            data=json.dumps(request).encode(),
            method="POST",
        )
        with urlopen(wire, timeout=5) as reply:
            raw = reply.read().decode()
        case = protocol.fixture_case(row, raw, finish, old.SEED)
        assert case["requests"][0] == request
        assert case["source_byte_fidelity"] is True
        assert case["semantic_sensitivity"] is None
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_scenario_report_7905_cli_fixture_and_replay(tmp_path: Path) -> None:
    """The real dated routes keep output and checkpoints in private storage."""
    from carnot import experiment_7905_v686_intervention_qualification as exp

    fixture = tmp_path / "fixture.json"
    for route in ("--fixture-e2e", "--cold-replay"):
        observed = subprocess.run(
            [
                sys.executable,
                "-u",
                str(exp.ROOT / exp.CLI),
                "--date",
                "20260930",
                route,
                str(fixture),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert observed.returncode == 0, observed.stdout + observed.stderr
    payload = json.loads(fixture.read_text())
    assert payload["independent_families"] == 24
    assert len(payload["rows"]) == 96
    assert payload["model_calls"] == 0
    failure = subprocess.run(
        [
            sys.executable,
            "-u",
            str(exp.ROOT / exp.CLI),
            "--date",
            "20260930",
            "--cold-replay",
            str(tmp_path / "missing.json"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert failure.returncode != 0 and "FileNotFoundError" in failure.stderr


def test_scenario_report_7905_prior_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing V685 authority closes with an exact blocked operand."""
    from carnot import experiment_7905_v686_intervention_qualification as exp

    monkeypatch.setattr(exp, "PRIOR", tmp_path / "missing.json")
    checks, hashes = exp.preflight(exp.command_manifest(tmp_path))
    assert any(row["upstream_id"] == "prior_exp7893" and not row["passed"] for row in checks)
    assert hashes["prior_exp7893"]["sha256"] is None


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "blocked",
        "required_failure",
        "terminal_failure",
        "publish_failure",
        "wrong_date",
        "manifest_drift",
        "missing_receipt",
        "candidate_drift",
        "terminal_first_failure",
        "terminal_final_failure",
    ],
)
def test_scenario_report_7905_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Private orchestration retains actual failures and checks final bytes twice."""
    from carnot import experiment_7905_v686_intervention_qualification as exp

    scratch = tmp_path / "scratch"
    output = tmp_path / "result.json"
    monkeypatch.setattr(exp, "RAW", tmp_path / "raw")
    if mode == "wrong_date":
        with pytest.raises(ValueError, match="run_date_mismatch"):
            exp.run("20260929", scratch, output)
        return
    if mode == "manifest_drift":
        exp.atomic_json(exp.RAW / "validation_command_manifest.json", {"drift": True})
        with pytest.raises(ValueError, match="validation_manifest_drift"):
            exp.run("20260930", scratch, output)
        return
    if mode == "blocked":
        monkeypatch.setattr(exp, "PRIOR", tmp_path / "absent.json")
        row = exp.run("20260930", scratch, output)
        assert row["verdict_class"] == "blocked"
        assert row["gate_check_summary"]
        assert output.is_file()
        return
    plan = exp.command_manifest(scratch)
    log = tmp_path / "child.log"
    log.write_text("completed")
    receipts = [
        {
            "name": item["name"],
            "classification": "required",
            "passed": mode != "required_failure" or index > 0,
            "command_argv": item["argv"],
            "log_path": str(log),
            "log_sha256": exp.sha256_file(log),
        }
        for index, item in enumerate(plan["commands"])
    ]
    for receipt in receipts:
        if receipt["name"] == "coverage_report":
            receipt["output_tail"] = (
                "python/carnot/experiment_7905_v686_intervention_qualification.py 177 0 100%"
            )
    monkeypatch.setattr(
        exp.prior_code,
        "execute",
        lambda *_args: receipts[:-1] if mode == "missing_receipt" else receipts,
    )
    if mode == "missing_receipt":
        with pytest.raises(ValueError, match="missing_required_command"):
            exp.run("20260930", scratch, output)
        return
    seen: list[str] = []

    def terminal(candidate: Path, _scratch: Path, _started: float, attempt: int):
        digest = exp.sha256_file(candidate)
        seen.append(digest)
        if mode == "terminal_first_failure":
            passed = attempt > 0
        elif mode == "terminal_final_failure":
            passed = attempt == 1
        else:
            passed = mode != "terminal_failure" or attempt == 0
        report = {
            "candidate_sha256": "sha256:wrong"
            if mode == "candidate_drift" and attempt == 0
            else digest,
            "reports": [{"name": "terminal_adversarial", "passed": passed, "log_path": str(log)}],
            "flagged_adversarial": not passed,
        }
        return passed, not passed, report

    monkeypatch.setattr(exp, "validate_terminal", terminal)
    if mode == "publish_failure":
        monkeypatch.setattr(
            exp.prior_code,
            "publish_exact",
            lambda *_args: (_ for _ in ()).throw(ValueError("publish_failed")),
        )
        with pytest.raises(ValueError, match="publish_failed"):
            exp.run("20260930", scratch, output)
        assert not output.exists()
        return
    if mode in {"terminal_failure", "terminal_final_failure"}:
        with pytest.raises(ValueError, match="terminal_revalidation_failed"):
            exp.run("20260930", scratch, output)
        assert not output.exists()
        return
    if mode == "candidate_drift":
        with pytest.raises(ValueError, match="terminal_candidate_drift"):
            exp.run("20260930", scratch, output)
        assert not output.exists()
        return
    row = exp.run("20260930", scratch, output)
    assert row["verdict_class"] == (
        "disqualified"
        if mode in {"required_failure", "terminal_first_failure"}
        else "circular_positive"
    )
    assert row["intervention_protocol_ready_score"] == int(mode == "success")
    assert len(seen) == (3 if mode == "terminal_first_failure" else 2)
    assert seen[-1] == seen[-2] == exp.sha256_file(output)
    assert row["coverage_statement_counts"]
    assert Path(row["fixture_rows_path"]).is_relative_to(exp.RAW)
    assert all(
        Path(item["log_path"]).is_relative_to(exp.RAW) for item in row["validation_receipts"]
    )


def test_scenario_report_7905_main_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bounded main route returns the current classification."""
    from carnot import experiment_7905_v686_intervention_qualification as exp

    monkeypatch.setattr(exp, "run", lambda *_args: {"verdict_class": "disqualified"})
    assert (
        exp.main(
            [
                "--date",
                "20260930",
                "--scratch",
                str(tmp_path),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 1
    )
    with pytest.raises(ValueError, match="run_date_mismatch"):
        exp.main(["--date", "20260929", "--fixture-e2e", str(tmp_path / "fixture.json")])


def test_scenario_report_7905_independent_exclusions() -> None:
    """Valid JSON cannot supply a missing sentence or bypass the request limit."""
    from carnot import experiment_7881_v684_intervention_protocol as old
    from carnot.verify import context_sufficiency_7854 as context
    from carnot.verify import intervention_protocol_7868 as protocol

    row, reply, finish = old.family(6)
    case = protocol.fixture_case(row, reply, finish, old.SEED)
    assert case["status"] == "excluded_no_complete_sentence"
    assert case["requests"] == []
    row, _reply, _finish = old.family(0)
    frozen = context.freeze_protocol(seed=old.SEED)
    frozen["context_ceiling_tokens"] = 1
    with pytest.raises(ValueError, match="context_budget"):
        context.build_request(
            row, "full_source", None, None, frozen, lambda text: len(text.split())
        )


def test_scenario_report_7905_historical_date_guard() -> None:
    """The V685 CLI still rejects a date outside its own frozen run."""
    from carnot import experiment_7893_v685_intervention_protocol as old

    with pytest.raises(ValueError, match="run_date_mismatch"):
        old.main(["--date", "20260930", "--fixture-e2e", "/tmp/unused-7905.json"])
