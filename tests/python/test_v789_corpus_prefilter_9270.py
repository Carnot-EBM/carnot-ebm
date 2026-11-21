from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v788_corpus_prefilter as p
from carnot.reporting import v789_corpus_prefilter_runner as r


def test_req_9270_scoped_required_checks_exclude_full_python_suite(tmp_path: Path) -> None:
    # REQ-REPORT-9270-V789 / SCENARIO-REPORT-9270-V789-CONTROLS: retain all 9 affected checks and exclude full_python_suite.
    scope = r.build_scope(tmp_path, tmp_path / "raw")
    expected_checks = [
        "target_module",
        "owned_tests",
        "v782_controls",
        "direct_consumers",
        "e2e018_021",
        "ruff_check",
        "ruff_format",
        "strict_mypy",
        "spec_coverage",
    ]
    assert list(r.REQUIRED_CHECKS) == expected_checks
    assert scope["required_check_names"] == expected_checks
    assert [cmd["name"] for cmd in scope["commands"]] == expected_checks
    assert "full_python_suite" not in scope["required_check_names"]
    by_name = {cmd["name"]: cmd for cmd in scope["commands"]}
    assert "coverage" not in by_name["target_module"]["argv"]
    assert "coverage" in by_name["owned_tests"]["argv"]
    assert "coverage" in by_name["v782_controls"]["argv"]
    assert "--unset=COVERAGE_PROCESS_START" in by_name["target_module"]["argv"]


def test_req_9270_compactness_guard_and_hashed_sidecars(tmp_path: Path) -> None:
    # REQ-REPORT-9270-V789 / SCENARIO-REPORT-9270-V789-CONTROLS: primary JSON < 10 MB and bulky tables live in hashed sidecars.
    r.compact_guard({"small": True})
    with pytest.raises(ValueError, match="primary_size_exceeded"):
        r.compact_guard({"bloat": "x" * r.MAX_PRIMARY_BYTES})

    av = p.load_av()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "pos.json").write_text(
        json.dumps({"inference_substrate": av.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE, "duration_s": 0.05}),
        encoding="utf-8",
    )
    (results_dir / "neg.json").write_text(
        json.dumps({"inference_substrate": "no_model_load", "duration_s": 0.1}),
        encoding="utf-8",
    )
    raw = tmp_path / "raw"
    scan = p.scan_corpus(results_dir, av)
    compact_scan = r.write_scan_sidecars(raw, scan)
    assert "manifest_rows" not in compact_scan
    assert "scan_rows" not in compact_scan
    m_ref = compact_scan["manifest_sidecar_reference"]
    s_ref = compact_scan["scan_rows_sidecar_reference"]
    assert Path(m_ref["path"]).is_file()
    assert Path(s_ref["path"]).is_file()
    assert p.file_ref(Path(m_ref["path"]), None)["sha256"] == m_ref["sha256"]
    assert p.file_ref(Path(s_ref["path"]), None)["sha256"] == s_ref["sha256"]
    assert s_ref["row_count"] == 2
    lines = Path(s_ref["path"]).read_text("utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["path"] == "neg.json"
    assert r.write_scan_sidecars(raw, compact_scan) == compact_scan
    Path(s_ref["path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scan_sidecar_changed"):
        r.write_scan_sidecars(raw, compact_scan)


def test_req_9270_gate_operands_and_exp9256_streaming_authentication(tmp_path: Path) -> None:
    # REQ-REPORT-9270-V789 / SCENARIO-REPORT-9270-V789: distinguish missing_file, missing_field, and observed 0; stream Exp9256 metadata.
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    path = root / r.UPSTREAM_PINS["exp9269-contract"][0]
    path.write_text(json.dumps({"contract_ready_score": 0}), encoding="utf-8")
    work = r.inspect_v789(root, tmp_path / "raw", authenticate_only=True)
    by_check = {row["check"]: row for row in work["gate_checks"]}
    assert by_check["exp9269_contract_ready_score"]["observed"] == 0
    assert by_check["exp9269_replay_authority_ready_score"]["observed"] == "missing_field"
    assert by_check["exp9256_sha256"]["observed"] == "missing_file"

    live = r.inspect_v789(p.ROOT, tmp_path / "live", authenticate_only=True)
    assert all(row["passed"] for row in live["gate_checks"]), [
        row for row in live["gate_checks"] if not row["passed"]
    ]
    assert live["exp9181_timeout_diagnostics"]["direct_consumers_exit_code"] == -15
    assert live["exp9256_historical_summary"]["corpus_scan_ready_score"] == 0
    assert live["exp9256_historical_summary"]["visited_files"] == 764150


def test_req_9270_target_module_and_historical_consumers_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-9270-V789 / SCENARIO-REPORT-9270-V789: exercise modified statements in target_module and historical consumers under coverage.
    target_path = p.ROOT / "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py"
    spec = importlib.util.spec_from_file_location("target_floor_9270", target_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "valid_no_llm.json").write_text(
        json.dumps(
            {
                "inference_substrate": mod.AV.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE,
                "duration_s": 0.05,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "REPO", tmp_path)
    mod.test_no_artifact_on_disk_is_newly_flagged_by_closing_this_hole()


def _child_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Use the real CLI outside the checkout so imports cannot depend on PYTHONPATH."""
    script = p.ROOT / "scripts/experiments/experiment_9270_v789_corpus_prefilter.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [sys.executable, "-u", str(script), *args]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, check=False)


def test_req_9270_private_cli_publication_refusal_tamper_and_cold_replay(tmp_path: Path) -> None:
    # SCENARIO-REPORT-9270-V789-CONTROLS: private CLI publication, refusal, tamper, and cold replay outside checkout.
    root = tmp_path / "private"
    outside = tmp_path / "outside"
    outside.mkdir()
    raw = root / "raw"
    out = root / r.OUT
    args = ("--date", r.RUN_DATE, "--root", str(root), "--raw-dir", str(raw), "--output", str(out))
    published = _child_cli(outside, *args)
    assert published.returncode == 0, published.stdout + published.stderr
    assert out.stat().st_size < r.MAX_PRIMARY_BYTES
    value = json.loads(out.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["corpus_scan_ready_score"] == 0
    assert "scan_rows" not in value
    assert "ordered_paths" not in value["corpus_manifest_reference"]
    assert _child_cli(outside, *args, "--cold-replay").returncode == 0
    assert _child_cli(outside, *args, "--from-work").returncode == 0
    assert _child_cli(outside, *args, "--checks-only").returncode == 1
    assert _child_cli(outside, "--date", "19990101").returncode == 1
    assert _child_cli(outside, "--from-work").returncode == 1
    pristine = out.read_bytes()

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * r.MAX_PRIMARY_BYTES)
    with pytest.raises(ValueError, match="primary_size_exceeded"):
        r.replay(oversized)

    value["corpus_scan_ready_score"] = 1
    out.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum_mismatch"):
        r.replay(out)
    value["reproducibility_checksum"] = r.checksum(value)
    out.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="replay_reduction_drift"):
        r.replay(out)
    assert _child_cli(outside, *args, "--cold-replay").returncode == 1

    for field, error in (
        ("work_reference", "work_reference_hash_mismatch"),
        ("corpus_manifest_reference", "manifest_sidecar_hash_mismatch"),
        ("scan_rows_sidecar_reference", "scan_rows_sidecar_hash_mismatch"),
    ):
        value = json.loads(pristine)
        value[field]["sha256"] = "sha256:bad"
        value["reproducibility_checksum"] = r.checksum(value)
        out.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            r.replay(out)
    out.write_bytes(pristine)


def test_req_9270_reduction_sidecar_verification_and_current_work_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-9270-V789 / SCENARIO-REPORT-9270-V789-CONTROLS: sidecar tamper and current_work tamper fail replay.
    av = p.load_av()
    root = tmp_path / "authority"
    for rel, _ in r.UPSTREAM_PINS.values():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(p.ROOT / rel)
    for rel in ("scripts/adversarial_verify.py", "scripts/verdict_row_consistency_lint.py"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(p.ROOT / rel)
    results = root / "results"
    (results / "positive.json").write_text(
        json.dumps({"inference_substrate": av.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE, "duration_s": 0.02}),
        encoding="utf-8",
    )
    raw = tmp_path / "raw"
    scan = p.scan_corpus(tmp_path / "empty", av)
    inspected = r.inspect_v789(root, raw, scan_summary=scan)
    assert all(row["passed"] for row in inspected["gate_checks"])
    assert inspected["controls"]["all_mutations_detected"]
    inspected_auto = r.inspect_v789(tmp_path / "absent", tmp_path / "raw_absent")
    assert any(not row["passed"] for row in inspected_auto["gate_checks"])

    pos_dir = tmp_path / "pos_results"
    pos_dir.mkdir()
    (pos_dir / "positive.json").write_text(
        json.dumps({"inference_substrate": av.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE, "duration_s": 0.02}),
        encoding="utf-8",
    )
    pos_scan = p.scan_corpus(pos_dir, av)
    inspected_pos = r.inspect_v789(root, raw, scan_summary=pos_scan)
    work = r.measure(root, raw, r.RUN_DATE, inspection=inspected_pos)
    value = r.build(root, raw, work, {}, r.RUN_DATE)
    assert value["verdict_class"] == "disqualified"
    assert value["corpus_scan_ready_score"] == 0
    assert value["inference_substrate"] == "read_only_live_artifact_inventory"
    candidate = raw / "candidate.json"
    r.atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]

    with monkeypatch.context() as patcher:
        patcher.setattr(r, "validate_current_work_receipt", lambda *a: ["bad"])
        with pytest.raises(ValueError, match="current_work_receipt_invalid"):
            r.replay(candidate)
    work["current_work_reference"]["sha256"] = "sha256:bad"
    r.atomic_json(raw / "work.json", work)
    value["work_reference"] = p.file_ref(raw / "work.json", None)
    value["reproducibility_checksum"] = r.checksum(value)
    r.atomic_json(candidate, value)
    with pytest.raises(ValueError, match="current_work_hash_mismatch"):
        r.replay(candidate)
    assert r._resolve_ref_path("relative", tmp_path) == tmp_path / "relative"


def test_req_9270_receipt_validation_coverage_and_main_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-9270-V789 / SCENARIO-REPORT-9270-V789: receipt validation, statement coverage, and main phases.
    from coverage import Coverage

    scope = r.build_scope(tmp_path, tmp_path / "raw")
    assert r.receipts_pass({"scope": scope}, {}) is False
    forged = {
        "checks": [
            {"name": name, "passed": True, "exit_code": 0} for name in scope["required_check_names"]
        ]
    }
    assert r.receipts_pass({"scope": scope}, forged) is False
    receipt = r.run(
        "real_unit_child",
        [sys.executable, "-c", "print('actual child')"],
        tmp_path / "logs",
        r.time.monotonic() + 10,
    )
    assert receipt["passed"]
    small = {
        "scope": {"required_check_names": ["real_unit_child"], "owned_changed_paths": list(r.OWNED)}
    }
    assert r.receipts_pass(small, {"checks": [receipt]}) is False
    Path(receipt["stdout_path"]).chmod(0o600)
    Path(receipt["stdout_path"]).write_text("tampered", encoding="utf-8")
    assert r.receipts_pass(small, {"checks": [receipt]}) is False

    source = tmp_path / "module.py"
    source.write_text("value = 2\n", encoding="utf-8")
    base = tmp_path / "raw/baseline"
    base.mkdir(parents=True)
    (base / source.name).write_text("value = 1\n", encoding="utf-8")
    data = tmp_path / ".coverage.fixture"
    tracer = Coverage(data_file=str(data), config_file=False)
    tracer.set_option("report:exclude_lines", [])
    tracer.start()
    exec(compile(source.read_text("utf-8"), str(source), "exec"), {})
    tracer.stop()
    tracer.save()
    with monkeypatch.context() as patcher:
        patcher.setattr(p, "ROOT", tmp_path)
        patcher.setattr(p, "RAW", "raw")
        rows = r.modified_coverage({"path": str(data)}, ["module.py"])
        assert rows[0]["statements"] == [1]
        assert rows[0]["missing"] == []
        assert rows[0]["excluded"] == []
        patcher.setattr(
            r.lifecycle,
            "measured",
            lambda *a: {
                "data_reference": {"path": str(data)},
                "complete": True,
                "percent": 100.0,
                "files": [{"path": "module.py", "statements": 1, "missing": [], "excluded": []}],
            },
        )
        test_scope = {
            "commands": [
                {"name": name, "argv": [sys.executable, "-c", "print('actual')"], "deadline_s": 10}
                for name in ("target_module", "owned_tests")
            ],
            "required_check_names": ["owned_tests"],
            "owned_changed_paths": ["module.py"],
            "modified_paths": ["module.py"],
        }
        assert r.execute_validation(tmp_path / "target", test_scope, target_only=True)["checks"][0][
            "passed"
        ]
        validation = r.execute_validation(tmp_path / "verify", test_scope)
        assert validation["checks"][0]["passed"]
        assert r.receipts_pass({"scope": test_scope}, validation) is False
        empty_files = dict(
            validation, owned_statement_coverage={"complete": True, "percent": 100.0, "files": []}
        )
        assert r.receipts_pass({"scope": test_scope}, empty_files) is False
        validation["checks"][0]["argv"] = ["changed"]
        assert r.receipts_pass({"scope": test_scope}, validation) is False

    raw = tmp_path / "raw_main"
    inspected = r.inspect_v789(tmp_path, raw, authenticate_only=True)
    inspected["gate_checks"] = []
    inspected["preconditions"] = []
    monkeypatch.setattr(r, "inspect_v789", lambda *a, **kw: inspected)
    monkeypatch.setattr(r, "execute_validation", lambda *a, **kw: {"checks": []})
    monkeypatch.setattr(
        r,
        "publish_primary",
        lambda path, value, validator: (r.atomic_json(path, value), {"unit": True})[1],
    )
    args = ["--date", r.RUN_DATE, "--root", str(tmp_path), "--raw-dir", str(raw)]
    assert r.main(args) == 0
    assert r.main([*args, "--checks-only"]) == 0
    out_file = tmp_path / "out.json"
    assert r.main([*args, "--output", str(out_file), "--from-work"]) == 0
    assert r.main([*args, "--output", str(out_file), "--cold-replay"]) == 0
    assert r.main(["--date", "19990101"]) == 1
    assert r.main(["--from-work"]) == 1
    with monkeypatch.context() as patcher:
        patcher.setattr(r, "run", lambda *a, **kw: {"passed": True})
        assert r.terminal(out_file, raw / "term")["passed"]
    r.atomic_json(raw / "corpus_scan_summary.json", inspected["corpus_scan"])
    assert r.main(args) == 0
    with monkeypatch.context() as patcher:
        patcher.setattr(p, "ROOT", tmp_path)
        assert r.main(["--date", r.RUN_DATE]) == 0
    times = iter([0.0, 4801.0])
    monkeypatch.setattr(r.time, "monotonic", lambda: next(times))
    assert r.main(args) == 1
