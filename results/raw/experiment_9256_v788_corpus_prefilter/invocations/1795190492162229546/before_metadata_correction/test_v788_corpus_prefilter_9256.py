from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v788_corpus_prefilter as p
from carnot.reporting import v788_corpus_prefilter_runner as r


def test_req_9256_reader_reaches_eof_and_hashes_matched_bytes(tmp_path: Path) -> None:
    # REQ-REPORT-9256-V788: a positive prefix cannot hide a changed suffix.
    target = tmp_path / "matched.json"
    raw = json.dumps(
        {"inference_substrate": p.load_av().ARC_LIVE_AGENT_NO_LLM_SUBSTRATE, "padding": "x" * 1000}
    ).encode()
    target.write_bytes(raw)
    result = p.stream_prefilter_path(
        target, p.derive_sentinels(p.load_av())["sentinels"], chunk_size=128
    )
    assert result["bytes_read"] == len(raw)
    assert result["sha256"] == p.digest_bytes(raw)


def test_req_9256_manifest_accounts_for_every_path_and_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SCENARIO-REPORT-9256-V788-CONTROLS: negative and newly added paths count too.
    results = tmp_path / "results"
    results.mkdir()
    for i in range(20):
        (results / f"{i:02}.json").write_text('{"old":0}')
    original = p.prefilter_evaluate_path

    def mutate(path: Path, av: object, **kwargs: object) -> dict:
        answer = original(path, av, **kwargs)
        if path.name == "19.json":
            (results / "00.json").write_text('{"new":1}')
            (results / "added.json").write_text("{}")
        return answer

    monkeypatch.setattr(p, "prefilter_evaluate_path", mutate)
    scan = p.scan_corpus(results, p.load_av())
    assert len(scan["manifest_rows"]) == 20
    assert len(scan["scan_rows"]) == 20
    assert scan["changed_files"] >= 1
    assert scan["added_paths"] == ["added.json"]
    assert scan["candidate_bytes_stable"] is False


def test_req_9256_failed_gate_does_not_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-9256-V788: missing authority ends before any corpus work.
    root = tmp_path / "absent"
    (root / "results").mkdir(parents=True)
    monkeypatch.setattr(p, "scan_corpus", lambda *a, **kw: pytest.fail("scanned after failed gate"))
    work = p.inspect(root, tmp_path / "raw")
    assert any(not row["passed"] for row in work["gate_checks"])
    assert work["controls"]["fixture_count"] == 0


def test_req_9256_every_chunk_boundary_for_sentinels_and_unicode(tmp_path: Path) -> None:
    # SCENARIO-REPORT-9256-V788-CONTROLS: each possible split retains the original answer.
    av = p.load_av()
    sentinels = p.derive_sentinels(av)["sentinels"]
    target = tmp_path / "boundary.json"
    prefix = b'{"pad":"' + b'","inference_substrate":"'
    for _, value, needle in sentinels:
        for split in range(1, len(needle)):
            padding = 128 - split - len(prefix)
            payload = b'{"pad":"' + b"x" * padding + b'","inference_substrate":"'
            assert len(payload) + split == 128
            target.write_bytes(payload + value.encode() + b'","duration_s":0.001}')
            assert p.prefilter_evaluate_path(target, av, chunk_size=128)["is_offender"]
    prefix = b'{"inference_substrate":"live'
    # The slash is byte 127 and the u is byte 128.
    target.write_bytes(
        b" " * (127 - len(prefix))
        + prefix
        + b'\\u005fagent_environment_interaction","duration_s":0.001}'
    )
    assert p.prefilter_evaluate_path(target, av, chunk_size=128)["is_offender"]


def test_five_sentinels_derived_from_unchanged_classifier_constants() -> None:
    # REQ-REPORT-9256-V788 / SCENARIO-REPORT-9256-V788
    av = p.load_av()
    info = p.derive_sentinels(av)
    assert len(info["sentinels"]) == 5
    names = [item[0] for item in info["sentinels"]]
    assert names == list(p.SENTINEL_CONSTANTS)
    assert info["max_needle_len"] == 73
    assert info["min_overlap_bytes"] == 72
    assert info["fast_common_subneedle"] == b"live_"
    for const_name, text, raw_bytes in info["sentinels"]:
        assert getattr(av, const_name) == text
        assert text.encode("ascii") == raw_bytes
        assert b"live_" in raw_bytes
    assert p.stratum_for_size(512) == "le_4kb"
    assert p.stratum_for_size(8192) == "4kb_to_64kb"
    assert p.stratum_for_size(131072) == "64kb_to_1mb"
    assert p.stratum_for_size(2097152) == "1mb_to_16mb"
    assert p.stratum_for_size(33554432) == "gt_16mb"


def test_semantic_parity_and_mutation_controls(tmp_path: Path) -> None:
    # REQ-REPORT-9256-V788 / SCENARIO-REPORT-9256-V788 / SCENARIO-REPORT-9256-V788-CONTROLS
    av = p.load_av()
    report = p.run_parity_and_mutation_controls(av, tmp_path / "controls")
    assert report["semantic_parity_exact"] is True
    assert report["parity_discrepancies"] == 0
    assert report["fixture_count"] >= 28
    assert report["all_mutations_detected"] is True
    assert len(report["mutation_controls"]) == 6
    for row in report["mutation_controls"]:
        assert row["mutation_detected"] is True
        assert row["missed_positive_count"] >= 1


def test_cross_chunk_boundaries_and_corpus_scan_summary(tmp_path: Path) -> None:
    # REQ-REPORT-9256-V788 / SCENARIO-REPORT-9256-V788 / SCENARIO-REPORT-9256-V788-CONTROLS
    av = p.load_av()
    results_dir = tmp_path / "results"
    nested = results_dir / "raw" / "sub"
    nested.mkdir(parents=True)
    chunk_size = 128
    pad = "x" * (chunk_size - 40)
    (results_dir / "a_pos_above.json").write_text(
        json.dumps(
            {
                "pad": pad,
                "inference_substrate": av.ARC_LIVE_AGENT_ENV_INTERACTION_SUBSTRATE,
                "duration_s": 0.05,
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "b_pos_below.json").write_text(
        json.dumps(
            {
                "pad": pad,
                "inference_substrate": {
                    "value": f"{av.ARC_SUPERVISOR_AB_NO_LLM_SUBSTRATE} -- note",
                    "principle": "wrapped",
                },
                "duration_s": 0.001,
            }
        ),
        encoding="utf-8",
    )
    (nested / "c_unicode_pos.json").write_bytes(
        (
            '{"pad":"'
            + ("y" * (chunk_size - 11))
            + '","inference\\u005fsubstrate":"live\\u005fagent_environment_interaction","duration_s":0.02}'
        ).encode("utf-8")
    )
    (nested / "d_neg_eof.json").write_text(
        json.dumps({"pad": "z" * 400, "inference_substrate": "no_model_load", "duration_s": 0.5}),
        encoding="utf-8",
    )
    (nested / "e_malformed.json").write_bytes(b"\xff\xfe{not valid json")
    (nested / "e_malformed_with_u.json").write_bytes(b'{"a":"\\u005f",\xff\xfe}')
    (nested / "f_nondict.json").write_text(
        json.dumps([{"inference_substrate": av.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE}]),
        encoding="utf-8",
    )
    out_file = tmp_path / "scan_out.json"
    summary = p.scan_corpus(
        results_dir,
        av,
        chunk_size=chunk_size,
        progress_every=2,
        scan_output=out_file,
    )
    assert out_file.exists()
    assert summary["total_files"] == 7
    assert summary["visited_files"] == 7
    assert summary["unfinished_files"] == 0
    assert summary["checked"] == 3
    assert len(summary["offenders"]) == 1
    assert summary["offenders"][0][0] == "b_pos_below.json"
    assert summary["safely_negative_eof_count"] == 2
    assert summary["malformed_count"] == 1
    assert summary["nondict_count"] == 1
    assert summary["candidate_bytes_stable"] is True
    assert summary["semantic_parity_exact"] is True

    costs = p.measure_stratum_costs(tmp_path, summary["manifest_rows"], av, per_stratum_limit=4)
    assert costs["full_corpus_speedup_claimed"] is False
    assert len(costs["strata"]) == 5

    expired = p.scan_corpus(
        results_dir,
        av,
        chunk_size=chunk_size,
        progress_every=10,
        stop_monotonic=0.0,
    )
    assert expired["unfinished_files"] == 7


def _child_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Use the real CLI outside the checkout so imports cannot depend on PYTHONPATH."""
    script = p.ROOT / "scripts/experiments/experiment_9256_v788_corpus_prefilter.py"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    argv = [sys.executable, "-u", str(script), *args]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, check=False)


def test_req_9256_gate_operands_and_historical_receipts(tmp_path: Path) -> None:
    # REQ-REPORT-9256-V788: byte pins do not replace named scalar gate operands.
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    path = root / p.UPSTREAM_PINS["exp9255-contract"][0]
    path.write_text(json.dumps({"contract_ready_score": 0}))
    work = p.inspect(root, tmp_path / "raw", authenticate_only=True)
    by_check = {row["check"]: row for row in work["gate_checks"]}
    assert by_check["exp9255_contract_ready_score"]["observed"] == 0
    assert by_check["exp9255_replay_authority_ready_score"]["observed"] == "missing_field"
    assert by_check["exp9241_sha256"]["observed"] == "missing_file"
    live = p.inspect(p.ROOT, tmp_path / "live", authenticate_only=True)
    assert all(row["passed"] for row in live["gate_checks"])
    assert live["exp9181_timeout_diagnostics"]["direct_consumers_exit_code"] == -15


def test_req_9256_private_cli_publication_refusal_and_cold_replay(tmp_path: Path) -> None:
    # SCENARIO-REPORT-9256-V788-CONTROLS: real refusal artifacts retain readiness zero.
    root = tmp_path / "private"
    outside = tmp_path / "outside"
    outside.mkdir()
    raw = root / "raw"
    out = root / p.OUT
    args = ("--date", p.RUN_DATE, "--root", str(root), "--raw-dir", str(raw), "--output", str(out))
    published = _child_cli(outside, *args)
    assert published.returncode == 0, published.stdout + published.stderr
    value = json.loads(out.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["inference_substrate_class"] == "blocked_no_run"
    assert value["corpus_scan_ready_score"] == 0
    assert _child_cli(outside, *args, "--cold-replay").returncode == 0
    assert _child_cli(outside, *args, "--from-work").returncode == 0
    assert _child_cli(outside, *args, "--checks-only").returncode == 1
    assert _child_cli(outside, "--date", "19990101").returncode == 1
    assert _child_cli(outside, "--from-work").returncode == 1
    pristine = out.read_bytes()
    value["corpus_scan_ready_score"] = 1
    out.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum_mismatch"):
        r.replay(out)
    value["reproducibility_checksum"] = r.checksum(value)
    out.write_text(json.dumps(value))
    assert _child_cli(outside, *args, "--cold-replay").returncode == 1
    out.write_bytes(pristine)
    for field, error in (("work_reference", "work_reference_hash_mismatch"),):
        value = json.loads(pristine)
        value[field]["sha256"] = "sha256:bad"
        value["reproducibility_checksum"] = r.checksum(value)
        out.write_text(json.dumps(value))
        with pytest.raises(ValueError, match=error):
            r.replay(out)
    out.write_bytes(pristine)


def test_req_9256_actual_receipt_validation_and_statement_coverage(tmp_path: Path) -> None:
    # REQ-REPORT-9256-V788: a fabricated green summary has no executed authority.
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
        "scope": {"required_check_names": ["real_unit_child"], "owned_changed_paths": list(p.OWNED)}
    }
    assert r.receipts_pass(small, {"checks": [receipt]}) is False
    Path(receipt["stdout_path"]).chmod(0o600)
    Path(receipt["stdout_path"]).write_text("tampered")
    assert r.receipts_pass(small, {"checks": [receipt]}) is False


def test_req_9256_bounded_stream_and_missing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SCENARIO-REPORT-9256-V788-CONTROLS: deadlines and disappearing files never certify a walk.
    av = p.load_av()
    path = tmp_path / "input.json"
    path.write_bytes(b"x" * 400)
    moments = iter([0, 61, 62, 63, 64])
    with monkeypatch.context() as patcher:
        patcher.setattr(p.time, "monotonic", lambda: next(moments))
        answer = p.stream_prefilter_path(path, (("other", "other", b"other"),), chunk_size=128)
        assert answer["bytes_read"] == 400
    with monkeypatch.context() as patcher:
        patcher.setenv("CARNOT_9256_DEADLINE", "0")
        with pytest.raises(TimeoutError, match="stream_deadline"):
            p.stream_prefilter_path(path, p.derive_sentinels(av)["sentinels"])
    results = tmp_path / "results"
    results.mkdir()
    (results / "a.json").write_text("{}")
    (results / "b.json").write_text("{}")
    original = p.prefilter_evaluate_path

    def remove(path: Path, *args: object, **kwargs: object) -> dict:
        answer = original(path, *args, **kwargs)
        if path.name == "b.json":
            (results / "a.json").unlink()
        return answer

    with monkeypatch.context() as patcher:
        patcher.setattr(p, "prefilter_evaluate_path", remove)
        scan = p.scan_corpus(results, av)
        assert scan["removed_paths"] == ["a.json"]
        assert scan["candidate_bytes_stable"] is False
    with monkeypatch.context() as patcher:
        patcher.setattr(
            p, "prefilter_evaluate_path", lambda *a, **kw: (_ for _ in ()).throw(TimeoutError())
        )
        scan = p.scan_corpus(results, av)
        assert scan["unfinished_files"] == 1


def test_req_9256_reduction_and_current_work_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-9256-V788: executed fixtures cannot replace failed validation authority.
    av = p.load_av()
    raw = tmp_path / "raw"
    inspected = p.inspect(tmp_path, raw, authenticate_only=True)
    results = tmp_path / "results"
    results.mkdir()
    (results / "positive.json").write_text(
        json.dumps({"inference_substrate": av.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE, "duration_s": 0.02})
    )
    inspected["corpus_scan"] = p.scan_corpus(results, av)
    inspected["controls"] = p.run_parity_and_mutation_controls(av, tmp_path / "controls")
    inspected["stratum_costs"] = p.measure_stratum_costs(
        tmp_path, inspected["corpus_scan"]["manifest_rows"], av
    )
    work = r.measure(tmp_path, raw, p.RUN_DATE, inspection=inspected)
    value = r.build(tmp_path, raw, work, {}, p.RUN_DATE)
    assert value["verdict_class"] == "blocked"
    assert all(row["status"] == "censored" for row in value["rows"])
    inspected["gate_checks"] = []
    inspected["preconditions"] = []
    r.atomic_json(raw / "work.json", work)
    value = r.build(tmp_path, raw, work, {}, p.RUN_DATE)
    assert value["verdict_class"] == "disqualified"
    assert value["corpus_scan_ready_score"] == 0
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


def test_req_9256_validation_executor_and_added_statement_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-REPORT-9256-V788: coverage reads actual SQLite statements with zero exclusions.
    from coverage import Coverage

    source = tmp_path / "module.py"
    source.write_text("value = 2\n")
    base = tmp_path / "raw/baseline"
    base.mkdir(parents=True)
    (base / source.name).write_text("value = 1\n")
    data = tmp_path / ".coverage.fixture"
    tracer = Coverage(data_file=str(data), config_file=False)
    tracer.set_option("report:exclude_lines", [])
    tracer.start()
    exec(compile(source.read_text(), str(source), "exec"), {})
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
        scope = {
            "commands": [
                {"name": name, "argv": [sys.executable, "-c", "print('actual')"], "deadline_s": 10}
                for name in ("target_module", "owned_tests")
            ],
            "required_check_names": ["owned_tests"],
            "owned_changed_paths": ["module.py"],
            "modified_paths": ["module.py"],
        }
        assert r.execute_validation(tmp_path / "target", scope, target_only=True)["checks"][0][
            "passed"
        ]
        validation = r.execute_validation(tmp_path / "verify", scope)
        assert validation["checks"][0]["passed"]
        assert r.receipts_pass({"scope": scope}, validation) is False
        empty_files = dict(
            validation, owned_statement_coverage={"complete": True, "percent": 100.0, "files": []}
        )
        assert r.receipts_pass({"scope": scope}, empty_files) is False
        receipt = validation["checks"][0]
        receipt["argv"] = ["changed"]
        assert r.receipts_pass({"scope": scope}, validation) is False


def test_req_9256_inspection_uses_authenticated_scan_and_controls(tmp_path: Path) -> None:
    # REQ-REPORT-9256-V788: only authenticated gates admit the owned reader and controls.
    root = tmp_path / "authority"
    for rel, _ in p.UPSTREAM_PINS.values():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(p.ROOT / rel)
    for rel in ("scripts/adversarial_verify.py", "scripts/verdict_row_consistency_lint.py"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(p.ROOT / rel)
    scan = p.scan_corpus(tmp_path / "empty", p.load_av())
    inspection = p.inspect(root, tmp_path / "inspection", scan_summary=scan)
    assert all(row["passed"] for row in inspection["gate_checks"])
    assert inspection["controls"]["all_mutations_detected"]
    assert inspection["corpus_scan"]["total_files"] == 0


def test_req_9256_main_phases_and_deadline_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # SCENARIO-REPORT-9256-V788-CONTROLS: publication cannot change a failed readiness decision.
    raw = tmp_path / "raw"
    inspected = p.inspect(tmp_path, raw, authenticate_only=True)
    inspected["gate_checks"] = []
    inspected["preconditions"] = []
    monkeypatch.setattr(p, "inspect", lambda *a, **kw: inspected)
    monkeypatch.setattr(r, "execute_validation", lambda *a, **kw: {"checks": []})
    monkeypatch.setattr(
        r,
        "publish_primary",
        lambda path, value, validator: (r.atomic_json(path, value), {"unit": True})[1],
    )
    args = ["--date", p.RUN_DATE, "--root", str(tmp_path), "--raw-dir", str(raw)]
    assert r.main(args) == 0
    assert r.main([*args, "--checks-only"]) == 0
    r.atomic_json(raw / "corpus_scan_summary.json", inspected["corpus_scan"])
    assert r.main(args) == 0
    with monkeypatch.context() as patcher:
        patcher.setattr(p, "ROOT", tmp_path)
        assert r.main(["--date", p.RUN_DATE]) == 0
    times = iter([0.0, 4801.0])
    monkeypatch.setattr(r.time, "monotonic", lambda: next(times))
    assert r.main(args) == 1


def test_req_9256_remaining_historical_consumers(tmp_path: Path) -> None:
    # REQ-REPORT-9256-V788: changed historical hash readers retain all their consumers.
    files = (
        "test_validation_schedule_9153.py",
        "test_v785_contract_9222.py",
        "test_v787_contract_9241.py",
        "test_v788_contract_9255.py",
    )
    argv = [
        "env",
        f"--chdir={tmp_path}",
        "--unset=PYTHONPATH",
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(p.ROOT / "pyproject.toml"),
        "--rootdir",
        str(p.ROOT),
        "-o",
        "addopts=",
        "--no-cov",
        "-n",
        "4",
        "-q",
        *(str(p.ROOT / "tests/python" / name) for name in files),
    ]
    receipt = r.run(
        "remaining_historical_consumers", argv, tmp_path / "receipts", r.time.monotonic() + 450
    )
    assert receipt["passed"], (
        Path(receipt["stdout_path"]).read_text()[-5000:]
        + Path(receipt["stderr_path"]).read_text()[-2000:]
    )
