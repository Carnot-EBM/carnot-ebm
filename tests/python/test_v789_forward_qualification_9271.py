"""REQ-REPORT-9271-V789 / SCENARIO-REPORT-9271-V789: complete causally corrected V789 validation union."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v789_forward_qualification as q


def test_req_9271_compactness_guard_and_sidecar_integrity(tmp_path: Path) -> None:
    """REQ-REPORT-9271-V789 / SCENARIO-REPORT-9271-V789-CONTROLS: primary JSON < 10 MB and hashed sidecars verify."""
    q.compact_guard({"ok": True})
    with pytest.raises(ValueError, match="primary_size_exceeded"):
        q.compact_guard({"bloat": "x" * q.MAX_PRIMARY_BYTES})

    raw = tmp_path / "raw"
    sample_nodes = [
        {
            "unit_id": "tests/python/test_a.py::test_one",
            "node_id": "tests/python/test_a.py::test_one",
            "module": "tests/python/test_a.py",
            "shard": "shard_a",
            "category": "original_621",
            "setup_outcome": "passed",
            "call_outcome": "passed",
            "teardown_outcome": "passed",
            "setup_s": 0.01,
            "call_s": 0.02,
            "teardown_s": 0.01,
            "duration_s": 0.04,
            "status": "completed",
            "numerator": 1,
            "denominator": 1,
            "censor_reason": None,
        }
    ]
    sidecar = q.write_node_sidecar(raw, sample_nodes, [{"shard": "shard_a", "completed": 1}])
    assert Path(sidecar["path"]).is_file()
    assert sidecar["row_count"] == 1
    assert q.verify_node_sidecar(sidecar)
    tampered = dict(sidecar, row_count=2)
    with pytest.raises(ValueError, match="node_sidecar_count_mismatch"):
        q.verify_node_sidecar(tampered)
    Path(sidecar["path"]).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="node_sidecar_hash_mismatch"):
        q.verify_node_sidecar(sidecar)


def test_req_9271_gate_operands_and_upstream_authentication(tmp_path: Path) -> None:
    """REQ-REPORT-9271-V789 / SCENARIO-REPORT-9271-V789: distinguish missing_file, missing_field, and observed 0."""
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    p9270 = root / q.UPSTREAM_PINS["exp9270-corpus-prefilter"][0]
    p9270.write_text(json.dumps({"corpus_scan_ready_score": 0}), encoding="utf-8")
    inspected = q.inspect_v789(root, tmp_path / "raw")
    by_check = {row["check"]: row for row in inspected["gate_checks"]}
    assert by_check["exp9270_corpus_scan_ready_score"]["observed"] == 0
    assert by_check["exp9270_corpus_scan_ready_score"]["passed"] is False
    assert by_check["exp9270_required_checks_passed"]["observed"] == "missing_field"
    assert by_check["exp9269_sha256"]["observed"] == "missing_file"

    live = q.inspect_v789(q.ROOT, tmp_path / "live")
    assert all(row["passed"] for row in live["gate_checks"]), [
        row for row in live["gate_checks"] if not row["passed"]
    ]
    assert all(row["available"] for row in live["preconditions"])
    assert len(live["obligation_manifest"]["original_node_ids"]) == 621
    assert len(live["obligation_manifest"]["v779_addition_ids"]) == 53


def test_req_9271_frozen_obligation_union_and_shard_partition(tmp_path: Path) -> None:
    """REQ-REPORT-9271-V789 / SCENARIO-REPORT-9271-V789: disjoint whole-module shards and per-shard coverage."""
    live = q.inspect_v789(q.ROOT, tmp_path / "live")
    scope = q.build_scope(q.ROOT, tmp_path / "raw", live["obligation_manifest"])
    shards = scope["shards"]
    seen_files: list[str] = []
    cov_files: list[str] = []
    for shard in shards:
        seen_files.extend(shard["test_files"])
        cov_files.append(shard["coverage_shard_path"])
        assert shard["coverage_shard_path"].endswith(f".coverage.{shard['name']}")
    assert len(seen_files) == len(set(seen_files)), "Each module must belong to at most one shard"
    assert len(cov_files) == len(set(cov_files))
    required_files = {
        "tests/python/test_causal_validation_9181.py",
        "tests/python/test_v782_contract_9180.py",
        "tests/python/test_v782_capstone_9193.py",
        "tests/python/test_v783_contract_9194.py",
        "tests/python/test_v783_capstone_9207.py",
        "tests/python/test_v784_contract_9208.py",
        "tests/python/test_v784_capstone_9221.py",
        "tests/python/test_label_ambiguity_9211.py",
        "tests/python/test_v785_contract_9222.py",
        "tests/python/test_selective_deferral_9225.py",
        "tests/python/test_v786_capstone_9240.py",
        "tests/python/test_v787_contract_9241.py",
        "tests/python/test_conformal_risk_9244.py",
        "tests/python/test_v788_contract_9255.py",
        "tests/python/test_v788_corpus_prefilter_9256.py",
        "tests/python/test_victris_energy_bot_9258.py",
        "tests/python/test_v788_capstone_9268.py",
        "tests/python/test_v789_contract_9269.py",
        "tests/python/test_v789_corpus_prefilter_9270.py",
        "tests/python/test_v789_forward_qualification_9271.py",
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_restricted_decision_audit_8210.py",
    }
    assert required_files.issubset(set(seen_files))
    for tail in ("ruff_check", "ruff_format", "strict_mypy", "spec_coverage"):
        assert tail in scope["required_check_names"]


def test_req_9271_node_phase_reduction_and_missing_duplicate_failed_censored_controls(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-9271-V789 / SCENARIO-REPORT-9271-V789-CONTROLS: phase reduction rejects missing, duplicate, failed, censored nodes."""
    n1 = "tests/python/test_a.py::test_1"
    n2 = "tests/python/test_b.py::test_2"
    manifest = {
        "original_node_ids": [n1],
        "v779_addition_ids": [n2],
        "affected_v780_v789_node_ids": [],
        "all_expected_node_ids": [n1, n2],
        "node_categories": {n1: "original_621", n2: "v779_addition_53"},
    }
    shards = [
        {
            "name": "shard_a",
            "test_files": ["tests/python/test_a.py"],
            "event_log": str(tmp_path / "events_a.jsonl"),
        },
        {
            "name": "shard_b",
            "test_files": ["tests/python/test_b.py"],
            "event_log": str(tmp_path / "events_b.jsonl"),
        },
    ]

    def write_events(path: str, events: list[dict[str, object]]) -> None:
        Path(path).write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")

    ok_n1 = [
        {"event": "start", "nodeid": n1, "ts_ns": 10},
        {
            "event": "phase",
            "nodeid": n1,
            "when": "setup",
            "outcome": "passed",
            "duration_s": 0.01,
            "ts_ns": 20,
        },
        {
            "event": "phase",
            "nodeid": n1,
            "when": "call",
            "outcome": "passed",
            "duration_s": 0.02,
            "ts_ns": 30,
        },
        {
            "event": "phase",
            "nodeid": n1,
            "when": "teardown",
            "outcome": "passed",
            "duration_s": 0.01,
            "ts_ns": 40,
        },
    ]
    ok_n2 = [
        {"event": "start", "nodeid": n2, "ts_ns": 50},
        {
            "event": "phase",
            "nodeid": n2,
            "when": "setup",
            "outcome": "passed",
            "duration_s": 0.01,
            "ts_ns": 60,
        },
        {
            "event": "phase",
            "nodeid": n2,
            "when": "call",
            "outcome": "passed",
            "duration_s": 0.03,
            "ts_ns": 70,
        },
        {
            "event": "phase",
            "nodeid": n2,
            "when": "teardown",
            "outcome": "passed",
            "duration_s": 0.01,
            "ts_ns": 80,
        },
    ]
    write_events(shards[0]["event_log"], ok_n1)
    write_events(shards[1]["event_log"], ok_n2)
    red_ok = q.reduce_node_completion(manifest, shards)
    assert red_ok["union_passed"] is True
    assert red_ok["completed_count"] == 2
    assert red_ok["missing_count"] == 0
    assert red_ok["duplicate_count"] == 0

    # Control 1: missing node (n2 absent)
    write_events(shards[1]["event_log"], [])
    red_missing = q.reduce_node_completion(manifest, shards)
    assert red_missing["union_passed"] is False
    assert red_missing["missing_count"] == 1

    # Control 2: duplicate node (n1 executed twice)
    write_events(shards[0]["event_log"], ok_n1 + ok_n1)
    write_events(shards[1]["event_log"], ok_n2)
    red_dup = q.reduce_node_completion(manifest, shards)
    assert red_dup["union_passed"] is False
    assert red_dup["duplicate_count"] == 1

    # Control 3: failed phase
    failed_n2 = deepcopy(ok_n2)
    failed_n2[2]["outcome"] = "failed"
    write_events(shards[0]["event_log"], ok_n1)
    write_events(shards[1]["event_log"], failed_n2)
    red_fail = q.reduce_node_completion(manifest, shards)
    assert red_fail["union_passed"] is False
    assert red_fail["failed_count"] == 1

    # Control 4: censored / incomplete phase (missing teardown)
    write_events(shards[1]["event_log"], ok_n2[:3])
    red_cens = q.reduce_node_completion(manifest, shards)
    assert red_cens["union_passed"] is False
    assert red_cens["censored_count"] == 1


def test_req_9271_verify_and_publish_controls_tamper_and_cli(tmp_path: Path) -> None:
    """REQ-REPORT-9271-V789 / SCENARIO-REPORT-9271-V789-CONTROLS: verify, publish, refusal, replay, and private CLI."""
    outside = tmp_path / "outside"
    outside.mkdir()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    help_proc = subprocess.run(
        [sys.executable, str(q.SCRIPT), "--help"],
        cwd=outside,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_proc.returncode == 0
    assert "9271" in help_proc.stdout

    # Test real single-command execution and real coverage combiner helper on tiny synthetic data
    cmd_row = q.run_checked_command(
        "tiny_echo", [sys.executable, "-c", "print('ok')"], tmp_path / "cmd_raw"
    )
    assert cmd_row["passed"] is True
    assert cmd_row["returncode"] == 0

    live = q.inspect_v789(q.ROOT, tmp_path / "live_inspect")
    manifest = live["obligation_manifest"]

    def fake_verify(root: Path, raw: Path) -> dict[str, object]:
        return q.simulate_verification_bundle(root, raw, manifest, all_pass=True)

    out = tmp_path / "pass_dir" / "experiment_9271_out.json"
    raw = tmp_path / "raw"
    payload = q.publish(out, raw, root=q.ROOT, verify_fn=fake_verify, run_linters=True)
    assert payload["forward_validation_ready_score"] == 1
    assert payload["honest_verdict"] == "complete_null_v789_forward_qualification"
    assert payload["verdict_class"] == "null"
    assert payload["required_checks_passed"] is True
    assert payload["current_llm_invocations"] == 0
    assert payload["generated_tokens"] == 0
    assert q.replay_from_payload(payload)["forward_validation_ready_score"] == 1

    # Failing verification sets ready_score=0 and refuses require_ready publication
    def fake_verify_fail(root: Path, raw: Path) -> dict[str, object]:
        return q.simulate_verification_bundle(root, raw, manifest, all_pass=False)

    with pytest.raises(ValueError, match="forward_qualification_check_failed"):
        q.publish(
            tmp_path / "fail_dir" / "experiment_9271_fail.json",
            tmp_path / "raw_fail",
            root=q.ROOT,
            verify_fn=fake_verify_fail,
            require_ready=True,
            run_linters=False,
        )
    failed_payload = q.publish(
        tmp_path / "fail_ok_dir" / "experiment_9271_fail_allowed.json",
        tmp_path / "raw_fail2",
        root=q.ROOT,
        verify_fn=fake_verify_fail,
        require_ready=False,
        run_linters=False,
    )
    assert failed_payload["forward_validation_ready_score"] == 0

    # Blocked upstream gate produces complete_blocked_v789_forward_qualification and blocked_no_run
    bad_root = tmp_path / "bad_root"
    (bad_root / "results").mkdir(parents=True)
    blocked_payload = q.publish(
        tmp_path / "blocked_dir" / "experiment_9271_blocked.json",
        tmp_path / "raw_blocked",
        root=bad_root,
        verify_fn=fake_verify,
        require_ready=False,
        run_linters=False,
    )
    assert blocked_payload["verdict_class"] == "blocked"
    assert blocked_payload["honest_verdict"] == "complete_blocked_v789_forward_qualification"
    assert blocked_payload["inference_substrate_class"] == "blocked_no_run"
    assert blocked_payload["forward_validation_ready_score"] == 0

    # Tampered receipt fails publish validation
    with pytest.raises(ValueError, match="receipt_validation_failed"):
        q.publish(
            tmp_path / "tamper_dir" / "experiment_9271_tamper.json",
            tmp_path / "raw_tamper",
            root=q.ROOT,
            verify_fn=fake_verify,
            tamper_receipt=True,
            run_linters=False,
        )

    # CLI main entry point with injected verify_fn
    rc = q.main(
        [
            "--date",
            "20261121",
            "--out",
            str(tmp_path / "cli_dir" / "experiment_9271_cli.json"),
            "--raw-dir",
            str(tmp_path / "cli_raw"),
        ],
        verify_fn=fake_verify,
        run_linters=False,
    )
    assert rc == 0


def test_req_9271_verify_orchestration_and_coverage_combiner(tmp_path: Path) -> None:
    """REQ-REPORT-9271-V789 / SCENARIO-REPORT-9271-V789: verify() executes shard schedule, tail checks, and coverage combiner."""
    live = q.inspect_v789(q.ROOT, tmp_path / "live_inspect")
    manifest = live["obligation_manifest"]
    raw = tmp_path / "verify_raw"

    def fast_shard_runner(
        name: str, cmd: list[str], raw_dir: Path, **kwargs: object
    ) -> dict[str, object]:
        for token in cmd:
            if token.startswith("CARNOT_9125_NODE_LOG="):
                log_path = Path(token.split("=", 1)[1])
                log_path.parent.mkdir(parents=True, exist_ok=True)
                shard_name = log_path.stem.removeprefix("events_")
                lines: list[str] = []
                for nid in manifest["shard_expected_nodes"].get(shard_name, []):
                    lines.extend(
                        [
                            json.dumps({"event": "start", "nodeid": nid, "ts_ns": 1}),
                            json.dumps(
                                {
                                    "event": "phase",
                                    "nodeid": nid,
                                    "when": "setup",
                                    "outcome": "passed",
                                    "duration_s": 0.001,
                                    "ts_ns": 2,
                                }
                            ),
                            json.dumps(
                                {
                                    "event": "phase",
                                    "nodeid": nid,
                                    "when": "call",
                                    "outcome": "passed",
                                    "duration_s": 0.002,
                                    "ts_ns": 3,
                                }
                            ),
                            json.dumps(
                                {
                                    "event": "phase",
                                    "nodeid": nid,
                                    "when": "teardown",
                                    "outcome": "passed",
                                    "duration_s": 0.001,
                                    "ts_ns": 4,
                                }
                            ),
                        ]
                    )
                log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            elif token.startswith("COVERAGE_FILE="):
                cov_path = Path(token.split("=", 1)[1])
                cov_path.parent.mkdir(parents=True, exist_ok=True)
                cov_path.write_bytes(b"placeholder")
        return {
            "name": name,
            "cmd": cmd,
            "returncode": 0,
            "passed": True,
            "duration_s": 0.01,
            "log": str(raw_dir / f"{name}.log"),
            "stdout_tail": "ok",
            "stderr_tail": "",
        }

    def fast_cov_combiner(raw_dir: Path, shards: list[dict[str, object]]) -> dict[str, object]:
        return q.parse_coverage_report(
            {
                "files": {
                    rel: {
                        "summary": {
                            "num_statements": 100,
                            "covered_lines": 100,
                            "percent_covered": 100.0,
                        },
                        "missing_lines": [],
                        "excluded_lines": [],
                    }
                    for rel in q.OWNED
                }
            },
            raw_dir / "coverage.json",
            [str(s["coverage_shard_path"]) for s in shards],
        )

    bundle = q.verify(
        q.ROOT,
        raw,
        obligation_manifest=manifest,
        run_fn=fast_shard_runner,
        cov_fn=fast_cov_combiner,
    )
    assert bundle["required_checks_passed"] is True
    assert bundle["coverage"]["passed"] is True
    assert bundle["node_reduction"]["union_passed"] is True

    def fake_cov_subproc(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "json" in cmd:
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_text(
                json.dumps(
                    {
                        "files": {
                            rel: {
                                "summary": {
                                    "num_statements": 10,
                                    "covered_lines": 10,
                                    "percent_covered": 100.0,
                                },
                                "missing_lines": [],
                                "excluded_lines": [],
                            }
                            for rel in q.OWNED
                        }
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    combined = q.combine_owned_coverage(raw, bundle["shards"], run_cmd=fake_cov_subproc)
    assert combined["passed"] is True

    bad_cov = q.parse_coverage_report({"files": {}}, raw / "bad_cov.json", [])
    assert bad_cov["passed"] is False


def test_req_9271_failed_gate_bypasses_execution(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9271-V789-TERMINAL: absent authority stops dependent work."""

    def forbidden(root: Path, raw: Path) -> dict[str, object]:
        pytest.fail("dependent execution after failed authentication")

    payload = q.publish(
        tmp_path / "experiment_9271_blocked.json",
        tmp_path / "raw",
        root=tmp_path / "missing",
        verify_fn=forbidden,
        require_ready=False,
        run_linters=False,
    )
    assert payload["honest_verdict"] == "complete_blocked_v789_forward_qualification"
    assert payload["forward_validation_ready_score"] == 0
    assert payload["attempted_count"] == 0
    assert payload["validation_receipts"]["commands"] == []


def test_req_9271_sidecars_memory_and_frozen_revision(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9271-V789-TERMINAL: sidecars and resources are real operands."""
    inspected = q.inspect_v789(q.ROOT, tmp_path / "raw")
    checks = {row["check"]: row for row in inspected["gate_checks"]}
    assert checks["exp9270_corpus_manifest_reference_sha256"]["passed"]
    assert checks["exp9270_scan_rows_sidecar_reference_sha256"]["passed"]
    assert checks["exp9269_source_ready_score"]["passed"]
    assert checks["exp9269_static_methods_ready_score"]["passed"]
    assert checks["exp9269_online_methods_ready_score"]["passed"]
    resources = {row["resource"]: row for row in inspected["preconditions"]}
    assert resources["memory_available_bytes"]["receipt"]["available_bytes"] > 0
    assert inspected["code_config_hashes"] == q._code_config_hashes()


def test_req_9271_all_historical_modules_are_scheduled(tmp_path: Path) -> None:
    """REQ-REPORT-9271-V789: old unisolated modules remain owed obligations."""
    manifest = q._build_obligation_manifest()
    scope = q.build_scope(q.ROOT, tmp_path, manifest)
    scheduled = [f for s in scope["shards"] for f in s["test_files"]]
    for name in (
        "test_v780_contract_9152.py",
        "test_v780_capstone_9165.py",
        "test_v781_contract_9166.py",
        "test_v781_capstone_9179.py",
        "test_v781_queue_binding.py",
    ):
        assert f"tests/python/{name}" in scheduled
    assert {n.split("::")[0] for n in manifest["all_expected_node_ids"]} <= set(scheduled)
    assert len(scheduled) == len(set(scheduled))


def test_req_9271_normal_deadline_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-9271-V789-TERMINAL: the normal path receives the frozen start."""
    seen = []

    def normal(root: Path, raw: Path, **kwargs: object) -> dict[str, object]:
        seen.append(kwargs)
        return q.simulate_verification_bundle(root, raw, kwargs["obligation_manifest"])

    monkeypatch.setattr(q, "verify", normal)
    payload = q.publish(
        tmp_path / "experiment_9271_clock.json",
        tmp_path / "raw",
        verify_fn=normal,
        run_linters=False,
    )
    assert seen[0]["started_monotonic"] > 0
    assert payload["forward_validation_ready_score"] == 1


def test_req_9271_unavailable_import_and_revision_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-9271-V789-TERMINAL: resource absence and changed code cannot qualify."""
    original_import = q.importlib.import_module

    def unavailable(name: str) -> object:
        if name == "ruff":
            raise ImportError("fixture missing tool")
        return original_import(name)

    with monkeypatch.context() as patch:
        patch.setattr(q.importlib, "import_module", unavailable)
        inspected = q.inspect_v789(q.ROOT, tmp_path / "missing_tool")
    assert not inspected["preconditions"][0]["available"]
    assert "fixture missing tool" in inspected["preconditions"][0]["receipt"]["ruff"]
    inspected = q.inspect_v789(q.ROOT, tmp_path / "live")
    fixture = q.simulate_verification_bundle(
        q.ROOT, tmp_path / "fixture", inspected["obligation_manifest"]
    )
    receipt = q.build_current_work_receipt(
        run_id="fixture",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="no_model_load",
        inference_substrate_details={},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=1,
        ended_monotonic_ns=2,
    )
    inspected["code_config_hashes"]["combined_revision_hash"] = "changed"
    payload = q.build_payload(inspected, fixture, receipt, 0.01)
    assert not payload["required_checks_passed"]
    assert payload["forward_validation_ready_score"] == 0


def test_req_9271_real_empty_coverage_is_not_ready(tmp_path: Path) -> None:
    """REQ-REPORT-9271-V789: the supervised coverage CLI cannot invent missing data."""
    q.build_scope(
        q.ROOT,
        tmp_path,
        {
            "original_node_ids": [],
            "v779_addition_ids": [],
            "affected_v780_v789_node_ids": [],
            "all_expected_node_ids": [],
        },
    )
    coverage = q.combine_owned_coverage(tmp_path, [])
    assert coverage["passed"] is False


def test_req_9271_authenticated_replay_reopens_logs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9271-V789-TERMINAL: changed streams invalidate current evidence."""
    inspected = q.inspect_v789(q.ROOT, tmp_path / "inspection")
    raw = tmp_path / "fixture"
    manifest = inspected["obligation_manifest"]
    bundle = q.simulate_verification_bundle(q.ROOT, raw, manifest)

    def bind(path: Path, value: dict[str, object]) -> dict[str, str]:
        q.atomic_json(path, value)
        return {"path": str(path), "sha256": q.sha256_file(path)}

    bundle["obligation_manifest_reference"] = bind(raw / "manifest.json", manifest)
    bundle["scope_reference"] = bind(raw / "scope.json", bundle["scope"])
    bundle["event_references"] = [
        {"path": s["event_log"], "sha256": q.sha256_file(Path(s["event_log"]))}
        for s in bundle["shards"]
    ]
    for command in bundle["checks"]:
        for stream in ("stdout", "stderr"):
            path = raw / f"{command['name']}.{stream}"
            path.write_text("explicit replay fixture\n", encoding="utf-8")
            command[f"{stream}_path"] = str(path)
            command[f"{stream}_sha256"] = q.sha256_file(path)
        command["receipt_reference"] = bind(raw / f"{command['name']}.receipt.json", command)
    receipt = q.build_current_work_receipt(
        run_id="fixture",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="no_model_load",
        inference_substrate_details={},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=1,
        ended_monotonic_ns=2,
    )
    payload = q.build_payload(inspected, bundle, receipt, 0.01)
    assert q.replay_from_payload(payload)["passed"]
    Path(bundle["checks"][0]["stdout_path"]).write_text("changed\n", encoding="utf-8")
    assert not q.replay_from_payload(payload)["passed"]
