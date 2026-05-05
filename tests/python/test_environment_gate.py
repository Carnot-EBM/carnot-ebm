import json
from pathlib import Path

import yaml

from carnot.conductor import environment_gate


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_gate_inputs(root: Path) -> None:
    (root / "ops").mkdir()
    (root / "scripts").mkdir()
    (root / "results").mkdir()
    (root / "ops/conductor-log.md").write_text(
        "\n".join(
            [
                "| 2026-05-05 07:25 UTC | Exp 1325 | FAIL | Codex CLI error: [Errno 122] Disk quota exceeded |",
                "| 2026-05-05 07:38 UTC | Exp 1328 | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed, 1 warning in 7.65s |",
                "| 2026-05-05 08:23 UTC | Exp 1336 | SKIP | Pre-tests failing, self-heal failed: 1 failed, 86 passed, 1 warning in 5.02s |",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "results/operational_retro_2026_04_103.json",
        {
            "preflight_suite": {
                "evidence": "ops/conductor-log.md shows repeated .103 SKIP rows with '1 failed, 86 passed, 1 warning' for exp1328-class through retrospective attempts."
            },
            "bottlenecks_identified": [
                "Disk quota failures hit exp1324/exp1325 in the conductor log.",
            ],
        },
    )
    _write_json(
        root / "results/experiment_1323_sota_gguf_token_health_prompt_runtime_diagnostic.json",
        {"status": "complete", "honest_verdict": "token_health_recovered"},
    )
    _write_json(
        root / "results/experiment_1325_triggered_certificate_extraction_v5_runtime_fixed_dccd_gbnf.json",
        {"status": "in_progress", "honest_verdict": "in_progress"},
    )
    _write_json(
        root / "results/experiment_1326_bootstrap_only.json",
        {"experiment_id": "exp1326", "status": "complete", "honest_verdict": "in_progress"},
    )
    (root / "research-roadmap.yaml").write_text(
        yaml.safe_dump(
            {
                "milestone": "2026.04.104",
                "tasks": [
                    {"id": "exp1337-environment-gate-disk-pretest-stale-skeleton-audit"},
                    {
                        "id": "exp1339-xgrammar2-tagdispatch-certificate-grammar-dryrun",
                        "gated_on": [
                            {
                                "upstream": "exp1337-environment-gate-disk-pretest-stale-skeleton-audit",
                                "artifact_field": "environment_ready",
                                "op": "==",
                                "value": True,
                            }
                        ],
                    },
                    {
                        "id": "exp1340-trigger-before-constrain-certificate-v6-sota",
                        "gated_on": [
                            {
                                "upstream": "exp1339-xgrammar2-tagdispatch-certificate-grammar-dryrun",
                                "artifact_field": "dynamic_grammar_ready",
                                "op": "==",
                                "value": True,
                            }
                        ],
                    },
                    {"id": "exp1341-independent-diagnostic", "gated_on": ["ignored malformed gate"]},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_gate_artifact_classifies_stale_103_state_REQ_INFRA_1337(tmp_path, monkeypatch):
    """REQ-INFRA-1337: passing disk and focused pre-test gates can proceed after stale .103 artifacts are classified."""
    _write_gate_inputs(tmp_path)
    focused_target = tmp_path / "tests/python/test_conductor_pretest_cache.py"
    focused_target.parent.mkdir(parents=True)
    focused_target.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    for script_name in ("validate_prior_failures.py", "audit_roadmap_gates.py"):
        (tmp_path / "scripts" / script_name).write_text("# available\n", encoding="utf-8")

    monkeypatch.setattr(
        environment_gate,
        "collect_filesystem_stats",
        lambda root: environment_gate.FileSystemStats(disk_free_gb=42.0, inode_free_pct=91.0),
    )

    def runner(cmd, cwd, timeout_s):
        joined = " ".join(cmd)
        if "test_conductor_pretest_cache.py" in joined:
            return environment_gate.CommandResult(0, "1 passed in 0.03s", "")
        if "validate_prior_failures.py" in joined:
            return environment_gate.CommandResult(1, "[SCHEMA ERRORS] research-roadmap-next.yaml\n  ERROR: File not found", "")
        return environment_gate.CommandResult(0, '{"roadmap_gate_audit_passed": true}', "")

    artifact = environment_gate.build_environment_gate_artifact(
        tmp_path,
        run_date="20260505",
        focused_pretest_target=focused_target,
        command_runner=runner,
    )

    assert artifact["status"] == "complete"
    assert artifact["disk_free_gb"] == 42.0
    assert artifact["inode_free_pct"] == 91.0
    assert artifact["disk_quota_ok"] is True
    assert artifact["focused_pretest_status"] == "passed"
    assert artifact["repeated_pretest_signature"]["disk_quota_signature"] == "Codex CLI error: [Errno 122] Disk quota exceeded"
    assert artifact["repeated_pretest_signature"]["pretest_signature"] == "Pre-tests failing, self-heal failed: 1 failed, 86 passed, 1 warning"
    assert artifact["repeated_pretest_signature"]["focused_pretest_signature_active"] is False
    assert artifact["stale_artifact_paths"] == [
        "results/experiment_1325_triggered_certificate_extraction_v5_runtime_fixed_dccd_gbnf.json",
        "results/experiment_1326_bootstrap_only.json",
    ]
    assert artifact["stale_skeleton_count"] == 2
    assert artifact["environment_ready"] is True
    assert artifact["recommended_task_pruning"] == []
    assert artifact["roadmap_health"]["prior_failures"]["status"] == "failed"
    assert artifact["honest_verdict"] == "environment_ready_stale_103_artifacts_classified"


def test_missing_focused_pretest_target_blocks_environment_REQ_INFRA_1337(tmp_path, monkeypatch):
    """SCENARIO-INFRA-1337: missing focused pre-test target is explicit and does not silently pass the gate."""
    _write_gate_inputs(tmp_path)
    monkeypatch.setattr(
        environment_gate,
        "collect_filesystem_stats",
        lambda root: environment_gate.FileSystemStats(disk_free_gb=42.0, inode_free_pct=91.0),
    )

    artifact = environment_gate.build_environment_gate_artifact(
        tmp_path,
        run_date="20260505",
        focused_pretest_target=tmp_path / "scripts/missing_focused_pretest.py",
        command_runner=lambda cmd, cwd, timeout_s: environment_gate.CommandResult(0, "", ""),
    )

    assert artifact["focused_pretest_status"] == "not_available"
    assert artifact["focused_pretest"]["missing_path"] == "scripts/missing_focused_pretest.py"
    assert artifact["environment_ready"] is False
    assert [item["task_id"] for item in artifact["recommended_task_pruning"]] == [
        "exp1339-xgrammar2-tagdispatch-certificate-grammar-dryrun",
        "exp1340-trigger-before-constrain-certificate-v6-sota",
    ]
    assert artifact["honest_verdict"] == "blocked_focused_pretest_not_available"


def test_repeated_pretest_signature_in_focused_output_blocks_environment_REQ_INFRA_1337(tmp_path, monkeypatch):
    """REQ-INFRA-1337: the .103 repeated failure signature remains blocking when the focused pre-test reproduces it."""
    _write_gate_inputs(tmp_path)
    focused_target = tmp_path / "scripts/focused_pretest.py"
    focused_target.write_text("# available\n", encoding="utf-8")
    monkeypatch.setattr(
        environment_gate,
        "collect_filesystem_stats",
        lambda root: environment_gate.FileSystemStats(disk_free_gb=42.0, inode_free_pct=91.0),
    )

    artifact = environment_gate.build_environment_gate_artifact(
        tmp_path,
        run_date="20260505",
        focused_pretest_target=focused_target,
        command_runner=lambda cmd, cwd, timeout_s: environment_gate.CommandResult(
            1,
            "Pre-tests failing, self-heal failed: 1 failed, 86 passed, 1 warning in 4.37s",
            "",
        ),
    )

    assert artifact["focused_pretest_status"] == "failed"
    assert artifact["repeated_pretest_signature"]["focused_pretest_signature_active"] is True
    assert artifact["environment_ready"] is False
    assert artifact["honest_verdict"] == "blocked_repeated_pretest_signature_active"


def test_disk_or_inode_gate_failure_blocks_environment_REQ_INFRA_1337(tmp_path, monkeypatch):
    """REQ-INFRA-1337: disk_quota_ok is false when free space or inodes are below the deterministic gate."""
    _write_gate_inputs(tmp_path)
    focused_target = tmp_path / "scripts/focused_pretest.py"
    focused_target.write_text("# available\n", encoding="utf-8")
    monkeypatch.setattr(
        environment_gate,
        "collect_filesystem_stats",
        lambda root: environment_gate.FileSystemStats(disk_free_gb=1.0, inode_free_pct=2.0),
    )

    artifact = environment_gate.build_environment_gate_artifact(
        tmp_path,
        run_date="20260505",
        focused_pretest_target=focused_target,
        command_runner=lambda cmd, cwd, timeout_s: environment_gate.CommandResult(0, "passed", ""),
    )

    assert artifact["disk_quota_ok"] is False
    assert artifact["environment_ready"] is False
    assert artifact["honest_verdict"] == "blocked_disk_quota_or_inode_gate"


def test_write_gate_artifact_writes_in_progress_then_complete_REQ_INFRA_1337(tmp_path, monkeypatch):
    """REQ-INFRA-1337: the artifact writer persists the required in-progress marker before final JSON."""
    _write_gate_inputs(tmp_path)
    focused_target = tmp_path / "scripts/focused_pretest.py"
    focused_target.write_text("# available\n", encoding="utf-8")
    output_path = tmp_path / "results/experiment_1337_environment_gate_disk_pretest_stale_skeleton_audit.json"
    seen_statuses = []

    monkeypatch.setattr(
        environment_gate,
        "collect_filesystem_stats",
        lambda root: environment_gate.FileSystemStats(disk_free_gb=42.0, inode_free_pct=91.0),
    )

    def observer(path, payload):
        seen_statuses.append(payload["status"])

    artifact = environment_gate.write_environment_gate_artifact(
        tmp_path,
        output_path=output_path,
        run_date="20260505",
        focused_pretest_target=focused_target,
        command_runner=lambda cmd, cwd, timeout_s: environment_gate.CommandResult(0, "passed", ""),
        write_observer=observer,
    )

    assert seen_statuses == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_collect_filesystem_stats_returns_nonnegative_values_REQ_INFRA_1337(tmp_path):
    """REQ-INFRA-1337: filesystem measurement uses the project root without deleting data."""
    stats = environment_gate.collect_filesystem_stats(tmp_path)

    assert stats.disk_free_gb >= 0.0
    assert 0.0 <= stats.inode_free_pct <= 100.0


def test_environment_gate_helper_edges_REQ_INFRA_1337(tmp_path, monkeypatch):
    """REQ-INFRA-1337: helper branches preserve explicit status for non-repeated failures and clean no-stale gates."""
    _write_gate_inputs(tmp_path)
    (tmp_path / "results/experiment_notes.json").write_text("{}", encoding="utf-8")
    focused_target = tmp_path / "scripts/focused_pretest.py"
    focused_target.write_text("# available\n", encoding="utf-8")
    monkeypatch.setattr(
        environment_gate,
        "collect_filesystem_stats",
        lambda root: environment_gate.FileSystemStats(disk_free_gb=42.0, inode_free_pct=91.0),
    )

    failed_artifact = environment_gate.build_environment_gate_artifact(
        tmp_path,
        run_date="20260505",
        focused_pretest_target=focused_target,
        command_runner=lambda cmd, cwd, timeout_s: environment_gate.CommandResult(1, "different failure", "stderr"),
    )

    assert failed_artifact["honest_verdict"] == "blocked_focused_pretest_failed"
    assert environment_gate._extract_pretest_signature("clean output") == ""
    assert environment_gate._relative_path(tmp_path, tmp_path.parent / "outside.json").endswith("outside.json")

    _write_json(
        tmp_path / "results/experiment_1325_triggered_certificate_extraction_v5_runtime_fixed_dccd_gbnf.json",
        {"status": "complete", "honest_verdict": "reconciled"},
    )
    _write_json(
        tmp_path / "results/experiment_1326_bootstrap_only.json",
        {"status": "complete", "honest_verdict": "substantive", "metric": 1},
    )
    ready_artifact = environment_gate.build_environment_gate_artifact(
        tmp_path,
        run_date="20260505",
        focused_pretest_target=focused_target,
        command_runner=lambda cmd, cwd, timeout_s: environment_gate.CommandResult(0, "passed", ""),
    )

    assert ready_artifact["stale_skeleton_count"] == 0
    assert ready_artifact["honest_verdict"] == "environment_ready"

    completed = environment_gate._run_subprocess(
        ["python3", "-c", "print('subprocess-ok')"],
        tmp_path,
        10,
    )
    assert completed.returncode == 0
    assert "subprocess-ok" in completed.stdout
