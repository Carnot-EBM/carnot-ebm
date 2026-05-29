"""Tests for Exp 3346 KV260 MMD-vs-CPU sequential-Gibbs continuity rerun.

Spec refs: REQ-HW-101, SCENARIO-HW-101.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.hardware import kv260_mmd_vs_cpu_sequential_gibbs as exp2938
from carnot.hardware import kv260_mmd_vs_cpu_sequential_gibbs_3346 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"


def _upload(n_spins: int = 64, max_degree: int = 2) -> dict[str, Any]:
    return {
        "layout": "ising_sampler_v2_n64_sparse_axi_q8_8",
        "max_degree": max_degree,
        "h_q88": [0 for _ in range(n_spins)],
        "adjacency": [
            [int((row + offset + 1) % n_spins) for offset in range(max_degree)]
            for row in range(n_spins)
        ],
        "couplings_q88": [[64, -32] for _ in range(n_spins)],
    }


def _exp2898_payload() -> dict[str, Any]:
    seeds = list(exp.RANDOM_SEEDS)
    problems = []
    specs = []
    for seed in seeds:
        problem = exp2938.generate_ising_problem(seed)
        problem["upload"] = _upload()
        problem["beta_final_q88"] = 256
        problems.append(problem)
        specs.append(exp2938.problem_spec(problem))
    return {
        "experiment_id": 2898,
        "bitstream_sha256": "a" * 64,
        "random_seeds_used": seeds,
        "problem_payload": {
            "n_spins": 64,
            "random_seeds_used": seeds,
            "ising_problem_specs": specs,
            "problems": problems,
        },
    }


def _write_exp2898(root: Path, payload: dict[str, Any] | None = None) -> None:
    path = root / exp.EXP2898_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _exp2898_payload(), sort_keys=True), encoding="utf-8")


def _fake_cpu_runner(problem, n_samples, burn_in_sweeps):
    # Deterministic CPU energies clustered well below the board energies so the
    # distinguishability finding reproduces in the test without a real chain.
    return exp.EnergyRunResult(
        seed=problem.seed,
        energies=[-5.0 - 0.001 * index for index in range(n_samples)],
        energy_sha256=exp2938.sha256_canonical([problem.seed, "cpu", n_samples]),
        update_schedule=exp.CPU_UPDATE_SCHEDULE,
        spin_orders_sha256=exp2938.sha256_canonical([problem.seed, "orders"]),
    )


def _board_evidence(n_samples: int) -> exp.BoardEvidence:
    return exp.BoardEvidence(
        ssh_reachable=True,
        board_uname="Linux kria 5.15.0-xilinx aarch64",
        xmutil_status="carnot_ising_v4 XRT_FLAT id_ok",
        uio_status="/dev/uio0\n/dev/uio4",
        bitstream_sha256="a" * 64,
        energies_by_seed={
            42: [5.0 + 0.001 * i for i in range(n_samples)],
            137: [5.5 + 0.001 * i for i in range(n_samples)],
            271: [6.0 + 0.001 * i for i in range(n_samples)],
        },
        command_transcript=[{"label": "precondition_ssh", "returncode": 0}],
        blocked_reasons=[],
        transcript_path=exp.TRANSCRIPT_REL_PATH.as_posix(),
        board_summary={"selected_uio": "/dev/uio4"},
    )


def test_req_hw_101_spec_anchor_exists() -> None:
    """REQ-HW-101: OpenSpec anchors the continuity rerun artifact and ssh path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-101" in spec
    assert "SCENARIO-HW-101" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "mmd_vs_cpu" in spec
    assert "ssh_reachable" in spec


def test_no_host_sd_card_check_anywhere() -> None:
    """REQ-HW-101: the retired host SD-card precondition must never appear."""

    source = Path(exp.__file__).read_text(encoding="utf-8")
    runner = (REPO_ROOT / "scripts" / "experiment_3346_kv260_mmd_vs_cpu_sequential_gibbs_v1.py").read_text(
        encoding="utf-8"
    )
    # The retired host SD-card check (any operational form) must never appear.
    # The module docstring may *name* the retired ``/dev/mmcblk*`` pattern to
    # explain why it is gone, but no command string may actually probe it.
    for retired in ("ls /dev/mmcblk", "test -e /dev/mmcblk", "/dev/mmcblk0", "mmcblk0p1"):
        assert retired not in source
    assert "/dev/mmcblk" not in runner
    # The sole hardware precondition is SSH reachability of the board.
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true" in source
    assert "BatchMode=yes" in source and "KV260_HOST" in source


def test_summarize_energies_handles_empty_and_populated() -> None:
    """REQ-HW-101: energy summaries are JSON-safe and checksummed."""

    empty = exp.summarize_energies([])
    assert empty["count"] == 0
    assert len(empty["sha256"]) == 64

    summary = exp.summarize_energies([1.0, 2.0, 3.0])
    assert summary["count"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(3.0)
    assert len(summary["sha256"]) == 64


def test_record_appends_transcript_entry_and_returns_combined_output() -> None:
    """REQ-HW-101: each board command is captured for the audit transcript."""

    transcript: list[dict[str, Any]] = []
    result = exp2938.CommandResult(
        cmd=["ssh", "kria", "uname", "-a"],
        returncode=0,
        stdout="Linux kria 5.15.0 aarch64",
        stderr="warn",
        duration_s=0.123456789,
    )
    combined = exp._record(transcript, "board_uname", result)

    assert "Linux kria" in combined and "warn" in combined
    assert len(transcript) == 1
    entry = transcript[0]
    assert entry["label"] == "board_uname"
    assert entry["returncode"] == 0
    assert entry["duration_s"] == pytest.approx(0.123457)
    assert entry["stdout_tail"] == "Linux kria 5.15.0 aarch64"
    assert entry["stderr_tail"] == "warn"


def test_run_experiment_blocks_missing_exp2898(tmp_path: Path) -> None:
    """REQ-HW-101: missing Exp 2898 provenance fails closed, not fabricated."""

    artifact = exp.run_experiment(
        root_path=tmp_path,
        board_collector=lambda problems: _board_evidence(4),
        started_s=0.0,
        now_s=1.0,
    )

    assert artifact["honest_verdict"] == "blocked_exp2898_artifact_missing"
    assert artifact["mmd_vs_cpu"] == {}
    assert artifact["blocked_reasons"]
    assert (tmp_path / exp.OUTPUT_REL_PATH).exists()


def test_run_experiment_blocks_bad_provenance(tmp_path: Path) -> None:
    """REQ-HW-101: malformed Exp 2898 provenance never becomes comparison data."""

    payload = _exp2898_payload()
    payload["problem_payload"]["random_seeds_used"] = [42]
    _write_exp2898(tmp_path, payload)

    artifact = exp.run_experiment(
        root_path=tmp_path,
        board_collector=lambda problems: _board_evidence(4),
        started_s=0.0,
        now_s=1.0,
    )
    assert artifact["honest_verdict"] == "blocked_exp2898_problem_reproduction_failed"
    assert artifact["blocked_reasons"]


def test_run_experiment_blocks_when_ssh_unreachable(tmp_path: Path) -> None:
    """SCENARIO-HW-101: an SSH failure preserves the transcript and blocks."""

    _write_exp2898(tmp_path)

    def unreachable(problems):
        return exp.BoardEvidence(
            ssh_reachable=False,
            command_transcript=[{"label": "precondition_ssh", "returncode": 255}],
            blocked_reasons=["blocked_kv260_ssh_unreachable: rc=255 timed out"],
            transcript_path=exp.TRANSCRIPT_REL_PATH.as_posix(),
        )

    artifact = exp.run_experiment(
        root_path=tmp_path,
        board_collector=unreachable,
        started_s=0.0,
        now_s=2.0,
    )

    assert artifact["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert artifact["ssh_reachable"] is False
    assert artifact["mmd_vs_cpu"] == {}
    assert artifact["command_transcript"]
    assert artifact["blocked_reasons"]


def test_run_experiment_blocks_on_board_verdict_without_prefix(tmp_path: Path) -> None:
    """REQ-HW-101: a non-prefixed blocked reason is normalised to blocked_*."""

    _write_exp2898(tmp_path)

    def odd_block(problems):
        return exp.BoardEvidence(
            ssh_reachable=True,
            blocked_reasons=["overlay went missing"],
            transcript_path=exp.TRANSCRIPT_REL_PATH.as_posix(),
        )

    artifact = exp.run_experiment(
        root_path=tmp_path,
        board_collector=odd_block,
        started_s=0.0,
        now_s=1.0,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_board_path_failed"
    assert artifact["blocked_reasons"] == ["overlay went missing"]


def test_run_experiment_blocks_incomplete_energy_trace(tmp_path: Path) -> None:
    """REQ-HW-101: a short hardware trace blocks instead of comparing garbage."""

    _write_exp2898(tmp_path)

    def short_trace(problems):
        evidence = _board_evidence(4)
        evidence.energies_by_seed[271] = [6.0, 6.1]  # too few
        return evidence

    artifact = exp.run_experiment(
        root_path=tmp_path,
        board_collector=short_trace,
        cpu_energy_runner=_fake_cpu_runner,
        n_samples=4,
        burn_in_sweeps=1,
        started_s=0.0,
        now_s=1.0,
    )
    assert artifact["honest_verdict"] == "blocked_kv260_energy_trace_incomplete"
    assert artifact["mmd_vs_cpu"] == {}
    assert artifact["sample_count_kv260"] == 2


def test_run_experiment_success_records_distinguishable_distributions(tmp_path: Path) -> None:
    """SCENARIO-HW-101: matched CPU/board traces yield a three-seed verdict."""

    _write_exp2898(tmp_path)

    artifact = exp.run_experiment(
        root_path=tmp_path,
        board_collector=lambda problems: _board_evidence(40),
        cpu_energy_runner=_fake_cpu_runner,
        n_samples=40,
        burn_in_sweeps=2,
        n_permutations=49,
        max_permutation_samples=60,
        started_s=0.0,
        now_s=75.0,
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["ssh_reachable"] is True
    assert artifact["board_uname"].startswith("Linux kria")
    assert "carnot_ising_v4" in artifact["xmutil_status"]
    assert "/dev/uio" in artifact["uio_status"]
    assert len(artifact["mmd_vs_cpu"]["per_seed"]) == 3
    assert artifact["mmd_vs_cpu"]["distributions_distinguishable"] is True
    assert artifact["paper_v6_recommendation"].startswith("retract")
    assert artifact["sample_count_cpu"] == artifact["sample_count_kv260"] == 40
    assert artifact["random_seed"] == 42
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["files_updated"] == [
        exp.OUTPUT_REL_PATH.as_posix(),
        exp.TRANSCRIPT_REL_PATH.as_posix(),
    ]
    assert set(artifact["cpu_baseline_summary"]) >= {"42", "137", "271", "update_schedule"}

    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))
    assert saved == artifact


def test_build_artifact_retain_recommendation_when_indistinguishable() -> None:
    """REQ-HW-101: identical distributions produce a retain recommendation."""

    problems = exp2938.recover_exp2898_problems(_exp2898_payload())
    n = 30
    cpu_runs = {p.seed: _fake_cpu_runner(p, n, 1) for p in problems}
    identical = {p.seed: list(cpu_runs[p.seed].energies) for p in problems}
    evidence = exp.BoardEvidence(
        ssh_reachable=True,
        bitstream_sha256="a" * 64,
        energies_by_seed=identical,
    )
    comparisons = {
        seed: {
            "mmd_squared": 0.0,
            "mmd_pvalue": 1.0,
            "ks_statistic": 0.0,
            "ks_pvalue": 1.0,
            "bandwidth": 1.0,
        }
        for seed in exp.RANDOM_SEEDS
    }
    artifact = exp.build_artifact(
        evidence=evidence,
        problems=problems,
        cpu_runs=cpu_runs,
        comparisons=comparisons,
        sample_count_cpu=n,
        sample_count_kv260=n,
        duration_s=70.0,
        files_updated=["a", "b"],
    )
    exp.validate_artifact(artifact)
    assert artifact["mmd_vs_cpu"]["distributions_distinguishable"] is False
    assert artifact["paper_v6_recommendation"].startswith("retain")


def test_validate_artifact_rejects_malformed_success() -> None:
    """REQ-HW-101: schema validation enforces the success contract."""

    problems = exp2938.recover_exp2898_problems(_exp2898_payload())
    cpu_runs = {p.seed: _fake_cpu_runner(p, 20, 1) for p in problems}
    evidence = _board_evidence(20)
    comparisons = {
        seed: {
            "mmd_squared": 1.0,
            "mmd_pvalue": 0.001,
            "ks_statistic": 0.99,
            "ks_pvalue": 0.0,
            "bandwidth": 1.0,
        }
        for seed in exp.RANDOM_SEEDS
    }
    good = exp.build_artifact(
        evidence=evidence,
        problems=problems,
        cpu_runs=cpu_runs,
        comparisons=comparisons,
        sample_count_cpu=20,
        sample_count_kv260=20,
        duration_s=65.0,
        files_updated=["a"],
    )
    exp.validate_artifact(good)

    missing = dict(good)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing"):
        exp.validate_artifact(missing)

    bad_substrate = dict(good, inference_substrate="cpu")
    with pytest.raises(ValueError, match="hardware_smoke"):
        exp.validate_artifact(bad_substrate)

    bad_prefix = dict(good, honest_verdict="kv260 ran fine")
    with pytest.raises(ValueError, match="terminal"):
        exp.validate_artifact(bad_prefix)

    bad_seed = dict(good, random_seed=999)
    with pytest.raises(ValueError, match="random_seed"):
        exp.validate_artifact(bad_seed)

    mismatched = dict(good, sample_count_kv260=21)
    with pytest.raises(ValueError, match="sample_count"):
        exp.validate_artifact(mismatched)

    short_checksum = dict(good, reproducibility_checksum="abc")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(short_checksum)

    too_few_seeds = json.loads(json.dumps(good))
    too_few_seeds["mmd_vs_cpu"]["per_seed"].pop("271")
    with pytest.raises(ValueError, match="three seeds"):
        exp.validate_artifact(too_few_seeds)


def test_validate_artifact_blocked_contract() -> None:
    """REQ-HW-101: blocked artifacts must keep reasons and empty mmd_vs_cpu."""

    evidence = exp.BoardEvidence(
        ssh_reachable=False,
        blocked_reasons=["blocked_kv260_ssh_unreachable: rc=255"],
    )
    blocked = exp.build_artifact(
        evidence=evidence,
        problems=[],
        cpu_runs={},
        comparisons={},
        sample_count_cpu=512,
        sample_count_kv260=0,
        duration_s=1.0,
        files_updated=["a"],
        blocked_verdict="blocked_kv260_ssh_unreachable",
    )
    exp.validate_artifact(blocked)

    no_reasons = dict(blocked, blocked_reasons=[])
    with pytest.raises(ValueError, match="blocked_reasons"):
        exp.validate_artifact(no_reasons)

    leaks_mmd = dict(blocked, mmd_vs_cpu={"per_seed": {"42": {}}})
    with pytest.raises(ValueError, match="mmd_vs_cpu empty"):
        exp.validate_artifact(leaks_mmd)


def test_main_outputs_summary_and_result_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-HW-101: the module CLI reports the deliverable path."""

    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root_path: {"honest_verdict": "complete: cli-ok"},
    )

    assert exp.main(["--root", str(tmp_path)]) == 0
    assert "complete: cli-ok" in capsys.readouterr().out

    assert exp.main(["--root", str(tmp_path), "--print-result-path"]) == 0
    assert str(tmp_path / exp.OUTPUT_REL_PATH) in capsys.readouterr().out
