"""Tests for scripts/autoresearch_conductor_round.py -- the missing wiring
between the already-built autoresearch pipeline (python/carnot/autoresearch/)
and the unattended conductor loop.

Spec refs: REQ-AUTO-019 (unattended conductor integration),
REQ-AUTO-020 (git-committed lineage per accepted hypothesis).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import scripts.autoresearch_conductor_round as acr
from carnot.autoresearch.constitution import ConstitutionChecker
from carnot.autoresearch.experiment_log import ExperimentEntry


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "README.md").write_text("seed\n")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=path, check=True)


def _make_entry(entry_id: str = "auto-001") -> ExperimentEntry:
    return ExperimentEntry(
        id=entry_id,
        timestamp="2026-09-11T00:00:00Z",
        hypothesis_code="def run(d): return {'double_well': {'final_energy': -6.0}}",
        hypothesis_description="smaller step size",
        sandbox_success=True,
        sandbox_metrics={"double_well": {"final_energy": -6.0}},
        eval_verdict="PASS",
        eval_reason="improved",
        outcome="accepted",
    )


class TestCommitAcceptedHypothesis:
    def test_commits_a_scoped_single_path(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        checker = ConstitutionChecker()
        entry = _make_entry()

        sha = acr.commit_accepted_hypothesis(
            checker, entry, "double_well", 0.05, -6.0, project_root=tmp_path
        )

        assert sha is not None
        show = subprocess.run(
            ["git", "show", "--stat", "--format=", sha],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        changed = [
            line.strip().split("|")[0].strip() for line in show.stdout.splitlines() if "|" in line
        ]
        assert changed == ["ops/autoresearch_discoveries/double_well/auto-001.json"]

        record_path = (
            tmp_path / "ops" / "autoresearch_discoveries" / "double_well" / "auto-001.json"
        )
        assert record_path.exists()

    def test_commit_message_carries_the_score(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        checker = ConstitutionChecker()
        entry = _make_entry()

        sha = acr.commit_accepted_hypothesis(
            checker, entry, "double_well", 0.05, -6.0, project_root=tmp_path
        )

        msg = subprocess.run(
            ["git", "log", "-1", "--format=%B", sha],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "0.05 -> -6.0" in msg
        assert "REQ-AUTO-019" in msg

    def test_forbidden_constitution_never_reaches_git(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        locked = ConstitutionChecker(allowed=(), forbidden=(r"create_file:",), requires_approval=())
        entry = _make_entry()

        sha = acr.commit_accepted_hypothesis(
            locked, entry, "double_well", 0.05, -6.0, project_root=tmp_path
        )

        assert sha is None
        log = subprocess.run(
            ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout
        assert log.count("\n") == 1  # only the seed commit
        assert not (tmp_path / "ops").exists()

    def test_forbidden_git_commit_action_never_reaches_git(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        # create_file is allowed, but git_commit itself is forbidden
        locked = ConstitutionChecker(
            allowed=(r"create_file:",), forbidden=(r"git_commit",), requires_approval=()
        )
        entry = _make_entry()

        sha = acr.commit_accepted_hypothesis(
            locked, entry, "double_well", 0.05, -6.0, project_root=tmp_path
        )

        assert sha is None
        log = subprocess.run(
            ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout
        assert log.count("\n") == 1


class TestEndpointReachable:
    def test_unreachable_returns_false(self) -> None:
        # Nothing listens on this port in a test environment.
        assert acr.endpoint_reachable("http://127.0.0.1:1", timeout=0.5) is False


class TestRunRound:
    def test_unreachable_endpoint_is_non_fatal(self, tmp_path: Path) -> None:
        rc = acr.run_round(
            api_base="http://127.0.0.1:1",
            model="m",
            max_iterations=5,
            project_root=tmp_path,
            receipt_path=tmp_path / "receipt.md",
        )
        assert rc == 0
        receipt = (tmp_path / "receipt.md").read_text()
        assert "BLOCKED" in receipt

    def test_bounded_run_commits_only_accepted_winners(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)

        def fake_generator(_cfg, _baselines, _failures, iteration, count=1):
            if iteration == 0:
                return [("better", "def run(d): return {'double_well': {'final_energy': -6.0}}")]
            return [("worse", "def run(d): return {'double_well': {'final_energy': 5.0}}")]

        with (
            patch.object(acr, "endpoint_reachable", return_value=True),
            patch.object(acr, "generate_hypotheses_batch", side_effect=fake_generator),
        ):
            rc = acr.run_round(
                api_base="http://127.0.0.1:9999",
                model="m",
                max_iterations=2,
                project_root=tmp_path,
                receipt_path=tmp_path / "receipt.md",
            )

        assert rc == 0
        discoveries = list((tmp_path / "ops" / "autoresearch_discoveries").rglob("*.json"))
        assert len(discoveries) == 1  # only the winner got a lineage commit
        log = subprocess.run(
            ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout
        assert log.count("\n") == 2  # seed + one accepted hypothesis

    def test_baseline_and_log_caches_stay_local_not_committed(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)

        def fake_generator(_cfg, _baselines, _failures, iteration, count=1):
            return []  # no hypotheses -> loop stops immediately, nothing accepted

        with (
            patch.object(acr, "endpoint_reachable", return_value=True),
            patch.object(acr, "generate_hypotheses_batch", side_effect=fake_generator),
        ):
            acr.run_round(
                api_base="http://127.0.0.1:9999",
                model="m",
                max_iterations=2,
                project_root=tmp_path,
                receipt_path=tmp_path / "receipt.md",
            )

        # Caches/receipt are written to disk (untracked is expected here -- the real
        # repo's .gitignore is what keeps them out of `git status` for real, this
        # fixture has none) but nothing was ever `git add`-ed or committed.
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert all(line.startswith("??") for line in status.splitlines())
        log = subprocess.run(
            ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout
        assert log.count("\n") == 1  # only the seed commit


class TestSeedBaselines:
    def test_matches_demo_autoresearch(self) -> None:
        record = acr.seed_baselines()
        assert record.benchmarks["double_well"].final_energy == 0.05
        assert record.benchmarks["rosenbrock"].final_energy == 0.5

    def test_load_falls_back_to_seed_when_cache_missing(self, tmp_path: Path) -> None:
        record = acr.load_baselines(tmp_path / "does_not_exist.json")
        assert record.benchmarks["double_well"].final_energy == 0.05

    def test_load_falls_back_to_seed_on_corrupt_cache(self, tmp_path: Path) -> None:
        cache = tmp_path / "corrupt.json"
        cache.write_text("not json")
        record = acr.load_baselines(cache)
        assert record.benchmarks["double_well"].final_energy == 0.05
