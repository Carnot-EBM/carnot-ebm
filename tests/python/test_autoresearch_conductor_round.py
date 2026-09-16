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

import pytest

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

    def test_generator_label_lands_in_the_commit_message(self, tmp_path: Path) -> None:
        """REQ-AUTO-024. Regression for the round-1 incident: every commit
        message said "via codex exec" even when Fable produced the
        hypothesis, because the text was hardcoded rather than passed in."""
        _init_repo(tmp_path)
        checker = ConstitutionChecker()
        entry = _make_entry()

        sha = acr.commit_accepted_hypothesis(
            checker,
            entry,
            "double_well",
            0.05,
            -6.0,
            project_root=tmp_path,
            generator_label="Fable 5.1 fallback (codex returned nothing this iteration)",
        )

        assert sha is not None
        msg = subprocess.run(
            ["git", "log", "-1", "--format=%B", sha],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "via Fable 5.1 fallback" in msg
        assert "via codex exec" not in msg

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

    def test_does_not_sweep_a_pre_staged_unrelated_file(self, tmp_path: Path) -> None:
        """Regression for adversarial review 2026-09-12 finding 2: a bare
        `git commit -m` commits the WHOLE index, not just the one added path.
        Reproduced by staging an unrelated file first."""
        _init_repo(tmp_path)
        (tmp_path / "unrelated_in_flight.py").write_text("# someone else's work\n")
        subprocess.run(["git", "add", "unrelated_in_flight.py"], cwd=tmp_path, check=True)
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
        # the unrelated file must still be staged, untouched by this commit
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "unrelated_in_flight.py" in status

    def test_rejects_a_path_traversal_benchmark_name(self, tmp_path: Path) -> None:
        """Regression for finding 3: benchmark_name is LLM-controlled data
        flowing into a filesystem path with no sanitization."""
        _init_repo(tmp_path)
        checker = ConstitutionChecker()
        entry = _make_entry()

        sha = acr.commit_accepted_hypothesis(
            checker,
            entry,
            "../../tests/injected",
            0.05,
            -6.0,
            project_root=tmp_path,
        )

        assert sha is None
        assert not (tmp_path / "tests" / "injected").exists()
        log = subprocess.run(
            ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout
        assert log.count("\n") == 1

    def test_non_json_serializable_metric_does_not_raise(self, tmp_path: Path) -> None:
        """Regression for finding 4: a numpy/jax scalar in the metrics dict
        used to crash json.dumps before any receipt could be written."""
        _init_repo(tmp_path)
        checker = ConstitutionChecker()

        class NotJsonable:
            pass

        entry = _make_entry()
        entry.sandbox_metrics = {"double_well": {"final_energy": NotJsonable()}}

        sha = acr.commit_accepted_hypothesis(
            checker, entry, "double_well", 0.05, -6.0, project_root=tmp_path
        )

        assert sha is None  # did not raise
        assert not (tmp_path / "ops" / "autoresearch_discoveries").exists()

    def test_failed_commit_leaves_no_untracked_residue(self, tmp_path: Path) -> None:
        """Regression for finding 6: a refused commit used to only unstage
        the file, leaving it untracked on disk for a later `git add -A` to
        sweep into someone else's commit."""
        _init_repo(tmp_path)
        checker = ConstitutionChecker()
        entry = _make_entry()

        def fake_git(project_root, *args, check=False):
            if args and args[0] == "commit":
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="hook refused")
            return subprocess.run(
                ["git", *args], cwd=project_root, check=check, capture_output=True, text=True
            )

        with patch.object(acr, "_git", side_effect=fake_git):
            sha = acr.commit_accepted_hypothesis(
                checker, entry, "double_well", 0.05, -6.0, project_root=tmp_path
            )

        assert sha is None
        assert not (
            tmp_path / "ops" / "autoresearch_discoveries" / "double_well" / "auto-001.json"
        ).exists()


class TestCodexAvailable:
    def test_returns_false_when_not_on_path(self) -> None:
        with patch.object(acr.shutil, "which", return_value=None):
            assert acr.codex_available() is False

    def test_returns_true_when_on_path(self) -> None:
        with patch.object(acr.shutil, "which", return_value="/usr/bin/codex"):
            assert acr.codex_available() is True


class TestCallCodex:
    def test_argv_matches_the_conductors_own_codex_pattern(self) -> None:
        """Same flags/shape as _build_agent_command's codex branch and
        pages_adversarial_audit.py's call_codex -- prompt piped via stdin,
        terminated with a bare '-', no repo tool access implied."""
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["input"] = kwargs.get("input")
            return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

        with patch.object(acr.subprocess, "run", side_effect=fake_run):
            ok, out = acr.call_codex("hello", "gpt-6-astra", 60)

        assert ok is True
        assert out == "ok"
        argv = captured["argv"]
        assert argv[:3] == ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox"]
        assert "--model" in argv and argv[argv.index("--model") + 1] == "gpt-6-astra"
        assert argv[-1] == "-"
        assert captured["input"] == "hello"

    def test_nonzero_exit_is_reported_not_raised(self) -> None:
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")

        with patch.object(acr.subprocess, "run", side_effect=fake_run):
            ok, out = acr.call_codex("hello", "gpt-6-astra", 60)
        assert ok is False
        assert "boom" in out

    def test_timeout_is_reported_not_raised(self) -> None:
        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 60))

        with patch.object(acr.subprocess, "run", side_effect=fake_run):
            ok, out = acr.call_codex("hello", "gpt-6-astra", 60)
        assert ok is False


class TestCodexGenerateHypotheses:
    def test_extracts_hypotheses_from_a_real_shaped_response(self) -> None:
        from carnot.autoresearch.baselines import BaselineRecord

        response = (
            "Try a smaller step size.\n\n"
            "```python\n"
            "def run(benchmark_data):\n"
            "    return {'double_well': {'final_energy': -6.0}}\n"
            "```\n"
        )
        with patch.object(acr, "call_codex", return_value=(True, response)):
            hyps = acr.codex_generate_hypotheses("gpt-6-astra", 60, BaselineRecord(), [], 0)
        assert len(hyps) == 1
        assert "def run(benchmark_data)" in hyps[0][1]

    def test_failed_call_returns_empty_and_records_the_failure(self) -> None:
        from carnot.autoresearch.baselines import BaselineRecord

        failures: list[dict] = []
        with patch.object(acr, "call_codex", return_value=(False, "codex exit 1")):
            hyps = acr.codex_generate_hypotheses("gpt-6-astra", 60, BaselineRecord(), failures, 0)
        assert hyps == []
        assert failures and failures[0]["description"] == "codex_call_failed"


class TestCallFable:
    def test_argv_uses_the_documented_fable_model_alias(self) -> None:
        """2026-09-12 operator directive: follow up a zero-hypothesis codex
        round with a Fable 5.1 attempt. `claude --help` documents 'fable' as
        a first-class --model alias -- verify the exact call shape, and that
        it is a stateless --print completion, never an agentic session (no
        --dangerously-skip-permissions, unlike call_codex's bypass flag)."""
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

        with patch.object(acr.subprocess, "run", side_effect=fake_run):
            ok, out = acr.call_fable("hello", 60)

        assert ok is True
        assert out == "ok"
        argv = captured["argv"]
        assert argv[0] == "claude"
        assert "--model" in argv and argv[argv.index("--model") + 1] == "fable"
        assert "--print" in argv and argv[argv.index("--print") + 1] == "hello"
        assert "--dangerously-skip-permissions" not in argv

    def test_nonzero_exit_is_reported_not_raised(self) -> None:
        def fake_run(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")

        with patch.object(acr.subprocess, "run", side_effect=fake_run):
            ok, out = acr.call_fable("hello", 60)
        assert ok is False
        assert "boom" in out

    def test_timeout_is_reported_not_raised(self) -> None:
        def fake_run(argv, **kwargs):
            raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 60))

        with patch.object(acr.subprocess, "run", side_effect=fake_run):
            ok, out = acr.call_fable("hello", 60)
        assert ok is False


class TestGenerateHypothesesWithFallback:
    def test_codex_success_never_calls_fable(self) -> None:
        from carnot.autoresearch.baselines import BaselineRecord

        fable_calls = []
        with (
            patch.object(acr, "codex_generate_hypotheses", return_value=[("d", "code")]),
            patch.object(
                acr,
                "fable_generate_hypotheses",
                side_effect=lambda *a, **kw: fable_calls.append(1) or [],
            ),
        ):
            log: list[int] = []
            hyps = acr.generate_hypotheses_with_fallback(
                "gpt-6-astra", 60, BaselineRecord(), [], 0, log
            )
        assert hyps == [("d", "code")]
        assert fable_calls == []
        assert log == []

    def test_codex_empty_falls_back_to_fable_and_records_the_iteration(self) -> None:
        """The exact scenario the operator asked for: codex returns nothing,
        Fable gets a real attempt at the same question before the round
        gives up on this iteration."""
        from carnot.autoresearch.baselines import BaselineRecord

        with (
            patch.object(acr, "codex_generate_hypotheses", return_value=[]),
            patch.object(acr, "fable_generate_hypotheses", return_value=[("fable-found", "code")]),
        ):
            log: list[int] = []
            hyps = acr.generate_hypotheses_with_fallback(
                "gpt-6-astra", 60, BaselineRecord(), [], 3, log
            )
        assert hyps == [("fable-found", "code")]
        assert log == [3]

    def test_both_generators_empty_returns_empty(self) -> None:
        from carnot.autoresearch.baselines import BaselineRecord

        with (
            patch.object(acr, "codex_generate_hypotheses", return_value=[]),
            patch.object(acr, "fable_generate_hypotheses", return_value=[]),
        ):
            log: list[int] = []
            hyps = acr.generate_hypotheses_with_fallback(
                "gpt-6-astra", 60, BaselineRecord(), [], 0, log
            )
        assert hyps == []
        assert log == [0]  # the attempt is still recorded even though it also failed


class TestGeneratorLabelForEntry:
    """REQ-AUTO-024: the commit-message attribution bug -- round 1's real
    production commits all said "via codex exec" even though
    fable_fallback_iterations showed codex failed every single iteration."""

    def test_iteration_not_in_fallback_list_is_codex(self) -> None:
        assert acr.generator_label_for_entry("llm-20260913-082215-000", [1, 2]) == "codex exec"

    def test_iteration_in_fallback_list_is_fable(self) -> None:
        label = acr.generator_label_for_entry("llm-20260913-082215-002", [0, 2, 4])
        assert "Fable" in label

    def test_unparseable_id_defaults_to_codex_not_a_crash(self) -> None:
        assert acr.generator_label_for_entry("not-the-expected-shape", [0]) == "codex exec"


class TestRecomputeMetrics:
    """Unit tests for the REQ-AUTO-021 fix to adversarial review finding 1."""

    def test_self_reported_final_energy_is_replaced_with_the_real_one(self) -> None:
        out = acr._recompute_metrics(
            {"double_well": {"final_energy": -999999.0, "final_state": [1.0, 1.0]}}
        )
        assert out["double_well"]["final_energy"] == 0.0

    def test_missing_final_state_drops_final_energy_entirely(self) -> None:
        out = acr._recompute_metrics({"double_well": {"final_energy": -999.0}})
        assert "final_energy" not in out["double_well"]

    def test_unknown_benchmark_name_never_gets_a_final_energy(self) -> None:
        out = acr._recompute_metrics({"made_up_bench": {"final_energy": 123.0}})
        assert "final_energy" not in out["made_up_bench"]

    def test_other_fields_survive_untouched(self) -> None:
        out = acr._recompute_metrics(
            {
                "double_well": {
                    "final_energy": -999.0,
                    "final_state": [1.0, 1.0],
                    "wall_clock_seconds": 1.2,
                }
            }
        )
        assert out["double_well"]["wall_clock_seconds"] == 1.2

    def test_verifier_auroc_self_reported_energy_is_replaced_with_the_real_one(self) -> None:
        """REQ-AUTO-025: the same trust boundary as double_well/rosenbrock
        above, now exercised for the third benchmark's own dispatch branch."""
        out = acr._recompute_metrics(
            {"verifier_auroc": {"final_energy": -999999.0, "final_state": [0.5, 0.5]}}
        )
        assert out["verifier_auroc"]["final_energy"] == acr.measure_default_weight_energy()

    def test_verifier_auroc_malformed_state_drops_final_energy_entirely(self) -> None:
        out = acr._recompute_metrics({"verifier_auroc": {"final_energy": -1.0}})
        assert "final_energy" not in out["verifier_auroc"]


class TestDefaultBenchmarkData:
    """REQ-AUTO-025: the training split must reach the hypothesis, and the
    heavy corpus load must not happen at import time (every test file pays
    it otherwise)."""

    def test_includes_the_toy_benchmark_dimension(self) -> None:
        data = acr.default_benchmark_data()
        assert data["dim"] == 2

    def test_includes_verifier_auroc_train_rows(self) -> None:
        data = acr.default_benchmark_data()
        rows = data["verifier_auroc_train_rows"]
        assert rows
        assert set(rows[0].keys()) == {"step_text", "label"}

    def test_train_rows_never_include_held_out_question_ids(self) -> None:
        from carnot.autoresearch.verifier_auroc_benchmark import _split_corpus

        _, held_out = _split_corpus()
        held_out_texts = {r["step_text"] for r in held_out}
        data = acr.default_benchmark_data()
        train_texts = {r["step_text"] for r in data["verifier_auroc_train_rows"]}
        assert not (train_texts & held_out_texts)


class TestSystemPromptMentionsVerifierAuroc:
    def test_prompt_describes_the_third_benchmark(self) -> None:
        assert "verifier_auroc" in acr.AUTORESEARCH_SYSTEM_PROMPT
        assert "PCIBProbe" in acr.AUTORESEARCH_SYSTEM_PROMPT

    def test_prompt_tells_the_hypothesis_its_own_auroc_is_not_trusted(self) -> None:
        assert "never trusted" in acr.AUTORESEARCH_SYSTEM_PROMPT.lower()


class TestVerifiedExecuteHypothesis:
    def test_a_fabricated_energy_with_no_state_never_reaches_the_evaluator(self) -> None:
        """The exact shape of the original adversarial-review reproduction,
        run through the REAL sandbox (not mocked) end to end."""
        code = "def run(d): return {'double_well': {'final_energy': -999999.0}}"
        result = acr._verified_execute_hypothesis(code, {"dim": 2})
        assert result.success is True
        assert "final_energy" not in result.metrics["double_well"]

    def test_a_real_state_recomputes_correctly_through_the_real_sandbox(self) -> None:
        code = "def run(d): return {'double_well': {'final_state': [1.0, 1.0]}}"
        result = acr._verified_execute_hypothesis(code, {"dim": 2})
        assert result.success is True
        assert result.metrics["double_well"]["final_energy"] == 0.0

    def test_sandbox_failure_passes_through_unmodified(self) -> None:
        code = "def run(d): raise ValueError('boom')"
        result = acr._verified_execute_hypothesis(code, {"dim": 2})
        assert result.success is False

    def test_verifier_auroc_hypothesis_runs_through_the_real_sandbox(self) -> None:
        """REQ-AUTO-025, end to end: a hypothesis that reads the training
        rows handed to it and reports weights it never validated itself
        still gets independently rescored against the (unseen-to-it)
        held-out split -- the real sandbox, real recompute (now a fresh
        subprocess per REQ-AUTO-025's own CRITICAL-2 fix), nothing mocked."""
        from carnot.autoresearch.verifier_auroc_benchmark import (
            recompute_verifier_auroc_energy,
        )

        code = (
            "def run(d):\n"
            "    assert 'verifier_auroc_train_rows' in d\n"
            "    return {'verifier_auroc': {'final_state': [-1.0, 1.0]}}\n"
        )
        result = acr._verified_execute_hypothesis(code, acr.default_benchmark_data())
        assert result.success is True
        energy = result.metrics["verifier_auroc"]["final_energy"]
        # Compared against a direct, in-process recompute (fine in test code --
        # no untrusted code has run here) to confirm the SUBPROCESS path used
        # in production gives the identical, correct number.
        assert energy == recompute_verifier_auroc_energy([-1.0, 1.0])

    def test_hypothesis_cannot_import_carnot_at_all(self) -> None:
        """REQ-AUTO-025 CRITICAL-2 (Variant A) fix: the sandbox must reject
        ANY `carnot` import from sandboxed hypothesis code, not just direct
        access to the held-out split -- this is what makes PCIBProbe's
        injection via benchmark_data (not `import carnot.verify.pcib_probe`)
        actually load-bearing rather than cosmetic."""
        code = "def run(d):\n    import carnot\n    return {}\n"
        config = acr.SandboxConfig(blocked_modules=acr.BLOCKED_MODULES | frozenset({"carnot"}))
        result = acr._verified_execute_hypothesis(code, {"dim": 2}, config)
        assert result.success is False

    def test_hypothesis_cannot_reach_the_held_out_split_via_direct_import(self) -> None:
        """The exact CRITICAL-2 Variant A reproduction from the 2026-09-16
        adversarial review, run through the real sandbox with the carnot
        import root blocked: a hypothesis that tries to import the benchmark
        module itself to read the held-out split must fail, not succeed."""
        code = (
            "def run(d):\n"
            "    import carnot.autoresearch.verifier_auroc_benchmark as vab\n"
            "    _, held_out = vab._split_corpus()\n"
            "    return {'verifier_auroc': {'final_state': [0.5, 0.5]}}\n"
        )
        config = acr.SandboxConfig(blocked_modules=acr.BLOCKED_MODULES | frozenset({"carnot"}))
        result = acr._verified_execute_hypothesis(code, acr.default_benchmark_data(), config)
        assert result.success is False

    def test_hypothesis_can_still_use_the_injected_pcibprobe(self) -> None:
        """The carnot-import block must not break the intended workflow:
        PCIBProbe handed via benchmark_data must still work end to end."""
        code = (
            "def run(d):\n"
            "    Probe = d['PCIBProbe']\n"
            "    probe = Probe(entity_weight=-1.0, falsifiability_weight=1.0)\n"
            "    probe.score(d['verifier_auroc_train_rows'][0]['step_text'], '')\n"
            "    return {'verifier_auroc': {'final_state': [-1.0, 1.0]}}\n"
        )
        config = acr.SandboxConfig(blocked_modules=acr.BLOCKED_MODULES | frozenset({"carnot"}))
        result = acr._verified_execute_hypothesis(code, acr.default_benchmark_data(), config)
        assert result.success is True
        assert "final_energy" in result.metrics["verifier_auroc"]

    def test_monkeypatched_binary_auroc_no_longer_fabricates_a_zero_energy(self) -> None:
        """The exact CRITICAL-2 (Variant B1) reproduction from the 2026-09-16
        adversarial review: a hypothesis that patches _binary_auroc to always
        return 1.0 must NOT get a fabricated final_energy of 0.0, because the
        recompute now runs in a fresh subprocess that never sees this
        process's monkeypatch."""
        code = (
            "def run(d):\n"
            "    import carnot.autoresearch.verifier_auroc_benchmark as vab\n"
            "    vab._binary_auroc = lambda labels, scores: 1.0\n"
            "    return {'verifier_auroc': {'final_state': [0.5, 0.5]}}\n"
        )
        # Deliberately NOT blocking carnot here -- this reproduces the
        # in-process monkeypatch attack even when the import itself succeeds,
        # to prove the SUBPROCESS recompute (not the import block) is what
        # actually closes it.
        result = acr._verified_execute_hypothesis(code, acr.default_benchmark_data())
        assert result.success is True
        energy = result.metrics["verifier_auroc"].get("final_energy")
        assert energy != 0.0
        assert energy == pytest.approx(acr.measure_default_weight_energy())

    def test_hypothesis_mutations_do_not_poison_later_parent_scoring(self) -> None:
        """SCENARIO-AUTO-025-E: cached rows and trusted registries recover too."""
        verifier = acr._verifier_auroc_module
        original_binary_auroc = verifier._binary_auroc
        original_split_corpus = verifier._split_corpus
        original_load_corpus_rows = verifier._load_corpus_rows
        original_energy_functions = dict(acr.BENCHMARK_ENERGY_FUNCTIONS)
        _, held_out = original_split_corpus()
        original_step_text = held_out[0]["step_text"]
        code = (
            "def run(d):\n"
            "    import carnot.autoresearch.toy_benchmarks as toy\n"
            "    import carnot.autoresearch.verifier_auroc_benchmark as vab\n"
            "    _, held_out = vab._split_corpus()\n"
            "    held_out[0]['step_text'] = 'poisoned cached row'\n"
            "    vab._binary_auroc = lambda labels, scores: 1.0\n"
            "    vab._split_corpus = lambda: ((), ())\n"
            "    vab._load_corpus_rows = lambda: ()\n"
            "    toy.BENCHMARK_ENERGY_FUNCTIONS.clear()\n"
            "    return {'double_well': {'final_state': [1.0, 1.0]}}\n"
        )

        result = acr._verified_execute_hypothesis(code, acr.default_benchmark_data())

        assert result.success is True
        assert result.metrics["double_well"]["final_energy"] == 0.0
        assert verifier._binary_auroc is original_binary_auroc
        assert verifier._split_corpus is original_split_corpus
        assert verifier._load_corpus_rows is original_load_corpus_rows
        assert acr.BENCHMARK_ENERGY_FUNCTIONS == original_energy_functions
        assert verifier._split_corpus()[1][0]["step_text"] == original_step_text


class TestEnergyVerificationPatch:
    def test_patches_and_restores_the_orchestrator_module(self) -> None:
        original = acr._orchestrator_module.execute_hypothesis
        with acr._energy_verification_patch():
            assert acr._orchestrator_module.execute_hypothesis is acr._verified_execute_hypothesis
        assert acr._orchestrator_module.execute_hypothesis is original


class TestRunRound:
    def test_fabricated_energy_claim_is_never_committed_end_to_end(self, tmp_path: Path) -> None:
        """The definitive close of adversarial review CRITICAL finding 1,
        reproduced exactly as the reviewer reported it, run through the real
        run_round -> real sandbox -> real evaluator -> real commit-or-not
        pipeline, no mocking below codex_generate_hypotheses."""
        _init_repo(tmp_path)

        def fake_generator(
            _model,
            _timeout,
            _baselines,
            _failures,
            iteration,
            _fallback_log=None,
            _fable_timeout=None,
        ):
            return [
                (
                    "claims an impossible energy",
                    "def run(d): return {'double_well': {'final_energy': -999999.0}}",
                )
            ]

        with (
            patch.object(acr, "codex_available", return_value=True),
            patch.object(acr, "generate_hypotheses_with_fallback", side_effect=fake_generator),
        ):
            rc = acr.run_round(
                model="gpt-6-astra",
                max_iterations=3,
                project_root=tmp_path,
                receipt_path=tmp_path / "receipt.md",
            )

        assert rc == 0
        log = subprocess.run(
            ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout
        assert log.count("\n") == 1  # only the seed commit -- the fabricated claim never landed
        assert not (tmp_path / "ops" / "autoresearch_discoveries").exists()

    def test_fable_produced_hypothesis_is_attributed_to_fable_end_to_end(
        self, tmp_path: Path
    ) -> None:
        """REQ-AUTO-024, full pipeline. Round 1's real production commits all
        said "via codex exec" although fable_fallback_iterations showed codex
        failed every iteration -- this reproduces the exact wiring
        (run_round -> generator closure -> commit_accepted_hypothesis) that
        must attribute correctly."""
        _init_repo(tmp_path)

        def fake_generator(
            _model,
            _timeout,
            _baselines,
            _failures,
            iteration,
            fallback_log,
            _fable_timeout,
        ):
            fallback_log.append(iteration)  # simulate: codex failed, Fable won
            return [
                ("fable win", "def run(d): return {'double_well': {'final_state': [1.0, 1.0]}}")
            ]

        with (
            patch.object(acr, "codex_available", return_value=True),
            patch.object(acr, "generate_hypotheses_with_fallback", side_effect=fake_generator),
        ):
            acr.run_round(
                model="gpt-6-astra",
                max_iterations=1,
                project_root=tmp_path,
                receipt_path=tmp_path / "receipt.md",
            )

        msg = subprocess.run(
            ["git", "log", "-1", "--format=%B"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "via Fable 5.1 fallback" in msg
        assert "via codex exec" not in msg

    def test_both_generators_failing_leaves_reasons_in_the_receipt(self, tmp_path: Path) -> None:
        """REQ-AUTO-022 / SCENARIO-AUTO-022-A. Real call_codex/call_fable
        (only subprocess.run is faked), so recent_failures is populated the
        same way a genuine timeout or non-zero exit would populate it."""
        _init_repo(tmp_path)

        def fake_run(argv, **kwargs):
            if argv[0] == "codex":
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="some stderr")
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="some other stderr")

        with (
            patch.object(acr, "codex_available", return_value=True),
            patch.object(acr.subprocess, "run", side_effect=fake_run),
        ):
            rc = acr.run_round(
                model="gpt-6-astra",
                max_iterations=1,
                project_root=tmp_path,
                receipt_path=tmp_path / "receipt.md",
            )

        assert rc == 0
        receipt = (tmp_path / "receipt.md").read_text()
        assert "## Generator failure reasons" in receipt
        assert "codex_call_failed: codex exit 1: some stderr" in receipt
        assert "fable_call_failed: claude exit 1: some other stderr" in receipt

    def test_a_clean_round_omits_the_failure_reasons_section(self, tmp_path: Path) -> None:
        """REQ-AUTO-022 / SCENARIO-AUTO-022-B."""
        _init_repo(tmp_path)

        def fake_generator(
            _model,
            _timeout,
            _baselines,
            _failures,
            iteration,
            _fallback_log=None,
            _fable_timeout=None,
        ):
            return [
                ("clean win", "def run(d): return {'double_well': {'final_state': [1.0, 1.0]}}")
            ]

        with (
            patch.object(acr, "codex_available", return_value=True),
            patch.object(acr, "generate_hypotheses_with_fallback", side_effect=fake_generator),
        ):
            acr.run_round(
                model="gpt-6-astra",
                max_iterations=1,
                project_root=tmp_path,
                receipt_path=tmp_path / "receipt.md",
            )

        receipt = (tmp_path / "receipt.md").read_text()
        assert "## Generator failure reasons" not in receipt

    def test_codex_unavailable_is_non_fatal(self, tmp_path: Path) -> None:
        with patch.object(acr, "codex_available", return_value=False):
            rc = acr.run_round(
                model="gpt-6-astra",
                max_iterations=5,
                project_root=tmp_path,
                receipt_path=tmp_path / "receipt.md",
            )
        assert rc == 0
        receipt = (tmp_path / "receipt.md").read_text()
        assert "BLOCKED" in receipt

    def test_bounded_run_commits_only_accepted_winners(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)

        def fake_generator(
            _model,
            _timeout,
            _baselines,
            _failures,
            iteration,
            _fallback_log=None,
            _fable_timeout=None,
        ):
            if iteration == 0:
                # double_well_energy([1.0, 1.0]) == 0.0, beats baseline 0.05
                return [
                    ("better", "def run(d): return {'double_well': {'final_state': [1.0, 1.0]}}")
                ]
            # double_well_energy([0.0, 0.0]) == 2.0, worse than the new baseline 0.0
            return [("worse", "def run(d): return {'double_well': {'final_state': [0.0, 0.0]}}")]

        with (
            patch.object(acr, "codex_available", return_value=True),
            patch.object(acr, "generate_hypotheses_with_fallback", side_effect=fake_generator),
        ):
            rc = acr.run_round(
                model="gpt-6-astra",
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

        def fake_generator(
            _model,
            _timeout,
            _baselines,
            _failures,
            iteration,
            _fallback_log=None,
            _fable_timeout=None,
        ):
            return []  # no hypotheses -> loop stops immediately, nothing accepted

        with (
            patch.object(acr, "codex_available", return_value=True),
            patch.object(acr, "generate_hypotheses_with_fallback", side_effect=fake_generator),
        ):
            acr.run_round(
                model="gpt-6-astra",
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

    def test_a_no_op_and_a_made_up_benchmark_are_never_committed(self, tmp_path: Path) -> None:
        """Regression for finding 1: PASS means 'no regression', not 'genuine
        improvement' -- an empty dict and an unknown benchmark name both used
        to reach a git commit. Only entry.eval_improvements should ever be
        committed."""
        _init_repo(tmp_path)

        def fake_generator(
            _model,
            _timeout,
            _baselines,
            _failures,
            iteration,
            _fallback_log=None,
            _fable_timeout=None,
        ):
            if iteration == 0:
                return [("no-op", "def run(d): return {}")]
            return [
                (
                    "made up benchmark",
                    "def run(d): return {'made_up_bench': {'final_energy': 123.0}}",
                )
            ]

        with (
            patch.object(acr, "codex_available", return_value=True),
            patch.object(acr, "generate_hypotheses_with_fallback", side_effect=fake_generator),
        ):
            acr.run_round(
                model="gpt-6-astra",
                max_iterations=2,
                project_root=tmp_path,
                receipt_path=tmp_path / "receipt.md",
            )

        log = subprocess.run(
            ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout
        assert log.count("\n") == 1  # only the seed commit -- neither reached a commit
        assert not (tmp_path / "ops" / "autoresearch_discoveries").exists()

    def test_commit_message_score_is_per_hypothesis_not_round_final(self, tmp_path: Path) -> None:
        """Regression for finding 5: the commit's before/after score must be
        this hypothesis's own numbers, not whatever the baseline happened to
        be after the WHOLE round finished."""
        _init_repo(tmp_path)

        def fake_generator(
            _model,
            _timeout,
            _baselines,
            _failures,
            iteration,
            _fallback_log=None,
            _fable_timeout=None,
        ):
            if iteration == 0:
                # double_well_energy([1.1, 1.0]) == 0.04410000000000008, beats baseline 0.05
                return [
                    (
                        "first win",
                        "def run(d): return {'double_well': {'final_state': [1.1, 1.0]}}",
                    )
                ]
            # double_well_energy([1.0, 1.0]) == 0.0, beats the new baseline 0.0441...
            return [
                ("second win", "def run(d): return {'double_well': {'final_state': [1.0, 1.0]}}")
            ]

        with (
            patch.object(acr, "codex_available", return_value=True),
            patch.object(acr, "generate_hypotheses_with_fallback", side_effect=fake_generator),
        ):
            acr.run_round(
                model="gpt-6-astra",
                max_iterations=2,
                project_root=tmp_path,
                receipt_path=tmp_path / "receipt.md",
            )

        log = subprocess.run(
            ["git", "log", "--format=%s%n%b", "--reverse"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        # first accepted hypothesis: baseline 0.05 -> its own real recomputed energy
        assert "0.05 -> 0.04410000000000008" in log
        # second accepted hypothesis: baseline 0.0441... (not some other value) -> 0.0
        assert "0.04410000000000008 -> 0.0" in log


class TestSeedBaselines:
    def test_matches_demo_autoresearch(self) -> None:
        record = acr.seed_baselines()
        assert record.benchmarks["double_well"].final_energy == 0.05
        assert record.benchmarks["rosenbrock"].final_energy == 0.5

    def test_verifier_auroc_seed_is_a_real_measurement(self) -> None:
        """REQ-AUTO-025: unlike the two illustrative placeholders above, this
        seed must equal an actual recomputation, not a hand-typed number."""
        record = acr.seed_baselines()
        assert record.benchmarks["verifier_auroc"].final_energy == (
            acr.measure_default_weight_energy()
        )

    def test_load_falls_back_to_seed_when_cache_missing(self, tmp_path: Path) -> None:
        record = acr.load_baselines(tmp_path / "does_not_exist.json")
        assert record.benchmarks["double_well"].final_energy == 0.05

    def test_load_falls_back_to_seed_on_corrupt_cache(self, tmp_path: Path) -> None:
        cache = tmp_path / "corrupt.json"
        cache.write_text("not json")
        record = acr.load_baselines(cache)
        assert record.benchmarks["double_well"].final_energy == 0.05

    def test_a_cache_predating_verifier_auroc_gets_it_seeded(self, tmp_path: Path) -> None:
        """REQ-AUTO-025 CRITICAL-1 fix (2026-09-16 adversarial review): the
        REAL production cache (ops/.autoresearch_baselines.json) predates
        this benchmark and holds only double_well/rosenbrock. Before the
        fix, load_baselines returned that cache verbatim, so verifier_auroc
        was silently ABSENT -- the evaluator then treated ANY reported
        weights as 'nothing to compare against' and accepted them
        unconditionally, landing whatever the first hypothesis proposed as
        the baseline instead of the measured seed."""
        cache = tmp_path / "old_baselines.json"
        old = acr.BaselineRecord(version="0.1.0")
        old.benchmarks["double_well"] = acr.BenchmarkMetrics(
            benchmark_name="double_well",
            final_energy=0.05,
            convergence_steps=5000,
            wall_clock_seconds=2.0,
        )
        old.benchmarks["rosenbrock"] = acr.BenchmarkMetrics(
            benchmark_name="rosenbrock",
            final_energy=0.5,
            convergence_steps=10000,
            wall_clock_seconds=5.0,
        )
        old.save(cache)
        assert "verifier_auroc" not in old.benchmarks  # sanity: reproduces the real gap

        record = acr.load_baselines(cache)

        assert "verifier_auroc" in record.benchmarks
        assert record.benchmarks["verifier_auroc"].final_energy == (
            acr.measure_default_weight_energy()
        )
        # The pre-existing entries must be UNCHANGED, not reset to the seed --
        # a real, evolved baseline must never be silently overwritten.
        assert record.benchmarks["double_well"].final_energy == 0.05
        assert record.benchmarks["rosenbrock"].final_energy == 0.5

    def test_a_cache_with_a_real_verifier_auroc_baseline_is_not_overwritten(
        self, tmp_path: Path
    ) -> None:
        """The migration must be additive-only: a benchmark the cache
        already tracks (even verifier_auroc itself, once a real round has
        run) keeps its real, evolved value -- never reset to the seed."""
        cache = tmp_path / "evolved_baselines.json"
        record = acr.seed_baselines()
        record.benchmarks["verifier_auroc"].final_energy = 0.1234  # a real "improved" value
        record.save(cache)

        loaded = acr.load_baselines(cache)

        assert loaded.benchmarks["verifier_auroc"].final_energy == 0.1234
