"""Tests for the conductor constitution: autonomous vs. human-approved actions.

Spec coverage: REQ-AUTO-015, SCENARIO-AUTO-015-001, SCENARIO-AUTO-015-002,
               SCENARIO-AUTO-015-003, SCENARIO-AUTO-015-004,
               SCENARIO-AUTO-015-005, SCENARIO-AUTO-015-006,
               SCENARIO-AUTO-015-007, SCENARIO-AUTO-015-008,
               SCENARIO-AUTO-015-009
"""

import pytest

from carnot.autoresearch.constitution import (
    ALLOWED_ACTIONS,
    ActionCategory,
    ConstitutionChecker,
    ConstitutionVerdict,
    FORBIDDEN_ACTIONS,
    REQUIRES_APPROVAL_ACTIONS,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _checker() -> ConstitutionChecker:
    """Return a default-configured checker using module-level policy tables."""
    return ConstitutionChecker()


# ---------------------------------------------------------------------------
# ALLOWED_ACTIONS tests
# ---------------------------------------------------------------------------


def test_allowed_create_file_python_carnot() -> None:
    """Create files under python/carnot/ is allowed autonomously.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    # The conductor writes new experiment results and model files here.
    verdict = _checker().check("create_file:python/carnot/models/new_model.py")
    assert verdict.category == ActionCategory.ALLOWED
    assert verdict.matched_rule != ""


def test_allowed_create_file_scripts() -> None:
    """Create files under scripts/ is allowed.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("create_file:scripts/experiment_999.py")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_create_file_tests() -> None:
    """Create files under tests/ is allowed.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("create_file:tests/python/test_new_thing.py")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_create_file_results() -> None:
    """Create files under results/ is allowed.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("create_file:results/experiment_999_results.json")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_modify_file_python_carnot() -> None:
    """Modify files under python/carnot/ is allowed.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("modify_file:python/carnot/autoresearch/evaluator.py")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_modify_file_scripts() -> None:
    """Modify scripts/ files (but not research_conductor.py) is allowed.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("modify_file:scripts/experiment_template.py")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_run_tests() -> None:
    """Running tests is always allowed autonomously.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("run_tests")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_run_pytest() -> None:
    """Running pytest is always allowed.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("run_pytest")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_git_commit() -> None:
    """Git commit (without push) is allowed.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("git_commit")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_git_stage() -> None:
    """Git staging is allowed.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("git_stage")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_read_file() -> None:
    """Reading any file is always allowed (read-only, no side effects).

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("read_file:some/sensitive/config.yaml")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_run_sandbox() -> None:
    """Running the autoresearch sandbox is allowed.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("run_sandbox")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_huggingface_fetch() -> None:
    """HTTP GET to huggingface.co is in the approved-domains allow-list.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("http_get:https://huggingface.co/Carnot-EBM/ising")
    assert verdict.category == ActionCategory.ALLOWED


def test_allowed_arxiv_fetch() -> None:
    """HTTP GET to arxiv.org is in the approved-domains allow-list.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-001
    """
    verdict = _checker().check("http_get:https://arxiv.org/abs/2603.25158")
    assert verdict.category == ActionCategory.ALLOWED


# ---------------------------------------------------------------------------
# FORBIDDEN_ACTIONS tests
# ---------------------------------------------------------------------------


def test_forbidden_delete_non_test_file() -> None:
    """Deleting a non-test file is forbidden — irreversible data loss.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("delete_file:python/carnot/models/ising.py")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_delete_env_file() -> None:
    """Deleting .env is forbidden.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("delete_file:.env")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_modify_env() -> None:
    """Modifying .env is forbidden to protect credentials at rest.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("modify_file:.env")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_modify_credentials() -> None:
    """Modifying credentials files is forbidden.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("modify_file:secrets/credentials.json")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_modify_conductor_itself() -> None:
    """The conductor must not modify its own source code (self-modification guard).

    This is a critical safety invariant: if the conductor could rewrite its own
    logic it could disable all safety checks in subsequent runs.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-003
    """
    verdict = _checker().check("modify_file:scripts/research_conductor.py")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_git_push() -> None:
    """Git push to remote without approval is forbidden.

    Pushing is irreversible from the conductor's perspective and can affect
    shared branches seen by the whole team.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("git_push")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_git_force_push() -> None:
    """Force push is always forbidden — it can destroy upstream history.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("git_force_push")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_http_post() -> None:
    """HTTP POST to external services is forbidden (network exfiltration guard).

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("http_post:https://example.com/api")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_exec_shell() -> None:
    """Raw shell execution is forbidden — the conductor uses specific commands only.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("exec_shell:rm -rf /")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_ssh_connect() -> None:
    """SSH connections are forbidden — no lateral movement to other machines.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("ssh_connect:prod-server.example.com")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_git_config() -> None:
    """Git config modification is forbidden.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("git_config:user.email=attacker@example.com")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_forbidden_modify_pem_key() -> None:
    """Modifying PEM/key files is forbidden.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-002
    """
    verdict = _checker().check("modify_file:deploy/server.pem")
    assert verdict.category == ActionCategory.FORBIDDEN


# ---------------------------------------------------------------------------
# REQUIRES_APPROVAL_ACTIONS tests
# ---------------------------------------------------------------------------


def test_requires_approval_modify_pyproject_toml() -> None:
    """Changing pyproject.toml dependencies requires human approval.

    A bad dependency change can break the entire build across CI and developer
    machines — it needs a human eye before landing.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-004
    """
    verdict = _checker().check("modify_file:pyproject.toml")
    assert verdict.category == ActionCategory.REQUIRES_APPROVAL


def test_requires_approval_modify_cargo_toml() -> None:
    """Changing Cargo.toml (Rust dependencies) requires human approval.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-004
    """
    verdict = _checker().check("modify_file:Cargo.toml")
    assert verdict.category == ActionCategory.REQUIRES_APPROVAL


def test_requires_approval_modify_claude_md() -> None:
    """Modifying CLAUDE.md requires approval — it defines the agent's own rules.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-004
    """
    verdict = _checker().check("modify_file:CLAUDE.md")
    assert verdict.category == ActionCategory.REQUIRES_APPROVAL


def test_requires_approval_archive_tests() -> None:
    """Archiving tests requires approval — reduces test coverage without review.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-004
    """
    verdict = _checker().check("archive_tests:tests/python/test_old_experiment.py")
    assert verdict.category == ActionCategory.REQUIRES_APPROVAL


def test_requires_approval_delete_test_file() -> None:
    """Deleting a test file requires approval (not outright forbidden but gated).

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-004
    """
    verdict = _checker().check("delete_file:tests/python/test_old_feature.py")
    assert verdict.category == ActionCategory.REQUIRES_APPROVAL


def test_requires_approval_modify_openspec() -> None:
    """Modifying spec documents requires human approval.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-004
    """
    verdict = _checker().check("modify_file:openspec/capabilities/autoresearch/spec.md")
    assert verdict.category == ActionCategory.REQUIRES_APPROVAL


def test_requires_approval_modify_bmad() -> None:
    """Modifying _bmad strategic documents requires human approval.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-004
    """
    verdict = _checker().check("modify_file:_bmad/architecture.md")
    assert verdict.category == ActionCategory.REQUIRES_APPROVAL


# ---------------------------------------------------------------------------
# Default-deny (unknown actions) tests
# ---------------------------------------------------------------------------


def test_default_deny_unknown_action() -> None:
    """Unknown actions are forbidden by default (fail-safe policy).

    If the conductor encounters an action verb not in any policy table, it must
    refuse.  This prevents a novel attack surface from being accidentally
    auto-approved.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-005
    """
    verdict = _checker().check("teleport_to_moon")
    assert verdict.category == ActionCategory.FORBIDDEN
    assert "default" in verdict.matched_rule


def test_default_deny_empty_action() -> None:
    """An empty action string is forbidden by default.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-005
    """
    verdict = _checker().check("")
    assert verdict.category == ActionCategory.FORBIDDEN


# ---------------------------------------------------------------------------
# FORBIDDEN precedence over ALLOWED tests
# ---------------------------------------------------------------------------


def test_forbidden_takes_precedence_over_allowed_for_conductor_script() -> None:
    """FORBIDDEN wins even if a pattern in ALLOWED would also match.

    research_conductor.py lives under scripts/ which is in ALLOWED, but the
    specific file is in FORBIDDEN — the more specific FORBIDDEN rule must win.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-006
    """
    # scripts/ matches ALLOWED modify_file:scripts/
    # but research_conductor.py is in FORBIDDEN
    verdict = _checker().check("modify_file:scripts/research_conductor.py")
    assert verdict.category == ActionCategory.FORBIDDEN


# ---------------------------------------------------------------------------
# ConstitutionVerdict structure tests
# ---------------------------------------------------------------------------


def test_verdict_has_matched_rule() -> None:
    """Every non-default verdict records which rule pattern matched.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-007
    """
    verdict = _checker().check("run_tests")
    assert verdict.matched_rule != ""
    assert verdict.matched_rule != "<default deny>"


def test_verdict_records_action_string() -> None:
    """The verdict echoes back the original action string for audit logging.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-007
    """
    action = "create_file:results/exp_42.json"
    verdict = _checker().check(action)
    assert verdict.action == action


# ---------------------------------------------------------------------------
# assert_allowed tests
# ---------------------------------------------------------------------------


def test_assert_allowed_passes_for_allowed_action() -> None:
    """assert_allowed does not raise for a permitted action.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-008
    """
    checker = _checker()
    checker.assert_allowed("run_tests")  # should not raise


def test_assert_allowed_raises_for_forbidden_action() -> None:
    """assert_allowed raises ValueError for a forbidden action.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-008
    """
    checker = _checker()
    with pytest.raises(ValueError, match="Constitution violation"):
        checker.assert_allowed("git_push")


def test_assert_allowed_raises_for_requires_approval_action() -> None:
    """assert_allowed raises ValueError for an action that requires approval.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-008
    """
    checker = _checker()
    with pytest.raises(ValueError, match="Constitution violation"):
        checker.assert_allowed("modify_file:pyproject.toml")


# ---------------------------------------------------------------------------
# Custom policy override tests
# ---------------------------------------------------------------------------


def test_custom_forbidden_extends_policy() -> None:
    """Callers can append extra FORBIDDEN rules without touching module constants.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-009
    """
    extra_forbidden = tuple(FORBIDDEN_ACTIONS) + (r"modify_file:my_critical_file\.py",)
    checker = ConstitutionChecker(forbidden=extra_forbidden)
    verdict = checker.check("modify_file:my_critical_file.py")
    assert verdict.category == ActionCategory.FORBIDDEN


def test_custom_allowed_extends_policy() -> None:
    """Callers can add extra ALLOWED rules for project-specific safe paths.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-009
    """
    extra_allowed = tuple(ALLOWED_ACTIONS) + (r"create_file:my_special_dir/",)
    checker = ConstitutionChecker(allowed=extra_allowed)
    verdict = checker.check("create_file:my_special_dir/output.json")
    assert verdict.category == ActionCategory.ALLOWED


# ---------------------------------------------------------------------------
# Orchestrator integration: constitution_checker wired into run_loop
# ---------------------------------------------------------------------------


def test_orchestrator_run_loop_respects_constitution_checker() -> None:
    """run_loop skips sandbox execution when the checker forbids run_sandbox.

    This verifies the constitution is actually wired into the loop, not just
    present as a dormant field.

    Spec: REQ-AUTO-015, SCENARIO-AUTO-015-006
    """
    from carnot.autoresearch.baselines import BaselineRecord, BenchmarkMetrics
    from carnot.autoresearch.orchestrator import AutoresearchConfig, run_loop

    # Build a checker that forbids run_sandbox (simulate a locked-down policy)
    locked_checker = ConstitutionChecker(
        allowed=(),  # nothing allowed
        forbidden=(r"run_sandbox",),
        requires_approval=(),
    )

    baselines = BaselineRecord(version="test")
    baselines.benchmarks["bench"] = BenchmarkMetrics(
        benchmark_name="bench",
        final_energy=-5.0,
        convergence_steps=100,
        wall_clock_seconds=1.0,
        peak_memory_mb=50.0,
    )

    hypotheses = [
        ("good hyp", "def run(d): return {'bench': {'final_energy': -6.0}}"),
    ]

    config = AutoresearchConfig(
        max_iterations=5,
        constitution_checker=locked_checker,
    )

    result = run_loop(hypotheses, baselines, {}, config)

    # The hypothesis was blocked before sandbox — counted as rejected, not accepted
    assert result.accepted == 0
    assert result.rejected == 1
    assert result.iterations == 1
