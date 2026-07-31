"""Tests that the autoresearch pipeline can actually reach the container sandbox.

Spec coverage: REQ-AUTO-004, REQ-AUTO-009, REQ-SEC-003

Origin: 2026-07-31 security audit, follow-up finding.

`sandbox_docker.run_in_docker` was exported from `carnot.autoresearch` and
invoked by NOTHING.  `orchestrator.py` called the in-process `run_in_sandbox`
at all three execution sites and contained zero references to
`CARNOT_USE_SANDBOX`.  So for the autoresearch pipeline -- the component that
exists to execute LLM-generated code -- the container boundary was unreachable
and the documented environment variable did nothing.

That is the same shape as the bug the audit started with: a control that reads
as available and never fires.  These tests assert the wiring exists, and in
particular that `CARNOT_REQUIRE_SANDBOX=1` FAILS CLOSED rather than quietly
running untrusted code in-process.
"""

import carnot.autoresearch.sandbox as sandbox_mod
import pytest
from carnot.autoresearch.sandbox import SandboxConfig, execute_hypothesis

HYPOTHESIS = "def run(d):\n    return {'ok': 1}\n"


@pytest.fixture(autouse=True)
def _clear_sandbox_env(monkeypatch):
    """Neither variable set unless a test sets it."""
    monkeypatch.delenv("CARNOT_USE_SANDBOX", raising=False)
    monkeypatch.delenv("CARNOT_REQUIRE_SANDBOX", raising=False)


class TestDefaultIsUnchanged:
    """With neither variable set, behaviour must be exactly as before."""

    def test_runs_in_process_by_default(self) -> None:
        result = execute_hypothesis(HYPOTHESIS, {})
        assert result.success, result.error
        assert result.metrics == {"ok": 1}

    def test_does_not_touch_docker_by_default(self, monkeypatch) -> None:
        """No Docker probe at all -- this must not add a Docker dependency."""
        import carnot.autoresearch.sandbox_docker as docker_mod

        def _boom() -> bool:
            raise AssertionError("Docker was probed on the default path")

        monkeypatch.setattr(docker_mod, "is_docker_available", _boom)
        assert execute_hypothesis(HYPOTHESIS, {}).success


class TestContainerRouting:
    """REQ-SEC-003: the env vars must actually reach run_in_docker."""

    def test_use_sandbox_routes_to_container_when_available(self, monkeypatch) -> None:
        import carnot.autoresearch.sandbox_docker as docker_mod

        calls: list[str] = []

        def _fake_run(code, data, config):  # type: ignore[no-untyped-def]
            calls.append("docker")
            return sandbox_mod.SandboxResult(success=True, metrics={"from": "container"})

        monkeypatch.setattr(docker_mod, "is_docker_available", lambda: True)
        monkeypatch.setattr(docker_mod, "is_image_available", lambda image: True)
        monkeypatch.setattr(docker_mod, "run_in_docker", _fake_run)
        monkeypatch.setenv("CARNOT_USE_SANDBOX", "1")

        result = execute_hypothesis(HYPOTHESIS, {})
        assert calls == ["docker"], "CARNOT_USE_SANDBOX=1 did not reach the container"
        assert result.metrics == {"from": "container"}

    def test_container_failure_is_not_retried_in_process(self, monkeypatch) -> None:
        """A failing hypothesis must NOT fall back and run again unsandboxed.

        `run_in_docker` returns a failed SandboxResult both when Docker is
        missing and when the hypothesis itself fails. Treating any failure as
        "infrastructure unavailable" would re-execute the code in-process --
        double execution, and it would drop isolation exactly when the code is
        misbehaving. Availability is therefore checked BEFORE launching, and the
        container's own verdict is final.
        """
        import carnot.autoresearch.sandbox_docker as docker_mod

        monkeypatch.setattr(docker_mod, "is_docker_available", lambda: True)
        monkeypatch.setattr(docker_mod, "is_image_available", lambda image: True)
        monkeypatch.setattr(
            docker_mod,
            "run_in_docker",
            lambda code, data, config: sandbox_mod.SandboxResult(
                success=False, error="hypothesis raised ZeroDivisionError"
            ),
        )
        monkeypatch.setenv("CARNOT_USE_SANDBOX", "1")

        result = execute_hypothesis(HYPOTHESIS, {})
        assert not result.success
        assert "ZeroDivisionError" in (result.error or ""), (
            "the container's failure was replaced by an in-process re-run"
        )

    def test_timeout_is_carried_across_to_the_container(self, monkeypatch) -> None:
        """Both backends must honour the same timeout, not silently diverge."""
        import carnot.autoresearch.sandbox_docker as docker_mod

        seen: dict[str, int] = {}

        def _fake_run(code, data, config):  # type: ignore[no-untyped-def]
            seen["timeout"] = config.timeout_seconds
            return sandbox_mod.SandboxResult(success=True, metrics={})

        monkeypatch.setattr(docker_mod, "is_docker_available", lambda: True)
        monkeypatch.setattr(docker_mod, "is_image_available", lambda image: True)
        monkeypatch.setattr(docker_mod, "run_in_docker", _fake_run)
        monkeypatch.setenv("CARNOT_USE_SANDBOX", "1")

        execute_hypothesis(HYPOTHESIS, {}, SandboxConfig(timeout_seconds=42))
        assert seen["timeout"] == 42


class TestUnavailableContainer:
    """The two variables differ ONLY here, and the difference is the point."""

    def test_use_sandbox_falls_back_with_a_loud_warning(self, monkeypatch) -> None:
        """USE = best effort, so a dev box without Docker still runs -- but noisily."""
        import carnot.autoresearch.sandbox_docker as docker_mod

        monkeypatch.setattr(docker_mod, "is_docker_available", lambda: False)
        monkeypatch.setenv("CARNOT_USE_SANDBOX", "1")

        with pytest.warns(RuntimeWarning, match="IN-PROCESS sandbox"):
            result = execute_hypothesis(HYPOTHESIS, {})
        assert result.success
        assert result.metrics == {"ok": 1}

    def test_require_sandbox_fails_closed(self, monkeypatch) -> None:
        """REQUIRE = the caller has declared the code untrusted.

        Running it in-process anyway would be the worst available outcome, so
        this must fail rather than degrade. This is the assertion that makes
        CARNOT_REQUIRE_SANDBOX worth setting.
        """
        import carnot.autoresearch.sandbox_docker as docker_mod

        monkeypatch.setattr(docker_mod, "is_docker_available", lambda: False)
        monkeypatch.setenv("CARNOT_REQUIRE_SANDBOX", "1")

        result = execute_hypothesis(HYPOTHESIS, {})
        assert not result.success, "REQUIRE_SANDBOX executed the hypothesis anyway"
        assert "blocked_sandbox_required_but_unavailable" in (result.error or "")

    def test_require_sandbox_does_not_execute_the_code(self, monkeypatch) -> None:
        """Prove nothing ran, rather than inferring it from the verdict."""
        import carnot.autoresearch.sandbox_docker as docker_mod

        monkeypatch.setattr(docker_mod, "is_docker_available", lambda: False)
        monkeypatch.setattr(
            sandbox_mod,
            "run_in_sandbox",
            lambda *a, **k: pytest.fail("in-process sandbox ran under REQUIRE_SANDBOX"),
        )
        monkeypatch.setenv("CARNOT_REQUIRE_SANDBOX", "1")

        assert not execute_hypothesis(HYPOTHESIS, {}).success


class TestOrchestratorIsWired:
    """The regression that started this: the orchestrator bypassed the dispatcher."""

    def test_orchestrator_calls_the_dispatcher_not_the_backend(self) -> None:
        """A source-level assertion, because this is a wiring bug, not a logic bug.

        `run_in_docker` was exported and called by nothing for as long as it had
        existed. Asserting behaviour alone would not have caught that; asserting
        the call site does.
        """
        from pathlib import Path

        import carnot.autoresearch.orchestrator as orch

        source = Path(orch.__file__).read_text(encoding="utf-8")
        assert "execute_hypothesis(" in source, "orchestrator does not use the dispatcher"
        assert "run_in_sandbox(" not in source, (
            "orchestrator still calls the in-process backend directly, which "
            "ignores CARNOT_USE_SANDBOX / CARNOT_REQUIRE_SANDBOX"
        )
