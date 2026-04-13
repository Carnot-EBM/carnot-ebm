"""Tests for gVisor-backed code verification sandboxing.

Spec coverage: REQ-CODE-001, SCENARIO-CODE-005
"""

from __future__ import annotations

import builtins
import json
import subprocess
from types import SimpleNamespace

import carnot.verify.python_types as python_types
import pytest
from carnot.verify import sandbox
from carnot.verify.python_types import safe_exec_function

CORRECT_ADD = "def add(a: int, b: int) -> int:\n    return a + b"


def _completed(
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


class TestSafeExecFunctionSandbox:
    """Regression tests for safe_exec_function sandbox delegation."""

    def test_import_error_falls_back_to_in_process_exec(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SCENARIO-CODE-005: opt-in sandbox import failures fall back safely."""

        original_import = builtins.__import__

        def fake_import(
            name: str,
            globalns: object | None = None,
            localns: object | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "carnot.verify.sandbox":
                raise ImportError("sandbox unavailable")
            return original_import(name, globalns, localns, fromlist, level)

        monkeypatch.setenv("CARNOT_USE_SANDBOX", "1")
        monkeypatch.setattr(builtins, "__import__", fake_import)

        result, error = safe_exec_function(CORRECT_ADD, "add", (1, 2))

        assert result == 3
        assert error is None

    def test_require_sandbox_without_runtime_returns_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-CODE-001: sandbox enforcement reports an unavailable runtime."""

        monkeypatch.setenv("CARNOT_REQUIRE_SANDBOX", "1")
        monkeypatch.setattr(sandbox, "_gvisor_available", lambda: False)

        result, error = safe_exec_function(CORRECT_ADD, "add", (1, 2))

        assert result is None
        assert isinstance(error, RuntimeError)
        assert "CARNOT_REQUIRE_SANDBOX=1" in str(error)

    def test_available_runtime_delegates_to_sandbox(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-CODE-001: opt-in sandbox mode delegates when gVisor is available."""

        calls: list[tuple[str, str, tuple[object, ...], float, bool]] = []

        def fake_sandbox_exec(
            code: str,
            func_name: str,
            args: tuple[object, ...],
            timeout: float = 1.0,
            *,
            allow_fallback: bool = True,
        ) -> tuple[object, Exception | None]:
            calls.append((code, func_name, args, timeout, allow_fallback))
            return ["ok"], None

        monkeypatch.setenv("CARNOT_USE_SANDBOX", "1")
        monkeypatch.delenv("CARNOT_REQUIRE_SANDBOX", raising=False)
        monkeypatch.setattr(sandbox, "_gvisor_available", lambda: True)
        monkeypatch.setattr(sandbox, "sandboxed_exec_function", fake_sandbox_exec)

        result, error = safe_exec_function(CORRECT_ADD, "add", (1, 2), timeout=2.5)

        assert result == ["ok"]
        assert error is None
        assert calls == [(CORRECT_ADD, "add", (1, 2), 2.5, True)]


class TestGvisorAvailability:
    """Coverage for sandbox runtime detection helpers."""

    def test_returns_false_when_docker_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-CODE-001: no docker binary means no sandbox runtime."""

        monkeypatch.setattr(sandbox.shutil, "which", lambda _: None)
        assert sandbox._gvisor_available() is False

    def test_returns_true_when_runtime_registered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-CODE-001: docker info including runsc enables sandboxing."""

        monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/docker")
        monkeypatch.setattr(
            sandbox.subprocess,
            "run",
            lambda *args, **kwargs: _completed(stdout="map[runc:{} runsc:{}]"),
        )
        assert sandbox._gvisor_available() is True

    def test_handles_runtime_probe_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCENARIO-CODE-005: runtime probe failures degrade cleanly."""

        monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/docker")

        def raise_oserror(*args: object, **kwargs: object) -> object:
            raise OSError("docker info failed")

        monkeypatch.setattr(sandbox.subprocess, "run", raise_oserror)
        assert sandbox._gvisor_available() is False


class TestSandboxedExecFunction:
    """Result decoding and sandbox execution branches."""

    def test_rejects_oversized_code(self) -> None:
        """REQ-CODE-001: absurdly large code is rejected before execution."""

        code = "x" * (sandbox._MAX_CODE_SIZE_BYTES + 1)
        result, error = sandbox.sandboxed_exec_function(code, "f", (), force_sandbox=True)

        assert result is None
        assert isinstance(error, ValueError)
        assert "byte limit" in str(error)

    def test_unavailable_runtime_falls_back_when_allowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-CODE-001: sandbox unavailability can fall back to local exec."""

        monkeypatch.setattr(sandbox, "_gvisor_available", lambda: False)
        monkeypatch.setattr(
            python_types,
            "safe_exec_function",
            lambda code, func_name, args, timeout=1.0: ("fallback", None),
        )

        result, error = sandbox.sandboxed_exec_function("code", "func", (), allow_fallback=True)

        assert result == "fallback"
        assert error is None

    def test_unavailable_runtime_raises_when_fallback_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-CODE-001: callers can require sandbox isolation."""

        monkeypatch.setattr(sandbox, "_gvisor_available", lambda: False)

        with pytest.raises(RuntimeError, match="gvisor sandbox required but unavailable"):
            sandbox.sandboxed_exec_function("code", "func", (), allow_fallback=False)

    def test_success_decodes_python_literal_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-CODE-001: tuple-like results survive sandbox transport."""

        payload = json.dumps({"status": "ok", "result_repr": "(1, 2)"})
        monkeypatch.setattr(
            sandbox.subprocess,
            "run",
            lambda cmd, **kwargs: _completed(stdout=payload),
        )

        result, error = sandbox.sandboxed_exec_function(
            CORRECT_ADD,
            "add",
            (1, 2),
            force_sandbox=True,
        )

        assert result == (1, 2)
        assert error is None

    def test_success_accepts_non_string_json_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-CODE-001: non-string JSON payloads pass through unchanged."""

        payload = json.dumps({"status": "ok", "result": 7})
        monkeypatch.setattr(
            sandbox.subprocess,
            "run",
            lambda cmd, **kwargs: _completed(stdout=payload),
        )

        result, error = sandbox.sandboxed_exec_function("code", "func", (), force_sandbox=True)

        assert result == 7
        assert error is None

    def test_success_keeps_non_literal_repr_as_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SCENARIO-CODE-005: non-literal reprs degrade to text instead of crashing."""

        payload = json.dumps({"status": "ok", "result_repr": "<Thing object>"})
        monkeypatch.setattr(
            sandbox.subprocess,
            "run",
            lambda cmd, **kwargs: _completed(stdout=payload),
        )

        result, error = sandbox.sandboxed_exec_function("code", "func", (), force_sandbox=True)

        assert result == "<Thing object>"
        assert error is None

    def test_apply_mutated_args_updates_caller_owned_mutables(self) -> None:
        """REQ-CODE-001: sandbox transport preserves in-place mutations."""

        nested = [0]
        items = [1]
        mapping = {"a": 1}
        members = {1}

        sandbox._apply_mutated_args(
            (items, mapping, members, (nested,)),
            repr([[2, 3], {"b": 2}, {3, 4}, ([9],)]),
        )

        assert items == [2, 3]
        assert mapping == {"b": 2}
        assert members == {3, 4}
        assert nested == [9]

    def test_timeout_returns_timeout_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCENARIO-CODE-005: timed-out containers report TimeoutError."""

        def raise_timeout(*args: object, **kwargs: object) -> object:
            raise subprocess.TimeoutExpired(cmd="docker run", timeout=1.0)

        monkeypatch.setattr(sandbox.subprocess, "run", raise_timeout)

        result, error = sandbox.sandboxed_exec_function("code", "func", (), force_sandbox=True)

        assert result is None
        assert isinstance(error, TimeoutError)

    def test_empty_stdout_returns_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCENARIO-CODE-005: silent containers surface a useful error."""

        monkeypatch.setattr(
            sandbox.subprocess,
            "run",
            lambda cmd, **kwargs: _completed(stderr="nothing happened"),
        )

        result, error = sandbox.sandboxed_exec_function("code", "func", (), force_sandbox=True)

        assert result is None
        assert isinstance(error, RuntimeError)
        assert "no output" in str(error)

    def test_invalid_json_returns_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SCENARIO-CODE-005: malformed stdout is reported cleanly."""

        monkeypatch.setattr(
            sandbox.subprocess,
            "run",
            lambda cmd, **kwargs: _completed(stdout="not json"),
        )

        result, error = sandbox.sandboxed_exec_function("code", "func", (), force_sandbox=True)

        assert result is None
        assert isinstance(error, RuntimeError)
        assert "not valid JSON" in str(error)

    def test_error_status_reconstructs_builtin_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SCENARIO-CODE-005: builtin exception types survive sandbox transport."""

        payload = json.dumps({"status": "error", "error_type": "ValueError", "error_msg": "boom"})
        monkeypatch.setattr(
            sandbox.subprocess,
            "run",
            lambda cmd, **kwargs: _completed(stdout=payload),
        )

        result, error = sandbox.sandboxed_exec_function("code", "func", (), force_sandbox=True)

        assert result is None
        assert isinstance(error, ValueError)
        assert str(error) == "boom"

    def test_error_status_uses_runtime_error_for_unknown_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SCENARIO-CODE-005: unknown exception names degrade to RuntimeError."""

        payload = json.dumps({"status": "error", "error_type": "CustomOops", "error_msg": "boom"})
        monkeypatch.setattr(
            sandbox.subprocess,
            "run",
            lambda cmd, **kwargs: _completed(stdout=payload),
        )

        result, error = sandbox.sandboxed_exec_function("code", "func", (), force_sandbox=True)

        assert result is None
        assert isinstance(error, RuntimeError)
        assert str(error) == "boom"

    def test_unexpected_status_returns_runtime_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SCENARIO-CODE-005: unknown status payloads are rejected."""

        payload = json.dumps({"status": "weird"})
        monkeypatch.setattr(
            sandbox.subprocess,
            "run",
            lambda cmd, **kwargs: _completed(stdout=payload),
        )

        result, error = sandbox.sandboxed_exec_function("code", "func", (), force_sandbox=True)

        assert result is None
        assert isinstance(error, RuntimeError)
        assert "Unexpected sandbox output" in str(error)

    def test_status_reports_configuration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-CODE-001: status exposes runtime configuration for callers."""

        monkeypatch.setattr(sandbox, "_gvisor_available", lambda: True)
        monkeypatch.setattr(sandbox.shutil, "which", lambda _: "/usr/bin/docker")

        status = sandbox.get_sandbox_status()

        assert status["available"] is True
        assert status["runtime"] == "runsc"
        assert status["docker"] is True
        assert status["image"] == "python:3.11-slim"
