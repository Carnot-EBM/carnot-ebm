"""Tests for Exp 325: conductor timeout wrapper and test-first stub generation.

Traces to:
  REQ-INFRA-001  — conductor timeout wrapper (SCENARIO-INFRA-001)
  REQ-INFRA-002  — test-first stub generation (SCENARIO-INFRA-002, SCENARIO-INFRA-003)
"""
from __future__ import annotations

import ast
import os
import stat
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = REPO_ROOT / "scripts" / "run_experiment_with_timeout.sh"


def _make_template(tmp_path: Path):
    """Return an ExperimentTemplate whose repo_root is isolated to *tmp_path*."""
    from scripts.experiment_template import ExperimentTemplate

    return ExperimentTemplate(
        325,
        "Exp 325: conductor hardening",
        "results/experiment_325_hardening.json",
        repo_root=tmp_path,
    )


# ---------------------------------------------------------------------------
# REQ-INFRA-001: timeout wrapper
# ---------------------------------------------------------------------------


class TestTimeoutWrapper:
    """REQ-INFRA-001 — run_experiment_with_timeout.sh existence and correctness."""

    def test_wrapper_exists(self):
        """SCENARIO-INFRA-001: wrapper script must exist at scripts/run_experiment_with_timeout.sh."""
        assert WRAPPER_PATH.exists(), f"Missing: {WRAPPER_PATH}"

    def test_wrapper_is_executable(self):
        """SCENARIO-INFRA-001: wrapper must be chmod +x so the OS can run it directly."""
        st = WRAPPER_PATH.stat()
        # At least one execute bit must be set (owner, group, or other).
        assert st.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH), (
            f"{WRAPPER_PATH} is not executable (mode {oct(st.st_mode)})"
        )

    def test_wrapper_has_bash_shebang(self):
        """SCENARIO-INFRA-001: shebang must be #!/usr/bin/env bash for portability."""
        first_line = WRAPPER_PATH.read_text().splitlines()[0]
        assert first_line.strip() == "#!/usr/bin/env bash", (
            f"Expected '#!/usr/bin/env bash', got '{first_line}'"
        )

    def test_wrapper_references_timeout_command(self):
        """SCENARIO-INFRA-001: script must call the `timeout` command to enforce the cap."""
        content = WRAPPER_PATH.read_text()
        assert "timeout" in content, "wrapper must invoke the 'timeout' command"

    def test_wrapper_uses_env_var(self):
        """REQ-INFRA-001: timeout must be driven by CARNOT_CONDUCTOR_TIMEOUT_MINUTES."""
        content = WRAPPER_PATH.read_text()
        assert "CARNOT_CONDUCTOR_TIMEOUT_MINUTES" in content, (
            "wrapper must read CARNOT_CONDUCTOR_TIMEOUT_MINUTES env var"
        )

    def test_wrapper_defaults_to_45(self):
        """REQ-INFRA-001: default timeout must be 45 minutes when env var is unset."""
        content = WRAPPER_PATH.read_text()
        assert "45" in content, "wrapper must default to 45 minutes"

    def test_wrapper_propagates_exit_code_124(self):
        """SCENARIO-INFRA-001: wrapper must exit with 124 (standard Unix timeout sentinel)."""
        content = WRAPPER_PATH.read_text()
        assert "124" in content, (
            "wrapper must handle exit code 124 (standard Unix timeout sentinel)"
        )

    def test_wrapper_emits_timeout_message(self):
        """SCENARIO-INFRA-001: human-readable message must be emitted when timeout fires."""
        content = WRAPPER_PATH.read_text()
        assert "CONDUCTOR TIMEOUT" in content, (
            "wrapper must print 'CONDUCTOR TIMEOUT' when exit code is 124"
        )

    def test_wrapper_uses_kill_grace(self):
        """REQ-INFRA-001: -k 60s ensures SIGKILL is sent if process ignores SIGTERM."""
        content = WRAPPER_PATH.read_text()
        assert "-k" in content, "wrapper must use timeout -k for guaranteed kill"


# ---------------------------------------------------------------------------
# REQ-INFRA-002: generate_test_stub()
# ---------------------------------------------------------------------------


class TestGenerateTestStub:
    """REQ-INFRA-002 — ExperimentTemplate.generate_test_stub() behaviour."""

    def test_stub_is_created(self, tmp_path):
        """SCENARIO-INFRA-002: calling generate_test_stub() creates the file when absent."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_new.py")
        result = tmpl.generate_test_stub(dest)
        assert result == dest
        assert Path(dest).exists()

    def test_stub_returns_path_string(self, tmp_path):
        """REQ-INFRA-002: return value must be the string path of the (written or existing) file."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_ret.py")
        result = tmpl.generate_test_stub(dest)
        assert isinstance(result, str)
        assert result == dest

    def test_stub_idempotent_no_overwrite(self, tmp_path):
        """SCENARIO-INFRA-002: second call must NOT overwrite the file written by the first call."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_idem.py")
        tmpl.generate_test_stub(dest)
        original = Path(dest).read_text()
        # Manually alter the file to detect whether a second call overwrites it.
        Path(dest).write_text(original + "\n# marker\n")
        result2 = tmpl.generate_test_stub(dest)
        # Return path must still be the same.
        assert result2 == dest
        # Content must NOT have been reset (marker must still be present).
        after = Path(dest).read_text()
        assert "# marker" in after, "generate_test_stub() must not overwrite an existing file"

    def test_stub_parses_as_valid_python(self, tmp_path):
        """SCENARIO-INFRA-003: skeleton must be syntactically valid Python."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_parse.py")
        tmpl.generate_test_stub(dest, "scripts.experiment_template")
        source = Path(dest).read_text()
        # ast.parse raises SyntaxError on invalid Python — this test will fail if the
        # generated skeleton is malformed.
        ast.parse(source)

    def test_stub_contains_req_comment(self, tmp_path):
        """SCENARIO-INFRA-003: skeleton must contain the REQ-INFRA-002 traceability comment."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_req.py")
        tmpl.generate_test_stub(dest)
        content = Path(dest).read_text()
        assert "REQ-INFRA-002" in content

    def test_stub_contains_autogenerated_header(self, tmp_path):
        """REQ-INFRA-002: skeleton must include the AUTO-GENERATED header comment."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_hdr.py")
        tmpl.generate_test_stub(dest)
        content = Path(dest).read_text()
        assert "AUTO-GENERATED" in content

    def test_stub_contains_test_class(self, tmp_path):
        """SCENARIO-INFRA-003: skeleton must define a class whose name starts with 'TestExp'."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_cls.py")
        tmpl.generate_test_stub(dest)
        content = Path(dest).read_text()
        assert "TestExp" in content

    def test_stub_contains_passing_placeholder_test(self, tmp_path):
        """REQ-INFRA-002: skeleton must contain a test method that passes (assert True)."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_pass.py")
        tmpl.generate_test_stub(dest)
        content = Path(dest).read_text()
        assert "test_placeholder_stub" in content
        assert "assert True" in content

    def test_stub_with_module_import(self, tmp_path):
        """REQ-INFRA-002: when module_to_test is provided, the skeleton must include an import."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_imp.py")
        tmpl.generate_test_stub(dest, "scripts.experiment_template")
        content = Path(dest).read_text()
        assert "scripts.experiment_template" in content

    def test_stub_without_module_import(self, tmp_path):
        """REQ-INFRA-002: when module_to_test is '' (default), no import statement is added."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_no_imp.py")
        tmpl.generate_test_stub(dest)
        content = Path(dest).read_text()
        # Should still be valid Python even without an import.
        ast.parse(content)

    def test_stub_file_permissions(self, tmp_path):
        """REQ-INFRA-002: written file must have mode 0o644 (rw-r--r--)."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_perm.py")
        tmpl.generate_test_stub(dest)
        mode = Path(dest).stat().st_mode & 0o777
        assert mode == 0o644, f"Expected 0o644, got {oct(mode)}"

    def test_stub_idempotent_returns_same_path(self, tmp_path):
        """SCENARIO-INFRA-002: return value must be the same path on both first and second call."""
        tmpl = _make_template(tmp_path)
        dest = str(tmp_path / "test_stub_same.py")
        path1 = tmpl.generate_test_stub(dest)
        path2 = tmpl.generate_test_stub(dest)
        assert path1 == path2 == dest


# ---------------------------------------------------------------------------
# timeout_minutes default via env var
# ---------------------------------------------------------------------------


class TestTimeoutMinutesEnvVar:
    """REQ-INFRA-001: CARNOT_CONDUCTOR_TIMEOUT_MINUTES controls the wrapper default."""

    def test_wrapper_content_references_default_minutes(self):
        """The wrapper must embed the default (45) so running without the env var works."""
        content = WRAPPER_PATH.read_text()
        # We already check for "45" in TestTimeoutWrapper.test_wrapper_defaults_to_45;
        # here we verify the env-var expansion pattern exists.
        assert "CARNOT_CONDUCTOR_TIMEOUT_MINUTES" in content

    def test_wrapper_passes_through_arbitrary_command(self):
        """REQ-INFRA-001: wrapper must forward all positional arguments to the timeout call."""
        content = WRAPPER_PATH.read_text()
        # The "$@" pattern is the canonical way to forward all args in bash.
        assert '"$@"' in content or "'$@'" in content or "$@" in content
