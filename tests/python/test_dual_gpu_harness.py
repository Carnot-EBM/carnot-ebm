"""Tests for DualGPUHarness and HarnessAudit.

Spec: REQ-INFRA-045, REQ-INFRA-046,
      SCENARIO-INFRA-053, SCENARIO-INFRA-054
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot.pipeline.dual_gpu_harness import AuditFinding, DualGPUHarness, HarnessAudit


# ---------------------------------------------------------------------------
# DualGPUHarness — eligibility
# ---------------------------------------------------------------------------


class TestDualGPUHarnessEligibility:
    """REQ-INFRA-045: DualGPUHarness.is_eligible gates apply()."""

    def test_not_eligible_in_ci_mode(self) -> None:
        """SCENARIO-INFRA-053: CI (live_mode=False) is never eligible."""
        h = DualGPUHarness(n_gpus=2, live_mode=False)
        assert h.is_eligible is False

    def test_not_eligible_single_gpu(self) -> None:
        """n_gpus=1 with live_mode=True is not eligible — need at least 2."""
        h = DualGPUHarness(n_gpus=1, live_mode=True)
        assert h.is_eligible is False

    def test_not_eligible_zero_gpus(self) -> None:
        """n_gpus=0 with live_mode=True is not eligible."""
        h = DualGPUHarness(n_gpus=0, live_mode=True)
        assert h.is_eligible is False

    def test_eligible_two_gpus_live(self) -> None:
        """n_gpus=2 and live_mode=True is eligible."""
        h = DualGPUHarness(n_gpus=2, live_mode=True)
        assert h.is_eligible is True

    def test_eligible_four_gpus_live(self) -> None:
        """n_gpus=4 and live_mode=True is also eligible."""
        h = DualGPUHarness(n_gpus=4, live_mode=True)
        assert h.is_eligible is True


# ---------------------------------------------------------------------------
# DualGPUHarness — apply() when eligible
# ---------------------------------------------------------------------------


class TestDualGPUHarnessApplyEligible:
    """SCENARIO-INFRA-053: apply() assigns cuda:0 to first, cuda:1 to second model."""

    def _make_specs(self, n: int) -> list[dict]:
        return [{"name": f"model_{i}", "hf_id": f"org/model_{i}"} for i in range(n)]

    def test_apply_assigns_cuda0_to_first(self) -> None:
        """SCENARIO-INFRA-053: first model gets gpu=0 and device_map cuda:0."""
        h = DualGPUHarness(n_gpus=2, live_mode=True)
        specs = self._make_specs(2)
        result = h.apply(specs)
        assert result[0]["gpu"] == 0
        assert result[0]["device_map"] == {"": "cuda:0"}

    def test_apply_assigns_cuda1_to_second(self) -> None:
        """SCENARIO-INFRA-053: second model gets gpu=1 and device_map cuda:1."""
        h = DualGPUHarness(n_gpus=2, live_mode=True)
        specs = self._make_specs(2)
        result = h.apply(specs)
        assert result[1]["gpu"] == 1
        assert result[1]["device_map"] == {"": "cuda:1"}

    def test_apply_returns_same_list(self) -> None:
        """apply() mutates in-place and returns the same list object."""
        h = DualGPUHarness(n_gpus=2, live_mode=True)
        specs = self._make_specs(2)
        result = h.apply(specs)
        assert result is specs

    def test_apply_preserves_existing_keys(self) -> None:
        """apply() does not remove existing keys from specs."""
        h = DualGPUHarness(n_gpus=2, live_mode=True)
        specs = [{"name": "A", "hf_id": "org/A", "extra": 42}, {"name": "B", "hf_id": "org/B"}]
        h.apply(specs)
        assert specs[0]["extra"] == 42

    def test_apply_three_models_last_gpu_overflow(self) -> None:
        """Third model with n_gpus=2 overflows to GPU 1 (last available)."""
        h = DualGPUHarness(n_gpus=2, live_mode=True)
        specs = self._make_specs(3)
        h.apply(specs)
        assert specs[2]["gpu"] == 1
        assert specs[2]["device_map"] == {"": "cuda:1"}

    def test_apply_single_model_eligible(self) -> None:
        """Single model with eligible harness gets gpu=0."""
        h = DualGPUHarness(n_gpus=2, live_mode=True)
        specs = [{"name": "solo", "hf_id": "org/solo"}]
        h.apply(specs)
        assert specs[0]["gpu"] == 0


# ---------------------------------------------------------------------------
# DualGPUHarness — apply() when NOT eligible
# ---------------------------------------------------------------------------


class TestDualGPUHarnessApplyNotEligible:
    """apply() is a no-op when not eligible."""

    def test_apply_noop_in_ci(self) -> None:
        """SCENARIO-INFRA-053: CI mode — apply() returns specs unchanged."""
        h = DualGPUHarness(n_gpus=2, live_mode=False)
        specs = [{"name": "A"}, {"name": "B"}]
        original = [dict(s) for s in specs]
        result = h.apply(specs)
        assert result is specs
        assert result[0] == original[0]
        assert result[1] == original[1]

    def test_apply_noop_single_gpu(self) -> None:
        """n_gpus=1 — apply() returns specs unchanged."""
        h = DualGPUHarness(n_gpus=1, live_mode=True)
        specs = [{"name": "A"}, {"name": "B"}]
        original = [dict(s) for s in specs]
        h.apply(specs)
        assert specs[0] == original[0]
        assert specs[1] == original[1]

    def test_apply_empty_specs_noop(self) -> None:
        """Empty specs list returns empty list unchanged."""
        h = DualGPUHarness(n_gpus=2, live_mode=False)
        result = h.apply([])
        assert result == []


# ---------------------------------------------------------------------------
# DualGPUHarness — from_env() factory
# ---------------------------------------------------------------------------


class TestDualGPUHarnessFromEnv:
    """DualGPUHarness.from_env() reads CARNOT_FORCE_LIVE from environment."""

    def test_from_env_ci_mode(self, monkeypatch) -> None:
        """CARNOT_FORCE_LIVE not set → live_mode=False, n_gpus=0."""
        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        h = DualGPUHarness.from_env()
        assert h._live_mode is False
        assert h._n_gpus == 0
        assert h.is_eligible is False

    def test_from_env_live_mode_no_torch(self, monkeypatch) -> None:
        """CARNOT_FORCE_LIVE=1 but torch not importable → n_gpus=0."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        import sys

        # Temporarily remove torch so import raises ImportError
        orig = sys.modules.pop("torch", None)
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            h = DualGPUHarness.from_env()
            assert h._live_mode is True
            assert h._n_gpus == 0
        finally:
            sys.modules.pop("torch", None)
            if orig is not None:
                sys.modules["torch"] = orig

    def test_from_env_live_mode_torch_available(self, monkeypatch) -> None:
        """CARNOT_FORCE_LIVE=1 and torch importable → n_gpus from device_count()."""
        import sys
        from types import SimpleNamespace
        import unittest.mock as mock

        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2))
        orig = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch  # type: ignore[assignment]
        try:
            h = DualGPUHarness.from_env()
            assert h._live_mode is True
            assert h._n_gpus == 2
            assert h.is_eligible is True
        finally:
            sys.modules.pop("torch", None)
            if orig is not None:
                sys.modules["torch"] = orig


# ---------------------------------------------------------------------------
# AuditFinding dataclass
# ---------------------------------------------------------------------------


class TestAuditFinding:
    """AuditFinding is a plain dataclass with the expected fields."""

    def test_needs_fix_true_when_dual_load_no_cuda1(self) -> None:
        """needs_fix=True when has_dual_model_load=True and has_cuda1_assignment=False."""
        f = AuditFinding(
            script_path="/scripts/exp_foo.py",
            has_dual_model_load=True,
            has_cuda1_assignment=False,
            needs_fix=True,
        )
        assert f.needs_fix is True

    def test_needs_fix_false_when_cuda1_present(self) -> None:
        """needs_fix=False when cuda:1 is already assigned."""
        f = AuditFinding(
            script_path="/scripts/exp_bar.py",
            has_dual_model_load=True,
            has_cuda1_assignment=True,
            needs_fix=False,
        )
        assert f.needs_fix is False


# ---------------------------------------------------------------------------
# HarnessAudit — scan()
# ---------------------------------------------------------------------------


class TestHarnessAuditScan:
    """REQ-INFRA-046: HarnessAudit.scan() flags scripts with dual loads missing cuda:1."""

    def _write_script(self, tmp_path: Path, name: str, content: str) -> Path:
        f = tmp_path / name
        f.write_text(content)
        return f

    def test_scan_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-054: empty scripts dir returns empty list."""
        audit = HarnessAudit(str(tmp_path))
        assert audit.scan() == []

    def test_scan_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-054: non-existent dir returns empty list with warning."""
        audit = HarnessAudit(str(tmp_path / "does_not_exist"))
        assert audit.scan() == []

    def test_scan_single_model_no_cuda1_not_flagged(self, tmp_path: Path) -> None:
        """Single-model script is never needs_fix (only one model)."""
        self._write_script(
            tmp_path,
            "exp_single.py",
            'MODEL_SPECS = [{"name": "A", "hf_id": "org/A"}]\n',
        )
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        dual = [f for f in findings if f.needs_fix]
        assert dual == []

    def test_scan_dual_model_no_cuda1_flagged(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-054: dual-model script without cuda:1 has needs_fix=True."""
        self._write_script(
            tmp_path,
            "exp_dual.py",
            'MODEL_SPECS = [\n'
            '    {"name": "A", "hf_id": "org/A"},\n'
            '    {"name": "B", "hf_id": "org/B"},\n'
            ']\n',
        )
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        assert len(findings) == 1
        assert findings[0].has_dual_model_load is True
        assert findings[0].has_cuda1_assignment is False
        assert findings[0].needs_fix is True

    def test_scan_dual_model_with_cuda1_not_flagged(self, tmp_path: Path) -> None:
        """Dual-model script that already has cuda:1 does not need a fix."""
        self._write_script(
            tmp_path,
            "exp_good.py",
            'MODEL_SPECS = [\n'
            '    {"name": "A", "hf_id": "org/A", "gpu": 0},\n'
            '    {"name": "B", "hf_id": "org/B", "gpu": 1},\n'
            ']\n'
            'device = "cuda:1"\n',
        )
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        assert len(findings) == 1
        assert findings[0].has_cuda1_assignment is True
        assert findings[0].needs_fix is False

    def test_scan_no_model_script_excluded(self, tmp_path: Path) -> None:
        """Non-harness scripts (no hf_id, no _load_* calls) are excluded from findings."""
        self._write_script(
            tmp_path,
            "utility.py",
            "def helper():\n    return 42\n",
        )
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        assert findings == []

    def test_scan_load_fn_calls_detected(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-054: _load_* function calls count as model loads."""
        self._write_script(
            tmp_path,
            "exp_loaders.py",
            "def main():\n"
            "    model_a = _load_gemma(gpu_index=0)\n"
            "    model_b = _load_qwen(hf_id='org/Q', gpu_index=0)\n",
        )
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        assert len(findings) == 1
        assert findings[0].has_dual_model_load is True
        assert findings[0].needs_fix is True

    def test_scan_returns_script_path(self, tmp_path: Path) -> None:
        """Findings include the full script path."""
        script = self._write_script(
            tmp_path,
            "exp_path_check.py",
            'x = {"hf_id": "org/A"}\ny = {"hf_id": "org/B"}\n',
        )
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        assert any(str(script) in f.script_path for f in findings)

    def test_scan_unreadable_file_skipped(self, tmp_path: Path, monkeypatch) -> None:
        """Files that cannot be read are skipped without crashing."""
        script = self._write_script(
            tmp_path,
            "exp_unreadable.py",
            'x = {"hf_id": "org/A"}\ny = {"hf_id": "org/B"}\n',
        )
        # Simulate read failure by patching Path.read_text
        original_read_text = Path.read_text

        def mock_read_text(self, **kwargs):
            if self.name == "exp_unreadable.py":
                raise OSError("Permission denied")
            return original_read_text(self, **kwargs)

        monkeypatch.setattr(Path, "read_text", mock_read_text)
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()  # must not raise
        assert all(str(script) not in f.script_path for f in findings)

    def test_scan_method_call_load_fn_detected(self, tmp_path: Path) -> None:
        """_load_* as attribute calls (obj._load_something()) also count."""
        self._write_script(
            tmp_path,
            "exp_method_load.py",
            "class Loader:\n"
            "    def run(self):\n"
            "        self._load_model_a()\n"
            "        self._load_model_b()\n",
        )
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        assert len(findings) == 1
        assert findings[0].has_dual_model_load is True
        assert findings[0].needs_fix is True

    def test_scan_syntax_error_counts_zero_load_fns(self, tmp_path: Path) -> None:
        """A file with a syntax error falls back to hf_id heuristic only."""
        self._write_script(
            tmp_path,
            "exp_syntax_err.py",
            'hf_id = "org/A"\nhf_id = "org/B"\ndef broken(\n',  # unclosed paren = SyntaxError
        )
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        # hf_id appears twice, so dual_model=True; no cuda:1 so needs_fix=True
        assert len(findings) == 1
        assert findings[0].has_dual_model_load is True
        assert findings[0].needs_fix is True
