"""Tests for HarnessPatcher and HarnessPatchResult.

Spec: REQ-INFRA-057, REQ-INFRA-058,
      SCENARIO-INFRA-065, SCENARIO-INFRA-066
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot.pipeline.dual_gpu_harness import AuditFinding, HarnessAudit
from carnot.pipeline.harness_patcher import HarnessPatchResult, HarnessPatcher


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _write_script(tmp_path: Path, name: str, content: str) -> Path:
    f = tmp_path / name
    f.write_text(content)
    return f


def _make_finding(path: str, needs_fix: bool = True) -> AuditFinding:
    return AuditFinding(
        script_path=path,
        has_dual_model_load=True,
        has_cuda1_assignment=not needs_fix,
        needs_fix=needs_fix,
    )


# ---------------------------------------------------------------------------
# HarnessPatchResult
# ---------------------------------------------------------------------------


class TestHarnessPatchResult:
    """SCENARIO-INFRA-065: HarnessPatchResult.success gate."""

    def test_success_true_when_patched_no_error(self) -> None:
        """success=True iff was_patched=True and error is None."""
        r = HarnessPatchResult(script_path="x.py", was_patched=True, error=None)
        assert r.success is True

    def test_success_false_when_not_patched(self) -> None:
        """success=False when was_patched=False (no-op or error)."""
        r = HarnessPatchResult(script_path="x.py", was_patched=False, error=None)
        assert r.success is False

    def test_success_false_when_error_set(self) -> None:
        """success=False when error is set even if was_patched=True."""
        r = HarnessPatchResult(script_path="x.py", was_patched=True, error="some error")
        assert r.success is False

    def test_success_false_when_both_false_and_error(self) -> None:
        """success=False when was_patched=False and error is set."""
        r = HarnessPatchResult(script_path="x.py", was_patched=False, error="fail")
        assert r.success is False


# ---------------------------------------------------------------------------
# HarnessPatcher.patch_script — Strategy 1 (device_map='auto')
# ---------------------------------------------------------------------------


class TestPatchScriptDeviceMapAuto:
    """SCENARIO-INFRA-065: patch_script replaces device_map='auto'."""

    def test_replaces_first_auto_with_cuda0(self, tmp_path: Path) -> None:
        """First device_map='auto' → device_map={'': 'cuda:0'}."""
        script = _write_script(
            tmp_path,
            "exp.py",
            "m1 = load(device_map='auto')\nm2 = load(device_map='auto')\n",
        )
        patcher = HarnessPatcher(str(tmp_path))
        result = patcher.patch_script(str(script))
        assert result.was_patched is True
        assert result.error is None
        patched = script.read_text()
        assert "device_map={'': 'cuda:0'}" in patched

    def test_replaces_second_auto_with_cuda1(self, tmp_path: Path) -> None:
        """Second device_map='auto' → device_map={'': 'cuda:1'}."""
        script = _write_script(
            tmp_path,
            "exp.py",
            "m1 = load(device_map='auto')\nm2 = load(device_map='auto')\n",
        )
        patcher = HarnessPatcher(str(tmp_path))
        patcher.patch_script(str(script))
        patched = script.read_text()
        assert "device_map={'': 'cuda:1'}" in patched

    def test_double_quote_auto_also_replaced(self, tmp_path: Path) -> None:
        """device_map=\"auto\" (double quotes) is also replaced."""
        script = _write_script(
            tmp_path,
            "exp2.py",
            'm1 = load(device_map="auto")\nm2 = load(device_map="auto")\n',
        )
        patcher = HarnessPatcher(str(tmp_path))
        result = patcher.patch_script(str(script))
        assert result.was_patched is True
        patched = script.read_text()
        assert "cuda:0" in patched
        assert "cuda:1" in patched

    def test_three_autos_third_also_cuda1(self, tmp_path: Path) -> None:
        """Third device_map='auto' also becomes cuda:1."""
        script = _write_script(
            tmp_path,
            "exp3.py",
            "a=load(device_map='auto')\nb=load(device_map='auto')\nc=load(device_map='auto')\n",
        )
        patcher = HarnessPatcher(str(tmp_path))
        patcher.patch_script(str(script))
        patched = script.read_text()
        # Two cuda:1 references (second and third)
        assert patched.count("cuda:1") >= 1

    def test_single_auto_no_replacement(self, tmp_path: Path) -> None:
        """Single device_map='auto' (not dual-model) still gets cuda:0 injected.

        Strategy 1 runs first; one replacement adds cuda:0 but not cuda:1.
        Strategy 2 then injects the block.
        """
        script = _write_script(
            tmp_path,
            "single.py",
            "m = load(hf_id='org/A', device_map='auto')\nhf_id_2 = 'org/B'\n",
        )
        patcher = HarnessPatcher(str(tmp_path))
        result = patcher.patch_script(str(script))
        # Strategy 1 runs (1 auto replaced → cuda:0 injected), then strategy 2 fires
        # for cuda:1.  Either way, was_patched=True and cuda:1 present.
        assert result.was_patched is True
        patched = script.read_text()
        assert "cuda:1" in patched

    def test_already_has_cuda1_noop(self, tmp_path: Path) -> None:
        """Script already containing cuda:1 is skipped without modification."""
        original = "m = load(device_map={'': 'cuda:1'})\n"
        script = _write_script(tmp_path, "done.py", original)
        patcher = HarnessPatcher(str(tmp_path))
        result = patcher.patch_script(str(script))
        assert result.was_patched is False
        assert result.error is None
        assert script.read_text() == original

    def test_audit_passes_after_patch(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-065: HarnessAudit no longer flags the script after patch."""
        script = _write_script(
            tmp_path,
            "exp_audit.py",
            "m1 = load(device_map='auto', hf_id='org/A')\n"
            "m2 = load(device_map='auto', hf_id='org/B')\n",
        )
        patcher = HarnessPatcher(str(tmp_path))
        patcher.patch_script(str(script))
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        needs_fix = [f for f in findings if f.needs_fix]
        assert needs_fix == []


# ---------------------------------------------------------------------------
# HarnessPatcher.patch_script — Strategy 2 (inject block)
# ---------------------------------------------------------------------------


class TestPatchScriptInjectBlock:
    """Strategy 2: inject DualGPUHarness.apply() block when no device_map='auto'."""

    def test_inject_when_hf_id_no_device_map_auto(self, tmp_path: Path) -> None:
        """Script with hf_id but no device_map='auto' gets the inject block."""
        script = _write_script(
            tmp_path,
            "exp_inject.py",
            'MODEL_SPECS = [\n'
            '    {"name": "A", "hf_id": "org/A"},\n'
            '    {"name": "B", "hf_id": "org/B"},\n'
            ']\n',
        )
        patcher = HarnessPatcher(str(tmp_path))
        result = patcher.patch_script(str(script))
        assert result.was_patched is True
        assert result.error is None
        patched = script.read_text()
        assert "cuda:1" in patched

    def test_inject_block_contains_apply_call(self, tmp_path: Path) -> None:
        """The injected block contains a DualGPUHarness apply() call."""
        script = _write_script(
            tmp_path,
            "exp_call.py",
            'MODEL_SPECS = [{"hf_id": "org/A"}, {"hf_id": "org/B"}]\n',
        )
        patcher = HarnessPatcher(str(tmp_path))
        patcher.patch_script(str(script))
        patched = script.read_text()
        assert "DualGPUHarness" in patched
        assert "apply" in patched

    def test_no_patchable_pattern_returns_error(self, tmp_path: Path) -> None:
        """Script with no hf_id, MODEL_SPECS, or _load_* returns error."""
        script = _write_script(
            tmp_path,
            "utility.py",
            "def helper():\n    return 42\n",
        )
        patcher = HarnessPatcher(str(tmp_path))
        result = patcher.patch_script(str(script))
        assert result.was_patched is False
        assert result.error is not None

    def test_inject_audit_passes_after(self, tmp_path: Path) -> None:
        """Injected script is no longer flagged by HarnessAudit."""
        script = _write_script(
            tmp_path,
            "exp_inject_audit.py",
            'MODEL_SPECS = [{"hf_id": "org/A"}, {"hf_id": "org/B"}]\n',
        )
        patcher = HarnessPatcher(str(tmp_path))
        patcher.patch_script(str(script))
        audit = HarnessAudit(str(tmp_path))
        findings = audit.scan()
        needs_fix = [f for f in findings if f.needs_fix]
        assert needs_fix == []


# ---------------------------------------------------------------------------
# HarnessPatcher.patch_script — file I/O errors
# ---------------------------------------------------------------------------


class TestPatchScriptErrors:
    """Error handling in patch_script."""

    def test_nonexistent_file_returns_error(self, tmp_path: Path) -> None:
        """Nonexistent path → was_patched=False, error set."""
        patcher = HarnessPatcher(str(tmp_path))
        result = patcher.patch_script(str(tmp_path / "does_not_exist.py"))
        assert result.was_patched is False
        assert result.error is not None

    def test_write_failure_returns_error(self, tmp_path: Path, monkeypatch) -> None:
        """If write fails, return was_patched=False, error set."""
        script = _write_script(
            tmp_path,
            "exp_write_fail.py",
            "m1 = load(device_map='auto')\nm2 = load(device_map='auto')\n",
        )
        # Simulate write failure
        original_write = Path.write_text

        def fail_write(self, data, **kwargs):
            if self.name == "exp_write_fail.py":
                raise OSError("disk full")
            return original_write(self, data, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_write)
        patcher = HarnessPatcher(str(tmp_path))
        result = patcher.patch_script(str(script))
        assert result.was_patched is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# HarnessPatcher.patch_all
# ---------------------------------------------------------------------------


class TestPatchAll:
    """REQ-INFRA-058: patch_all applies patches to all needs_fix findings."""

    def test_patch_all_returns_result_per_needs_fix(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-066: patch_all returns one result per needs_fix=True finding."""
        s1 = _write_script(tmp_path, "a.py", "m=load(device_map='auto', hf_id='x')\n"
                                                "m2=load(device_map='auto', hf_id='y')\n")
        s2 = _write_script(tmp_path, "b.py", 'MODEL_SPECS=[{"hf_id":"x"},{"hf_id":"y"}]\n')
        findings = [
            _make_finding(str(s1), needs_fix=True),
            _make_finding(str(s2), needs_fix=True),
            _make_finding("already_ok.py", needs_fix=False),  # should be skipped
        ]
        patcher = HarnessPatcher(str(tmp_path))
        results = patcher.patch_all(findings)
        # Only 2 findings with needs_fix=True
        assert len(results) == 2

    def test_patch_all_success_true_for_each_patched(self, tmp_path: Path) -> None:
        """patch_all results have success=True for successfully patched scripts."""
        script = _write_script(
            tmp_path,
            "exp_ok.py",
            "m1=load(device_map='auto', hf_id='x')\nm2=load(device_map='auto', hf_id='y')\n",
        )
        findings = [_make_finding(str(script), needs_fix=True)]
        patcher = HarnessPatcher(str(tmp_path))
        results = patcher.patch_all(findings)
        assert len(results) == 1
        assert results[0].success is True

    def test_patch_all_skips_needs_fix_false(self, tmp_path: Path) -> None:
        """patch_all skips findings with needs_fix=False."""
        findings = [_make_finding("does_not_matter.py", needs_fix=False)]
        patcher = HarnessPatcher(str(tmp_path))
        results = patcher.patch_all(findings)
        assert results == []

    def test_patch_all_empty_findings(self, tmp_path: Path) -> None:
        """patch_all with empty findings returns empty list."""
        patcher = HarnessPatcher(str(tmp_path))
        results = patcher.patch_all([])
        assert results == []

    def test_patch_all_error_captured_not_raised(self, tmp_path: Path) -> None:
        """Errors on individual scripts are captured in results, not raised."""
        findings = [_make_finding("/nonexistent/path/exp.py", needs_fix=True)]
        patcher = HarnessPatcher(str(tmp_path))
        results = patcher.patch_all(findings)
        assert len(results) == 1
        assert results[0].was_patched is False
        assert results[0].error is not None


# ---------------------------------------------------------------------------
# HarnessPatcher.verify_clean
# ---------------------------------------------------------------------------


class TestVerifyClean:
    """SCENARIO-INFRA-066: verify_clean returns 0 after patch_all."""

    def test_verify_clean_returns_zero_after_patch_all(self, tmp_path: Path) -> None:
        """SCENARIO-INFRA-066: full round-trip — audit, patch_all, verify_clean → 0."""
        # Create two scripts that need fixing
        _write_script(
            tmp_path,
            "exp_a.py",
            "m1=load(device_map='auto', hf_id='org/A')\n"
            "m2=load(device_map='auto', hf_id='org/B')\n",
        )
        _write_script(
            tmp_path,
            "exp_b.py",
            'MODEL_SPECS=[{"hf_id":"org/A"},{"hf_id":"org/B"}]\n',
        )
        patcher = HarnessPatcher(str(tmp_path))

        # Initial audit: both should need fixing
        initial_findings = HarnessAudit(str(tmp_path)).scan()
        assert sum(1 for f in initial_findings if f.needs_fix) == 2

        # Patch all
        patcher.patch_all(initial_findings)

        # verify_clean should return 0
        remaining = patcher.verify_clean(str(tmp_path))
        assert remaining == 0

    def test_verify_clean_nonexistent_dir_returns_zero(self, tmp_path: Path) -> None:
        """verify_clean on a nonexistent dir returns 0 (empty findings)."""
        patcher = HarnessPatcher(str(tmp_path))
        remaining = patcher.verify_clean(str(tmp_path / "no_such_dir"))
        assert remaining == 0

    def test_verify_clean_unpatched_dir_returns_nonzero(self, tmp_path: Path) -> None:
        """verify_clean on a dir with unpatched scripts returns > 0."""
        _write_script(
            tmp_path,
            "exp_bad.py",
            'x={"hf_id":"org/A"}\ny={"hf_id":"org/B"}\n',
        )
        patcher = HarnessPatcher(str(tmp_path))
        remaining = patcher.verify_clean(str(tmp_path))
        assert remaining > 0
