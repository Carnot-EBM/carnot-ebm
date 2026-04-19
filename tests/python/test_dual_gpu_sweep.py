"""Tests for DualGPUSweepResult — Exp 505 retroactive harness sweep metrics.

Spec: REQ-INFRA-059, REQ-INFRA-060,
      SCENARIO-INFRA-067, SCENARIO-INFRA-068
"""

from carnot.pipeline.dual_gpu_sweep import DualGPUSweepResult


class TestDualGPUSweepResultPatchRate:
    """SCENARIO-INFRA-067: patch_rate reflects patched/found ratio."""

    def test_patch_rate_all_patched(self):
        # SCENARIO-INFRA-067: patch_rate=1.0 when all found scripts are patched.
        result = DualGPUSweepResult(
            n_scripts_found=3,
            n_scripts_patched=3,
            n_scripts_skipped=0,
            patch_manifest=["a.py", "b.py", "c.py"],
        )
        assert result.patch_rate == 1.0

    def test_patch_rate_partial(self):
        # SCENARIO-INFRA-068: patch_rate is fractional when some scripts are skipped.
        result = DualGPUSweepResult(
            n_scripts_found=5,
            n_scripts_patched=3,
            n_scripts_skipped=2,
            patch_manifest=["x.py", "y.py", "z.py"],
        )
        assert abs(result.patch_rate - 0.6) < 1e-9

    def test_patch_rate_zero_when_none_found(self):
        # REQ-INFRA-060: patch_rate=0.0 when n_scripts_found==0 (no ZeroDivisionError).
        result = DualGPUSweepResult(
            n_scripts_found=0,
            n_scripts_patched=0,
            n_scripts_skipped=0,
            patch_manifest=[],
        )
        assert result.patch_rate == 0.0

    def test_patch_rate_none_patched(self):
        # All found but zero patched (already covered by prior sweep).
        result = DualGPUSweepResult(
            n_scripts_found=4,
            n_scripts_patched=0,
            n_scripts_skipped=4,
            patch_manifest=[],
        )
        assert result.patch_rate == 0.0


class TestDualGPUSweepResultToDict:
    """REQ-INFRA-060 / SCENARIO-INFRA-067/068: to_dict() returns all fields."""

    def test_to_dict_all_patched(self):
        # SCENARIO-INFRA-067: to_dict contains patch_rate=1.0 and full manifest.
        result = DualGPUSweepResult(
            n_scripts_found=3,
            n_scripts_patched=3,
            n_scripts_skipped=0,
            patch_manifest=["a.py", "b.py", "c.py"],
        )
        d = result.to_dict()
        assert d["n_scripts_found"] == 3
        assert d["n_scripts_patched"] == 3
        assert d["n_scripts_skipped"] == 0
        assert d["patch_manifest"] == ["a.py", "b.py", "c.py"]
        assert d["patch_rate"] == 1.0

    def test_to_dict_partial_patch(self):
        # SCENARIO-INFRA-068: to_dict skipped=2, manifest only has patched scripts.
        result = DualGPUSweepResult(
            n_scripts_found=5,
            n_scripts_patched=3,
            n_scripts_skipped=2,
            patch_manifest=["x.py", "y.py", "z.py"],
        )
        d = result.to_dict()
        assert d["n_scripts_skipped"] == 2
        assert len(d["patch_manifest"]) == 3
        assert abs(d["patch_rate"] - 0.6) < 1e-9

    def test_to_dict_is_json_serializable(self):
        # REQ-INFRA-060: to_dict output must be JSON-serializable.
        import json

        result = DualGPUSweepResult(
            n_scripts_found=2,
            n_scripts_patched=1,
            n_scripts_skipped=1,
            patch_manifest=["experiment_999_test.py"],
        )
        payload = json.dumps(result.to_dict())
        parsed = json.loads(payload)
        assert parsed["n_scripts_patched"] == 1

    def test_to_dict_empty_manifest(self):
        # Edge case: empty sweep — no scripts found, manifest is empty list.
        result = DualGPUSweepResult(
            n_scripts_found=0,
            n_scripts_patched=0,
            n_scripts_skipped=0,
            patch_manifest=[],
        )
        d = result.to_dict()
        assert d["patch_manifest"] == []
        assert d["patch_rate"] == 0.0

    def test_to_dict_manifest_is_copy(self):
        # to_dict() returns a copy of patch_manifest, not the original list.
        manifest = ["a.py"]
        result = DualGPUSweepResult(
            n_scripts_found=1,
            n_scripts_patched=1,
            n_scripts_skipped=0,
            patch_manifest=manifest,
        )
        d = result.to_dict()
        d["patch_manifest"].append("injected.py")
        assert result.patch_manifest == ["a.py"]


class TestDualGPUSweepResultDefaults:
    """REQ-INFRA-060: default patch_manifest is an empty list."""

    def test_default_patch_manifest(self):
        # patch_manifest defaults to [] when not supplied.
        result = DualGPUSweepResult(
            n_scripts_found=0,
            n_scripts_patched=0,
            n_scripts_skipped=0,
        )
        assert result.patch_manifest == []
        assert result.to_dict()["patch_manifest"] == []
