"""
Tests for scripts/experiment_962_preflight_v25.py

REQ-INFRA-072: exclusion manifest must contain all retired experiment IDs.
SCENARIO-PREFLIGHT-001: preflight script produces valid deliverable JSON.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Allow importing from scripts/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import experiment_962_preflight_v25 as preflight


class TestDiagnoseExp906:
    def test_missing_result_returns_not_found(self, tmp_path):
        # When no experiment_906 result file exists, diagnose_exp906 should
        # return found=False rather than raising an exception.
        with mock.patch.object(preflight, "RESULTS_DIR", tmp_path):
            result = preflight.diagnose_exp906()
        assert result["found"] is False
        assert result["root_cause"] == "no_result_file_found"

    def test_real_result_classified_as_scale_latency(self):
        # The actual exp 906 result file must yield the expected root-cause class.
        result = preflight.diagnose_exp906()
        assert result["found"] is True
        assert result["root_cause_class"] == "c_50q_scale_x_per_question_latency"
        # The experiment succeeded — should NOT be recommended for retirement.
        assert "strong_improvement" in result["honest_verdict_from_result"]


class TestCheckExp954:
    def test_returns_false_when_no_file(self, tmp_path):
        with mock.patch.object(preflight, "RESULTS_DIR", tmp_path):
            launched = preflight.check_exp954_launched()
        assert launched is False

    def test_returns_true_when_file_exists(self, tmp_path):
        (tmp_path / "experiment_954_fast_path.json").write_text("{}")
        with mock.patch.object(preflight, "RESULTS_DIR", tmp_path):
            launched = preflight.check_exp954_launched()
        assert launched is True


class TestCheckSotaModels:
    def test_missing_models_returns_false(self, tmp_path):
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            results = preflight.check_sota_models()
        assert all(not v for v in results.values())

    def test_present_models_returns_true(self, tmp_path):
        hf_hub = tmp_path / ".cache" / "huggingface" / "hub"
        for dir_name in preflight.SOTA_MODEL_CACHE_PATTERNS.values():
            (hf_hub / dir_name).mkdir(parents=True)
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            results = preflight.check_sota_models()
        assert all(results.values())


class TestVerifyExclusionManifest:
    def test_missing_file_returns_false(self, tmp_path):
        with mock.patch.object(preflight, "EXCLUSION_MANIFEST_PATH", tmp_path / "missing.yaml"):
            ok, found = preflight.verify_exclusion_manifest()
        assert ok is False
        assert found == []

    def test_all_entries_present(self, tmp_path):
        manifest = tmp_path / "exclusion_manifest.yaml"
        manifest.write_text(
            "retired:\n"
            "  - experiment_id: 786\n"
            "  - experiment_id: 627\n"
            "  - experiment_id: 603\n"
            "  - experiment_id: 641\n"
        )
        with mock.patch.object(preflight, "EXCLUSION_MANIFEST_PATH", manifest):
            ok, found = preflight.verify_exclusion_manifest()
        assert ok is True
        assert set(found) == {786, 627, 603, 641}

    def test_partial_entries_returns_false(self, tmp_path):
        manifest = tmp_path / "exclusion_manifest.yaml"
        manifest.write_text("retired:\n  - experiment_id: 786\n")
        with mock.patch.object(preflight, "EXCLUSION_MANIFEST_PATH", manifest):
            ok, found = preflight.verify_exclusion_manifest()
        assert ok is False
        assert 786 in found
        assert 627 not in found


class TestMainDeliverable:
    def test_deliverable_schema_complete(self, tmp_path):
        # Run main() against a temp results dir and verify all required fields
        # are present in the output JSON.
        deliverable = tmp_path / "experiment_962_preflight_v25.json"
        with (
            mock.patch.object(preflight, "RESULTS_DIR", tmp_path),
            mock.patch.object(preflight, "DELIVERABLE", str(deliverable)),
        ):
            preflight.main()

        assert deliverable.exists(), "deliverable JSON was not written"
        with open(deliverable) as f:
            artifact = json.load(f)

        required_fields = [
            "exp906_root_cause",
            "exp906_fix_applied",
            "exp954_never_launched",
            "sota_models_ready",
            "manifest_verified",
            "honest_verdict",
        ]
        for field in required_fields:
            assert field in artifact, f"missing required field: {field}"

        assert artifact["honest_verdict"] in ("preflight_complete", "preflight_partial")
