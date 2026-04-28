"""
Tests for scripts/experiment_974_preflight_v26.py

REQ-INFRA-072: exclusion manifest must contain all retired experiment IDs.
SCENARIO-PREFLIGHT-001: preflight script produces valid deliverable JSON.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

import experiment_974_preflight_v26 as preflight


class TestVerifyYamlManifest:
    def test_missing_file_returns_empty(self, tmp_path):
        # When the YAML file does not exist, the function returns three empty lists
        # rather than raising an exception.
        with mock.patch.object(preflight, "EXCLUSION_MANIFEST_YAML", tmp_path / "missing.yaml"):
            before, added, after = preflight.verify_yaml_manifest()
        assert before == []
        assert added == []
        assert after == []

    def test_all_present_no_append(self, tmp_path):
        # If all four required IDs are already in the YAML, nothing is appended.
        manifest = tmp_path / "exclusion_manifest.yaml"
        manifest.write_text(
            "retired:\n"
            "  - experiment_id: 786\n"
            "  - experiment_id: 627\n"
            "  - experiment_id: 603\n"
            "  - experiment_id: 641\n"
        )
        with mock.patch.object(preflight, "EXCLUSION_MANIFEST_YAML", manifest):
            before, added, after = preflight.verify_yaml_manifest()
        assert set(before) == {786, 627, 603, 641}
        assert added == []
        assert set(after) == {786, 627, 603, 641}
        # File should not have grown.
        assert "experiment_id: 786\n  - experiment_id: 627" in manifest.read_text()

    def test_missing_ids_are_appended(self, tmp_path):
        # When some IDs are absent the function appends YAML entries for them.
        manifest = tmp_path / "exclusion_manifest.yaml"
        manifest.write_text("retired:\n  - experiment_id: 603\n  - experiment_id: 627\n")
        with mock.patch.object(preflight, "EXCLUSION_MANIFEST_YAML", manifest):
            before, added, after = preflight.verify_yaml_manifest()
        assert set(before) == {603, 627}
        assert set(added) == {786, 641}
        assert set(after) == {786, 627, 603, 641}
        appended_text = manifest.read_text()
        assert "experiment_id: 786" in appended_text
        assert "experiment_id: 641" in appended_text
        assert "2026.04.75" in appended_text


class TestSyncConductorManifest:
    def test_missing_file_returns_not_ok(self, tmp_path):
        # When the conductor JSON file does not exist, synced_ok is False.
        with mock.patch.object(preflight, "CONDUCTOR_MANIFEST_JSON", tmp_path / "missing.json"):
            before, added, after, ok = preflight.sync_conductor_manifest()
        assert ok is False
        assert before == []

    def test_already_synced_no_write(self, tmp_path):
        # When both 786 and 641 are already in the JSON, nothing is written.
        data = {
            "excluded": [
                {"experiment_id": 786, "completed_milestone": "2026.04.75", "reason": "test"},
                {"experiment_id": 641, "completed_milestone": "2026.04.75", "reason": "test"},
            ]
        }
        conductor_path = tmp_path / "conductor_exclusion_manifest.json"
        conductor_path.write_text(json.dumps(data))
        with mock.patch.object(preflight, "CONDUCTOR_MANIFEST_JSON", conductor_path):
            before, added, after, ok = preflight.sync_conductor_manifest()
        assert ok is True
        assert added == []
        assert 786 in before
        assert 641 in before

    def test_missing_ids_are_appended(self, tmp_path):
        # When 786 and 641 are absent, they are appended and the file is updated.
        data = {
            "excluded": [
                {"experiment_id": 603, "completed_milestone": "2026.04.58", "reason": "old"},
            ]
        }
        conductor_path = tmp_path / "conductor_exclusion_manifest.json"
        conductor_path.write_text(json.dumps(data))
        with mock.patch.object(preflight, "CONDUCTOR_MANIFEST_JSON", conductor_path):
            before, added, after, ok = preflight.sync_conductor_manifest()
        assert ok is True
        assert set(added) == {786, 641}
        assert 786 in after
        assert 641 in after
        # Verify the file was actually written.
        updated = json.loads(conductor_path.read_text())
        updated_ids = {e["experiment_id"] for e in updated["excluded"]}
        assert {786, 641} <= updated_ids


class TestDiagnoseExp906:
    def test_missing_result_returns_no_file(self, tmp_path):
        with mock.patch.object(preflight, "RESULTS_DIR", tmp_path):
            root_cause, fix = preflight.diagnose_exp906()
        assert "no_result_file_found" in root_cause
        assert fix == "apply_timeout_cap_40min"

    def test_real_result_classified_scale_latency(self):
        # The actual experiment_906 JSON must yield root cause class (c).
        root_cause, fix = preflight.diagnose_exp906()
        assert "(c) 50q_scale_x_per_question_latency" in root_cause
        assert fix == "apply_timeout_cap_40min"

    def test_fallback_mode_forces_class_c(self, tmp_path):
        # Any result with inference_mode=fallback_transformers_only maps to cause (c).
        fake_result = {
            "inference_mode": "fallback_transformers_only",
            "qwen_results_per_problem": [{"n_attempts": 2} for _ in range(10)],
        }
        result_file = tmp_path / "experiment_906_fake.json"
        result_file.write_text(json.dumps(fake_result))
        with mock.patch.object(preflight, "RESULTS_DIR", tmp_path):
            root_cause, fix = preflight.diagnose_exp906()
        assert "(c)" in root_cause
        assert fix == "apply_timeout_cap_40min"


class TestCheckSotaModels:
    def test_missing_models_return_false(self, tmp_path):
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            results = preflight.check_sota_models()
        assert all(not v for v in results.values())

    def test_present_models_return_true(self, tmp_path):
        hf_hub = tmp_path / ".cache" / "huggingface" / "hub"
        for dir_name in preflight.SOTA_MODEL_CACHE_PATTERNS.values():
            (hf_hub / dir_name).mkdir(parents=True)
        with mock.patch.dict(os.environ, {"HOME": str(tmp_path)}):
            results = preflight.check_sota_models()
        assert all(results.values())


class TestMainDeliverable:
    def test_deliverable_has_required_fields(self, tmp_path):
        # Run main() against a controlled environment and verify all required
        # schema fields are present in the output JSON.
        deliverable = tmp_path / "experiment_974_preflight_v26.json"
        manifest = tmp_path / "exclusion_manifest.yaml"
        manifest.write_text(
            "retired:\n"
            "  - experiment_id: 786\n"
            "  - experiment_id: 627\n"
            "  - experiment_id: 603\n"
            "  - experiment_id: 641\n"
        )
        conductor_data = {
            "excluded": [
                {"experiment_id": 786, "completed_milestone": "2026.04.75", "reason": "t"},
                {"experiment_id": 641, "completed_milestone": "2026.04.75", "reason": "t"},
            ]
        }
        conductor_path = tmp_path / "conductor_exclusion_manifest.json"
        conductor_path.write_text(json.dumps(conductor_data))

        with (
            mock.patch.object(preflight, "RESULTS_DIR", tmp_path),
            mock.patch.object(preflight, "DELIVERABLE", str(deliverable)),
            mock.patch.object(preflight, "EXCLUSION_MANIFEST_YAML", manifest),
            mock.patch.object(preflight, "CONDUCTOR_MANIFEST_JSON", conductor_path),
        ):
            preflight.main()

        assert deliverable.exists(), "deliverable JSON was not written"
        with open(deliverable) as f:
            artifact = json.load(f)

        required_fields = [
            "manifest_entries_before",
            "manifest_entries_added",
            "manifest_entries_after",
            "exp906_root_cause",
            "exp906_fix",
            "sota_models_ready",
            "conductor_manifest_synced",
            "honest_verdict",
        ]
        for field in required_fields:
            assert field in artifact, f"missing required field: {field}"

        assert artifact["honest_verdict"] in ("preflight_complete", "preflight_partial")
        assert artifact["exp906_fix"] in ("apply_timeout_cap_40min", "retired_to_manifest")
