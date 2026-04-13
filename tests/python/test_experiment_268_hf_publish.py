"""Tests for Experiment 268: HuggingFace publishing of Exp 66 joint model and
FormalClaimVerifier ONNX artifact.

Spec: REQ-VERIFY-058, REQ-VERIFY-059
Run date: 20260413
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXP66_ARCH = {
    "embed_dim": 384,
    "n_constraints": 8,
    "hidden_dim": 64,
    "alpha": 10.0,
}


# ---------------------------------------------------------------------------
# Model card content validation
# ---------------------------------------------------------------------------


def _load_model_card_text(card_path: Path) -> str:
    return card_path.read_text()


class TestExp66ModelCard:
    """Verify that the Exp 66 model card contains required sections and honest disclaimers."""

    def test_model_card_has_proof_of_concept_warning(self, tmp_path: Path) -> None:
        """Model card must clearly state this is NOT production quality."""
        from scripts.experiment_268_hf_publish import build_exp66_model_card

        card = build_exp66_model_card()
        text_lower = card.lower()
        # Must include a strong disclaimer
        assert any(
            phrase in text_lower
            for phrase in [
                "proof-of-concept",
                "proof of concept",
                "not production",
                "research artifact",
            ]
        ), "Model card must contain a proof-of-concept / not-production disclaimer"

    def test_model_card_has_architecture_section(self) -> None:
        """Model card must describe the embedding + Ising → score architecture."""
        from scripts.experiment_268_hf_publish import build_exp66_model_card

        card = build_exp66_model_card()
        assert "embedding" in card.lower(), "Model card must mention embedding layer"
        assert "ising" in card.lower(), "Model card must mention Ising coupling"
        assert "384" in card, "Model card must state embed_dim=384"
        assert "auroc" in card.lower(), "Model card must report AUROC"

    def test_model_card_has_usage_example(self) -> None:
        """Model card must include a Python usage snippet."""
        from scripts.experiment_268_hf_publish import build_exp66_model_card

        card = build_exp66_model_card()
        assert "```" in card, "Model card must contain a fenced code block (usage example)"

    def test_model_card_has_training_details(self) -> None:
        """Model card must mention key hyperparameters."""
        from scripts.experiment_268_hf_publish import build_exp66_model_card

        card = build_exp66_model_card()
        assert "200" in card, "Model card must state n_epochs=200"
        assert "0.001" in card, "Model card must state best_lr=0.001"


class TestFormalClaimVerifierModelCard:
    """Verify that the FCV model card describes solver routing and abstention policy."""

    def test_model_card_has_routes(self) -> None:
        """Model card must list the five solver routes."""
        from scripts.experiment_268_hf_publish import build_fcv_model_card

        card = build_fcv_model_card()
        for route in ("arithmetic", "comparison", "cardinality", "set_membership", "boolean_entailment"):
            assert route in card, f"Model card missing route: {route}"

    def test_model_card_has_abstention_policy(self) -> None:
        """Model card must explain the abstention policy."""
        from scripts.experiment_268_hf_publish import build_fcv_model_card

        card = build_fcv_model_card()
        assert "abstain" in card.lower(), "Model card must describe abstention policy"

    def test_model_card_has_onnx_mention(self) -> None:
        """Model card must mention ONNX export for arithmetic and comparison routes."""
        from scripts.experiment_268_hf_publish import build_fcv_model_card

        card = build_fcv_model_card()
        assert "onnx" in card.lower(), "Model card must mention ONNX"
        assert "arithmetic" in card.lower()
        assert "comparison" in card.lower()

    def test_model_card_has_usage_example(self) -> None:
        """Model card must show `from carnot.pipeline import FormalClaimVerifier`."""
        from scripts.experiment_268_hf_publish import build_fcv_model_card

        card = build_fcv_model_card()
        assert "FormalClaimVerifier" in card, "Model card must show FormalClaimVerifier import"
        assert "from carnot" in card, "Model card must show carnot package import"


# ---------------------------------------------------------------------------
# Safetensors export
# ---------------------------------------------------------------------------


class TestExp66SafetensorsExport:
    """Verify that the Exp 66 model can be exported to safetensors format."""

    def test_safetensors_export_keys(self, tmp_path: Path) -> None:
        """Exported safetensors must contain all expected weight tensors."""
        from scripts.experiment_268_hf_publish import train_and_export_exp66

        out_dir = tmp_path / "exp66"
        out_dir.mkdir()
        export_path = train_and_export_exp66(out_dir=out_dir, fast=True)

        from safetensors import safe_open

        tensors = {}
        with safe_open(str(export_path), framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        required_keys = {
            "ising_biases",
            "ising_J",
            "mlp_w1",
            "mlp_b1",
            "mlp_w2",
            "mlp_b2",
        }
        assert required_keys <= set(tensors.keys()), (
            f"Missing safetensors keys: {required_keys - set(tensors.keys())}"
        )

    def test_safetensors_shapes(self, tmp_path: Path) -> None:
        """Tensor shapes must match the documented Exp 66 architecture."""
        from scripts.experiment_268_hf_publish import train_and_export_exp66
        from safetensors import safe_open

        out_dir = tmp_path / "exp66"
        out_dir.mkdir()
        export_path = train_and_export_exp66(out_dir=out_dir, fast=True)

        tensors = {}
        with safe_open(str(export_path), framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        n = EXP66_ARCH["n_constraints"]
        d = EXP66_ARCH["embed_dim"]
        h = EXP66_ARCH["hidden_dim"]
        mlp_in = d + n + n + 1

        assert tensors["ising_biases"].shape == (n,)
        assert tensors["ising_J"].shape == (n, n)
        assert tensors["mlp_w1"].shape == (mlp_in, h)
        assert tensors["mlp_b1"].shape == (h,)
        assert tensors["mlp_w2"].shape == (h, 1)
        assert tensors["mlp_b2"].shape == (1,)


# ---------------------------------------------------------------------------
# ONNX export schema validation
# ---------------------------------------------------------------------------


class TestFCVOnnxExport:
    """Verify that arithmetic and comparison routes export to valid ONNX."""

    def _export(self, tmp_path: Path) -> tuple[Path, Path]:
        from scripts.experiment_268_hf_publish import export_fcv_onnx

        return export_fcv_onnx(out_dir=tmp_path)

    def test_onnx_arithmetic_is_valid(self, tmp_path: Path) -> None:
        """Arithmetic ONNX model must pass onnx.checker.check_model."""
        arith_path, _cmp_path = self._export(tmp_path)
        model = onnx.load(str(arith_path))
        onnx.checker.check_model(model)

    def test_onnx_comparison_is_valid(self, tmp_path: Path) -> None:
        """Comparison ONNX model must pass onnx.checker.check_model."""
        _arith_path, cmp_path = self._export(tmp_path)
        model = onnx.load(str(cmp_path))
        onnx.checker.check_model(model)

    def test_onnx_arithmetic_inference(self, tmp_path: Path) -> None:
        """Arithmetic ONNX model must produce correct verdicts at runtime."""
        arith_path, _cmp_path = self._export(tmp_path)
        sess = ort.InferenceSession(str(arith_path), providers=["CPUExecutionProvider"])

        # 100 - 24 = 76 → supported (1)
        operands_ok = np.array([[100.0, 24.0, 76.0]], dtype=np.float32)
        [verdict_ok] = sess.run(None, {"operands": operands_ok})
        assert int(verdict_ok[0]) == 1, "100 - 24 = 76 should be supported"

        # 100 - 24 = 99 → violated (0)
        operands_bad = np.array([[100.0, 24.0, 99.0]], dtype=np.float32)
        [verdict_bad] = sess.run(None, {"operands": operands_bad})
        assert int(verdict_bad[0]) == 0, "100 - 24 = 99 should be violated"

    def test_onnx_comparison_inference(self, tmp_path: Path) -> None:
        """Comparison ONNX model must produce correct verdicts for less_than."""
        _arith_path, cmp_path = self._export(tmp_path)
        sess = ort.InferenceSession(str(cmp_path), providers=["CPUExecutionProvider"])

        # 3 < 7 → supported (1) for less_than
        operands_ok = np.array([[3.0, 7.0]], dtype=np.float32)
        [verdict_ok] = sess.run(None, {"operands": operands_ok})
        assert int(verdict_ok[0]) == 1, "3 < 7 should be supported (less_than)"

        # 10 < 3 → violated (0) for less_than
        operands_bad = np.array([[10.0, 3.0]], dtype=np.float32)
        [verdict_bad] = sess.run(None, {"operands": operands_bad})
        assert int(verdict_bad[0]) == 0, "10 < 3 should be violated (less_than)"

    def test_onnx_arithmetic_opset(self, tmp_path: Path) -> None:
        """Arithmetic ONNX must use opset >= 13 for broad compatibility."""
        arith_path, _cmp_path = self._export(tmp_path)
        model = onnx.load(str(arith_path))
        opsets = {op.domain: op.version for op in model.opset_import}
        default_opset = opsets.get("", 0)
        assert default_opset >= 13, f"Expected opset >= 13, got {default_opset}"


# ---------------------------------------------------------------------------
# Mock HF upload path
# ---------------------------------------------------------------------------


class TestMockHFUpload:
    """Verify the HF upload path without actually uploading."""

    def test_upload_returns_repo_urls(self, tmp_path: Path) -> None:
        """upload_artifacts must return a dict with exp66 and fcv HF URLs."""
        from scripts.experiment_268_hf_publish import upload_artifacts

        mock_api = MagicMock()
        mock_api.create_repo.return_value = MagicMock(repo_id="Carnot-EBM/test-exp66")
        mock_api.upload_folder.return_value = MagicMock()

        result = upload_artifacts(
            exp66_dir=tmp_path / "exp66",
            fcv_dir=tmp_path / "fcv",
            tag="v0.2.0-research",
            dry_run=True,
            hf_api=mock_api,
        )

        assert "exp66_repo" in result, "Must return exp66_repo key"
        assert "fcv_repo" in result, "Must return fcv_repo key"
        assert "tag" in result, "Must return tag key"
        assert result["tag"] == "v0.2.0-research"

    def test_dry_run_does_not_call_hf_api(self, tmp_path: Path) -> None:
        """In dry_run mode, no HF API calls should be made."""
        from scripts.experiment_268_hf_publish import upload_artifacts

        mock_api = MagicMock()
        upload_artifacts(
            exp66_dir=tmp_path / "exp66",
            fcv_dir=tmp_path / "fcv",
            tag="v0.2.0-research",
            dry_run=True,
            hf_api=mock_api,
        )

        mock_api.create_repo.assert_not_called()
        mock_api.upload_folder.assert_not_called()


# ---------------------------------------------------------------------------
# Results JSON schema
# ---------------------------------------------------------------------------


class TestResultsJsonSchema:
    """Verify the experiment_268_results.json has required fields."""

    @pytest.fixture
    def results(self) -> dict:
        results_path = (
            Path(__file__).parent.parent.parent / "results" / "experiment_268_results.json"
        )
        if not results_path.exists():
            pytest.skip("experiment_268_results.json not yet generated")
        with open(results_path) as f:
            return json.load(f)

    def test_has_experiment_id(self, results: dict) -> None:
        assert results.get("experiment") == 268

    def test_has_run_date(self, results: dict) -> None:
        assert results.get("run_date") == "20260413"

    def test_has_honest_verdict(self, results: dict) -> None:
        assert "honest_verdict" in results
        assert "explanation" in results["honest_verdict"]

    def test_has_artifacts(self, results: dict) -> None:
        assert "artifacts" in results
        artifacts = results["artifacts"]
        assert "exp66" in artifacts
        assert "fcv" in artifacts

    def test_no_fabricated_upload_confirmations(self, results: dict) -> None:
        """If upload failed or was dry-run, must NOT claim successful upload."""
        artifacts = results.get("artifacts", {})
        for name, art in artifacts.items():
            if art.get("upload_status") == "dry_run" or art.get("upload_status") == "skipped":
                # These are honest statuses — not fabricated
                pass
            elif art.get("upload_status") == "uploaded":
                # Must have a real HF URL
                assert art.get("hf_url"), f"{name}: claimed upload but no hf_url"
