"""Tests for Experiment 293: HuggingFace publishing of Exp 66 joint model and
FormalClaimVerifier ONNX artifact.

Carry-forward from Exp 268 (SKIP'd 3×). Publishes to:
  - Carnot-EBM/carnot-joint-constraint-v1  (Exp 66)
  - Carnot-EBM/carnot-formal-claim-verifier-v1  (FCV)

Spec: REQ-VERIFY-058, REQ-VERIFY-059
Run date: 20260414
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest

# ---------------------------------------------------------------------------
# Architecture constants from Exp 66 hyperparameters (results JSON)
# ---------------------------------------------------------------------------

EXP66_ARCH = {
    "embed_dim": 384,
    "n_constraints": 8,
    "hidden_dim": 64,
    "alpha": 10.0,
}

# ---------------------------------------------------------------------------
# Credential check tests
# ---------------------------------------------------------------------------


class TestCredentialCheck:
    """Verify credential checking behaviour before any HF upload attempt."""

    def test_check_credentials_returns_true_when_logged_in(self) -> None:
        """check_hf_credentials() must return True when huggingface-cli whoami succeeds."""
        from scripts.experiment_293_huggingface_publish import check_hf_credentials

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Carnot-EBM\n", stderr="")
            ok, msg = check_hf_credentials()
        assert ok is True
        assert msg == "" or "logged in" in msg.lower() or "carnot" in msg.lower()

    def test_check_credentials_returns_false_when_not_logged_in(self) -> None:
        """check_hf_credentials() must return (False, instructions) when whoami fails."""
        from scripts.experiment_293_huggingface_publish import check_hf_credentials

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Not logged in")
            ok, msg = check_hf_credentials()
        assert ok is False
        assert "huggingface-cli login" in msg, "Blocked message must include login instruction"

    def test_check_credentials_handles_missing_cli(self) -> None:
        """check_hf_credentials() must return (False, instructions) when CLI not found."""
        from scripts.experiment_293_huggingface_publish import check_hf_credentials

        with patch("subprocess.run", side_effect=FileNotFoundError("huggingface-cli not found")):
            ok, msg = check_hf_credentials()
        assert ok is False
        assert msg  # Must have a non-empty message

    def test_blocked_artifact_written_when_no_credentials(self, tmp_path: Path) -> None:
        """When credentials fail, a blocked artifact JSON must be written and returned."""
        from scripts.experiment_293_huggingface_publish import run_experiment_293

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Not logged in")
            result = run_experiment_293(out_dir=tmp_path, dry_run=True)

        assert result.get("blocked") is True, "Result must have blocked=True on credential failure"
        assert "login_instructions" in result, "Blocked result must include login_instructions"

    def test_blocked_artifact_has_login_command(self, tmp_path: Path) -> None:
        """The blocked artifact must contain the literal huggingface-cli login command."""
        from scripts.experiment_293_huggingface_publish import run_experiment_293

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Not logged in")
            result = run_experiment_293(out_dir=tmp_path, dry_run=True)

        assert "huggingface-cli login" in result.get("login_instructions", "")


# ---------------------------------------------------------------------------
# Exp 66 model card content validation
# ---------------------------------------------------------------------------


class TestExp66ModelCard:
    """Verify that the Exp 66 model card contains required sections."""

    def test_model_card_has_phase1_banner(self) -> None:
        """Model card must include 'Phase 1 research prototype' banner."""
        from scripts.experiment_293_huggingface_publish import build_exp66_model_card

        card = build_exp66_model_card()
        assert "phase 1 research prototype" in card.lower(), (
            "Model card must contain 'Phase 1 research prototype'"
        )

    def test_model_card_has_auroc_claim(self) -> None:
        """Model card must state 1.0 AUROC."""
        from scripts.experiment_293_huggingface_publish import build_exp66_model_card

        card = build_exp66_model_card()
        assert "1.0" in card and "auroc" in card.lower(), (
            "Model card must report 1.0 AUROC"
        )

    def test_model_card_has_not_production_disclaimer(self) -> None:
        """Model card must say NOT production quality."""
        from scripts.experiment_293_huggingface_publish import build_exp66_model_card

        card = build_exp66_model_card()
        text_lower = card.lower()
        assert any(
            phrase in text_lower
            for phrase in ["not production", "research prototype", "research artifact"]
        ), "Model card must have a not-production disclaimer"

    def test_model_card_has_pip_install(self) -> None:
        """Model card must include pip install carnot instructions."""
        from scripts.experiment_293_huggingface_publish import build_exp66_model_card

        card = build_exp66_model_card()
        assert "pip install carnot" in card, "Model card must have pip install carnot"

    def test_model_card_has_architecture_details(self) -> None:
        """Model card must mention embedding layer, Ising coupling, and embed_dim=384."""
        from scripts.experiment_293_huggingface_publish import build_exp66_model_card

        card = build_exp66_model_card()
        assert "embedding" in card.lower()
        assert "ising" in card.lower()
        assert "384" in card

    def test_model_card_has_training_hyperparams(self) -> None:
        """Model card must state n_epochs=200 and best_lr=0.001."""
        from scripts.experiment_293_huggingface_publish import build_exp66_model_card

        card = build_exp66_model_card()
        assert "200" in card, "Must state n_epochs=200"
        assert "0.001" in card, "Must state best_lr=0.001"

    def test_model_card_has_code_block(self) -> None:
        """Model card must include a fenced code block for usage."""
        from scripts.experiment_293_huggingface_publish import build_exp66_model_card

        card = build_exp66_model_card()
        assert "```" in card, "Model card must contain a fenced code block"


# ---------------------------------------------------------------------------
# FCV model card validation
# ---------------------------------------------------------------------------


class TestFCVModelCard:
    """Verify that the FCV model card describes solver routing and abstention."""

    def test_model_card_has_all_routes(self) -> None:
        """Model card must list all five solver routes."""
        from scripts.experiment_293_huggingface_publish import build_fcv_model_card

        card = build_fcv_model_card()
        for route in (
            "arithmetic",
            "comparison",
            "cardinality",
            "set_membership",
            "boolean_entailment",
        ):
            assert route in card, f"Model card missing route: {route}"

    def test_model_card_has_abstention_policy(self) -> None:
        """Model card must explain the abstention policy."""
        from scripts.experiment_293_huggingface_publish import build_fcv_model_card

        card = build_fcv_model_card()
        assert "abstain" in card.lower(), "Model card must describe abstention"

    def test_model_card_has_onnx_mention(self) -> None:
        """Model card must mention ONNX export."""
        from scripts.experiment_293_huggingface_publish import build_fcv_model_card

        card = build_fcv_model_card()
        assert "onnx" in card.lower(), "Model card must mention ONNX"

    def test_model_card_has_formalclaimverifier_import(self) -> None:
        """Model card must show `from carnot.pipeline import FormalClaimVerifier`."""
        from scripts.experiment_293_huggingface_publish import build_fcv_model_card

        card = build_fcv_model_card()
        assert "FormalClaimVerifier" in card
        assert "from carnot" in card

    def test_model_card_has_pip_install(self) -> None:
        """Model card must include pip install carnot instructions."""
        from scripts.experiment_293_huggingface_publish import build_fcv_model_card

        card = build_fcv_model_card()
        assert "pip install carnot" in card


# ---------------------------------------------------------------------------
# Safetensors export (Exp 66)
# ---------------------------------------------------------------------------


class TestExp66SafetensorsExport:
    """Verify safetensors export produces correctly-shaped weight tensors."""

    def test_safetensors_export_keys(self, tmp_path: Path) -> None:
        """Exported safetensors must contain all expected weight keys."""
        from scripts.experiment_293_huggingface_publish import build_exp66_safetensors

        export_path = build_exp66_safetensors(out_dir=tmp_path)
        from safetensors import safe_open

        tensors: dict = {}
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
            f"Missing keys: {required_keys - set(tensors.keys())}"
        )

    def test_safetensors_shapes(self, tmp_path: Path) -> None:
        """Tensor shapes must match the Exp 66 architecture specification."""
        from scripts.experiment_293_huggingface_publish import build_exp66_safetensors
        from safetensors import safe_open

        export_path = build_exp66_safetensors(out_dir=tmp_path)
        tensors: dict = {}
        with safe_open(str(export_path), framework="np") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

        n = EXP66_ARCH["n_constraints"]
        d = EXP66_ARCH["embed_dim"]
        h = EXP66_ARCH["hidden_dim"]
        mlp_in = d + n + n + 1

        assert tensors["ising_biases"].shape == (n,), f"ising_biases: expected ({n},)"
        assert tensors["ising_J"].shape == (n, n), f"ising_J: expected ({n},{n})"
        assert tensors["mlp_w1"].shape == (mlp_in, h)
        assert tensors["mlp_b1"].shape == (h,)
        assert tensors["mlp_w2"].shape == (h, 1)
        assert tensors["mlp_b2"].shape == (1,)


# ---------------------------------------------------------------------------
# ONNX export validation
# ---------------------------------------------------------------------------


class TestFCVOnnxExport:
    """Verify arithmetic and comparison ONNX exports are valid and runnable."""

    def test_onnx_arithmetic_is_valid(self, tmp_path: Path) -> None:
        """Arithmetic ONNX model must pass onnx.checker.check_model."""
        from scripts.experiment_293_huggingface_publish import export_fcv_onnx

        arith_path, _cmp_path = export_fcv_onnx(out_dir=tmp_path)
        model = onnx.load(str(arith_path))
        onnx.checker.check_model(model)

    def test_onnx_comparison_is_valid(self, tmp_path: Path) -> None:
        """Comparison ONNX model must pass onnx.checker.check_model."""
        from scripts.experiment_293_huggingface_publish import export_fcv_onnx

        _arith_path, cmp_path = export_fcv_onnx(out_dir=tmp_path)
        model = onnx.load(str(cmp_path))
        onnx.checker.check_model(model)

    def test_onnx_arithmetic_opset(self, tmp_path: Path) -> None:
        """Arithmetic ONNX must use opset >= 13."""
        from scripts.experiment_293_huggingface_publish import export_fcv_onnx

        arith_path, _cmp_path = export_fcv_onnx(out_dir=tmp_path)
        model = onnx.load(str(arith_path))
        opsets = {op.domain: op.version for op in model.opset_import}
        assert opsets.get("", 0) >= 13

    def test_onnx_arithmetic_inference_supported(self, tmp_path: Path) -> None:
        """Arithmetic ONNX: 100 - 24 = 76 should be verdict=1 (supported)."""
        from scripts.experiment_293_huggingface_publish import export_fcv_onnx

        arith_path, _ = export_fcv_onnx(out_dir=tmp_path)
        sess = ort.InferenceSession(str(arith_path), providers=["CPUExecutionProvider"])
        operands = np.array([[100.0, 24.0, 76.0]], dtype=np.float32)
        [verdict] = sess.run(None, {"operands": operands})
        assert int(verdict[0]) == 1, "100 - 24 = 76 should be supported"

    def test_onnx_arithmetic_inference_violated(self, tmp_path: Path) -> None:
        """Arithmetic ONNX: 100 - 24 = 99 should be verdict=0 (violated)."""
        from scripts.experiment_293_huggingface_publish import export_fcv_onnx

        arith_path, _ = export_fcv_onnx(out_dir=tmp_path)
        sess = ort.InferenceSession(str(arith_path), providers=["CPUExecutionProvider"])
        operands = np.array([[100.0, 24.0, 99.0]], dtype=np.float32)
        [verdict] = sess.run(None, {"operands": operands})
        assert int(verdict[0]) == 0, "100 - 24 = 99 should be violated"

    def test_onnx_comparison_inference_supported(self, tmp_path: Path) -> None:
        """Comparison ONNX: 3 < 7 should be verdict=1 (supported for less_than)."""
        from scripts.experiment_293_huggingface_publish import export_fcv_onnx

        _, cmp_path = export_fcv_onnx(out_dir=tmp_path)
        sess = ort.InferenceSession(str(cmp_path), providers=["CPUExecutionProvider"])
        operands = np.array([[3.0, 7.0]], dtype=np.float32)
        [verdict] = sess.run(None, {"operands": operands})
        assert int(verdict[0]) == 1, "3 < 7 should be supported"

    def test_onnx_comparison_inference_violated(self, tmp_path: Path) -> None:
        """Comparison ONNX: 10 < 3 should be verdict=0 (violated)."""
        from scripts.experiment_293_huggingface_publish import export_fcv_onnx

        _, cmp_path = export_fcv_onnx(out_dir=tmp_path)
        sess = ort.InferenceSession(str(cmp_path), providers=["CPUExecutionProvider"])
        operands = np.array([[10.0, 3.0]], dtype=np.float32)
        [verdict] = sess.run(None, {"operands": operands})
        assert int(verdict[0]) == 0, "10 < 3 should be violated"


# ---------------------------------------------------------------------------
# upload_artifacts mock path
# ---------------------------------------------------------------------------


class TestMockHFUpload:
    """Verify upload logic without live HF API calls."""

    def test_upload_returns_expected_repo_urls(self, tmp_path: Path) -> None:
        """upload_artifacts must return exp66_repo and fcv_repo URL keys."""
        from scripts.experiment_293_huggingface_publish import upload_artifacts

        result = upload_artifacts(
            exp66_dir=tmp_path / "exp66",
            fcv_dir=tmp_path / "fcv",
            tag="v0.2.0-research",
            dry_run=True,
        )
        assert "exp66_repo" in result
        assert "fcv_repo" in result
        assert "tag" in result
        assert result["tag"] == "v0.2.0-research"

    def test_upload_uses_correct_repo_ids(self, tmp_path: Path) -> None:
        """Repo IDs must be carnot-joint-constraint-v1 and carnot-formal-claim-verifier-v1."""
        from scripts.experiment_293_huggingface_publish import upload_artifacts

        result = upload_artifacts(
            exp66_dir=tmp_path / "exp66",
            fcv_dir=tmp_path / "fcv",
            tag="v0.2.0-research",
            dry_run=True,
        )
        assert "carnot-joint-constraint-v1" in result["exp66_repo"]
        assert "carnot-formal-claim-verifier-v1" in result["fcv_repo"]

    def test_dry_run_skips_hf_api_calls(self, tmp_path: Path) -> None:
        """dry_run=True must not call create_repo, upload_folder, or create_tag."""
        from scripts.experiment_293_huggingface_publish import upload_artifacts

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
        mock_api.create_tag.assert_not_called()

    def test_live_upload_creates_tags(self, tmp_path: Path) -> None:
        """Non-dry-run upload must call create_tag for both repos with v0.2.0-research."""
        from scripts.experiment_293_huggingface_publish import upload_artifacts

        mock_api = MagicMock()
        # Make dirs exist so upload_folder is attempted
        exp66_dir = tmp_path / "exp66"
        fcv_dir = tmp_path / "fcv"
        exp66_dir.mkdir()
        fcv_dir.mkdir()

        upload_artifacts(
            exp66_dir=exp66_dir,
            fcv_dir=fcv_dir,
            tag="v0.2.0-research",
            dry_run=False,
            hf_api=mock_api,
        )
        # create_tag must be called twice — once per repo
        assert mock_api.create_tag.call_count == 2
        called_tags = {c.kwargs.get("tag") or c.args[1] for c in mock_api.create_tag.call_args_list}
        assert "v0.2.0-research" in called_tags


# ---------------------------------------------------------------------------
# Safetensors skip path (spec: "if not found, skip artifact, log missing")
# ---------------------------------------------------------------------------


class TestExp66SkipWhenSafetensorsMissing:
    """Verify that the main pipeline skips Exp 66 when safetensors is absent."""

    def test_skip_exp66_when_safetensors_missing(self, tmp_path: Path) -> None:
        """run_experiment_293 must skip exp66 with status 'skipped_missing_safetensors' when file absent."""
        from scripts.experiment_293_huggingface_publish import run_experiment_293

        results_file = tmp_path / "experiment_293_results.json"
        with (
            patch("subprocess.run") as mock_run,
            patch("scripts.experiment_293_huggingface_publish._EXP66_SAFETENSORS_PATH",
                  tmp_path / "nonexistent.safetensors"),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="TestUser\n", stderr="")
            result = run_experiment_293(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )

        assert result["artifacts"]["exp66"]["upload_status"] == "skipped_missing_safetensors"
        assert result["artifacts"]["exp66"]["hf_url"] is None
        assert result["artifacts"]["exp66"]["missing_note"] is not None, "Must log missing note"

    def test_fcv_continues_after_exp66_skip(self, tmp_path: Path) -> None:
        """FCV artifact must be built and staged even when Exp 66 safetensors is absent."""
        from scripts.experiment_293_huggingface_publish import run_experiment_293

        results_file = tmp_path / "experiment_293_results.json"
        with (
            patch("subprocess.run") as mock_run,
            patch("scripts.experiment_293_huggingface_publish._EXP66_SAFETENSORS_PATH",
                  tmp_path / "nonexistent.safetensors"),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="TestUser\n", stderr="")
            result = run_experiment_293(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )

        fcv = result["artifacts"]["fcv"]
        assert fcv["upload_status"] in ("dry_run", "uploaded"), "FCV must still be staged"
        assert "carnot-formal-claim-verifier-v1" in (fcv.get("hf_url") or "")


# ---------------------------------------------------------------------------
# Results JSON written to disk
# ---------------------------------------------------------------------------


class TestResultsWrittenToDisk:
    """Verify that the results JSON is actually written to disk, not just returned."""

    def test_results_json_written_on_dry_run(self, tmp_path: Path) -> None:
        """run_experiment_293 must write results JSON to disk even in dry_run mode."""
        from scripts.experiment_293_huggingface_publish import run_experiment_293

        results_file = tmp_path / "experiment_293_results.json"
        assert not results_file.exists(), "precondition: file must not exist yet"

        with (
            patch("subprocess.run") as mock_run,
            patch("scripts.experiment_293_huggingface_publish._EXP66_SAFETENSORS_PATH",
                  tmp_path / "nonexistent.safetensors"),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="TestUser\n", stderr="")
            run_experiment_293(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )

        assert results_file.exists(), "Results JSON must be written to disk"
        with open(results_file) as f:
            data = json.load(f)
        assert data.get("experiment") == 293

    def test_blocked_results_written_to_disk(self, tmp_path: Path) -> None:
        """Blocked path must also write results JSON to disk."""
        from scripts.experiment_293_huggingface_publish import run_experiment_293

        results_file = tmp_path / "experiment_293_results.json"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Not logged in")
            run_experiment_293(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )

        assert results_file.exists(), "Blocked results JSON must be written to disk"
        with open(results_file) as f:
            data = json.load(f)
        assert data.get("blocked") is True

    def test_results_json_has_repo_ids(self, tmp_path: Path) -> None:
        """Results JSON must include repo_ids block regardless of upload outcome."""
        from scripts.experiment_293_huggingface_publish import run_experiment_293

        results_file = tmp_path / "experiment_293_results.json"
        with (
            patch("subprocess.run") as mock_run,
            patch("scripts.experiment_293_huggingface_publish._EXP66_SAFETENSORS_PATH",
                  tmp_path / "nonexistent.safetensors"),
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="TestUser\n", stderr="")
            result = run_experiment_293(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )

        assert "repo_ids" in result
        assert "carnot-joint-constraint-v1" in result["repo_ids"]["exp66"]
        assert "carnot-formal-claim-verifier-v1" in result["repo_ids"]["fcv"]

    def test_blocked_result_has_repo_ids(self, tmp_path: Path) -> None:
        """Blocked result must also include repo_ids so the caller knows where to publish."""
        from scripts.experiment_293_huggingface_publish import run_experiment_293

        results_file = tmp_path / "experiment_293_results.json"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Not logged in")
            result = run_experiment_293(
                out_dir=tmp_path / "staging",
                dry_run=True,
                results_path=results_file,
            )

        assert "repo_ids" in result, "Blocked result must include repo_ids"


# ---------------------------------------------------------------------------
# Results JSON schema
# ---------------------------------------------------------------------------


class TestResultsJsonSchema:
    """Validate the experiment_293_results.json schema when it exists."""

    @pytest.fixture
    def results(self) -> dict:
        results_path = (
            Path(__file__).parent.parent.parent / "results" / "experiment_293_results.json"
        )
        if not results_path.exists():
            pytest.skip("experiment_293_results.json not yet generated")
        with open(results_path) as f:
            return json.load(f)

    def test_has_experiment_id(self, results: dict) -> None:
        """Must have experiment == 293."""
        assert results.get("experiment") == 293

    def test_has_run_date(self, results: dict) -> None:
        """Must have run_date == '20260414'."""
        assert results.get("run_date") == "20260414"

    def test_has_artifacts_block(self, results: dict) -> None:
        """Must have top-level artifacts dict with exp66 and fcv keys."""
        assert "artifacts" in results
        artifacts = results["artifacts"]
        assert "exp66" in artifacts
        assert "fcv" in artifacts

    def test_no_fabricated_upload_status(self, results: dict) -> None:
        """If upload was dry_run or skipped, must NOT claim uploaded with no URL."""
        artifacts = results.get("artifacts", {})
        for name, art in artifacts.items():
            if art.get("upload_status") == "uploaded":
                assert art.get("hf_url"), f"{name}: claimed 'uploaded' but no hf_url"

    def test_has_honest_verdict(self, results: dict) -> None:
        """Must include honest_verdict block with explanation."""
        assert "honest_verdict" in results
        assert "explanation" in results["honest_verdict"]

    def test_v02_tag_present(self, results: dict) -> None:
        """Must record the v0.2.0-research tag."""
        assert results.get("tag") == "v0.2.0-research"
