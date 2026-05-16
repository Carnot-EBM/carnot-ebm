import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path
from carnot.pipeline.hf_publisher import HuggingFacePublisher

# REQ-PUBLISH-001: HuggingFace Model Card Requirements
# SCENARIO-PUBLISH-001: HuggingFace Artifact Preparation

@pytest.fixture
def dummy_artifact(tmp_path):
    artifact = tmp_path / "test.safetensors"
    artifact.write_bytes(b"dummy data")
    return artifact

def test_preconditions_blocked_no_cli(dummy_artifact):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        publisher = HuggingFacePublisher(artifact_path=str(dummy_artifact))
        result = publisher.run_publish()
        assert result["publish_mechanism"] == "blocked"
        assert result["honest_verdict"] == "blocked_huggingface_credentials_unavailable"
        assert result["acceptance_gate_passed"] is False

def test_preconditions_blocked_no_artifact():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Carnot-EBM")
        publisher = HuggingFacePublisher(artifact_path="nonexistent.safetensors")
        result = publisher.run_publish()
        assert result["publish_mechanism"] == "blocked"
        assert result["honest_verdict"] == "blocked_no_checkpoint_available"

def test_success_upload(dummy_artifact):
    with patch("subprocess.run") as mock_run, \
         patch("carnot.pipeline.hf_publisher.create_repo") as mock_create_repo, \
         patch("carnot.pipeline.hf_publisher.HfApi") as mock_hf_api, \
         patch("carnot.pipeline.hf_publisher.hf_hub_download") as mock_download, \
         patch("time.sleep"):
        mock_run.return_value = MagicMock(returncode=0, stdout="Carnot-EBM")
        mock_download.return_value = str(dummy_artifact)
        
        publisher = HuggingFacePublisher(artifact_path=str(dummy_artifact))
        result = publisher.run_publish()
        
        assert result["publish_mechanism"] == "hf_api_direct"
        assert result["hf_upload_succeeded"] is True
        assert result["external_load_verified"] is True
        assert result["honest_verdict"] == "complete: hf_upload_and_verify_success"
        assert result["acceptance_gate_passed"] is True
