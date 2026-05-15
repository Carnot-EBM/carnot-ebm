"""Tests for PyPI publisher.

REQ-PUBLISH-003: Gate Check Before Publication Actions
SCENARIO-PUBLISH-003: Primary Result Gate Fails (adapted for credentials)
"""
import os
import tempfile
import json
from unittest import mock
from carnot.pypi_publish import check_pypi_credentials, build_publish_artifact, run_publish

def test_check_pypi_credentials():
    with mock.patch.dict(os.environ, {"TWINE_USERNAME": "test", "TWINE_PASSWORD": "password"}):
        assert check_pypi_credentials() is True
    
    with mock.patch.dict(os.environ, {}, clear=True):
        assert check_pypi_credentials() is False

def test_build_publish_artifact_blocked():
    with mock.patch.dict(os.environ, {}, clear=True):
        artifact = build_publish_artifact(1711)
        assert artifact["honest_verdict"] == "blocked_pypi_credentials_unavailable"
        assert artifact["schema"] == "carnot.pypi_publish.v2"
        assert artifact["experiment"] == 1711
        assert artifact["methodology_note"] == "Per CLAUDE.md preconditions discipline, missing credentials → blocked verdict, NOT fabrication."

def test_run_publish():
    with tempfile.TemporaryDirectory() as tmpdir:
        result_path = os.path.join(tmpdir, "out.json")
        with mock.patch.dict(os.environ, {}, clear=True):
            run_publish(1711, result_path)
            
        assert os.path.exists(result_path)
        with open(result_path, "r") as f:
            data = json.load(f)
            assert data["experiment"] == 1711
            assert data["honest_verdict"] == "blocked_pypi_credentials_unavailable"
