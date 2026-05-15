import subprocess
from unittest import mock

import pytest

from carnot.verify.lean4_bridge import Lean4VerifierBackend


def test_lean4_backend_name():
    backend = Lean4VerifierBackend()
    assert backend.name == "lean4_verifier"


def test_parse_formal_constraint():
    backend = Lean4VerifierBackend()
    parsed = backend.parse_formal_constraint("true")
    assert "verify_constraint" in parsed
    assert "true" in parsed


@mock.patch("subprocess.run")
def test_verify_success(mock_run):
    backend = Lean4VerifierBackend()
    mock_run.return_value = mock.Mock(returncode=0)
    energy = backend.verify("def test : Bool := true")
    assert energy == 0.0
    mock_run.assert_called_once()


@mock.patch("subprocess.run")
def test_verify_failure(mock_run):
    backend = Lean4VerifierBackend()
    mock_run.return_value = mock.Mock(returncode=1)
    energy = backend.verify("def test : Bool := false")
    assert energy == float("inf")


@mock.patch("subprocess.run")
def test_verify_timeout(mock_run):
    backend = Lean4VerifierBackend()
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="lean", timeout=5.0)
    energy = backend.verify("def test : Bool := true")
    assert energy == float("inf")


@mock.patch("subprocess.run")
def test_verify_not_installed(mock_run):
    backend = Lean4VerifierBackend()
    mock_run.side_effect = FileNotFoundError()
    energy = backend.verify("def test : Bool := true")
    assert energy == float("inf")


@mock.patch("carnot.verify.lean4_bridge.Lean4VerifierBackend.verify")
def test_energy_convenience_method(mock_verify):
    backend = Lean4VerifierBackend()
    mock_verify.return_value = 0.0
    energy = backend.energy("true")
    assert energy == 0.0
    mock_verify.assert_called_once()
