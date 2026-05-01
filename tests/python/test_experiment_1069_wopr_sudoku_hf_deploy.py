"""Tests for Exp 1069 WOPR Sudoku HuggingFace Space deployment.

Spec: REQ-WOPR-001 — WOPR cartridge HuggingFace Space deployment + SOPS-encrypted
token handling.

These tests cover the deployment-helper functions in isolation —
they never actually hit huggingface.co. Network-touching paths
(``_hf_login``, ``_upload_space``, ``_verify_live``) are exercised
through monkeypatched stand-ins so the unit-test pass remains fast,
hermetic, and runs in CI without write credentials.

What we deliberately verify:

  - The SOPS decryption path returns the token *without* the
    ``HF_TOKEN:`` prefix, *without* surrounding quotes, and rejects
    the ``REPLACE_WITH_ACTUAL_TOKEN`` stub value.
  - The fallback chain (env -> keyring -> not_found) fires in the
    right order when the prior layer returns no value.
  - The artifact-building path in ``main`` honours each branch of
    the verdict matrix (token_missing, auth_fail, upload_fail,
    upload_ok+verify_fail, upload_ok+verify_ok).
  - ``_create_sops_stub`` refuses to clobber an existing encrypted
    file. This is the safety property that prevents wiping a real
    operator-supplied token if the experiment is re-run on a host
    that lacks the age key.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _d in [str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import scripts.experiment_1069_wopr_sudoku_hf_deploy as exp1069  # noqa: E402


# ---------------------------------------------------------------------------
# _decrypt_sops_token
# ---------------------------------------------------------------------------


def test_decrypt_sops_token_uses_sops_when_file_exists(monkeypatch, tmp_path):
    """When SOPS decrypts cleanly, we get the token back as 'sops' source."""
    fake_enc = tmp_path / "hf_token.enc.yaml"
    fake_enc.write_text("ENC[...]\n", encoding="utf-8")
    monkeypatch.setattr(exp1069, "SECRETS_PATH", fake_enc)
    monkeypatch.setattr(exp1069.shutil, "which", lambda name: "/usr/bin/sops")

    class _Proc:
        returncode = 0
        stdout = "HF_TOKEN: hf_test_real_token_value\n"
        stderr = ""

    monkeypatch.setattr(exp1069.subprocess, "run", lambda *a, **kw: _Proc())
    token, source = exp1069._decrypt_sops_token()
    assert token == "hf_test_real_token_value"
    assert source == "sops"


def test_decrypt_sops_token_strips_quotes(monkeypatch, tmp_path):
    """A quoted SOPS value should come back unquoted."""
    fake_enc = tmp_path / "hf_token.enc.yaml"
    fake_enc.write_text("dummy\n", encoding="utf-8")
    monkeypatch.setattr(exp1069, "SECRETS_PATH", fake_enc)
    monkeypatch.setattr(exp1069.shutil, "which", lambda name: "/usr/bin/sops")

    class _Proc:
        returncode = 0
        stdout = "HF_TOKEN: 'hf_quoted_token'\n"

    monkeypatch.setattr(exp1069.subprocess, "run", lambda *a, **kw: _Proc())
    token, source = exp1069._decrypt_sops_token()
    assert token == "hf_quoted_token"
    assert source == "sops"


def test_decrypt_sops_token_rejects_stub_placeholder(monkeypatch, tmp_path):
    """The placeholder string must NOT be returned as a real token."""
    fake_enc = tmp_path / "hf_token.enc.yaml"
    fake_enc.write_text("dummy\n", encoding="utf-8")
    monkeypatch.setattr(exp1069, "SECRETS_PATH", fake_enc)
    monkeypatch.setattr(exp1069.shutil, "which", lambda name: "/usr/bin/sops")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    class _Proc:
        returncode = 0
        stdout = "HF_TOKEN: REPLACE_WITH_ACTUAL_TOKEN\n"

    monkeypatch.setattr(exp1069.subprocess, "run", lambda *a, **kw: _Proc())
    # Force keyring import to fail so we exercise the not_found path
    # cleanly. We don't try to fake a keyring backend on this host.
    monkeypatch.setitem(sys.modules, "keyring", None)
    token, source = exp1069._decrypt_sops_token()
    assert token is None
    assert source == "not_found"


def test_decrypt_sops_token_falls_back_to_env(monkeypatch, tmp_path):
    """If SOPS unavailable, $HF_TOKEN is the second-tier source."""
    monkeypatch.setattr(exp1069, "SECRETS_PATH", tmp_path / "missing.yaml")
    monkeypatch.setattr(exp1069.shutil, "which", lambda name: None)
    monkeypatch.setenv("HF_TOKEN", "hf_env_token_xyz")
    token, source = exp1069._decrypt_sops_token()
    assert token == "hf_env_token_xyz"
    assert source == "env"


def test_decrypt_sops_token_returns_not_found_when_nothing(monkeypatch, tmp_path):
    """No SOPS, no env, no keyring => (None, 'not_found')."""
    monkeypatch.setattr(exp1069, "SECRETS_PATH", tmp_path / "missing.yaml")
    monkeypatch.setattr(exp1069.shutil, "which", lambda name: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "keyring", None)
    token, source = exp1069._decrypt_sops_token()
    assert token is None
    assert source == "not_found"


# ---------------------------------------------------------------------------
# _create_sops_stub
# ---------------------------------------------------------------------------


def test_create_sops_stub_refuses_to_clobber_existing(monkeypatch, tmp_path):
    """If the encrypted file already exists we must NOT overwrite it."""
    existing = tmp_path / "hf_token.enc.yaml"
    existing.write_text("real-encrypted-content\n", encoding="utf-8")
    monkeypatch.setattr(exp1069, "SECRETS_PATH", existing)
    assert exp1069._create_sops_stub() is False
    assert existing.read_text(encoding="utf-8") == "real-encrypted-content\n"


def test_create_sops_stub_returns_false_when_sops_missing(monkeypatch, tmp_path):
    """Without the sops CLI we cannot create an encrypted stub."""
    monkeypatch.setattr(exp1069, "SECRETS_PATH", tmp_path / "missing.yaml")
    monkeypatch.setattr(exp1069.shutil, "which", lambda name: None)
    assert exp1069._create_sops_stub() is False


# ---------------------------------------------------------------------------
# _read_exp1059_completion
# ---------------------------------------------------------------------------


def test_read_exp1059_completion_true_when_complete(monkeypatch, tmp_path):
    """Echoes the upstream artifact's space_code_complete=True flag."""
    p = tmp_path / "exp1059.json"
    p.write_text(json.dumps({"space_code_complete": True}), encoding="utf-8")
    monkeypatch.setattr(exp1069, "EXP1059_PATH", p)
    assert exp1069._read_exp1059_completion() is True


def test_read_exp1059_completion_false_when_missing(monkeypatch, tmp_path):
    """Missing or unreadable upstream artifact => False (do-not-deploy)."""
    monkeypatch.setattr(exp1069, "EXP1059_PATH", tmp_path / "absent.json")
    assert exp1069._read_exp1059_completion() is False


def test_read_exp1059_completion_false_on_corrupt(monkeypatch, tmp_path):
    """Corrupt JSON should not raise — just yield False."""
    p = tmp_path / "exp1059.json"
    p.write_text("not-json{{{", encoding="utf-8")
    monkeypatch.setattr(exp1069, "EXP1059_PATH", p)
    assert exp1069._read_exp1059_completion() is False


# ---------------------------------------------------------------------------
# main() artifact branches
# ---------------------------------------------------------------------------


def _patch_main_environment(monkeypatch, tmp_path):
    """Redirect on-disk state so main() never touches the real repo paths."""
    result = tmp_path / "exp1069.json"
    monkeypatch.setattr(exp1069, "RESULT_PATH", result)
    monkeypatch.setattr(exp1069, "EXP1059_PATH", tmp_path / "no_upstream.json")
    monkeypatch.setattr(exp1069, "SECRETS_PATH", tmp_path / "no_secrets.yaml")
    return result


def test_main_writes_blocked_artifact_when_no_token(monkeypatch, tmp_path):
    """No token, no env, no keyring => stub-created blocked verdict."""
    result_path = _patch_main_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(exp1069, "_decrypt_sops_token", lambda: (None, "not_found"))
    monkeypatch.setattr(exp1069, "_create_sops_stub", lambda: True)
    rc = exp1069.main()
    assert rc == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["honest_verdict"] == "hf_token_not_found_stub_created"
    assert payload["status"] == "blocked"
    assert payload["sops_stub_created"] is True
    assert payload["hf_token_found"] is False
    assert payload["deploy_attempted"] is False
    # Required schema fields per REQUIRED_RESULT_FIELDS
    for k in (
        "experiment",
        "schema",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "title",
    ):
        assert k in payload


def test_main_writes_failed_artifact_on_auth_failure(monkeypatch, tmp_path):
    """A token that fails whoami should result in honest_verdict=failed."""
    result_path = _patch_main_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(exp1069, "_decrypt_sops_token", lambda: ("hf_test", "sops"))
    monkeypatch.setattr(exp1069, "_hf_login", lambda token: (False, "401 invalid token"))
    rc = exp1069.main()
    assert rc == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["honest_verdict"] == "failed"
    assert payload["deploy_attempted"] is False
    assert payload["hf_token_found"] is True


def test_main_writes_failed_artifact_when_upload_raises(monkeypatch, tmp_path):
    """Upload exception path => honest_verdict=failed, upload_error set."""
    result_path = _patch_main_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(exp1069, "_decrypt_sops_token", lambda: ("hf_test", "sops"))
    monkeypatch.setattr(exp1069, "_hf_login", lambda token: (True, "ian"))
    monkeypatch.setattr(exp1069, "_upload_space", lambda token: (False, "upload exploded"))
    rc = exp1069.main()
    assert rc == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["honest_verdict"] == "failed"
    assert payload["deploy_attempted"] is True
    assert payload["space_deployed"] is False
    assert payload["upload_error"] == "upload exploded"


def test_main_records_verify_pending_when_http_not_200(monkeypatch, tmp_path):
    """Upload OK but live URL not yet 200 => deploy_attempted_verify_pending."""
    result_path = _patch_main_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(exp1069, "_decrypt_sops_token", lambda: ("hf_test", "sops"))
    monkeypatch.setattr(exp1069, "_hf_login", lambda token: (True, "ian"))
    monkeypatch.setattr(exp1069, "_upload_space", lambda token: (True, None))
    monkeypatch.setattr(exp1069, "_verify_live", lambda url: (False, 503))
    rc = exp1069.main()
    assert rc == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["honest_verdict"] == "deploy_attempted_verify_pending"
    assert payload["status"] == "success"
    assert payload["space_deployed"] is False
    assert payload["live_http_status"] == 503
    assert payload["live_url"].endswith(exp1069.SPACE_REPO_ID)


def test_main_records_deployed_live_when_http_200(monkeypatch, tmp_path):
    """The happy path: upload OK and the Space returns 200."""
    result_path = _patch_main_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(exp1069, "_decrypt_sops_token", lambda: ("hf_test", "sops"))
    monkeypatch.setattr(exp1069, "_hf_login", lambda token: (True, "ian"))
    monkeypatch.setattr(exp1069, "_upload_space", lambda token: (True, None))
    monkeypatch.setattr(exp1069, "_verify_live", lambda url: (True, 200))
    rc = exp1069.main()
    assert rc == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["honest_verdict"] == "deployed_live"
    assert payload["status"] == "success"
    assert payload["space_deployed"] is True
    assert payload["live_http_status"] == 200
