"""Tests for experiment 1125 — HuggingFace Spaces gallery update (Hashi).

REQ-WOPR-GALLERY-002: HashiGame must appear in ALL_GAMES.
SCENARIO-DEPLOY-002: gallery update script produces a valid artifact
with the schema fields the conductor expects.
SCENARIO-DEPLOY-003: README carries the .87-milestone benchmark markers.

Tests cover ONLY the experiment-1125 script and its artifact.  The
underlying cartridge code is exercised by the exp1124 test module.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiment_1125_hf_spaces_gallery_update.py"
RESULT_PATH = REPO_ROOT / "results" / "experiment_1125_hf_spaces_gallery_update.json"
INIT_PATH = REPO_ROOT / "spaces" / "wopr-games" / "games" / "__init__.py"
HASHI_PATH = REPO_ROOT / "spaces" / "wopr-games" / "games" / "hashi.py"
README_PATH = REPO_ROOT / "spaces" / "wopr-games" / "README.md"


# ---------- helpers ----------


def _load_module():
    """Import the experiment script as a module without running main()."""
    spec = importlib.util.spec_from_file_location("exp1125_module", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_result() -> dict:
    assert RESULT_PATH.exists(), f"Result artifact missing: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text())


# ---------- artifact-shape tests ----------


def test_result_artifact_exists() -> None:
    # SCENARIO-DEPLOY-002: artifact must be produced.
    assert RESULT_PATH.exists()


def test_result_required_fields_present() -> None:
    """Schema fields the conductor's task spec requires."""
    required = [
        "hashi_cartridge_deployed",
        "benchmark_results_updated",
        "space_url",
        "gallery_updated",
        "honest_verdict",
    ]
    result = _load_result()
    for field in required:
        assert field in result, f"Missing field: {field}"


def test_result_honest_verdict_in_enum() -> None:
    valid = {
        "deployed_live",
        "local_only_hf_token_unavailable",
        "partial",
        "failed",
    }
    assert _load_result()["honest_verdict"] in valid


def test_result_types_match_schema() -> None:
    """Field types must match the conductor's expected schema."""
    result = _load_result()
    assert isinstance(result["hashi_cartridge_deployed"], bool)
    assert isinstance(result["benchmark_results_updated"], bool)
    assert isinstance(result["gallery_updated"], bool)
    # space_url is str | None
    assert result["space_url"] is None or isinstance(result["space_url"], str)
    assert isinstance(result["honest_verdict"], str)


# ---------- local-state tests ----------


def test_hashi_cartridge_file_present() -> None:
    # REQ-WOPR-GALLERY-002 prerequisite — the cartridge source is on disk.
    assert HASHI_PATH.exists(), "spaces/wopr-games/games/hashi.py missing"


def test_hashi_registered_in_all_games() -> None:
    # REQ-WOPR-GALLERY-002 — Hashi wired into ALL_GAMES.
    assert INIT_PATH.exists()
    content = INIT_PATH.read_text()
    assert "HashiGame" in content
    assert "ALL_GAMES" in content
    assert "HashiGame()" in content


def test_readme_carries_benchmark_markers() -> None:
    # SCENARIO-DEPLOY-003 — every required marker present.
    module = _load_module()
    ok, missing = module._benchmark_results_present(README_PATH, module.BENCHMARK_MARKERS)
    assert ok, f"Missing benchmark markers: {missing}"


# ---------- helper-function unit tests ----------


def test_count_cartridges_at_least_six() -> None:
    """After exp1124, ALL_GAMES has at least 6 cartridges."""
    module = _load_module()
    assert module._count_cartridges(INIT_PATH) >= 6


def test_cartridge_registered_helper(tmp_path) -> None:
    """``_cartridge_registered`` returns True only when name AND
    instantiation appear together with ALL_GAMES marker."""
    module = _load_module()
    fake = tmp_path / "init.py"
    fake.write_text("ALL_GAMES = [HashiGame()]\n")
    assert module._cartridge_registered(fake, "HashiGame") is True
    fake.write_text("HashiGame  # mentioned but no ALL_GAMES list\n")
    assert module._cartridge_registered(fake, "HashiGame") is False
    assert module._cartridge_registered(tmp_path / "missing.py", "HashiGame") is False


def test_benchmark_results_present_missing(tmp_path) -> None:
    """Helper reports missing markers when the README is empty."""
    module = _load_module()
    empty = tmp_path / "README.md"
    empty.write_text("# stub\n")
    ok, missing = module._benchmark_results_present(empty, ["0.9946", "exp1111"])
    assert ok is False
    assert "0.9946" in missing and "exp1111" in missing


def test_decide_verdict_branches() -> None:
    """All four enum branches reachable for the expected input combos."""
    module = _load_module()

    # Local files broken → failed regardless of token.
    assert (
        module._decide_verdict(
            hf_token_found=True,
            hashi_cartridge_local=False,
            benchmark_results_updated=True,
            deploy_attempted=False,
            gallery_updated=False,
            live_http_status=0,
        )
        == "failed"
    )

    # Local OK but no token.
    assert (
        module._decide_verdict(
            hf_token_found=False,
            hashi_cartridge_local=True,
            benchmark_results_updated=True,
            deploy_attempted=False,
            gallery_updated=False,
            live_http_status=0,
        )
        == "local_only_hf_token_unavailable"
    )

    # Deploy succeeded and live.
    assert (
        module._decide_verdict(
            hf_token_found=True,
            hashi_cartridge_local=True,
            benchmark_results_updated=True,
            deploy_attempted=True,
            gallery_updated=True,
            live_http_status=200,
        )
        == "deployed_live"
    )

    # Deploy attempted but did not become live.
    assert (
        module._decide_verdict(
            hf_token_found=True,
            hashi_cartridge_local=True,
            benchmark_results_updated=True,
            deploy_attempted=True,
            gallery_updated=False,
            live_http_status=0,
        )
        == "partial"
    )


def test_retrieve_hf_token_env_fallback(monkeypatch) -> None:
    """When SOPS files are absent, env var is used."""
    module = _load_module()
    monkeypatch.setenv("HF_TOKEN", "hf_test_dummy_xyz")
    # Force SOPS to fail by pointing it at a nonexistent key file.
    monkeypatch.setenv("SOPS_AGE_KEY_FILE", "/nonexistent/age/key.txt")
    token, source = module._retrieve_hf_token()
    # Either SOPS still worked (real keys present) or env fallback fired.
    assert token is not None
    assert source.startswith("sops:") or source == "env:HF_TOKEN"


def test_retrieve_hf_token_not_found(monkeypatch, tmp_path) -> None:
    """When no SOPS file decrypts and env is empty, return None."""
    module = _load_module()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Run the function from a tmp working directory so SOPS finds no secrets.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    token, source = module._retrieve_hf_token()
    assert token is None
    assert source == "not_found"
