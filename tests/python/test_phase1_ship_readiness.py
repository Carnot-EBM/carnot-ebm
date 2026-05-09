"""Tests for the Exp 1582 Phase 1 software ship-readiness ledger.

Spec: REQ-PUBLISH-024, SCENARIO-PUBLISH-026.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.phase1_ship_readiness import (
    REQUIRED_ARTIFACT_FIELDS,
    REQUIRED_PER_TOKEN_EXPORTS,
    build_readiness_report,
    main,
    render_markdown,
    run,
    write_in_progress_artifact,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _base_repo(root: Path, *, ready: bool) -> None:
    package_name = "carnot-ebm"
    optional_deps = '[project.optional-dependencies]\nhf = ["huggingface_hub>=0.23"]' if ready else ""
    public_install = package_name if ready else "carnot"
    tool_count = "9" if ready else "7"
    mcp_command = '"args": ["-m", "carnot.mcp"]' if ready else '"args": ["tools/verify-mcp/server.py"]'
    hf_history = (
        "All documented artifacts have model cards and local staging files."
        if ready
        else "Action needed: Save the trained EBM weights from experiment 25 to safetensors."
    )

    _write(
        root / "pyproject.toml",
        f"""
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
dynamic = ["version"]
license = {{text = "Apache-2.0"}}
dependencies = [
    "jax>=0.4.30",
    "safetensors>=0.4",
]

{optional_deps}

[project.scripts]
carnot = "carnot.cli:main"

[tool.setuptools.package-data]
"carnot.schemas" = ["*.json"]
"carnot.sampling._vendored_thrml" = ["LICENSE", "py.typed"]

[tool.setuptools.dynamic]
version = {{attr = "carnot._version.__version__"}}
""",
    )
    _write(root / "LICENSE", "Apache License\nVersion 2.0\n")
    _write(root / "python/carnot/_version.py", '__version__ = "0.1.0b1"\n')
    _write(root / "python/carnot/schemas/session_memory_v1.json", "{}\n")
    _write(root / "python/carnot/sampling/_vendored_thrml/LICENSE", "Apache License\n")
    _write(root / "python/carnot/sampling/_vendored_thrml/py.typed", "")
    _write(
        root / "README.md",
        f"""
# Carnot

Install with `pip install {public_install}`.

The MCP server exposes **{tool_count}** tools.

{'Published model cards are synchronized with local exports.' if ready else 'Two Phase 1 research artifacts are published to HuggingFace.'}
""",
    )
    _write(
        root / "docs/getting-started.md",
        f"Use `pip install {public_install}` and run `python -m carnot.mcp`.\n",
    )
    _write(
        root / "docs/usage-guide.md",
        f"""
Install with `pip install {public_install}`.
Use `carnot verify examples/math_funcs.py --func gcd --test "(12,8):4"`.
Use `carnot verify-code examples/math_funcs.py --func gcd --pbt`.
{mcp_command}
""",
    )
    _write(root / "examples/README.md", f"Install with `pip install {public_install}`.\n")
    _write(root / ".mcp.json.example", "{\n  " + mcp_command + "\n}\n")
    _write(root / "docs/huggingface-plan.md", hf_history)
    if ready:
        _write(root / "docs/integrator-guide.md", "Clone, install, run CLI, run MCP.\n")
        _write(root / "data/token_activations_tqa_qwen35.safetensors", "placeholder")

    for model_id in REQUIRED_PER_TOKEN_EXPORTS:
        _write(root / f"exports/{model_id}/README.md", "---\nlicense: apache-2.0\n---\n")
        _write(root / f"exports/{model_id}/config.json", "{}\n")
        _write(root / f"exports/{model_id}/model.safetensors", "placeholder")
        _write(root / f"exports/{model_id}/training_metadata.json", "{}\n")

    mirror_payload: dict[str, object]
    if ready:
        mirror_payload = {
            "per_token_ebm_exports": {"cid": "QmPerToken"},
            "vjepa_v2": {"cid": "QmVjepa"},
            "estimation_verifier_v1": {"cid": "QmEstimation"},
            "pypi_sdist": {"cid": "QmSdist"},
        }
    else:
        mirror_payload = {
            "vjepa_v2": {"cid": "QmVjepa"},
            "estimation_verifier_v1": {"cid": "QmEstimation"},
        }
    _write_json(root / "results/ipfs_mirrors.json", mirror_payload)


def test_write_in_progress_artifact_seeds_required_fields(tmp_path: Path) -> None:
    """REQ-PUBLISH-024 requires the status=in_progress artifact to exist first."""

    path = tmp_path / "results/experiment_1582_phase1_ship_readiness_ledger.json"
    artifact = write_in_progress_artifact(path)

    assert artifact["status"] == "in_progress"
    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact


def test_report_records_blockers_for_unready_release_surface(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-026 blocks ship when software gate evidence is missing."""

    _base_repo(tmp_path, ready=False)

    report = build_readiness_report(tmp_path, safe_local_smoke_ran=True)

    assert report["status"] == "complete"
    assert report["phase1_ship_readiness_ledger_ready"] is True
    assert report["pypi_package_ready"] is False
    assert report["hf_mirror_ready"] is False
    assert report["second_mirror_ready"] is False
    assert report["mcp_cli_docs_ready"] is False
    assert report["independent_reproducer_path_ready"] is True
    assert report["safe_local_smoke_ran"] is True
    assert report["blocking_items_count"] == len(report["blocking_items"])
    assert report["honest_verdict"].startswith("blocked_")
    assert any("pip install carnot" in item["details"] for item in report["blocking_items"])
    assert any("token_activations_tqa_qwen35" in item["details"] for item in report["blocking_items"])
    assert "Resolve every blocker" in render_markdown(report)


def test_run_handles_empty_repo_with_all_missing_gate_evidence(tmp_path: Path) -> None:
    """REQ-PUBLISH-024 records exact blockers when metadata files are absent."""

    _write(
        tmp_path / "pyproject.toml",
        """
[project]
name = "wrong-name"
license = {text = "MIT"}

[project.scripts]
wrong = "wrong:main"

[tool.setuptools.package-data]
"carnot.schemas" = ["missing.json"]
""",
    )

    artifact = run(tmp_path, safe_local_smoke_ran=False)

    assert artifact["phase1_ship_readiness_ledger_ready"] is True
    assert artifact["pypi_package_ready"] is False
    assert artifact["independent_reproducer_path_ready"] is False
    assert artifact["blocking_items_count"] == len(artifact["blocking_items"])
    assert any("project.name" in item["details"] for item in artifact["blocking_items"])
    assert any("package-data" in item["details"] for item in artifact["blocking_items"])
    assert any("per-token HuggingFace export staging" in item["details"] for item in artifact["blocking_items"])


def test_report_handles_absent_pyproject(tmp_path: Path) -> None:
    """REQ-PUBLISH-024 keeps the ledger terminal when pyproject.toml is absent."""

    report = build_readiness_report(tmp_path, safe_local_smoke_ran=False)

    assert report["pypi_package_ready"] is False
    assert any("project.name" in item["details"] for item in report["blocking_items"])


def test_run_writes_ready_artifacts_when_all_gates_pass(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-026 emits ready verdict when every software gate passes."""

    _base_repo(tmp_path, ready=True)

    artifact = run(tmp_path, safe_local_smoke_ran=True)
    result_path = tmp_path / "results/experiment_1582_phase1_ship_readiness_ledger.json"
    ledger_path = tmp_path / "ops/phase1_ship_readiness.md"

    assert artifact["honest_verdict"] == "phase1_software_ship_ready"
    assert artifact["blocking_items_count"] == 0
    assert artifact["pypi_package_ready"] is True
    assert artifact["hf_mirror_ready"] is True
    assert artifact["second_mirror_ready"] is True
    assert artifact["mcp_cli_docs_ready"] is True
    assert artifact["independent_reproducer_path_ready"] is True
    assert result_path.exists()
    assert ledger_path.read_text(encoding="utf-8").startswith("# Phase 1 Ship Readiness")
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact


def test_main_returns_zero_after_writing_artifact(tmp_path: Path) -> None:
    """REQ-PUBLISH-024 exposes a non-publishing local ledger command."""

    _base_repo(tmp_path, ready=True)

    assert main([str(tmp_path), "--safe-local-smoke-ran"]) == 0
    artifact = json.loads(
        (tmp_path / "results/experiment_1582_phase1_ship_readiness_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["phase1_ship_readiness_ledger_ready"] is True
