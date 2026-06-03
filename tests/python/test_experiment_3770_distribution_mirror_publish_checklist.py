"""Tests for Exp 3770 distribution mirror readiness.

Spec refs: REQ-PUBLISH-3770, SCENARIO-PUBLISH-3770.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import distribution_mirror_publish_checklist_3770 as mod


SPEC_PATH = Path("openspec/capabilities/publication/spec.md")


def _seed_ready_repo(root: Path) -> None:
    (root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "carnot-ebm"',
                "[project.urls]",
                '"Model Hub" = "https://huggingface.co/Carnot-EBM"',
            ]
        ),
        encoding="utf-8",
    )
    (root / ".github" / "workflows" / "publish-pypi.yml").write_text(
        "\n".join(
            [
                "name: Publish carnot-ebm to PyPI",
                "on:",
                "  push:",
                "    tags:",
                '      - "v*"',
                "jobs:",
                "  publish:",
                "    environment:",
                "      name: pypi",
                "    permissions:",
                "      id-token: write",
                "    steps:",
                "      - uses: pypa/gh-action-pypi-publish@release/v1",
            ]
        ),
        encoding="utf-8",
    )
    (root / "docs" / "huggingface-plan.md").write_text(
        "\n".join(
            [
                "# HuggingFace Publishing Plan",
                "## Organization: Carnot-EBM",
                "#### 1. `Carnot-EBM/per-token-ebm-qwen3-06b`",
                "Upload with the operator's HuggingFace credentials.",
            ]
        ),
        encoding="utf-8",
    )
    (root / "docs" / "ipfs_mirror_table.md").write_text(
        "\n".join(
            [
                "# Carnot-EBM IPFS Mirror Manifest",
                "| Repo | Type | CID |",
                "|---|---|---|",
                "| `Carnot-EBM/per-token-ebm-qwen3-06b` | model | `Qmabcdef1234567890` |",
                "Pinning service: web3.storage / Storj / Filebase.",
            ]
        ),
        encoding="utf-8",
    )
    (root / "docs" / "ipfs_anchor_placeholder.md").write_text(
        "\n".join(
            [
                "# IPFS Distribution",
                "CID values are recorded after `ipfs add`.",
                "Durable pinning uses web3.storage, Storj, or Filebase.",
                "For PyPI sdist mirroring: ipfs add -r -Q --pin /tmp/carnot-sdist",
            ]
        ),
        encoding="utf-8",
    )
    (root / "results" / "ipfs_mirrors.json").write_text(
        json.dumps(
            {
                "entries": {
                    "Carnot-EBM/per-token-ebm-qwen3-06b": {
                        "cid": "Qmabcdef1234567890",
                        "pinned": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_req_publish_3770_spec_anchor_exists() -> None:
    """REQ-PUBLISH-3770: OpenSpec declares the exact checklist artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-PUBLISH-3770" in spec
    assert "SCENARIO-PUBLISH-3770" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "OPERATOR ACTION -- agent must not execute" in spec


def test_scenario_publish_3770_ready_artifact(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-3770: ready inputs produce an operator-only artifact."""
    _seed_ready_repo(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.25)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: distribution_mirror_readiness_audited_pypi_true_hf_true_"
        "ipfs_true_operator_checklist_emitted_agent_published_nothing"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["pypi_workflow_ready"] is True
    assert artifact["hf_mirror_documented"] is True
    assert artifact["ipfs_plan_documented"] is True
    assert artifact["agent_published_nothing"] is True
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["duration_s"] == pytest.approx(1.25)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    checklist = artifact["operator_publish_checklist"]
    assert [row["channel"] for row in checklist] == [
        "pypi",
        "pypi",
        "huggingface",
        "ipfs",
        "ipfs",
        "records",
    ]
    assert all(
        row["operator_only"] == "OPERATOR ACTION -- agent must not execute"
        for row in checklist
    )
    commands = "\n".join(row["command"] for row in checklist)
    assert "git tag v<version>" in commands
    assert "git push origin v<version>" in commands
    assert "huggingface-cli upload Carnot-EBM/<repo>" in commands
    assert "ipfs add -r -Q --pin" in commands
    assert "gh release create" not in commands
    assert "GGUF" not in json.dumps(artifact)
    assert "CUDA" not in json.dumps(artifact)


def test_req_publish_3770_negative_readiness_is_recorded(tmp_path: Path) -> None:
    """REQ-PUBLISH-3770: missing evidence records not-ready without publishing."""
    _seed_ready_repo(tmp_path)
    (tmp_path / ".github" / "workflows" / "publish-pypi.yml").write_text(
        "name: no oidc\npermissions:\n  contents: read\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "huggingface-plan.md").write_text(
        "No mirror repository named here.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "ipfs_mirror_table.md").write_text(
        "No content-addressed CID table yet.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "ipfs_anchor_placeholder.md").write_text(
        "No pinning plan yet.\n",
        encoding="utf-8",
    )
    (tmp_path / "results" / "ipfs_mirrors.json").write_text("{}", encoding="utf-8")

    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.5)

    mod.validate_artifact(artifact)
    assert artifact["pypi_workflow_ready"] is False
    assert artifact["hf_mirror_documented"] is False
    assert artifact["ipfs_plan_documented"] is False
    assert artifact["honest_verdict"] == (
        "complete: distribution_mirror_readiness_audited_pypi_false_hf_false_"
        "ipfs_false_operator_checklist_emitted_agent_published_nothing"
    )
    assert artifact["readiness_audit"]["pypi"]["missing"] == [
        "tag trigger v*",
        "pypa trusted publisher action",
        "pypi environment",
        "id-token: write",
    ]
    assert "Carnot-EBM/<repo>" in artifact["operator_publish_checklist"][2]["command"]
    assert artifact["agent_published_nothing"] is True


def test_req_publish_3770_write_artifact_and_validation_edges(tmp_path: Path) -> None:
    """REQ-PUBLISH-3770: writer persists JSON and rejects schema drift."""
    _seed_ready_repo(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=2.0)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert output == tmp_path / mod.OUTPUT_REL_PATH

    missing = dict(artifact)
    missing.pop("operator_publish_checklist")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_bool = dict(artifact, pypi_workflow_ready={"value": True})
    with pytest.raises(ValueError, match="pypi_workflow_ready"):
        mod.validate_artifact(bad_bool)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        mod.validate_artifact(bad_verdict)

    bad_substrate = dict(artifact, inference_substrate="live model")
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_agent = dict(artifact, agent_published_nothing=False)
    with pytest.raises(ValueError, match="agent_published_nothing"):
        mod.validate_artifact(bad_agent)

    empty_checklist = dict(artifact, operator_publish_checklist=[])
    with pytest.raises(ValueError, match="operator_publish_checklist"):
        mod.validate_artifact(empty_checklist)

    bad_checklist = dict(
        artifact,
        operator_publish_checklist=[
            {**artifact["operator_publish_checklist"][0], "operator_only": "agent action"}
        ],
    )
    with pytest.raises(ValueError, match="operator-only"):
        mod.validate_artifact(bad_checklist)

    bad_marker = dict(artifact, forbidden_marker="CUDA")
    with pytest.raises(ValueError, match="compute-bound markers"):
        mod.validate_artifact(bad_marker)

    bad_checksum = dict(artifact, reproducibility_checksum="0" * 64)
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_publish_3770_reader_helpers_cover_absent_and_invalid_json(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-3770: defensive local readers keep the audit read-only."""
    assert mod._read_text(tmp_path / "missing.txt") == ""
    assert mod._read_json_object(tmp_path / "missing.json") == {}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert mod._read_json_object(invalid) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json_object(list_json) == {}
