"""Read-only distribution mirror readiness audit for Exp 3770.

This module prepares the operator's publication checklist without doing any of
the publication work. The distinction matters because the repository can carry
working credentials and release automation, but external publication still
requires a human operator to review the artifact and run the final commands.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


OUTPUT_REL_PATH = Path("results/experiment_3770_distribution_mirror_publish_checklist.json")
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a readiness audit over "
    "configs/docs, no live model)."
)
RANDOM_SEED = 3770
OPERATOR_ONLY = "OPERATOR ACTION -- agent must not execute"
READY_VERDICT_PREFIX = "complete: distribution_mirror_readiness_audited"
COMPUTE_BOUND_MARKERS = ("GGUF", "CUDA")

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "pypi_workflow_ready",
    "hf_mirror_documented",
    "ipfs_plan_documented",
    "operator_publish_checklist",
    "agent_published_nothing",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the readiness-audit outcome.",
    "inference_substrate": "Readiness audit over configs/docs, no live model.",
    "pypi_workflow_ready": (
        "BARE bool -- is the OIDC trusted-publishing workflow present and "
        "configured."
    ),
    "hf_mirror_documented": (
        "BARE bool -- is the HuggingFace primary mirror channel named/ready."
    ),
    "ipfs_plan_documented": (
        "BARE bool -- is the IPFS content-addressed secondary channel planned."
    ),
    "operator_publish_checklist": (
        "Ordered OPERATOR-ONLY publish steps; the agent prepares but never executes."
    ),
    "agent_published_nothing": (
        "BARE bool, MUST be true -- capability does not imply authorization."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}


def _read_text(path: Path) -> str:
    """Return checked-in text when present and an empty string otherwise."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_json_object(path: Path) -> dict[str, Any]:
    """Load a local JSON object defensively for audit evidence.

    A missing or malformed file is not exceptional for readiness auditing; it
    simply means the corresponding readiness check must fail closed.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _bool_word(value: bool) -> str:
    return "true" if value else "false"


def _audit_pypi_workflow(root: Path) -> dict[str, Any]:
    workflow_path = root / ".github" / "workflows" / "publish-pypi.yml"
    text = _read_text(workflow_path)
    checks = {
        "workflow_present": workflow_path.exists(),
        "tag trigger v*": "tags:" in text and "v*" in text,
        "pypa trusted publisher action": "pypa/gh-action-pypi-publish" in text,
        "pypi environment": re.search(r"environment:\s*\n\s*name:\s*pypi", text) is not None
        or "environment: pypi" in text,
        "id-token: write": "id-token: write" in text,
    }
    return {
        "ready": all(checks.values()),
        "path": workflow_path.relative_to(root).as_posix(),
        "checks": checks,
        "missing": [name for name, passed in checks.items() if not passed],
    }


def _audit_hf_mirror(root: Path) -> dict[str, Any]:
    pyproject = _read_text(root / "pyproject.toml")
    hf_plan = _read_text(root / "docs" / "huggingface-plan.md")
    mirror_table = _read_text(root / "docs" / "ipfs_mirror_table.md")
    combined = "\n".join((pyproject, hf_plan, mirror_table))
    repo_matches = sorted(set(re.findall(r"Carnot-EBM/[A-Za-z0-9_.-]+", combined)))
    checks = {
        "org named": "Carnot-EBM" in combined or "https://huggingface.co/Carnot-EBM" in combined,
        "repository named": bool(repo_matches),
        "model hub url": "https://huggingface.co/Carnot-EBM" in combined,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "repositories": repo_matches,
        "primary_channel": "https://huggingface.co/Carnot-EBM",
        "missing": [name for name, passed in checks.items() if not passed],
    }


def _audit_ipfs_plan(root: Path) -> dict[str, Any]:
    mirror_table = _read_text(root / "docs" / "ipfs_mirror_table.md")
    anchor_plan = _read_text(root / "docs" / "ipfs_anchor_placeholder.md")
    manifest = _read_json_object(root / "results" / "ipfs_mirrors.json")
    entries = manifest.get("entries") if isinstance(manifest.get("entries"), dict) else {}
    combined = "\n".join((mirror_table, anchor_plan, json.dumps(manifest, sort_keys=True)))
    cid_matches = sorted(set(re.findall(r"\bQm[1-9A-HJ-NP-Za-km-z]{10,}\b", combined)))
    checks = {
        "cid documented": bool(cid_matches) or bool(entries),
        "pinning plan": any(
            service in combined
            for service in ("web3.storage", "Storj", "Filebase", "pinning service")
        ),
        "ipfs add plan": "ipfs add" in combined,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "cid_count": len(cid_matches),
        "manifest_entry_count": len(entries),
        "missing": [name for name, passed in checks.items() if not passed],
    }


def audit_readiness(root: Path) -> dict[str, Any]:
    """Audit release-channel readiness from local repository evidence only."""
    pypi = _audit_pypi_workflow(root)
    hf = _audit_hf_mirror(root)
    ipfs = _audit_ipfs_plan(root)
    return {"pypi": pypi, "huggingface": hf, "ipfs": ipfs}


def operator_publish_checklist() -> list[dict[str, str]]:
    """Return the exact operator-side commands without executing them."""
    return [
        {
            "channel": "pypi",
            "operator_only": OPERATOR_ONLY,
            "action": "create the release tag that triggers trusted publishing",
            "command": 'git tag v<version> -m "Release v<version>"',
        },
        {
            "channel": "pypi",
            "operator_only": OPERATOR_ONLY,
            "action": "push the tag so GitHub Actions publishes via PyPI OIDC",
            "command": "git push origin v<version>",
        },
        {
            "channel": "huggingface",
            "operator_only": OPERATOR_ONLY,
            "action": "upload the primary mirror artifact into the Carnot-EBM org",
            "command": (
                "huggingface-cli upload Carnot-EBM/<repo> <artifact_path> "
                "--repo-type model"
            ),
        },
        {
            "channel": "ipfs",
            "operator_only": OPERATOR_ONLY,
            "action": "create and locally pin the content-addressed secondary mirror",
            "command": "CID=$(ipfs add -r -Q --pin <artifact_path>)",
        },
        {
            "channel": "ipfs",
            "operator_only": OPERATOR_ONLY,
            "action": "durably pin the CID with an operator-controlled pinning service",
            "command": "w3 space add $CID  # or pin the CID via Storj/Filebase",
        },
        {
            "channel": "records",
            "operator_only": OPERATOR_ONLY,
            "action": "record the CID beside the HF mirror and in the IPFS manifest",
            "command": (
                "edit docs/ipfs_mirror_table.md and results/ipfs_mirrors.json "
                "with the new CID"
            ),
        },
    ]


def _stable_checksum_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: artifact[key]
        for key in sorted(artifact)
        if key not in {"duration_s", "reproducibility_checksum"}
    }


def reproducibility_checksum(artifact: dict[str, Any]) -> str:
    """Hash stable artifact content so later drift is detectable."""
    encoded = json.dumps(
        _stable_checksum_payload(artifact),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _verdict(pypi_ready: bool, hf_ready: bool, ipfs_ready: bool) -> str:
    return (
        f"{READY_VERDICT_PREFIX}_pypi_{_bool_word(pypi_ready)}_hf_"
        f"{_bool_word(hf_ready)}_ipfs_{_bool_word(ipfs_ready)}_"
        "operator_checklist_emitted_agent_published_nothing"
    )


def build_artifact(
    root: Path,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Build the Exp 3770 artifact without any external publication calls."""
    start = time.perf_counter() if started_s is None else started_s
    audit = audit_readiness(root)
    end = time.perf_counter() if now_s is None else now_s
    pypi_ready = bool(audit["pypi"]["ready"])
    hf_ready = bool(audit["huggingface"]["ready"])
    ipfs_ready = bool(audit["ipfs"]["ready"])
    artifact: dict[str, Any] = {
        "schema": "carnot.distribution_mirror_publish_checklist.v1",
        "experiment": 3770,
        "honest_verdict": _verdict(pypi_ready, hf_ready, ipfs_ready),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "pypi_workflow_ready": pypi_ready,
        "hf_mirror_documented": hf_ready,
        "ipfs_plan_documented": ipfs_ready,
        "readiness_audit": audit,
        "operator_publish_checklist": operator_publish_checklist(),
        "agent_published_nothing": True,
        "commands_not_executed": [
            "git tag",
            "huggingface-cli upload",
            "ipfs add",
            "gh release create",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(end - start, 0.0001), 6),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Fail closed when the artifact no longer satisfies REQ-PUBLISH-3770."""
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith(READY_VERDICT_PREFIX):
        raise ValueError("terminal verdict does not use the Exp 3770 prefix")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must declare aggregation/docs only")
    for field in (
        "pypi_workflow_ready",
        "hf_mirror_documented",
        "ipfs_plan_documented",
        "agent_published_nothing",
    ):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a bare bool")
    if artifact["agent_published_nothing"] is not True:
        raise ValueError("agent_published_nothing must remain true")
    checklist = artifact["operator_publish_checklist"]
    if not isinstance(checklist, list) or not checklist:
        raise ValueError("operator_publish_checklist must be a non-empty list")
    for row in checklist:
        if not isinstance(row, dict) or row.get("operator_only") != OPERATOR_ONLY:
            raise ValueError("every checklist entry must be operator-only")
    encoded = json.dumps(artifact, sort_keys=True)
    if any(marker in encoded for marker in COMPUTE_BOUND_MARKERS):
        raise ValueError("artifact contains forbidden compute-bound markers")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match artifact content")


def write_artifact(
    root: Path,
    *,
    output_path: str | Path = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the local readiness artifact and return its path."""
    artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    validate_artifact(artifact)
    output = root / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
