"""Stable resolver for local evaluation dataset manifests.

Exp 2849 writes date-suffixed JSONL manifest filenames.  Downstream evaluation
tasks should read those paths from the Exp 2849 artifact instead of guessing
plain aliases such as ``halueval.jsonl``.  This module makes that lookup small,
checksum-verified, and reusable across HaluEval, FEVER, and later corpora.

Spec: REQ-BENCH-2863, SCENARIO-BENCH-2863.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
MATERIALIZATION_ARTIFACT_REL_PATH = Path(
    "results/experiment_2849_local_dataset_materialization_v1.json"
)
OUTPUT_REL_PATH = Path("results/experiment_2863_eval_manifest_contract_v2.json")
CANONICAL_CORPORA = ("halueval", "fever", "mbpp", "humaneval", "truthfulqa")

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict for conductor classification.",
    "manifest_contract_ready": "True only when every requested corpus resolves and verifies.",
    "manifest_source_artifact": "Exp 2849 is the naming and checksum authority.",
    "resolved_manifest_paths": "Dated JSONL paths are echoed exactly from Exp 2849.",
    "resolved_manifest_sha256": "Checksums are copied from Exp 2849 and verified on disk.",
    "readiness_booleans": "Per-corpus readiness combines Exp 2849 readiness and checksum proof.",
    "synthetic_rows_created": "Always false; the resolver never builds benchmark rows.",
    "tests_run": "Commands used to validate the resolver contract.",
    "duration_s": "Measured wall-clock runtime; no sleep padding.",
    "run_date": "Fixed experiment date for milestone .270 artifacts.",
}


@dataclass(frozen=True)
class ResolvedManifest:
    """Resolved path, checksum, and readiness for one evaluation corpus."""

    corpus: str
    path: str
    sha256: str
    count: int
    source_ready: bool
    checksum_verified: bool
    contract_ready: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "corpus": self.corpus,
            "path": self.path,
            "sha256": self.sha256,
            "count": self.count,
            "source_ready": self.source_ready,
            "checksum_verified": self.checksum_verified,
            "contract_ready": self.contract_ready,
            "detail": self.detail,
        }


def _artifact_path(repo_root: Path, source_artifact: Path | str) -> Path:
    path = Path(source_artifact)
    return path if path.is_absolute() else repo_root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_contract(
    *,
    repo_root: Path = REPO_ROOT,
    source_artifact: Path | str = MATERIALIZATION_ARTIFACT_REL_PATH,
    corpora: Iterable[str] = CANONICAL_CORPORA,
) -> dict[str, ResolvedManifest]:
    """Resolve corpus names to Exp 2849 manifest paths and verified checksums."""

    repo_root = Path(repo_root)
    artifact_path = _artifact_path(repo_root, source_artifact)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    paths = dict(payload.get("manifest_paths") or {})
    checksums = dict(payload.get("manifest_sha256") or {})
    counts = dict(payload.get("manifest_counts") or {})
    status = dict(payload.get("dataset_status") or {})
    resolved: dict[str, ResolvedManifest] = {}

    for corpus in corpora:
        key = str(corpus).lower()
        path_value = str(paths.get(key) or "")
        manifest_path = Path(path_value)
        resolved_path = manifest_path if manifest_path.is_absolute() else repo_root / manifest_path
        declared_sha = str(checksums.get(key) or "")
        actual_sha = _sha256(resolved_path) if resolved_path.is_file() else ""
        checksum_verified = bool(declared_sha and actual_sha == declared_sha)
        source_ready = bool(payload.get(f"{key}_ready"))
        count = int(counts.get(key) or 0)
        detail = str(dict(status.get(key) or {}).get("detail") or "")
        if not resolved_path.is_file() or not checksum_verified:
            detail = (
                f"{detail}; resolved_file_exists={resolved_path.is_file()}; "
                f"checksum_verified={checksum_verified}"
            )
        resolved[key] = ResolvedManifest(
            corpus=key,
            path=str(resolved_path),
            sha256=declared_sha,
            count=count,
            source_ready=source_ready,
            checksum_verified=checksum_verified,
            contract_ready=bool(source_ready and checksum_verified),
            detail=detail.strip("; "),
        )
    return resolved


def build_contract_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    source_artifact: Path | str = MATERIALIZATION_ARTIFACT_REL_PATH,
    tests_run: Iterable[str] | None = None,
    started_at: float | None = None,
    clock: Any = time.time,
) -> dict[str, Any]:
    """Build the Exp 2863 manifest-contract artifact without inferring metrics."""

    started = clock() if started_at is None else started_at
    resolved = resolve_manifest_contract(
        repo_root=repo_root,
        source_artifact=source_artifact,
        corpora=CANONICAL_CORPORA,
    )
    ready = {key: resolved[key].contract_ready for key in CANONICAL_CORPORA}
    manifest_contract_ready = all(ready.values())
    return {
        "artifact": "experiment_2863_eval_manifest_contract_v2",
        "schema": "carnot.eval_manifest_contract.v2",
        "honest_verdict": (
            "complete: eval manifest contract ready"
            if manifest_contract_ready
            else "blocked_eval_manifest_contract"
        ),
        "manifest_contract_ready": manifest_contract_ready,
        "manifest_source_artifact": str(Path(source_artifact)),
        "resolved_manifest_paths": {key: resolved[key].path for key in CANONICAL_CORPORA},
        "resolved_manifest_sha256": {key: resolved[key].sha256 for key in CANONICAL_CORPORA},
        "resolved_manifest_counts": {key: resolved[key].count for key in CANONICAL_CORPORA},
        "checksum_verified": {key: resolved[key].checksum_verified for key in CANONICAL_CORPORA},
        "resolved_manifests": {key: resolved[key].as_dict() for key in CANONICAL_CORPORA},
        "halueval_ready": ready["halueval"],
        "fever_ready": ready["fever"],
        "mbpp_ready": ready["mbpp"],
        "humaneval_ready": ready["humaneval"],
        "truthfulqa_ready": ready["truthfulqa"],
        "synthetic_rows_created": False,
        "tests_run": list(tests_run or []),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": max(0.0, clock() - started),
    }


def write_contract_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    source_artifact: Path | str = MATERIALIZATION_ARTIFACT_REL_PATH,
    tests_run: Iterable[str] | None = None,
    started_at: float | None = None,
    clock: Any = time.time,
) -> dict[str, Any]:
    """Write the Exp 2863 contract artifact to ``results/``."""

    artifact = build_contract_artifact(
        repo_root=repo_root,
        source_artifact=source_artifact,
        tests_run=tests_run,
        started_at=started_at,
        clock=clock,
    )
    output_path = Path(repo_root) / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
