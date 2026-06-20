"""Experiment 4495: ARC human replay corpus staging.

Spec refs: REQ-ARC-FCP-4495, SCENARIO-ARC-FCP-4495.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from carnot.agentic import arc_human_replay_corpus as corpus


RESULT_RELATIVE_PATH = "results/experiment_4495_human_replay_corpus_staging.json"
DATA_RELATIVE_DIR = "data/arc_public_demo_human_replay_corpus"
REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OFFICIAL_ARC_BLOG_URL = "https://arcprize.org/blog/arc-agi-3-human-dataset"
OFFICIAL_ARC_DATA_SHORTLINK = "https://dub.link/vfwCqvb"
HF_DATASET_ID = "magic-sword/arc_agi_3_public_demo_human_testing"
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_DATASET_ID}"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET_ID}"
KAGGLE_FORMAT_MIRROR_URL = "https://www.kaggle.com/datasets/jihangli1121/arc-agi-3-replays-v1"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ "
        "(Verdict Terminal-Prefix Discipline)."
    ),
    "inference_substrate": (
        "explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | "
        "aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor."
    ),
    "preconditions_checked": (
        "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."
    ),
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "field_principles",
    "source_provenance",
    "license_status",
    "official_license_verified",
    "weights_committed",
    "training_shard_count",
    "training_example_count",
    "shard_checksums",
    "data_relative_dir",
)
ATTRIBUTION = (
    "Replay data provenance: ARC Prize Foundation ARC-AGI-3 Public Demo human testing "
    f"dataset ({OFFICIAL_ARC_BLOG_URL}); reachable Hugging Face mirror "
    f"{HF_DATASET_ID}; CC BY 4.0 format/license reference from {KAGGLE_FORMAT_MIRROR_URL}. "
    "No model weights are bundled or committed."
)


def _url_status(url: str, *, timeout: float = 15.0) -> dict[str, Any]:  # pragma: no cover
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "carnot-arc-replay-stager"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            return {"url": url, "reachable": 200 <= status < 400, "status_code": status}
    except urllib.error.HTTPError as exc:
        return {"url": url, "reachable": False, "status_code": int(exc.code), "error": str(exc)}
    except Exception as exc:
        return {"url": url, "reachable": False, "status_code": None, "error": repr(exc)}


def _read_hf_dataset_api() -> dict[str, Any]:  # pragma: no cover
    with urllib.request.urlopen(HF_API_URL, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _hf_tree_entries() -> list[dict[str, Any]]:  # pragma: no cover
    url = f"{HF_API_URL}/tree/main?recursive=1"
    with urllib.request.urlopen(url, timeout=30) as response:
        return list(json.loads(response.read().decode("utf-8")))


def _download_url(url: str, path: Path) -> None:  # pragma: no cover
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "carnot-arc-replay-stager"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def download_hf_mirror_parquets(root: Path | str) -> dict[str, Any]:  # pragma: no cover
    """Fetch the reachable Hugging Face mirror once into the gitignored data cache."""

    root_path = Path(root)
    raw_dir = root_path / DATA_RELATIVE_DIR / "raw_hf_mirror"
    dataset_api = _read_hf_dataset_api()
    entries = _hf_tree_entries()
    parquet_entries = [
        entry
        for entry in entries
        if str(entry.get("path", "")).startswith("data/")
        and str(entry.get("path", "")).endswith(".parquet")
    ]
    downloaded: list[dict[str, Any]] = []
    for entry in parquet_entries:
        relative = str(entry["path"])
        target = raw_dir / relative
        expected_size = int(entry.get("size") or 0)
        if not target.exists() or (expected_size > 0 and target.stat().st_size != expected_size):
            url = f"{HF_DATASET_URL}/resolve/main/{relative}"
            _download_url(url, target)
        downloaded.append(
            {
                "path": str(target.relative_to(root_path)),
                "source_path": relative,
                "size_bytes": target.stat().st_size,
                "oid": entry.get("oid"),
            }
        )
    return {
        "source_kind": "hf_mirror",
        "mirror_url": HF_DATASET_URL,
        "dataset_sha": dataset_api.get("sha"),
        "license_status": "mirror_attribution_required",
        "license_name": "not CC0/MIT-0 verified; staged with attribution and no weights",
        "official_license_verified": False,
        "downloaded_files": downloaded,
    }


def stage_from_hf_mirror(
    root: Path | str,
    *,
    max_examples: int | None = None,
) -> dict[str, Any]:  # pragma: no cover
    """Download missing mirror parquet files and convert them into local shards."""

    root_path = Path(root)
    download_manifest = download_hf_mirror_parquets(root_path)
    parquet_paths = [
        root_path / str(item["path"])
        for item in download_manifest.get("downloaded_files", [])
        if str(item.get("path", "")).endswith(".parquet")
    ]
    shard_manifest = corpus.write_training_shards_from_parquet(
        parquet_paths,
        root_path / DATA_RELATIVE_DIR,
        source_metadata=download_manifest,
        max_examples=max_examples,
    )
    return {**download_manifest, "staged_manifest": shard_manifest}


def check_preconditions(root: Path | str) -> dict[str, Any]:
    """Record the resources verified before staging or reporting the artifact."""

    root_path = Path(root)
    preconditions: dict[str, Any] = {
        "agents_md_read": (root_path / "AGENTS.md").exists(),
        "codex_md_read": (root_path / "CODEX.md").exists() or (root_path / "OPENCODE.md").exists(),
        "offline_arcade_import_smoke": False,
        "torch_import": False,
        "torch_version": "",
        "official_arc_shortlink_reachable": False,
        "hf_mirror_reachable": False,
        "source_shards_cached": bool(list((root_path / DATA_RELATIVE_DIR / "raw_hf_mirror").glob("data/*.parquet"))),
        "training_shards_present": (root_path / DATA_RELATIVE_DIR / corpus.MANIFEST_NAME).exists(),
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        preconditions["offline_arcade_import_smoke"] = True
    except Exception as exc:  # pragma: no cover
        preconditions["offline_arcade_error"] = repr(exc)
    try:
        import torch

        preconditions["torch_import"] = True
        preconditions["torch_version"] = str(torch.__version__)
    except Exception as exc:  # pragma: no cover
        preconditions["torch_error"] = repr(exc)

    official_status = _url_status(OFFICIAL_ARC_DATA_SHORTLINK)
    hf_status = _url_status(HF_API_URL)
    preconditions["official_arc_shortlink_reachable"] = bool(official_status["reachable"])
    preconditions["official_arc_shortlink_status"] = official_status
    preconditions["hf_mirror_reachable"] = bool(hf_status["reachable"])
    preconditions["hf_mirror_status"] = hf_status
    return preconditions


def _load_shard_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / DATA_RELATIVE_DIR / corpus.MANIFEST_NAME
    if not manifest_path.exists():
        return {
            "schema": corpus.SHARD_SCHEMA,
            "example_count": 0,
            "shard_count": 0,
            "shards": [],
            "source_metadata": {},
        }
    return corpus.load_manifest(manifest_path.parent)


def _verdict(shard_manifest: Mapping[str, Any], download_manifest: Mapping[str, Any]) -> str:
    if int(shard_manifest.get("example_count") or 0) <= 0:
        return "complete: blocked_human_replay_shards_missing"
    if bool(download_manifest.get("official_license_verified")):
        return "complete: official_human_replay_shards_staged_no_weights"
    return "complete: staged_attributed_mirror_no_weights"


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any],
    download_manifest: Mapping[str, Any],
    started: float | None = None,
    finished: float | None = None,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4495: build the terminal provenance and license artifact."""

    root_path = Path(root)
    shard_manifest = _load_shard_manifest(root_path)
    source_metadata = dict(shard_manifest.get("source_metadata") or {})
    source_provenance = {
        "official_blog_url": OFFICIAL_ARC_BLOG_URL,
        "official_data_shortlink": OFFICIAL_ARC_DATA_SHORTLINK,
        "hf_mirror_url": HF_DATASET_URL,
        "kaggle_cc_by_format_reference_url": KAGGLE_FORMAT_MIRROR_URL,
        "download_manifest": dict(download_manifest),
        "shard_source_metadata": source_metadata,
    }
    official_license_verified = bool(download_manifest.get("official_license_verified"))
    artifact = {
        "honest_verdict": _verdict(shard_manifest, download_manifest),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "requirements": ["REQ-ARC-FCP-4495"],
        "scenarios": ["SCENARIO-ARC-FCP-4495"],
        "preconditions_checked": dict(preconditions_checked),
        "data_relative_dir": DATA_RELATIVE_DIR,
        "source_provenance": source_provenance,
        "license_status": str(
            download_manifest.get("license_status")
            or source_metadata.get("license_status")
            or "mirror_attribution_required"
        ),
        "license_name": str(
            download_manifest.get("license_name")
            or source_metadata.get("license_name")
            or "not CC0/MIT-0 verified; staged with attribution and no weights"
        ),
        "official_license_verified": official_license_verified,
        "attribution": ATTRIBUTION,
        "weights_committed": False,
        "weights_commit_policy": "Do not commit weights unless official CC0/MIT-0-compatible licensing is verified.",
        "training_shard_count": int(shard_manifest.get("shard_count") or 0),
        "training_example_count": int(shard_manifest.get("example_count") or 0),
        "shard_checksums": list(shard_manifest.get("shards") or []),
        "duration_s": None if started is None or finished is None else max(0.0, float(finished) - float(started)),
    }
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must equal aggregation_from_upstream_artifacts")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match required field principles")
    if artifact.get("weights_committed") is True and not artifact.get("official_license_verified"):
        errors.append("weights require official CC0/MIT-0-compatible license verification")
    if int(artifact.get("training_shard_count") or 0) < 0:
        errors.append("training_shard_count must be non-negative")
    if int(artifact.get("training_example_count") or 0) < 0:
        errors.append("training_example_count must be non-negative")
    if not isinstance(artifact.get("source_provenance"), Mapping):
        errors.append("source_provenance must be a mapping")
    return errors


def run(
    *,
    root: Path | str = REPO_ROOT,
    preconditions_checked: Mapping[str, Any] | None = None,
    write: bool = True,
    fetch_if_missing: bool = True,
    max_examples: int | None = None,
    now: Any = time.monotonic,
) -> dict[str, Any]:
    """SCENARIO-ARC-FCP-4495: write stable JSON after reusing or staging shards."""

    root_path = Path(root)
    started = float(now())
    manifest_path = root_path / DATA_RELATIVE_DIR / corpus.MANIFEST_NAME
    download_manifest: Mapping[str, Any] = {"source_kind": "cached_local_shards"}
    needs_stage = not manifest_path.exists() or int(
        _load_shard_manifest(root_path).get("example_count") or 0
    ) <= 0
    if fetch_if_missing and needs_stage:
        try:
            download_manifest = stage_from_hf_mirror(root_path, max_examples=max_examples)
        except Exception as exc:  # pragma: no cover
            download_manifest = {"source_kind": "blocked", "error": repr(exc)}
    preconditions = (
        dict(preconditions_checked) if preconditions_checked is not None else check_preconditions(root_path)
    )
    preconditions["training_shards_present"] = manifest_path.exists()
    preconditions["source_shards_cached"] = bool(
        list((root_path / DATA_RELATIVE_DIR / "raw_hf_mirror").glob("data/*.parquet"))
    )
    artifact = build_artifact(
        root=root_path,
        preconditions_checked=preconditions,
        download_manifest=download_manifest,
        started=started,
        finished=float(now()),
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        out = root_path / RESULT_RELATIVE_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    artifact = run()
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
