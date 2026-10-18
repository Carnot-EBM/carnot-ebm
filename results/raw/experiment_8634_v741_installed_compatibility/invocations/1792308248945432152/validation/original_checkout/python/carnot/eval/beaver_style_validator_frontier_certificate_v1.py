"""Exp 3018 BEAVER-style frontier certificate over validator trees.

Spec refs: REQ-VERIFY-3018, SCENARIO-VERIFY-3018.

This module builds a small certificate over the already materialized Exp 3017
validator-tree corpus.  It does not run a live model and it does not claim full
BEAVER probability soundness.  The useful boundary is narrower: cached
known-good and known-bad candidates are replayed through deterministic
validator trees, prefix-closed assumptions are counted where they apply, and
non-prefix or unresolved rows remain explicit instead of being hidden inside a
binary pass/fail summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval import nsvif_instruction_validator_tree_expansion_v1 as exp3017


JsonDict = dict[str, Any]
RUN_DATE = "20260524"
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_FILENAME = "experiment_3018_beaver_style_validator_frontier_certificate_v1.json"
ARTIFACT = "experiment_3018_beaver_style_validator_frontier_certificate_v1"
SCHEMA = "carnot.beaver_style_validator_frontier_certificate.v1"
CERTIFICATE_MANIFEST_REL_PATH = Path(
    "results/beaver_style_validator_frontier_certificate_3018/certificate_manifest.jsonl"
)
TRANSCRIPT_REL_DIR = Path("results/beaver_style_validator_frontier_certificate_3018/transcripts")
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / ARTIFACT_FILENAME
INFERENCE_SUBSTRATE = "deterministic_cached_validator_frontier"

TERMINAL_SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
BLOCKED_PREFIXES = ("blocked:", "blocked_")
PREFIX_CLOSED_NODE_KINDS = frozenset(
    {
        "json_forbidden_tokens",
        "json_list_order",
        "string_forbidden_tokens",
        "string_ordered_substrings",
    }
)
PROBABILITY_NOT_COMPUTED: JsonDict = {
    "exact_probability_computed": False,
    "lower_bound": None,
    "upper_bound": None,
    "bound_type": "placeholder",
    "reason": "no token-trie or model-probability frontier was computed",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "frontier_certificate_ready",
        "certificate_manifest_path",
        "n_frontier_items",
        "n_prefix_closed_items",
        "certified_safe_count",
        "certified_violating_count",
        "unresolved_count",
        "enumerator_fallback_separated",
        "live_llm_evidence_used",
        "transcript_paths",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and clock hooks for deterministic Exp 3018 runs."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    certificate_manifest_path: Path | None = None
    transcript_dir: Path | None = None
    source_artifact_path: Path | None = None
    source_manifest_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = ()

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_certificate_manifest_path(self) -> Path:
        return self.certificate_manifest_path or self.repo_root / CERTIFICATE_MANIFEST_REL_PATH

    def resolved_transcript_dir(self) -> Path:
        return self.transcript_dir or self.repo_root / TRANSCRIPT_REL_DIR

    def resolved_source_artifact_path(self) -> Path:
        return (
            self.source_artifact_path
            or self.repo_root / "results" / exp3017.ARTIFACT_FILENAME
        )

    def resolved_source_manifest_path(self) -> Path:
        return self.source_manifest_path or self.repo_root / exp3017.VALIDATOR_MANIFEST_REL_PATH


def classify_validator_tree(
    validator_tree: Mapping[str, Any],
    *,
    cached_candidates_available: bool,
) -> JsonDict:
    """Classify which validator nodes support prefix or bounded exploration."""

    nodes = [dict(node) for node in validator_tree.get("nodes", [])]
    authoritative = [node for node in nodes if node.get("authoritative", True)]
    prefix_nodes = [
        str(node["node_id"])
        for node in authoritative
        if str(node.get("kind")) in PREFIX_CLOSED_NODE_KINDS
    ]
    prefix_kinds = _unique(
        str(node.get("kind"))
        for node in authoritative
        if str(node.get("kind")) in PREFIX_CLOSED_NODE_KINDS
    )
    bounded_kinds = _unique(
        str(node.get("kind"))
        for node in authoritative
        if str(node.get("kind")) not in PREFIX_CLOSED_NODE_KINDS
    )
    non_prefix_nodes = [
        str(node["node_id"])
        for node in nodes
        if not node.get("authoritative", True)
        or str(node.get("authority")) == exp3017.NON_AUTHORITATIVE_AUTHORITY
    ]
    return {
        "prefix_closed_node_ids": prefix_nodes,
        "prefix_closed_node_kinds": prefix_kinds,
        "bounded_frontier_node_kinds": bounded_kinds,
        "non_prefix_closed_node_ids": non_prefix_nodes,
        "frontier_explorable": bool(cached_candidates_available and authoritative),
        "frontier_exploration_mode": (
            "cached_candidate_set" if cached_candidates_available else "unavailable"
        ),
    }


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3018 frontier certificate artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    source = _load_exp3017_source(active)
    if not source["ready"]:
        artifact = _blocked_artifact(
            active,
            duration_s=round(active.clock() - started, 6),
            blocked_reason=str(source["blocked_reason"]),
        )
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    rows = build_certificate_rows(active, source["manifest_rows"], source["source_artifact"])
    _write_jsonl(active.resolved_certificate_manifest_path(), rows)
    artifact = build_artifact(
        active,
        rows,
        source_artifact=source["source_artifact"],
        duration_s=round(active.clock() - started, 6),
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def build_certificate_rows(
    config: ExperimentConfig,
    source_manifest_rows: Sequence[Mapping[str, Any]],
    source_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    """Replay cached candidates and build inspectable certificate rows."""

    candidates_by_id = {item.item_id: item for item in exp3017.build_instruction_items()}
    rows: list[JsonDict] = []
    for source_row in source_manifest_rows:
        item_id = str(source_row["item_id"])
        tree = source_row["validator_tree"]
        item = candidates_by_id.get(item_id)
        if item is None:
            rows.append(_unresolved_source_row(item_id, "cached_candidate_missing"))
            continue
        hash_mismatch = _candidate_hash_mismatch(source_row, item)
        if hash_mismatch:
            rows.append(_unresolved_source_row(item_id, hash_mismatch))
            continue

        classification = classify_validator_tree(tree, cached_candidates_available=True)
        item_rows = _candidate_certificate_rows(item, tree, classification)
        item_rows.extend(_non_prefix_node_rows(item_id, classification))
        transcript_info = _write_item_transcript(config, item_id, tree, classification, item_rows)
        for row in item_rows:
            row["transcript_path"] = transcript_info["path"]
            row["transcript_sha256"] = transcript_info["sha256"]
        rows.extend(item_rows)

    rows.extend(_source_rejected_rows(source_artifact))
    return rows


def build_artifact(
    config: ExperimentConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_artifact: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Build the terminal JSON artifact from certificate manifest rows."""

    counts = _status_counts(rows)
    transcript_paths = _unique(str(row["transcript_path"]) for row in rows if row.get("transcript_path"))
    live_llm_used = False
    fallback_separated = _enumerator_fallback_separated(config.repo_root)
    n_prefix_closed = sum(1 for row in rows if row.get("prefix_closed_assumption_applies"))
    ready = (
        bool(rows)
        and counts["certified_safe"] > 0
        and counts["certified_violating"] > 0
        and n_prefix_closed > 0
        and counts["non_prefix_closed"] > 0
        and counts["unresolved"] > 0
        and bool(transcript_paths)
        and not live_llm_used
        and fallback_separated
    )
    manifest_path = config.resolved_certificate_manifest_path()
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "frontier_certificate_ready": ready,
        "certificate_manifest_path": str(_relative_to(config.repo_root, manifest_path)),
        "n_frontier_items": len(rows),
        "n_prefix_closed_items": n_prefix_closed,
        "certified_safe_count": counts["certified_safe"],
        "certified_violating_count": counts["certified_violating"],
        "unresolved_count": counts["unresolved"],
        "non_prefix_closed_count": counts["non_prefix_closed"],
        "enumerator_fallback_separated": fallback_separated,
        "live_llm_evidence_used": live_llm_used,
        "transcript_paths": transcript_paths,
        "honest_verdict": (
            "complete: validator frontier certificate ready with explicit unresolved bounds"
            if ready
            else "blocked: validator frontier certificate gates did not clear"
        ),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts": _source_artifact_status(config.repo_root, source_artifact),
        "source_validator_manifest_path": str(
            _relative_to(config.repo_root, config.resolved_source_manifest_path())
        ),
        "certificate_manifest_sha256": sha256_file(manifest_path),
        "certificate_status_counts": counts,
        "prefix_closed_constraint_kinds": sorted(PREFIX_CLOSED_NODE_KINDS),
        "probability_bound_policy": dict(PROBABILITY_NOT_COMPUTED),
        "live_llm_provenance": {"used": False, "transcript_paths": []},
        "enumerator_fallback_provenance": _enumerator_fallback_provenance(config.repo_root),
        "tests_run": list(config.tests_run),
        "field_principles": field_principles(),
    }
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3018 artifact violates the certificate contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("live_llm_evidence_used") is not False:
        raise ValueError("live_llm_evidence_used must remain false")

    ready = bool(artifact.get("frontier_certificate_ready"))
    verdict = str(artifact.get("honest_verdict", ""))
    if ready and not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if not ready and not verdict.startswith(BLOCKED_PREFIXES):
        raise ValueError("honest_verdict must use a blocked prefix when not ready")
    if not ready:
        return

    if int(artifact.get("n_frontier_items") or 0) <= 0:
        raise ValueError("n_frontier_items must be positive")
    if artifact.get("enumerator_fallback_separated") is not True:
        raise ValueError("enumerator_fallback_separated must be true")
    if int(artifact.get("n_prefix_closed_items") or 0) <= 0:
        raise ValueError("n_prefix_closed_items must count BEAVER-style assumptions")
    if int(artifact.get("certified_safe_count") or 0) <= 0:
        raise ValueError("certified_safe_count must be positive")
    if int(artifact.get("certified_violating_count") or 0) <= 0:
        raise ValueError("certified_violating_count must be positive")
    if not artifact.get("transcript_paths"):
        raise ValueError("transcript_paths must be present")


def load_certificate_manifest(path: Path) -> list[JsonDict]:
    """Load the JSONL certificate manifest written by Exp 3018."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def field_principles() -> JsonDict:
    """Return short machine-readable reasons for required terminal fields."""

    return {
        "frontier_certificate_ready": (
            "Downstream FR-11 diagnostics gate on certificate availability."
        ),
        "certificate_manifest_path": "Certificate rows must be inspectable.",
        "n_frontier_items": "Frontier sample size must be explicit.",
        "n_prefix_closed_items": "BEAVER-style assumptions must be counted.",
        "unresolved_count": "Incomplete certificate rows must not be hidden.",
        "live_llm_evidence_used": "This run stays cached/exact unless explicitly changed.",
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by the focused Exp 3018 script."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--transcript-dir", type=Path, default=None)
    parser.add_argument("--source-artifact", type=Path, default=None)
    parser.add_argument("--source-manifest", type=Path, default=None)
    args = parser.parse_args(argv)

    artifact = run_experiment(
        ExperimentConfig(
            output_path=args.output,
            certificate_manifest_path=args.manifest,
            transcript_dir=args.transcript_dir,
            source_artifact_path=args.source_artifact,
            source_manifest_path=args.source_manifest,
        )
    )
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["frontier_certificate_ready"] else 1


def _candidate_certificate_rows(
    item: exp3017.InstructionItem,
    tree: Mapping[str, Any],
    classification: Mapping[str, Any],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    cached_candidates = (
        ("known_good", item.known_good_candidate),
        ("known_bad", item.known_bad_candidate),
    )
    for role, candidate_text in cached_candidates:
        outcome = exp3017.evaluate_validator_tree(tree, candidate_text)
        accepted = bool(outcome["accepted"])
        rows.append(
            {
                "row_id": f"{item.item_id}:{role}",
                "row_type": "candidate_frontier",
                "item_id": item.item_id,
                "candidate_role": role,
                "candidate_sha256": sha256_text(candidate_text),
                "certificate_status": (
                    "certified_safe" if accepted else "certified_violating"
                ),
                "prefix_closed_assumption_applies": bool(
                    classification["prefix_closed_node_ids"]
                ),
                "prefix_closed_node_ids": list(classification["prefix_closed_node_ids"]),
                "bounded_frontier_node_kinds": list(
                    classification["bounded_frontier_node_kinds"]
                ),
                "frontier_exploration": {
                    "mode": "cached_candidate_set",
                    "candidate_set_size": 2,
                    "bounded": True,
                    "candidate_role": role,
                },
                "deterministic_validator_outcome": {
                    "accepted": accepted,
                    "failing_node_ids": list(outcome["failing_node_ids"]),
                    "rejection_reasons": list(outcome["rejection_reasons"]),
                    "llm_judge_used": bool(outcome["llm_judge_used"]),
                },
                "probability_bound_placeholder": dict(PROBABILITY_NOT_COMPUTED),
                "live_llm_evidence_used": False,
                "enumerator_fallback_used": False,
            }
        )
    return rows


def _non_prefix_node_rows(item_id: str, classification: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "row_id": f"{item_id}:non_prefix:{node_id}",
            "row_type": "non_prefix_closed_node",
            "item_id": item_id,
            "node_id": node_id,
            "certificate_status": "non_prefix_closed",
            "prefix_closed_assumption_applies": False,
            "frontier_exploration": {
                "mode": "not_applicable",
                "bounded": False,
                "reason": "node is logged as non-authoritative semantic boundary",
            },
            "probability_bound_placeholder": dict(PROBABILITY_NOT_COMPUTED),
            "live_llm_evidence_used": False,
            "enumerator_fallback_used": False,
        }
        for node_id in classification["non_prefix_closed_node_ids"]
    ]


def _source_rejected_rows(source_artifact: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rejected in source_artifact.get("rejected_items", []):
        reason = str(rejected.get("rejection_reason", "unknown_rejection"))
        status = (
            "non_prefix_closed"
            if reason == "ambiguous_instruction"
            else "unresolved"
        )
        rows.append(
            {
                "row_id": f"{rejected.get('item_id', 'rejected')}:{status}",
                "row_type": "source_rejected_item",
                "item_id": str(rejected.get("item_id", "rejected")),
                "certificate_status": status,
                "source_rejection_reason": reason,
                "source_rejection_detail": str(rejected.get("detail", "")),
                "prefix_closed_assumption_applies": False,
                "frontier_exploration": {
                    "mode": "not_available",
                    "bounded": False,
                    "reason": "source item was rejected before validator authority",
                },
                "probability_bound_placeholder": dict(PROBABILITY_NOT_COMPUTED),
                "live_llm_evidence_used": False,
                "enumerator_fallback_used": False,
            }
        )
    return rows


def _unresolved_source_row(item_id: str, reason: str) -> JsonDict:
    return {
        "row_id": f"{item_id}:unresolved",
        "row_type": "source_manifest_item",
        "item_id": item_id,
        "certificate_status": "unresolved",
        "source_rejection_reason": reason,
        "prefix_closed_assumption_applies": False,
        "frontier_exploration": {
            "mode": "not_available",
            "bounded": False,
            "reason": reason,
        },
        "probability_bound_placeholder": dict(PROBABILITY_NOT_COMPUTED),
        "live_llm_evidence_used": False,
        "enumerator_fallback_used": False,
    }


def _candidate_hash_mismatch(
    source_row: Mapping[str, Any],
    item: exp3017.InstructionItem,
) -> str | None:
    if source_row.get("known_good_candidate_sha256") != sha256_text(item.known_good_candidate):
        return "known_good_candidate_hash_mismatch"
    if source_row.get("known_bad_candidate_sha256") != sha256_text(item.known_bad_candidate):
        return "known_bad_candidate_hash_mismatch"
    return None


def _write_item_transcript(
    config: ExperimentConfig,
    item_id: str,
    tree: Mapping[str, Any],
    classification: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    payload = {
        "item_id": item_id,
        "validator_tree_sha256": sha256_text(json.dumps(tree, sort_keys=True)),
        "classification": dict(classification),
        "certificate_rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"transcript_path", "transcript_sha256"}
            }
            for row in rows
        ],
        "live_llm_evidence_used": False,
        "enumerator_fallback_used": False,
    }
    directory = config.resolved_transcript_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{item_id}.json"
    _write_json(path, payload)
    return {
        "path": str(_relative_to(config.repo_root, path)),
        "sha256": sha256_file(path),
    }


def _load_exp3017_source(config: ExperimentConfig) -> JsonDict:
    artifact_path = config.resolved_source_artifact_path()
    manifest_path = config.resolved_source_manifest_path()
    if not artifact_path.is_file():
        return {
            "ready": False,
            "blocked_reason": "exp3017_artifact_missing",
            "source_artifact": {},
            "manifest_rows": [],
        }
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "ready": False,
            "blocked_reason": "exp3017_artifact_malformed",
            "source_artifact": {},
            "manifest_rows": [],
        }
    if artifact.get("instruction_validator_tree_ready") is not True:
        return {
            "ready": False,
            "blocked_reason": "exp3017_not_ready",
            "source_artifact": artifact,
            "manifest_rows": [],
        }
    if not manifest_path.is_file():
        return {
            "ready": False,
            "blocked_reason": "exp3017_manifest_missing",
            "source_artifact": artifact,
            "manifest_rows": [],
        }
    return {
        "ready": True,
        "blocked_reason": None,
        "source_artifact": artifact,
        "manifest_rows": exp3017.load_manifest(manifest_path),
    }


def _blocked_artifact(
    config: ExperimentConfig,
    *,
    duration_s: float,
    blocked_reason: str,
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "frontier_certificate_ready": False,
        "certificate_manifest_path": str(
            _relative_to(config.repo_root, config.resolved_certificate_manifest_path())
        ),
        "n_frontier_items": 0,
        "n_prefix_closed_items": 0,
        "certified_safe_count": 0,
        "certified_violating_count": 0,
        "unresolved_count": 1,
        "non_prefix_closed_count": 0,
        "enumerator_fallback_separated": True,
        "live_llm_evidence_used": False,
        "transcript_paths": [],
        "honest_verdict": f"blocked: {blocked_reason}",
        "blocked_reason": blocked_reason,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "certificate_status_counts": {
            "certified_safe": 0,
            "certified_violating": 0,
            "unresolved": 1,
            "non_prefix_closed": 0,
        },
        "probability_bound_policy": dict(PROBABILITY_NOT_COMPUTED),
        "live_llm_provenance": {"used": False, "transcript_paths": []},
        "enumerator_fallback_provenance": {"present": False, "paths": []},
        "tests_run": list(config.tests_run),
        "field_principles": field_principles(),
    }


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts = {
        "certified_safe": 0,
        "certified_violating": 0,
        "unresolved": 0,
        "non_prefix_closed": 0,
    }
    for row in rows:
        status = str(row.get("certificate_status", "unresolved"))
        if status in counts:
            counts[status] += 1
    return counts


def _source_artifact_status(repo_root: Path, source_artifact: Mapping[str, Any]) -> JsonDict:
    exp3004 = _read_json(repo_root / "results" / "experiment_3004_aquaforte_beaver_live_retry_provenance_v2.json")
    return {
        "exp3017": {
            "present": bool(source_artifact),
            "instruction_validator_tree_ready": bool(
                source_artifact.get("instruction_validator_tree_ready")
            ),
            "validator_manifest_path": source_artifact.get("validator_manifest_path"),
            "n_instruction_items": source_artifact.get("n_instruction_items"),
        },
        "exp3004_boundary_only": {
            "present": bool(exp3004),
            "enumerator_fallback_separated": exp3004.get("enumerator_fallback_separated")
            if exp3004
            else None,
            "live_transcript_paths": exp3004.get("live_transcript_paths", [])
            if exp3004
            else [],
            "enumerator_fallback_paths": exp3004.get("enumerator_fallback_paths", [])
            if exp3004
            else [],
        },
    }


def _enumerator_fallback_separated(repo_root: Path) -> bool:
    exp3004 = _read_json(repo_root / "results" / "experiment_3004_aquaforte_beaver_live_retry_provenance_v2.json")
    if not exp3004:
        return True
    live = {str(path) for path in exp3004.get("live_transcript_paths", [])}
    fallback = {str(path) for path in exp3004.get("enumerator_fallback_paths", [])}
    return bool(exp3004.get("enumerator_fallback_separated")) and live.isdisjoint(fallback)


def _enumerator_fallback_provenance(repo_root: Path) -> JsonDict:
    exp3004 = _read_json(repo_root / "results" / "experiment_3004_aquaforte_beaver_live_retry_provenance_v2.json")
    if not exp3004:
        return {"present": False, "paths": [], "separated_from_live": True}
    return {
        "present": True,
        "paths": list(exp3004.get("enumerator_fallback_paths", [])),
        "separated_from_live": _enumerator_fallback_separated(repo_root),
    }


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _unique(values: Any) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:  # pragma: no cover - external output paths are allowed.
        return path


__all__ = [
    "ARTIFACT_FILENAME",
    "CERTIFICATE_MANIFEST_REL_PATH",
    "DEFAULT_OUTPUT_PATH",
    "ExperimentConfig",
    "PROBABILITY_NOT_COMPUTED",
    "REQUIRED_ARTIFACT_FIELDS",
    "TRANSCRIPT_REL_DIR",
    "build_artifact",
    "build_certificate_rows",
    "classify_validator_tree",
    "field_principles",
    "load_certificate_manifest",
    "main",
    "run_experiment",
    "sha256_file",
    "sha256_text",
    "validate_artifact",
]
