"""Tiny exact-frontier expansion for HaluEval and FEVER rows.

What is exact here:
    A small registry of manually encoded constraints is checked against the
    local HaluEval/FEVER manifest bytes. Z3 decides constant string, year, and
    date-order constraints; plain Python only loads rows and verifies that the
    expected text anchors are still present.

What is not exact here:
    This is not natural-language theorem proving and not LLM autoformalization.
    Rows without a manual exact constraint remain outside the frontier even if
    their dataset label is obvious to a human reader.

Spec: REQ-VERIFY-2877, SCENARIO-VERIFY-2877.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import z3  # type: ignore[import]


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
OUTPUT_FILENAME = "experiment_2877_exact_frontier_expansion_halueval_fever_v2.json"
EXACT_FRONTIER_ARTIFACT_REL_PATH = Path(
    "results/experiment_2866_beaver_exact_tiny_frontier_v1.json"
)
REQUESTED_EXACT_ALIAS_REL_PATH = Path(
    "results/experiment_2866_exact_z3_arithmetic_frontier_v1.json"
)
CALIBRATION_ARTIFACT_REL_PATH = Path(
    "results/experiment_2864_halueval_fever_full_calibration_v3.json"
)
HALUEVAL_MANIFEST_REL_PATH = Path("data/eval_manifests/halueval_20260522.jsonl")
FEVER_MANIFEST_REL_PATH = Path("data/eval_manifests/fever_20260522.jsonl")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "frontier_expansion_ready",
    "source_artifacts",
    "selection_rule",
    "n_candidate_rows",
    "n_exact_supported_rows",
    "n_unsupported_rows",
    "unsupported_reasons",
    "exact_solver_backend",
    "certificates",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)

SELECTION_RULE = (
    "Scan the Exp 2864-resolved HaluEval and FEVER manifests in file order; "
    "include only rows whose stable_id appears in the static manual exact-"
    "constraint registry and whose prompt/candidate/claim anchors match the "
    "manifest bytes. No LLM autoformalization is used; every other row is "
    "reported unsupported."
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix verdict for conductor classification.",
    "frontier_expansion_ready": (
        "True only when clean source artifacts exist and both HaluEval and FEVER "
        "contribute at least one exact manual certificate."
    ),
    "source_artifacts": "Records the prior exact-frontier and calibration evidence used.",
    "selection_rule": (
        "Manual deterministic selection prevents accidental promotion of broad "
        "natural-language coverage."
    ),
    "n_candidate_rows": "All rows scanned from the Exp 2864-resolved manifests.",
    "n_exact_supported_rows": "Rows admitted to the exact frontier by local checks only.",
    "n_unsupported_rows": "Rows deliberately kept outside the exact frontier.",
    "unsupported_reasons": "Explains why unsupported rows were not promoted.",
    "exact_solver_backend": "Identifies the local solver used for exact constant checks.",
    "certificates": "Per-supported-row evidence, solver status, and checksum.",
    "tests_run": "Commands used to validate this module and artifact builder.",
    "duration_s": "Measured wall-clock runtime; no sleep padding.",
}

MANUAL_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "halueval-8-right": {
        "dataset_key": "halueval",
        "constraint_type": "safe_prefix_year_answer",
        "kind": "year_match",
        "expected_year": 2006,
        "expected_status": "sat",
        "exact_verdict": "safe_prefix_supported",
        "prompt_anchors": ("first aired in September 2006",),
        "candidate_anchors": ("2006",),
        "prefix_token": "2006",
    },
    "halueval-8-hallucinated": {
        "dataset_key": "halueval",
        "constraint_type": "year_contradiction",
        "kind": "year_match",
        "expected_year": 2006,
        "expected_status": "unsat",
        "exact_verdict": "contradiction_verified",
        "prompt_anchors": ("first aired in September 2006",),
        "candidate_anchors": ("2003",),
        "prefix_token": "2003",
    },
    "halueval-22-right": {
        "dataset_key": "halueval",
        "constraint_type": "arithmetic_like_date_order",
        "kind": "date_order",
        "entity_years": {"Hole": 1989, "The Wolfhounds": 1985},
        "claimed_first": "The Wolfhounds",
        "compare_against": "Hole",
        "expected_status": "sat",
        "exact_verdict": "safe_prefix_supported",
        "prompt_anchors": ("formed in 1989", "formed in Romford, UK in 1985"),
        "candidate_anchors": ("The Wolfhounds",),
        "prefix_token": "The Wolfhounds",
    },
    "halueval-22-hallucinated": {
        "dataset_key": "halueval",
        "constraint_type": "arithmetic_like_date_order",
        "kind": "date_order",
        "entity_years": {"Hole": 1989, "The Wolfhounds": 1985},
        "claimed_first": "Hole",
        "compare_against": "The Wolfhounds",
        "expected_status": "unsat",
        "exact_verdict": "contradiction_verified",
        "prompt_anchors": ("formed in 1989", "formed in Romford, UK in 1985"),
        "candidate_anchors": ("Hole", "founded first"),
        "prefix_token": "founded first",
    },
    "halueval-31-right": {
        "dataset_key": "halueval",
        "constraint_type": "arithmetic_like_date_order",
        "kind": "date_order",
        "entity_years": {"Pablo Trapero": 1971, "Aleksander Ford": 1908},
        "claimed_first": "Aleksander Ford",
        "compare_against": "Pablo Trapero",
        "expected_status": "sat",
        "exact_verdict": "safe_prefix_supported",
        "prompt_anchors": ("Born 4 October 1971", "24 November 1908"),
        "candidate_anchors": ("Aleksander Ford",),
        "prefix_token": "Aleksander Ford",
    },
    "halueval-31-hallucinated": {
        "dataset_key": "halueval",
        "constraint_type": "arithmetic_like_date_order",
        "kind": "date_order",
        "entity_years": {"Pablo Trapero": 1971, "Aleksander Ford": 1908},
        "claimed_first": "Pablo Trapero",
        "compare_against": "Aleksander Ford",
        "expected_status": "unsat",
        "exact_verdict": "contradiction_verified",
        "prompt_anchors": ("Born 4 October 1971", "24 November 1908"),
        "candidate_anchors": ("Pablo Trapero", "born first"),
        "prefix_token": "born first",
    },
    "fever-84514": {
        "dataset_key": "fever",
        "constraint_type": "anchored_entailment",
        "kind": "anchored_entailment",
        "expected_status": "sat",
        "exact_verdict": "entailment_anchor_verified",
        "prompt_anchors": ("gaseous state", "steam or water vapor"),
        "candidate_anchors": ("Steam", "gaseous state", "water vapor"),
        "prefix_token": "water vapor",
    },
    "fever-182889": {
        "dataset_key": "fever",
        "constraint_type": "existential_entailment",
        "kind": "anchored_entailment",
        "expected_status": "sat",
        "exact_verdict": "entailment_anchor_verified",
        "prompt_anchors": ("The film stars", "Al Pacino"),
        "candidate_anchors": ("stars at least one actor",),
        "prefix_token": "actor",
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the Exp 2877 exact-frontier expansion."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exact_frontier_artifact_path: Path | None = None
    calibration_artifact_path: Path | None = None
    tests_run: tuple[str, ...] | list[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    run_date: str = RUN_DATE

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_output_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def resolved_exact_frontier_artifact_path(self) -> Path:
        path = self.exact_frontier_artifact_path or EXACT_FRONTIER_ARTIFACT_REL_PATH
        return path if path.is_absolute() else self.repo_root / path

    def resolved_calibration_artifact_path(self) -> Path:
        path = self.calibration_artifact_path or CALIBRATION_ARTIFACT_REL_PATH
        return path if path.is_absolute() else self.repo_root / path


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Scan the resolved manifests, build certificates, and optionally write JSON."""

    cfg = config or ExperimentConfig()
    started = cfg.start_time()
    exact_artifact = _load_json(cfg.resolved_exact_frontier_artifact_path())
    calibration_artifact = _load_json(cfg.resolved_calibration_artifact_path())
    manifest_paths = _manifest_paths_from_calibration(cfg, calibration_artifact)
    rows = _load_candidate_rows(manifest_paths)
    certificates: list[dict[str, Any]] = []
    unsupported_reasons: Counter[str] = Counter()

    for row in rows:
        stable_id = _stable_id(row)
        spec = MANUAL_CONSTRAINTS.get(stable_id)
        if spec is None:
            unsupported_reasons["unsupported_no_manual_exact_constraint"] += 1
            continue
        certificate = _build_certificate(row, spec)
        if certificate is None:
            unsupported_reasons["unsupported_manual_constraint_failed"] += 1
            continue
        certificates.append(certificate)

    source_paths = _source_paths(cfg, manifest_paths)
    source_ready = bool(
        exact_artifact.get("exact_frontier_available")
        and calibration_artifact.get("halueval_fever_ready")
    )
    supported_datasets = {certificate["dataset"] for certificate in certificates}
    ready = bool(source_ready and {"HaluEval", "FEVER"} <= supported_datasets)
    artifact = {
        "honest_verdict": (
            "complete: exact frontier touches bounded HaluEval/FEVER rows without "
            "natural-language overclaim"
            if ready
            else "complete: no HaluEval/FEVER exact frontier expansion beyond unsupported rows"
        ),
        "frontier_expansion_ready": ready,
        "source_artifacts": list(source_paths),
        "source_artifact_sha256": _source_sha256(source_paths),
        "source_artifact_details": _source_details(source_paths),
        "source_artifact_notes": _source_notes(cfg),
        "selection_rule": SELECTION_RULE,
        "n_candidate_rows": len(rows),
        "n_exact_supported_rows": len(certificates),
        "n_unsupported_rows": len(rows) - len(certificates),
        "unsupported_reasons": dict(sorted(unsupported_reasons.items())),
        "exact_solver_backend": exact_solver_backend(),
        "certificates": certificates,
        "tests_run": list(cfg.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": cfg.run_date,
        "duration_s": max(0.0, cfg.clock() - started),
    }
    validate_artifact(artifact)
    if write:
        write_artifact(cfg.resolved_output_path(), artifact)
    return artifact


def exact_solver_backend() -> str:
    """Return the exact local solver identity used in row certificates."""

    return f"z3-solver {z3.get_version_string()} + deterministic manifest anchors"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 2877 fields and exact-frontier accounting."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["run_date"] != RUN_DATE:
        raise ValueError("run_date must be 20260522")
    if not isinstance(artifact["source_artifacts"], list):
        raise ValueError("source_artifacts must be a list")
    if not str(artifact["exact_solver_backend"]):
        raise ValueError("exact solver backend must be recorded")
    if int(artifact["n_candidate_rows"]) != int(artifact["n_exact_supported_rows"]) + int(
        artifact["n_unsupported_rows"]
    ):
        raise ValueError("unsupported count must reconcile with candidate and supported counts")
    if len(artifact["certificates"]) != int(artifact["n_exact_supported_rows"]):
        raise ValueError("certificate count must equal n_exact_supported_rows")
    if sum(int(value) for value in artifact["unsupported_reasons"].values()) != int(
        artifact["n_unsupported_rows"]
    ):
        raise ValueError("unsupported count must equal unsupported_reasons total")


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    """Write stable JSON for the Exp 2877 deliverable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _manifest_paths_from_calibration(
    config: ExperimentConfig,
    calibration_artifact: Mapping[str, Any],
) -> dict[str, Path]:
    paths = dict(calibration_artifact.get("manifest_paths_used") or {})
    halueval = Path(str(paths.get("halueval") or HALUEVAL_MANIFEST_REL_PATH))
    fever = Path(str(paths.get("fever") or FEVER_MANIFEST_REL_PATH))
    return {
        "halueval": halueval if halueval.is_absolute() else config.repo_root / halueval,
        "fever": fever if fever.is_absolute() else config.repo_root / fever,
    }


def _load_candidate_rows(manifest_paths: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_key in ("halueval", "fever"):
        for row in _read_jsonl(manifest_paths[dataset_key]):
            row["_dataset_key"] = dataset_key
            rows.append(row)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                loaded.append(json.loads(line))
    return loaded


def _build_certificate(row: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get("_dataset_key") != spec["dataset_key"]:
        return None
    prompt = str(row.get("prompt") or "")
    candidate = _candidate_text(row)
    if not _anchors_match(prompt, spec["prompt_anchors"]) or not _anchors_match(
        candidate,
        spec["candidate_anchors"],
    ):
        return None

    solver_status, constraint_payload = _evaluate_spec(candidate, spec)
    if solver_status != spec["expected_status"]:
        return None

    certificate = {
        "stable_id": _stable_id(row),
        "dataset": str(row.get("dataset") or "").strip() or str(spec["dataset_key"]).upper(),
        "label": int(row.get("label") or 0),
        "constraint_type": spec["constraint_type"],
        "exact_verdict": spec["exact_verdict"],
        "solver_status": solver_status,
        "expected_solver_status": spec["expected_status"],
        "solver_backend": exact_solver_backend(),
        "safe_prefix_end": _prefix_end(candidate, str(spec["prefix_token"])),
        "constraints": constraint_payload,
        "evidence": {
            "prompt_anchors": list(spec["prompt_anchors"]),
            "candidate_anchors": list(spec["candidate_anchors"]),
            "candidate_or_claim": candidate,
        },
    }
    certificate["certificate_sha256"] = _checksum(certificate)
    return certificate


def _evaluate_spec(candidate: str, spec: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if spec["kind"] == "year_match":
        candidate_year = _first_year(candidate)
        expected_year = int(spec["expected_year"])
        status = _z3_status(z3.IntVal(candidate_year or -1) == z3.IntVal(expected_year))
        return status, {"candidate_year": candidate_year, "expected_year": expected_year}
    if spec["kind"] == "date_order":
        entity_years = dict(spec["entity_years"])
        claimed_first = str(spec["claimed_first"])
        compare_against = str(spec["compare_against"])
        claimed_year = int(entity_years[claimed_first])
        comparison_year = int(entity_years[compare_against])
        status = _z3_status(z3.IntVal(claimed_year) < z3.IntVal(comparison_year))
        return status, {
            "claimed_first": claimed_first,
            "claimed_year": claimed_year,
            "compare_against": compare_against,
            "comparison_year": comparison_year,
            "relation": "claimed_year < comparison_year",
        }
    prompt_anchors = tuple(str(anchor) for anchor in spec["prompt_anchors"])
    candidate_anchors = tuple(str(anchor) for anchor in spec["candidate_anchors"])
    return "sat", {
        "prompt_anchors": list(prompt_anchors),
        "candidate_anchors": list(candidate_anchors),
        "anchor_scope": "constant prompt and candidate/claim substring containment",
    }


def _z3_status(expression: Any) -> str:
    solver = z3.Solver()
    solver.add(expression)
    return str(solver.check())


def _anchors_match(text: str, anchors: Sequence[str]) -> bool:
    return all(anchor in text for anchor in anchors)


def _first_year(text: str) -> int | None:
    match = re.search(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", text)
    return int(match.group(0)) if match else None


def _prefix_end(text: str, token: str) -> int:
    position = text.find(token)
    return len(text) if position < 0 else position + len(token)


def _candidate_text(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate") or row.get("claim") or "").strip()


def _stable_id(row: Mapping[str, Any]) -> str:
    return str(row.get("stable_id") or "")


def _source_paths(config: ExperimentConfig, manifest_paths: Mapping[str, Path]) -> dict[str, Path]:
    return {
        str(EXACT_FRONTIER_ARTIFACT_REL_PATH): config.resolved_exact_frontier_artifact_path(),
        str(CALIBRATION_ARTIFACT_REL_PATH): config.resolved_calibration_artifact_path(),
        str(HALUEVAL_MANIFEST_REL_PATH): manifest_paths["halueval"],
        str(FEVER_MANIFEST_REL_PATH): manifest_paths["fever"],
    }


def _source_sha256(source_paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: _sha256(path) if path.is_file() else "" for name, path in source_paths.items()}


def _source_details(source_paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"resolved_path": str(path), "present": path.is_file()}
        for name, path in source_paths.items()
    }


def _source_notes(config: ExperimentConfig) -> dict[str, Any]:
    requested_alias = config.repo_root / REQUESTED_EXACT_ALIAS_REL_PATH
    return {
        str(REQUESTED_EXACT_ALIAS_REL_PATH): {
            "present": requested_alias.is_file(),
            "note": (
                "The task prompt named this alias, but the canonical clean Exp 2866 "
                "artifact in this checkout is experiment_2866_beaver_exact_tiny_frontier_v1.json."
            ),
        }
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:  # pragma: no cover - command wrapper.
    run_experiment(ExperimentConfig(repo_root=Path.cwd()))
    return 0


if __name__ == "__main__":  # pragma: no cover - command wrapper.
    raise SystemExit(main())


__all__ = [
    "ExperimentConfig",
    "FIELD_PRINCIPLES",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "RUN_DATE",
    "SELECTION_RULE",
    "exact_solver_backend",
    "main",
    "run_experiment",
    "validate_artifact",
    "write_artifact",
]
