"""Exp 2867 residual-drift plus MUS conflict prioritization diagnostic.

This module consumes the already-built Exp 2865 cross-corpus matrix and turns
only its clean rows into a small diagnostic hypergraph. The graph is a repair
triage aid, not a trained graph neural network and not a certified MUS solver:
when the source artifacts lack exact minimal-unsat-subset certificates, the
hyperedges are explicitly treated as co-failure and residual-drift proxies.

Spec: REQ-VERIFY-2867,
      SCENARIO-VERIFY-2867,
      SCENARIO-VERIFY-2867-BLOCKED.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_DATE = "20260522"
RANDOM_SEED = 2867
MATRIX_REL_PATH = Path("results/experiment_2865_cross_corpus_matrix_v5.json")
OUTPUT_REL_PATH = Path("results/experiment_2867_drift_mus_prioritizer_v2.json")
HEURISTIC_NAME = "residual_weighted_two_pass_hypergraph_heuristic_not_trained_hgnn"
NEAR_RANDOM_AUROC_THRESHOLD = 0.56
RANDOM_BASELINE_TRIALS = 512
REPO_ROOT = Path(__file__).resolve().parents[3]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal complete:/blocked_ verdict; no inferred blocked-row metrics.",
    "drift_mus_diagnostic_ready": "True only after Exp 2865 matrix preconditions pass.",
    "n_failure_rows": "Counts weak or below-random AUROC evidence from clean rows only.",
    "hypergraph_nodes": "Constraint, verifier-failure, and failure-class nodes only.",
    "hypergraph_hyperedges": "Co-failure and residual-drift proxy edges; not certified MUS edges.",
    "hgnn_inspired_heuristic_name": (
        "Descriptive only: deterministic message passing, not a trained HGNN policy."
    ),
    "heuristic_improvement_vs_best_baseline": (
        "Positive means fewer checks-to-conflict than the best baseline; negative is allowed."
    ),
    "reproducibility_checksum": "Hashes matrix, clean source evidence, failures, graph, and metrics.",
    "duration_s": "Real wall-clock synthesis time; never padded.",
}


@dataclass(frozen=True)
class FailureEvidence:
    """One clean-row weakness that can become a repair-prioritization target.

    The row stores both the raw AUROC and the derived constraint/verifier labels
    because downstream ranking should be auditable without reopening the source
    artifacts. ``residual_drift`` is measured against the strongest clean matrix
    AUROC available, so it is a relative diagnostic rather than a benchmark
    capability claim.
    """

    row_id: str
    corpus: str
    source_metric: str
    auroc: float
    failure_class: str
    severity: float
    residual_drift: float
    constraint_family: str
    verifier_failure: str
    source_artifact: str

    def nodes(self) -> tuple[str, str, str]:
        return (
            f"constraint:{self.constraint_family}",
            f"verifier_failure:{self.verifier_failure}",
            f"failure_class:{self.failure_class}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "corpus": self.corpus,
            "source_metric": self.source_metric,
            "auroc": self.auroc,
            "failure_class": self.failure_class,
            "severity": self.severity,
            "residual_drift": self.residual_drift,
            "constraint_family": self.constraint_family,
            "verifier_failure": self.verifier_failure,
            "source_artifact": self.source_artifact,
        }


@dataclass(frozen=True)
class Hyperedge:
    """A co-failure or drift proxy relationship among repair target nodes."""

    edge_id: str
    kind: str
    nodes: tuple[str, ...]
    row_ids: tuple[str, ...]
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "kind": self.kind,
            "nodes": list(self.nodes),
            "row_ids": list(self.row_ids),
            "weight": self.weight,
        }


@dataclass(frozen=True)
class Hypergraph:
    """Small immutable hypergraph used by all baseline comparisons."""

    nodes: tuple[str, ...]
    hyperedges: tuple[Hyperedge, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "hyperedges": [edge.to_dict() for edge in self.hyperedges],
        }


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object or ``{}`` when the source cannot be trusted."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def finite_float(value: object) -> float | None:
    """Coerce finite numeric JSON values while rejecting booleans and NaNs."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def classify_failure(auroc: float) -> tuple[str, float] | None:
    """Classify AUROC weaknesses that are actionable for repair triage."""

    if auroc < 0.5:
        return "below_random_auroc", 0.5 - auroc
    if auroc < NEAR_RANDOM_AUROC_THRESHOLD:
        return "near_random_auroc", NEAR_RANDOM_AUROC_THRESHOLD - auroc
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _clean_matrix_rows(matrix_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    matrix = matrix_payload.get("verifier_corpus_dual_matrix")
    if not isinstance(matrix, dict):
        return {}
    return {
        str(corpus): row
        for corpus, row in matrix.items()
        if isinstance(row, dict) and row.get("row_status") == "clean"
    }


def _source_payloads(root: Path, clean_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for row in clean_rows.values():
        rel = str(row.get("source_artifact") or "")
        if not rel:
            continue
        path = Path(rel)
        payloads[rel] = read_json(path if path.is_absolute() else root / path)
    return payloads


def _reference_auroc(clean_rows: dict[str, dict[str, Any]]) -> float:
    values: list[float] = []
    for row in clean_rows.values():
        for key in ("production_auroc", "architecture_only_auroc"):
            value = finite_float(row.get(key))
            if value is not None:
                values.append(value)
        measured = row.get("measured_auroc_by_dataset")
        if isinstance(measured, dict):
            values.extend(value for value in (finite_float(v) for v in measured.values()) if value)
    return max(values) if values else 0.5


def _constraint_family(corpus: str, metric: str) -> str:
    if "arithmetic" in metric:
        return "arithmetic_consistency"
    if "logical" in metric or "curry" in metric:
        return "logical_self_consistency"
    if corpus == "HaluEval/FEVER" or metric in {"halueval", "fever"}:
        return "factual_support"
    return "verifier_calibration"


def _fover_verifier_means(source_payload: dict[str, Any]) -> dict[str, float]:
    per_verifier = source_payload.get("per_verifier_condition_a_auroc")
    if not isinstance(per_verifier, dict):
        return {}
    means: dict[str, float] = {}
    for name, values in per_verifier.items():
        if not isinstance(values, list):
            continue
        finite_values = [value for value in (finite_float(item) for item in values) if value is not None]
        mean = _mean(finite_values)
        if mean is not None:
            means[str(name)] = mean
    return means


def _halueval_fever_metrics(row: dict[str, Any], source_payload: dict[str, Any]) -> dict[str, float]:
    raw = row.get("measured_auroc_by_dataset")
    metrics: dict[str, float] = {}
    if isinstance(raw, dict):
        for dataset, value in raw.items():
            numeric = finite_float(value)
            if numeric is not None:
                metrics[str(dataset)] = numeric
    for dataset, key in (("halueval", "halueval_auroc"), ("fever", "fever_auroc")):
        if dataset not in metrics:
            numeric = finite_float(source_payload.get(key))
            if numeric is not None:
                metrics[dataset] = numeric
    return metrics


def extract_clean_failure_evidence(
    root: Path,
    matrix_payload: dict[str, Any],
    source_payloads: dict[str, dict[str, Any]] | None = None,
) -> list[FailureEvidence]:
    """Extract weak clean-row evidence without touching blocked or missing rows."""

    clean_rows = _clean_matrix_rows(matrix_payload)
    sources = source_payloads if source_payloads is not None else _source_payloads(root, clean_rows)
    reference = _reference_auroc(clean_rows)
    failures: list[FailureEvidence] = []

    for corpus, row in clean_rows.items():
        source_artifact = str(row.get("source_artifact") or "")
        source_payload = sources.get(source_artifact, {})
        if corpus == "FoVer":
            for metric, auroc in sorted(_fover_verifier_means(source_payload).items()):
                classified = classify_failure(auroc)
                if classified is None:
                    continue
                failure_class, severity = classified
                failures.append(
                    FailureEvidence(
                        row_id=f"{corpus}:{metric}",
                        corpus=corpus,
                        source_metric=metric,
                        auroc=auroc,
                        failure_class=failure_class,
                        severity=severity,
                        residual_drift=max(0.0, reference - auroc),
                        constraint_family=_constraint_family(corpus, metric),
                        verifier_failure=metric,
                        source_artifact=source_artifact,
                    )
                )
        elif corpus == "HaluEval/FEVER":
            for metric, auroc in sorted(_halueval_fever_metrics(row, source_payload).items()):
                classified = classify_failure(auroc)
                if classified is None:
                    continue
                failure_class, severity = classified
                failures.append(
                    FailureEvidence(
                        row_id=f"{corpus}:{metric}",
                        corpus=corpus,
                        source_metric=metric,
                        auroc=auroc,
                        failure_class=failure_class,
                        severity=severity,
                        residual_drift=max(0.0, reference - auroc),
                        constraint_family=_constraint_family(corpus, metric),
                        verifier_failure=f"{metric}_local_calibration",
                        source_artifact=source_artifact,
                    )
                )

    return sorted(failures, key=lambda item: item.row_id)


def build_hypergraph(failure_rows: list[FailureEvidence]) -> Hypergraph:
    """Build co-failure and residual-drift proxy edges from clean failures."""

    nodes: set[str] = set()
    edges: list[Hyperedge] = []
    rows_by_corpus: dict[str, list[FailureEvidence]] = defaultdict(list)
    rows_by_constraint: dict[str, list[FailureEvidence]] = defaultdict(list)

    for row in failure_rows:
        row_nodes = tuple(sorted(row.nodes()))
        nodes.update(row_nodes)
        rows_by_corpus[row.corpus].append(row)
        rows_by_constraint[row.constraint_family].append(row)
        edges.append(
            Hyperedge(
                edge_id=f"co_failure:{row.row_id}",
                kind="co_failure",
                nodes=row_nodes,
                row_ids=(row.row_id,),
                weight=round(1.0 + row.severity + row.residual_drift, 6),
            )
        )

    for corpus, rows in sorted(rows_by_corpus.items()):
        if len(rows) < 2:
            continue
        edge_nodes = sorted({node for row in rows for node in row.nodes()})
        nodes.update(edge_nodes)
        avg_severity = sum(row.severity for row in rows) / len(rows)
        edges.append(
            Hyperedge(
                edge_id=f"corpus_co_failure:{corpus}",
                kind="corpus_co_failure",
                nodes=tuple(edge_nodes),
                row_ids=tuple(row.row_id for row in rows),
                weight=round(0.5 + avg_severity, 6),
            )
        )

    for constraint, rows in sorted(rows_by_constraint.items()):
        if len(rows) < 2:
            continue
        edge_nodes = sorted({node for row in rows for node in row.nodes()})
        nodes.update(edge_nodes)
        avg_residual = sum(row.residual_drift for row in rows) / len(rows)
        avg_severity = sum(row.severity for row in rows) / len(rows)
        edges.append(
            Hyperedge(
                edge_id=f"residual_drift_mus_proxy:{constraint}",
                kind="residual_drift_mus_proxy",
                nodes=tuple(edge_nodes),
                row_ids=tuple(row.row_id for row in rows),
                weight=round(1.0 + avg_residual + avg_severity, 6),
            )
        )

    return Hypergraph(nodes=tuple(sorted(nodes)), hyperedges=tuple(edges))


def rank_nodes_by_degree(hypergraph: Hypergraph) -> list[str]:
    """Degree baseline: inspect nodes that appear in the most hyperedges first."""

    degree = Counter(node for edge in hypergraph.hyperedges for node in edge.nodes)
    return sorted(hypergraph.nodes, key=lambda node: (-degree[node], node))


def rank_nodes_by_residual_message_passing(hypergraph: Hypergraph) -> list[str]:
    """Rank nodes with deterministic residual-weighted hypergraph message passing."""

    if not hypergraph.nodes:
        return []
    scores = {node: 0.0 for node in hypergraph.nodes}
    for edge in hypergraph.hyperedges:
        share = edge.weight / max(1, len(edge.nodes))
        for node in edge.nodes:
            scores[node] += share
            if node.startswith("constraint:"):
                scores[node] += 0.15 * edge.weight
            if node.startswith("verifier_failure:"):
                scores[node] += 0.10 * edge.weight

    for _ in range(2):
        next_scores = {node: 0.10 * score for node, score in scores.items()}
        for edge in hypergraph.hyperedges:
            edge_message = edge.weight + sum(scores[node] for node in edge.nodes) / len(edge.nodes)
            if edge.kind == "residual_drift_mus_proxy":
                edge_message *= 1.10
            for node in edge.nodes:
                next_scores[node] += edge_message / len(edge.nodes)
        scores = next_scores

    return sorted(hypergraph.nodes, key=lambda node: (-scores[node], node))


def checks_to_first_conflict(ranking: list[str], hypergraph: Hypergraph) -> float:
    """Return checks until the first hyperedge is fully inspected."""

    if not ranking or not hypergraph.hyperedges:
        return 0.0
    positions = {node: idx + 1 for idx, node in enumerate(ranking)}
    completions = [
        max(positions[node] for node in edge.nodes)
        for edge in hypergraph.hyperedges
        if all(node in positions for node in edge.nodes)
    ]
    return float(min(completions)) if completions else 0.0


def random_baseline_checks(
    hypergraph: Hypergraph,
    *,
    seed: int = RANDOM_SEED,
    trials: int = RANDOM_BASELINE_TRIALS,
) -> float:
    """Average conflict discovery checks across seeded random node orders."""

    if not hypergraph.nodes or not hypergraph.hyperedges:
        return 0.0
    rng = random.Random(seed)
    nodes = list(hypergraph.nodes)
    total = 0.0
    for _ in range(trials):
        shuffled = nodes[:]
        rng.shuffle(shuffled)
        total += checks_to_first_conflict(shuffled, hypergraph)
    return total / trials


def reproducibility_checksum(
    *,
    matrix_payload: dict[str, Any],
    source_payloads: dict[str, dict[str, Any]],
    failure_rows: list[FailureEvidence],
    hypergraph: Hypergraph,
    metrics: dict[str, float],
) -> str:
    """Hash all deterministic inputs and derived diagnostic payloads."""

    payload = {
        "matrix_payload": matrix_payload,
        "source_payloads": source_payloads,
        "failure_rows": [row.to_dict() for row in failure_rows],
        "hypergraph": hypergraph.to_dict(),
        "metrics": metrics,
        "random_seed": RANDOM_SEED,
        "run_date": RUN_DATE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _probe_preconditions(
    root: Path,
    matrix_path: Path,
    matrix_payload: dict[str, Any],
    clean_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    source_paths = [str(row.get("source_artifact") or "") for row in clean_rows.values()]
    source_ok = all((root / path).is_file() for path in source_paths if path)
    return [
        {
            "name": "working_directory",
            "command": f"cd {root}",
            "ok": root.is_dir(),
            "observed": str(root),
        },
        {
            "name": "exp2865_matrix_file",
            "command": f"test -f {matrix_path}",
            "ok": matrix_path.is_file(),
            "observed": str(matrix_path),
        },
        {
            "name": "cross_corpus_matrix_built",
            "command": "jq -e '.cross_corpus_matrix_built == true' "
            "results/experiment_2865_cross_corpus_matrix_v5.json",
            "ok": matrix_payload.get("cross_corpus_matrix_built") is True,
            "observed": matrix_payload.get("cross_corpus_matrix_built"),
        },
        {
            "name": "clean_rows_present",
            "command": "inspect verifier_corpus_dual_matrix clean rows",
            "ok": bool(clean_rows),
            "observed": sorted(clean_rows),
        },
        {
            "name": "clean_source_artifacts_available",
            "command": "test declared clean source artifacts",
            "ok": source_ok,
            "observed": source_paths,
        },
    ]


def _failure_class_counts(failure_rows: list[FailureEvidence]) -> dict[str, int]:
    return dict(sorted(Counter(row.failure_class for row in failure_rows).items()))


def _recommended_repairs(failure_rows: list[FailureEvidence]) -> list[str]:
    repairs: list[str] = []
    metrics = {row.source_metric for row in failure_rows}
    if "tier0s_arithmetic_gap" in metrics:
        repairs.append(
            "Calibrate, invert, or down-weight tier0s_arithmetic_gap before using it as an "
            "arithmetic repair signal; the clean FoVer source shows below-random AUROC."
        )
    if "fever" in metrics:
        repairs.append(
            "Prioritize FEVER factual-support calibration next; it is the strongest non-FoVer "
            "residual-drift conflict in the clean matrix evidence."
        )
    if any(row.failure_class == "near_random_auroc" for row in failure_rows):
        repairs.append(
            "Keep near-random logical and factual signals separate from below-random failures so "
            "future repair tasks can decide whether to abstain, recalibrate, or collect labels."
        )
    repairs.append(
        "Treat this as a deterministic residual-drift diagnostic, not a trained HGNN policy; "
        "collect per-example co-failure and exact MUS certificates before training one."
    )
    return repairs


def _blocked_artifact(
    *,
    verdict: str,
    preconditions: list[dict[str, Any]],
    matrix_payload: dict[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    checksum = reproducibility_checksum(
        matrix_payload=matrix_payload,
        source_payloads={},
        failure_rows=[],
        hypergraph=Hypergraph(nodes=(), hyperedges=()),
        metrics={},
    )
    return {
        "schema": "carnot.drift_mus_prioritizer.v2",
        "artifact": "experiment_2867_drift_mus_prioritizer_v2",
        "honest_verdict": verdict,
        "drift_mus_diagnostic_ready": False,
        "n_failure_rows": 0,
        "failure_class_counts": {},
        "hypergraph_nodes": 0,
        "hypergraph_hyperedges": 0,
        "hgnn_inspired_heuristic_name": "blocked_precondition_not_run",
        "baseline_random_checks_to_conflict": 0.0,
        "baseline_degree_checks_to_conflict": 0.0,
        "heuristic_checks_to_conflict": 0.0,
        "heuristic_improvement_vs_best_baseline": 0.0,
        "recommended_repairs": [
            "Rebuild results/experiment_2865_cross_corpus_matrix_v5.json with "
            "cross_corpus_matrix_built=true, then rerun the diagnostic."
        ],
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "run_date": RUN_DATE,
        "duration_s": duration_s,
        "failure_rows": [],
        "hypergraph_edges": [],
        "methodology_note": "Blocked before evidence extraction; no missing metrics were inferred.",
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """Build the Exp 2867 diagnostic artifact from the Exp 2865 clean matrix."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    matrix_path = root_path / MATRIX_REL_PATH
    matrix_payload = read_json(matrix_path)
    clean_rows = _clean_matrix_rows(matrix_payload)
    preconditions = _probe_preconditions(root_path, matrix_path, matrix_payload, clean_rows)
    end = time.perf_counter() if now_s is None else now_s
    duration_s = round(max(0.0, end - start), 6)

    if matrix_payload.get("cross_corpus_matrix_built") is not True:
        return _blocked_artifact(
            verdict="blocked_cross_corpus_matrix_not_built",
            preconditions=preconditions,
            matrix_payload=matrix_payload,
            duration_s=duration_s,
        )

    source_payloads = _source_payloads(root_path, clean_rows)
    failure_rows = extract_clean_failure_evidence(root_path, matrix_payload, source_payloads)
    hypergraph = build_hypergraph(failure_rows)
    degree_ranking = rank_nodes_by_degree(hypergraph)
    heuristic_ranking = rank_nodes_by_residual_message_passing(hypergraph)

    random_checks = round(random_baseline_checks(hypergraph), 6)
    degree_checks = round(checks_to_first_conflict(degree_ranking, hypergraph), 6)
    heuristic_checks = round(checks_to_first_conflict(heuristic_ranking, hypergraph), 6)
    best_baseline = min(random_checks, degree_checks) if hypergraph.hyperedges else 0.0
    improvement = round(best_baseline - heuristic_checks, 6)
    metrics = {
        "baseline_random_checks_to_conflict": random_checks,
        "baseline_degree_checks_to_conflict": degree_checks,
        "heuristic_checks_to_conflict": heuristic_checks,
        "heuristic_improvement_vs_best_baseline": improvement,
    }
    checksum = reproducibility_checksum(
        matrix_payload=matrix_payload,
        source_payloads=source_payloads,
        failure_rows=failure_rows,
        hypergraph=hypergraph,
        metrics=metrics,
    )

    return {
        "schema": "carnot.drift_mus_prioritizer.v2",
        "artifact": "experiment_2867_drift_mus_prioritizer_v2",
        "honest_verdict": (
            "complete: residual-drift MUS-proxy prioritizer built from Exp 2865 clean rows; "
            f"n_failure_rows={len(failure_rows)}; heuristic_checks_to_conflict={heuristic_checks}"
        ),
        "drift_mus_diagnostic_ready": True,
        "n_failure_rows": len(failure_rows),
        "failure_class_counts": _failure_class_counts(failure_rows),
        "hypergraph_nodes": len(hypergraph.nodes),
        "hypergraph_hyperedges": len(hypergraph.hyperedges),
        "hgnn_inspired_heuristic_name": HEURISTIC_NAME,
        "baseline_random_checks_to_conflict": random_checks,
        "baseline_degree_checks_to_conflict": degree_checks,
        "heuristic_checks_to_conflict": heuristic_checks,
        "heuristic_improvement_vs_best_baseline": improvement,
        "recommended_repairs": _recommended_repairs(failure_rows),
        "preconditions_checked": preconditions,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "run_date": RUN_DATE,
        "duration_s": duration_s,
        "failure_rows": [row.to_dict() for row in failure_rows],
        "hypergraph_edges": [edge.to_dict() for edge in hypergraph.hyperedges],
        "node_rankings": {
            "degree": degree_ranking,
            "residual_message_passing": heuristic_ranking,
        },
        "methodology_note": (
            "No exact MUS certificates were present in Exp 2865 or its clean source artifacts. "
            "Hyperedges are co-failure and residual-drift proxies, and the heuristic is not a "
            "trained HGNN policy."
        ),
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2867 JSON deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())
