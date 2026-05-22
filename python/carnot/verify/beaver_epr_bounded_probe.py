"""Bounded-prefix BEAVER/EPR proxy for small arithmetic semantic constraints.

What this approximates:
    BEAVER v2 needs a sound token-trie/frontier proof before its probability
    bounds can be called exact.  This module does not implement that frontier
    proof.  It implements a bounded-prefix feasibility probe for one Carnot-
    expressible semantic constraint: an arithmetic trace must not contain a
    completed false equality.  Once such a prefix exists, later text cannot make
    that prefix valid under the trace discipline.

Where top-k logprobs are present in prior local telemetry, the module also
computes entropy-production features.  Missing top-k data is reported as
unavailable rather than simulated.

Spec: REQ-VERIFY-2843, SCENARIO-VERIFY-2843
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HEADLINE_REQUIRED_ANY_OF: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LEGACY_CPU_SMOKE_ONLY: tuple[str, ...] = ("Qwen3.5-0.8B", "gemma-4-E4B-it")
EXP2836_FILENAME = "experiment_2836_sota_runtime_preflight.json"
OUTPUT_FILENAME = "experiment_2843_beaver_epr_bounded_probe.json"
DEFAULT_RUN_DATE = "20260522"
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "beaver_exact",
    "bounded_prefix_probe_auc",
    "entropy_production_features_available",
    "topk_logprob_source",
    "n_examples",
    "model_specs",
    "preconditions_checked",
    "duration_s",
)
DEFAULT_FOVER_PATHS: tuple[str, ...] = (
    "data/fover_corpus.jsonl",
    "data/fover_test_v4.json",
    "data/fover_corpus_v4.json",
    "data/fover_test.json",
)
DEFAULT_TELEMETRY_PATHS: tuple[str, ...] = (
    "results/live_sota_balanced_telemetry_manifest_1480.jsonl",
    "results/live_sota_telemetry_manifest_1468.jsonl",
    "results/arm_ebm_logprob_telemetry_manifest_1556.jsonl",
)

_ARITHMETIC_EQUALITY_RE = re.compile(
    r"(?P<expr>[-+]?\d+(?:\.\d+)?(?:\s*(?:\+|-|\*|/)\s*[-+]?\d+(?:\.\d+)?)+)"
    r"\s*=\s*(?:<<[^>]*>>\s*)?\$?(?P<claimed>[-+]?\d+(?:\.\d+)?)"
)
_TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")


@dataclass(frozen=True)
class PreconditionCheck:
    """One pre-launch resource check recorded in the Exp 2843 artifact."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "resource": self.resource,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LabeledExample:
    """Local FoVer-style row normalized for bounded-prefix scoring."""

    example_id: str
    text: str
    label: int
    source: str


@dataclass(frozen=True)
class ArithmeticClaim:
    """A completed arithmetic equality observed in a prefix."""

    expression: str
    claimed: Decimal
    computed: Decimal
    satisfied: bool


@dataclass(frozen=True)
class PrefixState:
    """One bounded prefix and whether it already violates the constraint."""

    prefix_length: int
    terminal: bool
    checked_claim_count: int
    false_claim_count: int
    violates_constraint: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "prefix_length": self.prefix_length,
            "terminal": self.terminal,
            "checked_claim_count": self.checked_claim_count,
            "false_claim_count": self.false_claim_count,
            "violates_constraint": self.violates_constraint,
        }


@dataclass(frozen=True)
class PrefixProbeResult:
    """Bounded-prefix score for one response trace."""

    states: tuple[PrefixState, ...]
    checked_claim_count: int
    false_claim_count: int
    first_violation_prefix_length: int | None
    score: float

    @property
    def final_state(self) -> PrefixState:
        return self.states[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "checked_claim_count": self.checked_claim_count,
            "false_claim_count": self.false_claim_count,
            "first_violation_prefix_length": self.first_violation_prefix_length,
            "score": self.score,
            "final_state": self.final_state.to_dict(),
        }


@dataclass(frozen=True)
class EntropyProductionFeatures:
    """Entropy-production summary derived only from observed top-k logprobs."""

    available: bool
    position_count: int
    mean_entropy: float
    max_entropy: float
    total_positive_entropy_delta: float

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "position_count": self.position_count,
            "mean_entropy": self.mean_entropy,
            "max_entropy": self.max_entropy,
            "total_positive_entropy_delta": self.total_positive_entropy_delta,
        }


@dataclass(frozen=True)
class EntropyTelemetrySummary:
    """Aggregate EPR-style features over all telemetry rows with top-k data."""

    available: bool
    source: str
    n_examples: int
    entropy_production_auc: float | None
    mean_total_positive_entropy_delta: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "source": self.source,
            "n_examples": self.n_examples,
            "entropy_production_auc": self.entropy_production_auc,
            "mean_total_positive_entropy_delta": self.mean_total_positive_entropy_delta,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 2843."""

    repo_root: Path = Path(".")
    output_path: Path | None = None
    run_date: str = DEFAULT_RUN_DATE
    n_examples: int = 100
    random_seed: int = 42
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    fover_paths: tuple[str, ...] = DEFAULT_FOVER_PATHS
    telemetry_paths: tuple[str, ...] = DEFAULT_TELEMETRY_PATHS


class ArithmeticFalseClaimConstraint:
    """Prefix-closed "no completed false arithmetic equality" constraint."""

    def explore_prefixes(self, text: str, *, prefix_stride: int = 80) -> PrefixProbeResult:
        offsets = _prefix_offsets(len(text), prefix_stride)
        states = tuple(self._score_prefix(text[:offset], terminal=offset == len(text)) for offset in offsets)
        final_state = states[-1]
        first_violation = next(
            (state.prefix_length for state in states if state.violates_constraint),
            None,
        )
        score = (
            final_state.false_claim_count / final_state.checked_claim_count
            if final_state.checked_claim_count
            else 0.0
        )
        return PrefixProbeResult(
            states=states,
            checked_claim_count=final_state.checked_claim_count,
            false_claim_count=final_state.false_claim_count,
            first_violation_prefix_length=first_violation,
            score=score,
        )

    def _score_prefix(self, prefix: str, *, terminal: bool) -> PrefixState:
        claims = _extract_arithmetic_claims(prefix, terminal=terminal)
        false_claim_count = sum(not claim.satisfied for claim in claims)
        return PrefixState(
            prefix_length=len(prefix),
            terminal=terminal,
            checked_claim_count=len(claims),
            false_claim_count=false_claim_count,
            violates_constraint=false_claim_count > 0,
        )


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute binary AUROC with average ranks for tied scores."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    positives = sum(int(label) == 1 for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both positive and negative labels")

    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for row_index in range(index, end):
            ranks[ranked[row_index][0]] = average_rank
        index = end

    positive_rank_sum = sum(rank for label, rank in zip(labels, ranks, strict=True) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def entropy_production_from_topk(top_logprobs: Sequence[Mapping[str, Any]]) -> EntropyProductionFeatures:
    """Compute EPR-style entropy increases from observed top-k logprob rows."""

    entropies = [_entropy_from_logprob_dict(row) for row in top_logprobs]
    entropies = [value for value in entropies if value is not None]
    if not entropies:
        return EntropyProductionFeatures(False, 0, 0.0, 0.0, 0.0)
    positive_deltas = [
        entropies[index] - entropies[index - 1]
        for index in range(1, len(entropies))
        if entropies[index] > entropies[index - 1]
    ]
    return EntropyProductionFeatures(
        available=True,
        position_count=len(entropies),
        mean_entropy=sum(entropies) / len(entropies),
        max_entropy=max(entropies),
        total_positive_entropy_delta=sum(positive_deltas),
    )


def run_experiment(config: ExperimentConfig | None = None) -> dict[str, object]:
    """Run Exp 2843 and write the bounded-prefix/EPR proxy artifact."""

    cfg = config or ExperimentConfig()
    started_at = cfg.started_at if cfg.started_at is not None else cfg.clock()
    repo_root = cfg.repo_root
    output_path = cfg.output_path or repo_root / "results" / OUTPUT_FILENAME
    exp2836_path = repo_root / "results" / EXP2836_FILENAME
    exp2836 = _read_json(exp2836_path)
    preconditions = [
        PreconditionCheck("exp2836_artifact", bool(exp2836), _display_path(exp2836_path)),
        PreconditionCheck(
            "exp2836_sota_runtime_ready",
            bool(exp2836.get("sota_runtime_ready")),
            str(exp2836.get("sota_runtime_ready")),
        ),
        PreconditionCheck(
            "selected_loader_token_probabilities",
            _loader_can_expose_token_probabilities(exp2836),
            _loader_probability_detail(exp2836),
        ),
    ]

    if not exp2836.get("sota_runtime_ready"):
        artifact = _base_artifact(
            cfg=cfg,
            exp2836=exp2836,
            preconditions=preconditions,
            honest_verdict="blocked_exp2836_sota_runtime_not_ready",
            duration_s=cfg.clock() - started_at,
        )
        _write_json(output_path, artifact)
        return artifact

    examples, source_path = _select_labeled_examples(repo_root, cfg.fover_paths, cfg.n_examples)
    preconditions.append(
        PreconditionCheck(
            "fover_style_labeled_examples",
            len(examples) == cfg.n_examples,
            f"{_display_path(source_path)} count={len(examples)}",
        )
    )
    if len(examples) != cfg.n_examples:
        artifact = _base_artifact(
            cfg=cfg,
            exp2836=exp2836,
            preconditions=preconditions,
            honest_verdict="blocked_insufficient_labeled_examples",
            duration_s=cfg.clock() - started_at,
        )
        _write_json(output_path, artifact)
        return artifact

    constraint = ArithmeticFalseClaimConstraint()
    probe_rows = [_score_example(example, constraint) for example in examples]
    labels = [int(row["label"]) for row in probe_rows]
    scores = [float(row["score"]) for row in probe_rows]
    auc = compute_auroc(labels, scores)
    entropy_summary = _load_entropy_telemetry(repo_root, cfg.telemetry_paths)
    preconditions.append(
        PreconditionCheck(
            "topk_logprob_telemetry",
            entropy_summary.available,
            entropy_summary.source,
        )
    )

    artifact = _base_artifact(
        cfg=cfg,
        exp2836=exp2836,
        preconditions=preconditions,
        honest_verdict="complete: bounded-prefix/EPR proxy evaluated on local FoVer-style labels",
        duration_s=cfg.clock() - started_at,
    )
    artifact.update(
        {
            "bounded_prefix_probe_auc": auc,
            "entropy_production_features_available": entropy_summary.available,
            "topk_logprob_source": entropy_summary.source,
            "n_examples": len(examples),
            "example_source": _display_path(source_path),
            "bounded_prefix_constraint": {
                "name": "no_false_completed_arithmetic_equalities",
                "prefix_closed": True,
                "already_expressible_in_carnot": "arithmetic equality constraint energy",
            },
            "bounded_prefix_summary": _summarize_probe_rows(probe_rows),
            "entropy_production_summary": entropy_summary.to_dict(),
            "sample_rows": probe_rows[:5],
            "reproducibility_checksum": _reproducibility_checksum(
                cfg.random_seed,
                source_path,
                entropy_summary.source,
            ),
        }
    )
    _validate_terminal_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised through the script wrapper.
    run_experiment(ExperimentConfig(repo_root=Path.cwd()))
    return 0


def _prefix_offsets(text_length: int, prefix_stride: int) -> list[int]:
    stride = max(1, int(prefix_stride))
    offsets = list(range(stride, text_length + 1, stride))
    if not offsets or offsets[-1] != text_length:
        offsets.append(text_length)
    return offsets


def _extract_arithmetic_claims(text: str, *, terminal: bool = True) -> list[ArithmeticClaim]:
    normalized = _normalize_arithmetic_text(text)
    claims: list[ArithmeticClaim] = []
    for match in _ARITHMETIC_EQUALITY_RE.finditer(normalized):
        if not terminal and match.end() == len(normalized):
            continue
        computed = _safe_eval_arithmetic(match.group("expr"))
        claimed = _to_decimal(match.group("claimed"))
        if computed is None or claimed is None:
            continue
        claims.append(
            ArithmeticClaim(
                expression=match.group("expr"),
                claimed=claimed,
                computed=computed,
                satisfied=abs(computed - claimed) <= Decimal("1e-9"),
            )
        )
    return claims


def _normalize_arithmetic_text(text: str) -> str:
    normalized = text.replace("\\times", "*").replace("×", "*").replace("−", "-")
    normalized = normalized.replace(",", "").replace("$", "")
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
    return normalized


def _safe_eval_arithmetic(expression: str) -> Decimal | None:
    if not re.fullmatch(r"[-+*/().\d\s]+", expression):
        return None
    try:
        parsed = ast.parse(expression, mode="eval")
        return _eval_ast(parsed.body)
    except (SyntaxError, ValueError, InvalidOperation, DivisionByZero):
        return None


def _eval_ast(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"unsupported arithmetic expression: {ast.dump(node)}")


def _to_decimal(value: object) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _entropy_from_logprob_dict(row: Mapping[str, Any]) -> float | None:
    logprobs = [float(value) for value in row.values() if isinstance(value, int | float)]
    if not logprobs:
        return None
    max_logprob = max(logprobs)
    weights = [math.exp(value - max_logprob) for value in logprobs]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    return -sum(probability * math.log(probability) for probability in probabilities if probability)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    return [dict(row) for row in payload] if isinstance(payload, list) else []


def _select_labeled_examples(
    repo_root: Path,
    candidate_paths: Sequence[str],
    limit: int,
) -> tuple[list[LabeledExample], Path]:
    for relative_path in candidate_paths:
        path = repo_root / relative_path
        rows = [_example_from_row(row, path, index) for index, row in enumerate(_read_rows(path))]
        examples = [row for row in rows if row is not None]
        selected = _balanced_prefix(examples, limit)
        if len(selected) == limit:
            return selected, path
    return [], repo_root / candidate_paths[0]


def _balanced_prefix(examples: Sequence[LabeledExample], limit: int) -> list[LabeledExample]:
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    half = limit // 2
    if len(positives) >= half and len(negatives) >= limit - half:
        selected: list[LabeledExample] = []
        selected.extend(negatives[: limit - half])
        selected.extend(positives[:half])
        return selected
    return list(examples[:limit])


def _example_from_row(row: Mapping[str, Any], path: Path, index: int) -> LabeledExample | None:
    label = _label_to_incorrect(row)
    text = str(
        row.get("step_text")
        or row.get("response")
        or row.get("completion")
        or row.get("model_output")
        or ""
    ).strip()
    if label is None or not text:
        return None
    return LabeledExample(
        example_id=str(row.get("question_id") or row.get("case_id") or f"row-{index}"),
        text=text,
        label=label,
        source=_display_path(path),
    )


def _label_to_incorrect(row: Mapping[str, Any]) -> int | None:
    label = row.get("label", row.get("correctness_label"))
    if isinstance(label, str):
        lowered = label.lower()
        if lowered == "incorrect":
            return 1
        if lowered == "correct":
            return 0
    correct = row.get("correct")
    if isinstance(correct, bool):
        return 0 if correct else 1
    known = row.get("known_verifier_label")
    if isinstance(known, int | float) and not isinstance(known, bool):
        return 0 if int(known) == 1 else 1
    return None


def _score_example(
    example: LabeledExample,
    constraint: ArithmeticFalseClaimConstraint,
) -> dict[str, object]:
    result = constraint.explore_prefixes(example.text)
    return {
        "example_id": example.example_id,
        "label": example.label,
        "score": result.score,
        "checked_claim_count": result.checked_claim_count,
        "false_claim_count": result.false_claim_count,
        "first_violation_prefix_length": result.first_violation_prefix_length,
    }


def _load_entropy_telemetry(repo_root: Path, candidate_paths: Sequence[str]) -> EntropyTelemetrySummary:
    for relative_path in candidate_paths:
        path = repo_root / relative_path
        rows = _read_rows(path)
        features_and_labels = _entropy_rows(rows)
        if features_and_labels:
            labels = [label for label, _features in features_and_labels]
            scores = [
                features.total_positive_entropy_delta + features.mean_entropy
                for _label, features in features_and_labels
            ]
            auc = _optional_auroc(labels, scores)
            mean_delta = sum(
                features.total_positive_entropy_delta for _label, features in features_and_labels
            ) / len(features_and_labels)
            return EntropyTelemetrySummary(True, _display_path(path), len(features_and_labels), auc, mean_delta)
    return EntropyTelemetrySummary(False, "unavailable", 0, None, None)


def _entropy_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[int, EntropyProductionFeatures]]:
    parsed: list[tuple[int, EntropyProductionFeatures]] = []
    for row in rows:
        topk = row.get("top_logprobs") or row.get("topk_logprobs") or row.get("top_k_logprobs")
        if not isinstance(topk, list):
            continue
        features = entropy_production_from_topk(
            [item for item in topk if isinstance(item, Mapping)]
        )
        label = _label_to_incorrect(row)
        if features.available and label is not None:
            parsed.append((label, features))
    return parsed


def _optional_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    try:
        return compute_auroc(labels, scores)
    except ValueError:
        return None


def _loader_can_expose_token_probabilities(exp2836: Mapping[str, Any]) -> bool:
    loader_probe = exp2836.get("loader_probe")
    return isinstance(loader_probe, Mapping) and bool(loader_probe.get("llama_cpp_import_ok"))


def _loader_probability_detail(exp2836: Mapping[str, Any]) -> str:
    if _loader_can_expose_token_probabilities(exp2836):
        loader_probe = exp2836.get("loader_probe")
        return str((loader_probe or {}).get("llama_cpp_origin") or "llama_cpp_import_ok")
    return "llama_cpp loader unavailable or not recorded"


def _base_artifact(
    *,
    cfg: ExperimentConfig,
    exp2836: Mapping[str, Any],
    preconditions: Sequence[PreconditionCheck],
    honest_verdict: str,
    duration_s: float,
) -> dict[str, object]:
    artifact = {
        "artifact": "experiment_2843_beaver_epr_bounded_probe",
        "run_date": cfg.run_date,
        "schema_version": 1,
        "honest_verdict": honest_verdict,
        "beaver_exact": False,
        "beaver_method_label": "bounded-prefix/EPR proxy, not exact BEAVER",
        "bounded_prefix_probe_auc": None,
        "entropy_production_features_available": False,
        "topk_logprob_source": "unavailable",
        "n_examples": 0,
        "model_specs": _model_specs(exp2836),
        "preconditions_checked": [check.to_dict() for check in preconditions],
        "duration_s": float(duration_s),
        "random_seed": cfg.random_seed,
        "field_principles": _field_principles(),
        "failure_modes": {
            "proxy_not_exact_beaver": True,
            "frontier_proof_missing": True,
            "topk_entropy_only_when_available": True,
        },
    }
    _validate_terminal_artifact(artifact)
    return artifact


def _model_specs(exp2836: Mapping[str, Any]) -> dict[str, object]:
    return {
        "headline_required_any_of": list(HEADLINE_REQUIRED_ANY_OF),
        "legacy_cpu_smoke_only": list(LEGACY_CPU_SMOKE_ONLY),
        "exp2836_sota_runtime_ready": bool(exp2836.get("sota_runtime_ready")),
        "exp2836_selected_python": exp2836.get("selected_python"),
        "exp2836_model_specs": exp2836.get("model_specs", {}),
        "sota_models_cached": exp2836.get("sota_models_cached", []),
    }


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal/blocked prefix discipline.",
        "beaver_exact": "Prevents overstating a proxy as sound BEAVER.",
        "bounded_prefix_probe_auc": "Measured only if labels and scores exist.",
        "entropy_production_features_available": "Depends on loader logprob support.",
        "topk_logprob_source": "Reproducibility for EPR-style features.",
        "n_examples": "Sample-size transparency.",
        "model_specs": "Mandated SOTA GGUF recorded.",
        "preconditions_checked": "Explains blocks honestly.",
        "duration_s": "Real compute wall-time; no sleep padding.",
    }


def _summarize_probe_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    false_claim_rows = [row for row in rows if int(row["false_claim_count"]) > 0]
    no_claim_rows = [row for row in rows if int(row["checked_claim_count"]) == 0]
    return {
        "scored_examples": len(rows),
        "false_claim_rows": len(false_claim_rows),
        "no_arithmetic_claim_rows": len(no_claim_rows),
        "mean_score": sum(float(row["score"]) for row in rows) / len(rows),
    }


def _reproducibility_checksum(random_seed: int, source_path: Path, topk_source: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(random_seed).encode("utf-8"))
    digest.update(_display_path(source_path).encode("utf-8"))
    digest.update(topk_source.encode("utf-8"))
    if source_path.exists():
        digest.update(source_path.read_bytes()[:4096])
    return digest.hexdigest()[:16]


def _validate_terminal_artifact(artifact: Mapping[str, object]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not verdict.startswith(_TERMINAL_PREFIXES):
        raise ValueError(f"honest_verdict has disallowed prefix: {verdict}")
    if artifact["beaver_exact"] is not False:
        raise ValueError("Exp 2843 must not claim exact BEAVER")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


__all__ = [
    "ArithmeticFalseClaimConstraint",
    "DEFAULT_RUN_DATE",
    "EXP2836_FILENAME",
    "ExperimentConfig",
    "HEADLINE_REQUIRED_ANY_OF",
    "LEGACY_CPU_SMOKE_ONLY",
    "OUTPUT_FILENAME",
    "REQUIRED_ARTIFACT_FIELDS",
    "compute_auroc",
    "entropy_production_from_topk",
    "run_experiment",
]
