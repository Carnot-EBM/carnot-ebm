"""Tiny exact arithmetic frontier check for Exp 2866.

What is exact here:
    Z3 decides ground arithmetic equalities on a small, deterministic FoVer
    subset. The reported frontier is the first text prefix that ends at a
    solver-proved false completed equality.

What is not exact here:
    This is not full BEAVER. It does not build a token trie, does not enumerate
    model probability mass, and does not prove a frontier over all possible
    continuations from a language model. Exp 2858 proxy scores are compared
    against the exact local arithmetic result, not promoted into exact BEAVER.

Spec: REQ-VERIFY-2866, SCENARIO-VERIFY-2866
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from carnot.verify.beaver_epr_bounded_probe import (
    ArithmeticFalseClaimConstraint,
    LabeledExample,
    _example_from_row,
    _read_rows,
)

try:  # pragma: no cover - exercised through monkeypatch in tests.
    import z3  # type: ignore[import]
except Exception:  # pragma: no cover - environment-dependent import fallback.
    z3 = None  # type: ignore[assignment]

_Z3_IMPORT = z3

OUTPUT_FILENAME = "experiment_2866_beaver_exact_tiny_frontier_v1.json"
RUN_DATE = "20260522"
RANDOM_SEED = 2866
N_EXAMPLES = 6
REPO_ROOT = Path(__file__).resolve().parents[3]
FOVER_PATH = Path("data/fover_corpus.jsonl")
PROXY_ARTIFACT_PATH = Path("results/experiment_2858_beaver_epr_clean_bounded_proxy_v2.json")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "exact_beaver_implemented",
    "exact_frontier_available",
    "n_examples",
    "solver_used",
    "blocked_reason",
    "exact_vs_proxy_comparison",
    "sample_rows",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "exact_beaver_implemented": (
        "false: no token-trie or model probability frontier proof is implemented."
    ),
    "exact_frontier_available": (
        "true only for the bounded FoVer subset whose completed arithmetic "
        "equalities are all Z3-decidable."
    ),
    "proxy_boundary": (
        "Exp 2858 bounded-prefix scores are comparison baselines, not exact evidence."
    ),
    "solver_boundary": (
        "Z3 checks ground arithmetic equalities in observed text prefixes only."
    ),
}

_NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
_OPERATOR = r"(?:\+|-|\*|/|\\times|×|[xX])"
_EQUALITY_RE = re.compile(
    rf"(?P<expr>{_NUMBER}(?:\s*{_OPERATOR}\s*{_NUMBER})+)"
    rf"\s*=\s*(?:<<[^>]*>>\s*)?\$?(?P<claimed>{_NUMBER})"
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for the Exp 2866 local exact-frontier attempt."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    run_date: str = RUN_DATE
    n_examples: int = N_EXAMPLES
    random_seed: int = RANDOM_SEED
    tests_run: tuple[str, ...] | list[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    fover_path: Path = FOVER_PATH
    proxy_artifact_path: Path = PROXY_ARTIFACT_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_fover_path(self) -> Path:
        return self.repo_root / self.fover_path

    def resolved_proxy_artifact_path(self) -> Path:
        return self.repo_root / self.proxy_artifact_path

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        return self.repo_root / "results" / OUTPUT_FILENAME


@dataclass(frozen=True)
class ExactArithmeticClaim:
    """One completed equality decided by the local exact arithmetic solver."""

    expression: str
    claimed: str
    prefix_length: int
    satisfied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "claimed": self.claimed,
            "prefix_length": self.prefix_length,
            "satisfied": self.satisfied,
        }


def check_preconditions(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Return the resource checks that decide whether exact local scoring can run."""

    solver = solver_used()
    return [
        {
            "step": "test -f data/fover_corpus.jsonl",
            "passed": config.resolved_fover_path().is_file(),
            "observed": str(config.resolved_fover_path()),
        },
        {
            "step": "test -f results/experiment_2858_beaver_epr_clean_bounded_proxy_v2.json",
            "passed": config.resolved_proxy_artifact_path().is_file(),
            "observed": str(config.resolved_proxy_artifact_path()),
        },
        {
            "step": ".venv/bin/python -c \"import z3; print(z3.get_version_string())\"",
            "passed": solver is not None,
            "observed": solver or "missing z3-solver",
        },
    ]


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, Any]:
    """Run Exp 2866 and optionally write the terminal JSON artifact."""

    cfg = config or ExperimentConfig()
    started_at = cfg.start_time()
    preconditions = check_preconditions(cfg)
    duration = lambda: cfg.clock() - started_at

    if not preconditions[2]["passed"]:
        artifact = _blocked_artifact(
            cfg,
            preconditions,
            duration(),
            "blocked_dependency: z3-solver unavailable for exact frontier",
        )
    elif not preconditions[0]["passed"] or not preconditions[1]["passed"]:
        artifact = _blocked_artifact(
            cfg,
            preconditions,
            duration(),
            "blocked_dependency: required FoVer corpus or Exp 2858 proxy artifact missing",
        )
    else:
        proxy_artifact = _read_proxy_artifact(cfg.resolved_proxy_artifact_path())
        selected_rows = _load_selected_exact_rows(cfg)
        labels = {int(row["label"]) for row in selected_rows}
        if len(selected_rows) != cfg.n_examples or labels != {0, 1}:
            artifact = _blocked_artifact(
                cfg,
                preconditions,
                duration(),
                "blocked_dependency: insufficient solver-decidable labeled FoVer subset",
            )
        else:
            artifact = _success_artifact(
                cfg,
                preconditions,
                selected_rows,
                proxy_artifact,
                duration(),
            )

    validate_artifact(artifact)
    if write:
        write_artifact(cfg.resolved_output_path(), artifact)
    return artifact


def score_example_exact_frontier(example: LabeledExample) -> dict[str, Any]:
    """Score one FoVer row with Z3 exact arithmetic and the Exp 2858 proxy."""

    claims, unsupported_claim_count = _extract_exact_claims(example.text)
    false_claims = [claim for claim in claims if not claim.satisfied]
    exact_frontier_available = bool(z3 is not None and claims and unsupported_claim_count == 0)
    exact_score = len(false_claims) / len(claims) if claims else 0.0
    proxy = ArithmeticFalseClaimConstraint().explore_prefixes(example.text)
    exact_decision = bool(false_claims)
    proxy_decision = proxy.score > 0.0
    return {
        "example_id": example.example_id,
        "label": int(example.label),
        "source": example.source,
        "exact_frontier_available": exact_frontier_available,
        "solver_claim_count": len(claims),
        "unsupported_claim_count": unsupported_claim_count,
        "exact_false_claim_count": len(false_claims),
        "first_exact_frontier_prefix_length": (
            min(claim.prefix_length for claim in false_claims) if false_claims else None
        ),
        "exact_score": exact_score,
        "bounded_prefix_proxy_score": proxy.score,
        "proxy_checked_claim_count": proxy.checked_claim_count,
        "proxy_false_claim_count": proxy.false_claim_count,
        "proxy_first_violation_prefix_length": proxy.first_violation_prefix_length,
        "exact_matches_proxy_decision": exact_decision == proxy_decision,
        "solver_claims": [claim.to_dict() for claim in claims],
    }


def solver_used() -> str | None:
    """Return the exact solver identity, or None when the dependency is absent."""

    if z3 is None:
        return None
    return f"z3-solver {z3.get_version_string()}"


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    """Write stable JSON for the Exp 2866 deliverable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 2866 schema and the exact/proxy boundary."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["run_date"] != RUN_DATE:
        raise ValueError("run_date must be 20260522")
    if artifact["exact_beaver_implemented"] is not False:
        raise ValueError("full exact BEAVER is not implemented by Exp 2866")
    if artifact["exact_frontier_available"] and artifact["solver_used"] is None:
        raise ValueError("exact_frontier_available requires solver_used")
    if artifact["exact_frontier_available"] and artifact["blocked_reason"] is not None:
        raise ValueError("successful exact frontier artifacts cannot have blocked_reason")
    if not artifact["exact_frontier_available"] and artifact["blocked_reason"] is None:
        raise ValueError("blocked artifacts require blocked_reason")


def main() -> int:  # pragma: no cover - command wrapper.
    run_experiment(ExperimentConfig(repo_root=Path.cwd()))
    return 0


def _success_artifact(
    config: ExperimentConfig,
    preconditions: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    proxy_artifact: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    comparison = _compare_exact_to_proxy(selected_rows, proxy_artifact)
    artifact = _base_artifact(config, preconditions, duration_s)
    artifact.update(
        {
            "honest_verdict": (
                "complete: tiny exact Z3 arithmetic frontier available on a bounded "
                "FoVer subset; full exact BEAVER remains unimplemented"
            ),
            "exact_frontier_available": True,
            "n_examples": len(selected_rows),
            "solver_used": solver_used(),
            "blocked_reason": None,
            "exact_vs_proxy_comparison": comparison,
            "sample_rows": [dict(row) for row in selected_rows],
        }
    )
    artifact["reproducibility_checksum"] = _checksum(
        {
            "run_date": config.run_date,
            "random_seed": config.random_seed,
            "solver_used": artifact["solver_used"],
            "sample_rows": artifact["sample_rows"],
            "exact_vs_proxy_comparison": comparison,
        }
    )
    return artifact


def _blocked_artifact(
    config: ExperimentConfig,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
    honest_verdict: str,
) -> dict[str, Any]:
    artifact = _base_artifact(config, preconditions, duration_s)
    artifact.update(
        {
            "honest_verdict": honest_verdict,
            "blocked_reason": "blocked_dependency",
            "solver_used": solver_used(),
            "exact_vs_proxy_comparison": {
                "comparison_status": "blocked",
                "boundary": FIELD_PRINCIPLES["proxy_boundary"],
            },
        }
    )
    artifact["reproducibility_checksum"] = _checksum(
        {
            "run_date": config.run_date,
            "random_seed": config.random_seed,
            "honest_verdict": honest_verdict,
            "preconditions_checked": list(preconditions),
        }
    )
    return artifact


def _base_artifact(
    config: ExperimentConfig,
    preconditions: Sequence[Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_2866_beaver_exact_tiny_frontier_v1",
        "schema": "carnot.beaver_exact_tiny_frontier.v1",
        "honest_verdict": "blocked_dependency: not evaluated",
        "exact_beaver_implemented": False,
        "exact_frontier_available": False,
        "n_examples": 0,
        "solver_used": None,
        "blocked_reason": "blocked_dependency",
        "exact_vs_proxy_comparison": {},
        "sample_rows": [],
        "random_seed": config.random_seed,
        "reproducibility_checksum": "",
        "preconditions_checked": [dict(check) for check in preconditions],
        "tests_run": list(config.tests_run),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": config.run_date,
        "duration_s": float(duration_s),
    }


def _load_selected_exact_rows(config: ExperimentConfig) -> list[dict[str, Any]]:
    examples = [
        example
        for index, row in enumerate(_read_rows(config.resolved_fover_path()))
        if (example := _example_from_row(row, config.resolved_fover_path(), index)) is not None
    ]
    scored_rows = [score_example_exact_frontier(example) for example in examples]
    exact_rows = [row for row in scored_rows if row["exact_frontier_available"]]
    positives = [row for row in exact_rows if int(row["label"]) == 1]
    negatives = [row for row in exact_rows if int(row["label"]) == 0]
    half = config.n_examples // 2
    negative_needed = config.n_examples - half
    if len(positives) < half or len(negatives) < negative_needed:
        return []
    rng = random.Random(config.random_seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    selected = negatives[:negative_needed] + positives[:half]
    rng.shuffle(selected)
    return selected


def _extract_exact_claims(text: str) -> tuple[list[ExactArithmeticClaim], int]:
    claims: list[ExactArithmeticClaim] = []
    unsupported_claim_count = 0
    for match in _EQUALITY_RE.finditer(text):
        expression = _normalize_expression(match.group("expr"))
        claimed = _normalize_expression(match.group("claimed"))
        try:
            satisfied = _z3_equality_holds(expression, claimed)
        except ValueError:
            unsupported_claim_count += 1
            continue
        claims.append(
            ExactArithmeticClaim(
                expression=expression,
                claimed=claimed,
                prefix_length=match.end("claimed"),
                satisfied=satisfied,
            )
        )
    return claims, unsupported_claim_count


def _z3_equality_holds(expression: str, claimed: str) -> bool:
    if z3 is None:
        raise ValueError("z3 unavailable")
    left = _z3_from_ast(_parse_arithmetic(expression))
    right = _z3_from_ast(_parse_arithmetic(claimed))
    solver = z3.Solver()
    solver.add(left == right)
    result = solver.check()
    if result == z3.sat:
        return True
    if result == z3.unsat:
        return False
    raise ValueError(f"z3 returned {result}")  # pragma: no cover - ground arithmetic is sat/unsat.


def _parse_arithmetic(expression: str) -> ast.AST:
    if not re.fullmatch(r"[-+*/().\d\s]+", expression):
        raise ValueError(f"unsupported arithmetic expression: {expression}")
    try:
        return ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        raise ValueError(f"unsupported arithmetic expression: {expression}") from exc


def _z3_from_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return _z3_real(Fraction(str(node.value)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_z3_from_ast(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _z3_from_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _z3_from_ast(node.left)
        right = _z3_from_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if _fraction_from_ast(node.right) == 0:
                raise ValueError("division by zero is not a supported exact claim")
            return left / right
    raise ValueError(f"unsupported arithmetic AST: {ast.dump(node)}")


def _fraction_from_ast(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_fraction_from_ast(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _fraction_from_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _fraction_from_ast(node.left)
        right = _fraction_from_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
    raise ValueError(f"unsupported arithmetic AST: {ast.dump(node)}")


def _z3_real(value: Fraction) -> Any:
    if z3 is None:
        raise ValueError("z3 unavailable")
    if value.denominator == 1:
        return z3.RealVal(value.numerator)
    return z3.RealVal(f"{value.numerator}/{value.denominator}")


def _normalize_expression(expression: str) -> str:
    normalized = expression.replace("\\times", "*").replace("×", "*")
    normalized = normalized.replace(",", "").replace("$", "")
    normalized = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", normalized)
    return normalized.strip()


def _read_proxy_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - malformed local artifact.
        return {}
    return payload if isinstance(payload, dict) else {}


def _compare_exact_to_proxy(
    selected_rows: Sequence[Mapping[str, Any]],
    proxy_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    agreements = [bool(row["exact_matches_proxy_decision"]) for row in selected_rows]
    exact_scores = [float(row["exact_score"]) for row in selected_rows]
    proxy_scores = [float(row["bounded_prefix_proxy_score"]) for row in selected_rows]
    return {
        "comparison_status": "complete",
        "proxy_artifact": str(PROXY_ARTIFACT_PATH),
        "exp2858_proxy_auc": float(proxy_artifact.get("bounded_prefix_proxy_auc") or 0.0),
        "exp2858_n_examples": int(proxy_artifact.get("n_examples") or 0),
        "exp2858_exact_beaver_implemented": bool(
            proxy_artifact.get("exact_beaver_implemented")
        ),
        "exact_frontier_available_on_subset": all(
            bool(row["exact_frontier_available"]) for row in selected_rows
        ),
        "comparison_subset_n": len(selected_rows),
        "exact_positive_rate": _mean([score > 0.0 for score in exact_scores]),
        "proxy_positive_rate": _mean([score > 0.0 for score in proxy_scores]),
        "decision_agreement_rate": _mean(agreements),
        "score_mean_absolute_delta": _mean(
            [abs(exact - proxy) for exact, proxy in zip(exact_scores, proxy_scores, strict=True)]
        ),
        "disagreements": [
            str(row["example_id"]) for row in selected_rows if not row["exact_matches_proxy_decision"]
        ],
        "boundary": FIELD_PRINCIPLES["proxy_boundary"],
    }


def _mean(values: Sequence[float | bool]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ExperimentConfig",
    "FIELD_PRINCIPLES",
    "FOVER_PATH",
    "N_EXAMPLES",
    "OUTPUT_FILENAME",
    "PROXY_ARTIFACT_PATH",
    "RANDOM_SEED",
    "REQUIRED_ARTIFACT_FIELDS",
    "RUN_DATE",
    "_Z3_IMPORT",
    "check_preconditions",
    "main",
    "run_experiment",
    "score_example_exact_frontier",
    "solver_used",
    "validate_artifact",
    "write_artifact",
]
