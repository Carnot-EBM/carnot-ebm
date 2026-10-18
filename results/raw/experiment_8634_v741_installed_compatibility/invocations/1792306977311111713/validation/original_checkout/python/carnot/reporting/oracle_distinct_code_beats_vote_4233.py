"""Exp 4233 oracle-distinct code pass-predictor beats-vote gate.

Spec refs: REQ-VERIFY-4233, SCENARIO-VERIFY-4233,
SCENARIO-VERIFY-4233-NO-HEADROOM, SCENARIO-VERIFY-4233-BLOCKED.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import random
import re
import subprocess
import sys
import time
import tokenize
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


RANDOM_SEED = 4233
BOOTSTRAP_RESAMPLES = 2000
HASH_BUCKETS = 64
OUTPUT_REL = Path("results/experiment_4233_oracle_distinct_code_beats_vote.json")
SPEC_REFS = [
    "REQ-VERIFY-4233",
    "SCENARIO-VERIFY-4233",
    "SCENARIO-VERIFY-4233-NO-HEADROOM",
    "SCENARIO-VERIFY-4233-BLOCKED",
]
INFERENCE_SUBSTRATE = "cached_code_candidates_oracle_distinct_sklearn_cpu"
BLOCKED_VERDICT = "blocked_code_candidate_pool_missing"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A code beats-vote win, a code ties-at-power null, "
        "or an honest no-pool/no-headroom is COMPLETE -- each disambiguates the ARC null."
    ),
    "code_oracle_distinct_beats_vote": (
        "BARE bool: predictor@1 - vote@1 CI95 excludes 0 AND delta>0 AND "
        "headroom -- a LEARNED (non-executing) code verifier beating vote; "
        "NOT the circular execution result."
    ),
    "code_predictor_minus_vote_delta": (
        "predictor@1 - vote@1 on held-out code -- the oracle-distinct lift on "
        "a balanced, high-power domain."
    ),
    "code_predictor_minus_vote_ci95": (
        "Task-level bootstrap CI95 of the code delta -- excluding 0 "
        "distinguishes a real win from noise at power code provides."
    ),
    "oracle_at_k": (
        "code positive-control ceiling (any candidate passes) -- if ~=vote the "
        "null is uninformative."
    ),
    "held_out_task_n": (
        "BARE int: the code gate's N -- expected >> the .391 ARC n=14, so a "
        "code null is high-power."
    ),
    "disambiguation_read": (
        "ARC_null_is_data_sparsity (code wins -> build a bigger ARC pool) / "
        "selection_thesis_bounded (code also ties -> the oracle-distinct "
        "selection thesis is bounded) / code_no_headroom -- the load-bearing "
        "cross-domain read on the .391 ARC null."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the predictor scores code WITHOUT executing tests "
        "at inference (the hidden-test label is the training target only); "
        "this keeps the result oracle-distinct, not circular."
    ),
    "model_specs": (
        "The pass-predictor architecture + oracle-distinct code feature set + "
        "calibrated loss; required methodology."
    ),
    "random_seed": (
        "Determinism precondition; fold split + init seeded."
    ),
    "reproducibility_checksum": (
        "Hash of the code candidate pool + fold split; lets a third party re-run."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "code_oracle_distinct_beats_vote",
    "code_predictor_minus_vote_delta",
    "code_predictor_minus_vote_ci95",
    "oracle_at_k",
    "held_out_task_n",
    "disambiguation_read",
    "verifier_is_oracle",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)

BASE_FEATURE_NAMES = (
    "char_len",
    "line_count",
    "blank_line_frac",
    "comment_line_frac",
    "max_line_len",
    "mean_line_len",
    "token_count",
    "unique_token_frac",
    "operator_token_frac",
    "name_token_frac",
    "literal_token_frac",
    "ast_parse_ok",
    "ast_node_count",
    "ast_max_depth",
    "function_def_count",
    "class_def_count",
    "return_count",
    "import_count",
    "assign_count",
    "if_count",
    "loop_count",
    "try_count",
    "except_count",
    "raise_count",
    "call_count",
    "compare_count",
    "boolop_count",
    "binop_count",
    "subscript_count",
    "listcomp_count",
    "lambda_count",
    "assert_count",
    "placeholder_count",
    "docstring_count",
    "prompt_overlap_frac",
)
HASH_FEATURE_NAMES = tuple(f"hash_bucket_{index:02d}" for index in range(HASH_BUCKETS))
CROSS_FEATURE_NAMES = (
    "task_candidate_count",
    "task_signature_count",
    "task_signature_frac",
    "task_signature_is_modal",
    "task_signature_margin",
    "task_signature_unique_count",
    "task_signature_entropy",
    "code_len_zscore",
    "token_count_zscore",
    "line_count_zscore",
    "mean_pairwise_token_jaccard",
    "max_pairwise_token_jaccard",
    "modal_token_jaccard",
    "self_consistency_margin",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + HASH_FEATURE_NAMES + CROSS_FEATURE_NAMES

LABEL_FIELDS = (
    "hidden_pass",
    "passes_hidden_tests",
    "pass_hidden_tests",
    "hidden_tests_passed",
)
CODE_FIELDS = (
    "completion",
    "candidate",
    "code",
    "source",
    "generated_code",
    "text",
)
TASK_FIELDS = ("task_id", "task", "problem_id", "id")


@dataclass(frozen=True)
class PoolSpec:
    source_id: str
    paths: tuple[Path, ...]


DEFAULT_POOL_SPECS = (
    PoolSpec(
        "exp2830_2837_2838_ensemble_json",
        (
            Path("results/experiment_2830_humaneval_full_ensemble_eval.json"),
            Path("results/experiment_2837_mbpp_ensemble_eval.json"),
            Path("results/experiment_2838_humaneval_full_ensemble_eval.json"),
        ),
    ),
    PoolSpec("exp1607_dsl_humaneval_json", (Path("results/experiment_1607_dsl_humaneval.json"),)),
    PoolSpec(
        "verifier_reward_3arm_lora_rft_a83b52882c198954",
        (
            Path(
                "results/verifier_reward_3arm_lora_rft/"
                "code_verifier_reward_lora_rft_a83b52882c198954/corpora/arm_A.jsonl"
            ),
            Path(
                "results/verifier_reward_3arm_lora_rft/"
                "code_verifier_reward_lora_rft_a83b52882c198954/corpora/arm_B.jsonl"
            ),
            Path(
                "results/verifier_reward_3arm_lora_rft/"
                "code_verifier_reward_lora_rft_a83b52882c198954/corpora/arm_C.jsonl"
            ),
        ),
    ),
    PoolSpec(
        "verifier_reward_3arm_lora_rft_91b7244bb09edd32",
        (
            Path(
                "results/verifier_reward_3arm_lora_rft/"
                "code_verifier_reward_lora_rft_91b7244bb09edd32/corpora/arm_A.jsonl"
            ),
            Path(
                "results/verifier_reward_3arm_lora_rft/"
                "code_verifier_reward_lora_rft_91b7244bb09edd32/corpora/arm_B.jsonl"
            ),
            Path(
                "results/verifier_reward_3arm_lora_rft/"
                "code_verifier_reward_lora_rft_91b7244bb09edd32/corpora/arm_C.jsonl"
            ),
        ),
    ),
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""

    def __init__(self, reason: str, attempted_sources: list[dict[str, Any]]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempted_sources = attempted_sources


@dataclass(frozen=True)
class CodeCandidate:
    source_id: str
    source_path: str
    row_index: int
    task_id: str
    candidate_id: str
    candidate_index: int
    code: str
    prompt: str
    passes_hidden_tests: bool
    vote_signature: str


@dataclass(frozen=True)
class FeatureRow:
    candidate: CodeCandidate
    features: dict[str, float]
    token_set: frozenset[str]
    learned_score: float = 0.0
    fold: int = -1
    train_task_excluded: bool = False


@dataclass(frozen=True)
class CandidatePool:
    source_id: str
    rows: list[CodeCandidate]
    source_paths: list[Path]
    source_sha256: dict[str, str]
    task_n: int
    candidate_n: int
    positive_n: int
    pass_rate: float
    attempted_sources: list[dict[str, Any]]
    vote_signature_source: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _stable_bucket(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], "big") % HASH_BUCKETS


def _as_bool_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _first_string(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _label_from_row(row: dict[str, Any]) -> bool | None:
    for key in LABEL_FIELDS:
        if key in row:
            label = _as_bool_label(row[key])
            if label is not None:
                return label
    return None


def _candidate_index(row: dict[str, Any], fallback: int) -> int:
    for key in ("candidate_index", "source_draw_index", "draw_index", "sample_index", "seed"):
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return fallback


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _json_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
        return records
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(_walk_dicts(payload))


def _normalize_fallback_text(code: str) -> str:
    lines = [" ".join(line.strip().split()) for line in code.splitlines() if line.strip()]
    return "\n".join(lines)


def normalized_code_signature(code: str) -> str:
    """Return a non-executing answer signature for vote/self-consistency."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        normalized = _normalize_fallback_text(code)
    else:
        normalized = ast.dump(tree, include_attributes=False, annotate_fields=True)
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def _candidate_from_record(
    source_id: str,
    source_path: Path,
    row: dict[str, Any],
    row_index: int,
) -> CodeCandidate | None:
    task_id = _first_string(row, TASK_FIELDS)
    code = _first_string(row, CODE_FIELDS)
    label = _label_from_row(row)
    if not task_id or not code or label is None:
        return None
    prompt = row.get("prompt") if isinstance(row.get("prompt"), str) else ""
    candidate_index = _candidate_index(row, row_index)
    signature = normalized_code_signature(code)
    raw_id = {
        "source_id": source_id,
        "task_id": task_id,
        "candidate_index": candidate_index,
        "row_index": row_index,
        "signature": signature,
    }
    candidate_id = hashlib.sha256(
        json.dumps(raw_id, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CodeCandidate(
        source_id=source_id,
        source_path=str(source_path),
        row_index=row_index,
        task_id=task_id,
        candidate_id=f"{source_id}:{task_id}:{candidate_index}:{candidate_id[:12]}",
        candidate_index=candidate_index,
        code=code,
        prompt=prompt,
        passes_hidden_tests=label,
        vote_signature=signature,
    )


def _load_rows_for_spec(repo_root: Path, spec: PoolSpec) -> tuple[list[CodeCandidate], dict[str, Any]]:
    rows: list[CodeCandidate] = []
    path_reports: list[dict[str, Any]] = []
    for rel_path in spec.paths:
        path = rel_path if rel_path.is_absolute() else repo_root / rel_path
        if not path.exists():
            path_reports.append({"path": str(rel_path), "exists": False, "candidate_rows": 0})
            continue
        try:
            records = _json_records(path)
        except Exception as exc:
            path_reports.append(
                {
                    "path": str(rel_path),
                    "exists": True,
                    "candidate_rows": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        start = len(rows)
        for row_index, record in enumerate(records):
            candidate = _candidate_from_record(spec.source_id, path, record, row_index)
            if candidate is not None:
                rows.append(candidate)
        path_reports.append(
            {
                "path": str(rel_path),
                "exists": True,
                "candidate_rows": len(rows) - start,
            }
        )
    task_counts = Counter(row.task_id for row in rows)
    labels = [row.passes_hidden_tests for row in rows]
    viable_task_ids = {task_id for task_id, count in task_counts.items() if count >= 2}
    viable_rows = [row for row in rows if row.task_id in viable_task_ids]
    report = {
        "source_id": spec.source_id,
        "paths": path_reports,
        "candidate_rows": len(rows),
        "viable_candidate_rows": len(viable_rows),
        "task_n": len(viable_task_ids),
        "positive_n": sum(1 for row in viable_rows if row.passes_hidden_tests),
        "has_both_labels": len(set(labels)) == 2,
    }
    return viable_rows, report


def load_candidate_pool(
    repo_root: Path | str,
    pool_specs: tuple[PoolSpec, ...] = DEFAULT_POOL_SPECS,
) -> CandidatePool:
    root = Path(repo_root)
    attempted_sources: list[dict[str, Any]] = []
    for spec in pool_specs:
        rows, report = _load_rows_for_spec(root, spec)
        attempted_sources.append(report)
        labels = [row.passes_hidden_tests for row in rows]
        if rows and len({row.task_id for row in rows}) > 0 and len(set(labels)) == 2:
            source_paths = [
                path if path.is_absolute() else root / path
                for path in spec.paths
                if (path if path.is_absolute() else root / path).exists()
            ]
            source_sha = {str(path): _sha256_file(path) for path in source_paths}
            return CandidatePool(
                source_id=spec.source_id,
                rows=rows,
                source_paths=source_paths,
                source_sha256=source_sha,
                task_n=len({row.task_id for row in rows}),
                candidate_n=len(rows),
                positive_n=sum(1 for row in rows if row.passes_hidden_tests),
                pass_rate=sum(1 for row in rows if row.passes_hidden_tests) / float(len(rows)),
                attempted_sources=attempted_sources,
                vote_signature_source="normalized_code_text_signature",
            )
    raise BlockedRun(BLOCKED_VERDICT, attempted_sources)


def _token_items(code: str) -> tuple[list[str], int, int, int]:
    tokens: list[str] = []
    operators = 0
    names = 0
    literals = 0
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for tok in stream:
            if tok.type in (tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE):
                continue
            token_text = tok.string.strip()
            if not token_text:
                continue
            tokens.append(f"{tok.type}:{token_text}")
            if tok.type == tokenize.OP:
                operators += 1
            elif tok.type == tokenize.NAME:
                names += 1
            elif tok.type in (tokenize.NUMBER, tokenize.STRING):
                literals += 1
    except tokenize.TokenError:
        words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]", code)
        tokens = [f"fallback:{word}" for word in words]
        names = sum(1 for word in words if re.match(r"[A-Za-z_]", word))
        literals = sum(1 for word in words if word.isdigit())
        operators = max(0, len(tokens) - names - literals)
    return tokens, operators, names, literals


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_ast_depth(child) for child in children)


def _ast_counts(code: str) -> tuple[dict[str, float], bool]:
    names = {
        "ast_node_count": 0.0,
        "ast_max_depth": 0.0,
        "function_def_count": 0.0,
        "class_def_count": 0.0,
        "return_count": 0.0,
        "import_count": 0.0,
        "assign_count": 0.0,
        "if_count": 0.0,
        "loop_count": 0.0,
        "try_count": 0.0,
        "except_count": 0.0,
        "raise_count": 0.0,
        "call_count": 0.0,
        "compare_count": 0.0,
        "boolop_count": 0.0,
        "binop_count": 0.0,
        "subscript_count": 0.0,
        "listcomp_count": 0.0,
        "lambda_count": 0.0,
        "assert_count": 0.0,
        "docstring_count": 0.0,
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return names, False
    nodes = list(ast.walk(tree))
    names["ast_node_count"] = float(len(nodes))
    names["ast_max_depth"] = float(_ast_depth(tree))
    names["function_def_count"] = float(sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in nodes))
    names["class_def_count"] = float(sum(isinstance(n, ast.ClassDef) for n in nodes))
    names["return_count"] = float(sum(isinstance(n, ast.Return) for n in nodes))
    names["import_count"] = float(sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in nodes))
    names["assign_count"] = float(sum(isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for n in nodes))
    names["if_count"] = float(sum(isinstance(n, ast.If) for n in nodes))
    names["loop_count"] = float(sum(isinstance(n, (ast.For, ast.AsyncFor, ast.While)) for n in nodes))
    names["try_count"] = float(sum(isinstance(n, ast.Try) for n in nodes))
    names["except_count"] = float(sum(isinstance(n, ast.ExceptHandler) for n in nodes))
    names["raise_count"] = float(sum(isinstance(n, ast.Raise) for n in nodes))
    names["call_count"] = float(sum(isinstance(n, ast.Call) for n in nodes))
    names["compare_count"] = float(sum(isinstance(n, ast.Compare) for n in nodes))
    names["boolop_count"] = float(sum(isinstance(n, ast.BoolOp) for n in nodes))
    names["binop_count"] = float(sum(isinstance(n, ast.BinOp) for n in nodes))
    names["subscript_count"] = float(sum(isinstance(n, ast.Subscript) for n in nodes))
    names["listcomp_count"] = float(sum(isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) for n in nodes))
    names["lambda_count"] = float(sum(isinstance(n, ast.Lambda) for n in nodes))
    names["assert_count"] = float(sum(isinstance(n, ast.Assert) for n in nodes))
    names["docstring_count"] = float(
        sum(
            1
            for n in nodes
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and ast.get_docstring(n) is not None
        )
    )
    return names, True


def _safe_frac(numer: float, denom: float) -> float:
    return numer / denom if denom else 0.0


def _extract_base_features(candidate: CodeCandidate) -> tuple[dict[str, float], frozenset[str]]:
    code = candidate.code
    lines = code.splitlines() or [""]
    nonempty = [line for line in lines if line.strip()]
    comment_lines = [line for line in lines if line.strip().startswith("#")]
    tokens, operators, names, literals = _token_items(code)
    token_count = len(tokens)
    token_set = frozenset(tokens)
    ast_feature_values, ast_ok = _ast_counts(code)
    prompt_words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", candidate.prompt.lower()))
    code_words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code.lower()))
    placeholder_count = sum(code.lower().count(term) for term in ("todo", "notimplemented", "pass"))
    features: dict[str, float] = {
        "char_len": float(len(code)),
        "line_count": float(len(lines)),
        "blank_line_frac": _safe_frac(len(lines) - len(nonempty), len(lines)),
        "comment_line_frac": _safe_frac(len(comment_lines), len(lines)),
        "max_line_len": float(max((len(line) for line in lines), default=0)),
        "mean_line_len": sum(len(line) for line in lines) / float(len(lines)),
        "token_count": float(token_count),
        "unique_token_frac": _safe_frac(len(token_set), token_count),
        "operator_token_frac": _safe_frac(operators, token_count),
        "name_token_frac": _safe_frac(names, token_count),
        "literal_token_frac": _safe_frac(literals, token_count),
        "ast_parse_ok": 1.0 if ast_ok else 0.0,
        "placeholder_count": float(placeholder_count),
        "prompt_overlap_frac": _safe_frac(len(prompt_words & code_words), len(prompt_words)),
    }
    features.update(ast_feature_values)
    hash_counts = [0.0] * HASH_BUCKETS
    for token in tokens:
        hash_counts[_stable_bucket(token)] += 1.0
    for index, count in enumerate(hash_counts):
        features[f"hash_bucket_{index:02d}"] = _safe_frac(count, token_count)
    return features, token_set


def _zscore(value: float, values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / float(len(values))
    var = sum((item - mean) ** 2 for item in values) / float(len(values))
    std = math.sqrt(var)
    return (value - mean) / std if std > 0.0 else 0.0


def _entropy(counts: Iterable[int]) -> float:
    values = [count for count in counts if count > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in values)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = len(left | right)
    return len(left & right) / float(union) if union else 0.0


def build_feature_matrix(rows: list[CodeCandidate]) -> tuple[list[FeatureRow], list[str]]:
    base_rows: list[tuple[CodeCandidate, dict[str, float], frozenset[str]]] = []
    for candidate in rows:
        features, token_set = _extract_base_features(candidate)
        base_rows.append((candidate, features, token_set))

    by_task: dict[str, list[tuple[CodeCandidate, dict[str, float], frozenset[str]]]] = defaultdict(list)
    for item in base_rows:
        by_task[item[0].task_id].append(item)

    enriched: list[FeatureRow] = []
    for task_items in by_task.values():
        n = len(task_items)
        signature_counts = Counter(item[0].vote_signature for item in task_items)
        count_values = sorted(signature_counts.values(), reverse=True)
        modal_signature, modal_count = signature_counts.most_common(1)[0]
        second_count = count_values[1] if len(count_values) > 1 else 0
        len_values = [item[1]["char_len"] for item in task_items]
        token_values = [item[1]["token_count"] for item in task_items]
        line_values = [item[1]["line_count"] for item in task_items]
        modal_tokens: frozenset[str] = frozenset()
        for candidate, _features, token_set in task_items:
            if candidate.vote_signature == modal_signature:
                modal_tokens = token_set
                break
        for candidate, features, token_set in task_items:
            sig_count = signature_counts[candidate.vote_signature]
            is_modal = candidate.vote_signature == modal_signature
            other_jaccards = [
                _jaccard(token_set, other_tokens)
                for other_candidate, _other_features, other_tokens in task_items
                if other_candidate.candidate_id != candidate.candidate_id
            ]
            margin_numer = sig_count - (second_count if is_modal else modal_count)
            cross = {
                "task_candidate_count": float(n),
                "task_signature_count": float(sig_count),
                "task_signature_frac": sig_count / float(n),
                "task_signature_is_modal": 1.0 if is_modal else 0.0,
                "task_signature_margin": margin_numer / float(n),
                "task_signature_unique_count": float(len(signature_counts)),
                "task_signature_entropy": _entropy(signature_counts.values()),
                "code_len_zscore": _zscore(features["char_len"], len_values),
                "token_count_zscore": _zscore(features["token_count"], token_values),
                "line_count_zscore": _zscore(features["line_count"], line_values),
                "mean_pairwise_token_jaccard": (
                    sum(other_jaccards) / float(len(other_jaccards)) if other_jaccards else 1.0
                ),
                "max_pairwise_token_jaccard": max(other_jaccards, default=1.0),
                "modal_token_jaccard": _jaccard(token_set, modal_tokens),
                "self_consistency_margin": (modal_count - second_count) / float(n),
            }
            merged = dict(features)
            merged.update(cross)
            enriched.append(FeatureRow(candidate=candidate, features=merged, token_set=token_set))
    return sorted(enriched, key=lambda row: (row.candidate.task_id, row.candidate.candidate_index, row.candidate.row_index)), list(FEATURE_NAMES)


def _matrix(feature_rows: list[FeatureRow], feature_names: list[str]) -> list[list[float]]:
    return [[float(row.features.get(name, 0.0)) for name in feature_names] for row in feature_rows]


def _labels(feature_rows: list[FeatureRow]) -> list[int]:
    return [1 if row.candidate.passes_hidden_tests else 0 for row in feature_rows]


def _task_folds(task_ids: list[str], random_seed: int, n_folds: int = 5) -> list[list[str]]:
    shuffled = list(task_ids)
    random.Random(random_seed).shuffle(shuffled)
    fold_n = min(max(2, n_folds), len(shuffled))
    folds = [[] for _ in range(fold_n)]
    for index, task_id in enumerate(shuffled):
        folds[index % fold_n].append(task_id)
    return [sorted(fold) for fold in folds if fold]


def auroc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [score for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return 0.5
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    pos_rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label == 1)
    return (pos_rank_sum - len(positives) * (len(positives) + 1) / 2.0) / (
        len(positives) * len(negatives)
    )


def train_oof_predictor(
    feature_rows: list[FeatureRow],
    feature_names: list[str],
    *,
    random_seed: int,
    n_folds: int = 5,
) -> tuple[list[FeatureRow], list[list[str]], float]:
    task_ids = sorted({row.candidate.task_id for row in feature_rows})
    folds = _task_folds(task_ids, random_seed, n_folds=n_folds)
    rows_by_id = {row.candidate.candidate_id: row for row in feature_rows}
    scored: dict[str, FeatureRow] = {}
    for fold_index, held_out_tasks in enumerate(folds):
        held_set = set(held_out_tasks)
        train_rows = [row for row in feature_rows if row.candidate.task_id not in held_set]
        test_rows = [row for row in feature_rows if row.candidate.task_id in held_set]
        train_labels = _labels(train_rows)
        if len(set(train_labels)) < 2:
            fallback_score = sum(train_labels) / float(len(train_labels)) if train_labels else 0.5
            for row in test_rows:
                scored[row.candidate.candidate_id] = FeatureRow(
                    row.candidate,
                    row.features,
                    row.token_set,
                    learned_score=float(fallback_score),
                    fold=fold_index,
                    train_task_excluded=True,
                )
            continue
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed + fold_index,
            solver="liblinear",
        )
        train_x = _matrix(train_rows, feature_names)
        test_x = _matrix(test_rows, feature_names)
        model.fit(train_x, train_labels)
        raw_train = [float(value) for value in model.predict_proba(train_x)[:, 1]]
        raw_test = [float(value) for value in model.predict_proba(test_x)[:, 1]]
        if len(set(raw_train)) > 1:
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(raw_train, train_labels)
            fold_scores = [float(value) for value in calibrator.transform(raw_test)]
        else:
            fold_scores = raw_test
        for row, score in zip(test_rows, fold_scores, strict=True):
            scored[row.candidate.candidate_id] = FeatureRow(
                row.candidate,
                row.features,
                row.token_set,
                learned_score=max(0.0, min(1.0, score)),
                fold=fold_index,
                train_task_excluded=True,
            )
    ordered = [scored.get(row.candidate.candidate_id, rows_by_id[row.candidate.candidate_id]) for row in feature_rows]
    return ordered, folds, auroc([row.learned_score for row in ordered], _labels(ordered))


def _group_by_task(rows: list[FeatureRow]) -> list[list[FeatureRow]]:
    grouped: dict[str, list[FeatureRow]] = defaultdict(list)
    for row in rows:
        grouped[row.candidate.task_id].append(row)
    return [
        sorted(task_rows, key=lambda row: (row.candidate.candidate_index, row.candidate.row_index))
        for _task_id, task_rows in sorted(grouped.items())
    ]


def _select_predictor(task_rows: list[FeatureRow]) -> FeatureRow:
    return max(
        task_rows,
        key=lambda row: (
            row.learned_score,
            row.features["task_signature_count"],
            -row.candidate.candidate_index,
            -row.candidate.row_index,
        ),
    )


def _select_control(task_rows: list[FeatureRow]) -> FeatureRow:
    return min(task_rows, key=lambda row: (row.candidate.candidate_index, row.candidate.row_index))


def _select_vote_signature(task_rows: list[FeatureRow]) -> tuple[str, list[FeatureRow]]:
    counts = Counter(row.candidate.vote_signature for row in task_rows)
    first_index: dict[str, tuple[int, int]] = {}
    for row in task_rows:
        first_index.setdefault(row.candidate.vote_signature, (row.candidate.candidate_index, row.candidate.row_index))
    signature = max(counts, key=lambda sig: (counts[sig], -first_index[sig][0], -first_index[sig][1]))
    return signature, [row for row in task_rows if row.candidate.vote_signature == signature]


def _rate(values: list[bool]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _bootstrap_ci95(deltas: list[float], *, random_seed: int, resamples: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    rng = random.Random(random_seed)
    n = len(deltas)
    means = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / float(n)
        for _ in range(int(resamples))
    ]
    means.sort()
    return [
        _round_metric(means[int(0.025 * (len(means) - 1))]),
        _round_metric(means[int(0.975 * (len(means) - 1))]),
    ]


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return bool(len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0))


def measure_gate(
    scored_rows: list[FeatureRow],
    *,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    tasks = _group_by_task(scored_rows)
    predictor_hits: list[bool] = []
    vote_hits: list[bool] = []
    control_hits: list[bool] = []
    oracle_hits: list[bool] = []
    deltas: list[float] = []
    control_deltas: list[float] = []
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        predictor_pick = _select_predictor(task)
        control_pick = _select_control(task)
        vote_signature, vote_rows = _select_vote_signature(task)
        oracle_hit = any(row.candidate.passes_hidden_tests for row in task)
        vote_hit = any(row.candidate.passes_hidden_tests for row in vote_rows)
        predictor_hit = predictor_pick.candidate.passes_hidden_tests
        control_hit = control_pick.candidate.passes_hidden_tests
        predictor_hits.append(predictor_hit)
        vote_hits.append(vote_hit)
        control_hits.append(control_hit)
        oracle_hits.append(oracle_hit)
        deltas.append(float(predictor_hit) - float(vote_hit))
        control_deltas.append(float(predictor_hit) - float(control_hit))
        task_rows.append(
            {
                "task_id": task[0].candidate.task_id,
                "oracle_hit": oracle_hit,
                "vote_signature": vote_signature,
                "vote_signature_count": len(vote_rows),
                "vote_correct": vote_hit,
                "predictor_candidate_id": predictor_pick.candidate.candidate_id,
                "predictor_correct": predictor_hit,
                "predictor_score": _round_metric(predictor_pick.learned_score),
                "matched_control_candidate_id": control_pick.candidate.candidate_id,
                "matched_control_correct": control_hit,
                "train_task_excluded": all(row.train_task_excluded for row in task),
            }
        )
    vote_at_1 = _rate(vote_hits)
    predictor_at_1 = _rate(predictor_hits)
    control_at_1 = _rate(control_hits)
    oracle_at_k = _rate(oracle_hits)
    delta = _round_metric(predictor_at_1 - vote_at_1)
    ci95 = _bootstrap_ci95(deltas, random_seed=random_seed, resamples=bootstrap_resamples)
    headroom_exists = oracle_at_k > vote_at_1 + 1e-12
    beats_vote = bool(headroom_exists and delta > 0.0 and ci95[0] > 0.0)
    if not headroom_exists:
        disambiguation = "code_no_headroom"
        honest_verdict = "complete: code_no_headroom"
    elif beats_vote:
        disambiguation = "ARC_null_is_data_sparsity"
        honest_verdict = "complete: code_oracle_distinct_beats_vote"
    else:
        disambiguation = "selection_thesis_bounded"
        honest_verdict = "complete: selection_thesis_bounded"
    return {
        "honest_verdict": honest_verdict,
        "code_oracle_distinct_beats_vote": beats_vote,
        "code_predictor_minus_vote_delta": delta,
        "code_predictor_minus_vote_ci95": ci95,
        "oracle_at_k": _round_metric(oracle_at_k),
        "held_out_task_n": len(tasks),
        "disambiguation_read": disambiguation,
        "pass_rates": {
            "predictor_at_1": _round_metric(predictor_at_1),
            "vote_at_1": _round_metric(vote_at_1),
            "matched_control_at_1": _round_metric(control_at_1),
        },
        "matched_control_delta": _round_metric(
            sum(control_deltas) / float(len(control_deltas)) if control_deltas else 0.0
        ),
        "oracle_minus_vote": _round_metric(oracle_at_k - vote_at_1),
        "headroom_exists": headroom_exists,
        "ci95_excludes_zero": _ci_excludes_zero(ci95),
        "bootstrap_resamples": int(bootstrap_resamples),
        "task_rows": task_rows,
    }


def _model_specs(feature_names: list[str]) -> dict[str, Any]:
    return {
        "architecture": "task_held_out_code_text_logistic_pass_predictor",
        "loss": "class_weight_balanced_logistic_loss",
        "calibration": "train_fold_isotonic_on_raw_probabilities",
        "feature_set": (
            "code text hash embeddings, lexical/AST statistics, normalized code "
            "signature duplicate agreement, and cross-candidate self-consistency margins"
        ),
        "feature_names": feature_names,
        "forbidden_inference_signals": [
            "hidden_pass",
            "passes_hidden_tests",
            "visible_perfect",
            "arm",
            "test_execution",
            "runtime_output_signature",
        ],
        "verifier_is_oracle": False,
    }


def reproducibility_checksum(
    pool: CandidatePool,
    folds: list[list[str]],
    feature_names: list[str],
    random_seed: int,
) -> str:
    payload = {
        "candidate_ids": [row.candidate_id for row in pool.rows],
        "folds": folds,
        "feature_names": feature_names,
        "random_seed": int(random_seed),
        "source_id": pool.source_id,
        "source_sha256": pool.source_sha256,
        "vote_signature_source": pool.vote_signature_source,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_checksum(reason: str, random_seed: int, attempted_sources: list[dict[str, Any]]) -> str:
    raw = json.dumps(
        {"reason": reason, "random_seed": random_seed, "attempted_sources": attempted_sources},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(
    attempted_sources: list[dict[str, Any]],
    *,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    checksum = _blocked_checksum(BLOCKED_VERDICT, random_seed, attempted_sources)
    return {
        "experiment": "experiment_4233_oracle_distinct_code_beats_vote",
        "schema": "carnot.oracle_distinct_code_beats_vote_4233.v1",
        "status": "complete",
        "honest_verdict": BLOCKED_VERDICT,
        "code_oracle_distinct_beats_vote": False,
        "code_predictor_minus_vote_delta": 0.0,
        "code_predictor_minus_vote_ci95": [0.0, 0.0],
        "oracle_at_k": 0.0,
        "held_out_task_n": 0,
        "disambiguation_read": BLOCKED_VERDICT,
        "verifier_is_oracle": False,
        "model_specs": _model_specs(list(FEATURE_NAMES)),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "off_fold_auroc": 0.0,
        "pass_rates": {
            "predictor_at_1": 0.0,
            "vote_at_1": 0.0,
            "matched_control_at_1": 0.0,
        },
        "matched_control_delta": 0.0,
        "oracle_minus_vote": 0.0,
        "headroom_exists": False,
        "ci95_excludes_zero": False,
        "bootstrap_resamples": 0,
        "candidate_pool": {
            "source_id": "",
            "candidate_n": 0,
            "task_n": 0,
            "positive_n": 0,
            "pass_rate": 0.0,
        },
        "attempted_candidate_sources": attempted_sources,
        "vote_signature_source": "normalized_code_text_signature",
        "task_rows": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "methodology_note": (
            "Blocked before training because no cached code candidate pool had both "
            "multi-candidate task structure and per-candidate hidden-test labels."
        ),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    pool: CandidatePool,
    metrics: dict[str, Any],
    *,
    off_fold_auroc: float,
    folds: list[list[str]],
    feature_names: list[str],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    checksum = reproducibility_checksum(pool, folds, feature_names, random_seed)
    return {
        "experiment": "experiment_4233_oracle_distinct_code_beats_vote",
        "schema": "carnot.oracle_distinct_code_beats_vote_4233.v1",
        "status": "complete",
        **metrics,
        "verifier_is_oracle": False,
        "model_specs": _model_specs(feature_names),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "off_fold_auroc": _round_metric(off_fold_auroc),
        "candidate_pool": {
            "source_id": pool.source_id,
            "candidate_n": pool.candidate_n,
            "task_n": pool.task_n,
            "positive_n": pool.positive_n,
            "pass_rate": _round_metric(pool.pass_rate),
            "source_paths": [str(path) for path in pool.source_paths],
        },
        "attempted_candidate_sources": pool.attempted_sources,
        "vote_signature_source": pool.vote_signature_source,
        "fold_task_ids": folds,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "methodology_note": (
            "Hidden-test pass labels are supervised targets only. Candidate scoring "
            "uses code text and cross-candidate signature features without executing "
            "candidate code or tests at inference."
        ),
        "adversarial_verify": {"status": "pending"},
    }


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "adversarial_verify.py"), "--json", str(artifact_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def _clean_adversarial_report(report: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    for item in report.get("reports", []):
        if isinstance(item, dict):
            flags.extend(flag for flag in item.get("flags", []) if isinstance(flag, dict))
    circular_clean = not any(flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags)
    return {
        "status": "clean" if not flags else "flagged",
        "circular_moat_overclaim_clean": circular_clean,
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict == BLOCKED_VERDICT
    ):
        raise ValueError("honest_verdict must be terminal-prefixed or a blocked verdict")
    if type(artifact["code_oracle_distinct_beats_vote"]) is not bool:
        raise ValueError("code_oracle_distinct_beats_vote must be a bare bool")
    for field in ("code_predictor_minus_vote_delta", "oracle_at_k"):
        if isinstance(artifact[field], bool) or not isinstance(artifact[field], (int, float)):
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["code_predictor_minus_vote_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("code_predictor_minus_vote_ci95 must be a two-number ci95")
    if type(artifact["held_out_task_n"]) is not int:
        raise ValueError("held_out_task_n must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be present")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4233")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4233")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    pool_specs: tuple[PoolSpec, ...] = DEFAULT_POOL_SPECS,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        pool = load_candidate_pool(root, pool_specs)
        feature_rows, feature_names = build_feature_matrix(pool.rows)
        scored_rows, folds, off_fold_auroc = train_oof_predictor(
            feature_rows,
            feature_names,
            random_seed=random_seed,
        )
        metrics = measure_gate(
            scored_rows,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        artifact = _complete_artifact(
            pool,
            metrics,
            off_fold_auroc=off_fold_auroc,
            folds=folds,
            feature_names=feature_names,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        artifact = _blocked_artifact(
            exc.attempted_sources,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_report = (
        adversarial_runner(output_path)
        if adversarial_runner is not None
        else _run_adversarial_verify(root, output_path)
    )
    artifact["adversarial_verify"] = _clean_adversarial_report(raw_report)
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
