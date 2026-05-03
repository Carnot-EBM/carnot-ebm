"""SC-Energy verifier for set-level logical compatibility.

This module implements the Exp 1168 verifier contract.  It keeps RoBERTa CLS
pooling as the preferred encoder path, then learns a small diagonal
compatibility metric with the SC-Energy margin objective.

Spec: REQ-VERIFY-1168, SCENARIO-VERIFY-1168
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[+\-*/=<>]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_EQUATION_RE = re.compile(r"(?P<left>[-+*/().\d\s]+?)\s*=\s*(?P<right>[-+*/().\d\s]+)")


@dataclass(frozen=True)
class _Pair:
    response: str
    context: str = ""


class _DeterministicCLSBackend:
    """Small local encoder used when transformer weights are unavailable."""

    name = "deterministic_cls"

    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    def encode_cls(self, statements: list[str]) -> np.ndarray:
        rows = [
            _deterministic_statement_vector(statement, self.output_dim) for statement in statements
        ]
        if not rows:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        return np.vstack(rows).astype(np.float32)


class _TransformersCLSBackend:  # pragma: no cover - exercised only when weights are available.
    """RoBERTa CLS-pooling encoder backed by HuggingFace transformers."""

    name = "transformers_roberta_cls"

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self.output_dim = int(self._model.config.hidden_size)

    def encode_cls(self, statements: list[str]) -> np.ndarray:
        if not statements:
            return np.zeros((0, self.output_dim), dtype=np.float32)

        batch = self._tokenizer(
            statements,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            outputs = self._model(**batch)
        cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return cls.astype(np.float32)


class SCEnergyVerifier:
    """Set-Consistency Energy verifier for question/response statement sets.

    `encode()` returns one CLS-style embedding per statement. `energy()` scores
    context/response compatibility: lower energy means the statements are more
    mutually compatible; higher energy means the response is less compatible
    with the context. `train()` uses the SC-Energy hinge objective.

    Spec: REQ-VERIFY-1168
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        hidden_dim: int = 128,
        *,
        margin: float = 1.0,
        learning_rate: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.hidden_dim = int(hidden_dim)
        self.margin = float(margin)
        self.learning_rate = float(learning_rate)
        self._backend = _load_backend(model_name, self.hidden_dim)
        self.encoder_backend = self._backend.name
        self._projection: np.ndarray | None = None
        self._metric = -np.ones(self.hidden_dim, dtype=np.float32)
        self._bias = 1.0
        self._loss_history: list[float] = []

    @property
    def name(self) -> str:
        return "SCEnergyVerifier"

    @property
    def satisfaction_threshold(self) -> float:
        return 0.5

    @property
    def loss_history(self) -> list[float]:
        return list(self._loss_history)

    def encode(self, statements: list[str]) -> np.ndarray:
        """Return projected RoBERTa-style CLS embeddings for each statement.

        Spec: REQ-VERIFY-1168-2
        """
        normalized = _normalize_statements(statements)
        if not normalized:
            return np.zeros((0, self.hidden_dim), dtype=np.float32)
        raw = self._backend.encode_cls(normalized)
        return self._project(raw)

    def energy(self, response: str, context: str = "") -> float:
        """Return a scalar incompatibility energy for response against context.

        Spec: REQ-VERIFY-1168-3
        """
        feature = self._feature_for_pair(_Pair(response=response, context=context))
        return self._energy_from_feature(feature)

    def score(self, response: str, context: str = "") -> float:
        """Return energy clipped into the ensemble-friendly [0, 1] range."""
        return _clamp01(self.energy(response, context))

    def train(
        self,
        coherent_pairs: Sequence[Any],
        incoherent_pairs: Sequence[Any],
        n_epochs: int = 10,
    ) -> "SCEnergyVerifier":
        """Train the diagonal compatibility metric with SC-Energy margin loss.

        Spec: REQ-VERIFY-1168-4
        """
        coherent_features = [self._feature_for_pair(_coerce_pair(item)) for item in coherent_pairs]
        incoherent_features = [
            self._feature_for_pair(_coerce_pair(item)) for item in incoherent_pairs
        ]
        n_pairs = min(len(coherent_features), len(incoherent_features))
        if n_pairs == 0:
            self._loss_history.append(0.0)
            return self

        coherent_features = coherent_features[:n_pairs]
        incoherent_features = incoherent_features[:n_pairs]
        for _ in range(max(0, int(n_epochs))):
            for coherent, incoherent in zip(coherent_features, incoherent_features):
                gap = self._energy_from_feature(incoherent) - self._energy_from_feature(coherent)
                if gap < self.margin:
                    self._metric += self.learning_rate * (incoherent - coherent)
            self._metric = np.clip(self._metric, -8.0, 8.0)
            self._loss_history.append(
                _margin_loss(
                    coherent_features, incoherent_features, self._metric, self._bias, self.margin
                )
            )
        return self

    def contrastive_loss(
        self, coherent_pairs: Sequence[Any], incoherent_pairs: Sequence[Any]
    ) -> float:
        """Return the current mean margin loss for contrastive pairs."""
        coherent_features = [self._feature_for_pair(_coerce_pair(item)) for item in coherent_pairs]
        incoherent_features = [
            self._feature_for_pair(_coerce_pair(item)) for item in incoherent_pairs
        ]
        n_pairs = min(len(coherent_features), len(incoherent_features))
        if n_pairs == 0:
            return 0.0
        return _margin_loss(
            coherent_features[:n_pairs],
            incoherent_features[:n_pairs],
            self._metric,
            self._bias,
            self.margin,
        )

    def grad_energy(self, x: Any) -> np.ndarray:
        """Return a zero gradient placeholder for ConstraintTerm compatibility."""
        return np.zeros_like(np.asarray(x, dtype=float))

    def is_satisfied(self, response: str, context: str = "") -> bool:
        """Return whether the response energy is below the verifier threshold."""
        return self.score(response, context) < self.satisfaction_threshold

    def _feature_for_pair(self, pair: _Pair) -> np.ndarray:
        context_statements = _split_statements(pair.context)
        response_statements = _split_statements(pair.response)
        if not context_statements and not response_statements:
            return np.zeros(self.hidden_dim, dtype=np.float32)
        context_embeddings = self.encode(context_statements)
        response_embeddings = self.encode(response_statements)
        return _pair_feature(context_embeddings, response_embeddings)

    def _energy_from_feature(self, feature: np.ndarray) -> float:
        return float(self._bias + np.dot(self._metric, feature))

    def _project(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[1] == self.hidden_dim:
            return _l2_normalize(raw)
        if self._projection is None:
            self._projection = _make_projection(raw.shape[1], self.hidden_dim, self.model_name)
        return _l2_normalize(raw @ self._projection)


def _load_backend(model_name: str, hidden_dim: int) -> Any:
    if model_name.lower() in {"deterministic", "hash", "local"}:
        return _DeterministicCLSBackend(hidden_dim)
    try:
        return _TransformersCLSBackend(model_name)
    except Exception:
        return _DeterministicCLSBackend(hidden_dim)


def _normalize_statements(statements: list[str]) -> list[str]:
    return [statement.strip() for statement in statements if statement and statement.strip()]


def _split_statements(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.replace("\\n", "\n"))
    return _normalize_statements(parts)


def _coerce_pair(item: Any) -> _Pair:
    if isinstance(item, _Pair):
        return item
    if isinstance(item, dict):
        statements = item.get("statements")
        if isinstance(statements, list):
            return _coerce_pair(statements)
        response = item.get("response") or item.get("model_response") or item.get("step_text") or ""
        context = item.get("context") or item.get("question") or item.get("prompt") or ""
        return _Pair(str(response), str(context))
    if isinstance(item, tuple) and len(item) == 2:
        return _Pair(str(item[0]), str(item[1]))
    if isinstance(item, list):
        statements = [str(part) for part in item if str(part).strip()]
        if len(statements) <= 1:
            return _Pair(" ".join(statements), "")
        return _Pair(statements[-1], " ".join(statements[:-1]))
    return _Pair(str(item), "")


def _pair_feature(context_embeddings: np.ndarray, response_embeddings: np.ndarray) -> np.ndarray:
    if len(response_embeddings) == 0 and len(context_embeddings) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(response_embeddings) == 0:
        response_embeddings = context_embeddings
        context_embeddings = np.zeros((0, response_embeddings.shape[1]), dtype=np.float32)

    if len(context_embeddings) > 0:
        cross = (context_embeddings[:, None, :] * response_embeddings[None, :, :]).mean(axis=(0, 1))
    elif len(response_embeddings) > 1:
        products = [
            response_embeddings[i] * response_embeddings[j]
            for i in range(len(response_embeddings))
            for j in range(i + 1, len(response_embeddings))
        ]
        cross = np.vstack(products).mean(axis=0)
    else:
        cross = np.zeros(response_embeddings.shape[1], dtype=np.float32)

    unary = (response_embeddings * response_embeddings).mean(axis=0)
    return (0.85 * cross + 0.15 * unary).astype(np.float32)


def _margin_loss(
    coherent_features: Sequence[np.ndarray],
    incoherent_features: Sequence[np.ndarray],
    metric: np.ndarray,
    bias: float,
    margin: float,
) -> float:
    losses = []
    for coherent, incoherent in zip(coherent_features, incoherent_features):
        e_coherent = float(bias + np.dot(metric, coherent))
        e_incoherent = float(bias + np.dot(metric, incoherent))
        losses.append(max(0.0, margin - (e_incoherent - e_coherent)))
    return float(np.mean(losses)) if losses else 0.0


def _make_projection(input_dim: int, hidden_dim: int, model_name: str) -> np.ndarray:
    seed_bytes = hashlib.sha256(f"{model_name}:{input_dim}:{hidden_dim}".encode()).digest()
    seed = int.from_bytes(seed_bytes[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    scale = 1.0 / max(1.0, np.sqrt(float(input_dim)))
    return rng.normal(0.0, scale, size=(input_dim, hidden_dim)).astype(np.float32)


def _deterministic_statement_vector(statement: str, output_dim: int) -> np.ndarray:
    vector = np.zeros(output_dim, dtype=np.float32)
    tokens = _statement_tokens(statement)
    if not tokens:
        return vector
    for token in tokens:
        vector[_hash_token(token, output_dim)] += 1.0
    return _l2_normalize(vector[None, :])[0]


def _statement_tokens(statement: str) -> list[str]:
    tokens = [token.lower() for token in _WORD_RE.findall(statement)]
    marker = _equation_marker(statement)
    if marker:
        tokens.extend([marker] * 4)
    return tokens


def _equation_marker(statement: str) -> str:
    saw_equation = False
    for match in _EQUATION_RE.finditer(statement):
        left = match.group("left").strip()
        right = match.group("right").strip()
        if not left or not right:  # pragma: no cover - regex requires both groups.
            continue
        saw_equation = True
        try:
            if abs(_safe_eval_arithmetic(left) - _safe_eval_arithmetic(right)) > 1e-9:
                return "__arith_invalid__"
        except Exception:
            continue
    return "__arith_valid__" if saw_equation else ""


def _safe_eval_arithmetic(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return float(_eval_ast(tree.body))


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ValueError("division by zero")
        return left / right
    raise ValueError("unsupported arithmetic expression")


def _hash_token(token: str, output_dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % output_dim


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return (arr / norms).astype(np.float32)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
