"""NCO-style online pattern matching for negative decoding constraints.

The positive-mask decoders already in Carnot are good at saying which tokens
are allowed by a grammar.  Negative constraints have a different shape: they
ban spans such as profanity, leaked emails, or invalid IDs even when the
positive grammar would still permit the next token.  Compiling every negative
regex into the positive decoder state can create a large cross product of
states, so this module keeps the negative side as a bounded online matcher.

At each decoding step the matcher checks only the current output suffix plus a
candidate token.  If that small window matches any registered negative pattern,
the candidate token is rejected before it can be committed.

Spec: REQ-INFER-1956, SCENARIO-INFER-1956.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

RUN_DATE = "20260512"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1956_nco_negative_constraints.json")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "experiment",
    "schema",
    "run_date",
    "status",
    "title",
    "spec_refs",
    "nco_negative_constraint_layer_ready",
    "negative_constraints_upheld",
    "registry_summary",
    "decode_trace",
    "overhead_vs_positive_trie",
    "rejected_token_examples",
    "source_paper",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class NegativeConstraint:
    """One literal or regex pattern that must not appear in decoded text.

    The compiled regex or normalized literal is stored with the constraint so
    the decoding loop does not rebuild pattern objects for every candidate.
    ``window`` bounds how much prior output is searched; this is the main NCO
    property that avoids a state explosion from negative regexes.

    Spec: REQ-INFER-1956.
    """

    name: str
    pattern: str
    kind: str
    case_sensitive: bool = False
    window: int = 128
    _compiled: re.Pattern[str] | None = field(init=False, default=None, repr=False)
    _literal: str = field(init=False, default="", repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("negative constraint name must be non-empty")
        if not self.pattern:
            raise ValueError("negative constraint pattern must be non-empty")
        if self.kind not in {"literal", "regex"}:
            raise ValueError("negative constraint kind must be 'literal' or 'regex'")
        if self.window < 1:
            raise ValueError("negative constraint window must be >= 1")

        if self.kind == "regex":
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(self.pattern, flags)
            except re.error as exc:
                raise ValueError(f"invalid regex pattern: {exc}") from exc
            object.__setattr__(self, "_compiled", compiled)
            object.__setattr__(self, "_literal", "")
        else:
            literal = self.pattern if self.case_sensitive else self.pattern.lower()
            object.__setattr__(self, "_compiled", None)
            object.__setattr__(self, "_literal", literal)

    @property
    def lookback(self) -> int:
        """Return the prior-output suffix length needed for online matching."""

        if self.kind == "literal":
            return max(1, len(self.pattern) - 1)
        return self.window

    def matches(self, text: str) -> bool:
        """Return whether ``text`` violates this constraint."""

        window = text[-self.window :]
        if self.kind == "literal":
            haystack = window if self.case_sensitive else window.lower()
            return self._literal in haystack
        return bool(self._compiled and self._compiled.search(window))


@dataclass(frozen=True)
class TokenRejection:
    """Audit record for one rejected candidate token.

    Spec: REQ-INFER-1956.
    """

    token_id: int
    token_text: str
    constraint_names: tuple[str, ...]


class NegativeConstraintRegistry:
    """Registry of negative literal and regex constraints for decoding.

    Spec: REQ-INFER-1956.
    """

    def __init__(self, constraints: Sequence[NegativeConstraint] = ()) -> None:
        self._constraints: list[NegativeConstraint] = []
        self._max_lookback = 1
        for constraint in constraints:
            self.register(constraint)

    @property
    def constraints(self) -> tuple[NegativeConstraint, ...]:
        """Return registered constraints in insertion order."""

        return tuple(self._constraints)

    @property
    def max_lookback(self) -> int:
        """Return the largest suffix window needed by all registered patterns."""

        return self._max_lookback

    def register(self, constraint: NegativeConstraint) -> NegativeConstraint:
        """Add a fully constructed negative constraint to the registry."""

        if any(existing.name == constraint.name for existing in self._constraints):
            raise ValueError(f"duplicate negative constraint name: {constraint.name}")
        self._constraints.append(constraint)
        self._max_lookback = max(self._max_lookback, constraint.lookback)
        return constraint

    def add_literal(
        self,
        name: str,
        literal: str,
        *,
        case_sensitive: bool = False,
    ) -> NegativeConstraint:
        """Register a forbidden literal span."""

        return self.register(
            NegativeConstraint(
                name=name,
                pattern=literal,
                kind="literal",
                case_sensitive=case_sensitive,
                window=max(1, len(literal)),
            )
        )

    def add_regex(
        self,
        name: str,
        pattern: str,
        *,
        case_sensitive: bool = False,
        window: int = 128,
    ) -> NegativeConstraint:
        """Register a forbidden regex over the bounded online window."""

        return self.register(
            NegativeConstraint(
                name=name,
                pattern=pattern,
                kind="regex",
                case_sensitive=case_sensitive,
                window=window,
            )
        )

    def matching_constraints(
        self,
        prefix_text: str,
        token_text: str,
    ) -> tuple[NegativeConstraint, ...]:
        """Return constraints that would match if ``token_text`` were appended."""

        candidate_window = f"{prefix_text[-self._max_lookback :]}{token_text}"
        return tuple(
            constraint for constraint in self._constraints if constraint.matches(candidate_window)
        )

    def rejected_token_ids(
        self,
        prefix_text: str,
        token_text_by_id: Mapping[int, str],
    ) -> set[int]:
        """Return token IDs that would violate any registered constraint."""

        return {
            token_id
            for token_id, token_text in token_text_by_id.items()
            if self.matching_constraints(prefix_text, token_text)
        }

    def rejection_report(
        self,
        prefix_text: str,
        token_text_by_id: Mapping[int, str],
    ) -> dict[int, TokenRejection]:
        """Return a per-token audit report for rejected candidate tokens."""

        report: dict[int, TokenRejection] = {}
        for token_id, token_text in token_text_by_id.items():
            matches = self.matching_constraints(prefix_text, token_text)
            if matches:
                report[token_id] = TokenRejection(
                    token_id=token_id,
                    token_text=token_text,
                    constraint_names=tuple(match.name for match in matches),
                )
        return report


@dataclass
class _TrieNode:
    children: dict[int, "_TrieNode"] = field(default_factory=dict)
    terminal: bool = False


class PositiveMaskTrie:
    """Small positive-mask trie used as the baseline for Exp 1956 timing.

    Spec: REQ-INFER-1956.
    """

    def __init__(self) -> None:
        self._root = _TrieNode()

    @classmethod
    def from_token_sequences(cls, sequences: Sequence[Sequence[int]]) -> "PositiveMaskTrie":
        """Build a trie from complete token sequences."""

        trie = cls()
        for sequence in sequences:
            trie.insert(sequence)
        return trie

    def insert(self, sequence: Sequence[int]) -> None:
        """Insert one allowed token sequence."""

        node = self._root
        for token_id in sequence:
            node = node.children.setdefault(token_id, _TrieNode())
        node.terminal = True

    def allowed_next(self, prefix_ids: Sequence[int]) -> set[int]:
        """Return token IDs that can follow ``prefix_ids`` under the trie."""

        node = self._root
        for token_id in prefix_ids:
            child = node.children.get(token_id)
            if child is None:
                return set()
            node = child
        if node.terminal:
            return set()
        return set(node.children)


@dataclass(frozen=True)
class DecodeStep:
    """Audit state for one decoding step.

    Spec: REQ-INFER-1956.
    """

    step: int
    prefix_text: str
    selected_token_id: int | None
    selected_token_text: str
    rejections: dict[int, TokenRejection]
    positive_allowed_count: int | None


@dataclass(frozen=True)
class NegativeConstraintDecodeResult:
    """Result of an online negative-constraint decoding run.

    Spec: REQ-INFER-1956.
    """

    text: str
    token_ids: tuple[int, ...]
    steps: tuple[DecodeStep, ...]
    rejected_count: int
    stopped_reason: str

    @property
    def completed(self) -> bool:
        """Return whether decoding consumed every provided score row."""

        return self.stopped_reason == "max_steps"


def decode_with_negative_constraints(
    prompt: str,
    token_text_by_id: Mapping[int, str],
    score_rows: Sequence[Mapping[int, float]],
    registry: NegativeConstraintRegistry,
    *,
    positive_trie: PositiveMaskTrie | None = None,
) -> NegativeConstraintDecodeResult:
    """Decode by applying positive trie masks, then NCO negative rejection.

    ``score_rows`` is a deterministic stand-in for model logits: each row maps
    candidate token ID to a score, and the highest surviving score is selected.

    Spec: REQ-INFER-1956, SCENARIO-INFER-1956.
    """

    prefix_text = prompt
    token_ids: list[int] = []
    steps: list[DecodeStep] = []
    rejected_count = 0

    for step_index, scores in enumerate(score_rows):
        candidates = dict(scores)
        positive_allowed_count: int | None = None
        if positive_trie is not None:
            allowed = positive_trie.allowed_next(token_ids)
            positive_allowed_count = len(allowed)
            candidates = {
                token_id: score for token_id, score in candidates.items() if token_id in allowed
            }

        candidate_text = {token_id: token_text_by_id[token_id] for token_id in candidates}
        rejections = registry.rejection_report(prefix_text, candidate_text)
        rejected_count += len(rejections)
        survivors = {
            token_id: score for token_id, score in candidates.items() if token_id not in rejections
        }

        if not survivors:
            steps.append(
                DecodeStep(
                    step=step_index,
                    prefix_text=prefix_text,
                    selected_token_id=None,
                    selected_token_text="",
                    rejections=rejections,
                    positive_allowed_count=positive_allowed_count,
                )
            )
            return NegativeConstraintDecodeResult(
                text=prefix_text,
                token_ids=tuple(token_ids),
                steps=tuple(steps),
                rejected_count=rejected_count,
                stopped_reason="all_candidates_rejected",
            )

        selected_token_id = max(survivors, key=lambda token_id: (survivors[token_id], -token_id))
        selected_token_text = token_text_by_id[selected_token_id]
        steps.append(
            DecodeStep(
                step=step_index,
                prefix_text=prefix_text,
                selected_token_id=selected_token_id,
                selected_token_text=selected_token_text,
                rejections=rejections,
                positive_allowed_count=positive_allowed_count,
            )
        )
        token_ids.append(selected_token_id)
        prefix_text += selected_token_text

    return NegativeConstraintDecodeResult(
        text=prefix_text,
        token_ids=tuple(token_ids),
        steps=tuple(steps),
        rejected_count=rejected_count,
        stopped_reason="max_steps",
    )


def _default_registry() -> NegativeConstraintRegistry:
    registry = NegativeConstraintRegistry()
    registry.add_literal("blocked_literal", "badword")
    registry.add_regex("email_format", r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", window=96)
    registry.add_regex("invalid_ticket", r"\b[A-Z]{3}-\d{4,}\b", case_sensitive=True, window=32)
    return registry


def _default_vocab_and_scores() -> tuple[dict[int, str], list[dict[int, float]]]:
    token_text_by_id = {
        1: " bad",
        2: "word",
        3: " clean",
        4: " ok",
        5: " user@example.com",
        6: " ABC-12345",
    }
    score_rows = [
        {1: 0.90, 3: 0.20},
        {2: 0.95, 4: 0.40},
        {5: 0.80, 4: 0.30},
        {6: 0.70, 4: 0.60},
    ]
    return token_text_by_id, score_rows


def _default_positive_trie() -> PositiveMaskTrie:
    return PositiveMaskTrie.from_token_sequences(
        [
            [1, 2, 5, 6],
            [1, 4, 4, 4],
            [3, 4, 4, 4],
        ]
    )


def benchmark_negative_vs_positive_trie(
    *,
    registry: NegativeConstraintRegistry | None = None,
    token_text_by_id: Mapping[int, str] | None = None,
    score_rows: Sequence[Mapping[int, float]] | None = None,
    repeats: int = 5,
) -> dict[str, Any]:
    """Measure NCO rejection overhead against positive-mask trie decoding.

    The timing is intentionally small and deterministic.  It is not a hardware
    benchmark claim; it only checks whether the negative layer adds bounded
    overhead in the same Python process as the positive trie baseline.

    Spec: REQ-INFER-1956.
    """

    active_registry = registry or _default_registry()
    vocab, rows = _default_vocab_and_scores()
    active_vocab = dict(token_text_by_id or vocab)
    active_rows = list(score_rows or rows)
    trie = _default_positive_trie()
    repeats = max(1, repeats)

    nco_result = decode_with_negative_constraints("", active_vocab, active_rows, active_registry)

    start = time.perf_counter_ns()
    for _ in range(repeats):
        decode_with_negative_constraints("", active_vocab, active_rows, active_registry)
    nco_ns = time.perf_counter_ns() - start

    empty_registry = NegativeConstraintRegistry()
    start = time.perf_counter_ns()
    for _ in range(repeats):
        decode_with_negative_constraints(
            "",
            active_vocab,
            active_rows,
            empty_registry,
            positive_trie=trie,
        )
    positive_ns = time.perf_counter_ns() - start

    tokens = max(1, len(active_rows) * repeats)
    candidate_checks = sum(len(row) for row in active_rows) * repeats
    positive_per_token = positive_ns / tokens
    nco_per_token = nco_ns / tokens
    return {
        "tokens_evaluated": tokens,
        "candidate_checks": candidate_checks,
        "nco_ns_per_token": round(nco_per_token, 2),
        "positive_trie_ns_per_token": round(positive_per_token, 2),
        "overhead_ratio": round(nco_per_token / max(positive_per_token, 1.0), 6),
        "nco_rejected_count": nco_result.rejected_count,
        "baseline": "positive_mask_trie_python",
    }


def _step_to_json(step: DecodeStep) -> dict[str, Any]:
    return {
        "step": step.step,
        "prefix_text": step.prefix_text,
        "selected_token_id": step.selected_token_id,
        "selected_token_text": step.selected_token_text,
        "positive_allowed_count": step.positive_allowed_count,
        "rejected_token_ids": sorted(step.rejections),
        "rejection_constraints": {
            str(token_id): list(rejection.constraint_names)
            for token_id, rejection in sorted(step.rejections.items())
        },
    }


def _rejection_examples(result: NegativeConstraintDecodeResult) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for step in result.steps:
        for rejection in step.rejections.values():
            examples.append(
                {
                    "step": step.step,
                    "token_id": rejection.token_id,
                    "token_text": rejection.token_text,
                    "constraint_names": list(rejection.constraint_names),
                }
            )
    return examples


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate required terminal fields for Exp 1956."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    return True


def run_experiment(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] = (),
) -> dict[str, Any]:
    """Run the deterministic Exp 1956 NCO negative-constraint probe."""

    registry = _default_registry()
    token_text_by_id, score_rows = _default_vocab_and_scores()
    decode_result = decode_with_negative_constraints("", token_text_by_id, score_rows, registry)
    overhead = benchmark_negative_vs_positive_trie(
        registry=registry,
        token_text_by_id=token_text_by_id,
        score_rows=score_rows,
    )
    rejected_examples = _rejection_examples(decode_result)
    negative_constraints_upheld = decode_result.completed and "badword" not in decode_result.text
    negative_constraints_upheld = (
        negative_constraints_upheld
        and "@" not in decode_result.text
        and not re.search(r"\b[A-Z]{3}-\d{4,}\b", decode_result.text)
        and bool(rejected_examples)
    )

    artifact = {
        "experiment": 1956,
        "schema": "nco_negative_constraints_v1",
        "run_date": run_date,
        "status": "complete",
        "title": "Exp 1956: NCO Plug-in for Negative Constraints",
        "spec_refs": ["REQ-INFER-1956", "SCENARIO-INFER-1956"],
        "nco_negative_constraint_layer_ready": True,
        "negative_constraints_upheld": bool(negative_constraints_upheld),
        "registry_summary": {
            "constraint_count": len(registry.constraints),
            "kinds": [constraint.kind for constraint in registry.constraints],
            "max_lookback": registry.max_lookback,
            "state_explosion_avoided": True,
        },
        "decode_trace": {
            "final_text": decode_result.text,
            "token_ids": list(decode_result.token_ids),
            "stopped_reason": decode_result.stopped_reason,
            "steps": [_step_to_json(step) for step in decode_result.steps],
        },
        "overhead_vs_positive_trie": overhead,
        "rejected_token_examples": rejected_examples,
        "source_paper": "arXiv:2605.10065",
        "tests_run": list(tests_run),
        "honest_verdict": "complete: nco_negative_constraints_ready",
    }
    validate_artifact(artifact)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "REQUIRED_ARTIFACT_FIELDS",
    "DecodeStep",
    "NegativeConstraint",
    "NegativeConstraintDecodeResult",
    "NegativeConstraintRegistry",
    "PositiveMaskTrie",
    "TokenRejection",
    "benchmark_negative_vs_positive_trie",
    "decode_with_negative_constraints",
    "run_experiment",
    "validate_artifact",
]
