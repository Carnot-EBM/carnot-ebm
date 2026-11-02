"""Diffusion-of-Thought verifier-guided text refinement.

Diffusion of Thought (DoT) treats a response as a small discrete state and
uses Carnot's verifier energy as the denoising guide.  Instead of adding
Gaussian noise to a full generated answer, DoT masks the tokens whose removal
most reduces verifier energy, proposes local replacements, and keeps the
replacement with the lowest composite energy.

Spec: REQ-INFER-017, SCENARIO-INFER-017-001
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

MASK_TOKEN = "[MASK]"
MAX_TIMESTEPS = 125

_TOKEN_AFFIX_RE = re.compile(r"^(\W*)(.*?)(\W*)$")
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_EQUATION_RE = re.compile(
    r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*([+\-*/])\s*"
    r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*=\s*"
    r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"
)

_OPERATIONS: dict[str, Callable[[float, float], float]] = {
    "+": lambda left, right: left + right,
    "-": lambda left, right: left - right,
    "*": lambda left, right: left * right,
    "/": lambda left, right: left / right,
}

_WORD_SUBSTITUTIONS: dict[str, tuple[str, ...]] = {
    "false": ("true", "correct", "valid"),
    "incorrect": ("correct", "valid", "consistent"),
    "invalid": ("valid", "correct", "consistent"),
    "not": ("therefore", "so", "thus"),
    "never": ("always", "usually", "therefore"),
    "less": ("more", "equal", "greater"),
    "more": ("less", "equal", "greater"),
}
_GENERIC_CANDIDATES = ("therefore", "because", "correct", "valid", "consistent")


class DiffusionOfThought:
    """Iteratively lower verifier energy by repairing high-energy tokens.

    Args:
        k5_ensemble: Existing verifier ensemble or energy object.  The class
            accepts Carnot's k=5 `verify(question, response)` style, a direct
            `energy(response, context)` method, a single-text `score(text)`
            method, or a callable `(response, context) -> energy`.
        n_candidates_per_step: Number of deterministic fallback replacements
            considered for each masked token.  Values are clamped to the
            requested 3-5 candidate band.
    """

    def __init__(self, k5_ensemble: Any, n_candidates_per_step: int = 5) -> None:
        self.k5_ensemble = k5_ensemble
        self.n_candidates_per_step = max(3, min(5, int(n_candidates_per_step)))

    def composite_energy(self, response: str, context: str = "") -> float:
        """Return the scalar verifier energy for one context/response pair."""

        ensemble = self.k5_ensemble
        if hasattr(ensemble, "verify"):
            result = ensemble.verify(context, response)
            scores = getattr(result, "per_verifier_scores", None)
            if scores:
                return float(sum(float(value) for value in scores.values()) / len(scores))
            energy = getattr(result, "energy", None)
            if energy is not None:  # pragma: no cover - compatibility fallback.
                return float(energy)
            verified = bool(getattr(result, "verified", False))  # pragma: no cover
            return 0.0 if verified else 1.0  # pragma: no cover

        if hasattr(ensemble, "energy"):
            return float(ensemble.energy(response, context))

        if hasattr(ensemble, "score"):
            text = f"{context}\n{response}" if context.strip() else response
            return float(ensemble.score(text))

        return float(ensemble(response, context))

    def compute_token_energies(self, response: str, context: str) -> list[float]:
        """Estimate each token's violation energy by masking it.

        A positive value means the response energy drops when that token is
        replaced by `[MASK]`, so the token is a useful repair candidate.
        """

        tokens = response.split()
        base_energy = self.composite_energy(response, context)
        token_energies: list[float] = []
        for position in range(len(tokens)):
            masked_tokens = list(tokens)
            masked_tokens[position] = MASK_TOKEN
            masked_response = " ".join(masked_tokens)
            masked_energy = self.composite_energy(masked_response, context)
            token_energies.append(max(0.0, base_energy - masked_energy))
        return token_energies

    def propose_correction(self, token: str, context: str, position: int) -> list[str]:
        """Generate deterministic replacement candidates for one token.

        This is the no-LLM fallback path.  Numeric tokens receive nearby
        values and arithmetic targets found in the context; text tokens use a
        small substitution vocabulary plus generic reasoning words.
        """

        prefix, core, suffix = _split_affixes(token)
        numeric_value = _parse_number(core)
        raw_candidates: list[str] = []

        if numeric_value is not None:
            raw_candidates.extend(_arithmetic_targets(context))
            raw_candidates.extend(
                _format_number(value)
                for value in (
                    numeric_value - 2,
                    numeric_value - 1,
                    numeric_value + 1,
                    numeric_value + 2,
                    0.0,
                    1.0,
                )
            )

        raw_candidates.extend(_WORD_SUBSTITUTIONS.get(core.lower(), ()))
        raw_candidates.extend(_GENERIC_CANDIDATES)

        return _format_candidate_tokens(
            raw_candidates,
            original_core=core,
            prefix=prefix,
            suffix=suffix,
            limit=self.n_candidates_per_step,
        )

    def refine(self, response: str, context: str, n_steps: int) -> tuple[str, list[float]]:
        """Run DoT refinement for `n_steps` and return text plus energy trace."""

        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")

        tokens = response.split()
        current_energy = self.composite_energy(response, context)
        energy_trace = [current_energy]

        for _step in range(n_steps):
            if tokens:
                token_energies = self.compute_token_energies(" ".join(tokens), context)
                for position in _top_mask_positions(token_energies, n_steps, len(tokens)):
                    best_token = tokens[position]
                    best_energy = current_energy
                    proposal_context = f"{context}\n{' '.join(tokens)}".strip()
                    for candidate in self.propose_correction(
                        tokens[position],
                        proposal_context,
                        position,
                    ):
                        candidate_tokens = list(tokens)
                        candidate_tokens[position] = candidate
                        candidate_response = " ".join(candidate_tokens)
                        candidate_energy = self.composite_energy(candidate_response, context)
                        if candidate_energy < best_energy:
                            best_token = candidate
                            best_energy = candidate_energy
                    tokens[position] = best_token
                    current_energy = best_energy
            energy_trace.append(current_energy)

        return " ".join(tokens), energy_trace


def _top_mask_positions(token_energies: list[float], n_steps: int, n_tokens: int) -> list[int]:
    n_mask = max(1, min(n_tokens, math.ceil(1.5 * max(1, n_steps) / MAX_TIMESTEPS)))
    return sorted(range(n_tokens), key=lambda idx: token_energies[idx], reverse=True)[:n_mask]


def _split_affixes(token: str) -> tuple[str, str, str]:
    match = _TOKEN_AFFIX_RE.match(token)
    if match is None:  # pragma: no cover - regex is total for strings.
        return "", token, ""
    return match.group(1), match.group(2), match.group(3)


def _parse_number(text: str) -> float | None:
    candidate = text.replace(",", "")
    if _NUMBER_RE.fullmatch(candidate):
        return float(candidate)
    return None


def _arithmetic_targets(text: str) -> list[str]:
    normalized = text.replace("\\times", "*").replace("×", "*").replace("÷", "/").replace("−", "-")
    targets: list[str] = []
    for match in _EQUATION_RE.finditer(normalized):
        left = _parse_number(match.group(1))
        right = _parse_number(match.group(3))
        claimed = _parse_number(match.group(4))
        if left is None or right is None or claimed is None:  # pragma: no cover
            continue
        try:
            expected = _OPERATIONS[match.group(2)](left, right)
        except ZeroDivisionError:  # pragma: no cover - invalid equation candidate.
            continue
        if abs(expected - claimed) > 1e-9:
            targets.append(_format_number(expected))
    return targets


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _format_candidate_tokens(
    raw_candidates: list[str],
    *,
    original_core: str,
    prefix: str,
    suffix: str,
    limit: int,
) -> list[str]:
    formatted: list[str] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        if not raw:  # pragma: no cover - candidate tables do not emit empties.
            continue
        core = _match_case(str(raw), original_core)
        if core == original_core:
            continue
        candidate = f"{prefix}{core}{suffix}"
        if candidate not in seen:
            seen.add(candidate)
            formatted.append(candidate)
        if len(formatted) >= limit:
            break
    return formatted


def _match_case(candidate: str, original: str) -> str:
    if original.isupper():
        return candidate.upper()
    if original.istitle():
        return candidate.capitalize()
    return candidate


__all__ = ["DiffusionOfThought", "MASK_TOKEN", "MAX_TIMESTEPS"]
