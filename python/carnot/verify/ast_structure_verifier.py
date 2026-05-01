"""AST and bracket-structure verifier for diverse verifier ensembles.

Spec: REQ-VERIFY-1107, SCENARIO-VERIFY-1107
"""

from __future__ import annotations

import ast
import re


class ASTStructureVerifier:
    """AST-based structural verifier for code and structured text.

    Python-looking text is checked with ``ast.parse`` plus bracket balance.
    General prose is checked with cheap structural heuristics such as balanced
    brackets and sentence boundaries. The kernel is syntactic structure, not
    token-level statistical fluency.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def score(self, text: str) -> float:
        """Return structural violation energy in [0, 1]."""
        if not text or not text.strip():
            return 0.5

        source = _extract_code_block(text).strip()
        if not source or not any(ch.isalnum() for ch in source):
            return 0.5

        bracket_energy = _bracket_violation_energy(source)
        if _looks_like_python(source):
            syntax_energy = 0.0
            try:
                ast.parse(source)
            except SyntaxError:
                syntax_energy = 0.75
            except Exception:
                return 0.5
            return _clamp(max(bracket_energy, syntax_energy))

        return _clamp(max(bracket_energy, _prose_structure_energy(source)))


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    return text


def _looks_like_python(text: str) -> bool:
    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not stripped_lines:
        return False
    python_line = re.compile(
        r"^(def|class|async\s+def|import|from\s+\w|return|if|elif|else:|for|while|try:|"
        r"except|finally:|with|@\w)"
    )
    if any(python_line.match(line) for line in stripped_lines):
        return True
    if any(
        line.endswith(":") and re.match(r"^(if|for|while|def|class|with|try)\b", line)
        for line in stripped_lines
    ):
        return True
    assignment_like = sum(1 for line in stripped_lines if re.match(r"^[A-Za-z_]\w*\s*=", line))
    indented_lines = sum(1 for line in text.splitlines() if line.startswith(("    ", "\t")))
    return assignment_like >= 2 or indented_lines >= 2


def _bracket_violation_energy(text: str) -> float:
    pairs = {")": "(", "]": "[", "}": "{"}
    openers = set(pairs.values())
    stack: list[str] = []
    errors = 0
    for ch in text:
        if ch in openers:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                errors += 1
            else:
                stack.pop()
    errors += len(stack)
    if errors == 0:
        return 0.0
    return min(1.0, 0.25 + 0.25 * errors)


def _prose_structure_energy(text: str) -> float:
    violations = 0.0
    checks = 3.0
    stripped = text.strip()

    if len(stripped) > 120 and not re.search(r"[.!?]\s*$", stripped):
        violations += 1.0
    if re.search(r"[!?.,;:]{4,}", stripped):
        violations += 1.0

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", stripped) if s.strip()]
    if not sentences:
        return 0.5

    alpha_sentences = [s for s in sentences if any(ch.isalpha() for ch in s)]
    if alpha_sentences:
        lowercase_starts = sum(1 for s in alpha_sentences if s[0].islower())
        if lowercase_starts / len(alpha_sentences) > 0.5:
            violations += 1.0
    else:
        checks = 2.0

    return min(0.5, violations / checks)


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
