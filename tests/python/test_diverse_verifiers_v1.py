"""Tests for Exp 1107 diverse verifier kernels.

Spec: REQ-VERIFY-1107, SCENARIO-VERIFY-1107
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.verify.ast_structure_verifier import ASTStructureVerifier
from carnot.verify.semantic_consistency_verifier import SemanticConsistencyVerifier
from carnot.verify.z3_math_verifier import Z3MathVerifier


def test_z3_math_verifier_catches_arithmetic_error():
    """REQ-VERIFY-1107: incorrect explicit arithmetic has nonzero energy."""
    verifier = Z3MathVerifier()
    assert verifier.score("47 + 28 = 76") > 0.0


def test_z3_math_verifier_passes_correct_arithmetic():
    """REQ-VERIFY-1107: correct explicit arithmetic has zero energy."""
    verifier = Z3MathVerifier()
    assert verifier.score("47 + 28 = 75") == 0.0


def test_ast_verifier_scores_invalid_syntax_higher_than_valid():
    """REQ-VERIFY-1107: invalid Python syntax scores higher than valid Python."""
    verifier = ASTStructureVerifier()
    valid = "def add(a, b):\n    return a + b\n"
    invalid = "def add(a, b):\n    return (a + b\n"
    assert verifier.score(invalid) > verifier.score(valid)


def test_semantic_consistency_verifier_catches_numeric_contradiction():
    """REQ-VERIFY-1107: repeated numeric claims with conflicting values score nonzero."""
    verifier = SemanticConsistencyVerifier()
    text = "The total is 100. Therefore, the total is 150."
    assert verifier.score(text) > 0.0


def test_semantic_consistency_verifier_passes_consistent_text():
    """REQ-VERIFY-1107: consistent repeated numeric claims score zero."""
    verifier = SemanticConsistencyVerifier()
    text = "The total is 100. Therefore, the total remains 100."
    assert verifier.score(text) == 0.0


def test_all_verifiers_return_float_in_0_1_range_on_fover_corpus():
    """REQ-VERIFY-1107: all diverse verifiers return bounded float energies."""
    corpus = json.loads((_REPO_ROOT / "data" / "fover_corpus_v4.json").read_text())
    verifiers = [Z3MathVerifier(), ASTStructureVerifier(), SemanticConsistencyVerifier()]
    for example in corpus[:20]:
        text = example["step_text"]
        for verifier in verifiers:
            score = verifier.score(text)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    for verifier in verifiers:
        assert verifier.score("") == 0.5
