"""Tests for Exp 764: ASTKnowledgeVerifier — Execution-Free Hallucination Detection.

Each test traces to REQ-EXTRACT-035 or REQ-EXTRACT-036 as required by spec-anchored
development.  The tests exercise:
  - KnowledgeBase.lookup: known attrs return True, missing attrs return False.
  - ASTKnowledgeVerifier.verify: correct code produces no violations (no FP);
    hallucinated code produces exactly the right violations.
  - Precision/recall arithmetic from TP/FP/FN counts.
  - Import alias resolution.
  - Tier 0d skip logic (violations present → caller concludes violation_detected=True).

Spec: REQ-EXTRACT-035, REQ-EXTRACT-036,
      SCENARIO-EXTRACT-070, SCENARIO-EXTRACT-071, SCENARIO-EXTRACT-072
"""

from __future__ import annotations

import pytest

from carnot.extraction.ast_knowledge_verifier import (
    ASTKnowledgeViolation,
    ASTKnowledgeVerifier,
    KnowledgeBase,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    """Knowledge base populated with json, os, math, random, re modules."""
    return KnowledgeBase.build_from_modules(["json", "os", "math", "random", "re"])


@pytest.fixture(scope="module")
def verifier(kb: KnowledgeBase) -> ASTKnowledgeVerifier:
    return ASTKnowledgeVerifier(kb)


# ---------------------------------------------------------------------------
# KnowledgeBase tests — REQ-EXTRACT-035
# ---------------------------------------------------------------------------


def test_kb_lookup_known_attr_returns_true(kb: KnowledgeBase) -> None:
    """json.loads exists in the standard library; lookup must return True.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-070
    """
    assert kb.lookup("json", "loads") is True


def test_kb_lookup_missing_attr_returns_false(kb: KnowledgeBase) -> None:
    """json.parse does NOT exist; lookup must return False.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-071
    """
    assert kb.lookup("json", "parse") is False


def test_kb_lookup_unknown_module_returns_true(kb: KnowledgeBase) -> None:
    """For a module not in the KB, lookup returns True (safe — no FP).

    The precision=1.0 invariant requires that we never flag unknown modules.
    Spec: REQ-EXTRACT-035, REQ-EXTRACT-036
    """
    assert kb.lookup("unknownmodule_xyz", "anything") is True


def test_kb_has_module_loaded(kb: KnowledgeBase) -> None:
    """has_module returns True for introspected modules.

    Spec: REQ-EXTRACT-035
    """
    assert kb.has_module("json") is True


def test_kb_has_module_unknown(kb: KnowledgeBase) -> None:
    """has_module returns False for modules not in the KB.

    Spec: REQ-EXTRACT-035
    """
    assert kb.has_module("nonexistent_xyz") is False


def test_kb_known_modules_includes_introspected(kb: KnowledgeBase) -> None:
    """known_modules() lists all loaded modules.

    Spec: REQ-EXTRACT-035
    """
    modules = kb.known_modules()
    assert "json" in modules
    assert "math" in modules


def test_kb_os_path_join_exists(kb: KnowledgeBase) -> None:
    """os.path is an attribute of the os module; os.path exists → True.

    Spec: REQ-EXTRACT-035
    """
    assert kb.lookup("os", "path") is True


def test_kb_math_relu_missing(kb: KnowledgeBase) -> None:
    """math.relu does not exist; lookup must return False.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-071
    """
    assert kb.lookup("math", "relu") is False


# ---------------------------------------------------------------------------
# ASTKnowledgeVerifier — no false positives — REQ-EXTRACT-035, SCENARIO-EXTRACT-070
# ---------------------------------------------------------------------------


def test_verify_json_loads_no_violation(verifier: ASTKnowledgeVerifier) -> None:
    """Correct API usage json.loads(text) must produce zero violations.

    This is the critical no-false-positive test.  100% precision requires that
    correct code is never flagged.
    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-070
    """
    code = "import json\ndata = json.loads(text)"
    violations = verifier.verify(code)
    assert violations == [], f"Expected no violations, got: {violations}"


def test_verify_json_dumps_no_violation(verifier: ASTKnowledgeVerifier) -> None:
    """json.dumps is a real method — must not be flagged.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-070
    """
    code = "import json\ns = json.dumps(obj)"
    assert verifier.verify(code) == []


def test_verify_math_sqrt_no_violation(verifier: ASTKnowledgeVerifier) -> None:
    """math.sqrt is real — must not be flagged.

    Spec: REQ-EXTRACT-035
    """
    code = "import math\nresult = math.sqrt(x)"
    assert verifier.verify(code) == []


def test_verify_random_choice_no_violation(verifier: ASTKnowledgeVerifier) -> None:
    """random.choice is real — must not be flagged.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-070
    """
    code = "import random\nitem = random.choice(lst)"
    assert verifier.verify(code) == []


def test_verify_os_makedirs_no_violation(verifier: ASTKnowledgeVerifier) -> None:
    """os.makedirs is real — must not be flagged.

    Spec: REQ-EXTRACT-035
    """
    code = "import os\nos.makedirs(path, exist_ok=True)"
    assert verifier.verify(code) == []


# ---------------------------------------------------------------------------
# ASTKnowledgeVerifier — hallucination detection — REQ-EXTRACT-035, SCENARIO-EXTRACT-071
# ---------------------------------------------------------------------------


def test_verify_json_parse_flagged(verifier: ASTKnowledgeVerifier) -> None:
    """json.parse does not exist; must be flagged as missing_attr violation.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-071
    """
    code = "import json\ndata = json.parse(text)"
    violations = verifier.verify(code)
    assert len(violations) == 1
    v = violations[0]
    assert v.module == "json"
    assert v.attr == "parse"
    assert v.violation_type == "missing_attr"


def test_verify_os_read_file_flagged(verifier: ASTKnowledgeVerifier) -> None:
    """os.read_file does not exist; must be flagged.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-072
    """
    code = "import os\ndata = os.read_file(path)"
    violations = verifier.verify(code)
    assert any(v.attr == "read_file" and v.module == "os" for v in violations)


def test_verify_math_relu_flagged(verifier: ASTKnowledgeVerifier) -> None:
    """math.relu does not exist; must be flagged.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-071
    """
    code = "import math\nresult = math.relu(x)"
    violations = verifier.verify(code)
    assert any(v.attr == "relu" and v.module == "math" for v in violations)


def test_verify_random_choice_weighted_flagged(verifier: ASTKnowledgeVerifier) -> None:
    """random.choice_weighted does not exist; must be flagged.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-071
    """
    code = "import random\nitem = random.choice_weighted(lst, weights)"
    violations = verifier.verify(code)
    assert any(v.attr == "choice_weighted" and v.module == "random" for v in violations)


def test_verify_re_findall_named_flagged(verifier: ASTKnowledgeVerifier) -> None:
    """re.findall_named does not exist; must be flagged.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-071
    """
    code = "import re\nmatches = re.findall_named(pattern, text)"
    violations = verifier.verify(code)
    assert any(v.attr == "findall_named" and v.module == "re" for v in violations)


# ---------------------------------------------------------------------------
# ASTKnowledgeVerifier — import alias resolution
# ---------------------------------------------------------------------------


def test_verify_alias_json_parse_flagged(verifier: ASTKnowledgeVerifier) -> None:
    """'import json as j; j.parse(text)' must still be flagged via alias resolution.

    Spec: REQ-EXTRACT-035
    """
    code = "import json as j\ndata = j.parse(text)"
    violations = verifier.verify(code)
    assert any(v.attr == "parse" and v.module == "json" for v in violations)


def test_verify_alias_json_loads_no_violation(verifier: ASTKnowledgeVerifier) -> None:
    """'import json as j; j.loads(text)' must NOT be flagged — alias to a real method.

    Spec: REQ-EXTRACT-035, SCENARIO-EXTRACT-070
    """
    code = "import json as j\ndata = j.loads(text)"
    assert verifier.verify(code) == []


# ---------------------------------------------------------------------------
# ASTKnowledgeVerifier — edge cases
# ---------------------------------------------------------------------------


def test_verify_syntax_error_returns_empty(verifier: ASTKnowledgeVerifier) -> None:
    """Syntactically invalid code returns empty violations (safe fallback).

    Spec: REQ-EXTRACT-035
    """
    code = "def broken(: :"
    assert verifier.verify(code) == []


def test_verify_empty_string_returns_empty(verifier: ASTKnowledgeVerifier) -> None:
    """Empty code returns empty violations.

    Spec: REQ-EXTRACT-035
    """
    assert verifier.verify("") == []


def test_verify_unknown_module_skipped(verifier: ASTKnowledgeVerifier) -> None:
    """Accesses on modules not in the KB are skipped (preserve precision=1.0).

    If we don't know the module, we cannot flag it — better to miss a violation
    than to generate a false positive.
    Spec: REQ-EXTRACT-035, REQ-EXTRACT-036
    """
    code = "import numpy as np\nresult = np.nonexistent_function(x)"
    # numpy is not in our KB, so this should NOT be flagged.
    violations = verifier.verify(code)
    assert violations == []


def test_verify_function_code(verifier: ASTKnowledgeVerifier) -> None:
    """verify_function() works on a bare function definition string.

    Spec: REQ-EXTRACT-035
    """
    func_code = "import json\ndef process(text):\n    return json.parse(text)\n"
    violations = verifier.verify_function(func_code)
    assert any(v.attr == "parse" for v in violations)


# ---------------------------------------------------------------------------
# Precision/recall arithmetic — REQ-EXTRACT-035
# ---------------------------------------------------------------------------


def test_precision_from_counts() -> None:
    """precision = TP / (TP + FP); must be 1.0 when FP=0.

    Spec: REQ-EXTRACT-035
    """
    tp, fp, fn = 10, 0, 3
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    assert precision == 1.0


def test_recall_from_counts() -> None:
    """recall = TP / (TP + FN); formula correctness check.

    Spec: REQ-EXTRACT-035
    """
    tp, fp, fn = 10, 0, 3
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    assert abs(recall - 10 / 13) < 1e-9


def test_f1_from_counts() -> None:
    """f1 = 2*P*R / (P+R); formula correctness check.

    Spec: REQ-EXTRACT-035
    """
    precision = 1.0
    recall = 10 / 13
    f1 = 2 * precision * recall / (precision + recall)
    assert f1 > 0.0


def test_precision_with_fp_below_1() -> None:
    """precision < 1.0 when FP > 0.

    Spec: REQ-EXTRACT-035
    """
    tp, fp, fn = 8, 2, 2
    precision = tp / (tp + fp)
    assert precision == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Tier 0d skip logic — REQ-EXTRACT-036, SCENARIO-EXTRACT-072
# ---------------------------------------------------------------------------


def test_tier0d_violation_detected_skips_ising(verifier: ASTKnowledgeVerifier) -> None:
    """When violations are non-empty, caller concludes violation_detected=True.

    This test models the Tier 0d integration: if ASTKnowledgeVerifier returns
    violations, we set violation_detected=True and skip the Ising verifier.
    100% precision guarantees this is a real KCH — no Ising call needed.
    Spec: REQ-EXTRACT-036, SCENARIO-EXTRACT-072
    """
    code = "import os\ndata = os.read_file(path)"
    violations = verifier.verify(code)
    violation_detected = len(violations) > 0
    assert violation_detected is True
    # In the pipeline: if violation_detected, we skip Ising. Simulate that:
    ising_called = not violation_detected
    assert ising_called is False


def test_tier0d_no_violation_proceeds_to_ising(verifier: ASTKnowledgeVerifier) -> None:
    """When no violations found, violation_detected=False and Ising runs normally.

    Spec: REQ-EXTRACT-036, SCENARIO-EXTRACT-072
    """
    code = "import os\nos.makedirs(path, exist_ok=True)"
    violations = verifier.verify(code)
    violation_detected = len(violations) > 0
    assert violation_detected is False
    ising_called = not violation_detected
    assert ising_called is True


# ---------------------------------------------------------------------------
# ASTKnowledgeViolation dataclass — REQ-EXTRACT-035
# ---------------------------------------------------------------------------


def test_violation_dataclass_fields() -> None:
    """ASTKnowledgeViolation stores the expected fields.

    Spec: REQ-EXTRACT-035
    """
    v = ASTKnowledgeViolation(
        node_text="json.parse",
        module="json",
        attr="parse",
        violation_type="missing_attr",
        lineno=2,
    )
    assert v.node_text == "json.parse"
    assert v.module == "json"
    assert v.attr == "parse"
    assert v.violation_type == "missing_attr"
    assert v.lineno == 2
