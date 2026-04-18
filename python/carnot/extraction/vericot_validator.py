"""VeriCoT Step Validator — FOL formalization + Z3 consistency checking for IT model CoT.

**Problem this solves (IT model incompatibility with regex extractors):**

    Instruction-tuned (IT) models like Gemma4-E4B-it and Qwen3.5-0.8B write reasoning
    in natural prose:

        "the total is 47 plus 28, which gives 75"

    The existing ArithmeticExtractor uses the regex pattern ``(-?\\d+)\\s*([+\\-])\\s*(-?\\d+)\\s*=\\s*(-?\\d+)``
    which only matches equation-style expressions like "47 + 28 = 75".  On IT model
    outputs this regex finds ZERO matches, so the verify-repair loop has 0% net effect
    even when the model is wrong.

**VeriCoT fix (arXiv 2511.04662 — 46% relative pass rate improvement):**

    VeriCoT formalizes each Chain-of-Thought step into First-Order Logic (FOL) premises
    using an LLM call, then feeds those premises to Z3 for consistency checking.

    Pipeline:
        1.  Split CoT text into individual reasoning steps.
        2.  For each step, call an LLM (or mock rule-based extractor) to emit a
            Z3-compatible assertion string, e.g. ``"47 + 28 == 75"``.
        3.  Add all FOL premises for the step as a Z3 conjunction.
        4.  Run Z3.check():
            - SAT   → premises are consistent, no violation detected.
            - UNSAT → premises are contradictory, violation detected (arithmetic error).

    "the total is 47 plus 28, which gives 76"
        → FOL: 47 + 28 == 76
        → Z3: UNSAT (47 + 28 evaluates to 75, not 76)
        → StepVerdict(status='unsat')  ← violation detected

    "the total is 47 plus 28, which gives 75"
        → FOL: 47 + 28 == 75
        → Z3: SAT (correct)
        → StepVerdict(status='sat')   ← no violation

**Why Z3 instead of a neural judge?**

    Z3 is a sound SMT (Satisfiability Modulo Theories) solver.  It cannot hallucinate a
    "violation" for a correct constraint — if it returns UNSAT, the premises are provably
    inconsistent under integer arithmetic axioms, deterministically, every time.  A
    neural judge can be reward-hacked, produce inconsistent verdicts across runs, or
    suffer distribution shift on unusual number formats.

**Why use_mock=True in unit tests?**

    Production mode loads Qwen/Qwen3.5-0.8B via HuggingFace transformers, which requires
    a GPU or slow CPU inference (~minutes per step).  Tests that import this module in CI
    must not block on model loading.  use_mock=True substitutes a deterministic
    rule-based extractor that handles common arithmetic patterns without any model call.
    The mock extractor is accurate enough to validate the Z3 checking logic; production
    mode is required for evaluation on real IT model outputs.

Spec: REQ-EXTRACT-024, REQ-EXTRACT-025, REQ-EXTRACT-026,
      SCENARIO-EXTRACT-049, SCENARIO-EXTRACT-050, SCENARIO-EXTRACT-051
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import z3

# ---------------------------------------------------------------------------
# FOLPremise — one formalized premise extracted from a CoT step
# ---------------------------------------------------------------------------

_OP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "47 plus 28" / "47 added to 28"
    (re.compile(r"(\d+(?:\.\d+)?)\s+(?:plus|added to)\s+(\d+(?:\.\d+)?)"), "+"),
    # "47 minus 28" / "47 subtracted by 28"
    (re.compile(r"(\d+(?:\.\d+)?)\s+(?:minus|subtracted by)\s+(\d+(?:\.\d+)?)"), "-"),
    # "subtract 28 from 75" / "subtracting 28 from 75"
    (re.compile(r"subtract(?:ing)?\s+(\d+(?:\.\d+)?)\s+from\s+(\d+(?:\.\d+)?)"), "from-sub"),
    # "47 times 28" / "47 multiplied by 28"
    (re.compile(r"(\d+(?:\.\d+)?)\s+(?:times|multiplied by)\s+(\d+(?:\.\d+)?)"), "*"),
    # "47 divided by 7"
    (re.compile(r"(\d+(?:\.\d+)?)\s+divided by\s+(\d+(?:\.\d+)?)"), "/"),
]

# Trailing claim: "gives N", "gives us N", "is N", "equals N", "= N"
_RESULT_PATTERN = re.compile(
    r"(?:gives us|gives|equals|is)\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _mock_extract_expression(step_text: str) -> str | None:
    """Rule-based FOL extraction for common IT arithmetic prose.

    Handles patterns like:
      - "47 plus 28, which gives 75"       → "47 + 28 == 75"
      - "5 times 6 gives us 30"            → "5 * 6 == 30"
      - "subtracting 15 from 100 gives 85" → "100 - 15 == 85"

    Returns a Z3-compatible Python expression string, or None if no pattern matched.
    """
    for op_pat, op_sym in _OP_PATTERNS:
        op_match = op_pat.search(step_text)
        if not op_match:
            continue

        res_match = _RESULT_PATTERN.search(step_text, op_match.end())
        if not res_match:
            continue

        a_str, b_str = op_match.group(1), op_match.group(2)
        c_str = res_match.group(1)

        if op_sym == "from-sub":
            # "subtract B from A" means A - B == C
            a_str, b_str = b_str, a_str
            op_sym = "-"

        # Use integers when possible to avoid floating-point Z3 sort mismatches
        a = int(a_str) if a_str.isdigit() else float(a_str)
        b = int(b_str) if b_str.isdigit() else float(b_str)
        c = int(c_str) if c_str.isdigit() else float(c_str)

        return f"{a} {op_sym} {b} == {c}"

    return None


@dataclass
class FOLPremise:
    """One First-Order Logic premise extracted from a Chain-of-Thought step.

    Attributes
    ----------
    expression : str
        A Z3-compatible Python expression string such as ``"47 + 28 == 75"``.
        This is the formalized version of one arithmetic claim in the step.
    source_step : str
        The original natural-language step text this premise was extracted from.
        Preserved for traceability and human review.

    Why this format?
        The expression string is deliberately kept as a simple ground-truth arithmetic
        assertion (no variables, no quantifiers) so that Z3 can evaluate it in O(1)
        without search.  The string representation is also readable by humans, making
        debugging straightforward.

    Spec: REQ-EXTRACT-024, SCENARIO-EXTRACT-049, SCENARIO-EXTRACT-050
    """

    expression: str
    source_step: str

    def to_z3_assertion(self) -> z3.BoolRef | None:
        """Convert the expression to a Z3 BoolRef for solver input.

        Parses ``expression`` as ``"A OP B == C"`` where A, B, C are integer or
        float literals and OP is one of ``+``, ``-``, ``*``, ``/``.

        Returns None when the expression cannot be parsed (e.g. it came from a
        production LLM call that emitted a malformed assertion).

        Why Z3 Int rather than Python eval?
            Python ``eval("47 + 28 == 75")`` returns a Python bool (True), which is
            not a Z3 BoolRef.  Z3 IntVal objects must be used so that the solver can
            reason about the constraint symbolically and report SAT/UNSAT reliably.
        """
        # Pattern: "A OP B == C" where OP is +, -, *, /
        match = re.fullmatch(
            r"\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*==\s*(-?\d+(?:\.\d+)?)\s*",
            self.expression,
        )
        if not match:
            return None

        raw_a, op, raw_b, raw_c = match.groups()

        def _val(s: str) -> z3.ArithRef:
            return z3.IntVal(int(s)) if "." not in s else z3.RealVal(s)

        a, b, c = _val(raw_a), _val(raw_b), _val(raw_c)

        if op == "+":
            lhs = a + b
        elif op == "-":
            lhs = a - b
        elif op == "*":
            lhs = a * b
        elif op == "/":
            if raw_b == "0":
                return None
            lhs = a / b
        else:
            return None

        return lhs == c


# ---------------------------------------------------------------------------
# StepVerdict — Z3 satisfiability verdict for one CoT step
# ---------------------------------------------------------------------------


@dataclass
class StepVerdict:
    """Z3 satisfiability verdict for one Chain-of-Thought reasoning step.

    Attributes
    ----------
    step_idx : int
        Zero-based index of this step within the CoT text.
    step_text : str
        The full text of this reasoning step.
    status : str
        One of ``'sat'``, ``'unsat'``, ``'unknown'``.
        - ``'sat'``     — step is internally consistent; no violation detected.
        - ``'unsat'``   — step contains a contradiction (arithmetic error).
        - ``'unknown'`` — no FOL premises could be extracted (step may be
                          definitional, hedged, or purely qualitative).
    fol_premises : list[FOLPremise]
        The FOL premises that were extracted from this step and fed to Z3.

    How to interpret status:
        UNSAT is the actionable signal — it means Z3 proved the step is
        self-contradictory under integer/real arithmetic axioms.  The repair loop
        should focus on steps with status='unsat' first.
    """

    step_idx: int
    step_text: str
    status: str
    fol_premises: list[FOLPremise] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "step_text": self.step_text,
            "status": self.status,
            "fol_premises": [
                {"expression": p.expression, "source_step": p.source_step}
                for p in self.fol_premises
            ],
        }


# ---------------------------------------------------------------------------
# VeriCoTStepValidator — top-level API
# ---------------------------------------------------------------------------

# Split on newlines or sentence-ending punctuation (not commas — commas can appear
# mid-claim in IT prose like "47 plus 28, which gives 75").
_STEP_SPLITTER = re.compile(r"\n+|(?<=[.?!;])\s+")


def _split_steps(cot_text: str) -> list[str]:
    """Split CoT text into individual reasoning steps.

    Splits on newlines and sentence-ending punctuation.  Empty chunks are dropped.
    """
    parts = _STEP_SPLITTER.split(cot_text.strip())
    return [p.strip() for p in parts if p.strip()]


class VeriCoTStepValidator:
    """VeriCoT pipeline: extract FOL premises from IT model CoT, verify with Z3.

    **What this does:**
        For each reasoning step in a Chain-of-Thought trace, this validator:
        1. Calls an LLM (or rule-based mock) to extract Z3-compatible assertion
           strings representing the arithmetic claims in the step.
        2. Feeds those assertions to Z3 as a conjunction.
        3. Returns UNSAT when Z3 finds a contradiction — meaning the step claims
           a result that is arithmetically impossible.

    **Why the LLM prompt asks for Z3-compatible assertions:**
        Z3 assertions in the form "A OP B == C" are machine-checkable without
        any additional parsing.  The prompt asks the model to canonicalize verbal
        arithmetic into this form, making the formalization step fully automated.
        The model never needs to VERIFY the arithmetic — it only needs to TRANSCRIBE
        the claim, which is a much easier task and far less error-prone than asking
        the model to judge correctness.

    Parameters
    ----------
    extractor_llm : str
        HuggingFace model ID for the FOL extraction LLM.  Only used when
        ``use_mock=False``.
    use_mock : bool
        When True (default for tests), uses a rule-based extractor instead of
        calling the LLM.  This avoids any model-loading overhead in unit tests
        while preserving the full Z3 checking logic.

    Spec: REQ-EXTRACT-024, REQ-EXTRACT-025, REQ-EXTRACT-026
    """

    def __init__(
        self,
        extractor_llm: str = "Qwen/Qwen3.5-0.8B",
        use_mock: bool = False,
    ) -> None:
        self.extractor_llm = extractor_llm
        self.use_mock = use_mock
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # FOL extraction
    # ------------------------------------------------------------------

    def extract_fol(self, step_text: str) -> list[FOLPremise]:
        """Extract FOL premises from one CoT reasoning step.

        In mock mode, applies rule-based parsing for common arithmetic prose.
        In production mode, calls ``extractor_llm`` with a structured prompt.

        Returns an empty list when no arithmetic claim is detectable.
        """
        if self.use_mock:
            return self._mock_extract_fol(step_text)
        return self._llm_extract_fol(step_text)

    def _mock_extract_fol(self, step_text: str) -> list[FOLPremise]:
        """Rule-based FOL extraction — deterministic, no model call.

        Covers the most common IT model arithmetic patterns so tests remain
        fast and hermetic.  For production use, set use_mock=False.
        """
        expr = _mock_extract_expression(step_text)
        if expr is None:
            return []
        return [FOLPremise(expression=expr, source_step=step_text)]

    def _llm_extract_fol(self, step_text: str) -> list[FOLPremise]:
        """LLM-based FOL extraction — calls extractor_llm via transformers pipeline.

        Constructs a prompt that asks the model to emit exactly one Z3-compatible
        assertion per arithmetic claim.  Parses the output for lines starting with
        ``ASSERT:`` and builds FOLPremise objects.

        Why this prompt format?
            Asking for ``ASSERT: A OP B == C`` lines is an easily parse-able
            structured output that small models (0.8B–1B params) handle reliably.
            Free-form output risks noise that could corrupt the Z3 assertion; the
            prefix token anchors the model to produce only the formalized claim.
        """
        self._ensure_model_loaded()

        prompt = (
            "You are a formal verification assistant.  Given one step from a "
            "chain-of-thought, emit Z3-compatible arithmetic assertions.\n\n"
            "Rules:\n"
            "- For each arithmetic claim in the step, emit one line: ASSERT: A OP B == C\n"
            "- OP must be one of: +  -  *  /\n"
            "- A, B, C must be integer or decimal literals (no variables)\n"
            "- If the step contains no arithmetic claim, emit: ASSERT: none\n\n"
            f"Step: {step_text}\n\n"
            "Assertions:"
        )

        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        raw = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        premises: list[FOLPremise] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("ASSERT:"):
                continue
            expr = line[len("ASSERT:"):].strip()
            if expr.lower() == "none" or not expr:
                continue
            premises.append(FOLPremise(expression=expr, source_step=step_text))

        return premises

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the extractor LLM on first use.

        Deferred to avoid importing transformers at module import time, which
        would break CI environments without GPU or transformers installed.
        """
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            self._tokenizer = AutoTokenizer.from_pretrained(self.extractor_llm)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.extractor_llm,
                torch_dtype="auto",
                device_map="auto",
            )
        except Exception as exc:
            raise RuntimeError(
                f"VeriCoTStepValidator: failed to load extractor LLM '{self.extractor_llm}'. "
                "Set use_mock=True for tests. Original error: " + str(exc)
            ) from exc

    # ------------------------------------------------------------------
    # Z3 verification
    # ------------------------------------------------------------------

    def verify_step(self, step_text: str) -> StepVerdict:
        """Verify one CoT step by extracting FOL premises and checking with Z3.

        Returns
        -------
        StepVerdict
            ``status='unsat'`` when Z3 proves the step is arithmetically
            contradictory; ``status='sat'`` when consistent; ``status='unknown'``
            when no premises were extracted.

        Why check the conjunction?
            VeriCoT's key insight is that natural language steps often contain
            TWO arithmetic claims that must be mutually consistent: a computation
            (``47 + 28``) and a stated result (``gives 75``).  The conjunction
            ``47 + 28 == 75`` is what Z3 actually checks — this is the
            "claim not the arithmetic" pattern: we are not asking Z3 to evaluate
            arithmetic, we are asking it to determine if the stated claim is
            internally self-consistent.
        """
        premises = self.extract_fol(step_text)
        return self._check_premises(0, step_text, premises)

    def _check_premises(
        self, step_idx: int, step_text: str, premises: list[FOLPremise]
    ) -> StepVerdict:
        if not premises:
            return StepVerdict(
                step_idx=step_idx,
                step_text=step_text,
                status="unknown",
                fol_premises=[],
            )

        solver = z3.Solver()
        for p in premises:
            assertion = p.to_z3_assertion()
            if assertion is not None:
                solver.add(assertion)

        result = solver.check()
        if result == z3.sat:
            status = "sat"
        elif result == z3.unsat:
            status = "unsat"
        else:
            status = "unknown"

        return StepVerdict(
            step_idx=step_idx,
            step_text=step_text,
            status=status,
            fol_premises=premises,
        )

    # ------------------------------------------------------------------
    # CoT-level detection
    # ------------------------------------------------------------------

    def detect_violations(self, cot_text: str) -> list[StepVerdict]:
        """Detect arithmetic violations across all steps in a CoT trace.

        Splits ``cot_text`` into individual reasoning steps, verifies each,
        and returns only the steps where Z3 reported UNSAT (violations).

        This is the main entry point for the verify-repair loop — downstream
        repair logic should iterate over the returned verdicts and attempt
        to fix the identified contradictions.

        Parameters
        ----------
        cot_text : str
            Full chain-of-thought text from an IT model response.

        Returns
        -------
        list[StepVerdict]
            Only verdicts with status='unsat'.  Empty list means no violations
            were detected (either all steps are correct or none could be parsed).
        """
        steps = _split_steps(cot_text)
        violations: list[StepVerdict] = []
        for idx, step in enumerate(steps):
            premises = self.extract_fol(step)
            verdict = self._check_premises(idx, step, premises)
            if verdict.status == "unsat":
                violations.append(verdict)
        return violations
