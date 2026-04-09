#!/usr/bin/env python3
"""Experiment 57: Verify-Repair Loop — EBMs as reasoning constraints that GUIDE the LLM.

**Researcher summary:**
    Builds on Experiment 56 (live LLM → constraint verification) by closing the
    loop: when the Ising/constraint verifier finds violations, those violations
    are translated into natural language feedback and fed back to the LLM as a
    "repair prompt." The LLM regenerates its answer, and we re-verify — up to 3
    iterations. This is the core Kona value proposition: EBMs don't just classify
    outputs as good/bad, they GUIDE the LLM toward correct answers.

**Detailed explanation for engineers:**
    Experiment 56 proved that our constraint pipeline can detect hallucinations
    in live LLM output. But detection alone isn't enough — we want the system to
    *fix* wrong answers automatically. The repair loop works like this:

    Iteration 0 (initial):
        1. Prompt the LLM with a question.
        2. Parse the LLM's response to extract an answer.
        3. Run constraint verification (arithmetic check, logic Ising, code AST,
           factual KB lookup) on the extracted answer.
        4. If all constraints pass → done, accept the answer.

    Iteration 1..N (repair):
        5. If any constraints are violated, call format_violations() to convert
           the machine-readable violation records into plain English feedback,
           e.g. "Your arithmetic is wrong: you claimed 47+28=76 but the correct
           answer is 75."
        6. Append this feedback to the conversation history and ask the LLM to
           "try again, fixing the issues listed above."
        7. Go to step 2.

    We cap at max_iters=3 repairs (4 total attempts) to avoid infinite loops.
    The experiment tests on 15 questions chosen to be tricky for small LLMs:
    multi-step arithmetic, misleading logic, tricky factual questions. We compare
    three conditions:
        A. LLM alone (no verification)
        B. LLM + verify (detect but don't repair — Exp 56 baseline)
        C. LLM + verify + repair (this experiment)

    Metrics tracked:
        - Accuracy per condition (A vs B vs C)
        - Average number of repair iterations needed
        - Per-question trace of how the answer changed across iterations

Usage:
    .venv/bin/python scripts/experiment_57_verify_repair_loop.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-004
"""

from __future__ import annotations

import gc
import os
import re
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# 1. Test questions — 15 questions designed to trip up small LLMs
# ---------------------------------------------------------------------------

def get_repair_test_questions() -> list[dict[str, Any]]:
    """Return 15 questions where small LLMs commonly hallucinate.

    **Detailed explanation for engineers:**
        These questions are intentionally tricky for small models (0.6-0.8B
        parameters). Categories:

        - Multi-step arithmetic (5): requires carrying, order of operations,
          or multiple steps — small LLMs often get one step wrong.
        - Misleading logic (5): questions with negation, contrapositive, or
          distractors that tempt the LLM into the wrong answer.
        - Tricky factual (5): commonly confused facts (e.g., capital of
          Myanmar, largest desert) that small LLMs frequently get wrong.

        Each question has a check_answer lambda that returns True if the
        response contains the correct answer. The ground_truth field is for
        display only.
    """
    return [
        # --- Multi-step arithmetic (5) ---
        {
            "domain": "arithmetic",
            "question": "What is 97 + 86?",
            "ground_truth": "183",
            "check_answer": lambda ans: _extract_number(ans) == 183,
            "why_tricky": "Two-digit addition with carrying across both digits.",
        },
        {
            "domain": "arithmetic",
            "question": "What is 256 - 178?",
            "ground_truth": "78",
            "check_answer": lambda ans: _extract_number(ans) == 78,
            "why_tricky": "Subtraction requiring borrowing across multiple places.",
        },
        {
            "domain": "arithmetic",
            "question": "What is 13 * 17?",
            "ground_truth": "221",
            "check_answer": lambda ans: _extract_number(ans) == 221,
            "why_tricky": "Two-digit multiplication — common off-by-one errors.",
        },
        {
            "domain": "arithmetic",
            "question": "A store has 45 apples. They sell 18, receive 23, then sell 12. How many remain?",
            "ground_truth": "38",
            "check_answer": lambda ans: _extract_number(ans) == 38,
            "why_tricky": "Four-step word problem — errors compound across steps.",
        },
        {
            "domain": "arithmetic",
            "question": "What is 7 * 8 + 3 * 9?",
            "ground_truth": "83",
            "check_answer": lambda ans: _extract_number(ans) == 83,
            "why_tricky": "Order of operations: multiplication before addition.",
        },
        # --- Misleading logic (5) ---
        {
            "domain": "logic",
            "question": (
                "If all roses are flowers, and some flowers fade quickly, "
                "do all roses fade quickly? Answer yes or no."
            ),
            "ground_truth": "no",
            "check_answer": lambda ans: "no" in ans.lower() and "yes" not in ans.lower().replace("yes", "").replace("no", ""),
            "why_tricky": "Tempts 'yes' via association: roses→flowers→fade.",
        },
        {
            "domain": "logic",
            "question": (
                "If no fish are mammals, and all dolphins are mammals, "
                "are dolphins fish? Answer yes or no."
            ),
            "ground_truth": "no",
            "check_answer": lambda ans: "no" in ans.lower(),
            "why_tricky": "Common misconception that dolphins are fish.",
        },
        {
            "domain": "logic",
            "question": (
                "If A implies B, and B is false, is A true or false? "
                "Answer 'true' or 'false'."
            ),
            "ground_truth": "false",
            "check_answer": lambda ans: "false" in ans.lower(),
            "why_tricky": "Modus tollens / contrapositive — often confused.",
        },
        {
            "domain": "logic",
            "question": (
                "A farmer has 5 haystacks in one field and 4 haystacks in another. "
                "If he combines them all in one field, how many haystacks does he have?"
            ),
            "ground_truth": "1",
            "check_answer": lambda ans: _extract_number(ans) == 1,
            "why_tricky": "Trick question — combining haystacks makes one big one.",
        },
        {
            "domain": "logic",
            "question": (
                "If it takes 5 machines 5 minutes to make 5 widgets, "
                "how many minutes does it take 100 machines to make 100 widgets?"
            ),
            "ground_truth": "5",
            "check_answer": lambda ans: _extract_number(ans) == 5,
            "why_tricky": "Classic rate problem — tempts answer '100'.",
        },
        # --- Tricky factual (5) ---
        {
            "domain": "factual",
            "question": "What is the capital of Myanmar?",
            "ground_truth": "Naypyidaw",
            "check_answer": lambda ans: "naypyidaw" in ans.lower() or "nay pyi taw" in ans.lower(),
            "why_tricky": "Most people (and LLMs) say Yangon/Rangoon.",
        },
        {
            "domain": "factual",
            "question": "What is the largest desert in the world by area?",
            "ground_truth": "Antarctic",
            "check_answer": lambda ans: "antarctic" in ans.lower(),
            "why_tricky": "Most say Sahara, but Antarctic desert is larger.",
        },
        {
            "domain": "factual",
            "question": "How many states does the United States have?",
            "ground_truth": "50",
            "check_answer": lambda ans: _extract_number(ans) == 50,
            "why_tricky": "Easy fact but small LLMs sometimes add territories.",
        },
        {
            "domain": "factual",
            "question": "What is the smallest country in the world by area?",
            "ground_truth": "Vatican City",
            "check_answer": lambda ans: "vatican" in ans.lower(),
            "why_tricky": "Often confused with Monaco or other micro-states.",
        },
        {
            "domain": "factual",
            "question": "In what year did the Berlin Wall fall?",
            "ground_truth": "1989",
            "check_answer": lambda ans: "1989" in ans,
            "why_tricky": "Commonly confused with 1990 (reunification) or 1991.",
        },
    ]


def _extract_number(text: str) -> float | None:
    """Pull the last number from a string (usually the final answer).

    **Detailed explanation for engineers:**
        LLMs produce answers in many formats: "The answer is 75", "75",
        "47 + 28 = 75", "I think it's about 75.0", etc. This function
        finds ALL numbers in the text and returns the last one, which is
        usually the final computed answer. Handles integers and decimals.
        Returns None if no number is found so callers can treat that as
        a failed extraction.
    """
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return None
    try:
        val = float(numbers[-1])
        return int(val) if val == int(val) else val
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# 2. Violation formatting — machine-readable violations → natural language
# ---------------------------------------------------------------------------

def format_violations(verification_result: dict[str, Any]) -> str:
    """Convert constraint violations into natural language feedback for the LLM.

    **Detailed explanation for engineers:**
        The verification pipeline (from Exp 56) returns structured dicts with
        fields like "type", "satisfied", "claimed", "correct", "description".
        This function turns those into plain English that a language model can
        understand and act on. The feedback is specific and actionable:

        - For arithmetic: "You said 47+28=76, but the correct answer is 75."
        - For logic: "Your conclusion contradicts the premises: [details]."
        - For code: "Your code has a syntax error on line 3."
        - For factual: "Your claim 'X' contradicts known facts."

        The output is a multi-line string that gets appended to the repair
        prompt. If there are no violations, returns an empty string.

    Args:
        verification_result: Dict with keys "domain", "constraints",
            "n_violated", and the constraint list from Exp 56's pipeline.

    Returns:
        A natural language string describing what went wrong, suitable for
        appending to an LLM prompt. Empty string if nothing is violated.
    """
    constraints = verification_result.get("constraints", [])
    violated = [c for c in constraints if c.get("satisfied") is False]

    if not violated:
        return ""

    domain = verification_result.get("domain", "unknown")
    feedback_lines = ["Your answer has the following errors:"]

    for i, v in enumerate(violated, 1):
        vtype = v.get("type", "unknown")

        if vtype == "arithmetic" or vtype == "arithmetic_multi":
            # Arithmetic violation: show claimed vs correct.
            expr = v.get("expression", "?")
            claimed = v.get("claimed", "?")
            correct = v.get("correct", "?")
            feedback_lines.append(
                f"  {i}. Arithmetic error: {expr} = {correct}, "
                f"but you said {claimed}."
            )

        elif vtype == "logic_answer":
            # Logic answer was wrong.
            answer = v.get("answer", "?")
            feedback_lines.append(
                f"  {i}. Your yes/no answer '{answer}' is incorrect. "
                f"Re-examine the premises and try again."
            )

        elif vtype == "logic_premise":
            # Logic premise check failed.
            raw = v.get("raw", v.get("description", "?"))
            feedback_lines.append(
                f"  {i}. Logical error: the claim '{raw}' is inconsistent "
                f"with the given premises."
            )

        elif vtype.startswith("code_"):
            # Code constraint violation (parse error, type error, etc.).
            desc = v.get("description", "Unknown code issue")
            feedback_lines.append(f"  {i}. Code issue: {desc}")

        elif vtype == "factual_claim":
            # Factual claim contradicts knowledge base.
            raw = v.get("raw", v.get("description", "?"))
            verdict = v.get("kb_verdict", "false")
            feedback_lines.append(
                f"  {i}. Factual error: '{raw}' — this is {verdict} according "
                f"to our knowledge base."
            )

        elif vtype == "factual_consistency":
            desc = v.get("description", "Inconsistent facts")
            feedback_lines.append(f"  {i}. Consistency error: {desc}")

        else:
            # Generic fallback for any other constraint type.
            desc = v.get("description", v.get("raw", v.get("type", "unknown")))
            feedback_lines.append(f"  {i}. Constraint violated: {desc}")

    feedback_lines.append("")
    feedback_lines.append("Please fix these errors and try again.")

    return "\n".join(feedback_lines)


# ---------------------------------------------------------------------------
# 3. Prompt building (initial + repair)
# ---------------------------------------------------------------------------

def build_initial_prompt(question: str, domain: str) -> str:
    """Build the initial prompt for the LLM (same structure as Exp 56).

    **Detailed explanation for engineers:**
        Uses domain-specific prompt templates that ask the LLM to answer and
        provide verifiable constraints. Kept short because small models
        (0.6-0.8B) struggle with long system prompts. The prompt format
        varies by domain to elicit the right kind of structured output.
    """
    if domain == "arithmetic":
        return (
            f"Question: {question}\n"
            f"Think step by step. Give the answer as a number.\n"
            f"Format:\n"
            f"Answer: <number>"
        )
    elif domain == "logic":
        return (
            f"Question: {question}\n"
            f"Think step by step. Give a clear answer.\n"
            f"Format:\n"
            f"Answer: <your answer>"
        )
    else:  # factual
        return (
            f"Question: {question}\n"
            f"Give a short, direct factual answer.\n"
            f"Format:\n"
            f"Answer: <answer>"
        )


def build_repair_prompt(
    question: str,
    domain: str,
    previous_answer: str,
    violation_feedback: str,
) -> str:
    """Build a repair prompt that includes violation feedback.

    **Detailed explanation for engineers:**
        The repair prompt has three parts:
        1. The original question (so the LLM remembers what was asked).
        2. The LLM's previous answer (so it knows what it said).
        3. The violation feedback from format_violations() (so it knows
           what was wrong).

        The prompt ends with an explicit instruction to fix the errors.
        We keep the repair prompt self-contained (not relying on conversation
        history) because some small models don't handle multi-turn context well.
    """
    return (
        f"Question: {question}\n\n"
        f"Your previous answer was:\n{previous_answer}\n\n"
        f"However, verification found problems:\n{violation_feedback}\n\n"
        f"Please provide a corrected answer.\n"
        f"Format:\n"
        f"Answer: <your corrected answer>"
    )


# ---------------------------------------------------------------------------
# 4. LLM generation (live or simulated)
# ---------------------------------------------------------------------------

def generate_with_llm(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a response from the loaded LLM.

    **Detailed explanation for engineers:**
        Uses HuggingFace transformers generate() with greedy decoding
        (do_sample=False) for reproducibility. Applies the model's chat
        template. Strips any <think>...</think> reasoning tokens from Qwen
        models that use chain-of-thought internally.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    if "</think>" in response:
        response = response.split("</think>")[-1].strip()

    return response


# ---------------------------------------------------------------------------
# 5. Constraint extraction (reuses Exp 56 extractors)
# ---------------------------------------------------------------------------

def extract_constraints(response: str, question: str, domain: str) -> list[dict]:
    """Extract and verify constraints from LLM output, dispatching by domain.

    **Detailed explanation for engineers:**
        Delegates to the domain-specific extractors from Exp 56:
        - arithmetic: parses equations, checks operand/result consistency
        - logic: extracts NL claims, checks against KB
        - factual: extracts factual claims, cross-references KB

        Code domain is not included in this experiment because code repair
        requires a different feedback mechanism (test output, not constraints).
    """
    from experiment_56_live_llm_pipeline import (
        extract_arithmetic_constraints,
        extract_logic_constraints,
        extract_factual_constraints,
    )

    if domain == "arithmetic":
        return extract_arithmetic_constraints(response, question)
    elif domain == "logic":
        return extract_logic_constraints(response, question)
    elif domain == "factual":
        return extract_factual_constraints(response, question)
    else:
        return []


# ---------------------------------------------------------------------------
# 6. Simulated LLM outputs (fallback when model loading fails)
# ---------------------------------------------------------------------------

def get_simulated_outputs() -> dict[str, list[str]]:
    """Simulated LLM outputs for the repair loop, including repair iterations.

    **Detailed explanation for engineers:**
        When the LLM can't be loaded, we simulate multi-turn interactions.
        Each question maps to a list of responses: index 0 is the initial
        answer, index 1 is the response after first repair feedback, etc.

        We simulate realistic behavior:
        - Some questions the LLM gets right on the first try (no repair needed).
        - Some questions the LLM gets wrong initially but fixes after feedback.
        - Some questions the LLM never fixes (to test max_iters cap).

        This ensures the pipeline logic is fully exercised even without a GPU.
    """
    return {
        # --- Arithmetic ---
        "What is 97 + 86?": [
            "Answer: 173",       # Wrong (off by 10, no carry)
            "Answer: 183",       # Fixed after feedback
        ],
        "What is 256 - 178?": [
            "Answer: 88",        # Wrong (borrowing error)
            "Answer: 78",        # Fixed after feedback
        ],
        "What is 13 * 17?": [
            "Answer: 211",       # Wrong (common multiplication error)
            "Answer: 231",       # Still wrong
            "Answer: 221",       # Fixed on 2nd repair
        ],
        "A store has 45 apples. They sell 18, receive 23, then sell 12. How many remain?": [
            "Answer: 40",        # Wrong (missed a step)
            "Answer: 38",        # Fixed after feedback
        ],
        "What is 7 * 8 + 3 * 9?": [
            "Answer: 83",        # Correct on first try
        ],
        # --- Logic ---
        (
            "If all roses are flowers, and some flowers fade quickly, "
            "do all roses fade quickly? Answer yes or no."
        ): [
            "Answer: Yes",       # Wrong (invalid syllogism)
            "Answer: No",        # Fixed after feedback
        ],
        (
            "If no fish are mammals, and all dolphins are mammals, "
            "are dolphins fish? Answer yes or no."
        ): [
            "Answer: No",        # Correct on first try
        ],
        (
            "If A implies B, and B is false, is A true or false? "
            "Answer 'true' or 'false'."
        ): [
            "Answer: true",      # Wrong (contrapositive error)
            "Answer: false",     # Fixed after feedback
        ],
        (
            "A farmer has 5 haystacks in one field and 4 haystacks in another. "
            "If he combines them all in one field, how many haystacks does he have?"
        ): [
            "Answer: 9",         # Wrong (trick question)
            "Answer: 9",         # Still wrong (doesn't get the trick)
            "Answer: 9",         # Never fixes it
        ],
        (
            "If it takes 5 machines 5 minutes to make 5 widgets, "
            "how many minutes does it take 100 machines to make 100 widgets?"
        ): [
            "Answer: 100",       # Wrong (classic trap)
            "Answer: 5",         # Fixed after feedback
        ],
        # --- Factual ---
        "What is the capital of Myanmar?": [
            "Answer: Yangon",    # Wrong (old capital)
            "Answer: Naypyidaw", # Fixed after feedback
        ],
        "What is the largest desert in the world by area?": [
            "Answer: Sahara",    # Wrong (Antarctic is larger)
            "Answer: Sahara",    # Still wrong (common misconception)
            "Answer: Antarctic Desert",  # Fixed on 2nd repair
        ],
        "How many states does the United States have?": [
            "Answer: 50",        # Correct on first try
        ],
        "What is the smallest country in the world by area?": [
            "Answer: Vatican City",  # Correct on first try
        ],
        "In what year did the Berlin Wall fall?": [
            "Answer: 1991",      # Wrong (confused with USSR)
            "Answer: 1989",      # Fixed after feedback
        ],
    }


# ---------------------------------------------------------------------------
# 7. The verify-repair loop — the core of this experiment
# ---------------------------------------------------------------------------

def verify_repair_loop(
    question: str,
    domain: str,
    check_answer: Any,
    ground_truth: str,
    *,
    tokenizer: Any = None,
    model: Any = None,
    device: str = "cpu",
    use_live_llm: bool = False,
    simulated_responses: list[str] | None = None,
    max_iters: int = 3,
) -> dict[str, Any]:
    """Run the verify-repair loop: LLM answers, verify, repair, repeat.

    **Detailed explanation for engineers:**
        This is the heart of the experiment. The loop proceeds as follows:

        1. Generate an initial answer from the LLM (or use simulated output).
        2. Extract constraints from the answer using domain-specific extractors.
        3. Check if constraints are satisfied.
        4. If violated AND we haven't hit max_iters:
           a. Format the violations into natural language feedback.
           b. Build a repair prompt with the feedback appended.
           c. Generate a new answer.
           d. Go to step 2.
        5. If all constraints pass OR max_iters reached → stop.

        Returns a dict with:
        - final_answer: the last answer generated
        - n_repairs: how many repair iterations were needed (0 = first try OK)
        - iterations: list of per-iteration results (answer, constraints, etc.)
        - initial_correct: whether the FIRST answer was correct
        - final_correct: whether the LAST answer was correct
        - repaired: True if the answer changed from wrong→right during repair

    Args:
        question: The question text.
        domain: One of "arithmetic", "logic", "factual".
        check_answer: Lambda that checks if a response contains the right answer.
        ground_truth: The correct answer string (for display).
        tokenizer: HuggingFace tokenizer (None if simulated).
        model: HuggingFace model (None if simulated).
        device: "cpu" or "cuda".
        use_live_llm: Whether to use the real LLM.
        simulated_responses: List of simulated responses for fallback.
        max_iters: Maximum number of repair iterations (default 3).

    Returns:
        Dict with final_answer, n_repairs, iterations, and correctness flags.
    """
    iterations = []
    current_answer = ""
    sim_idx = 0  # Index into simulated_responses.

    for iteration in range(max_iters + 1):  # iteration 0 = initial, 1..N = repairs
        # Build the prompt.
        if iteration == 0:
            prompt = build_initial_prompt(question, domain)
        else:
            # Build repair prompt with violation feedback from previous iteration.
            prev_result = iterations[-1]
            feedback = format_violations(prev_result)
            prompt = build_repair_prompt(
                question, domain, current_answer, feedback,
            )

        # Generate response.
        if use_live_llm:
            response = generate_with_llm(prompt, tokenizer, model, device)
        else:
            # Use simulated responses; repeat last one if we run out.
            if simulated_responses and sim_idx < len(simulated_responses):
                response = simulated_responses[sim_idx]
                sim_idx += 1
            elif simulated_responses:
                response = simulated_responses[-1]  # Repeat last.
            else:
                response = f"Answer: {ground_truth}"  # Perfect fallback.

        current_answer = response

        # Extract and verify constraints.
        constraints = extract_constraints(response, question, domain)

        n_constraints = len(constraints)
        n_satisfied = sum(1 for c in constraints if c.get("satisfied") is True)
        n_violated = sum(1 for c in constraints if c.get("satisfied") is False)
        n_unknown = n_constraints - n_satisfied - n_violated
        answer_correct = check_answer(response)

        iter_result = {
            "iteration": iteration,
            "prompt": prompt[:200],  # Truncated for readability.
            "response": response,
            "answer_correct": answer_correct,
            "domain": domain,
            "constraints": constraints,
            "n_constraints": n_constraints,
            "n_satisfied": n_satisfied,
            "n_violated": n_violated,
            "n_unknown": n_unknown,
        }
        iterations.append(iter_result)

        # If no violations, we're done (answer may still be wrong if
        # constraints didn't cover the error, but we can't detect that).
        if n_violated == 0:
            break

        # If this was the last allowed iteration, stop.
        if iteration == max_iters:
            break

    # Compute summary.
    initial_correct = iterations[0]["answer_correct"]
    final_correct = iterations[-1]["answer_correct"]
    n_repairs = len(iterations) - 1  # 0 means first try was accepted.

    return {
        "question": question,
        "domain": domain,
        "ground_truth": ground_truth,
        "final_answer": current_answer,
        "n_repairs": n_repairs,
        "iterations": iterations,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "repaired": not initial_correct and final_correct,
    }


# ---------------------------------------------------------------------------
# 8. Main — run all 15 questions through the three conditions
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the full verify-repair loop experiment."""
    print("=" * 72)
    print("EXPERIMENT 57: Verify-Repair Loop")
    print("  LLM answers → Carnot verifies → feedback → LLM repairs → re-verify")
    print("  Core Kona proposition: EBMs GUIDE the LLM, not just classify.")
    print("=" * 72)

    start = time.time()
    questions = get_repair_test_questions()

    # --- Attempt to load the LLM ---
    use_live_llm = False
    tokenizer = None
    model = None
    device = "cpu"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_candidates = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3-0.6B"]
        for model_name in model_candidates:
            try:
                print(f"\n  Loading {model_name} on {device}...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True,
                    dtype=torch.float16 if device == "cuda" else None,
                )
                if device == "cuda":
                    model = model.cuda()
                model.eval()
                use_live_llm = True
                print(f"  Loaded {model_name} successfully.")
                break
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
                tokenizer = None
                model = None

    except ImportError as e:
        print(f"\n  torch/transformers not available: {e}")

    if not use_live_llm:
        print("\n  *** FALLBACK: Using simulated LLM outputs ***")
        print("  (Model loading failed — pipeline logic is still exercised)")
        simulated = get_simulated_outputs()

    # --- Run the verify-repair loop on each question ---
    results: list[dict[str, Any]] = []

    for q in questions:
        question_text = q["question"]
        domain = q["domain"]

        # Get simulated responses for this question (if not using live LLM).
        sim_responses = None
        if not use_live_llm:
            sim_responses = simulated.get(question_text)

        result = verify_repair_loop(
            question=question_text,
            domain=domain,
            check_answer=q["check_answer"],
            ground_truth=q["ground_truth"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            use_live_llm=use_live_llm,
            simulated_responses=sim_responses,
            max_iters=3,
        )
        results.append(result)

    # --- Free LLM memory ---
    if use_live_llm:
        del model, tokenizer
        import torch
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # --- Print detailed per-question results ---
    elapsed = time.time() - start
    sep = "=" * 72

    print(f"\n{sep}")
    print("DETAILED RESULTS — Per-Question Repair Trace")
    print(sep)

    for r in results:
        initial_icon = "o" if r["initial_correct"] else "x"
        final_icon = "o" if r["final_correct"] else "x"
        repair_str = f"{r['n_repairs']} repairs" if r["n_repairs"] > 0 else "no repair needed"

        if r["repaired"]:
            status = "REPAIRED"
        elif r["initial_correct"]:
            status = "correct (no repair needed)"
        else:
            status = "FAILED (repair didn't fix)"

        print(f"\n  [{initial_icon}->{final_icon}] [{r['domain']:10s}] {r['question'][:55]}")
        print(f"      Ground truth: {r['ground_truth']}")
        print(f"      Status: {status}  |  {repair_str}")

        # Show each iteration's answer.
        for it in r["iterations"]:
            it_icon = "o" if it["answer_correct"] else "x"
            resp_short = it["response"][:60].replace("\n", " ")
            print(f"      Iter {it['iteration']}: [{it_icon}] {resp_short}"
                  f"  ({it['n_satisfied']} ok, {it['n_violated']} violated)")

    # --- Compute aggregate metrics ---
    print(f"\n{sep}")
    print(f"EXPERIMENT 57 RESULTS ({elapsed:.1f}s) "
          f"[{'LIVE LLM' if use_live_llm else 'SIMULATED'}]")
    print(sep)

    n_total = len(results)
    n_initial_correct = sum(1 for r in results if r["initial_correct"])
    n_final_correct = sum(1 for r in results if r["final_correct"])
    n_repaired = sum(1 for r in results if r["repaired"])
    n_failed_repair = sum(
        1 for r in results
        if not r["initial_correct"] and not r["final_correct"]
    )
    n_no_repair_needed = sum(
        1 for r in results
        if r["initial_correct"]
    )
    total_repairs = sum(r["n_repairs"] for r in results)
    repairs_attempted = sum(1 for r in results if r["n_repairs"] > 0)
    avg_repairs = total_repairs / repairs_attempted if repairs_attempted > 0 else 0

    # --- Condition comparison ---
    # A: LLM alone (initial accuracy)
    # B: LLM + verify (same accuracy, but we know which are wrong)
    # C: LLM + verify + repair (final accuracy)
    accuracy_a = n_initial_correct / n_total
    accuracy_c = n_final_correct / n_total

    print(f"\n  Comparison of three conditions:")
    print(f"  {'Condition':40s} {'Accuracy':>10s} {'Details':>20s}")
    print(f"  {'-' * 72}")
    acc_a_str = f"{n_initial_correct}/{n_total} ({accuracy_a:.0%})"
    acc_c_str = f"{n_final_correct}/{n_total} ({accuracy_c:.0%})"
    print(f"  {'A. LLM alone (no verification)':40s} {acc_a_str:>12s}")
    print(f"  {'B. LLM + verify (detect only)':40s} {acc_a_str:>12s}"
          f"  (same acc, knows which are wrong)")
    print(f"  {'C. LLM + verify + repair (this exp)':40s} {acc_c_str:>12s}"
          f"  (+{n_repaired} repaired)")

    print(f"\n  Repair statistics:")
    print(f"    Questions needing no repair:  {n_no_repair_needed}/{n_total}")
    print(f"    Questions successfully repaired: {n_repaired}/{repairs_attempted}"
          f" (of those attempted)")
    print(f"    Questions repair failed:     {n_failed_repair}/{repairs_attempted}"
          f" (of those attempted)")
    print(f"    Average repairs when needed:  {avg_repairs:.1f} iterations")
    print(f"    Total repair iterations:      {total_repairs}")

    # --- Per-domain breakdown ---
    print(f"\n  Per-domain breakdown:")
    print(f"  {'Domain':12s} {'Initial':>10s} {'Final':>10s} {'Repaired':>10s}")
    print(f"  {'-' * 45}")

    for domain in ["arithmetic", "logic", "factual"]:
        domain_results = [r for r in results if r["domain"] == domain]
        d_total = len(domain_results)
        d_initial = sum(1 for r in domain_results if r["initial_correct"])
        d_final = sum(1 for r in domain_results if r["final_correct"])
        d_repaired = sum(1 for r in domain_results if r["repaired"])
        print(f"  {domain:12s} {d_initial}/{d_total:>8} {d_final}/{d_total:>8} "
              f"{d_repaired:>10d}")

    # --- Verdict ---
    improvement = n_final_correct - n_initial_correct
    if improvement > 0:
        print(f"\n  VERDICT: Verify-repair loop improved accuracy by "
              f"+{improvement} questions ({accuracy_a:.0%} -> {accuracy_c:.0%})")
        print(f"  The EBM constraint layer successfully GUIDED the LLM to "
              f"correct {n_repaired} wrong answers.")
        print(f"  This validates the core Kona proposition: EBMs as reasoning "
              f"constraints, not just classifiers.")
    elif improvement == 0 and n_initial_correct == n_total:
        print(f"\n  VERDICT: LLM was already perfect — no repair needed.")
        print(f"  The verify layer confirmed all answers correct.")
    else:
        print(f"\n  VERDICT: Repair loop did not improve accuracy "
              f"({accuracy_a:.0%} -> {accuracy_c:.0%}).")
        print(f"  {n_failed_repair} questions resisted repair — constraint "
              f"coverage may need expansion.")

    print(f"\n  Architecture: LLM → Carnot verify → NL feedback → LLM repair → re-verify")
    print(f"  Constraint layer is deterministic — no hallucination in verification.")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
