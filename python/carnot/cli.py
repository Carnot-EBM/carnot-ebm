"""Carnot verification CLI — verify Python functions via EBM energy constraints.

**Researcher summary:**
    Command-line interface for Carnot's code verification pipeline. Takes a
    Python file and function name, runs structural constraints and property-based
    tests, then reports energy scores and a pass/fail verdict.

**Detailed explanation for engineers:**
    This module provides the ``carnot`` CLI command, installed via
    ``pip install -e .`` as a console_scripts entry point. It wraps
    ``scripts/carnot_cli.py`` for package-level access.

Usage:
    carnot verify examples/math_funcs.py --func gcd --test "(12,8):4"

Spec: REQ-CODE-001, REQ-CODE-006
"""

from __future__ import annotations

import argparse
import ast
import os
import sys

_TYPE_MAP: dict[str, type] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "None": type(None),
    "NoneType": type(None),
}


def _find_separator_colon(s: str) -> int:
    """Find the separator colon scanning right-to-left.

    Spec: REQ-CODE-006
    """
    for i in range(len(s) - 1, 0, -1):
        if s[i] == ":":
            left = s[:i].strip()
            right = s[i + 1:].strip()
            if not left or not right:
                continue
            try:
                ast.literal_eval(left)
                ast.literal_eval(right)
                return i
            except (ValueError, SyntaxError):
                continue
    return -1


def _parse_test_pair(raw: str) -> tuple[tuple, object]:
    """Parse 'input:expected' into (args_tuple, expected_value).

    Spec: REQ-CODE-006
    """
    colon_idx = _find_separator_colon(raw)
    if colon_idx == -1:
        raise ValueError(
            f"Invalid test format: {raw!r}. Expected 'input:expected', "
            f"e.g. '(12,8):4'."
        )

    input_str = raw[:colon_idx].strip()
    expected_str = raw[colon_idx + 1:].strip()

    try:
        input_val = ast.literal_eval(input_str)
    except (ValueError, SyntaxError) as e:  # pragma: no cover — guarded by _find_separator_colon
        raise ValueError(f"Cannot parse input {input_str!r}: {e}") from e

    try:
        expected_val = ast.literal_eval(expected_str)
    except (ValueError, SyntaxError) as e:  # pragma: no cover — guarded by _find_separator_colon
        raise ValueError(f"Cannot parse expected {expected_str!r}: {e}") from e

    if not isinstance(input_val, tuple):
        input_val = (input_val,)

    return input_val, expected_val


def _resolve_type(name: str) -> type:
    """Resolve type name to Python type.

    Spec: REQ-CODE-006
    """
    t = _TYPE_MAP.get(name)
    if t is None:
        raise ValueError(
            f"Unknown type {name!r}. Supported: {', '.join(sorted(_TYPE_MAP))}"
        )
    return t


def cmd_verify(args: argparse.Namespace) -> int:
    """Execute the verify subcommand.

    Spec: REQ-CODE-001, REQ-CODE-006
    """
    from carnot.verify.property_test import format_violations_for_llm, property_test
    from carnot.verify.python_types import build_code_energy, code_to_embedding, safe_exec_function

    file_path = args.file
    if not os.path.isfile(file_path):
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 1

    with open(file_path) as f:
        code = f.read()

    func_name = args.func
    expected_type = _resolve_type(args.type)

    test_cases: list[tuple[tuple, object]] = []
    for raw in args.test or []:
        try:
            tc = _parse_test_pair(raw)
            test_cases.append(tc)
        except ValueError as e:
            print(f"Error parsing test: {e}", file=sys.stderr)
            return 1

    if not test_cases:
        print("Error: at least one --test is required.", file=sys.stderr)
        return 1

    print("=" * 60)
    print("CARNOT VERIFY")
    print(f"  File:     {file_path}")
    print(f"  Function: {func_name}")
    print(f"  Type:     {expected_type.__name__}")
    print(f"  Tests:    {len(test_cases)}")
    print("=" * 60)

    energy_fn = build_code_energy(code, func_name, test_cases, expected_type=expected_type)
    embedding = code_to_embedding(code)
    result = energy_fn.verify(embedding)

    print("\n--- Structural Tests ---")
    for input_args, expected in test_cases:
        actual, error = safe_exec_function(code, func_name, input_args)
        passed = error is None and actual == expected
        icon = "PASS" if passed else "FAIL"
        detail = ""
        if error is not None:
            detail = f" (error: {error})"
        elif not passed:
            detail = f" (got {actual!r})"
        print(f"  [{icon}] {func_name}{input_args} == {expected!r}{detail}")

    print("\n--- Energy Breakdown ---")
    for report in result.constraints:
        icon = "OK" if report.satisfied else "!!"
        print(
            f"  [{icon}] {report.name}: "
            f"energy={report.energy:.4f} "
            f"(weighted={report.weighted_energy:.4f})"
        )

    prop_result = None
    if args.properties:
        import random  # noqa: TC003

        def gen_single_int(rng: random.Random) -> tuple:
            return (rng.randint(-100, 100),)

        properties = [{
            "name": f"{func_name}_no_exception_on_int",
            "gen_args": gen_single_int,
            "check": lambda result, *args: result is not None or True,
        }]

        print(f"\n--- Property-Based Tests ({args.prop_samples} samples) ---")
        prop_result = property_test(
            code, func_name, properties,
            n_samples=args.prop_samples, seed=args.prop_seed,
        )
        print(
            f"  Ran {prop_result.n_tests} tests: "
            f"{prop_result.n_passed} passed, {prop_result.n_failed} failed "
            f"(energy={prop_result.energy:.4f}, "
            f"{prop_result.wall_clock_seconds:.3f}s)"
        )
        if prop_result.violations:
            feedback = format_violations_for_llm(prop_result)
            for line in feedback.splitlines():
                print(f"  {line}")

    has_violations = not result.verdict.verified
    if prop_result and prop_result.n_failed > 0:
        has_violations = True

    print(f"\n{'=' * 60}")
    print(f"  Total energy: {float(result.total_energy):.4f}")

    if has_violations:
        failing = result.verdict.failing
        if prop_result and prop_result.n_failed > 0:
            failing = [*failing, "property_tests"]
        print("  Verdict:      FAIL")
        print(f"  Violations:   {', '.join(failing)}")
        print(f"{'=' * 60}")
        return 1

    print("  Verdict:      PASS")
    print(f"{'=' * 60}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score activation vectors using a pre-trained EBM from HuggingFace.

    Spec: REQ-INFER-015
    """
    if args.list_models:
        from carnot.inference.ebm_loader import KNOWN_MODELS

        print("Available pre-trained EBM models:")
        print(f"{'Model ID':45s} {'Accuracy':>10s} {'Source Model':>25s} {'Thinking':>10s}")
        print("-" * 95)
        for mid, info in KNOWN_MODELS.items():
            acc, src, th = info["accuracy"], info["source_model"], info["thinking"]
            print(f"{mid:45s} {acc:>10s} {src:>25s} {th:>10s}")
        print("\nRecommended: per-token-ebm-qwen35-08b-nothink")
        return 0

    if args.activations_file is None:
        print("Error: provide --activations-file or use --list-models", file=sys.stderr)
        return 1

    from carnot.inference.ebm_loader import get_model_info, load_ebm
    from carnot.inference.ebm_rejection import score_activations_with_ebm

    print(f"Loading EBM: {args.model}...")
    ebm = load_ebm(args.model)
    info = get_model_info(args.model)

    # Load activations from safetensors
    from safetensors.numpy import load_file

    data = load_file(args.activations_file)
    activations = data["activations"]
    labels = data.get("labels")

    import numpy as np

    mean_energy = score_activations_with_ebm(ebm, activations)

    print(f"\n{'=' * 60}")
    print("CARNOT SCORE")
    print(f"  Model:       {args.model}")
    print(f"  Source LLM:  {info.get('source_model', 'unknown')}")
    print(f"  Accuracy:    {info.get('accuracy', 'unknown')}")
    print(f"  Tokens:      {len(activations)}")
    print(f"  Mean energy: {mean_energy:.4f}")

    if labels is not None:
        correct_mask = labels == 1
        wrong_mask = labels == 0
        if correct_mask.sum() > 0 and wrong_mask.sum() > 0:
            import jax.numpy as jnp

            c_energies = [float(ebm.energy(jnp.array(activations[i])))
                          for i in range(len(activations)) if labels[i] == 1]
            w_energies = [float(ebm.energy(jnp.array(activations[i])))
                          for i in range(len(activations)) if labels[i] == 0]
            n_eval = min(200, len(c_energies), len(w_energies))
            c_e = c_energies[:n_eval]
            w_e = w_energies[:n_eval]
            thresh = (np.mean(c_e) + np.mean(w_e)) / 2
            tp = sum(1 for e in w_e if e > thresh)
            tn = sum(1 for e in c_e if e <= thresh)
            acc = (tp + tn) / (len(c_e) + len(w_e))
            gap = np.mean(w_e) - np.mean(c_e)
            print(f"  Detection:   {acc:.1%} (gap={gap:.4f})")

    print(f"{'=' * 60}")
    return 0


def main() -> int:
    """CLI entry point.

    Spec: REQ-CODE-006, REQ-INFER-015
    """
    parser = argparse.ArgumentParser(
        prog="carnot",
        description="Carnot EBM verification CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- verify subcommand ---
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify a Python function against test cases and properties",
    )
    verify_parser.add_argument("file", help="Path to the Python source file")
    verify_parser.add_argument("--func", required=True, help="Function name to verify")
    verify_parser.add_argument(
        "--test", action="append", metavar="INPUT:EXPECTED",
        help="Test case in 'input:expected' format, e.g. '(12,8):4'. May be repeated.",
    )
    verify_parser.add_argument(
        "--type", default="int",
        help="Expected return type (default: int)",
    )
    verify_parser.add_argument(
        "--properties", action="store_true",
        help="Also run property-based tests",
    )
    verify_parser.add_argument(
        "--prop-samples", type=int, default=100,
        help="Random samples per property (default: 100)",
    )
    verify_parser.add_argument(
        "--prop-seed", type=int, default=42,
        help="Random seed for property tests (default: 42)",
    )

    # --- score subcommand ---
    score_parser = subparsers.add_parser(
        "score",
        help="Score activations using a pre-trained EBM from HuggingFace",
    )
    score_parser.add_argument(
        "--model", default="per-token-ebm-qwen35-08b-nothink",
        help="EBM model ID (default: per-token-ebm-qwen35-08b-nothink)",
    )
    score_parser.add_argument(
        "--activations-file", default=None,
        help="Path to safetensors file with 'activations' key",
    )
    score_parser.add_argument(
        "--list-models", action="store_true",
        help="List available pre-trained EBM models",
    )

    parsed = parser.parse_args()
    if parsed.command is None:
        parser.print_help()
        return 1
    if parsed.command == "verify":
        return cmd_verify(parsed)
    if parsed.command == "score":
        return cmd_score(parsed)
    parser.print_help()  # pragma: no cover
    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
