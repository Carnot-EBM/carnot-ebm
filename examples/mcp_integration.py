#!/usr/bin/env python3
"""Configure and test the Carnot MCP server for use with Claude Code.

This example shows how to:
1. Print the correct MCP server configuration for your settings.json
2. Test the MCP server tools programmatically (without starting the server)
3. Show example tool calls that Claude Code would make via MCP

Use case: You want to add Carnot verification as an MCP tool in Claude
Code so the LLM can self-verify its own outputs during conversations.

Usage:
    JAX_PLATFORMS=cpu python examples/mcp_integration.py
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        from carnot.pipeline import VerifyRepairPipeline
    except ImportError:
        print("ERROR: carnot is not installed. Run: pip install -e '.[dev]'")
        return 1

    # --- Step 1: Print MCP configuration ---
    print("=" * 60)
    print("Step 1: MCP Server Configuration")
    print("=" * 60)
    print()
    print("Add this to your Claude Code settings.json")
    print("(~/.claude/settings.json or .claude/settings.json):")
    print()

    # Detect the Python executable path for the config.
    python_path = sys.executable
    config = {
        "mcpServers": {
            "carnot-verify": {
                "command": python_path,
                "args": ["-m", "carnot.mcp"],
                "env": {
                    "JAX_PLATFORMS": "cpu"
                }
            }
        }
    }
    print(json.dumps(config, indent=2))

    print()
    print("After adding this config, restart Claude Code. The following")
    print("tools will be available to the LLM:")
    print()
    print("  - verify_code: Run structural tests on Python functions")
    print("  - verify_with_properties: Property-based testing with random inputs")
    print("  - verify_llm_output: Verify LLM responses via constraint extraction")
    print("  - verify_and_repair: Verify and get natural-language repair feedback")
    print("  - list_domains: List available constraint extraction domains")
    print("  - health_check: Check server liveness")

    # --- Step 2: Test the verification tools directly ---
    print()
    print("=" * 60)
    print("Step 2: Testing verification tools (same logic as MCP server)")
    print("=" * 60)

    pipeline = VerifyRepairPipeline()

    # Simulate what Claude Code would do via verify_llm_output.
    print()
    print("Simulating verify_llm_output tool call:")
    question = "What is 25 * 4?"
    response = "25 * 4 = 100. The answer is 100."
    result = pipeline.verify(question, response)
    tool_result = {
        "verified": result.verified,
        "energy": float(result.energy),
        "n_constraints": len(result.constraints),
        "n_violations": len(result.violations),
        "constraints": [
            {
                "type": c.constraint_type,
                "description": c.description,
                "satisfied": c.metadata.get("satisfied"),
            }
            for c in result.constraints
        ],
    }
    print(f"  Tool input:  question={question!r}, response={response!r}")
    print(f"  Tool output: {json.dumps(tool_result, indent=4)}")

    # Simulate verify_and_repair with a wrong answer.
    print()
    print("Simulating verify_and_repair tool call (with violation):")
    question = "What is 15 + 28?"
    response = "15 + 28 = 42"
    result = pipeline.verify(question, response)
    violations_feedback = []
    for i, v in enumerate(result.violations, 1):
        line = f"{i}. [{v.constraint_type}] {v.description}"
        if v.constraint_type == "arithmetic" and "correct_result" in v.metadata:
            line += f" (correct answer: {v.metadata['correct_result']})"
        violations_feedback.append(line)

    repair_result = {
        "verified": result.verified,
        "energy": float(result.energy),
        "repair_feedback": "\n".join(violations_feedback) if violations_feedback else "No violations found.",
        "n_violations": len(result.violations),
    }
    print(f"  Tool input:  question={question!r}, response={response!r}")
    print(f"  Tool output: {json.dumps(repair_result, indent=4)}")

    # --- Step 3: Show the MCP tool schema ---
    print()
    print("=" * 60)
    print("Step 3: Example Claude Code conversation flow")
    print("=" * 60)
    print()
    print("  User: What is 347 + 258?")
    print("  Claude: Let me calculate... 347 + 258 = 605.")
    print("  [Claude calls verify_llm_output tool]")
    print("  Tool result: {verified: true, n_violations: 0}")
    print("  Claude: 347 + 258 = 605. (verified by Carnot)")
    print()
    print("  User: What is 99 * 7?")
    print("  Claude: 99 * 7 = 693.")
    print("  [Claude calls verify_llm_output tool]")
    print("  Tool result: {verified: false, repair_feedback: '99 * 7 = 693 (correct: ...)'}")
    print("  Claude: Let me recalculate... [uses feedback to self-correct]")

    print()
    print("Done. All examples completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
