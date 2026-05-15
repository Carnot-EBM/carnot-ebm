"""
Constraint-Aware Retrieval Module (CARM) Benchmark Suite.
Traces to: REQ-BENCH-1771
"""

import json
from typing import Any, Dict, List

def get_benchmark_cases() -> List[Dict[str, Any]]:
    """Return the 20 benchmark cases."""
    return [
        {
            "id": "case_1",
            "constraint_type": "tool-use",
            "instruction": "Fetch the weather for Seattle and print it using the `get_weather` tool.",
            "ground_truth": {"tools_required": ["get_weather"], "arguments": {"location": "Seattle"}}
        },
        {
            "id": "case_2",
            "constraint_type": "arithmetic",
            "instruction": "Calculate the sum of the first 10 prime numbers.",
            "ground_truth": {"operation": "sum", "sequence": "primes", "limit": 10, "expected_result": 129}
        },
        {
            "id": "case_3",
            "constraint_type": "logic",
            "instruction": "If all bloops are blarps, and some blarps are blips, are all bloops blips?",
            "ground_truth": {"type": "syllogism", "expected_answer": False}
        },
        {
            "id": "case_4",
            "constraint_type": "tool-use",
            "instruction": "Use the `search_arxiv` tool to find recent papers on Q-Learning.",
            "ground_truth": {"tools_required": ["search_arxiv"], "arguments": {"query": "Q-Learning", "sort_by": "recent"}}
        },
        {
            "id": "case_5",
            "constraint_type": "arithmetic",
            "instruction": "Divide 100 by 4 and add 7.",
            "ground_truth": {"operation": "add", "operands": [{"operation": "divide", "operands": [100, 4]}, 7], "expected_result": 32}
        },
        {
            "id": "case_6",
            "constraint_type": "logic",
            "instruction": "Evaluate the boolean expression: (True AND False) OR True",
            "ground_truth": {"expression": "(True AND False) OR True", "expected_result": True}
        },
        {
            "id": "case_7",
            "constraint_type": "tool-use",
            "instruction": "List all files in the directory using `list_directory`.",
            "ground_truth": {"tools_required": ["list_directory"]}
        },
        {
            "id": "case_8",
            "constraint_type": "arithmetic",
            "instruction": "What is the square root of 144?",
            "ground_truth": {"operation": "sqrt", "operand": 144, "expected_result": 12}
        },
        {
            "id": "case_9",
            "constraint_type": "logic",
            "instruction": "A requires B, B excludes C. Can A and C both be true?",
            "ground_truth": {"type": "constraint_satisfaction", "expected_answer": False}
        },
        {
            "id": "case_10",
            "constraint_type": "tool-use",
            "instruction": "Send an email to admin@example.com using the `send_email` tool.",
            "ground_truth": {"tools_required": ["send_email"], "arguments": {"to": "admin@example.com"}}
        },
        {
            "id": "case_11",
            "constraint_type": "arithmetic",
            "instruction": "Multiply 13 by 13.",
            "ground_truth": {"operation": "multiply", "operands": [13, 13], "expected_result": 169}
        },
        {
            "id": "case_12",
            "constraint_type": "logic",
            "instruction": "Solve: x AND y is True. Is x True?",
            "ground_truth": {"type": "boolean_algebra", "expected_answer": True}
        },
        {
            "id": "case_13",
            "constraint_type": "tool-use",
            "instruction": "Format the Python code in file.py using the `ruff_format` tool.",
            "ground_truth": {"tools_required": ["ruff_format"], "arguments": {"file": "file.py"}}
        },
        {
            "id": "case_14",
            "constraint_type": "arithmetic",
            "instruction": "Compute 2 to the power of 10.",
            "ground_truth": {"operation": "power", "base": 2, "exponent": 10, "expected_result": 1024}
        },
        {
            "id": "case_15",
            "constraint_type": "logic",
            "instruction": "XOR of True and True is what?",
            "ground_truth": {"operation": "XOR", "operands": [True, True], "expected_result": False}
        },
        {
            "id": "case_16",
            "constraint_type": "tool-use",
            "instruction": "Read the system metrics using `get_metrics`.",
            "ground_truth": {"tools_required": ["get_metrics"]}
        },
        {
            "id": "case_17",
            "constraint_type": "arithmetic",
            "instruction": "Find the factorial of 5.",
            "ground_truth": {"operation": "factorial", "operand": 5, "expected_result": 120}
        },
        {
            "id": "case_18",
            "constraint_type": "logic",
            "instruction": "If it is raining, the street is wet. The street is wet. Is it raining?",
            "ground_truth": {"type": "fallacy", "fallacy_type": "affirming_consequent", "expected_answer": "unknown"}
        },
        {
            "id": "case_19",
            "constraint_type": "tool-use",
            "instruction": "Write 'Hello World' to hello.txt using `write_file`.",
            "ground_truth": {"tools_required": ["write_file"], "arguments": {"path": "hello.txt", "content": "Hello World"}}
        },
        {
            "id": "case_20",
            "constraint_type": "arithmetic",
            "instruction": "Subtract 45 from 100.",
            "ground_truth": {"operation": "subtract", "operands": [100, 45], "expected_result": 55}
        }
    ]

def generate_carm_benchmark(primary_path: str, backup_path: str) -> None:
    """Generate the CARM benchmark suite JSON files."""
    cases = get_benchmark_cases()
    payload = {
        "schema": "carnot.carm.benchmark.v1",
        "num_cases": len(cases),
        "cases": cases
    }
    
    with open(primary_path, "w") as f:
        json.dump(payload, f, indent=2)
        
    with open(backup_path, "w") as f:
        json.dump(payload, f, indent=2)
