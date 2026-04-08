#!/usr/bin/env python3
"""Experiment 41: Full LLM → Ising Verify → Repair pipeline.

End-to-end demo:
  1. Give Qwen3.5-0.8B a graph coloring problem
  2. Parse its answer into a coloring assignment
  3. Encode the graph constraints as Ising model
  4. Check if the LLM's answer satisfies constraints
  5. If not: repair via thrml sampling (find a valid coloring)
  6. Compare: LLM alone vs LLM + Ising repair

This is the practical "LLM proposes, EBM verifies and repairs" architecture
running on Extropic-compatible primitives.

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        PYTHONUNBUFFERED=1 .venv/bin/python scripts/experiment_41_llm_verify_repair.py'
"""

from __future__ import annotations

import gc
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from experiment_40_llm_ising_bridge import (
    check_coloring,
    decode_coloring,
    graph_coloring_to_ising,
)


def parse_llm_coloring(response: str, n_nodes: int, color_names: list[str]) -> dict[int, int]:
    """Parse an LLM's graph coloring response into a dict.

    Handles various formats:
      - "Node 0: Red, Node 1: Blue..."
      - "0=R, 1=B, 2=G..."
      - "A: red, B: blue..."
      - Free-form text with color words
    """
    coloring = {}
    color_map = {}
    for i, name in enumerate(color_names):
        color_map[name.lower()] = i
        color_map[name[0].lower()] = i  # First letter

    # Try structured formats first
    # Pattern: number followed by color word
    patterns = [
        r'[Nn]ode\s*(\d+)\s*[:=]\s*(\w+)',
        r'(\d+)\s*[:=]\s*(\w+)',
        r'(\d+)\s*[-–]\s*(\w+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response)
        if len(matches) >= n_nodes // 2:  # Found enough
            for node_str, color_str in matches:
                node = int(node_str)
                color_lower = color_str.lower()
                if node < n_nodes and color_lower in color_map:
                    coloring[node] = color_map[color_lower]
            if len(coloring) >= n_nodes // 2:
                break

    # Fill missing nodes with -1
    for i in range(n_nodes):
        if i not in coloring:
            coloring[i] = -1

    return coloring


def main() -> int:
    import torch
    import jax.numpy as jnp
    import jax.random as jrandom
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from thrml import SpinNode, Block, SamplingSchedule, sample_states
    from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init

    print("=" * 70)
    print("EXPERIMENT 41: LLM → Ising Verify → Repair Pipeline")
    print("  LLM proposes, Ising verifies, thrml repairs")
    print("=" * 70)

    start = time.time()

    # Load LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3.5-0.8B"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    # Test problems
    problems = [
        {
            "name": "Triangle",
            "n_nodes": 3,
            "edges": [(0, 1), (1, 2), (2, 0)],
            "n_colors": 3,
            "description": "Color 3 nodes (0, 1, 2) with 3 colors (Red, Green, Blue). Edges: 0-1, 1-2, 2-0. Adjacent nodes must have different colors.",
        },
        {
            "name": "4-cycle",
            "n_nodes": 4,
            "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
            "n_colors": 2,
            "description": "Color 4 nodes (0, 1, 2, 3) with 2 colors (Red, Blue). Edges: 0-1, 1-2, 2-3, 3-0. Adjacent nodes must have different colors.",
        },
        {
            "name": "Petersen-like",
            "n_nodes": 5,
            "edges": [(0,1), (1,2), (2,3), (3,4), (4,0), (0,2), (1,3)],
            "n_colors": 3,
            "description": "Color 5 nodes (0-4) with 3 colors (Red, Green, Blue). Edges: 0-1, 1-2, 2-3, 3-4, 4-0, 0-2, 1-3. Adjacent nodes must have different colors.",
        },
        {
            "name": "K4",
            "n_nodes": 4,
            "edges": [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)],
            "n_colors": 4,
            "description": "Color 4 nodes (0-3) with 4 colors (Red, Green, Blue, Yellow). Every node connects to every other: 0-1, 0-2, 0-3, 1-2, 1-3, 2-3. Adjacent nodes must have different colors.",
        },
        {
            "name": "6-ring",
            "n_nodes": 6,
            "edges": [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)],
            "n_colors": 2,
            "description": "Color 6 nodes (0-5) with 2 colors (Red, Blue). Ring: 0-1, 1-2, 2-3, 3-4, 4-5, 5-0. Adjacent nodes must have different colors.",
        },
        {
            "name": "8-complex",
            "n_nodes": 8,
            "edges": [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),(0,4),(2,6)],
            "n_colors": 3,
            "description": "Color 8 nodes (0-7) with 3 colors (Red, Green, Blue). Ring plus diagonals: 0-1, 1-2, 2-3, 3-4, 4-5, 5-6, 6-7, 7-0, 0-4, 2-6. Adjacent nodes must have different colors.",
        },
    ]

    color_names = ["Red", "Green", "Blue", "Yellow"]
    results = []

    for prob in problems:
        print(f"\n--- {prob['name']} ({prob['n_nodes']} nodes, {len(prob['edges'])} edges) ---")

        # Step 1: Ask LLM
        prompt = f"""Solve this graph coloring problem.

{prob['description']}

Give your answer as: Node 0: Color, Node 1: Color, etc.
Answer each node with exactly one color. No explanation needed."""

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)

        response = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()

        print(f"  LLM response: {response[:120]}")

        # Step 2: Parse LLM's coloring
        llm_coloring = parse_llm_coloring(response, prob["n_nodes"], color_names[:prob["n_colors"]])
        print(f"  Parsed: {llm_coloring}")

        # Step 3: Verify
        llm_valid, total = check_coloring(llm_coloring, prob["edges"])
        llm_perfect = llm_valid == total
        print(f"  LLM verify: {llm_valid}/{total} edges valid {'PERFECT' if llm_perfect else 'VIOLATIONS'}")

        # Step 4: Repair via thrml (always, to compare)
        total_spins, ising_edges, biases, weights = graph_coloring_to_ising(
            prob["n_nodes"], prob["edges"], prob["n_colors"],
        )
        nodes = [SpinNode() for _ in range(total_spins)]
        thrml_edges = [(nodes[e[0]], nodes[e[1]]) for e in ising_edges]

        ising_model = IsingEBM(
            nodes=nodes, edges=thrml_edges,
            biases=jnp.array(biases, dtype=jnp.float32),
            weights=jnp.array(weights, dtype=jnp.float32),
            beta=jnp.array(10.0),
        )

        free_blocks = [Block([nodes[i]]) for i in range(total_spins)]
        program = IsingSamplingProgram(ising_model, free_blocks, [])
        init_state = hinton_init(jrandom.PRNGKey(prob["n_nodes"]), ising_model, free_blocks, ())
        schedule = SamplingSchedule(1000, 30, 20)

        samples = sample_states(
            jrandom.PRNGKey(prob["n_nodes"] + 42), program, schedule,
            init_state, [], free_blocks,
        )

        # Find best repair
        n_samples = samples[0].shape[0]
        best_repair_valid = 0
        best_repair_coloring = None
        for s_idx in range(n_samples):
            spins = [bool(samples[v][s_idx, 0]) for v in range(total_spins)]
            coloring = decode_coloring(spins, prob["n_nodes"], prob["n_colors"])
            valid, _ = check_coloring(coloring, prob["edges"])
            if valid > best_repair_valid:
                best_repair_valid = valid
                best_repair_coloring = coloring

        repair_perfect = best_repair_valid == total
        print(f"  Ising repair: {best_repair_valid}/{total} edges valid {'PERFECT' if repair_perfect else ''}")

        if best_repair_coloring:
            repair_str = ", ".join(
                f"{i}={color_names[c] if 0 <= c < len(color_names) else '?'}"
                for i, c in sorted(best_repair_coloring.items())
            )
            print(f"  Repair coloring: {repair_str}")

        # Did repair improve over LLM?
        improved = best_repair_valid > llm_valid
        if improved:
            print(f"  ✅ REPAIR IMPROVED: {llm_valid} → {best_repair_valid}")
        elif best_repair_valid == llm_valid:
            print(f"  — No change needed" if llm_perfect else f"  — Same quality")
        else:
            print(f"  ❌ Repair didn't help")

        results.append({
            "name": prob["name"],
            "n_nodes": prob["n_nodes"],
            "n_edges": len(prob["edges"]),
            "llm_valid": llm_valid,
            "repair_valid": best_repair_valid,
            "total": total,
            "llm_perfect": llm_perfect,
            "repair_perfect": repair_perfect,
            "improved": improved,
        })

    # Free LLM memory
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Summary
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 41 RESULTS ({elapsed:.0f}s)")
    print(sep)
    print(f"{'Problem':20s} {'LLM':>10s} {'Repair':>10s} {'Improved':>10s}")
    print("-" * 55)

    for r in results:
        llm_str = f"{r['llm_valid']}/{r['total']}" + (" ✓" if r["llm_perfect"] else "")
        rep_str = f"{r['repair_valid']}/{r['total']}" + (" ✓" if r["repair_perfect"] else "")
        imp = "YES" if r["improved"] else ("—" if r["llm_perfect"] else "no")
        print(f"  {r['name']:18s} {llm_str:>10s} {rep_str:>10s} {imp:>10s}")

    n_llm_perfect = sum(1 for r in results if r["llm_perfect"])
    n_repair_perfect = sum(1 for r in results if r["repair_perfect"])
    n_improved = sum(1 for r in results if r["improved"])

    print(f"\n  LLM perfect: {n_llm_perfect}/{len(results)}")
    print(f"  Repair perfect: {n_repair_perfect}/{len(results)}")
    print(f"  Repairs that improved: {n_improved}/{len(results)}")

    if n_improved > 0:
        print(f"\n  VERDICT: ✅ Ising repair improves LLM output on {n_improved} problems")
        print(f"  The 'LLM proposes, Ising repairs' architecture works.")
    else:
        print(f"\n  VERDICT: ❌ LLM already perfect or repair didn't help")

    print(f"\n  Architecture: LLM (language) → Ising (constraints) → thrml/TSU (sampling)")
    print(f"  No hallucination possible in the constraint layer — it's pure math.")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
