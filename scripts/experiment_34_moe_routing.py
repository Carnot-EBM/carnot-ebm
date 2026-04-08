#!/usr/bin/env python3
"""Experiment 34: MoE routing patterns as hallucination signal.

Exp 32 showed Qwen3.5-35B has genuinely specialized experts (overlap 0.008).
Do correct vs hallucinated tokens activate different routing patterns?

For each token during generation, the MoE router produces:
  - Top-k expert selection (which experts are active)
  - Router logits (how confident the routing decision is)
  - Router entropy (how spread out the routing distribution is)

Hypothesis: hallucinated tokens may show higher routing entropy (the model
is "confused" about which expert should handle this token) or activate
different expert subsets than correct tokens.

This requires NO EBM training — just observe the router during inference.

Uses Qwen3.5-35B-A3B (256 experts, already downloaded).

Usage:
    sg render -c 'TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        PYTHONUNBUFFERED=1 .venv/bin/python scripts/experiment_34_moe_routing.py'
"""

from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from collect_truthfulqa_activations import check_truthfulqa_answer


def collect_routing_data(n_questions: int = 100) -> dict:
    """Generate on TruthfulQA and capture MoE routing decisions per token.

    Returns dict with routing data for correct and wrong answers.
    """
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3.5-35B-A3B"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype="auto" if device == "cuda" else None,
    )
    if device == "cuda":
        model = model.cuda()
    model.eval()

    ds = load_dataset("truthful_qa", "generation")
    questions = list(ds["validation"])[:n_questions]

    # Hook into MoE router to capture routing decisions
    routing_data = {"correct": [], "wrong": []}
    captured_router_logits = []

    def router_hook(module, input, output):
        """Capture router logits from MoE layer."""
        # output is typically (hidden_states, router_logits) or similar
        # The exact format depends on the model implementation
        if isinstance(output, tuple) and len(output) >= 2:
            router_out = output[1]
            if router_out is not None and hasattr(router_out, 'shape'):
                captured_router_logits.append(router_out.detach().cpu())

    # Find MoE layers and attach hooks
    hooks = []
    moe_layers = []
    for name, module in model.named_modules():
        # Qwen3.5 MoE uses 'mlp' with experts
        if "moe" in name.lower() or (hasattr(module, 'experts') and hasattr(module, 'gate')):
            moe_layers.append(name)
            # Hook the gate/router specifically
            if hasattr(module, 'gate'):
                hooks.append(module.gate.register_forward_hook(
                    lambda m, i, o, name=name: captured_router_logits.append(
                        ("gate", name, o.detach().cpu() if isinstance(o, torch.Tensor) else o)
                    )
                ))

    if not hooks:
        # Try alternative: hook all modules and look for router-like outputs
        print("  No standard MoE gate found, trying alternative hook strategy...")
        for name, module in model.named_modules():
            if "gate" in name.lower() and "proj" not in name.lower():
                hooks.append(module.register_forward_hook(
                    lambda m, i, o, name=name: captured_router_logits.append(
                        ("alt_gate", name, o.detach().cpu() if isinstance(o, torch.Tensor) else o)
                    )
                ))
                moe_layers.append(name)

    print(f"  Hooked {len(hooks)} router modules: {moe_layers[:5]}...")

    n_correct = 0
    n_wrong = 0

    for qi, example in enumerate(questions):
        question = example["question"]
        correct_answers = example["correct_answers"]
        incorrect_answers = example["incorrect_answers"]
        best_answer = example.get("best_answer", "")

        prompt = f"Answer briefly and factually in one sentence. {question}"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        captured_router_logits.clear()

        try:
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)

            gen_ids = outputs[0, prompt_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()

            is_correct = check_truthfulqa_answer(
                response, correct_answers, incorrect_answers, best_answer,
            )

            if is_correct:
                n_correct += 1
            else:
                n_wrong += 1

            # Process captured routing data
            if captured_router_logits:
                # Compute routing entropy per captured logit tensor
                entropies = []
                top_expert_ids = []
                for item in captured_router_logits:
                    if isinstance(item, tuple) and len(item) == 3:
                        _, layer_name, logits = item
                        if isinstance(logits, torch.Tensor) and logits.ndim >= 2:
                            # logits shape: (batch*seq_len, num_experts) or (seq_len, num_experts)
                            probs = torch.softmax(logits.float(), dim=-1)
                            ent = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                            entropies.extend(ent.numpy().tolist())
                            top = logits.argmax(dim=-1)
                            top_expert_ids.extend(top.numpy().tolist())

                if entropies:
                    record = {
                        "mean_entropy": float(np.mean(entropies)),
                        "std_entropy": float(np.std(entropies)),
                        "n_routing_decisions": len(entropies),
                        "top_expert_distribution": {},
                    }
                    # Count top expert frequency
                    for eid in top_expert_ids:
                        record["top_expert_distribution"][str(eid)] = \
                            record["top_expert_distribution"].get(str(eid), 0) + 1

                    target = "correct" if is_correct else "wrong"
                    routing_data[target].append(record)

        except Exception as e:
            print(f"  WARNING: Question {qi} failed: {e}")
            n_wrong += 1

        if (qi + 1) % 10 == 0:
            n_c_routes = len(routing_data["correct"])
            n_w_routes = len(routing_data["wrong"])
            print(f"  [{qi+1:3d}/{n_questions}] correct={n_correct} wrong={n_wrong} "
                  f"routing_data: {n_c_routes}c/{n_w_routes}w")

    # Remove hooks
    for h in hooks:
        h.remove()

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return routing_data, n_correct, n_wrong


def analyze_routing(routing_data: dict) -> dict:
    """Analyze routing patterns for correct vs wrong answers."""
    results = {}

    for label in ["correct", "wrong"]:
        records = routing_data[label]
        if not records:
            results[label] = {"count": 0}
            continue

        entropies = [r["mean_entropy"] for r in records]
        results[label] = {
            "count": len(records),
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "median_entropy": float(np.median(entropies)),
        }

    # Compare
    if results["correct"]["count"] > 0 and results["wrong"]["count"] > 0:
        c_ent = results["correct"]["mean_entropy"]
        w_ent = results["wrong"]["mean_entropy"]
        results["entropy_gap"] = w_ent - c_ent
        results["entropy_ratio"] = w_ent / (c_ent + 1e-10)

        # Statistical test: are the distributions different?
        c_vals = [r["mean_entropy"] for r in routing_data["correct"]]
        w_vals = [r["mean_entropy"] for r in routing_data["wrong"]]

        # Simple threshold-based accuracy
        all_vals = c_vals + w_vals
        all_labels = [0] * len(c_vals) + [1] * len(w_vals)
        thresh = (np.mean(c_vals) + np.mean(w_vals)) / 2
        tp = sum(1 for v in w_vals if v > thresh)
        tn = sum(1 for v in c_vals if v <= thresh)
        results["entropy_detection_accuracy"] = (tp + tn) / (len(c_vals) + len(w_vals))

    return results


def main() -> int:
    print("=" * 70)
    print("EXPERIMENT 34: MoE Routing Patterns as Hallucination Signal")
    print("  Model: Qwen/Qwen3.5-35B-A3B (256 experts)")
    print("=" * 70)

    start = time.time()

    routing_data, n_correct, n_wrong = collect_routing_data(100)
    elapsed_collect = time.time() - start

    print(f"\nCollection done ({elapsed_collect:.0f}s): {n_correct} correct, {n_wrong} wrong")
    print(f"  Routing records: {len(routing_data['correct'])} correct, {len(routing_data['wrong'])} wrong")

    if not routing_data["correct"] or not routing_data["wrong"]:
        print("\n  WARNING: No routing data captured. The model may not expose router logits")
        print("  through standard hooks. This is a negative result — the router is not")
        print("  accessible via forward hooks in this architecture.")

        # Still try to analyze what we got
        if routing_data["correct"]:
            print(f"\n  Correct routing entropy: {np.mean([r['mean_entropy'] for r in routing_data['correct']]):.4f}")
        if routing_data["wrong"]:
            print(f"  Wrong routing entropy: {np.mean([r['mean_entropy'] for r in routing_data['wrong']]):.4f}")

        print("\n  VERDICT: ⚠️ Router hooks didn't capture routing decisions")
        return 0

    results = analyze_routing(routing_data)

    sep = "=" * 70
    print(f"\n{sep}")
    print("EXPERIMENT 34 RESULTS")
    print(sep)
    print(f"  Correct answers: entropy = {results['correct']['mean_entropy']:.4f} "
          f"(±{results['correct']['std_entropy']:.4f})")
    print(f"  Wrong answers:   entropy = {results['wrong']['mean_entropy']:.4f} "
          f"(±{results['wrong']['std_entropy']:.4f})")
    print(f"  Entropy gap:     {results.get('entropy_gap', 0):.4f}")
    print(f"  Detection accuracy (entropy threshold): {results.get('entropy_detection_accuracy', 0):.1%}")

    gap = results.get("entropy_gap", 0)
    acc = results.get("entropy_detection_accuracy", 0.5)

    if acc > 0.65:
        print(f"\n  VERDICT: ✅ Router entropy distinguishes correct from wrong ({acc:.1%})")
    elif acc > 0.55:
        print(f"\n  VERDICT: ⚠️ Weak routing signal ({acc:.1%})")
    else:
        print(f"\n  VERDICT: ❌ Router entropy does not distinguish ({acc:.1%})")

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
