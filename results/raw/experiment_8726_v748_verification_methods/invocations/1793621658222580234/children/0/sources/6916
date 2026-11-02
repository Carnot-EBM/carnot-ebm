#!/usr/bin/env python3
"""Experiment 1972: Draft EBT Decoding Loop using local SOTA models.

Spec: REQ-EBT-1972, SCENARIO-EBT-1972
"""

import sys
import json
from pathlib import Path

# Add python directory and project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "python"))
sys.path.insert(0, str(root_dir))

from scripts.experiment_template import ExperimentTemplate
from carnot.ebt_decoding import EBTDecodingLoop

def main():
    tmpl = ExperimentTemplate(
        exp_id=1972,
        title="Draft EBT Decoding Loop",
        deliverable="results/experiment_1972_ebt_baseline.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # Define 10 prompt cases
    prompts = [
        "Prompt case 1: Explain thermodynamics.",
        "Prompt case 2: Translate 'hello' to French.",
        "Prompt case 3: Write a poem about spring.",
        "Prompt case 4: Solve 2x + 3 = 11.",
        "Prompt case 5: Summarize the plot of Hamlet.",
        "Prompt case 6: What is the capital of Japan?",
        "Prompt case 7: Describe photosynthesis.",
        "Prompt case 8: Who wrote 1984?",
        "Prompt case 9: Explain the theory of relativity.",
        "Prompt case 10: How do airplanes fly?"
    ]

    # Step 1: Initialize the decoder loop
    # Using the required model: unsloth/Qwen3.6-35B-A3B-GGUF
    model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
    loop = EBTDecodingLoop(model_hf_id=model_id)
    
    # Step 2: Track multi-step energy minimization on 10 prompt cases
    with tmpl.phase("optimization_loop"):
        results = loop.decode_batch(prompts, max_steps=5)

    # Calculate average final energy
    total_energy = sum(r["final_energy"] for r in results)
    avg_energy = total_energy / len(prompts) if prompts else 0.0

    # Step 3: Write summary
    artifact = tmpl.build_result(
        {
            "model_used": model_id,
            "average_final_energy": avg_energy,
            "batch_results": results,
            "cases_evaluated": len(prompts)
        },
        status="success",
        code_files=[__file__, str(root_dir / "python" / "carnot" / "ebt_decoding.py")]
    )
    
    tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()

if __name__ == "__main__":
    main()
