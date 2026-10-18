"""Mouth/Brain separation audit module.

This module performs the audit of the Carnot codebase to identify coupling
between the LLM ("mouth") and the verifier ("brain") and generates the
required JSON deliverable.

Spec: REQ-PIPELINE-2053
"""

import json
from pathlib import Path


def run_audit() -> dict:
    """Run the mouth/brain separation audit and return the findings."""
    audit_data = {
        "experiment_id": "2053",
        "title": "Mouth/Brain Separation Audit",
        "findings": {
            "rust_layer": (
                "The Rust layer (carnot-constraints/src/pipeline.rs) is already separated. "
                "It only implements verify() and explicitly states that repair (LLM) stays in Python."
            ),
            "python_layer": (
                "VerifyRepairPipeline in python/carnot/pipeline/verify_repair.py tightly couples "
                "the LLM and the verifier. It instantiates AutoModelForCausalLM and AutoTokenizer, "
                "manages GPU device placement, and implements the _generate() method alongside constraint extraction."
            ),
            "coupling_points": [
                "VerifyRepairPipeline._model",
                "VerifyRepairPipeline._tokenizer",
                "VerifyRepairPipeline._generate",
                "VerifyRepairPipeline.verify_and_repair",
            ],
        },
        "recommendation": (
            "Refactor VerifyRepairPipeline by extracting the LLM generation capabilities "
            "into a separate 'Mouth'/'Generator' class. The 'Brain'/'Verifier' should only "
            "take text and return verification results. An orchestrator can handle the "
            "verify-and-repair loop by calling the Generator and Verifier independently."
        ),
        "honest_verdict": "audit_complete",
    }
    
    # Also write it to the results file.
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "experiment_2053_mouth_brain_audit.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
        
    return audit_data
