#!/usr/bin/env python3
"""Experiment 3566 — FR-11 Multi-corpus Deploy Verifier Diversity Grounding v1."""

from __future__ import annotations

import json
import os
import sys
import random
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from carnot.fr11.multicorpus_deploy_verifier_diversity_grounding_v1 import (
    run_multicorpus_deploy,
    RANDOM_SEED,
)

RESULT_PATH = os.path.join(
    REPO_ROOT,
    "results",
    "experiment_3566_fr11_multicorpus_deploy_verifier_diversity_grounding_v1.json",
)

def load_battery() -> list[list[dict[str, Any]]]:
    """Assemble 3 non-degenerate corpora."""
    battery = []
    
    # Corpus 1: p01 flattened
    p01_path = os.path.join(REPO_ROOT, "data", "p01_difficulty_matched_generations_flattened_v2.jsonl")
    c1 = []
    if os.path.exists(p01_path):
        with open(p01_path) as f:
            for line in f:
                d = json.loads(line)
                d["is_correct"] = bool(d.get("is_correct", d.get("correct", False)))
                c1.append(d)
                if len(c1) >= 500: break
    battery.append(c1)
    
    # Corpus 2 and 3: Subsampled fover to hit ~45% and ~55% accuracy
    fover_path = os.path.join(REPO_ROOT, "data", "fover_corpus.jsonl")
    correct_recs = []
    incorrect_recs = []
    if os.path.exists(fover_path):
        with open(fover_path) as f:
            for line in f:
                d = json.loads(line)
                d["is_correct"] = (d.get("label") == "correct")
                if d["is_correct"]:
                    correct_recs.append(d)
                else:
                    incorrect_recs.append(d)
                if len(correct_recs) > 1000 and len(incorrect_recs) > 1000:
                    break
    
    rng = random.Random(RANDOM_SEED)
    
    # Corpus 2: target 45% correct
    c2 = correct_recs[:225] + incorrect_recs[:275]
    rng.shuffle(c2)
    battery.append(c2)
    
    # Corpus 3: target 55% correct
    c3 = correct_recs[225:500] + incorrect_recs[275:500]
    rng.shuffle(c3)
    battery.append(c3)
    
    return battery

def main():
    battery = load_battery()
    artifact = run_multicorpus_deploy(battery)

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written to {RESULT_PATH}")
    print(f"Honest verdict: {artifact.get('honest_verdict')}")

if __name__ == "__main__":
    main()
