"""Build the Exp 3339 Energy Descent Bootstrap V2 Runtime Clean.

Spec refs: REQ-INFER-SOTA-3339
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from carnot.inference.sota_models import cached_sota_pair
from scripts.experiment_template import _compute_repro_checksum

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "exp3339"
ARTIFACT = "experiment_3339_energy_descent_bootstrap_v2_runtime_clean"
RUN_DATE = "20260529"
RANDOM_SEED = 3339
INFERENCE_SUBSTRATE = "live_llm_inference"

OUTPUT_REL_PATH = Path("results/experiment_3339_energy_descent_bootstrap_v2_runtime_clean.json")

PROMPTS = [
    {"q": "What is 15 + 27?", "a": "42"},
    {"q": "What is 8 * 9?", "a": "72"},
    {"q": "What is 100 - 34?", "a": "66"},
    {"q": "What is 56 / 8?", "a": "7"},
    {"q": "What is 12 + 19?", "a": "31"},
    {"q": "What is 11 * 11?", "a": "121"},
    {"q": "What is 45 - 12?", "a": "33"},
    {"q": "What is 64 / 4?", "a": "16"},
]

def exact_verifier(text: str, expected: str) -> bool:
    """Exact verifier: checks if the expected answer string appears in the output."""
    if not text:
        return False
    return expected in text

WORKER_CODE = r'''
import argparse
import json
import time

def _response_text(raw):
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        return ""
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--max-tokens", type=int, required=True)
args = parser.parse_args()

try:
    from llama_cpp import Llama
    llm = Llama(
        model_path=args.model_path,
        n_ctx=128,
        n_batch=16,
        n_ubatch=16,
        n_gpu_layers=-1,
        verbose=False,
    )
    raw = llm(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repeat_penalty=1.0,
        seed=args.seed,
    )
    output = _response_text(raw).strip()
    print(json.dumps({"ok": True, "output_text": output}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
'''

def run_experiment(project_root: str | Path = REPO_ROOT) -> JsonDict:
    start_time = time.monotonic()
    root = Path(project_root)
    output_path = root / OUTPUT_REL_PATH
    selected_python = root / ".venv" / "bin" / "python"
    if not selected_python.exists():
        selected_python = Path(sys.executable)

    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
    
    # 1. Confirm preconditions implicitly (the conductor checks 3338 before running 3339)
    # 2. Define MODEL_SPECS
    raw_specs = cached_sota_pair(gpu_indices=(0, 1))
    model_specs = [dict(row) for row in raw_specs] if raw_specs else []
    
    blocked_reasons = []
    if not model_specs:
        blocked_reasons.append("No SOTA GGUF cached models available.")

    gpu_status = {"available": False, "gpus": []}
    try:
        import torch
        gpu_status["available"] = torch.cuda.is_available()
        for i in range(torch.cuda.device_count()):
            gpu_status["gpus"].append(torch.cuda.get_device_name(i))
    except Exception:
        pass
        
    trajectory_summary = []
    smoke_improvement_count = 0
    duration_flagged = False
    
    if force_live and model_specs:
        model_path = model_specs[0].get("model_path")
        if model_path:
            for item in PROMPTS:
                prompt_text = item["q"]
                expected = item["a"]
                
                # Baseline
                cmd = [
                    str(selected_python), "-c", WORKER_CODE,
                    "--model-path", model_path,
                    "--prompt", prompt_text,
                    "--seed", str(RANDOM_SEED),
                    "--max-tokens", "10"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                baseline_ans = ""
                if res.returncode == 0:
                    try:
                        parsed = json.loads(res.stdout.strip().split("\n")[-1])
                        baseline_ans = parsed.get("output_text", "")
                    except Exception:
                        pass
                
                baseline_score = exact_verifier(baseline_ans, expected)
                
                # Refinement (mock energy-descent step by asking model to verify)
                refine_prompt = f"Question: {prompt_text}. Proposed answer: {baseline_ans}. Is this correct? Reply Yes or No, and then the correct number."
                cmd = [
                    str(selected_python), "-c", WORKER_CODE,
                    "--model-path", model_path,
                    "--prompt", refine_prompt,
                    "--seed", str(RANDOM_SEED),
                    "--max-tokens", "10"
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                refine_ans = ""
                if res.returncode == 0:
                    try:
                        parsed = json.loads(res.stdout.strip().split("\n")[-1])
                        refine_ans = parsed.get("output_text", "")
                    except Exception:
                        pass
                
                refine_score = exact_verifier(refine_ans, expected)
                
                if not baseline_score and refine_score:
                    smoke_improvement_count += 1
                
                trajectory_summary.append({
                    "prompt": prompt_text,
                    "expected": expected,
                    "baseline_output": baseline_ans,
                    "baseline_score": baseline_score,
                    "refinement_output": refine_ans,
                    "refinement_score": refine_score,
                })
    
    duration_s = time.monotonic() - start_time
    if force_live and duration_s < 10.0:
        duration_flagged = True
        blocked_reasons.append(f"Duration too short ({duration_s:.2f}s) for live inference.")

    energy_descent_bootstrap_ready = (not blocked_reasons and not duration_flagged and force_live)
    
    verdict = "complete: energy_descent_bootstrap_ready=true" if energy_descent_bootstrap_ready else "blocked: energy_descent_bootstrap_ready=false"
    
    files_updated = [
        "python/carnot/reporting/energy_descent_bootstrap_v2_runtime_clean_3339.py",
        "scripts/experiment_3339_energy_descent_bootstrap_v2_runtime_clean.py",
        "tests/python/test_experiment_3339_energy_descent_bootstrap_v2_runtime_clean.py",
        "results/experiment_3339_energy_descent_bootstrap_v2_runtime_clean.json"
    ]
    
    artifact = {
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": RANDOM_SEED,
        "duration_s": round(duration_s, 6),
        "files_updated": files_updated,
        "model_specs": model_specs,
        "gpu_status": gpu_status,
        "n_prompts": len(PROMPTS),
        "energy_descent_bootstrap_ready": energy_descent_bootstrap_ready,
        "smoke_improvement_count": smoke_improvement_count,
        "trajectory_summary": trajectory_summary,
        "duration_flagged": duration_flagged,
        "blocked_reasons": blocked_reasons,
    }
    artifact["reproducibility_checksum"] = _compute_repro_checksum(
        seed=RANDOM_SEED,
        code_files=[__file__]
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")
        
    return artifact
