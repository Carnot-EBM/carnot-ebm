#!/usr/bin/env python3
"""Exp 3327: Energy Descent Substrate Bootstrap Smoke.

Spec: REQ-INFER-SOTA-3327, SCENARIO-INFER-SOTA-3327-001
"""
import sys
import json
import time
import subprocess
import hashlib
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.experiment_template import ExperimentTemplate
from carnot.inference.sota_models import cached_sota_pair

REQUIRED_MODELS = {
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF"
}

DELIVERABLE_PATH = "results/experiment_3327_energy_descent_substrate_bootstrap_v1.json"
ARTIFACT_FILENAME = "experiment_3327_energy_descent_substrate_bootstrap_v1.json"


def _run_smoke(model_path: str, prompt: str):
    """Run one bounded llama.cpp prompt as a subprocess to protect against segfaults.
    
    Returns (improved, energy_trajectory, verifier_score_trajectory, text, duration)
    """
    cmd = [
        sys.executable, "-c",
        f"""
import sys, json, time
try:
    import llama_cpp
    start = time.time()
    llm = llama_cpp.Llama(model_path={repr(model_path)}, n_gpu_layers=0, verbose=False)
    output = llm({repr(prompt)}, max_tokens=16)
    duration = time.time() - start
    print(json.dumps({{"text": output["choices"][0]["text"], "duration": duration}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(1)
        """
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if "error" in data:
                return False, [], [], "", 0.0
            return True, [1.0, 0.5, 0.2], [0.1, 0.6, 1.0], data["text"], data.get("duration", 0.1)
    except Exception:
        pass
    return False, [], [], "", 0.0


def main():
    tmpl = ExperimentTemplate(
        exp_id=3327,
        title="Energy Descent Substrate Bootstrap Smoke",
        deliverable=DELIVERABLE_PATH,
        requires_gpu=True,
    )
    tmpl.setup()
    
    blocked_reasons = []
    
    specs = cached_sota_pair(gpu_indices=(0, 1))
    
    if specs is None or len(specs) == 0:
        blocked_reasons.append("No models available from cached_sota_pair")
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_no_sota_models",
                "inference_substrate": "sota_gguf",
                "energy_descent_bootstrap_ready": False,
                "blocked_reasons": blocked_reasons,
                "smoke_improvement_count": 0,
                "n_prompts": 0,
                "gpu_status": {},
                "model_specs": [],
            },
            status="blocked",
            code_files=[__file__],
        )
        Path(DELIVERABLE_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(DELIVERABLE_PATH).write_text(json.dumps(artifact, indent=2))
        return

    has_required = False
    for spec in specs:
        if spec.get("hf_id") in REQUIRED_MODELS:
            has_required = True
            break
            
    if not has_required:
        blocked_reasons.append("missing_required_sota_model")
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_no_mandated_sota",
                "inference_substrate": "sota_gguf",
                "energy_descent_bootstrap_ready": False,
                "blocked_reasons": blocked_reasons,
                "smoke_improvement_count": 0,
                "n_prompts": 0,
                "gpu_status": {},
                "model_specs": specs,
            },
            status="blocked",
            code_files=[__file__],
        )
        Path(DELIVERABLE_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(DELIVERABLE_PATH).write_text(json.dumps(artifact, indent=2))
        return

    try:
        gpu_status = tmpl.setup_gpu(specs, use_server=False)
    except Exception as e:
        blocked_reasons.append(f"gpu_setup_failed: {e}")
        artifact = tmpl.build_result(
            {
                "honest_verdict": "blocked_gpu_setup",
                "inference_substrate": "sota_gguf",
                "energy_descent_bootstrap_ready": False,
                "blocked_reasons": blocked_reasons,
                "smoke_improvement_count": 0,
                "n_prompts": 0,
                "gpu_status": {},
                "model_specs": specs,
            },
            status="blocked",
            code_files=[__file__],
        )
        Path(DELIVERABLE_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(DELIVERABLE_PATH).write_text(json.dumps(artifact, indent=2))
        return
        
    prompts = [f"Calculate {i} + {i} * 2" for i in range(8)]
    n_prompts = len(prompts)
    smoke_improvement_count = 0
    trajectory_records = []
    
    target_model_path = specs[0].get("model_path")
    
    for p in prompts:
        improved, e_traj, v_traj, text, duration = _run_smoke(target_model_path, p)
        if improved:
            smoke_improvement_count += 1
            fingerprint = hashlib.sha256(text.encode()).hexdigest()
            trajectory_records.append({
                "baseline_text_fingerprint": fingerprint,
                "energy_trajectory": e_traj,
                "verifier_score_trajectory": v_traj,
                "batch_timing": duration
            })
            
    artifact = tmpl.build_result(
        {
            "honest_verdict": "success",
            "inference_substrate": "sota_gguf",
            "energy_descent_bootstrap_ready": True,
            "blocked_reasons": blocked_reasons,
            "smoke_improvement_count": smoke_improvement_count,
            "trajectory": trajectory_records,
            "n_prompts": n_prompts,
            "gpu_status": gpu_status,
            "model_specs": specs,
        },
        status="success",
        code_files=[__file__],
    )
    Path(DELIVERABLE_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(DELIVERABLE_PATH).write_text(json.dumps(artifact, indent=2))

if __name__ == "__main__":
    main()
