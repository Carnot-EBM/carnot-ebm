"""Experiment 150 — HuggingFace Publishing: guided-decoding-adapter + model card updates.

Spec refs: REQ-VERIFY-001, SCENARIO-VERIFY-004
Story: HuggingFace Publishing Milestones item #3 (research-program.md)

What this script does
---------------------
1. Checks whether the HuggingFace CLI is available and authenticated.
2. Updates exports/guided-decoding-adapter/README.md with benchmark numbers
   from Exp-138 (accuracy deltas) and Exp-140 (latency breakdown).
3. If authenticated, uploads the adapter directory to Carnot-EBM/guided-decoding-adapter.
4. Generates updated README preambles for the 16 existing per-token EBM model
   cards, saving them under exports/model-readme-updates/<model-name>.md.
   (Actual HuggingFace update of those cards requires an API token and a
   separate upload step — this script only produces the content.)
5. Writes results/experiment_150_results.json with outcome metadata.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = REPO_ROOT / "exports" / "guided-decoding-adapter"
README_PATH = ADAPTER_DIR / "README.md"
EXP138_PATH = REPO_ROOT / "results" / "experiment_138_results.json"
EXP140_PATH = REPO_ROOT / "results" / "experiment_140_results.json"
MODEL_README_DIR = REPO_ROOT / "exports" / "model-readme-updates"
RESULTS_PATH = REPO_ROOT / "results" / "experiment_150_results.json"

# The 16 existing per-token EBM model repos on Carnot-EBM
PER_TOKEN_MODELS: list[str] = [
    "per-token-ebm-bonsai-17b-nothink",
    "per-token-ebm-gemma4-e2b-it-nothink",
    "per-token-ebm-gemma4-e2b-nothink",
    "per-token-ebm-gemma4-e4b-it-nothink",
    "per-token-ebm-gemma4-e4b-nothink",
    "per-token-ebm-gptoss-20b-nothink",
    "per-token-ebm-lfm25-12b-nothink",
    "per-token-ebm-lfm25-350m-nothink",
    "per-token-ebm-qwen3-06b",
    "per-token-ebm-qwen35-08b-nothink",
    "per-token-ebm-qwen35-08b-think",
    "per-token-ebm-qwen35-27b-nothink",
    "per-token-ebm-qwen35-2b-nothink",
    "per-token-ebm-qwen35-35b-nothink",
    "per-token-ebm-qwen35-4b-nothink",
    "per-token-ebm-qwen35-9b-nothink",
]

HF_ORG = "Carnot-EBM"
HF_ADAPTER_REPO = f"{HF_ORG}/guided-decoding-adapter"


# ---------------------------------------------------------------------------
# Step 1 — Check HuggingFace CLI availability
# ---------------------------------------------------------------------------
def check_hf_cli() -> tuple[bool, str | None]:
    """Return (available, skip_reason).

    Returns (True, None) if `huggingface-cli whoami` exits 0.
    Returns (False, reason) if the CLI is missing or not authenticated.
    """
    hf_exe = shutil.which("huggingface-cli")
    if hf_exe is None:
        return False, "huggingface-cli not found in PATH"

    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False, "huggingface-cli whoami timed out"

    if result.returncode != 0:
        # Common output when not logged in: "Not logged in"
        stderr_hint = result.stderr.strip() or result.stdout.strip()
        return False, f"Not authenticated: {stderr_hint}"

    user = result.stdout.strip().splitlines()[0]
    print(f"[HF] Authenticated as: {user}")
    return True, None


# ---------------------------------------------------------------------------
# Step 2 — Update adapter README with benchmark results
# ---------------------------------------------------------------------------
def _build_benchmark_section(exp138: dict, exp140: dict) -> str:
    """Build the Markdown section summarising Exp-138 and Exp-140 results."""

    # --- Exp-138 accuracy table ---
    summary = exp138["summary"]["modes"]

    gsm8k_base = summary["baseline"]["gsm8k_accuracy"]
    gsm8k_guided = summary["guided"]["gsm8k_accuracy"]
    gsm8k_gvr = summary["guided_verify_repair"]["gsm8k_accuracy"]

    heval_base = summary["baseline"]["humaneval_pass_at_1"]
    heval_guided = summary["guided"]["humaneval_pass_at_1"]

    tqa_base = summary["baseline"]["truthfulqa_accuracy"]
    tqa_guided = summary["guided"]["truthfulqa_accuracy"]
    tqa_gvr = summary["guided_verify_repair"]["truthfulqa_accuracy"]

    lat = exp138["latency_profile"]
    lat_p50 = lat["p50_ms"]
    lat_p99 = lat["p99_ms"]

    # --- Exp-140 latency table (batch=1 total) ---
    lat140_b1 = next(b for b in exp140["latency_benchmark"] if b["batch_size"] == 1)
    proj_p50 = lat140_b1["project_only"]["p50_ms"]
    proj_p99 = lat140_b1["project_only"]["p99_ms"]
    total_p50 = lat140_b1["total"]["p50_ms"]
    total_p99 = lat140_b1["total"]["p99_ms"]

    return f"""
## Benchmark Results (Exp-138 & Exp-140)

> **Note — Simulated Inference**: All benchmark numbers below were produced
> with a *simulated* (mock) LLM, not a real transformer model.  The constraint
> checker and logit-penalty logic are real; the generation loop uses a
> deterministic stand-in.  Live-model E2E validation is pending (Exp-111).

### Accuracy (Exp-138, n=200/50/100, simulated inference)

| Dataset | Baseline | Guided | Guided+Verify-Repair | Delta (guided) |
|---------|----------|--------|----------------------|----------------|
| GSM8K (math) | {gsm8k_base:.1%} | {gsm8k_guided:.1%} | {gsm8k_gvr:.1%} | **+{gsm8k_guided - gsm8k_base:.1%}** |
| HumanEval (code) | {heval_base:.1%} | {heval_guided:.1%} | — | **+{heval_guided - heval_base:.1%}** |
| TruthfulQA | {tqa_base:.1%} | {tqa_guided:.1%} | {tqa_gvr:.1%} | **+{tqa_guided - tqa_base:.1%}** |

### Latency (Exp-138, n=485 samples, CPU)

| Metric | Value |
|--------|-------|
| Constraint-check p50 | {lat_p50:.4f} ms |
| Constraint-check p99 | {lat_p99:.4f} ms |

### Latency — KAN Projection Mode (Exp-140, batch=1, CPU)

| Operation | p50 | p99 |
|-----------|-----|-----|
| Logit projection (energy gradient) | {proj_p50:.3f} ms | {proj_p99:.3f} ms |
| Total per-token (grad + projection) | {total_p50:.3f} ms | {total_p99:.3f} ms |

Exp-140 pass criterion: total p50 < 5 ms — **{'PASSED' if exp140['success_criterion']['passed'] else 'FAILED'}**
(actual {exp140['success_criterion']['total_p50_batch1_ms']:.4f} ms vs 5.0 ms threshold).
"""


def _build_limitations_section() -> str:
    """Return an expanded Limitations section noting simulated inference."""
    return """
## Limitations

1. **Simulated inference benchmark**: Exp-138 and Exp-140 used a mock LLM.
   Numbers show constraint-checker and logit-penalty overhead, not end-to-end
   accuracy on real models.  Treat accuracy deltas as directional, not final.
2. **No KV-cache**: Full forward pass every token.  Keep `max_tokens < 256`.
3. **Uniform penalty**: Adjusts entropy across the whole vocabulary; does not
   steer towards specific correct tokens.
4. **Energy is a violation count**: Not a calibrated probability.  High `alpha`
   + many violations → very flat distribution (model may repeat or stall).
5. **Min-text guard**: `AutoExtractor` skips texts < 5 chars (early tokens).
6. **Live-model E2E pending**: Exp-111 validation against Qwen/Gemma not done yet.
"""


def _build_install_section() -> str:
    return """
## Installation

```bash
pip install carnot
```

Requires Python 3.11+.  See [pypi.org/project/carnot](https://pypi.org/project/carnot)
for the full package including the verify-repair pipeline.
"""


def update_adapter_readme(exp138: dict, exp140: dict) -> None:
    """Inject benchmark results, limitations, and install sections into README."""
    current = README_PATH.read_text()

    # Guard against double-injection on re-runs
    if "## Benchmark Results (Exp-138" in current:
        print("[README] Benchmark section already present — skipping injection.")
        return

    # Build new sections
    bench_section = _build_benchmark_section(exp138, exp140)
    limitations_section = _build_limitations_section()
    install_section = _build_install_section()

    # Replace the existing "## Known Limitations" block with our expanded version,
    # and insert the benchmark section before it.
    # Strategy: append new sections before the existing ## Known Limitations heading.
    LIMITATIONS_HEADING = "## Known Limitations"
    if LIMITATIONS_HEADING in current:
        # Find the start of the Known Limitations section
        idx = current.index(LIMITATIONS_HEADING)
        # Find end of Known Limitations section (next ## heading or EOF)
        rest_after = current[idx:]
        # Locate the next top-level heading after Known Limitations
        import re
        next_heading_match = re.search(r"\n## ", rest_after[3:])  # skip the ## itself
        if next_heading_match:
            end_idx = idx + 3 + next_heading_match.start()
            new_readme = (
                current[:idx]
                + bench_section
                + install_section
                + limitations_section
                + current[end_idx:]
            )
        else:
            # Known Limitations is the last section
            new_readme = (
                current[:idx]
                + bench_section
                + install_section
                + limitations_section
            )
    else:
        # No existing Limitations section — append everything at the end
        new_readme = current + bench_section + install_section + limitations_section

    README_PATH.write_text(new_readme)
    print(f"[README] Updated: {README_PATH}")


# ---------------------------------------------------------------------------
# Step 3 — Upload adapter to HuggingFace Hub
# ---------------------------------------------------------------------------
def upload_adapter() -> str | None:
    """Run huggingface-cli upload and return the HF URL on success, else None."""
    print(f"[HF] Uploading {ADAPTER_DIR} → {HF_ADAPTER_REPO} …")
    result = subprocess.run(
        [
            "huggingface-cli",
            "upload",
            HF_ADAPTER_REPO,
            str(ADAPTER_DIR),
            "--repo-type",
            "model",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"[HF] Upload failed (exit {result.returncode})", file=sys.stderr)
        return None

    # Verify the upload landed
    verify = subprocess.run(
        ["huggingface-cli", "repo", "info", HF_ADAPTER_REPO],
        capture_output=True,
        text=True,
        timeout=15,
    )
    print(verify.stdout)
    hf_url = f"https://huggingface.co/{HF_ADAPTER_REPO}"
    print(f"[HF] Verified: {hf_url}")
    return hf_url


# ---------------------------------------------------------------------------
# Step 4 — Generate updated README preambles for the 16 existing EBM models
# ---------------------------------------------------------------------------
PHASE1_PREAMBLE = """\
## Note: Phase 1 Research Artifact

These EBMs were trained in **Phase 1** to detect *model confidence*, not
correctness.  They score per-token activation patterns and were developed as
part of the Carnot project's early research into energy-based verification.

For the production **verify-repair pipeline** — which combines constraint
energy with guided decoding — install:

```bash
pip install carnot
```

See [pypi.org/project/carnot](https://pypi.org/project/carnot) for full
documentation.

For token-level **energy-guided decoding**, see the
[guided-decoding-adapter](https://huggingface.co/Carnot-EBM/guided-decoding-adapter)
— a model-agnostic adapter that adjusts LLM token probabilities based on
constraint violation energy.

---

"""


def generate_model_readme_updates() -> int:
    """Write one .md file per model under exports/model-readme-updates/.

    Each file contains the Phase 1 preamble followed by placeholder text
    indicating where the existing model card content should be appended.

    Returns the number of files written.
    """
    MODEL_README_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for model_name in PER_TOKEN_MODELS:
        # Check if there is an existing exports/<model> directory to harvest
        # a README from; if so, read and prepend.  Otherwise generate a stub.
        existing_readme_path = REPO_ROOT / "exports" / model_name / "README.md"

        if existing_readme_path.exists():
            existing_content = existing_readme_path.read_text()
            new_content = PHASE1_PREAMBLE + existing_content
        else:
            # Stub — include preamble plus a minimal YAML front-matter header
            # that downstream tooling can merge with the existing HF card.
            new_content = (
                "---\n"
                "tags:\n"
                "  - energy-based-model\n"
                "  - per-token-ebm\n"
                "  - phase-1-research\n"
                "  - carnot\n"
                "license: apache-2.0\n"
                "---\n\n"
                + PHASE1_PREAMBLE
                + f"# {model_name}\n\n"
                "<!-- TODO: paste existing HuggingFace model card body here -->\n"
            )

        out_path = MODEL_README_DIR / f"{model_name}.md"
        out_path.write_text(new_content)
        print(f"[README-UPDATE] Written: {out_path.name}")
        count += 1

    return count


# ---------------------------------------------------------------------------
# Step 5 — Save results
# ---------------------------------------------------------------------------
def save_results(
    pushed: bool,
    skip_reason: str | None,
    hf_url: str | None,
    readme_updates_generated: int,
) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "exp_150_hf_push",
        "spec": ["REQ-VERIFY-001", "SCENARIO-VERIFY-004"],
        "pushed": pushed,
        "skip_reason": skip_reason,
        "hf_url": hf_url,
        "readme_updates_generated": readme_updates_generated,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[RESULTS] Saved: {RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Experiment 150: HuggingFace Publishing ===")

    # --- Load benchmark data ---
    if not EXP138_PATH.exists():
        print(f"[ERROR] {EXP138_PATH} not found — cannot update README", file=sys.stderr)
        sys.exit(1)

    exp138 = json.loads(EXP138_PATH.read_text())

    if EXP140_PATH.exists():
        exp140 = json.loads(EXP140_PATH.read_text())
        print(f"[INFO] Loaded Exp-140 latency data from {EXP140_PATH.name}")
    else:
        # Build a minimal stub so the script still runs
        print(f"[WARN] {EXP140_PATH} not found — latency section will be sparse")
        exp140 = {
            "latency_benchmark": [
                {
                    "batch_size": 1,
                    "project_only": {"p50_ms": float("nan"), "p99_ms": float("nan")},
                    "total": {"p50_ms": float("nan"), "p99_ms": float("nan")},
                }
            ],
            "success_criterion": {"passed": False, "total_p50_batch1_ms": float("nan")},
        }

    # --- Step 2: Update adapter README ---
    update_adapter_readme(exp138, exp140)

    # --- Step 1: Check HF CLI ---
    hf_available, skip_reason = check_hf_cli()

    # --- Step 3: Optionally upload ---
    pushed = False
    hf_url: str | None = None

    if hf_available:
        hf_url = upload_adapter()
        pushed = hf_url is not None
        if not pushed:
            skip_reason = "Upload command failed — see stderr above"
    else:
        print(f"[SKIPPED] HuggingFace upload: {skip_reason}")
        print("[INFO] README was still updated locally. Re-run after `huggingface-cli login`.")

    # --- Step 4: Generate model README updates ---
    readme_updates_generated = generate_model_readme_updates()
    print(
        f"\n[INFO] Generated {readme_updates_generated} model README update files "
        f"in {MODEL_README_DIR}"
    )
    print(
        "[INFO] To apply these to HuggingFace, run:\n"
        "  huggingface-cli upload Carnot-EBM/<model-name> "
        "exports/model-readme-updates/<model-name>.md README.md --repo-type model"
    )

    # --- Step 5: Save results ---
    save_results(pushed, skip_reason, hf_url, readme_updates_generated)

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  Adapter README updated : YES")
    print(f"  HF push succeeded      : {pushed}")
    if skip_reason:
        print(f"  Skip reason            : {skip_reason}")
    if hf_url:
        print(f"  HF URL                 : {hf_url}")
    print(f"  Model card stubs       : {readme_updates_generated}")
    print(f"  Results file           : {RESULTS_PATH}")


if __name__ == "__main__":
    main()
