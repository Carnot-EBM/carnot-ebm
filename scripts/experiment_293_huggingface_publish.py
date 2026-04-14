"""Experiment 293: HuggingFace publishing of Exp 66 joint EBM and FormalClaimVerifier.

Carry-forward from Exp 268 (SKIP'd 3×). Publishes two artifacts to HuggingFace Hub:

  1. Carnot-EBM/carnot-joint-constraint-v1  — Exp 66 differentiable constraint model
     (embed_dim=384, Ising coupling, MLP head, 1.0 AUROC on held-out validation set)
  2. Carnot-EBM/carnot-formal-claim-verifier-v1  — FormalClaimVerifier with ONNX
     exports for arithmetic and comparison solver routes

Credential check runs FIRST.  If `huggingface-cli whoami` fails, the script emits
a blocked artifact JSON with login instructions and exits without uploading.

Spec: REQ-VERIFY-058, REQ-VERIFY-059
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Architecture constants from Exp 66 results JSON
# ---------------------------------------------------------------------------

_EXP66_ARCH = {
    "embed_dim": 384,
    "n_constraints": 8,
    "hidden_dim": 64,
    "alpha": 10.0,
    "best_lr": 0.001,
    "n_epochs": 200,
    "test_auroc": 1.0,
}

# HuggingFace repo IDs for this experiment
_EXP66_REPO_ID = "Carnot-EBM/carnot-joint-constraint-v1"
_FCV_REPO_ID = "Carnot-EBM/carnot-formal-claim-verifier-v1"

# Tag applied to both repos
_TAG = "v0.2.0-research"

# Results file path (relative to repo root)
_RESULTS_PATH = Path(__file__).parent.parent / "results" / "experiment_293_results.json"

# Exp 66 existing results JSON (used for eval numbers)
_EXP66_RESULTS_PATH = Path(__file__).parent.parent / "results" / "experiment_66_results.json"

# Exp 66 trained model weights — published artifact if it exists, skip if absent.
# Do NOT synthesize random weights in the main pipeline; that would publish noise under
# the guise of the trained model.  Only build_exp66_safetensors() (used in tests) creates
# synthetic weights for schema / shape validation.
_EXP66_SAFETENSORS_PATH = Path(__file__).parent.parent / "results" / "experiment_66_model.safetensors"


# ---------------------------------------------------------------------------
# Credential check
# ---------------------------------------------------------------------------


def check_hf_credentials() -> tuple[bool, str]:
    """Check whether HuggingFace credentials are configured.

    Runs ``huggingface-cli whoami`` as a subprocess.  This is the authoritative
    credential check — the HF Python API does not surface login state until an
    actual API call is attempted, making subprocess the most reliable method.

    Returns:
        (True, "") when logged in.
        (False, instructions) when not logged in or CLI not found.
    """
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError:
        return False, (
            "huggingface-cli not found. "
            "Install it with: pip install huggingface_hub\n"
            "Then authenticate with: huggingface-cli login"
        )
    except Exception as exc:
        return False, (
            f"Error running huggingface-cli whoami: {exc}\n"
            "Authenticate with: huggingface-cli login"
        )

    if result.returncode != 0:
        return False, (
            "Not logged in to HuggingFace Hub.\n"
            "Run: huggingface-cli login\n"
            "or set the HF_TOKEN environment variable."
        )

    # Logged in — return the username from stdout if available
    username = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    msg = f"logged in as {username}" if username else ""
    return True, msg


# ---------------------------------------------------------------------------
# Model card builders
# ---------------------------------------------------------------------------


def build_exp66_model_card() -> str:
    """Build the model card for the Exp 66 joint EBM + Ising constraint model.

    The card is written for a public HuggingFace audience: it prominently
    states the Phase 1 research prototype disclaimer, reports the 1.0 AUROC
    result with honest provenance caveats, explains the architecture, and
    provides a minimal usage example with pip install carnot instructions.

    Returns:
        Markdown string suitable for README.md in the HF repo.
    """
    card = """---
tags:
  - energy-based-model
  - constraint-verification
  - research-prototype
  - carnot
license: apache-2.0
---

# carnot-joint-constraint-v1

> **Phase 1 research prototype.** 1.0 AUROC on held-out validation data.
> **Not production quality.** This is a research artifact for reproducibility.

## Overview

`carnot-joint-constraint-v1` is the Exp 66 differentiable constraint model from
the [Carnot](https://github.com/ianblenke/carnot) project.  It combines:

- **Embedding layer** — text input projected to 384-dimensional space (embed_dim=384)
- **Ising coupling** — learned pairwise interactions among 8 latent constraint nodes
- **MLP scoring head** — hidden_dim=64 projection to a scalar confidence score

The joint model achieves **AUROC 1.0** on the held-out validation split across
arithmetic, code, logic, factual, and scheduling domains.

> ⚠️ **Provenance note:** Exp 66 metrics were produced in a simulated training run
> (JAX CPU, synthetic data). The 1.0 AUROC should be treated as an in-distribution
> fit metric, not as a live-inference benchmark.  See `results/experiment_66_results.json`
> for full details.

## Architecture

```
text → sentence-embedding (384-d)
     → Ising coupling (8 × 8 pairwise J matrix)
     → concat([embedding, Ising_activations, Ising_biases, alpha])
     → Linear(393 → 64) + ReLU
     → Linear(64 → 1) + sigmoid
     → confidence score
```

| Parameter | Value |
|-----------|-------|
| embed_dim | 384 |
| n_constraints | 8 |
| hidden_dim | 64 |
| alpha | 10.0 |
| n_epochs | 200 |
| best_lr | 0.001 |
| optimizer | Adam |

## Eval Numbers

| Split | AUROC (joint) | AUROC (Ising only) | AUROC (embed only) |
|-------|:---:|:---:|:---:|
| Validation | **1.0** | 0.540 | 0.979 |

## Install and Use

```bash
pip install carnot
```

```python
from safetensors.numpy import load_file
import numpy as np

# Load weights
weights = load_file("exp66.safetensors")

# Access Ising coupling matrix (8 × 8)
J = weights["ising_J"]
biases = weights["ising_biases"]
print(f"J shape: {J.shape}, biases shape: {biases.shape}")
```

## Files

- `exp66.safetensors` — model weights in safetensors format
- `config.json` — architecture hyperparameters
- `README.md` — this file

## Citation

```bibtex
@misc{carnot-exp66-2026,
  title={Exp 66: Differentiable Constraint Verification via Joint EBM + Ising Architecture},
  author={Carnot Research},
  year={2026},
  url={https://github.com/ianblenke/carnot}
}
```
""".strip()
    return card


def build_fcv_model_card() -> str:
    """Build the model card for the FormalClaimVerifier.

    Documents all five solver routes, the abstention policy, ONNX exports
    for the arithmetic and comparison routes, and a standalone usage example
    using ``from carnot.pipeline import FormalClaimVerifier``.

    Returns:
        Markdown string suitable for README.md in the HF repo.
    """
    card = """---
tags:
  - formal-verification
  - claim-verifier
  - onnx
  - research-prototype
  - carnot
license: apache-2.0
---

# carnot-formal-claim-verifier-v1

Solver-routed verifier for typed formal claims.  Routes each claim to the
narrowest deterministic checker; claims that cannot be safely formalized
receive an explicit **abstain** verdict rather than a heuristic guess.

## Solver Routes

| Route | Description |
|-------|-------------|
| `arithmetic` | Verifies that `a OP b == result` (exhaustive over +, −, ×, ÷) |
| `comparison` | Checks `less_than` / `greater_than` / `between` relations |
| `cardinality` | Count constraints: exact equality or between-range |
| `set_membership` | Element `in` / `not in` set checks against bound variables |
| `boolean_entailment` | Attribute/property equality claims against a known vocabulary |

## Abstention Policy

The verifier **abstains** (rather than guessing) when:

- `formalization_status != "formalized"` — claim was not fully parsed
- `candidate_solver_route` is not in the supported set above
- The selected checker lacks sufficient operands (e.g. < 3 for arithmetic)
- `not_contains` set-membership (requires scanning text not in claim struct)

This explicit abstain policy means downstream consumers always receive a
signal about verifier confidence — there are no silent failures.

## ONNX Exports

Two routes are exported as portable ONNX models for hardware-accelerated inference:

- `arithmetic_route.onnx` — input `operands [batch, 3]` → output `verdict [batch]`
  - `verdict[i] = 1` if `|a − b − result| < 0.5`, else 0
- `comparison_route.onnx` — input `operands [batch, 2]` → output `verdict [batch]`
  - `verdict[i] = 1` if `x < y` (less_than), else 0

Both models use opset 13 and support CPU and CUDA execution providers.

The `boolean_entailment` and `set_membership` routes are packaged as Python
modules (pure Python, no ONNX required).

## Install and Use

```bash
pip install carnot
```

```python
from carnot.pipeline import FormalClaimVerifier
from carnot.pipeline.formal_claim_verifier import normalize_claim

# Initialize the verifier
fcv = FormalClaimVerifier()

# Verify a single raw claim dict
raw_claim = {
    "claim_id": "c1",
    "claim_text": "100 minus 24 equals 76",
    "candidate_solver_route": "arithmetic",
    "formalization_status": "formalized",
    "relation_type": "equation",
    "operands": [100.0, 24.0, 76.0],
    "target": "",
    "bound_variables": [],
}
claim = normalize_claim(raw_claim)
verdict = fcv.verify_claim(claim)
print(verdict.verdict)   # "supported"
print(verdict.route)     # "arithmetic"
```

### Batch verification

```python
from carnot.pipeline.formal_claim_verifier import verify_formal_claims

results = verify_formal_claims([raw_claim1, raw_claim2, raw_claim3])
print(results.counts)       # {"supported": 2, "violated": 1, "abstain": 0}
print(results.route_counts) # {"arithmetic": 2, "comparison": 1}
```

### ONNX inference (arithmetic route)

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("arithmetic_route.onnx", providers=["CPUExecutionProvider"])
operands = np.array([[100.0, 24.0, 76.0]], dtype=np.float32)
[verdict] = sess.run(None, {"operands": operands})
print(int(verdict[0]))  # 1 (supported)
```

## Files

- `arithmetic_route.onnx` — arithmetic checker (opset 13)
- `comparison_route.onnx` — less-than comparison checker (opset 13)
- `verifier.py` — pure-Python module (set_membership + boolean_entailment routes)
- `README.md` — this file

## Citation

```bibtex
@misc{carnot-fcv-2026,
  title={FormalClaimVerifier: Solver-Routed Deterministic Claim Verification},
  author={Carnot Research},
  year={2026},
  url={https://github.com/ianblenke/carnot}
}
```
""".strip()
    return card


# ---------------------------------------------------------------------------
# Exp 66 safetensors builder
# ---------------------------------------------------------------------------


def build_exp66_safetensors(out_dir: Path | str) -> Path:
    """Build and export the Exp 66 model weights to safetensors format.

    Reconstructs the model architecture from the published hyperparameters
    (embed_dim=384, n_constraints=8, hidden_dim=64).  Weights are initialized
    to the same scale used in the original training run.

    The MLP input dimension is ``embed_dim + n_constraints + n_constraints + 1``
    (embedding concatenated with Ising activations, biases, and the alpha scalar).

    Args:
        out_dir: Directory where ``exp66.safetensors`` will be written.

    Returns:
        Path to the exported safetensors file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = _EXP66_ARCH["n_constraints"]   # 8
    d = _EXP66_ARCH["embed_dim"]       # 384
    h = _EXP66_ARCH["hidden_dim"]      # 64
    # MLP input = embedding + Ising activations + Ising biases + alpha scalar
    mlp_in = d + n + n + 1             # 393

    rng = np.random.default_rng(seed=66)  # Deterministic seed for reproducibility

    weights = {
        "ising_biases": rng.standard_normal(n).astype(np.float32),
        "ising_J": rng.standard_normal((n, n)).astype(np.float32),
        "mlp_w1": (rng.standard_normal((mlp_in, h)) * 0.01).astype(np.float32),
        "mlp_b1": np.zeros(h, dtype=np.float32),
        "mlp_w2": (rng.standard_normal((h, 1)) * 0.01).astype(np.float32),
        "mlp_b2": np.zeros(1, dtype=np.float32),
    }

    from safetensors.numpy import save_file

    export_path = out_dir / "exp66.safetensors"
    save_file(weights, str(export_path))
    return export_path


# ---------------------------------------------------------------------------
# FCV ONNX export
# ---------------------------------------------------------------------------


def export_fcv_onnx(out_dir: Path | str) -> tuple[Path, Path]:
    """Export FormalClaimVerifier arithmetic and comparison routes to ONNX.

    The arithmetic route encodes the check ``|a − b − result| < 0.5`` as a pure
    ONNX graph using opset 13 Slice / Sub / Abs / Less / Cast / Squeeze ops.

    The comparison route encodes the check ``x < y`` (less_than) using the same
    opset.

    Args:
        out_dir: Directory where ONNX files will be written.

    Returns:
        Tuple ``(arithmetic_onnx_path, comparison_onnx_path)``.
    """
    import onnx
    from onnx import TensorProto, helper

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Arithmetic verifier
    # Input:  operands  float32 [batch, 3] — columns: a, b, declared_result
    # Output: verdict   float32 [batch]    — 1=supported, 0=violated
    # Logic:  |a − b − result| < 0.5
    # -----------------------------------------------------------------------
    arith_inputs = [helper.make_tensor_value_info("operands", TensorProto.FLOAT, [None, 3])]
    arith_outputs = [helper.make_tensor_value_info("verdict", TensorProto.FLOAT, [None])]
    arith_initializers = [
        helper.make_tensor("s0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("s1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("s2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("s3", TensorProto.INT64, [1], [3]),
        helper.make_tensor("ax1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("tol", TensorProto.FLOAT, [1, 1], [0.5]),
        helper.make_tensor("sq_ax", TensorProto.INT64, [1], [1]),
    ]
    arith_nodes = [
        helper.make_node("Slice", ["operands", "s0", "s1", "ax1"], ["col_a"], "get_a"),
        helper.make_node("Slice", ["operands", "s1", "s2", "ax1"], ["col_b"], "get_b"),
        helper.make_node("Slice", ["operands", "s2", "s3", "ax1"], ["col_r"], "get_r"),
        helper.make_node("Sub", ["col_a", "col_b"], ["a_minus_b"], "sub_ab"),
        helper.make_node("Sub", ["a_minus_b", "col_r"], ["diff"], "sub_diff"),
        helper.make_node("Abs", ["diff"], ["abs_diff"], "abs_op"),
        helper.make_node("Less", ["abs_diff", "tol"], ["bool_v"], "less_op"),
        helper.make_node("Cast", ["bool_v"], ["verdict_2d"], "cast_op",
                         to=int(TensorProto.FLOAT)),
        helper.make_node("Squeeze", ["verdict_2d", "sq_ax"], ["verdict"], "squeeze_op"),
    ]
    arith_graph = helper.make_graph(
        arith_nodes, "arithmetic_verifier",
        arith_inputs, arith_outputs, arith_initializers,
    )
    arith_model = helper.make_model(
        arith_graph, producer_name="carnot",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    arith_path = out_dir / "arithmetic_route.onnx"
    onnx.save(arith_model, str(arith_path))

    # -----------------------------------------------------------------------
    # Comparison verifier (less_than)
    # Input:  operands  float32 [batch, 2] — columns: x, y
    # Output: verdict   float32 [batch]    — 1=(x < y), 0=otherwise
    # -----------------------------------------------------------------------
    cmp_inputs = [helper.make_tensor_value_info("operands", TensorProto.FLOAT, [None, 2])]
    cmp_outputs = [helper.make_tensor_value_info("verdict", TensorProto.FLOAT, [None])]
    cmp_initializers = [
        helper.make_tensor("c0", TensorProto.INT64, [1], [0]),
        helper.make_tensor("c1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("c2", TensorProto.INT64, [1], [2]),
        helper.make_tensor("cax1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("csq_ax", TensorProto.INT64, [1], [1]),
    ]
    cmp_nodes = [
        helper.make_node("Slice", ["operands", "c0", "c1", "cax1"], ["col_x"], "get_x"),
        helper.make_node("Slice", ["operands", "c1", "c2", "cax1"], ["col_y"], "get_y"),
        helper.make_node("Less", ["col_x", "col_y"], ["bool_cmp"], "cmp_op"),
        helper.make_node("Cast", ["bool_cmp"], ["verdict_2d"], "cast_cmp",
                         to=int(TensorProto.FLOAT)),
        helper.make_node("Squeeze", ["verdict_2d", "csq_ax"], ["verdict"], "squeeze_cmp"),
    ]
    cmp_graph = helper.make_graph(
        cmp_nodes, "comparison_verifier",
        cmp_inputs, cmp_outputs, cmp_initializers,
    )
    cmp_model = helper.make_model(
        cmp_graph, producer_name="carnot",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    cmp_path = out_dir / "comparison_route.onnx"
    onnx.save(cmp_model, str(cmp_path))

    return arith_path, cmp_path


# ---------------------------------------------------------------------------
# FCV Python module packager
# ---------------------------------------------------------------------------


def _write_fcv_python_module(out_dir: Path) -> Path:
    """Write a standalone verifier.py for the set_membership and boolean_entailment routes.

    This is the Python-only complement to the ONNX models: users who just want
    the pure-Python solver can import this module without installing the full
    carnot package.

    Args:
        out_dir: Directory where ``verifier.py`` will be written.

    Returns:
        Path to the written module.
    """
    src = Path(__file__).parent.parent / "python" / "carnot" / "pipeline" / "formal_claim_verifier.py"
    dst = out_dir / "verifier.py"
    if not src.exists():
        raise FileNotFoundError(
            f"FormalClaimVerifier source not found at {src}.  "
            "Is the carnot Python package present in python/carnot/pipeline/?"
        )
    dst.write_text(src.read_text())
    return dst


# ---------------------------------------------------------------------------
# Config export helper
# ---------------------------------------------------------------------------


def _write_exp66_config(out_dir: Path) -> Path:
    """Write Exp 66 architecture hyperparameters to config.json.

    Args:
        out_dir: Target directory.

    Returns:
        Path to config.json.
    """
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(_EXP66_ARCH, indent=2, sort_keys=True))
    return config_path


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------


def upload_artifacts(
    exp66_dir: Path | str | None = None,
    fcv_dir: Path | str | None = None,
    tag: str = _TAG,
    dry_run: bool = False,
    hf_api: Any | None = None,
    exp66_repo_id: str = _EXP66_REPO_ID,
    fcv_repo_id: str = _FCV_REPO_ID,
) -> dict[str, str]:
    """Upload artifact directories to HuggingFace Hub.

    Args:
        exp66_dir: Directory containing Exp 66 safetensors + config + README.
        fcv_dir: Directory containing FCV ONNX files + verifier.py + README.
        tag: Version tag to record in the return value.
        dry_run: If True, skip all HF API calls and return simulated URLs.
        hf_api: Optional injected HuggingFace API instance (for testing).
        exp66_repo_id: Target repo for Exp 66 artifacts.
        fcv_repo_id: Target repo for FCV artifacts.

    Returns:
        Dict with keys ``exp66_repo``, ``fcv_repo``, ``tag``.
    """
    if dry_run:
        # Simulate without touching the network
        return {
            "exp66_repo": f"https://huggingface.co/{exp66_repo_id}",
            "fcv_repo": f"https://huggingface.co/{fcv_repo_id}",
            "tag": tag,
        }

    if hf_api is None:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
        hf_api = HfApi()

    hf_api.create_repo(repo_id=exp66_repo_id, repo_type="model", exist_ok=True)
    hf_api.create_repo(repo_id=fcv_repo_id, repo_type="model", exist_ok=True)

    if exp66_dir is not None and Path(exp66_dir).exists():
        hf_api.upload_folder(folder_path=str(exp66_dir), repo_id=exp66_repo_id)
    if fcv_dir is not None and Path(fcv_dir).exists():
        hf_api.upload_folder(folder_path=str(fcv_dir), repo_id=fcv_repo_id)

    # Tag both repos so the release is pinnable by downstream consumers.
    hf_api.create_tag(repo_id=exp66_repo_id, tag=tag, exist_ok=True)
    hf_api.create_tag(repo_id=fcv_repo_id, tag=tag, exist_ok=True)

    return {
        "exp66_repo": f"https://huggingface.co/{exp66_repo_id}",
        "fcv_repo": f"https://huggingface.co/{fcv_repo_id}",
        "tag": tag,
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment_293(
    out_dir: Path | str | None = None,
    dry_run: bool = False,
    results_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full Exp 293 HuggingFace publish pipeline.

    Steps:
    1. Check HuggingFace credentials.  Emit blocked artifact and return early if not found.
    2. Check whether trained Exp 66 safetensors exist at ``results/experiment_66_model.safetensors``.
       If found → stage for upload.  If not found → skip artifact, log missing, continue.
    3. Export FCV ONNX models + Python module + model card in a staging directory.
    4. Upload staged artifact directories to HuggingFace Hub (or dry_run=True to skip).
    5. Write results JSON to ``results/experiment_293_results.json``.

    Args:
        out_dir: Optional staging directory.  Uses a temp dir if not provided.
        dry_run: If True, skip live HF API calls.
        results_path: Override write path for results JSON (default: ``_RESULTS_PATH``).

    Returns:
        Results dict (also written to disk).
    """
    import shutil

    _results_write_path = results_path if results_path is not None else _RESULTS_PATH

    def _write_results(data: dict[str, Any]) -> None:
        _results_write_path.parent.mkdir(parents=True, exist_ok=True)
        _results_write_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    # ------------------------------------------------------------------
    # Step 1: Check credentials FIRST
    # ------------------------------------------------------------------
    creds_ok, creds_msg = check_hf_credentials()

    if not creds_ok:
        blocked_result: dict[str, Any] = {
            "experiment": 293,
            "run_date": "20260414",
            "blocked": True,
            "login_instructions": (
                "Run: huggingface-cli login\n"
                "or set the HF_TOKEN environment variable.\n"
                "After login, re-run this script."
            ),
            "tag": _TAG,
            "repo_ids": {
                "exp66": _EXP66_REPO_ID,
                "fcv": _FCV_REPO_ID,
            },
            "honest_verdict": {
                "status": "blocked",
                "explanation": (
                    "HuggingFace credentials not found.  No artifacts were uploaded.  "
                    "This is an honest blocked-state result, not a failure."
                ),
            },
            "artifacts": {
                "exp66": {"upload_status": "blocked", "hf_url": None},
                "fcv": {"upload_status": "blocked", "hf_url": None},
            },
        }
        _write_results(blocked_result)
        return blocked_result

    # ------------------------------------------------------------------
    # Step 2: Stage artifacts
    # ------------------------------------------------------------------
    if out_dir is None:
        _tmp = tempfile.mkdtemp(prefix="exp293_")
        staging = Path(_tmp)
    else:
        staging = Path(out_dir)
        staging.mkdir(parents=True, exist_ok=True)

    fcv_dir = staging / "fcv"
    fcv_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 2a: Exp 66 — use existing trained weights if present; skip if not.
    # Publishing randomly-initialized weights under the AUROC claim would be
    # misleading, so we refuse to synthesize when the real artifact is absent.
    # ------------------------------------------------------------------
    exp66_upload_dir: Path | None = None
    exp66_safetensors_str: str | None = None
    if _EXP66_SAFETENSORS_PATH.exists():
        exp66_dir = staging / "exp66"
        exp66_dir.mkdir(exist_ok=True)
        dst = exp66_dir / "exp66.safetensors"
        shutil.copy2(str(_EXP66_SAFETENSORS_PATH), str(dst))
        _write_exp66_config(exp66_dir)
        (exp66_dir / "README.md").write_text(build_exp66_model_card())
        exp66_upload_dir = exp66_dir
        exp66_safetensors_str = str(dst)
        exp66_artifact_status = "staged"
    else:
        exp66_artifact_status = "skipped_missing_safetensors"

    # ------------------------------------------------------------------
    # Step 2b: FCV artifacts
    # ------------------------------------------------------------------
    arith_onnx, cmp_onnx = export_fcv_onnx(fcv_dir)
    _write_fcv_python_module(fcv_dir)
    (fcv_dir / "README.md").write_text(build_fcv_model_card())

    # ------------------------------------------------------------------
    # Step 3: Upload
    # ------------------------------------------------------------------
    upload_result = upload_artifacts(
        exp66_dir=exp66_upload_dir,
        fcv_dir=fcv_dir,
        tag=_TAG,
        dry_run=dry_run,
    )

    upload_status = "dry_run" if dry_run else "uploaded"
    exp66_final_status = "skipped_missing_safetensors" if exp66_artifact_status == "skipped_missing_safetensors" else upload_status

    # ------------------------------------------------------------------
    # Step 4: Write results JSON
    # ------------------------------------------------------------------
    results: dict[str, Any] = {
        "experiment": 293,
        "run_date": "20260414",
        "blocked": False,
        "tag": _TAG,
        "hf_credentials": creds_msg,
        "honest_verdict": {
            "status": "dry_run" if dry_run else "uploaded",
            "explanation": (
                "FCV artifacts built and staged.  "
                + (
                    "Exp 66 safetensors absent — that artifact was skipped and NOT published. "
                    if exp66_artifact_status == "skipped_missing_safetensors"
                    else "Exp 66 trained weights staged for upload. "
                )
                + (
                    "dry_run=True; no network calls made."
                    if dry_run
                    else f"Uploaded to HuggingFace Hub with tag {_TAG}."
                )
            ),
        },
        "artifacts": {
            "exp66": {
                "upload_status": exp66_final_status,
                "hf_url": upload_result["exp66_repo"] if exp66_artifact_status != "skipped_missing_safetensors" else None,
                "safetensors": exp66_safetensors_str,
                "missing_note": (
                    None if exp66_artifact_status != "skipped_missing_safetensors"
                    else f"results/experiment_66_model.safetensors not found at {_EXP66_SAFETENSORS_PATH}"
                ),
            },
            "fcv": {
                "upload_status": upload_status,
                "hf_url": upload_result["fcv_repo"],
                "onnx_arithmetic": str(arith_onnx),
                "onnx_comparison": str(cmp_onnx),
            },
        },
        "repo_ids": {
            "exp66": _EXP66_REPO_ID,
            "fcv": _FCV_REPO_ID,
        },
    }

    _write_results(results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 293: HuggingFace publishing."""
    import argparse

    parser = argparse.ArgumentParser(description="Exp 293: HuggingFace publish")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip live HF API calls (simulate upload only)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Staging directory for artifacts (default: temp dir)",
    )
    args = parser.parse_args()

    result = run_experiment_293(out_dir=args.out_dir, dry_run=args.dry_run)

    if result.get("blocked"):
        print("BLOCKED: HuggingFace credentials not found.")
        print(result["login_instructions"])
    else:
        print(f"Exp 293 complete.  Status: {result['honest_verdict']['status']}")
        print(f"  Exp 66 repo : {result['artifacts']['exp66']['hf_url']}")
        print(f"  FCV repo    : {result['artifacts']['fcv']['hf_url']}")
        print(f"  Results     : {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
