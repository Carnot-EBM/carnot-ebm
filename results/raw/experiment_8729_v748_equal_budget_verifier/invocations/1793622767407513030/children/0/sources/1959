"""Experiment 5718: build the offline ARC exploration-playbook embedding index
(the static asset behind the retrieval-based Phase-3 injection, REQ-ARC-WMTE-5718).

WHAT THIS BUILDS
----------------
A shippable static asset under models/arc_playbook_index/:
  - index.json      : {model, dim, patterns:[{pattern_id, theme, statement,
                       mechanic_tags, source_games, citation}...]} -- the "graph"
                       (pattern -> mechanic-tags -> corpus citations).
  - embeddings.npy  : (N, dim) float32; row i is the embedding of patterns[i].statement,
                       computed with the LIVE Qwen3.5-9B GGUF so the vector space MATCHES
                       the query embedding the live agent extracts at stall time.
  - kit_reference.json : a compact AST-derived reference (signature + one-line docstring,
                       NOT full bodies) of the Phase-2 arc_solver_kit exploration primitives
                       -- the same "compact structural summary instead of full text"
                       principle applied to code.

SUBSTRATE
---------
inference_substrate = live_llm_embedding_extraction (CLAUDE.md): loads the real GGUF and
extracts LAST-token pooled embeddings via llama_cpp.Llama(embedding=True), a single forward
pass per pattern (no autoregressive generation), 2.0s duration floor. If the GGUF is missing
it emits a blocked_* verdict rather than fabricating.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from carnot.agentic.arc_playbook_patterns import playbook_patterns
from carnot.agentic.arc_playbook_retrieval import validate_mechanic_tags

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "models" / "arc_playbook_index"
KIT_PATH = REPO_ROOT / "python" / "carnot" / "agentic" / "arc_solver_kit.py"
# The Phase-2 exploration primitives to summarize into the compact kit reference.
KIT_PRIMITIVES = (
    "probe_action_semantics",
    "read_absolute_trajectory",
    "find_unexplained_glyphs",
    "bounded_reachability_search",
    "bisect_death_prefix",
    "settled_grid",
)

RANDOM_SEED = 5718
INFERENCE_SUBSTRATE = "live_llm_embedding_extraction"
MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "playbook-pattern embedder (matches the live query-embedding space)",
        "quant": "Q4_K_M",
    }
]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed self-declared state (Verdict Terminal-Prefix)."
    },
    "inference_substrate": {
        "principle": "live_llm_embedding_extraction -> adversarial_verify applies the 2.0s "
        "embedding-forward-pass floor; real GGUF load + real forward passes, no generation."
    },
    "preconditions_checked": {
        "principle": "records the GGUF was verified before any load; pre-empts silent-missing "
        "fabrication (Pre-Launch Preconditions)."
    },
    "random_seed": {"principle": "harness determinism precondition."},
    "reproducibility_checksum": {
        "principle": "content hash over patterns + embeddings catches drift."
    },
    "n_patterns": {"principle": "count of embedded patterns; must equal the embeddings row count."},
    "embedding_dim": {"principle": "vector dimension; the live query embedding must match it."},
}


def _one_line_doc(node: ast.FunctionDef) -> str:
    doc = ast.get_docstring(node) or ""
    return doc.strip().splitlines()[0].strip() if doc.strip() else ""


def _signature(node: ast.FunctionDef) -> str:
    """Reconstruct a compact `name(args) -> ret` signature from the AST (no body)."""
    src = ast.unparse(node.args)
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{node.name}({src}){ret}"


def build_kit_reference(
    kit_path: Path = KIT_PATH, names: tuple[str, ...] = KIT_PRIMITIVES
) -> list[dict[str, Any]]:
    """AST-derive signature + one-line docstring (NOT bodies) for the named kit primitives."""
    tree = ast.parse(kit_path.read_text())
    wanted = set(names)
    out: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            out.append(
                {"name": node.name, "signature": _signature(node), "doc": _one_line_doc(node)}
            )
    out.sort(key=lambda r: names.index(r["name"]) if r["name"] in names else 999)
    return out


def _resolve_gguf() -> Optional[str]:
    from carnot.agentic.arc_executable_world_model import _resolve_gguf

    return _resolve_gguf("Qwen3.5-9B-MTP") or _resolve_gguf("Qwen3.5-9B")


def preconditions() -> dict[str, Any]:
    gguf = _resolve_gguf()
    return {
        "gguf_path": gguf,
        "preconditions_checked": [{"resource": "qwen3.5_9b_gguf_cached", "available": bool(gguf)}],
    }


def _embed_statements(gguf_path: str, statements: list[str]) -> np.ndarray:
    from llama_cpp import Llama
    from llama_cpp.llama_cpp import LLAMA_POOLING_TYPE_LAST

    llm = Llama(
        model_path=gguf_path,
        embedding=True,
        pooling_type=LLAMA_POOLING_TYPE_LAST,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    vecs: list[np.ndarray] = []
    for text in statements:
        raw = llm.embed(text, normalize=False, truncate=True)
        arr = np.asarray(raw, dtype=np.float32)
        vecs.append(arr if arr.ndim == 1 else arr.reshape(-1))
    return np.vstack(vecs)


def _checksum(patterns: list[dict[str, Any]], embeddings: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(patterns, sort_keys=True).encode())
    h.update(np.ascontiguousarray(embeddings.astype(np.float32)).tobytes())
    return "sha256:" + h.hexdigest()


def build(index_dir: Path = INDEX_DIR) -> dict[str, Any]:
    started = time.time()
    patterns = [p.as_dict() for p in playbook_patterns()]
    bad_tags = validate_mechanic_tags(patterns)
    base: dict[str, Any] = {
        "experiment": "exp5718-playbook-index-build",
        "req": "REQ-ARC-WMTE-5718",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "random_seed": RANDOM_SEED,
        "field_provenance": FIELD_PRINCIPLES,
        "n_patterns": len(patterns),
        "invalid_mechanic_tags": bad_tags,
    }
    if bad_tags:
        base["honest_verdict"] = f"complete: blocked_invalid_mechanic_tags_{'_'.join(bad_tags)}"
        base["duration_s"] = round(time.time() - started, 3)
        base["reproducibility_checksum"] = _checksum(patterns, np.zeros((0, 0), dtype=np.float32))
        return base

    preconds = preconditions()
    base["preconditions_checked"] = preconds["preconditions_checked"]
    base["gguf_path"] = preconds["gguf_path"]
    if not preconds["gguf_path"]:
        base["honest_verdict"] = "complete: blocked_qwen3.5_9b_gguf_cached"
        base["duration_s"] = round(time.time() - started, 3)
        base["reproducibility_checksum"] = _checksum(patterns, np.zeros((0, 0), dtype=np.float32))
        return base

    statements = [p["statement"] for p in patterns]
    embeddings = _embed_statements(preconds["gguf_path"], statements)
    dim = int(embeddings.shape[1])
    kit_reference = build_kit_reference()

    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "index.json").write_text(
        json.dumps(
            {
                "model": "Qwen3.5-9B-MTP-Q4_K_M",
                "dim": dim,
                "embed_pooling": "last_token",
                "built_by": "experiment_5718",
                "patterns": patterns,
            },
            indent=2,
        )
    )
    np.save(index_dir / "embeddings.npy", embeddings.astype(np.float32))
    (index_dir / "kit_reference.json").write_text(json.dumps(kit_reference, indent=2))

    base.update(
        {
            "honest_verdict": "complete_playbook_index_built",
            "embedding_dim": dim,
            "kit_reference_count": len(kit_reference),
            "index_dir": str(index_dir.relative_to(REPO_ROOT)),
            "reproducibility_checksum": _checksum(patterns, embeddings),
            "duration_s": round(time.time() - started, 3),
            "verifier_is_oracle": False,
        }
    )
    return base


def main() -> None:
    result = build()
    out = REPO_ROOT / "results" / "experiment_5718_playbook_index_build.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"verdict: {result.get('honest_verdict')}")
    print(
        f"n_patterns={result.get('n_patterns')} dim={result.get('embedding_dim')} "
        f"kit_ref={result.get('kit_reference_count')} dur={result.get('duration_s')}s"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
