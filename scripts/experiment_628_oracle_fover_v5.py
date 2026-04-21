#!/usr/bin/env python3
"""Experiment 628: ORACLE FOVER v5 Corpus Builder.

**Context (RETRO-066):**
    JEPA v13 ECE=0.207 (target <0.10).  All training data came from synthetic violations
    or binary correct/incorrect response labels — not step-level constraint labels that
    match live LLM output style.  ORACLE (arXiv 2603.21140, AAAI 2026) generates
    multi-step reasoning data where EACH STEP has a symbolic verification label derived
    from the same model that will later be verified.  This closes the offline/live
    distribution gap by construction.

**What this experiment produces:**
    results/fover_corpus_v5_oracle.json — every live response in fover_corpus_v5 and
    live_pairs_578 annotated with per-step violation labels from SymCodeVerifier.
    The corpus is the training data source for JEPA v14.

Spec: REQ-DATA-012, REQ-DATA-013,
      SCENARIO-DATA-019, SCENARIO-DATA-020
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.oracle_corpus_builder import OracleCorpusBuilder
from carnot.pipeline.symcode_verifier import SymCodeVerifier
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
_log = logging.getLogger(__name__)


def main() -> None:
    # Step 1: self-inject CARNOT_FORCE_LIVE if GPU is present but var was absent.
    apply_env_autofix()

    # Step 2: arm watchdog — 45 minute hard limit.
    ExperimentTimeoutWatchdog(628, timeout_minutes=45)

    # Step 3: standard experiment scaffolding (requires_gpu=False: runs on CPU).
    tmpl = ExperimentTemplate(
        628,
        "ORACLE FOVER v5 Corpus Builder",
        "results/experiment_628_oracle_fover_v5.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # Step 4: optionally load a live LLM caller for higher SymCode extraction accuracy.
    # Use CARNOT_ORACLE_LIVE=1 to explicitly opt into LLM loading for corpus building.
    # CARNOT_FORCE_LIVE is used only for GPU gating; LLM loading here is opt-in via a
    # separate flag to avoid accidentally loading a model during fast corpus-only runs.
    llm_caller = None
    force_live = os.environ.get("CARNOT_ORACLE_LIVE", "0") == "1"
    if force_live:
        _log.info("CARNOT_FORCE_LIVE=1: loading Qwen3.5-0.8B on CPU for SymCode extraction")
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415

            _pipe = hf_pipeline(
                "text-generation",
                model="Qwen/Qwen3.5-0.8B",
                device="cpu",
                max_new_tokens=64,
            )

            def _llm(prompt: str) -> str:
                out = _pipe(prompt, return_full_text=False)
                return out[0]["generated_text"] if out else ""

            llm_caller = _llm
        except Exception as exc:
            _log.warning("Could not load Qwen3.5-0.8B (%s); using regex fallback", exc)

    # Step 5: load and merge live pairs from all available sources.
    # Deduplication is by question_id (question_index as str, or hash of question).
    sources = [
        _REPO_ROOT / "results" / "live_pairs_578.json",
        _REPO_ROOT / "results" / "fover_corpus_v5.json",
    ]
    seen_ids: set[str] = set()
    live_pairs: list[dict] = []

    for src in sources:
        if not src.exists():
            _log.warning("Source not found, skipping: %s", src)
            continue
        _log.info("Loading %s", src)
        raw = json.loads(src.read_text())

        # fover_corpus_v5.json wraps pairs under a 'pairs' key.
        rows: list[dict] = raw if isinstance(raw, list) else raw.get("pairs", [])

        for row in rows:
            q_id = str(
                row.get("question_id")
                or row.get("question_index")
                or hash(row.get("question", ""))
            )
            if q_id in seen_ids:
                continue
            seen_ids.add(q_id)
            # Normalise: ensure 'model_id' is present (some sources use 'model').
            row.setdefault("model_id", row.get("model", "unknown"))
            live_pairs.append(row)

    _log.info("Total unique live pairs loaded: %d", len(live_pairs))

    # Step 6: build SymCodeVerifier and OracleCorpusBuilder.
    verifier = SymCodeVerifier(llm_caller=llm_caller)
    builder = OracleCorpusBuilder(verifier)

    # Step 7: label every chain.
    chains = builder.build_corpus(live_pairs)

    # Step 8: compute statistics.
    n_chains = len(chains)
    n_with_violation = sum(c.has_violation for c in chains)
    n_without_violation = n_chains - n_with_violation
    n_total_steps = sum(len(c.step_labels) for c in chains)
    n_violated_steps = sum(c.n_violated_steps for c in chains)
    step_violation_rate = n_violated_steps / max(n_total_steps, 1)

    _log.info(
        "Corpus stats: n_chains=%d  with_violation=%d  total_steps=%d  "
        "violated_steps=%d  step_violation_rate=%.3f",
        n_chains,
        n_with_violation,
        n_total_steps,
        n_violated_steps,
        step_violation_rate,
    )

    # Step 9: serialise and write oracle corpus JSON.
    oracle_path = _REPO_ROOT / "results" / "fover_corpus_v5_oracle.json"
    chains_dicts = [dataclasses.asdict(c) for c in chains]
    oracle_path.write_text(json.dumps(chains_dicts, indent=2))
    _log.info("Wrote oracle corpus to %s (%d chains)", oracle_path, n_chains)

    # Step 10: build and write experiment artifact.
    corpus_ready = n_chains >= 100
    honest_verdict = "oracle_corpus_ready" if n_chains >= 100 else "oracle_corpus_partial"

    artifact = tmpl.build_result(
        {
            "schema": "carnot.oracle_fover_v5.v1",
            "n_chains": n_chains,
            "n_with_violation": n_with_violation,
            "n_without_violation": n_without_violation,
            "n_total_steps": n_total_steps,
            "n_violated_steps": n_violated_steps,
            "step_violation_rate": step_violation_rate,
            "oracle_corpus_path": str(oracle_path.relative_to(_REPO_ROOT)),
            "corpus_ready": corpus_ready,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    out_path = _REPO_ROOT / "results" / "experiment_628_oracle_fover_v5.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(artifact, f, indent=2)
    _log.info("Artifact written to %s", out_path)

    # FINAL: assert deliverable was written (raises if missing/empty).
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
