#!/usr/bin/env python3
"""Experiment 236: explicit code-spec corpus from checked-in HumanEval traces.

Writes:
- ``data/research/code_spec_corpus_236.jsonl``
- ``results/experiment_236_results.json``

Spec: REQ-CODE-023, REQ-CODE-024,
SCENARIO-CODE-020, SCENARIO-CODE-021
"""

from __future__ import annotations

from carnot.pipeline.code_spec_corpus import main


if __name__ == "__main__":
    raise SystemExit(main())
