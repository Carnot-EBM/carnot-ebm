#!/usr/bin/env python3
"""Write the Exp 2911 deterministic code-hallucination taxonomy artifact."""

from __future__ import annotations

import json

from carnot.eval.code_hallucination_taxonomy_verifier import run_experiment


def main() -> None:
    artifact = run_experiment()
    print(json.dumps({"honest_verdict": artifact["honest_verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
