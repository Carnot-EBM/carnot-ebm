#!/usr/bin/env python3
"""Write the Exp 2925 code-hallucination taxonomy provenance corrigendum."""

from __future__ import annotations

import json

from carnot.eval.code_hallucination_taxonomy_provenance_corrigendum import (
    DEFAULT_OUTPUT_PATH,
    write_artifact,
)


def main() -> None:
    output_path = write_artifact()
    print(json.dumps({"artifact_path": str(DEFAULT_OUTPUT_PATH), "written": output_path.exists()}))


if __name__ == "__main__":
    main()
