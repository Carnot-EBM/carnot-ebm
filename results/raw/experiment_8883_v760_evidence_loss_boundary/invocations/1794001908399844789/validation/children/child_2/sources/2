"""REQ-REPORT-8883: a separately installed verifier receives visible records alone."""

import hashlib
import json
from pathlib import Path
import sys

import carnot.verify.size_aware_witness as installed


def main(argv: list[str]) -> None:
    """Return deterministic exact decisions and imported bytes without receiving source truth."""
    payloads = json.loads(Path(argv[0]).read_bytes())["payloads"]
    rows = []
    for i, payload in enumerate(payloads):
        result = installed.verify_size_aware_witness(payload, cap=200000)
        rows.append({k: result[k] for k in ("decision", "budget_exhausted")})
        if (i + 1) % 30 == 0:
            print(f"[exp8883-worker] completed={i + 1} pending={len(payloads) - i - 1}", flush=True)
    module = Path(installed.__file__).resolve()
    Path(argv[1]).write_text(
        json.dumps(
            dict(
                rows=rows,
                imported_path=str(module),
                imported_sha256="sha256:" + hashlib.sha256(module.read_bytes()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
