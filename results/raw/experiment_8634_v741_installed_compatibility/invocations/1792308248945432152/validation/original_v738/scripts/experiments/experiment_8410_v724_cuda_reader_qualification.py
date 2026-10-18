"""REQ-REPORT-8410: route the actual invocation to bounded CUDA reader qualification."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.verify.cuda_reader_runner_8410 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
