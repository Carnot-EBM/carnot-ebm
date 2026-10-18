"""REQ-VERIFY-8403: direct execution uses the bounded no-model custody runner."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from carnot.reporting.released_question_identity_runner_8403 import main

if __name__ == "__main__":
    raise SystemExit(main())
