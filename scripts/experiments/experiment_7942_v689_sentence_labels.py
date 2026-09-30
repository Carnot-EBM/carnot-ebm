"""REQ-REPORT-7942: run the host-only sentence annotation transport."""

from pathlib import Path
import sys

# Direct execution must resolve repository-owned publication readers before
# importing the producer, regardless of the caller's working directory.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.experiment_7942_v689_sentence_labels import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
