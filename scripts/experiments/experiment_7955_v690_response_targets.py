"""REQ-REPORT-7955: direct host-only response target transport."""

from pathlib import Path
import sys

# Resolve both public consumers even when replay starts outside the checkout.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.experiment_7955_v690_response_targets import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
