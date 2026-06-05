import sys
from pathlib import Path

# Ensure the carnot package is importable
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "python"))

from carnot.verify.experiment_3826_fover_ablation_faithful import main

if __name__ == "__main__":
    main()
