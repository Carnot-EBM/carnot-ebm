#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure python/ is in sys.path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "python"))

from carnot.reporting.experiment_1924_retro import generate_retro

if __name__ == "__main__":
    generate_retro("results/experiment_1924_retro.json")
    print("Generated results/experiment_1924_retro.json")
