#!/usr/bin/env python3
import sys
import os

# Ensure the python directory is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from carnot.research.experiment_3841 import ResearchRefresh3841

def main():
    refresh = ResearchRefresh3841()
    try:
        artifact = refresh.generate_artifact()
        print(artifact["honest_verdict"])
        if artifact["honest_verdict"].startswith("complete:"):
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"blocked_exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
