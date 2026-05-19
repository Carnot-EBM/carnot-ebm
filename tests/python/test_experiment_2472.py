import sys
from pathlib import Path
import pytest

# Add the scripts directory to path to import the experiment script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.experiment_2472_llm_judge import extract_float

def test_extract_float():
    assert extract_float("Answer: YES. Confidence: 0.9", 0.5) == 0.9
    assert extract_float("YES. Confidence: 1.0", 0.5) == 1.0
    assert extract_float("NO. Confidence: 0.0", 0.5) == 0.0
    assert extract_float("Confidence: 0.42", 0.5) == 0.42
    assert extract_float("YES 0.8", 0.5) == 0.8
    assert extract_float("NO 0.2", 0.5) == 0.2
    assert extract_float("I am not sure.", 0.5) == 0.5
