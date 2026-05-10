import pytest
import json
import os
from pathlib import Path
from carnot.models.kan.glorokan_robustness import GloroKANBounder
from carnot.pipeline.eidoku_gate import EidokuGate
# We need to figure out what FR-11 is represented by.
