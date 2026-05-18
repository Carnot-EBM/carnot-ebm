import pytest
from carnot.verify.hive_ensemble_v4 import HiveEnsembleV4Detector

def test_hive_ensemble_v4_init():
    detector = HiveEnsembleV4Detector()
    assert detector is not None
