from pathlib import Path
import pytest
from carnot.reporting.evidence_archive_8777 import configured
@pytest.fixture(autouse=True)
def original_authority():
    with configured(Path('/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8777_v752_evidence_method_qualification/startup/scoped-221033/v751_authority')):
        yield
