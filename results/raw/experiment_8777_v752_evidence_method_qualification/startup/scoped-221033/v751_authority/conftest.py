import os
from pathlib import Path
def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.is_relative_to(Path('/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8777_v752_evidence_method_qualification/startup/scoped-221033/v751_authority')):
            item._nodeid = os.path.relpath(item.path, '/home/ianblenke/github.com/ianblenke/carnot') + '::' + item.name
