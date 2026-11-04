import pytest
from pathlib import Path
from carnot.reporting import v753_branch_qualification as api
@pytest.fixture(autouse=True)
def historical_bindings(monkeypatch):
    snapshot, freeze = api.snapshot, api.freeze
    root = Path('/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8801_v754_repaired_memory_qualification/invocations/1793794575899961572/v753_authority')
    monkeypatch.setattr(api, 'snapshot', lambda source, target: snapshot(root if source == api.ROOT else source, target))
    monkeypatch.setattr(api, 'freeze', lambda source, raw, date: freeze(root if source == api.ROOT else source, raw, date))
    monkeypatch.setattr(api, 'CLI', '/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8801_v754_repaired_memory_qualification/invocations/1793794575899961572/repaired_family/original_cli.py')
