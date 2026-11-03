import pytest
from pathlib import Path
from carnot.reporting import v748_collection_contract as api
from carnot.reporting import v749_contract_runner as runner
@pytest.fixture(autouse=True)
def archived_dependencies(monkeypatch):
    monkeypatch.setattr(api.m, "ROOT", Path('/tmp/exp8754-private-8e0m0xz5/748'))
    monkeypatch.setattr(api.m, "CLI", '/tmp/exp8754-private-8e0m0xz5/748/scripts/experiments/experiment_8723_v748_collection_contract.py')
