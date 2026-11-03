import pytest
from pathlib import Path
from carnot.reporting import v748_collection_contract as api
from carnot.reporting import v749_contract_runner as runner
@pytest.fixture(autouse=True)
def archived_dependencies(monkeypatch):
    monkeypatch.setattr(api.m, "ROOT", Path('/tmp/exp8749-fixture-7bcbnlc6/archive_748'))
    monkeypatch.setattr(api.m, "CLI", '/tmp/exp8749-fixture-7bcbnlc6/archive_748/scripts/experiments/experiment_8723_v748_collection_contract.py')
