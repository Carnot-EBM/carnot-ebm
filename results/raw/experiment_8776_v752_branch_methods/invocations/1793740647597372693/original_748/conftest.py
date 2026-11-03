import pytest
from pathlib import Path
from carnot.reporting import v748_collection_contract as api
from carnot.reporting import v749_contract_runner as runner
@pytest.fixture(autouse=True)
def archived_dependencies(monkeypatch):
    monkeypatch.setattr(api.m, "ROOT", Path('/tmp/exp8776-archives-_9x9f2jg/archive_748'))
    monkeypatch.setattr(api.m, "CLI", '/tmp/exp8776-archives-_9x9f2jg/archive_748/scripts/experiments/experiment_8723_v748_collection_contract.py')
from carnot.reporting.v751_branch_contract import output_binding
@pytest.fixture(autouse=True)
def archived_output_guard():
    with output_binding(Path('/tmp/exp8776-archives-_9x9f2jg/archive_748')):
        yield
