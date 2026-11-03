import pytest
from pathlib import Path
from carnot.reporting import v749_contract_methods as api
from carnot.reporting import v749_contract_runner as runner
@pytest.fixture(autouse=True)
def archived_dependencies(monkeypatch):
    monkeypatch.setattr(api, "ROOT", Path('/tmp/exp8776-archives-_9x9f2jg/archive_749'))
    monkeypatch.setattr(runner, "CLI", '/tmp/exp8776-archives-_9x9f2jg/archive_749/scripts/experiments/experiment_8737_v749_contract_methods.py')
from carnot.reporting.v751_branch_contract import output_binding
@pytest.fixture(autouse=True)
def archived_output_guard():
    with output_binding(Path('/tmp/exp8776-archives-_9x9f2jg/archive_749')):
        yield
