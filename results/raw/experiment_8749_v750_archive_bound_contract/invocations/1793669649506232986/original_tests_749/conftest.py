import pytest
from pathlib import Path
from carnot.reporting import v749_contract_methods as api
from carnot.reporting import v749_contract_runner as runner
@pytest.fixture(autouse=True)
def archived_dependencies(monkeypatch):
    monkeypatch.setattr(api, "ROOT", Path('/tmp/exp8749-fixture-7bcbnlc6/archive_749'))
    monkeypatch.setattr(runner, "CLI", '/tmp/exp8749-fixture-7bcbnlc6/archive_749/scripts/experiments/experiment_8737_v749_contract_methods.py')
