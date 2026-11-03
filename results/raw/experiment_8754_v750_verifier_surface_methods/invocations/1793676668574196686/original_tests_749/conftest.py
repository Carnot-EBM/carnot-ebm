import pytest
from pathlib import Path
from carnot.reporting import v749_contract_methods as api
from carnot.reporting import v749_contract_runner as runner
@pytest.fixture(autouse=True)
def archived_dependencies(monkeypatch):
    monkeypatch.setattr(api, "ROOT", Path('/tmp/exp8754-private-8e0m0xz5/749'))
    monkeypatch.setattr(runner, "CLI", '/tmp/exp8754-private-8e0m0xz5/749/scripts/experiments/experiment_8737_v749_contract_methods.py')
