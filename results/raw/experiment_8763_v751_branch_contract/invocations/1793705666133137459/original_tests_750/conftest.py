import pytest
from pathlib import Path
from carnot.reporting import v750_archive_bound as a, v750_archive_runner as r
@pytest.fixture(autouse=True)
def archived_dependencies(monkeypatch):
    monkeypatch.setattr(a, "ROOT", Path('/tmp/v751-original750-lyr1afji/archive'))
    monkeypatch.setattr(r.terminal_runner, "CLI", '/tmp/v751-original750-lyr1afji/archive/scripts/experiments/experiment_8749_v750_archive_bound_contract.py')
