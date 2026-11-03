import pytest
from pathlib import Path
from carnot.reporting import v750_archive_bound as a, v750_archive_runner as r
@pytest.fixture(autouse=True)
def archived_dependencies(monkeypatch):
    monkeypatch.setattr(a, "ROOT", Path('/tmp/exp8776-archives-4uqoo68f/archive_750'))
    monkeypatch.setattr(r.terminal_runner, "CLI", '/tmp/exp8776-archives-4uqoo68f/archive_750/scripts/experiments/experiment_8749_v750_archive_bound_contract.py')
