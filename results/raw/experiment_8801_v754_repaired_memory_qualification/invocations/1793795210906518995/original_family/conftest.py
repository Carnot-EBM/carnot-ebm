import pytest
from carnot.reporting import v752_branch_methods as api
from carnot.reporting import v752_branch_runner as runner
@pytest.fixture(autouse=True)
def historical_bindings(monkeypatch):
    monkeypatch.setattr(api, 'DESIGN', 'openspec/change-proposals/research-roadmap-v752-preserved-20261103.md')
    monkeypatch.setattr(api, 'CLI', '/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8801_v754_repaired_memory_qualification/invocations/1793795210906518995/original_family/original_cli.py')
    monkeypatch.setattr(runner.terminal, 'CLI', '/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8801_v754_repaired_memory_qualification/invocations/1793795210906518995/original_family/original_cli.py')
