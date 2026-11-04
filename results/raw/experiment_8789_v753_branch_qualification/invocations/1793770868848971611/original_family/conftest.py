import pytest
from carnot.reporting import v752_branch_methods as api
@pytest.fixture(autouse=True)
def historical_bindings(monkeypatch):
    monkeypatch.setattr(api, 'DESIGN', 'openspec/change-proposals/research-roadmap-v752-preserved-20261103.md')
    monkeypatch.setattr(api, 'CLI', '/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8789_v753_branch_qualification/invocations/1793770868848971611/original_family/original_cli.py')
