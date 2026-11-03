import pytest
from pathlib import Path
from carnot.reporting import arc_outcome_frontier_8746 as binding
@pytest.fixture(scope="module", autouse=True)
def archived_dependencies():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(binding.e, "ROOT", Path('/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8760_v750_arc_horizon_sensitivity/invocations/1793690645486397327/archived_v749'))
        patch.setattr(binding.e, "CLI", '/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8760_v750_arc_horizon_sensitivity/invocations/1793690645486397327/archived_v749/scripts/experiments/experiment_8746_v749_arc_outcome_frontier.py')
        yield
