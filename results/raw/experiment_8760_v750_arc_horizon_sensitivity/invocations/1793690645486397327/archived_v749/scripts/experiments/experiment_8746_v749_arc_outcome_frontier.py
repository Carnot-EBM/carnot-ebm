from pathlib import Path
import sys
sys.path[:0] = ['/home/ianblenke/github.com/ianblenke/carnot/python', '/home/ianblenke/github.com/ianblenke/carnot']
from carnot.reporting.arc_horizon_sensitivity_8760 import historical_frontier_main
raise SystemExit(historical_frontier_main(Path('/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8760_v750_arc_horizon_sensitivity/invocations/1793690645486397327/archived_v749')))
