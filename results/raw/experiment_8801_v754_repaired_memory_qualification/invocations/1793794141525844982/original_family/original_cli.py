import sys
from carnot.reporting import v752_branch_methods as api
api.DESIGN = 'openspec/change-proposals/research-roadmap-v752-preserved-20261103.md'
from carnot.reporting import v752_branch_runner as runner
runner.terminal.CLI = __file__
raise SystemExit(runner.main(sys.argv[1:] + ['--root', '/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8801_v754_repaired_memory_qualification/invocations/1793794141525844982/unit_authority']))
