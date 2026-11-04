import sys
from carnot.reporting import v752_branch_methods as api
api.DESIGN = 'openspec/change-proposals/research-roadmap-v752-preserved-20261103.md'
from carnot.reporting import v752_branch_runner as runner
runner.terminal.CLI = __file__
raise SystemExit(runner.main(sys.argv[1:] + ['--root', '/home/ianblenke/github.com/ianblenke/carnot/results/raw/carnot-v753-recheck-qs7qlka3/raw/experiment_8789_v753_branch_qualification/invocations/1793774193674346292/unit_authority']))
