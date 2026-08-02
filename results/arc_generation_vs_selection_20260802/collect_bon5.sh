#!/usr/bin/env bash
set -uo pipefail
cd /home/ianblenke/github.com/ianblenke/carnot
export CARNOT_REPO=$PWD
export SCRATCH_E3=/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/e3_scratch
export GVS_ENGINE_TIMEOUT_S=45
D=results/arc_generation_vs_selection_20260802
mkdir -p $D/out/cells_bestofn
L=$D/out/collect_bon5.log
# ft09 is EXCLUDED and the exclusion is recorded, not silent: its worker exceeded a 2700s
# game budget twice and was reaped. A game that cannot be scored inside its budget is
# DROPPED with its reason, never scored 0.
for g in lp85 sc25 tn36 tu93 vc33; do
  # extract this game's frozen completions to .py, then score
  .venv/bin/python - "$g" >> "$L" 2>&1 <<'PY'
import json, os, pathlib, sys
sys.path.insert(0, "python")
from carnot.agentic import arc_executable_world_model as e3
g=sys.argv[1]
BON=pathlib.Path("results/arc_induce_bestofn_20260731")
CODE=pathlib.Path(os.environ["SCRATCH_E3"])/"bon_code"; CODE.mkdir(parents=True, exist_ok=True)
sc=json.loads((BON/"bestofn_scored.json").read_text())
jobs=[]
for c in sc["candidates"]:
    if c["game"]!=g: continue
    tag=c.get("tag") or "gpu1"
    t=BON/"harness"/"bon"/tag/f"{g}_k{c['candidate']}.txt"
    if not t.exists(): continue
    cp=CODE/f"{g}_k{c['candidate']}.py"
    if not cp.exists(): cp.write_text(e3._extract_python(t.read_text(errors="replace")) or "")
    jobs.append({"cell": f"{g}__k{c['candidate']}", "path": str(cp)})
pathlib.Path(f"results/arc_generation_vs_selection_20260802/out/.bonjobs_{g}.json").write_text(json.dumps(jobs))
print(f"prepared {g}: {len(jobs)} candidates")
PY
  timeout 900 .venv/bin/python $D/bestofn_worker.py "$g" \
      "$D/out/.bonjobs_$g.json" "$D/out/cells_bestofn/$g.json" >> "$L" 2>&1 \
      || echo "DROPPED $g (timeout/error, recorded not zeroed)" >> "$L"
  echo "-- done $g $(date -u +%FT%TZ)" >> "$L"
done
echo "BON5 ALL DONE" >> "$L"
