#!/bin/bash
# Sequential four-arm supervisor. ONE server per arm, on ONE card, never two at once:
# 24 GiB cannot hold two 13.5 GiB servers, and the prior lane's RUN_LOG records a leaked
# server on port 8953 blocking the next arm for 600s. Each attempt gets a FRESH port so a
# leaked listener from a crashed attempt can never be silently reused as a healthy server.
V=/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python
D=results/arc_wm_four_arm_20260727
GPU=${GPU:-1}
BUDGET=${BUDGET:-200}
VARIANTS=${VARIANTS:-1}
GAMES=${GAMES:-}
SUF=${SUF:-}
PORTBASE=${PORTBASE:-9200}
i=0
for arm in wm_A0_control wm_A1_mask wm_A2_gate wm_A3_both; do
  i=$((i+1))
  for attempt in 1 2 3; do
    if [ -f $D/run_${arm}${SUF}.json ]; then echo "=== $arm complete ==="; break; fi
    port=$((PORTBASE + i*10 + attempt))
    n=$(ls $D/cells/ 2>/dev/null | grep -c "^${arm}__")
    echo "=== $(date -u +%H:%M:%SZ) $arm attempt $attempt: $n cells banked, port $port ==="
    if [ -n "$GAMES" ]; then
      $V -u $D/fourarm.py --arm $arm --k 4 --variants $VARIANTS --budget $BUDGET \
         --port $port --gpu $GPU --games "$GAMES" --tag "$SUF"
    else
      $V -u $D/fourarm.py --arm $arm --k 4 --variants $VARIANTS --budget $BUDGET \
         --port $port --gpu $GPU --tag "$SUF"
    fi
    echo "=== $arm attempt $attempt returned (exit $?) ==="
    sleep 10
  done
done
echo "=== CHAIN${SUF} DONE $(date -u +%H:%M:%SZ) ==="
for arm in wm_A0_control wm_A1_mask wm_A2_gate wm_A3_both; do
  echo "$arm: $(ls $D/cells/ 2>/dev/null | grep -c "^${arm}__") cells"
done
