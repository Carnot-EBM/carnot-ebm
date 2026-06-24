#!/usr/bin/env bash
# Run the slow online exp4710 arms SEQUENTIALLY (full CPU, no contention) so they do not
# torch-segfault under the 4-way parallel CPU collision that killed the parallel sweep at ~32min.
# Each arm runs ~15-20min solo. frozen + online-warm already have valid artifacts from the smoke.
set -u
cd /home/ianblenke/github.com/ianblenke/carnot
unset CARNOT_ARC_ONLINE_BUDGET
export PYTHONPATH=python
ARMS="${1:-online-scratch online-warm-propose}"
for arm in $ARMS; do
  log="/tmp/exp4710_seq_${arm//-/_}.log"
  echo "START $arm $(date -u +%FT%TZ)" > "$log"
  CARNOT_ARC_ONLINE_ARM="$arm" nice -n 10 .venv/bin/python -m carnot.experiment_4710_online_action_learning_arms >> "$log" 2>&1
  echo "EXIT_${arm}=$? $(date -u +%FT%TZ)" >> "$log"
done
echo "SEQ_ALL_DONE $(date -u +%FT%TZ)" > /tmp/exp4710_seq_status.log
