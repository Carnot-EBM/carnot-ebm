#!/bin/bash
# conductor-stop.sh — stop the conductor ON PURPOSE, machine-readably.
#
# WHY (REQ-CONDUCTOR-RESTART-1). The janitor STARTS a dead conductor
# within two 30-minute cycles unless ~/.carnot/conductor-hold exists. A
# bare `systemctl --user stop` therefore gets overridden in under an
# hour. This wrapper writes the hold (with your reason) FIRST, then
# stops — the order matters: hold-then-stop leaves no window in which
# the janitor sees "dead, no hold".
#
# Resume:  rm ~/.carnot/conductor-hold && systemctl --user start carnot-conductor.service
# A hold older than 48h WARNs daily in ops/conductor-log.md.
set -euo pipefail
HOLD="${CARNOT_JANITOR_HOLD:-$HOME/.carnot/conductor-hold}"
REASON="${1:-no reason given}"
mkdir -p "$(dirname "$HOLD")"
printf '%s stopped by %s: %s\n' "$(date -u +%FT%TZ)" "${USER:-unknown}" "$REASON" > "$HOLD"
systemctl --user stop carnot-conductor.service
echo "conductor stopped; hold written to $HOLD"
echo "resume with: rm $HOLD && systemctl --user start carnot-conductor.service"
