# Carnot conductor systemd setup

Defense-in-depth against the orphan-pytest accumulation pattern observed
2026-05-05 10:12 UTC (load average reached 90 with stuck pytest workers
from prior conductor instances reparented to PID 1).

## Layer 1 — `carnot-conductor.service`

Wraps the conductor in a systemd cgroup. `KillMode=control-group` +
`SendSIGKILL=yes` ensures every pytest worker / xdist pool / codex
subprocess inherits the conductor's cgroup. Stopping the service
propagates SIGTERM/SIGKILL to the entire process tree, structurally
preventing orphan accumulation.

## Layer 2 — `carnot-orphan-cleanup.{service,timer}` + `orphan-cleanup.sh`

Janitor running every 30 min via systemd timer. Reads the active
conductor PID from `ops/conductor-heartbeat.json`, enumerates all
descendants, and kills any `python3` / `pytest` process with elapsed
>2hr that is NOT a conductor descendant. Logs to
`/tmp/orphan-cleanup.log`.

## Install

```bash
# Copy unit files
cp ops/systemd/carnot-conductor.service \
   ops/systemd/carnot-orphan-cleanup.service \
   ops/systemd/carnot-orphan-cleanup.timer \
   ~/.config/systemd/user/

# Copy cleanup script (mode +x preserved by cp -a; otherwise chmod)
cp ops/systemd/orphan-cleanup.sh ~/.carnot/
chmod +x ~/.carnot/orphan-cleanup.sh

# Reload + enable
systemctl --user daemon-reload
systemctl --user enable --now carnot-orphan-cleanup.timer
```

To switch the conductor from manual launch to systemd-managed:

```bash
# Stop manual conductor first
kill -TERM <conductor-pid>

# Start systemd-managed
systemctl --user enable --now carnot-conductor.service
```

The systemd-managed conductor inherits the same env from
`~/.carnot/conductor_state.sh` (referenced via `EnvironmentFile=`).
