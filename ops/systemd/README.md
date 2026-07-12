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

## Layer 3 — `arc-news-watch.{service,timer}` + `scripts/arc_news_watch.py`

Added 2026-07-11 (operator request: "we should check on this daily to make
sure we keep apprised for our eventual November submissions"). Independent
of the conductor -- runs once daily via `codex exec` (verified to have real
web-search tool access) against the ARC Prize blog and the Kaggle
competition page/discussion, and appends a dated entry to
`docs/research-notes/arc-agi3-news-watch.md` only when it finds something
not already recorded in `ops/.arc_news_watch_state.json`. A "checked,
nothing new" run still logs its timestamp so the watch's liveness is
auditable without re-reading the full history every day.

```bash
# Copy unit files
cp ops/systemd/arc-news-watch.service \
   ops/systemd/arc-news-watch.timer \
   ~/.config/systemd/user/

# Reload + enable
systemctl --user daemon-reload
systemctl --user enable --now arc-news-watch.timer

# Force an immediate run (don't wait for the daily schedule)
systemctl --user start arc-news-watch.service
```

Fires daily at 09:07 local (`RandomizedDelaySec=300` to avoid a thundering
herd against arcprize.org/Kaggle at the exact minute mark).
`Persistent=true` catches up a missed run after the machine was
asleep/off at the scheduled time. Retires when the November 2026
submission deadline passes and ARC-AGI-3 news no longer needs daily
tracking, per the same two-condition retirement pattern as CLAUDE.md's
"ARC-AGI-3 November-Submission Standing Floor."
