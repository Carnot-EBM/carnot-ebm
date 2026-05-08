# Outer-Loop Hybrid Pattern (codex hourly + claude on-demand)

## What this is

`carnot-outer-loop.{service,timer}` runs `outer-loop-codex.sh` every hour
via systemd. The script gathers status snapshot (heartbeat, recent log,
unpushed commits, load, active procs), pushes any commit >30min old, and
asks codex/gpt-5.5 for a single-line verdict: `OK` or `ESCALATE`.

Output journals to `/tmp/carnot-outer-loop.log` and `journalctl --user -u
carnot-outer-loop`.

## Escalation criteria (codex writes ESCALATE: ...)

- conductor service inactive (systemd should auto-restart; escalate only if didn't)
- single codex subprocess >2hr elapsed without log activity
- load average >10 (orphan accumulation despite cgroup wrap)
- unrecovered Traceback/Exception in recent log
- persistent FAIL across multiple distinct tasks (not just one retry)
- task retried 3+ times bootstrap-only with no progress

## When to engage claude session

- Codex emits `ESCALATE: <reason>` — operator pings claude-code with the reason
- Operator notices a pattern codex missed (codex/gpt-5.5 has weaker
  structural-diagnosis ceiling than Sonnet/Opus)
- New structural conductor bug: claude better at code archaeology +
  surgical patch design + verification across diagnostic-fix-validate cycle

## Quota implications

- Codex hourly: ~3,000-5,000 tokens per check, $0 Claude burn
- Operator pings claude on escalation: ~1-3% Claude weekly quota per event
- Most weeks: zero escalations (8 conductor structural fixes already shipped
  this session arc + systemd cgroup wrap + orphan janitor + classifier patches
  cover most known failure modes)

## Install

```bash
cp ops/systemd/carnot-outer-loop.{service,timer} ~/.config/systemd/user/
cp ops/systemd/outer-loop-codex.sh ~/.carnot/
chmod +x ~/.carnot/outer-loop-codex.sh
systemctl --user daemon-reload
systemctl --user enable --now carnot-outer-loop.timer
```

## Status

```bash
systemctl --user list-timers carnot-outer-loop.timer
journalctl --user -u carnot-outer-loop --since "24 hours ago"
tail -100 /tmp/carnot-outer-loop.log
```
