import os
metrics_path = "ops/metrics.md"

new_metrics = """
## Session: 2026-06-02 Milestone 2026.06.342 Operational Retrospective

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-06-02T23:49:08Z | 2026-06-02T23:51:00Z | Wrote operational retro for 2026.06.342 | ~3k |
"""

with open(metrics_path, "a") as f:
    f.write(new_metrics)

print("METRICS APPENDED")