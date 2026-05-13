import os
from datetime import datetime

end_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

session_text = f"""
## Session: 2026-05-13 Milestone 2026.05.159 Operational Retrospective

### Turn Log

| Turn | Start | End | Description | Tokens (est) |
|------|-------|-----|-------------|------|
| 1 | 2026-05-13T08:22:02Z | {end_time} | Write operational retrospective for milestone 2026.05.159 | TBD |
"""

with open("ops/metrics.md", "a") as f:
    f.write(session_text)
