"""Independently reduce the necessary chronology operand without producer helpers."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime

path = Path(sys.argv[1])
print('[8200-independent] before completed=0 pending=1', flush=True)
body = path.read_bytes()
data = json.loads(body)
available = []
for row in data['records']:
    try:
        stamp = datetime.fromisoformat(str(row.get('issued_at', '')).replace('Z', '+00:00'))
        if stamp.utcoffset() is not None:
            available.append(row)
    except ValueError:
        pass
assert not available, 'Nonempty chronology needs the complete independent identity reducer.'
value = dict(inventory_sha256='sha256:' + hashlib.sha256(body).hexdigest(),
             inventory_count=len(data['records']), authenticated_issue_timestamp_count=0,
             eligible_count=0, completed_count=0, independent_count=0,
             excluded_count=len(data['records']), original_exact_repeat_frequency=None,
             replay_exact_repeat_frequency=None, request_trace_ready_score=0,
             rationale='Every request needs an authenticated issue timestamp; none exists.')
Path(sys.argv[2]).write_text(json.dumps(value, indent=2) + '\n')
print('[8200-independent] after completed=1 pending=0', flush=True)
