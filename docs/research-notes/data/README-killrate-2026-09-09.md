# Why this JSON is committed

`killrate-2026-09-09-journal-sessions.json` supports
`../wall-clock-idle-kill-rate-investigation-2026-09-09.md`.

It is the ONLY finding in that investigation that cannot be rebuilt. Every other
input — the conductor log rows, the kill rows, the roadmap-task join — comes from
`ops/conductor-log.md` and from the git history of `research-roadmap.yaml`, both
permanent. This file was segmented from the systemd journal, and that journal keeps
about 2 days 9 hours (measured 2026-09-09: oldest entry 2026-09-07T01:56Z). The
conductor's children also flood it, so the window is shorter than the disk implies:
2,321 suppression events were counted against a default limit of 10,000 messages per
30 seconds.

So the sessions this file holds — including the 09-07 shift that is the whole
mechanism claim — are already unreachable through `journalctl` for the earliest part
of the window, and all of it expires within days.

The note's medians (305 s, 556 s, 459 s) are enough to run the proposed falsifier.
This file is what allows anyone to check those medians, or to ask a question the note
did not, rather than taking them on trust.
