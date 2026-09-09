# Wall-clock+idle kill rise on 2026-09-07: composition or rate?

Measured 2026-09-09 10:33Z to 11:05Z. Read-only investigation. No repository file was changed.
All scripts and intermediate JSON live next to this file in `killrate/`.

## 1. Answer

**It is a rate rise inside a stable population.** The same kind of task, with the same prompt
shape, the same agent, the same model name, and the same conductor code, started to die more
often. The mix of tasks did not move in a direction that explains it.

The mechanism is measured, not guessed. Each long conductor task has ONE long silent phase.
The phase sits between "the RED test was written and run" and "the implementation is written".
No tool call is in flight during that phase. The child prints the `codex` message header, then
prints nothing until the assistant message arrives. Its length roughly doubled on 2026-09-07
between the sessions that started at 12:03Z and 13:02Z. The conductor's idle rule is 600 s. The
new median of that phase is 556 s. So about half of the exposed tasks now cross the line.

### 1a. The label is a function of the prompt; the kill is a function of one silent step

From `scripts/research_conductor.py` (read 2026-09-09):

| Line | What it does |
|---|---|
| `:855` `_LIVE_MODEL_PROMPT_MARKERS` | `cached_sota_pair`, `live_llm_inference`, `llama_cpp`, `n_gpu_layers`, `live gpu` |
| `:969-979` | `STALL_TIMEOUT = 0` (claude), `1800` if the prompt has a marker, else `600` |
| `:1072-1085` | the stall check runs FIRST on every silent poll: silence > `STALL_TIMEOUT` -> `Stalled after Ns` |
| `:1220` | `IDLE_GRACE = 600`, a separate constant |
| `:1393` | `elapsed > 1200 AND silence > 600` -> `Wall-clock+idle timeout after Ns (Ms silence)` |

Consequence. A task WITHOUT a marker dies as `Stalled after` at 600 s of silence. A task WITH a
marker cannot die that way before 1800 s, so it dies as `Wall-clock+idle` once elapsed passes
1200 s. Both labels record the same event: one silent step longer than 600 s. The
`Wall-clock+idle` count is therefore "silent step > 600 s, in a task whose prompt names a live
model". All 18 such kills since 08-31 carry `Codex CLI error`, and the conductor's process
environment (pid 1797075, `/proc/1797075/environ`) shows `AGENT_TYPE=codex`.

### 1b. Per-day census (my own count, lenient parser)

Source: `ops/conductor-log.md`, 17,233 dated rows, 16,725 parsed, 508 unparsed (all
`ESCALATE_OPUS_100` rows from April-May; none carry a kill string). Denominator = rows with
status OK, FAIL, or FLAGGED. Kill kinds come from the literal strings the source emits.

| day | terminal | wall_idle | stall | hard_cap | other FAIL | wall_idle rate |
|---|---|---|---|---|---|---|
| 08-24 | 31 | 0 | 0 | 4 | 5 | 0.0% |
| 08-25 | 41 | 0 | 0 | 2 | 3 | 0.0% |
| 08-26 | 27 | 0 | 0 | 7 | 3 | 0.0% |
| 08-27 | 30 | 0 | 0 | 5 | 9 | 0.0% |
| 08-28 | 19 | 0 | 0 | 9 | 1 | 0.0% |
| 08-29 | 30 | 0 | 0 | 4 | 3 | 0.0% |
| 08-30 | 47 | 0 | 0 | 2 | 18 | 0.0% |
| 08-31 | 39 | 4 | 3 | 0 | 7 | 10.3% |
| 09-01 | 39 | 0 | 0 | 2 | 5 | 0.0% |
| 09-02 | 36 | 1 | 0 | 1 | 3 | 2.8% |
| 09-03 | 47 | 1 | 1 | 1 | 11 | 2.1% |
| 09-04 | 44 | 0 | 0 | 2 | 1 | 0.0% |
| 09-05 | 46 | 0 | 0 | 0 | 10 | 0.0% |
| 09-06 | 58 | 0 | 0 | 0 | 23 | 0.0% |
| 09-07 | 37 | 3 | 0 | 0 | 1 | 8.1% |
| 09-08 | 42 | 8 | 0 | 3 | 0 | 19.0% |
| 09-09 (to 10:30Z) | 16 | 1 | 0 | 2 | 3 | 6.2% |

My denominators differ from the lead's by 1-2 rows per day. The two parsers split rows with a
`|` inside the detail field differently. The rates agree to within one point.

Two things this table adds. First, 08-31 (a Monday) had a prior burst: 4 wall_idle + 3 stall in
39 rows. Second, since 08-31 every one of the 18 wall_idle kills PRINTED before it went quiet
(silence < elapsed in all 18). The 100-of-145 "never printed" population in `ops/known-issues.md`
is older; it is not what is happening now.

Rows are not tasks. 09-08's 8 kill rows are 5 distinct tasks (three were killed twice in a row).
09-07's 3 rows are 3 tasks.

### 1c. Composition did not move in the direction that would explain it

Roadmap join: 47 activated roadmap versions since 08-27 (87 since 08-09) taken from the git
history of `research-roadmap.yaml`; 711 task rows. Each terminal log row was joined to the task
whose `title[:50]` matches and whose milestone was activated most recently before the row. 980
terminal rows 08-10..09-09; 625 matched to a roadmap task; 355 are conductor-internal (planner,
activation, re-exec). All numbers below are on the matched rows.

| day | rows | distinct tasks | wall_idle rows / tasks | median prompt chars | requires_gpu | live-marker share | median est_min |
|---|---|---|---|---|---|---|---|
| 09-01 | 30 | 26 | 0 / 0 | 4117 | 17% | 17% | 210 |
| 09-02 | 22 | 17 | 1 / 1 | 4897 | 27% | 36% | 240 |
| 09-03 | 33 | 23 | 0 / 0 | 4605 | 21% | 12% | 180 |
| 09-04 | 33 | 30 | 0 / 0 | 4692 | 21% | 21% | 180 |
| 09-05 | 29 | 27 | 0 / 0 | 4634 | 17% | 17% | 120 |
| 09-06 | 27 | 22 | 0 / 0 | 4857 | 44% | 52% | 180 |
| 09-07 | 24 | 21 | 3 / 3 | 5051 | 17% | 29% | 150 |
| 09-08 | 31 | 20 | 8 / 5 | 4671 | 45% | 52% | 60 |
| 09-09 | 10 | 6 | 1 / 1 | 5547 | 60% | 70% | 58 |

The decisive row pair is 09-06 against 09-08. Both have 52% live-marker rows and 44-45%
`requires_gpu`. 09-06 had 0 kills in 14 marker rows. 09-08 had 7 kills in 16 marker rows.

Kill rate inside the exposed (live-marker) population only:

| day | marker rows | wall_idle | rate |
|---|---|---|---|
| 09-01 | 5 | 0 | 0% |
| 09-02 | 8 | 1 | 12% |
| 09-03 | 4 | 0 | 0% |
| 09-04 | 7 | 0 | 0% |
| 09-05 | 5 | 0 | 0% |
| 09-06 | 14 | 0 | 0% |
| 09-07 | 7 | 3 | 43% |
| 09-08 | 16 | 7 | 44% |
| 09-09 | 7 | 1 | 14% |

Track does not isolate it either. Kills 09-07..09-09 fall in infrastructure (2), verification (2),
self_learning (4), arc (2), sampling (1), constraint_reasoning (1).

Prompt shape is stable across the onset. Per activated milestone, the share of prompts that
demand a RED test before implementation is 100% for .612-.624 (09-04 to 09-07) and 67-92% for
.625-.629. Median prompt length is 3990-5752 chars from .599 to .629 with no step at .624/.625.
The median count of "read first" files is 10-14 throughout. `CLAUDE.md` (which every prompt
lists first) was 304,897 bytes on every day 09-01..09-07 and grew on 09-08, after the onset.

Generated code did not get bigger. Experiment modules first committed 09-06: n=5, median 70,810
bytes. 09-07: n=20, median 56,592. 09-08: n=18, median 55,875. 09-09: n=6, median 57,009.

### 1d. The silent step, measured from the systemd journal

The brief says child stdout is never written to a file. That is true of the conductor's own
sinks. But the conductor prints every child line to its stdout (`research_conductor.py:1064`),
and systemd captures that into the journal with a per-line timestamp. The journal holds the
conductor's output from 2026-09-07 01:56Z onward; earlier entries were rotated out (1.8 GB cap,
about 2.5 M lines per day). So the journal covers every kill from 09-07 on, plus 13 clean long
sessions from the morning of 09-07 as a control, but not 09-06.

I split the journal into `run_agent` sessions on the conductor's own `Calling Codex CLI` and
`Codex CLI completed|failed|stalled|past soft wall-clock cap|exceeded HARD wall-clock cap|
produced stable deliverable` lines (95 sessions). Lines without the `[conductor]` prefix are
child output. For each session I took every child-output gap over 240 s.

Finding: nearly every session of 15 min or more has exactly ONE gap over 240 s. It starts 400 to
1200 s into the run. Around it, the same three things appear in every case I inspected: before
the gap, the tail of the RED test patch (`+ assert ...`) and the `codex` message header; after
the gap, an assistant message such as "The RED run failed at import as intended. I'm
implementing the checker now" (survivors) or nothing (killed). No shell command is in flight.
This is the model's reasoning step before it writes the implementation.

Its length by window (sessions of 900 s or more):

| window | sessions | median longest gap | gaps >= 500 s | gaps >= 600 s | killed by silence |
|---|---|---|---|---|---|
| 09-07 01:50Z-13:00Z (.622, .623, early .624) | 13 | 305 s | 1 | 0 | 0 |
| 09-07 13:00Z - 09-08 23:59Z (.624 late, .625-.628) | 45 | 556 s | 34 | 14 | 12 |
| 09-09 00:00Z-10:00Z (.629, .630) | 10 | 459 s | 3 | 2 | 1 |

Survivors pile up just under the line: 598, 599, 595, 592, 591, 595, 585, 577 s. Killed sessions
read 601 s (the conductor's own timer). Tasks that survived with a gap over 600 s (685, 686,
706, 752 s) did so only because the gap ended before elapsed reached 1200 s. Non-marker tasks in
the same window show the same 500-595 s gaps; one of them crossed 600 and was logged
`Stalled after 601s`. Same task, different attempt: "Three-family symbolic grounding comparison"
gaps were 601 (killed), 646 (killed), 318 (OK); "Verifier-balanced external-memory" gaps were
601 (killed), 706 (killed), 686 (survived, ended at 1172 s elapsed).

The shift happened on the same conductor process (pid 1797075, started 09-06 22:40Z), the same
code (no re-exec between 09-05 13:10Z and 09-07 23:28Z), the same binary, the same config, the
same model name, and the same prompt shape, between 12:03Z and 13:02Z on 09-07.

## 2. Candidates, kept and dropped

| Candidate | Verdict | Evidence |
|---|---|---|
| Conductor code change | **Dropped** | The log's `Conductor re-exec` rows show the process picked up new source at 09-05 13:10Z and next at 09-07 23:28Z. The clean 09-06 day and the first two 09-07 kills ran on the same code. |
| Codex CLI upgrade (0.153.4 installed 09-04 18:53 local) | **Dropped** | That binary is `~/.local/bin/codex`. The conductor's `PATH` (from `/proc/1797075/environ`) is `.venv/bin:/usr/local/bin:/usr/bin`, so it runs `/usr/bin/codex` = 0.149.1, installed by pacman 08-25 with no later upgrade. The 09-08 rollout that carries the conductor's prompt shape reports `cli_version 0.149.1`. |
| Reasoning effort raised to xhigh | **Dropped** | `~/.codex/config.toml` sets `model_reasoning_effort = "xhigh"` and was written 09-06 23:04Z, which looked aligned. But every rollout under `~/.codex/sessions` from 08-20 through 09-09 already carries `"effort":"xhigh"` (e.g. 08-21: 40 of 40, 09-06: 349 of 349). The effort did not change. The `model` line in that file is overridden by the conductor's `--model` argument. |
| Systemd restart 09-06 22:40Z | **Not tied to onset** | Real: `ExecMainStartTimestamp=Sun 2026-09-06 18:40:36 EDT`, `NRestarts=0`. The post-restart environment shows `AGENT_MODEL=gpt-5.6-sol`; the drop-ins on disk gave the same value before (60- since 08-23; 70- only touches planner/retro/audit). The 13 clean long sessions on the morning of 09-07 ran under the restarted process with a median gap of 305 s. The pre-restart process environment cannot be verified; the process is gone and the journal does not reach back. |
| Task mix (live-marker share, GPU share, track) | **Dropped** | 09-06 and 09-08 have the same marker and GPU shares; 0 vs 7 kills. Kills spread over six tracks. |
| Prompt shape (RED-first contract, prompt length, read-first count) | **Ingredient, not trigger** | Stable since .612 (09-04). It is what creates one large silent reasoning step per session; it did not change on 09-07. |
| Bigger implementations to generate | **Dropped** | Module medians 70.8 KB (09-06), 56.6 KB (09-07), 55.9 KB (09-08). |
| `CLAUDE.md` growth | **Dropped** | Unchanged 09-01..09-07; grew 09-08. |
| Task Progress-Line Requirement (09-08) | **Dropped (already by the lead), and it cannot help this class** | The silence is the model reasoning before the experiment module exists. A flushed line inside a script cannot fill a gap in which no script is running. Consistent with the known-issues note that exp7130's killed attempt left no module. |
| GPU or host contention from outer-loop agents | **Not evidenced** | No tool call is in flight during the gap in any inspected session, so local load would have to slow the codex client itself by minutes. Not measured; not claimed. |
| Slower model service on 09-07/09-08 | **Kept as the only date-aligned candidate; not proven** | (a) The doubling happened mid-day 09-07 with no local change. (b) Control: the outer loop's own short single-turn codex sessions on the same model, from `~/.codex/sessions`, show median time to first assistant message 10, 11, 9, 10 s on 09-03..09-06, then 15 and 14 s on 09-07/09-08, then 10 s on 09-09; p90 28, 24, 18, 19 -> 44, 38 -> 33. (c) All 18 wall_idle kills since 08-20 fell on Mon-Thu; 0 of 376 terminal rows on Fri-Sun. Limits: the control's prompt mix is not fixed day to day (no prompt family recurs on 3+ days), n=48/49 on the two days, and 08-24..08-28 were weekdays with 0 kills, though the RED-first contract that creates the vulnerable step was not yet universal then. **Falsifier:** run a fixed prompt against gpt-5.6-sol at xhigh once an hour and log the latency. If the conductor's longest-gap median stays above 500 s on 09-12/09-13 while that latency is flat, this story is wrong. If both move together, it is supported. |

Secondary cost, recorded because it compounds: 5 of the 12 kills on 09-07..09-09 were followed
within 3 minutes by `SKIP Pre-tests failing, self-heal failed`. The killed attempt leaves a RED
test with no module, and the next attempt's pretest fails on it. 5 of 12 retried to OK.

## 3. What I could not determine, and what would have settled it

- **The 09-06 gap distribution.** The journal is the only record of child output and it holds
  about 2.5 days. The conductor's own codex sessions are `--ephemeral`: no rollout file, no row in
  `~/.codex/state_5.sqlite` (checked by matching each kill's start time against thread
  `created_at`; the only matches within 180 s were unrelated outer-loop review threads). A
  copy of the journal from 09-06, or a per-task stdout file, would have given a same-code
  before/after.
- **Why the journal is so large.** Inside one session the child printed 3,943 lines in one
  second, 1,525 of them distinct, in repeated bursts of the same test-file diff every 10-25 s.
  The session banner appears once, so these are not whole-transcript re-dumps. I did not find
  the origin. It is why retention is ~2 days. Not related to the kills; recorded so nobody
  re-measures it.
- **The model service's latency directly.** No request-level timing exists on this host. The
  short-session control above is the nearest thing and its prompt mix is uncontrolled.
- **The pre-restart process environment.** Gone with the process.
- **Whether the conductor's `--model` is honoured over `config.toml`'s `model = "gpt-6-astra"`.**
  The `Calling Codex CLI (model: gpt-5.6-sol)` lines and the rollout `turn_context` for the one
  conductor-shaped 09-08 rollout both say gpt-5.6-sol, so I treat it as honoured.

## 4. Draft for `ops/known-issues.md`

### 2026-09-09: the 09-07 wall-clock+idle rise is a rate rise, not a mix shift; the silent step is the model's pre-implementation reasoning

Follow-up to "The wall-clock+idle kill rate rose ~9x on 2026-09-07, cause unidentified".
Joined every terminal log row 09-01..09-09 to its roadmap task (625 matched rows, 47 activated
roadmap versions from git). The live-marker share, `requires_gpu` share, and prompt shape are
the same on 09-06 (0 kills in 14 marker rows) and 09-08 (7 kills in 16). Track, prompt length,
read-first count, and generated module size do not separate the days. The systemd journal keeps
the conductor's child output for about 2.5 days with per-line timestamps (this is the child
stdout record the earlier entry said did not exist). Segmented into 95 `run_agent` sessions from
09-07 01:56Z, it shows that each long session has one silent step, between the RED test patch
and the assistant message that announces the implementation, with no shell command in flight.
Its median length was 305 s in the 13 long sessions before 09-07 13:00Z, 556 s in the 45
sessions from 09-07 13:00Z to 09-08 23:59Z, and 459 s on 09-09 morning. `IDLE_GRACE` is 600 s,
so about half of the exposed tasks now cross it; survivors read 591-599 s, kills read 601 s. The
label is set by the prompt: a task whose prompt names a live model has `STALL_TIMEOUT` 1800 and
dies as `Wall-clock+idle`; the same gap in any other task dies as `Stalled after`. Ruled out with
dates: conductor code (no re-exec 09-05 13:10Z to 09-07 23:28Z), the codex binary (the
conductor runs `/usr/bin/codex` 0.149.1, not the 09-04 `~/.local/bin` install), reasoning effort
(xhigh in every rollout since 08-20), `CLAUDE.md` size (unchanged until 09-08), and the
progress-line rule (the gap has no script to print from). The only date-aligned candidate is a
slower model service on 09-07/09-08: the outer loop's short codex sessions on the same model
show median first-message latency 9-11 s on 09-03..09-06, 14-15 s on 09-07/09-08, 10 s on 09-09,
and all 18 such kills since 08-20 fell Mon-Thu. That is consistent, not proven; the control's
prompt mix is not fixed. Falsifier: a fixed hourly probe against gpt-5.6-sol at xhigh, compared
with the journal's longest-gap median over the weekend of 09-12. Secondary cost: 5 of the 12
kills left a RED test with no module, so the retry's pretest failed and the task was SKIPped.
