# TokenEconomizer and context-mode: reading note (2026-11-01)

Status: reading note. No experiment was run. Nothing here is a Carnot result.

## What was read

- TokenEconomizer (Microsoft and UC Berkeley, Apache 2.0). Repo cloned at commit c53a4a6dcd (2026-10-31). Read: README, `docs/ARCHITECTURE.md`, `docs/PLUGIN.md`, `loopguard/` README, `rules.py`, `judge.py`, `microskills/` README. Paper: arXiv 2610.22732, HTML body read to character 100000 of 150223.
- context-mode (`mksglu/context-mode`) and contextmode.ai. Read the README and the product page only, through a summarizing fetch. The last 15k characters of the README, `BENCHMARK.md`, the LICENSE file and the code were NOT read.
- No code from either project was run.

## TokenEconomizer

An external controller for coding agents (Claude Code, Codex, Copilot CLI, Pi). It has three mechanisms:

1. **Micro-skills.** Repo facts (for example "bare `pytest` is missing, use `python -m pytest`") are learned from finished sessions. A fact becomes active after confirmation on 3 sessions. It retires on contradiction.
2. **Tool pruning.** A tool never used over 10 prompts across at least 2 sessions, or over 5 sessions, goes on a deny list in the repo's own settings file.
3. **Failure-loop guard.** A free gate checks the recent trace. If it opens, a judge model decides stuck or productive. A stuck verdict at confidence 0.7 or higher sends one redirect. The plugin never stops a turn.

### Reported results (paper, SWE-Bench, three seeds, mean and standard deviation)

| Variant (Claude Code) | Sonnet 5 cost change | Opus 4.8 cost change |
|---|---|---|
| Micro-skills | -11.8 +/- 2.1 | -14.3 +/- 2.3 |
| Tool pruning | -13.2 +/- 1.3 | -17.7 +/- 1.5 |
| Failure-loop guard | -7.1 +/- 2.2 | -5.4 +/- 1.8 |
| All three | -23.6 +/- 1.8 | -30.9 +/- 2.1 |

- Baseline: the unmodified agent at high reasoning effort. Cost is cache-aware and includes the controller's own model calls.
- Pass@1 stays within 3 points of default. The Opus 4.8 run shows -1.0.
- The ablation table covers Claude Code only. Codex and Copilot have no per-mechanism breakdown.
- The paper's own limits: the task streams are "related but independent fixes rather than actual user sessions". Each mechanism can err (a stale fact, a wrongly pruned tool, a productive task read as a loop).
- The plugin docs add: the loop guard ran only on provoked loops, and "whether a nudge changes a looping model's course is not established" (on Pi, one run stopped after the nudge and another went on).

### Correction to my earlier summary

In chat I said the redirect-instead-of-kill idea was the strongest match to Carnot. That was too strong, for three reasons.

1. The loop guard is the SMALLEST of the three mechanisms (5 to 7 points of cost).
2. The guard runs on tool-call hooks (`PostToolUse`). Our largest kill population is silence: 100 of 145 soft kills never printed a line, and 96 died at exactly 1201 s. A silent task fires no tool-call hook, so this guard cannot see it. The progress-line rule is the right fix for that population, and nothing here replaces it.
3. The authors do not establish that a nudge changes a looping model's course.

### What is usable

1. **Tool pruning for this outer-loop session.** It is the largest single saver in the paper. This session loads many MCP tools (desktop-commander, jcodemunch, jdocmunch, Claude Docs, a failed carnot-verify server). I have not measured how much context their schemas and names cost. Measure first. If the cost is real, prune servers that sessions never call. This is a settings change, and it needs no new software.
2. **Confirmation threshold for facts.** Their rule: a fact needs 3 confirming sessions, retires on contradiction, and is injected only while the failure it prevents keeps occurring. Our memory notes enter on one observation and never retire. Memory entries about tool traps (`--no-cov`, `--basetemp`, `-st` for llama-cli) are micro-skills in their sense. A retirement rule for memory notes is the transferable part.
3. **Gate before judge.** Their free gate decides whether a paid look is worth it: a repeated identical failure, or 8 or more error outputs after 18 calls. The point is that the judge is not a per-step tax. We can test the same gate offline on our own logs with no model call.

### Open questions (not answered by what I read)

- Does the conductor's `codex exec --dangerously-bypass-approvals-and-sandbox` run honor plugin hooks? PLUGIN.md verified Codex 0.153.1 with a separate flag, `--dangerously-bypass-hook-trust`. Our process path shows Codex 0.159.1. Untested here.
- What is the makeup of our 198 hard-cap kills (4800 s, task kept printing)? If most show a repeated-failure pattern, the gate fits them. If they show steady progress, it does not. Check against a base rate of completed tasks before drawing any conclusion.
- The paper's task streams are SWE-Bench fixes in six repositories. Carnot tasks are experiment runs with long model calls. Transfer is unknown.

## context-mode

A local MCP server plus hooks. Large tool output runs in a sandbox and only stdout returns to the conversation. Output is indexed in SQLite with BM25 search. Hooks snapshot state before compaction. Reported: 98% context reduction on selected examples. The site reports a $7.20 to $2.49 drop on one long session and 168 paired runs, all vendor-run.

### Assessment

- "Think in code" (compute in a script, print the answer) is what we already do. The only gap I saw this session: one `journalctl` call pulled a full planner prompt into context. A filter in the command fixes that without new software.
- It helps the outer-loop Claude session only. The conductor runs codex and gains nothing.
- Its hooks block Bash, Read and web fetches. That may conflict with the deny-hook and cwd-drift trap already in memory.
- Session data is deleted at startup unless `--continue` is used. That cuts against the write-it-down rule.
- The license looks like Elastic License 2.0 (source-available, not open source). I did not read the LICENSE file. Check it before any use, against the Decentralization rules.
- contextmode.ai is a hosted proxy. It works by changing the agent's base URL, so every prompt passes through the vendor. That conflicts with the per-call data-minimization rule. Do not use it. I could not confirm that the site and the repo share an owner.

## Rule checks

- No verifier-value claim is made by either project. The Circularity / Oracle-Distinctness rule does not apply.
- Neither project touches the north star, the calibrated-decision floor or ARC.
- All cost numbers are self-reported and I did not verify any of them.

## Verdict

Low to medium value. No Carnot experiment follows. Three small, separable ideas are worth a look: measure MCP tool cost and prune unused servers, add a retirement rule to memory notes, and test their stuck gate offline on our own kill logs. Do not install either product on the conductor.
