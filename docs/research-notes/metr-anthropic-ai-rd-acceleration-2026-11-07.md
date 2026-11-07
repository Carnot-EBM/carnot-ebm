# METR assessment of AI R&D acceleration at Anthropic: reading note (2026-11-07)

Status: reading note. No experiment was run. Nothing here is a Carnot result.

Source: METR blog post dated 2026-11-05, "Anthropic AI R&D acceleration assessment", https://metr.org/blog/2026-11-05-anthropic-ai-rd-acceleration-assessment/. The outer loop read the page text (78,000 characters) and checked the figures below against it. It did not read the linked data, charts or appendices in full, and it did not check replications.

Access note: `curl -sL -A "Mozilla/5.0"` returns the page. Strip the tags before reading.

## What the post reports

| Item | Result |
|---|---|
| Engagement | High-access evaluation of Anthropic, 2026-08-12 to 2026-10-29 |
| Main estimate | AECI (Anthropic's version of the Epoch Capabilities Index) grows at 1.9x the pre-October-2025 rate, 95% CI 1.5 to 2.8x |
| Breakpoint | 2025-10-07, one breakpoint imposed, estimated in the main specification |
| Robustness | 1.1x to 2.9x overall. Link function alone: 1.1x to 2.9x. Snapshots used: 1.9x to 2.2x. Benchmarks used: 1.9x to 2.2x |
| Haiku 5.5 system card | 1.7x. The post says the difference comes mainly from internal-only models it included |
| Attribution | "perhaps 80%" of the October 2026 acceleration is due to AI, with substantial uncertainty |
| Self-reported uplift | About 12x average for the most capable internal model in October 2026. The post says it is hard to extrapolate |
| Tokens per attempt | About 2.5M input and 72k output on internal benchmarks. About 10M input and 94k output on agentic coding benchmarks |
| Redaction | Anthropic had the right to redact. METR had the right to disclose that redaction happened |

## Limits the post states

- Anthropic redacted much of the most informative evidence. Readers must rely on METR's judgment.
- The benchmark trend covers well-specified tasks. The post notes extra uncertainty in how it transfers to less well-specified tasks.
- The result moves with analysis choices. The post gives the 1.1x to 2.9x range itself.
- One reading of the post: if Anthropic models are more benchmark-optimized than others, the 1.9x is overstated. The post gives the opposite case too.

## What it means for Carnot

1. **Report spread, not one number.** The post puts a robustness range next to its headline. Our artifacts report one F1 or AUROC. A claim should carry its sensitivity to analysis choices. Cheap to add.
2. **Self-report is weak evidence.** The 12x self-reports sit far from the roughly 2x benchmark estimate. This supports our rule that an agent's own verdict does not count. It also supports the covert-underperformance watch: the agent and model are not recorded per artifact.
3. **Withheld evidence means trust.** METR says its readers must trust its judgment. Our label audit hit the same wall in a smaller form: a `verifier: heuristic` field that was a backfilled default. Provenance must be inspectable.
4. **Saturation.** The post flags benchmark saturation as a source of error. Our "saturated null" verdicts (exp4396, exp4407) have the same shape, and the step 1 audit shows a worse cause there.
5. **AI-assisted analysis.** The post says Claude was used in data collection and analysis, and that METR found no systematic bias. This matches our rule that a reviewer re-derives from raw source.
6. **Verifiable-task scope.** The measured acceleration is on well-specified tasks, where an executable check exists. Our open frontier is the part with no cheap oracle. This weakly supports the re-anchored north star. The post tests no verifier.
7. **Cost per attempt.** The post reports tokens per attempt. Our task artifacts do not record tokens consistently.

## Cheap follow-ups (none queued)

- Add a `sensitivity` field to calibration artifacts: the same metric under two or three reasonable analysis choices.
- Record tokens and wall time per attempt in the BFS pilot artifact. It already needs wall time.
- Record agent and model per artifact. This item is already open.

## Rule checks

- Circularity / Oracle-Distinctness: no verifier claim in the post. The rule does not apply.
- No weight update to the mandated generator. No external publication.

## Verdict

Moderate value as a reporting template. No evidence about verifiers, labels or calibration. No experiment follows from this note.
