# V682 method registration — 2026-09-29

This registration precedes Exp7851 measurement. The [V682 source review](../../research-references.md#v682-planning-review--2026-09-28-recorded-before-design) defines the prior scope. This map fixes local comparisons and their limits. Exp7851 checks administrative authority only; it does not estimate any scientific benefit.

| Primary method | Local adaptation | Limit |
| --- | --- | --- |
| [Verification Without Sufficiency, §III and §VII–XII](https://arxiv.org/html/2608.00585v1) | Exp7854 and Exp7857 compare a full evidence set, a nominated witness, a neighboring context, a matched disjoint context, and one support removed. Distinguish missing support from contradiction. | A context edit has no new human truth label. Source risk sensitivity cannot become answer accuracy. |
| [RECAP, §2–3](https://arxiv.org/html/2606.06698v4) | Exp7853 and Exp7858 use predict-before-feedback and add/edit/delete constraint episodes, then check old constraints after each update. | This adapts the evaluation protocol, not prompt optimization or the paper's architecture. |
| [Retrieval-warmed EBR, §III–IV](https://arxiv.org/html/2606.26476v1) | Exp7858 and Exp7859 compare aligned past memory, shuffled past memory, no-write, frozen, and oracle/immediate-feedback diagnostic arms. | Oracle information is not deployable. A prior memory value must change prospective choices to count. |
| [FPGA/ASIC co-design, §3–5](https://arxiv.org/html/2602.15985v2) | Exp7861 and Exp7862 measure CPU orchestration, transfers, construction, updates, durable commits and persistence before an acceleration bound. | Their physical chip and FPGA timings are external; a local bound is not measured Carnot board speed. |

## Frozen role and analysis plan

- Source family, not seed or source view, is the independent bootstrap unit. Keep family IDs and all censored, excluded and failed rows in primitive output.
- Source roles: fitting 256 families, tuning 64, policy selection 64, online update 96, online admission 64, evaluation 64, retention 32. No family crosses roles. Natural human-labeled data already exposed in development remains development evidence.
- Primary contrasts: full source versus witness-only and support-removed risk; calibrated typed decisions versus length/source controls; aligned delayed feedback versus shuffled, frozen and no-write controls; complete CPU service versus the measured current baseline. A fixture/oracle contrast is diagnostic.
- Family bootstrap: 2,000 resamples of independent source families, seed 68201. Report intervals even when null or adverse. The family count, not number of views or random seeds, is the effective sample size.
- Multiplicity: Holm correction within each named primary contrast family, familywise alpha 0.05. All secondary and oracle analyses remain diagnostic.
- Compute budgets: Exp7851 has zero model loads and zero generation calls; Exp7857 has a bounded Qwen3.8-27B generation budget set in its own preregistration; no present administrative receipt spends that budget. Each owned child has its frozen deadline. The repository health diagnostic has a separate 180-second deadline.

## V681 dispositions and current prerequisites

Exp7837 remains disqualified after its unsupported coverage API call and success-prefixed partial candidate. The V681 eight producer artifacts comprise five disqualified and three blocked; six science producers are missing. Their old required failures remain open. Exp7851 and Exp7854 are the two infrastructure slots. Exp7855 owns calibrated-decision training; Exp7858/7859 own continuous-learning contrasts; Exp7860 owns ARC outcome audit; Exp7862 owns hardware custody. Exp7853 first has to remove the fixture-ID feature shortcut and separate natural fit from calibration batches. The oracle-distinct corrigendum is retained: deterministic fixture agreement is circular evidence, not human-labeled natural benefit. Administrative agreement creates no scientific gate.
