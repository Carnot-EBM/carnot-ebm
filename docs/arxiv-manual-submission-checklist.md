# Carnot arXiv Manual Submission Checklist

Run date: 2026-05-05

Upload URL: https://arxiv.org/submit

Ready bundle:
- Relative path: `results/arxiv_bundle_v11.tar.gz`
- Absolute path: `/home/ianblenke/github.com/ianblenke/carnot/results/arxiv_bundle_v11.tar.gz`
- Verified non-empty source archive: yes

## Pre-Filled Metadata

Title:

```text
Carnot: An Architectural Framework for Mapping the Empirical Bounds of LLM Verification
```

Authors:

```text
Ian Blenke <ian@blenke.com>
```

Primary category:

```text
cs.LG
```

License:

```text
CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)
```

Abstract:

```text
Verifier-filtered self-distillation has become a central paradigm for aligning open-weight LLMs without closed-frontier dependence. Naively, the recipe is ``compose more verifiers, sample faster, and the residual error vanishes.'' We show empirically that this naive scaling collides with three structural walls --- a verifier correlation ceiling, an exact-sampling detailed-balance limit, and an out-of-distribution energy-ordering inversion on highly-optimized SOTA outputs --- that no amount of compute or engineering effort can push past in their current form. We position Carnot not as a finished verification system but as an architectural framework designed to make these walls measurable, to bound them with closed-form theory, and to deploy fall-backs that remain mathematically exact under the bounds. Specifically, we (i) replace the naive geometric-mean joint-volume approximation with the correct $\sqrt{\det(\Sigma)}$ factor and apply the Welch/Rankin Simplex bound to derive the verifier-composition ceiling $k^* \leq 3.125$ from the empirical $\alpha^2 = 0.66$ of three deployed text probes ($D_{\mathrm{eff}} = 1.603$, exp1093); (ii) replace the prior synchronous parallel Glauber hardware story with a $\chi \leq 4$ sparse-constraint FPGA Fast-Path plus CPU fallback design after auditing the sampler's software-proxy $\mathrm{KL} = 3.07$ against single-site Gibbs (exp1094 simulation; bitstream on-board KL not yet measured), while deferring same-basis CPU-vs-FPGA speedup claims until matching measurements exist; (iii) document the pre-retrain energy-ordering inversion on SOTA outputs ($\overline{E}_{\mathrm{correct}} = 0.689 > \overline{E}_{\mathrm{incorrect}} = 0.621$, exp1100) as an out-of-distribution Goodhart-class anomaly that constrains the class of energy functions any decentralized verifier may use; (iv) show that retraining the SOS-KAN verifier on a 7{,}329-pair SOTA-inclusive corpus fixes the observed inversion with AUROC=0.9774 and correct energy ordering restored (exp1120); and (v) report the brittleness of pre-filtered self-distillation (exp1099) as evidence that the energy signal has to drive the filter, not be re-derived from accept/reject labels. Carnot's positive baseline results --- SOS-KAN $\mathrm{AUROC} = 0.9545$ on the FoVer corpus (n=6{,}548; exp1072), GRPO with ThinkPRM v2 as an explicit energy reward improving held-out GSM8K by $+4$ pp (n=25, 95% CI: [0.1%, 20.4%]) to $+8.51$ pp (n=25, 95% CI: [1.0%, 26.0%]) (exp1118/1129), post-retrain $\alpha_t = 0.52$ on Qwen3.6-35B-A3B (up from $0.38$, exp1130), and the HumanEval extraction-fix anomaly ($+36$ pp after the harness fix, Appendix~\ref{app:harness-anomalies}) --- demonstrate that the framework is operational in-distribution and can repair measured failure modes when harness artifacts are separated from model behaviour; the foregrounded negatives demonstrate that it is rigorous out of it. The .106 closeout adds the first positive evidence chain for the prior publication-hold blockers: live local Qwen3.6-35B certificate generation with tag-first CRANE prefix injection reaches certificate parse rate $1.0$ (exp1366); semantic validation, VERGE MCS repair, and conservative scheduler triage run end-to-end with repair hint precision $1.0$ and false acceptance $0.0$ (exp1369--1371); self-learning uses the primary semantic-verified path with four fresh verified samples and self-learning delta $=1.596429$ (exp1374); and a CPU-only GS-KAN PWA/MILP check verifies the stated energy bound (exp1372). Complementary CPU probes add DiffuTruth-style falsehood energy AUROC $0.867$ with KAN correlation $0.961$ (exp1367) and Eidoku CSP feasibility rate $0.740$ (exp1365). We position Carnot's Apache-2.0, local-first, hardware-portable design as the engineering substrate that makes cross-mechanism verifier diversity (the only known route past the Welch ceiling) physically deployable on consumer hardware.
```

Comments:

```text
Position paper draft v3; arXiv source bundle v11 prepared 2026-05-05.
```

Secondary categories, if the arXiv form offers them and the operator wants the
same routing as the existing metadata file:

```text
cs.AI, cs.NE, quant-ph
```

## Browser Upload Steps

1. Screen: Start. Open `https://arxiv.org/submit` and sign in to the operator arXiv account.
2. Screen: New submission. Choose to start a new submission and select the compressed TeX/source upload path.
3. Screen: Upload source. Upload `/home/ianblenke/github.com/ianblenke/carnot/results/arxiv_bundle_v11.tar.gz`.
4. Screen: Process source. Wait for AutoTeX to process the archive. If arXiv reports a fatal TeX error, stop and fix the local source before submitting.
5. Screen: Preview. Open the generated PDF preview and compare it with `docs/arxiv-paper/main.pdf`.
6. Screen: Classification. Set the primary category to `cs.LG`.
7. Screen: Metadata. Paste the title, author, abstract, comments, and license exactly from the pre-filled metadata above.
8. Screen: License. Choose Creative Commons Attribution 4.0 International (`CC-BY-4.0`).
9. Screen: Final review. Confirm figures, references, title, abstract, author, category, and license render correctly.
10. Screen: Submit. Submit the paper and record the returned arXiv identifier in `results/experiment_1390_arxiv_submission_sword_api.json`.
