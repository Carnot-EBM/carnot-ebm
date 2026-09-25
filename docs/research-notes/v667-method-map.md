# V667 source-witness method map

Date: 2026-09-25. Scope: contract and method limits for the V667 plan.
These papers inform controls; their results are not Carnot measurements.

| Source | Local use | Limit |
|---|---|---|
| [Static-analysis detection study](https://arxiv.org/abs/2604.07755) | Exp7644 records structural witness coverage and explicit unknowns; Exp7648 compares the strongest cheap structural control. | The reviewed abstract reports partial detection of library hallucinations. It does not validate arbitrary claims or transfer an accuracy rate to supplied Python source. |
| [Hallucination Inspector](https://arxiv.org/abs/2604.20202) | Exp7644 ties AST symbols to exact supplied source spans; Exp7646 checks source-group coverage and erasure. | Its API migration setting differs from this source-question setting. A symbol match does not prove full-answer truth. |
| [Calibeating Made Simple](https://arxiv.org/html/2603.22167v1) | Exp7649 predicts before delayed feedback, uses source-conditioned proper-loss updates, and compares matched scalar-only updates. | Immediate online guarantees do not automatically cover delayed admission, sparse source strata, or retention after restart. |
| [Proper Calibeating](https://arxiv.org/abs/2605.26703) | Exp7648 and Exp7649 keep proper loss separate from typed decision value and retention. | A finite held-back study is not a theorem reproduction. |
| [EAEV](https://arxiv.org/html/2609.08267v1) and [Beyond Document Grounding](https://arxiv.org/abs/2607.00895) | Exp7646 and Exp7650 preserve source offsets, erasure, derangement, and label provenance. | Existing exposed code groups and injected labels do not establish fresh oracle-distinct gain. |
| [JSONSchemaBench](https://arxiv.org/abs/2501.10868) | Exp7651 separates grammar validity, semantic support, and cost. | A bounded Qwen challenge is not the benchmark. |
| [EBT](https://arxiv.org/abs/2507.02092) and [ARM-EBM](https://arxiv.org/abs/2512.15605) | Exp7647 normalizes a small two-state decision energy. | Low energy is not a factual-correctness guarantee; no generator weights change. |
| [FPGA-ASIC co-design](https://arxiv.org/abs/2602.15985) | Exp7655 counts parsing, calls, persistence, dispatch, and device cost. | Arithmetic-only speed does not prove whole-consumer advantage. |

Static analysis and the Inspector motivate a narrow source witness. The witness
may abstain, and an exact structural answer is an oracle fixture for that
claim only. The planned learned advantage requires independent labels and
held-back source groups. Probability loss, decision value, retention, and
freshness each need their own observations.

The V666 GPU capacity block remains a resource finding. The ARC goal-guard
validation failure remains a separate invalid-validation finding. Neither is a
new source-witness result. The existing publication gates G1-G4 remain fixed;
this administrative contract opens none of them.
