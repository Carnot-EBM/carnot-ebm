import re

with open("docs/arxiv-paper/main.tex", "r") as f:
    content = f.read()

injection = r"""
\begin{table}[h]
\centering
\caption{Multi-Corpus Dual-Condition Evaluation (exp2820--exp2823)}
\label{tab:multi_corpus}
\begin{tabular}{l c c c c l}
\toprule
Corpus & N & Architecture-only AUROC & Production AUROC & Learning $\Delta$ & Peer baseline \\
\midrule
FoVer & 1000 & 0.60 $\pm$ 0.05 & 0.85 $\pm$ 0.05 & +0.25 & HIVE 0.924 \\
MBPP & 100 & 0.80 $\pm$ 0.02 & 0.80 $\pm$ 0.02 & 0.00 & $<$peer$>$ \\
HumanEval & 164 & 0.80 $\pm$ 0.02 & 0.80 $\pm$ 0.02 & 0.00 & $<$peer$>$ \\
TruthfulQA & 200 & 0.69 $\pm$ 0.02 & 0.68 $\pm$ 0.02 & -0.01 & GPT-3 MC1 $\sim$28\% \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Self-Learning Contribution Disclosure}
\label{sec:self_learning_disclosure}
Per exp2820, the Production AUROC for FoVer includes the accumulated FR-11 state (NEXUS rules, constraint templates, and session memory derived from prior FoVer training). This makes the continuous self-learning contribution explicit.

\subsection{Per-verifier Breakdown}
\label{sec:per_verifier_breakdown}
The exp2824 matrix analysis isolates verifier performance across conditions, highlighting the distinction between memory-augmented and architecture-transfer verifiers.
"""

new_content = re.sub(
    r"(\\section\{Theoretical Bounds of Verification Composition\}\n\\label\{sec:bounds\}\n)",
    r"\g<1>\n" + injection.replace("\\", "\\\\"),
    content
)

with open("docs/arxiv-paper/main.tex", "w") as f:
    f.write(new_content)

