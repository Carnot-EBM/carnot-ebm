# Energy-Based Transformer / NRGPT Capabilities

## REQ-NRGPT-001: Diagnostic Loss Function
The system must support evaluating a diagnostic loss function based on NRGPT energy descent to compare gradient stability against standard cross-entropy.

## REQ-NRGPT-002: EBT Reasoning Bridge
The system must connect the continuous latent sampler with the EBT energy minimizer, using explicit EBT gradients to measure end-to-end convergence on a logical reasoning trace.

## SCENARIO-NRGPT-001: Local Overfitting Resistance
Given a toy reasoning dataset, the diagnostic loss mechanism can simulate descent to show asymptotic stability or overfitting resistance.

## SCENARIO-NRGPT-002: Continuous Latent Reasoning Convergence
Given fixed context embeddings, the reasoning bridge connects the ContinuousLatentSampler with EBT to refine candidate embeddings and measures convergence properties.
