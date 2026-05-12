# Energy-Based Transformer / NRGPT Capabilities

## REQ-NRGPT-001: Diagnostic Loss Function
The system must support evaluating a diagnostic loss function based on NRGPT energy descent to compare gradient stability against standard cross-entropy.

## REQ-NRGPT-002: EBT Reasoning Bridge
The system must connect the continuous latent sampler with the EBT energy minimizer, using explicit EBT gradients to measure end-to-end convergence on a logical reasoning trace.

## REQ-NRGPT-003: EBT Compatibility Energy
The system must provide a basic EBT compatibility checking prototype that outputs a scalar compatibility energy for sequence pairs.

## SCENARIO-NRGPT-001: Local Overfitting Resistance
Given a toy reasoning dataset, the diagnostic loss mechanism can simulate descent to show asymptotic stability or overfitting resistance.

## SCENARIO-NRGPT-002: Continuous Latent Reasoning Convergence
Given fixed context embeddings, the reasoning bridge connects the ContinuousLatentSampler with EBT to refine candidate embeddings and measures convergence properties.

## SCENARIO-NRGPT-003: Compatibility Checking
Given an input sequence and a predicted sequence, the EBT compatibility prototype computes the scalar compatibility energy, compares this approach with conditional log-probability, and reports the energy descent curve during optimization.
