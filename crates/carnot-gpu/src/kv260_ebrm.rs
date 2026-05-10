//! KV260 binding for EBRM trace-scoring constraints.
//!
//! The Python EBRM scorer works with extracted logical trace rows.  This module
//! keeps the same scoring semantics, but packages each trace as a q=3 Potts
//! problem so callers can swap the deterministic software executor for a KV260
//! MMIO executor later without changing the score schema.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const EXPERIMENT_ID: u32 = 1657;
pub const POTTS_Q_STATES: u8 = 3;
pub const DEFAULT_ARTIFACT_PATH: &str = "results/experiment_1657_kv260_ebrm_binding.json";
pub const DEFAULT_SYNTHESIS_ARTIFACT_PATH: &str = "results/experiment_1649_vivado_synthesis.json";
pub const DEFAULT_POTTS_RTL_PATH: &str = "hardware/kv260/potts_sampler_v1.v";
pub const SOFTWARE_KV260_BACKEND: &str = "software-kv260-potts";
pub const SPEC_TRACES: [&str; 2] = ["REQ-VERIFY-1657", "SCENARIO-VERIFY-1657"];
pub const REQUIRED_ARTIFACT_FIELDS: &[&str] = &[
    "status",
    "experiment_id",
    "schema",
    "kv260_ebrm_binding_ready",
    "continuous_energy_used",
    "potts_q_states",
    "potts_rtl_path",
    "synthesis_artifact_path",
    "upstream_synthesis_success",
    "hardware_execution_available",
    "software_fallback_used",
    "synthetic_cases_total",
    "consistent_cases",
    "inconsistent_cases",
    "consistent_mean_energy",
    "inconsistent_mean_energy",
    "energy_gap",
    "score_accuracy",
    "case_scores",
    "spec_traces",
    "tests_run",
    "honest_verdict",
];

const POTTS_STATE_SATISFIED: u8 = 0;
const POTTS_STATE_PARTIAL: u8 = 1;
const POTTS_STATE_VIOLATED: u8 = 2;
const ACCURACY_GATE: f32 = 0.8;

#[derive(Error, Debug)]
pub enum Kv260EbrmError {
    #[error("invalid trace: {0}")]
    InvalidTrace(String),
    #[error("executor failed: {0}")]
    Executor(String),
    #[error("artifact validation failed: {0}")]
    InvalidArtifact(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LogicalTraceStep {
    pub step_id: String,
    pub proposition: String,
    pub truth_value: bool,
    pub confidence: f32,
    pub supports: Vec<String>,
    pub contradicts: Vec<String>,
    pub constraint_ids: Vec<String>,
}

impl LogicalTraceStep {
    pub fn new(
        step_id: impl Into<String>,
        proposition: impl Into<String>,
        truth_value: bool,
    ) -> Self {
        Self {
            step_id: step_id.into(),
            proposition: proposition.into(),
            truth_value,
            confidence: 1.0,
            supports: Vec::new(),
            contradicts: Vec::new(),
            constraint_ids: Vec::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_supports<I, S>(mut self, supports: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.supports = supports.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_contradicts<I, S>(mut self, contradicts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.contradicts = contradicts.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_constraint_ids<I, S>(mut self, constraint_ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.constraint_ids = constraint_ids.into_iter().map(Into::into).collect();
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Kv260Trace {
    pub trace_id: String,
    pub expected_inconsistent: Option<bool>,
    pub steps: Vec<LogicalTraceStep>,
}

impl Kv260Trace {
    pub fn new(
        trace_id: impl Into<String>,
        expected_inconsistent: Option<bool>,
        steps: Vec<LogicalTraceStep>,
    ) -> Self {
        Self {
            trace_id: trace_id.into(),
            expected_inconsistent,
            steps,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Kv260EbrmConfig {
    pub contradiction_weight: f32,
    pub explicit_contradiction_bonus: f32,
    pub unsupported_weight: f32,
    pub confidence_weight: f32,
    pub coverage_weight: f32,
    pub ordering_weight: f32,
    pub min_confidence: f32,
    pub prediction_threshold: f32,
    pub coherence_temperature: f32,
}

impl Default for Kv260EbrmConfig {
    fn default() -> Self {
        Self {
            contradiction_weight: 3.0,
            explicit_contradiction_bonus: 0.75,
            unsupported_weight: 1.25,
            confidence_weight: 1.0,
            coverage_weight: 0.65,
            ordering_weight: 0.4,
            min_confidence: 0.75,
            prediction_threshold: 1.0,
            coherence_temperature: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ComponentEnergies {
    pub contradiction_energy: f32,
    pub unsupported_energy: f32,
    pub confidence_energy: f32,
    pub coverage_energy: f32,
    pub ordering_energy: f32,
}

impl ComponentEnergies {
    pub fn total(&self) -> f32 {
        round6(
            self.contradiction_energy
                + self.unsupported_energy
                + self.confidence_energy
                + self.coverage_energy
                + self.ordering_energy,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Kv260TraceProblem {
    pub trace_id: String,
    pub potts_q_states: u8,
    pub potts_states: Vec<u8>,
    pub step_ids: Vec<String>,
    pub component_energies: ComponentEnergies,
    pub violation_count: u32,
    pub energy: f32,
    pub coherence_score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Kv260ExecutionResult {
    pub energy: f32,
    pub coherence_score: f32,
    pub component_energies: ComponentEnergies,
    pub violation_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Kv260TraceScore {
    pub trace_id: String,
    pub energy: f32,
    pub coherence_score: f32,
    pub component_energies: ComponentEnergies,
    pub violation_count: u32,
    pub continuous_energy_used: bool,
    pub potts_q_states: u8,
    pub potts_states: Vec<u8>,
    pub backend: String,
    pub hardware_execution_available: bool,
}

pub trait Kv260TraceExecutor {
    fn backend_name(&self) -> &'static str;
    fn hardware_available(&self) -> bool;
    fn execute(&self, problem: &Kv260TraceProblem) -> Result<Kv260ExecutionResult, Kv260EbrmError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SoftwareKv260Executor;

impl Kv260TraceExecutor for SoftwareKv260Executor {
    fn backend_name(&self) -> &'static str {
        SOFTWARE_KV260_BACKEND
    }

    fn hardware_available(&self) -> bool {
        false
    }

    fn execute(&self, problem: &Kv260TraceProblem) -> Result<Kv260ExecutionResult, Kv260EbrmError> {
        Ok(Kv260ExecutionResult {
            energy: problem.energy,
            coherence_score: problem.coherence_score,
            component_energies: problem.component_energies,
            violation_count: problem.violation_count,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct Kv260EbrmScorer {
    pub config: Kv260EbrmConfig,
}

impl Kv260EbrmScorer {
    pub fn new(config: Kv260EbrmConfig) -> Self {
        Self { config }
    }

    pub fn encode_trace(&self, trace: &Kv260Trace) -> Result<Kv260TraceProblem, Kv260EbrmError> {
        validate_trace(trace)?;

        let step_index: HashMap<&str, usize> = trace
            .steps
            .iter()
            .enumerate()
            .map(|(index, step)| (step.step_id.as_str(), index))
            .collect();
        let mut seen_by_prop: HashMap<&str, Vec<&LogicalTraceStep>> = HashMap::new();
        let mut potts_states = vec![POTTS_STATE_SATISFIED; trace.steps.len()];
        let mut component_energies = ComponentEnergies::default();
        let mut violation_count = 0u32;

        for (index, step) in trace.steps.iter().enumerate() {
            let bounded_confidence = clamp01(step.confidence);
            let confidence_delta = self.config.confidence_weight
                * (self.config.min_confidence - bounded_confidence).max(0.0);
            if confidence_delta > 0.0 {
                potts_states[index] = potts_states[index].max(POTTS_STATE_PARTIAL);
            }
            component_energies.confidence_energy += confidence_delta;

            if step.constraint_ids.is_empty() {
                potts_states[index] = potts_states[index].max(POTTS_STATE_PARTIAL);
                component_energies.coverage_energy += self.config.coverage_weight;
            }

            for prior in seen_by_prop
                .get(step.proposition.as_str())
                .into_iter()
                .flatten()
            {
                if prior.truth_value != step.truth_value {
                    component_energies.contradiction_energy += self.config.contradiction_weight
                        * bounded_confidence
                        * clamp01(prior.confidence);
                    violation_count += 1;
                    potts_states[index] = POTTS_STATE_VIOLATED;
                }
            }

            for linked_step_id in &step.supports {
                let linked_index = step_index.get(linked_step_id.as_str()).copied();
                let unsupported = linked_index.is_none();
                let out_of_order = linked_index.is_some_and(|linked| linked >= index);
                if unsupported || out_of_order {
                    violation_count += 1;
                    potts_states[index] = POTTS_STATE_VIOLATED;
                }
                component_energies.unsupported_energy +=
                    self.config.unsupported_weight * f32::from(unsupported);
                component_energies.ordering_energy +=
                    self.config.ordering_weight * f32::from(out_of_order);
            }

            for linked_step_id in &step.contradicts {
                let linked_index = step_index.get(linked_step_id.as_str()).copied();
                let linked_exists = linked_index.is_some();
                let out_of_order = linked_index.is_some_and(|linked| linked >= index);
                if linked_exists || out_of_order {
                    violation_count += 1;
                    potts_states[index] = POTTS_STATE_VIOLATED;
                }
                component_energies.contradiction_energy +=
                    self.config.explicit_contradiction_bonus * f32::from(linked_exists);
                component_energies.ordering_energy +=
                    self.config.ordering_weight * f32::from(out_of_order);
            }

            seen_by_prop
                .entry(step.proposition.as_str())
                .or_default()
                .push(step);
        }

        component_energies.contradiction_energy = round6(component_energies.contradiction_energy);
        component_energies.unsupported_energy = round6(component_energies.unsupported_energy);
        component_energies.confidence_energy = round6(component_energies.confidence_energy);
        component_energies.coverage_energy = round6(component_energies.coverage_energy);
        component_energies.ordering_energy = round6(component_energies.ordering_energy);

        let energy = component_energies.total();
        let coherence_score = round6((-energy / self.config.coherence_temperature).exp());

        Ok(Kv260TraceProblem {
            trace_id: trace.trace_id.clone(),
            potts_q_states: POTTS_Q_STATES,
            potts_states,
            step_ids: trace
                .steps
                .iter()
                .map(|step| step.step_id.clone())
                .collect(),
            component_energies,
            violation_count,
            energy,
            coherence_score,
        })
    }

    pub fn score_trace<E: Kv260TraceExecutor>(
        &self,
        trace: &Kv260Trace,
        executor: &E,
    ) -> Result<Kv260TraceScore, Kv260EbrmError> {
        let problem = self.encode_trace(trace)?;
        let result = executor.execute(&problem)?;
        Ok(Kv260TraceScore {
            trace_id: problem.trace_id,
            energy: result.energy,
            coherence_score: result.coherence_score,
            component_energies: result.component_energies,
            violation_count: result.violation_count,
            continuous_energy_used: true,
            potts_q_states: problem.potts_q_states,
            potts_states: problem.potts_states,
            backend: executor.backend_name().to_string(),
            hardware_execution_available: executor.hardware_available(),
        })
    }

    pub fn score_traces<E: Kv260TraceExecutor>(
        &self,
        traces: &[Kv260Trace],
        executor: &E,
    ) -> Result<Vec<Kv260TraceScore>, Kv260EbrmError> {
        traces
            .iter()
            .map(|trace| self.score_trace(trace, executor))
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ArtifactOptions {
    pub run_date: String,
    pub tests_run: Vec<String>,
    pub synthesis_artifact_path: PathBuf,
    pub potts_rtl_path: PathBuf,
}

impl Default for ArtifactOptions {
    fn default() -> Self {
        Self {
            run_date: "20260509".to_string(),
            tests_run: Vec::new(),
            synthesis_artifact_path: PathBuf::from(DEFAULT_SYNTHESIS_ARTIFACT_PATH),
            potts_rtl_path: PathBuf::from(DEFAULT_POTTS_RTL_PATH),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Kv260EbrmArtifact {
    pub status: String,
    pub experiment_id: u32,
    pub schema: String,
    pub run_date: String,
    pub kv260_ebrm_binding_ready: bool,
    pub continuous_energy_used: bool,
    pub potts_q_states: u8,
    pub potts_rtl_path: String,
    pub synthesis_artifact_path: String,
    pub upstream_synthesis_success: bool,
    pub hardware_execution_available: bool,
    pub software_fallback_used: bool,
    pub synthetic_cases_total: usize,
    pub consistent_cases: usize,
    pub inconsistent_cases: usize,
    pub consistent_mean_energy: f32,
    pub inconsistent_mean_energy: f32,
    pub energy_gap: f32,
    pub score_accuracy: f32,
    pub case_scores: Vec<Kv260TraceScore>,
    pub spec_traces: Vec<String>,
    pub tests_run: Vec<String>,
    pub honest_verdict: String,
}

pub fn default_trace_cases() -> Vec<Kv260Trace> {
    vec![
        Kv260Trace::new(
            "inventory-consistent",
            Some(false),
            vec![
                LogicalTraceStep::new("s1", "inventory_total_is_five", true)
                    .with_confidence(0.96)
                    .with_constraint_ids(["counting"]),
                LogicalTraceStep::new("s2", "answer_uses_inventory_total", true)
                    .with_confidence(0.93)
                    .with_supports(["s1"])
                    .with_constraint_ids(["answer_grounding"]),
            ],
        ),
        Kv260Trace::new(
            "inventory-contradiction",
            Some(true),
            vec![
                LogicalTraceStep::new("s1", "inventory_total_is_five", true)
                    .with_confidence(0.96)
                    .with_constraint_ids(["counting"]),
                LogicalTraceStep::new("s2", "inventory_total_is_five", false)
                    .with_confidence(0.95)
                    .with_contradicts(["s1"])
                    .with_constraint_ids(["counting"]),
            ],
        ),
        Kv260Trace::new(
            "route-consistent",
            Some(false),
            vec![
                LogicalTraceStep::new("s1", "east_bridge_open", true)
                    .with_confidence(0.91)
                    .with_constraint_ids(["route"]),
                LogicalTraceStep::new("s2", "route_uses_east_bridge", true)
                    .with_confidence(0.9)
                    .with_supports(["s1"])
                    .with_constraint_ids(["route"]),
            ],
        ),
        Kv260Trace::new(
            "route-unsupported",
            Some(true),
            vec![
                LogicalTraceStep::new("s1", "east_bridge_open", true)
                    .with_confidence(0.91)
                    .with_constraint_ids(["route"]),
                LogicalTraceStep::new("s2", "route_is_valid", true)
                    .with_confidence(0.52)
                    .with_supports(["missing-route-link"]),
            ],
        ),
    ]
}

pub fn build_artifact<E: Kv260TraceExecutor>(
    cases: &[Kv260Trace],
    options: ArtifactOptions,
    executor: &E,
) -> Result<Kv260EbrmArtifact, Kv260EbrmError> {
    let scorer = Kv260EbrmScorer::default();
    let scores = scorer.score_traces(cases, executor)?;
    let metrics = aggregate_scores(cases, &scores);
    let upstream_synthesis_success =
        load_upstream_synthesis_success(&options.synthesis_artifact_path)?;
    let continuous_energy_used = scores.iter().all(|score| score.continuous_energy_used);
    let ready = continuous_energy_used
        && metrics.energy_gap > 0.0
        && metrics.score_accuracy >= ACCURACY_GATE;

    let artifact = Kv260EbrmArtifact {
        status: if ready { "complete" } else { "blocked" }.to_string(),
        experiment_id: EXPERIMENT_ID,
        schema: "kv260_ebrm_binding_v1".to_string(),
        run_date: options.run_date,
        kv260_ebrm_binding_ready: ready,
        continuous_energy_used,
        potts_q_states: POTTS_Q_STATES,
        potts_rtl_path: options.potts_rtl_path.to_string_lossy().into_owned(),
        synthesis_artifact_path: options
            .synthesis_artifact_path
            .to_string_lossy()
            .into_owned(),
        upstream_synthesis_success,
        hardware_execution_available: executor.hardware_available(),
        software_fallback_used: !executor.hardware_available()
            || executor.backend_name() == SOFTWARE_KV260_BACKEND,
        synthetic_cases_total: scores.len(),
        consistent_cases: metrics.consistent_cases,
        inconsistent_cases: metrics.inconsistent_cases,
        consistent_mean_energy: metrics.consistent_mean_energy,
        inconsistent_mean_energy: metrics.inconsistent_mean_energy,
        energy_gap: metrics.energy_gap,
        score_accuracy: metrics.score_accuracy,
        case_scores: scores,
        spec_traces: SPEC_TRACES
            .iter()
            .map(|trace| (*trace).to_string())
            .collect(),
        tests_run: options.tests_run,
        honest_verdict: honest_verdict(ready, metrics.score_accuracy),
    };
    validate_artifact(&artifact)?;
    Ok(artifact)
}

pub fn write_artifact(
    output_path: impl AsRef<Path>,
    options: ArtifactOptions,
) -> Result<Kv260EbrmArtifact, Kv260EbrmError> {
    let artifact = build_artifact(&default_trace_cases(), options, &SoftwareKv260Executor)?;
    let path = output_path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(&artifact)? + "\n")?;
    Ok(artifact)
}

pub fn write_default_artifact() -> Result<Kv260EbrmArtifact, Kv260EbrmError> {
    write_artifact(DEFAULT_ARTIFACT_PATH, ArtifactOptions::default())
}

pub fn validate_artifact(artifact: &Kv260EbrmArtifact) -> Result<(), Kv260EbrmError> {
    if artifact.experiment_id != EXPERIMENT_ID {
        return Err(Kv260EbrmError::InvalidArtifact(
            "experiment_id mismatch".to_string(),
        ));
    }
    if artifact.schema != "kv260_ebrm_binding_v1" {
        return Err(Kv260EbrmError::InvalidArtifact(
            "schema mismatch".to_string(),
        ));
    }
    if artifact.spec_traces != SPEC_TRACES {
        return Err(Kv260EbrmError::InvalidArtifact(
            "spec_traces mismatch".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&artifact.score_accuracy) {
        return Err(Kv260EbrmError::InvalidArtifact(
            "score_accuracy out of range".to_string(),
        ));
    }
    if artifact.status == "complete" {
        if !artifact.kv260_ebrm_binding_ready {
            return Err(Kv260EbrmError::InvalidArtifact(
                "complete artifact requires ready binding".to_string(),
            ));
        }
        if !artifact.continuous_energy_used {
            return Err(Kv260EbrmError::InvalidArtifact(
                "complete artifact requires continuous energy".to_string(),
            ));
        }
        if artifact.energy_gap <= 0.0 {
            return Err(Kv260EbrmError::InvalidArtifact(
                "complete artifact requires positive energy gap".to_string(),
            ));
        }
        if artifact.score_accuracy < ACCURACY_GATE {
            return Err(Kv260EbrmError::InvalidArtifact(
                "complete artifact requires accuracy gate".to_string(),
            ));
        }
    }
    Ok(())
}

pub fn load_upstream_synthesis_success(path: impl AsRef<Path>) -> Result<bool, Kv260EbrmError> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(false);
    }
    let payload: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(payload
        .get("synthesis_success")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false))
}

#[derive(Clone, Copy, Debug)]
struct AggregateMetrics {
    consistent_cases: usize,
    inconsistent_cases: usize,
    consistent_mean_energy: f32,
    inconsistent_mean_energy: f32,
    energy_gap: f32,
    score_accuracy: f32,
}

fn aggregate_scores(cases: &[Kv260Trace], scores: &[Kv260TraceScore]) -> AggregateMetrics {
    let mut consistent = Vec::new();
    let mut inconsistent = Vec::new();
    let mut correct = 0usize;
    let threshold = Kv260EbrmConfig::default().prediction_threshold;

    for (case, score) in cases.iter().zip(scores.iter()) {
        let expected = case.expected_inconsistent.unwrap_or(false);
        let predicted = score.energy >= threshold;
        correct += usize::from(predicted == expected);
        if expected {
            inconsistent.push(score.energy);
        } else {
            consistent.push(score.energy);
        }
    }

    let consistent_mean_energy = mean(&consistent);
    let inconsistent_mean_energy = mean(&inconsistent);
    AggregateMetrics {
        consistent_cases: consistent.len(),
        inconsistent_cases: inconsistent.len(),
        consistent_mean_energy,
        inconsistent_mean_energy,
        energy_gap: round6(inconsistent_mean_energy - consistent_mean_energy),
        score_accuracy: round6(correct as f32 / scores.len().max(1) as f32),
    }
}

fn validate_trace(trace: &Kv260Trace) -> Result<(), Kv260EbrmError> {
    if trace.steps.is_empty() {
        return Err(Kv260EbrmError::InvalidTrace(
            "logical trace must contain at least one step".to_string(),
        ));
    }

    let mut seen = HashSet::new();
    for step in &trace.steps {
        if step.step_id.trim().is_empty() {
            return Err(Kv260EbrmError::InvalidTrace(
                "step_id must be non-empty".to_string(),
            ));
        }
        if step.proposition.trim().is_empty() {
            return Err(Kv260EbrmError::InvalidTrace(
                "proposition must be non-empty".to_string(),
            ));
        }
        if !seen.insert(step.step_id.as_str()) {
            return Err(Kv260EbrmError::InvalidTrace(format!(
                "duplicate step_id: {}",
                step.step_id
            )));
        }
    }
    Ok(())
}

fn honest_verdict(ready: bool, score_accuracy: f32) -> String {
    if ready {
        format!(
            "complete: KV260 EBRM binding separates trace constraints with score_accuracy={score_accuracy}"
        )
    } else {
        format!(
            "blocked: KV260 EBRM binding did not satisfy separation gate; score_accuracy={score_accuracy}"
        )
    }
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn mean(values: &[f32]) -> f32 {
    round6(values.iter().sum::<f32>() / values.len().max(1) as f32)
}

fn round6(value: f32) -> f32 {
    (value * 1_000_000.0).round() / 1_000_000.0
}
