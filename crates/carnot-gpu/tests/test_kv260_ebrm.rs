//! Tests for the Exp 1657 KV260 EBRM trace-scoring binding.
//! Spec: REQ-VERIFY-1657, SCENARIO-VERIFY-1657.

use std::cell::Cell;
use std::path::{Path, PathBuf};

use carnot_gpu::kv260_ebrm::{
    build_artifact, default_trace_cases, load_upstream_synthesis_success, validate_artifact,
    write_artifact, write_default_artifact, ArtifactOptions, Kv260EbrmConfig, Kv260EbrmError,
    Kv260EbrmScorer, Kv260ExecutionResult, Kv260Trace, Kv260TraceExecutor, Kv260TraceProblem,
    LogicalTraceStep, SoftwareKv260Executor, DEFAULT_ARTIFACT_PATH, REQUIRED_ARTIFACT_FIELDS,
    SPEC_TRACES,
};

fn coherent_trace() -> Kv260Trace {
    Kv260Trace::new(
        "coherent",
        Some(false),
        vec![
            LogicalTraceStep::new("s1", "inventory_total_is_five", true)
                .with_confidence(0.94)
                .with_constraint_ids(["counting"]),
            LogicalTraceStep::new("s2", "answer_uses_inventory_total", true)
                .with_confidence(0.91)
                .with_supports(["s1"])
                .with_constraint_ids(["counting", "answer_grounding"]),
        ],
    )
}

fn contradictory_trace() -> Kv260Trace {
    Kv260Trace::new(
        "contradictory",
        Some(true),
        vec![
            LogicalTraceStep::new("s1", "inventory_total_is_five", true)
                .with_confidence(0.94)
                .with_constraint_ids(["counting"]),
            LogicalTraceStep::new("s2", "inventory_total_is_five", false)
                .with_confidence(0.93)
                .with_contradicts(["s1"])
                .with_constraint_ids(["counting"]),
        ],
    )
}

#[test]
fn test_req_verify_1657_scores_preserve_ebrm_components() {
    let scorer = Kv260EbrmScorer::default();
    let executor = SoftwareKv260Executor;

    let coherent_problem = scorer
        .encode_trace(&coherent_trace())
        .expect("coherent trace encodes");
    let contradictory_problem = scorer
        .encode_trace(&contradictory_trace())
        .expect("contradictory trace encodes");

    assert_eq!(coherent_problem.potts_q_states, 3);
    assert_eq!(coherent_problem.potts_states, vec![0, 0]);
    assert_eq!(contradictory_problem.potts_states, vec![0, 2]);

    let coherent = scorer
        .score_trace(&coherent_trace(), &executor)
        .expect("coherent trace scores");
    let contradictory = scorer
        .score_trace(&contradictory_trace(), &executor)
        .expect("contradictory trace scores");

    assert!(coherent.energy >= 0.0);
    assert!(contradictory.energy > coherent.energy);
    assert!(contradictory.coherence_score < coherent.coherence_score);
    assert!(contradictory.component_energies.contradiction_energy > 0.0);
    assert_eq!(
        contradictory.component_energies.total(),
        contradictory.energy
    );
    assert_eq!(contradictory.backend, "software-kv260-potts");
    assert!(contradictory.continuous_energy_used);

    let weighted = Kv260EbrmScorer::new(Kv260EbrmConfig {
        contradiction_weight: 5.0,
        ..Kv260EbrmConfig::default()
    })
    .score_trace(&contradictory_trace(), &executor)
    .expect("weighted trace scores");
    assert!(weighted.energy > contradictory.energy);
}

struct RecordingExecutor {
    called: Cell<bool>,
}

impl RecordingExecutor {
    fn new() -> Self {
        Self {
            called: Cell::new(false),
        }
    }
}

impl Kv260TraceExecutor for RecordingExecutor {
    fn backend_name(&self) -> &'static str {
        "recording-kv260"
    }

    fn hardware_available(&self) -> bool {
        true
    }

    fn execute(&self, problem: &Kv260TraceProblem) -> Result<Kv260ExecutionResult, Kv260EbrmError> {
        self.called.set(true);
        assert_eq!(problem.potts_q_states, 3);
        assert_eq!(problem.trace_id, "contradictory");
        assert_eq!(problem.potts_states, vec![0, 2]);
        SoftwareKv260Executor.execute(problem)
    }
}

struct HardwarePassthroughExecutor;

impl Kv260TraceExecutor for HardwarePassthroughExecutor {
    fn backend_name(&self) -> &'static str {
        "hardware-kv260-mmio"
    }

    fn hardware_available(&self) -> bool {
        true
    }

    fn execute(&self, problem: &Kv260TraceProblem) -> Result<Kv260ExecutionResult, Kv260EbrmError> {
        SoftwareKv260Executor.execute(problem)
    }
}

struct FailingExecutor;

impl Kv260TraceExecutor for FailingExecutor {
    fn backend_name(&self) -> &'static str {
        "failing-kv260"
    }

    fn hardware_available(&self) -> bool {
        true
    }

    fn execute(
        &self,
        _problem: &Kv260TraceProblem,
    ) -> Result<Kv260ExecutionResult, Kv260EbrmError> {
        Err(Kv260EbrmError::Executor("forced failure".to_string()))
    }
}

#[test]
fn test_scenario_verify_1657_executor_trait_receives_encoded_problem() {
    let scorer = Kv260EbrmScorer::default();
    let executor = RecordingExecutor::new();

    let score = scorer
        .score_trace(&contradictory_trace(), &executor)
        .expect("recording executor scores");

    assert!(executor.called.get());
    assert_eq!(score.backend, "recording-kv260");
    assert!(score.hardware_execution_available);
    assert_eq!(score.potts_q_states, 3);
}

#[test]
fn test_req_verify_1657_invalid_inputs_fail_closed_before_execution() {
    let scorer = Kv260EbrmScorer::default();
    let executor = RecordingExecutor::new();

    let empty = Kv260Trace::new("empty", None, vec![]);
    assert!(matches!(
        scorer.score_trace(&empty, &executor),
        Err(Kv260EbrmError::InvalidTrace(_))
    ));
    assert!(!executor.called.get());

    let duplicate = Kv260Trace::new(
        "duplicate",
        None,
        vec![
            LogicalTraceStep::new("s1", "alpha", true),
            LogicalTraceStep::new("s1", "beta", true),
        ],
    );
    assert!(matches!(
        scorer.score_trace(&duplicate, &executor),
        Err(Kv260EbrmError::InvalidTrace(_))
    ));
    assert!(!executor.called.get());

    let empty_step_id = Kv260Trace::new(
        "empty-step-id",
        None,
        vec![LogicalTraceStep::new("", "alpha", true)],
    );
    assert!(matches!(
        scorer.score_trace(&empty_step_id, &executor),
        Err(Kv260EbrmError::InvalidTrace(_))
    ));

    let empty_proposition = Kv260Trace::new(
        "empty-proposition",
        None,
        vec![LogicalTraceStep::new("s1", " ", true)],
    );
    assert!(matches!(
        scorer.score_trace(&empty_proposition, &executor),
        Err(Kv260EbrmError::InvalidTrace(_))
    ));
}

#[test]
fn test_req_verify_1657_ordering_and_executor_error_paths() {
    let scorer = Kv260EbrmScorer::default();

    let repeated_same_truth = Kv260Trace::new(
        "repeated-same-truth",
        Some(false),
        vec![
            LogicalTraceStep::new("s1", "alpha", true).with_constraint_ids(["logic"]),
            LogicalTraceStep::new("s2", "alpha", true).with_constraint_ids(["logic"]),
        ],
    );
    let repeated_score = scorer
        .score_trace(&repeated_same_truth, &SoftwareKv260Executor)
        .expect("same-truth repetition scores");
    assert_eq!(repeated_score.component_energies.contradiction_energy, 0.0);

    let future_support = Kv260Trace::new(
        "future-support",
        Some(true),
        vec![
            LogicalTraceStep::new("s1", "route_is_valid", true)
                .with_supports(["s2"])
                .with_constraint_ids(["route"]),
            LogicalTraceStep::new("s2", "bridge_is_open", true).with_constraint_ids(["route"]),
        ],
    );
    let future_support_score = scorer
        .score_trace(&future_support, &SoftwareKv260Executor)
        .expect("future support scores");
    assert!(future_support_score.component_energies.ordering_energy > 0.0);
    assert_eq!(future_support_score.potts_states[0], 2);

    let future_contradiction = Kv260Trace::new(
        "future-contradiction",
        Some(true),
        vec![
            LogicalTraceStep::new("s1", "alpha", false)
                .with_contradicts(["s2"])
                .with_constraint_ids(["logic"]),
            LogicalTraceStep::new("s2", "alpha", true).with_constraint_ids(["logic"]),
        ],
    );
    let future_contradiction_score = scorer
        .score_trace(&future_contradiction, &SoftwareKv260Executor)
        .expect("future contradiction scores");
    assert!(
        future_contradiction_score
            .component_energies
            .ordering_energy
            > 0.0
    );
    assert!(
        future_contradiction_score
            .component_energies
            .contradiction_energy
            > 0.0
    );

    assert!(matches!(
        scorer.score_trace(&coherent_trace(), &FailingExecutor),
        Err(Kv260EbrmError::Executor(_))
    ));
}

#[test]
fn test_scenario_verify_1657_artifact_schema_and_writer() {
    let unique = format!(
        "carnot-kv260-ebrm-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    );
    let tmp_dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&tmp_dir).expect("tmp dir");
    let synthesis_path = tmp_dir.join("experiment_1649_vivado_synthesis.json");
    std::fs::write(&synthesis_path, r#"{"synthesis_success":true}"#).expect("synthesis fixture");
    let output_path = tmp_dir.join("experiment_1657_kv260_ebrm_binding.json");

    let options = ArtifactOptions {
        run_date: "20260509".to_string(),
        tests_run: vec!["crates/carnot-gpu/tests/test_kv260_ebrm.rs".to_string()],
        synthesis_artifact_path: synthesis_path.clone(),
        potts_rtl_path: PathBuf::from("hardware/kv260/potts_sampler_v1.v"),
    };

    let artifact = write_artifact(&output_path, options).expect("artifact writes");
    let persisted: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&output_path).expect("artifact file"))
            .expect("artifact json");

    validate_artifact(&artifact).expect("artifact validates");
    for field in REQUIRED_ARTIFACT_FIELDS {
        assert!(persisted.get(field).is_some(), "missing {field}");
    }
    assert_eq!(artifact.status, "complete");
    assert_eq!(artifact.experiment_id, 1657);
    assert_eq!(artifact.schema, "kv260_ebrm_binding_v1");
    assert!(artifact.kv260_ebrm_binding_ready);
    assert_eq!(artifact.potts_q_states, 3);
    assert_eq!(
        artifact.synthesis_artifact_path,
        synthesis_path.to_string_lossy()
    );
    assert!(artifact.upstream_synthesis_success);
    assert!(artifact.software_fallback_used);
    assert!(artifact.continuous_energy_used);
    assert!(artifact.energy_gap > 0.0);
    assert!(artifact.score_accuracy >= 0.8);
    assert_eq!(artifact.spec_traces, SPEC_TRACES);
    assert_eq!(
        artifact.tests_run,
        vec!["crates/carnot-gpu/tests/test_kv260_ebrm.rs"]
    );
    assert!(artifact.honest_verdict.starts_with("complete:"));

    std::fs::remove_dir_all(&tmp_dir).expect("cleanup");
}

#[test]
fn test_req_verify_1657_artifact_validation_catches_schema_drift() {
    let mut artifact = build_artifact(
        &[coherent_trace(), contradictory_trace()],
        ArtifactOptions::default(),
        &SoftwareKv260Executor,
    )
    .expect("artifact builds");
    validate_artifact(&artifact).expect("artifact validates");

    artifact.spec_traces = vec![];
    assert!(matches!(
        validate_artifact(&artifact),
        Err(Kv260EbrmError::InvalidArtifact(_))
    ));

    let mut bad_experiment = build_artifact(
        &[coherent_trace(), contradictory_trace()],
        ArtifactOptions::default(),
        &SoftwareKv260Executor,
    )
    .expect("artifact builds");
    bad_experiment.experiment_id = 0;
    assert!(matches!(
        validate_artifact(&bad_experiment),
        Err(Kv260EbrmError::InvalidArtifact(_))
    ));

    let mut bad_schema = build_artifact(
        &[coherent_trace(), contradictory_trace()],
        ArtifactOptions::default(),
        &SoftwareKv260Executor,
    )
    .expect("artifact builds");
    bad_schema.schema = "wrong".to_string();
    assert!(matches!(
        validate_artifact(&bad_schema),
        Err(Kv260EbrmError::InvalidArtifact(_))
    ));

    let mut out_of_range_accuracy = build_artifact(
        &[coherent_trace(), contradictory_trace()],
        ArtifactOptions::default(),
        &SoftwareKv260Executor,
    )
    .expect("artifact builds");
    out_of_range_accuracy.score_accuracy = 1.5;
    assert!(matches!(
        validate_artifact(&out_of_range_accuracy),
        Err(Kv260EbrmError::InvalidArtifact(_))
    ));

    let mut low_accuracy = build_artifact(
        &[coherent_trace(), contradictory_trace()],
        ArtifactOptions::default(),
        &SoftwareKv260Executor,
    )
    .expect("artifact builds");
    low_accuracy.status = "complete".to_string();
    low_accuracy.score_accuracy = 0.0;
    assert!(matches!(
        validate_artifact(&low_accuracy),
        Err(Kv260EbrmError::InvalidArtifact(_))
    ));

    let mut not_ready = build_artifact(
        &[coherent_trace(), contradictory_trace()],
        ArtifactOptions::default(),
        &SoftwareKv260Executor,
    )
    .expect("artifact builds");
    not_ready.status = "complete".to_string();
    not_ready.kv260_ebrm_binding_ready = false;
    assert!(matches!(
        validate_artifact(&not_ready),
        Err(Kv260EbrmError::InvalidArtifact(_))
    ));

    let mut no_continuous_energy = build_artifact(
        &[coherent_trace(), contradictory_trace()],
        ArtifactOptions::default(),
        &SoftwareKv260Executor,
    )
    .expect("artifact builds");
    no_continuous_energy.status = "complete".to_string();
    no_continuous_energy.continuous_energy_used = false;
    assert!(matches!(
        validate_artifact(&no_continuous_energy),
        Err(Kv260EbrmError::InvalidArtifact(_))
    ));

    let mut no_energy_gap = build_artifact(
        &[coherent_trace(), contradictory_trace()],
        ArtifactOptions::default(),
        &SoftwareKv260Executor,
    )
    .expect("artifact builds");
    no_energy_gap.status = "complete".to_string();
    no_energy_gap.energy_gap = 0.0;
    assert!(matches!(
        validate_artifact(&no_energy_gap),
        Err(Kv260EbrmError::InvalidArtifact(_))
    ));
}

#[test]
fn test_scenario_verify_1657_artifact_edge_paths_and_hardware_status() {
    let unique = format!(
        "carnot-kv260-ebrm-edge-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    );
    let tmp_dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&tmp_dir).expect("tmp dir");

    let missing_synthesis_path = tmp_dir.join("missing-synthesis.json");
    assert!(!load_upstream_synthesis_success(&missing_synthesis_path).expect("missing is false"));

    let invalid_synthesis_path = tmp_dir.join("invalid-synthesis.json");
    std::fs::write(&invalid_synthesis_path, "{not-json").expect("invalid fixture");
    assert!(matches!(
        load_upstream_synthesis_success(&invalid_synthesis_path),
        Err(Kv260EbrmError::Json(_))
    ));

    let blocked = build_artifact(&[], ArtifactOptions::default(), &SoftwareKv260Executor)
        .expect("blocked artifact builds");
    assert_eq!(blocked.status, "blocked");
    assert!(!blocked.kv260_ebrm_binding_ready);
    assert!(blocked.honest_verdict.starts_with("blocked:"));
    validate_artifact(&blocked).expect("blocked artifact validates");

    let hardware_artifact = build_artifact(
        &default_trace_cases(),
        ArtifactOptions::default(),
        &HardwarePassthroughExecutor,
    )
    .expect("hardware artifact builds");
    assert!(hardware_artifact.hardware_execution_available);
    assert!(!hardware_artifact.software_fallback_used);

    let no_parent_output = format!("kv260-ebrm-test-artifact-{}.json", std::process::id());
    let no_parent_artifact =
        write_artifact(&no_parent_output, ArtifactOptions::default()).expect("no-parent write");
    assert_eq!(no_parent_artifact.status, "complete");
    std::fs::remove_file(&no_parent_output).expect("cleanup no-parent artifact");

    let parent_file = tmp_dir.join("parent-file");
    std::fs::write(&parent_file, "not a directory").expect("parent fixture");
    let nested_under_file = parent_file.join("child.json");
    assert!(matches!(
        write_artifact(&nested_under_file, ArtifactOptions::default()),
        Err(Kv260EbrmError::Io(_))
    ));

    assert!(matches!(
        write_artifact(&tmp_dir, ArtifactOptions::default()),
        Err(Kv260EbrmError::Io(_))
    ));
    assert!(matches!(
        write_artifact("", ArtifactOptions::default()),
        Err(Kv260EbrmError::Io(_))
    ));

    let default_path = Path::new(DEFAULT_ARTIFACT_PATH);
    let previous_default_artifact = std::fs::read(default_path).ok();
    let default_artifact = write_default_artifact().expect("default artifact writes");
    assert_eq!(default_artifact.status, "complete");
    if let Some(previous) = previous_default_artifact {
        std::fs::write(default_path, previous).expect("restore default artifact");
    } else {
        std::fs::remove_file(default_path).expect("cleanup default artifact");
    }

    std::fs::remove_dir_all(&tmp_dir).expect("cleanup");
}
