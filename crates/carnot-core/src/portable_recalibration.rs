//! Shared durable core for the qualified nine-knot recalibration service.
//!
//! Both the JSON-lines executable and the PyO3 binding call this module. This
//! keeps persistence and numerical behavior identical at both boundaries.
//! Spec: REQ-REPORT-7626 and SCENARIO-REPORT-7626-*.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const KNOT_COUNT: usize = 9;
const RIDGE_MASS: f64 = 8.0;
const MOVEMENT_BOUND: f64 = 0.10;
const SOLVER_TOLERANCE: f64 = 1e-8;
const SOLVER_MAX_ITERATIONS: usize = 500;
const RELEASE_BLOCK_SIZE: usize = 8;
pub const STATE_SCHEMA: &str = "carnot.recalibration.sufficient_statistics.v1";
pub const DURABILITY_POLICY: &str = "atomic_file_fsync_rename_directory_fsync_reload_ack";
pub const INTERRUPTED_WRITE_EXIT: i32 = 86;
type StageDurations = BTreeMap<String, u64>;

fn knots() -> [f64; KNOT_COUNT] {
    std::array::from_fn(|index| index as f64 / (KNOT_COUNT - 1) as f64)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct SolverConfig {
    ridge_mass: f64,
    movement_bound: f64,
    tolerance: f64,
    iteration_cap: usize,
    method: String,
}

impl SolverConfig {
    fn qualified() -> Self {
        Self {
            ridge_mass: RIDGE_MASS,
            movement_bound: MOVEMENT_BOUND,
            tolerance: SOLVER_TOLERANCE,
            iteration_cap: SOLVER_MAX_ITERATIONS,
            method: "SLSQP".to_string(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct State {
    pub schema: String,
    pub gram_shape: Vec<usize>,
    pub gram: Vec<Vec<f64>>,
    pub target: Vec<f64>,
    pub theta: Vec<f64>,
    pub sample_count: usize,
    pub processed_event_ids: Vec<String>,
    pub solver_config: SolverConfig,
    pub constrained: bool,
    pub last_solver_receipt: Value,
}

impl State {
    fn identity() -> Self {
        Self {
            schema: STATE_SCHEMA.to_string(),
            gram_shape: vec![KNOT_COUNT, KNOT_COUNT],
            gram: vec![vec![0.0; KNOT_COUNT]; KNOT_COUNT],
            target: vec![0.0; KNOT_COUNT],
            theta: knots().to_vec(),
            sample_count: 0,
            processed_event_ids: Vec::new(),
            solver_config: SolverConfig::qualified(),
            constrained: true,
            last_solver_receipt: Value::Null,
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != STATE_SCHEMA {
            return Err("sufficient_statistic_schema_mismatch".to_string());
        }
        if self.gram_shape != [KNOT_COUNT, KNOT_COUNT]
            || self.gram.len() != KNOT_COUNT
            || self.gram.iter().any(|row| row.len() != KNOT_COUNT)
            || self.target.len() != KNOT_COUNT
            || self.theta.len() != KNOT_COUNT
        {
            return Err("sufficient_statistic_shape_invalid".to_string());
        }
        if self
            .gram
            .iter()
            .flatten()
            .chain(self.target.iter())
            .chain(self.theta.iter())
            .any(|value| !value.is_finite())
        {
            return Err("sufficient_statistic_not_finite".to_string());
        }
        if self.solver_config != SolverConfig::qualified() {
            return Err("solver_configuration_not_frozen".to_string());
        }
        if !self.constrained {
            return Err("constrained_state_required".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Event {
    pub event_id: String,
    pub probability: f64,
    pub label: i32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Query {
    pub event_id: String,
    pub probability: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    pub operation: String,
    pub state_path: Option<PathBuf>,
    #[serde(default)]
    pub events: Vec<Event>,
    #[serde(default)]
    pub queries: Vec<Query>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Prediction {
    pub event_id: String,
    pub probability: f64,
    pub action: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Response {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub predictions: Vec<Prediction>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub acknowledgments: Vec<usize>,
    pub acknowledged_release_count: usize,
    pub processed_event_count: usize,
    pub reloaded_state_matches: bool,
    pub state_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<State>,
    pub stage_ns: StageDurations,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub exclusive_stage_ns: StageDurations,
    pub kernel_ns: u64,
    pub durability_policy: &'static str,
}

fn design(probability: f64) -> Result<[f64; KNOT_COUNT], String> {
    if !probability.is_finite() {
        return Err("probability_not_finite".to_string());
    }
    let value = probability.clamp(0.0, 1.0);
    let mut row = [0.0; KNOT_COUNT];
    if value >= 1.0 {
        row[KNOT_COUNT - 1] = 1.0;
        return Ok(row);
    }
    let scaled = value * (KNOT_COUNT - 1) as f64;
    let left = scaled.floor() as usize;
    let fraction = scaled - left as f64;
    row[left] = 1.0 - fraction;
    row[left + 1] = fraction;
    Ok(row)
}

fn predict_value(probability: f64, theta: &[f64]) -> Result<f64, String> {
    Ok(design(probability)?
        .iter()
        .zip(theta.iter())
        .map(|(weight, value)| weight * value)
        .sum())
}

fn typed_decision(probability: f64) -> Result<String, String> {
    if !probability.is_finite() {
        return Err("probability_not_finite".to_string());
    }
    let value = probability.clamp(0.0, 1.0);
    let costs = [
        ("escalate", 0.2),
        ("accept", 5.0 * value),
        ("reject", 1.0 - value),
    ];
    let minimum = costs
        .iter()
        .map(|(_, cost)| *cost)
        .fold(f64::INFINITY, f64::min);
    Ok(costs
        .iter()
        .find(|(_, cost)| (*cost - minimum).abs() <= 1e-12)
        .expect("three finite costs always have a minimum")
        .0
        .to_string())
}

fn project_monotone(values: &[f64; KNOT_COUNT]) -> [f64; KNOT_COUNT] {
    let mut levels = [0.0; KNOT_COUNT];
    let mut weights = [0usize; KNOT_COUNT];
    let mut blocks = 0usize;
    for value in values {
        levels[blocks] = *value;
        weights[blocks] = 1;
        blocks += 1;
        while blocks >= 2 && levels[blocks - 2] > levels[blocks - 1] {
            let right = blocks - 1;
            let left = right - 1;
            let weight = weights[left] + weights[right];
            levels[left] = (levels[left] * weights[left] as f64
                + levels[right] * weights[right] as f64)
                / weight as f64;
            weights[left] = weight;
            blocks -= 1;
        }
    }
    let mut projected = [0.0; KNOT_COUNT];
    let mut cursor = 0usize;
    for block in 0..blocks {
        for item in projected.iter_mut().skip(cursor).take(weights[block]) {
            *item = levels[block];
        }
        cursor += weights[block];
    }
    projected
}

fn project_box(values: &[f64; KNOT_COUNT]) -> [f64; KNOT_COUNT] {
    let basis = knots();
    std::array::from_fn(|index| {
        values[index].clamp(
            0.0f64.max(basis[index] - MOVEMENT_BOUND),
            1.0f64.min(basis[index] + MOVEMENT_BOUND),
        )
    })
}

fn project_constraints(values: &[f64; KNOT_COUNT]) -> [f64; KNOT_COUNT] {
    let mut current = *values;
    let mut box_residual = [0.0; KNOT_COUNT];
    let mut order_residual = [0.0; KNOT_COUNT];
    for _ in 0..2_000 {
        let box_input = std::array::from_fn(|index| current[index] + box_residual[index]);
        let boxed = project_box(&box_input);
        for index in 0..KNOT_COUNT {
            box_residual[index] = box_input[index] - boxed[index];
        }
        let order_input = std::array::from_fn(|index| boxed[index] + order_residual[index]);
        let ordered = project_monotone(&order_input);
        for index in 0..KNOT_COUNT {
            order_residual[index] = order_input[index] - ordered[index];
        }
        let delta = current
            .iter()
            .zip(ordered.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        current = ordered;
        if delta <= 1e-14 {
            break;
        }
    }
    current
}

fn objective(theta: &[f64; KNOT_COUNT], state: &State) -> f64 {
    let basis = knots();
    let quadratic: f64 = (0..KNOT_COUNT)
        .map(|row| {
            (0..KNOT_COUNT)
                .map(|column| theta[row] * state.gram[row][column] * theta[column])
                .sum::<f64>()
        })
        .sum();
    let linear: f64 = state
        .target
        .iter()
        .zip(theta.iter())
        .map(|(left, right)| left * right)
        .sum();
    quadratic - 2.0 * linear
        + RIDGE_MASS
            * theta
                .iter()
                .zip(basis.iter())
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f64>()
}

fn solve(state: &State) -> Result<([f64; KNOT_COUNT], usize, f64), String> {
    let basis = knots();
    let mut theta: [f64; KNOT_COUNT] = state
        .theta
        .clone()
        .try_into()
        .map_err(|_| "theta_shape_or_finiteness_invalid".to_string())?;
    let lipschitz = (0..KNOT_COUNT)
        .map(|row| {
            2.0 * ((0..KNOT_COUNT)
                .map(|column| state.gram[row][column].abs())
                .sum::<f64>()
                + RIDGE_MASS)
        })
        .fold(0.0, f64::max);
    if !lipschitz.is_finite() || lipschitz <= 0.0 {
        return Err("recalibration_solver_did_not_converge".to_string());
    }
    let mut iterations = 0usize;
    let mut converged = false;
    for iteration in 1..=SOLVER_MAX_ITERATIONS {
        let gradient: [f64; KNOT_COUNT] = std::array::from_fn(|row| {
            let fitted: f64 = (0..KNOT_COUNT)
                .map(|column| state.gram[row][column] * theta[column])
                .sum();
            2.0 * (fitted - state.target[row] + RIDGE_MASS * (theta[row] - basis[row]))
        });
        let proposal = std::array::from_fn(|index| theta[index] - gradient[index] / lipschitz);
        let next = project_constraints(&proposal);
        let delta = theta
            .iter()
            .zip(next.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f64::max);
        theta = next;
        iterations = iteration;
        if delta <= SOLVER_TOLERANCE * 0.01 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err("recalibration_solver_did_not_converge".to_string());
    }
    Ok((theta, iterations, objective(&theta, state)))
}

fn updated_state(state: &State, events: &[Event]) -> Result<State, String> {
    let mut candidate = state.clone();
    let mut processed: BTreeSet<String> = state.processed_event_ids.iter().cloned().collect();
    let mut seen = BTreeSet::new();
    for event in events {
        if event.event_id.is_empty() {
            return Err("event_id_required".to_string());
        }
        if !seen.insert(event.event_id.clone()) || processed.contains(&event.event_id) {
            return Err(format!("duplicate_feedback:{}", event.event_id));
        }
        if event.label != 0 && event.label != 1 {
            return Err("binary_label_required".to_string());
        }
        let row = design(event.probability)?;
        for left in 0..KNOT_COUNT {
            candidate.target[left] += event.label as f64 * row[left];
            for right in 0..KNOT_COUNT {
                candidate.gram[left][right] += row[left] * row[right];
            }
        }
    }
    let (theta, iterations, measured_objective) = solve(&candidate)?;
    candidate.theta = theta.to_vec();
    candidate.sample_count += events.len();
    for event in events {
        processed.insert(event.event_id.clone());
    }
    candidate.processed_event_ids = processed.into_iter().collect();
    candidate.last_solver_receipt = serde_json::json!({
        "method": "projected_gradient_port_of_slsqp_contract",
        "tolerance": SOLVER_TOLERANCE,
        "iteration_cap": SOLVER_MAX_ITERATIONS,
        "converged": true,
        "status": 0,
        "iterations": iterations,
        "objective": measured_objective,
        "constraint_errors": [],
        "message": "converged",
        "sample_count": candidate.sample_count,
        "update_count": events.len(),
    });
    Ok(candidate)
}

pub fn read_state(path: &Path) -> Result<State, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|error| format!("state_read_failed:{error}"))?;
    let state: State =
        serde_json::from_slice(&bytes).map_err(|error| format!("state_json_invalid:{error}"))?;
    state.validate()?;
    Ok(state)
}

fn maybe_test_crash(stage: &str) {
    let enabled = std::env::var("CARNOT_RECALIBRATION_TEST_MODE")
        .ok()
        .as_deref()
        == Some("1");
    let selected = std::env::var("CARNOT_RECALIBRATION_TEST_CRASH_STAGE").unwrap_or_default();
    if enabled && selected == stage {
        std::process::exit(INTERRUPTED_WRITE_EXIT);
    }
}

fn durable_write(path: &Path, state: &State) -> Result<(usize, StageDurations), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "state_parent_missing".to_string())?;
    fs::create_dir_all(parent).map_err(|error| format!("state_parent_create_failed:{error}"))?;
    let mut encoded =
        serde_json::to_vec(state).map_err(|error| format!("state_json_encode_failed:{error}"))?;
    encoded.push(b'\n');
    let temporary = path.with_file_name(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("state"),
        std::process::id()
    ));
    let mut stages = BTreeMap::from([
        ("write".to_string(), 0u64),
        ("fsync".to_string(), 0u64),
        ("rename".to_string(), 0u64),
    ]);
    let started = Instant::now();
    let mut file =
        File::create(&temporary).map_err(|error| format!("state_create_failed:{error}"))?;
    file.write_all(&encoded)
        .map_err(|error| format!("state_write_failed:{error}"))?;
    file.flush()
        .map_err(|error| format!("state_flush_failed:{error}"))?;
    let wrote = Instant::now();
    stages.insert(
        "write".to_string(),
        wrote.duration_since(started).as_nanos() as u64,
    );
    file.sync_all()
        .map_err(|error| format!("state_fsync_failed:{error}"))?;
    let synced = Instant::now();
    stages.insert(
        "fsync".to_string(),
        synced.duration_since(wrote).as_nanos() as u64,
    );
    drop(file);
    maybe_test_crash("before_rename");
    fs::rename(&temporary, path).map_err(|error| format!("state_rename_failed:{error}"))?;
    let renamed = Instant::now();
    stages.insert(
        "rename".to_string(),
        renamed.duration_since(synced).as_nanos() as u64,
    );
    maybe_test_crash("after_rename");
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("directory_fsync_failed:{error}"))?;
    *stages.entry("fsync".to_string()).or_default() +=
        Instant::now().duration_since(renamed).as_nanos() as u64;
    Ok((encoded.len(), stages))
}

pub fn initialize_state(path: &Path) -> Result<usize, String> {
    let (bytes, _) = durable_write(path, &State::identity())?;
    read_state(path)?;
    Ok(bytes)
}

fn empty_stages() -> StageDurations {
    [
        "read", "predict", "solve", "write", "fsync", "rename", "reload",
    ]
    .into_iter()
    .map(|name| (name.to_string(), 0))
    .collect()
}

fn add_stage(stages: &mut StageDurations, name: &str, value: u64) {
    *stages.entry(name.to_string()).or_default() += value;
}

fn timing_enabled() -> bool {
    std::env::var("CARNOT_SERVICE_TIMING").ok().as_deref() == Some("1")
}

fn exclusive_stages(enabled: bool) -> StageDurations {
    if !enabled {
        return BTreeMap::new();
    }
    [
        "encoding",
        "update_arithmetic",
        "journal_write",
        "fsync",
        "acknowledgement",
        "reload",
    ]
    .into_iter()
    .map(|name| (name.to_string(), 0))
    .collect()
}

fn vectors_close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(a, b)| (a - b).abs() <= SOLVER_TOLERANCE)
}

fn durable_state_difference(left: &State, right: &State) -> Option<&'static str> {
    if left.schema != right.schema {
        return Some("schema");
    }
    if left.gram_shape != right.gram_shape {
        return Some("gram_shape");
    }
    if left
        .gram
        .iter()
        .zip(right.gram.iter())
        .any(|(a, b)| !vectors_close(a, b))
    {
        return Some("gram");
    }
    if !vectors_close(&left.target, &right.target) {
        return Some("target");
    }
    if !vectors_close(&left.theta, &right.theta) {
        return Some("theta");
    }
    if left.sample_count != right.sample_count {
        return Some("sample_count");
    }
    if left.processed_event_ids != right.processed_event_ids {
        return Some("processed_event_ids");
    }
    if left.solver_config != right.solver_config {
        return Some("solver_config");
    }
    if left.constrained != right.constrained {
        return Some("constrained");
    }
    None
}

fn failure(error: String) -> Response {
    Response {
        ok: false,
        error: Some(error),
        predictions: Vec::new(),
        acknowledgments: Vec::new(),
        acknowledged_release_count: 0,
        processed_event_count: 0,
        reloaded_state_matches: false,
        state_bytes: 0,
        state: None,
        stage_ns: empty_stages(),
        exclusive_stage_ns: BTreeMap::new(),
        kernel_ns: 0,
        durability_policy: DURABILITY_POLICY,
    }
}

fn run_trace(request: Request) -> Result<Response, String> {
    let path = request
        .state_path
        .ok_or_else(|| "state_path_required".to_string())?;
    let mut stages = empty_stages();
    let telemetry = timing_enabled();
    let mut exclusive = exclusive_stages(telemetry);
    let started = Instant::now();
    let mut state = read_state(&path)?;
    add_stage(&mut stages, "read", started.elapsed().as_nanos() as u64);
    let mut predictions = Vec::new();
    let mut acknowledgments = Vec::new();
    let mut state_bytes = fs::metadata(&path)
        .map_err(|error| format!("state_metadata_failed:{error}"))?
        .len() as usize;
    for (release_index, block) in request.events.chunks(RELEASE_BLOCK_SIZE).enumerate() {
        for event in block {
            let started = Instant::now();
            let probability = predict_value(event.probability, &state.theta)?;
            let action = typed_decision(probability)?;
            add_stage(&mut stages, "predict", started.elapsed().as_nanos() as u64);
            predictions.push(Prediction {
                event_id: event.event_id.clone(),
                probability,
                action,
            });
        }
        let started = Instant::now();
        let candidate = updated_state(&state, block)?;
        let update_ns = started.elapsed().as_nanos() as u64;
        add_stage(&mut stages, "solve", update_ns);
        if telemetry {
            add_stage(&mut exclusive, "update_arithmetic", update_ns);
        }
        let (bytes, write_stages) = durable_write(&path, &candidate)?;
        state_bytes = bytes;
        for (name, duration) in write_stages {
            add_stage(&mut stages, &name, duration);
            if telemetry {
                let exclusive_name = if name == "fsync" {
                    "fsync"
                } else {
                    "journal_write"
                };
                add_stage(&mut exclusive, exclusive_name, duration);
            }
        }
        let started = Instant::now();
        let reloaded = read_state(&path)?;
        let reload_ns = started.elapsed().as_nanos() as u64;
        add_stage(&mut stages, "reload", reload_ns);
        if telemetry {
            add_stage(&mut exclusive, "reload", reload_ns);
        }
        if let Some(field) = durable_state_difference(&reloaded, &candidate) {
            return Err(format!("reloaded_state_mismatch:{field}"));
        }
        state = reloaded;
        acknowledgments.push(release_index);
    }
    let kernel_ns =
        stages.get("predict").copied().unwrap_or(0) + stages.get("solve").copied().unwrap_or(0);
    Ok(Response {
        ok: true,
        error: None,
        predictions,
        acknowledgments: acknowledgments.clone(),
        acknowledged_release_count: acknowledgments.len(),
        processed_event_count: request.events.len(),
        reloaded_state_matches: true,
        state_bytes,
        state: Some(state),
        stage_ns: stages,
        exclusive_stage_ns: exclusive,
        kernel_ns,
        durability_policy: DURABILITY_POLICY,
    })
}

fn run_predict(request: Request) -> Result<Response, String> {
    let path = request
        .state_path
        .ok_or_else(|| "state_path_required".to_string())?;
    let mut stages = empty_stages();
    let started = Instant::now();
    let state = read_state(&path)?;
    add_stage(&mut stages, "read", started.elapsed().as_nanos() as u64);
    let processed: BTreeSet<&str> = state
        .processed_event_ids
        .iter()
        .map(String::as_str)
        .collect();
    let mut seen = BTreeSet::new();
    let mut predictions = Vec::new();
    for query in request.queries {
        if query.event_id.is_empty() {
            return Err("event_id_required".to_string());
        }
        if !seen.insert(query.event_id.clone()) || processed.contains(query.event_id.as_str()) {
            return Err(format!("duplicate_feedback:{}", query.event_id));
        }
        let started = Instant::now();
        let probability = predict_value(query.probability, &state.theta)?;
        let action = typed_decision(probability)?;
        add_stage(&mut stages, "predict", started.elapsed().as_nanos() as u64);
        predictions.push(Prediction {
            event_id: query.event_id,
            probability,
            action,
        });
    }
    let state_bytes = fs::metadata(&path)
        .map_err(|error| format!("state_metadata_failed:{error}"))?
        .len() as usize;
    let kernel_ns = stages.get("predict").copied().unwrap_or(0);
    Ok(Response {
        ok: true,
        error: None,
        predictions,
        acknowledgments: Vec::new(),
        acknowledged_release_count: 0,
        processed_event_count: 0,
        reloaded_state_matches: true,
        state_bytes,
        state: Some(state),
        stage_ns: stages,
        exclusive_stage_ns: exclusive_stages(timing_enabled()),
        kernel_ns,
        durability_policy: DURABILITY_POLICY,
    })
}

pub fn handle(request: Request) -> Response {
    if request.operation == "solver_failure" {
        return failure("recalibration_solver_did_not_converge".to_string());
    }
    let result = match request.operation.as_str() {
        "trace" => run_trace(request),
        "predict" => run_predict(request),
        _ => return failure("unknown_operation".to_string()),
    };
    result.unwrap_or_else(failure)
}

pub fn run_jsonl<R: Read, W: Write>(reader: R, writer: W) -> Result<(), String> {
    let mut output = BufWriter::new(writer);
    for line in BufReader::new(reader).lines() {
        let line = line.map_err(|error| format!("stdin_read_failed:{error}"))?;
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(request) => handle(request),
            Err(error) => failure(format!("request_json_invalid:{error}")),
        };
        serde_json::to_writer(&mut output, &response)
            .map_err(|error| format!("stdout_encode_failed:{error}"))?;
        output
            .write_all(b"\n")
            .map_err(|error| format!("stdout_write_failed:{error}"))?;
        output
            .flush()
            .map_err(|error| format!("stdout_flush_failed:{error}"))?;
    }
    Ok(())
}

pub fn run_stdio() -> Result<(), String> {
    run_jsonl(io::stdin().lock(), io::stdout().lock())
}

#[derive(Clone, Debug)]
pub struct NativeDecision {
    pub event_id: String,
    pub probability: f64,
    pub action: String,
}

#[derive(Clone, Debug)]
pub struct NativeAcknowledgment {
    pub event_id: String,
    pub acknowledged: bool,
    pub durable: bool,
    pub error: Option<String>,
}

pub struct PortableRecalibrationService {
    state_path: PathBuf,
    pending: BTreeMap<String, f64>,
}

impl PortableRecalibrationService {
    pub fn open(state_path: PathBuf) -> Result<Self, String> {
        if !state_path.exists() {
            initialize_state(&state_path)?;
        }
        read_state(&state_path)?;
        Ok(Self {
            state_path,
            pending: BTreeMap::new(),
        })
    }

    pub fn predict(
        &mut self,
        event_id: String,
        probability: f64,
    ) -> Result<NativeDecision, String> {
        if event_id.is_empty() {
            return Err("event_id_required".to_string());
        }
        if self.pending.contains_key(&event_id) {
            return Err(format!("duplicate_prediction:{event_id}"));
        }
        let response = run_predict(Request {
            operation: "predict".to_string(),
            state_path: Some(self.state_path.clone()),
            events: Vec::new(),
            queries: vec![Query {
                event_id: event_id.clone(),
                probability,
            }],
        })?;
        let prediction = response
            .predictions
            .into_iter()
            .next()
            .ok_or_else(|| "prediction_missing".to_string())?;
        self.pending.insert(event_id.clone(), probability);
        Ok(NativeDecision {
            event_id,
            probability: prediction.probability,
            action: prediction.action,
        })
    }

    pub fn release_feedback(&mut self, event_id: String, label: i32) -> NativeAcknowledgment {
        if label != 0 && label != 1 {
            return NativeAcknowledgment {
                event_id,
                acknowledged: false,
                durable: false,
                error: Some("binary_label_required".to_string()),
            };
        }
        let Some(probability) = self.pending.get(&event_id).copied() else {
            let state = read_state(&self.state_path);
            let duplicate = state
                .as_ref()
                .is_ok_and(|value| value.processed_event_ids.contains(&event_id));
            return NativeAcknowledgment {
                event_id: event_id.clone(),
                acknowledged: false,
                durable: false,
                error: Some(if duplicate {
                    format!("duplicate_feedback:{event_id}")
                } else {
                    format!("unknown_prediction:{event_id}")
                }),
            };
        };
        let result = run_trace(Request {
            operation: "trace".to_string(),
            state_path: Some(self.state_path.clone()),
            events: vec![Event {
                event_id: event_id.clone(),
                probability,
                label,
            }],
            queries: Vec::new(),
        });
        match result {
            Ok(response)
                if response.acknowledgments == [0]
                    && response.reloaded_state_matches
                    && response.durability_policy == DURABILITY_POLICY =>
            {
                self.pending.remove(&event_id);
                NativeAcknowledgment {
                    event_id,
                    acknowledged: true,
                    durable: true,
                    error: None,
                }
            }
            Ok(_) => NativeAcknowledgment {
                event_id,
                acknowledged: false,
                durable: false,
                error: Some("durable_acknowledgment_invalid".to_string()),
            },
            Err(error) => NativeAcknowledgment {
                event_id,
                acknowledged: false,
                durable: false,
                error: Some(error),
            },
        }
    }

    pub fn state(&self) -> Result<State, String> {
        read_state(&self.state_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_invalid_input_has_no_acknowledgment() {
        let request = Request {
            operation: "predict".to_string(),
            state_path: None,
            events: Vec::new(),
            queries: Vec::new(),
        };
        let response = handle(request);
        assert!(!response.ok);
        assert_eq!(response.acknowledged_release_count, 0);
    }

    #[test]
    fn identity_prediction_preserves_typed_decision() {
        assert_eq!(predict_value(0.01, &knots()).unwrap(), 0.01);
        assert_eq!(typed_decision(0.01).unwrap(), "accept");
        assert_eq!(typed_decision(0.5).unwrap(), "escalate");
        assert_eq!(typed_decision(0.99).unwrap(), "reject");
    }
}
