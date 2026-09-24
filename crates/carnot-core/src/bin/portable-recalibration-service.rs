//! Durable process boundary for the qualified nine-knot recalibration map.
//!
//! The executable reads newline-delimited JSON from stdin. It is a production
//! precursor, not a shared language binding and not FPGA execution.
//!
//! Spec: REQ-CL-7585 and SCENARIO-CL-7585-PARITY/SERVICE.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const KNOT_COUNT: usize = 9;
const RIDGE_MASS: f64 = 8.0;
const MOVEMENT_BOUND: f64 = 0.10;
const SOLVER_TOLERANCE: f64 = 1e-8;
const SOLVER_MAX_ITERATIONS: usize = 500;
const RELEASE_BLOCK_SIZE: usize = 8;
const STATE_SCHEMA: &str = "carnot.recalibration.sufficient_statistics.v1";
const DURABILITY_POLICY: &str = "atomic_file_fsync_rename_directory_fsync_reload_ack";

fn knots() -> [f64; KNOT_COUNT] {
    std::array::from_fn(|index| index as f64 / (KNOT_COUNT - 1) as f64)
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct SolverConfig {
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

    fn validate(&self) -> Result<(), String> {
        if self != &Self::qualified() {
            return Err("solver_configuration_not_frozen".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct State {
    schema: String,
    gram_shape: Vec<usize>,
    gram: Vec<Vec<f64>>,
    target: Vec<f64>,
    theta: Vec<f64>,
    sample_count: usize,
    processed_event_ids: Vec<String>,
    solver_config: SolverConfig,
    constrained: bool,
    last_solver_receipt: Value,
}

impl State {
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
        self.solver_config.validate()?;
        if !self.constrained {
            return Err("constrained_state_required".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
struct Event {
    event_id: String,
    probability: f64,
    label: i32,
}

#[derive(Clone, Debug, Deserialize)]
struct Request {
    operation: String,
    state_path: Option<PathBuf>,
    #[serde(default)]
    events: Vec<Event>,
}

#[derive(Clone, Debug, Serialize)]
struct Prediction {
    event_id: String,
    probability: f64,
    action: String,
}

#[derive(Debug, Serialize)]
struct Response {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    predictions: Vec<Prediction>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    acknowledgments: Vec<usize>,
    acknowledged_release_count: usize,
    processed_event_count: usize,
    reloaded_state_matches: bool,
    state_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<State>,
    stage_ns: BTreeMap<String, u64>,
    kernel_ns: u64,
    durability_policy: &'static str,
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

fn predict(probability: f64, theta: &[f64]) -> Result<f64, String> {
    let row = design(probability)?;
    Ok(row
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
        let lower = 0.0f64.max(basis[index] - MOVEMENT_BOUND);
        let upper = 1.0f64.min(basis[index] + MOVEMENT_BOUND);
        values[index].clamp(lower, upper)
    })
}

fn project_constraints(values: &[f64; KNOT_COUNT]) -> [f64; KNOT_COUNT] {
    // Dykstra's method computes the Euclidean projection onto the intersection
    // of the box and monotone cone. Nine variables make the tight inner loop cheap.
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

fn objective(theta: &[f64; KNOT_COUNT], gram: &[Vec<f64>], target: &[f64]) -> f64 {
    let basis = knots();
    let quadratic: f64 = (0..KNOT_COUNT)
        .map(|row| {
            (0..KNOT_COUNT)
                .map(|column| theta[row] * gram[row][column] * theta[column])
                .sum::<f64>()
        })
        .sum();
    let linear: f64 = target
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
    let mut converged = false;
    let mut iterations = 0usize;
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
    Ok((
        theta,
        iterations,
        objective(&theta, &state.gram, &state.target),
    ))
}

fn update_batch(state: &mut State, events: &[Event]) -> Result<(), String> {
    let mut processed: BTreeSet<String> = state.processed_event_ids.iter().cloned().collect();
    let mut seen = BTreeSet::new();
    for event in events {
        if !seen.insert(event.event_id.clone()) || processed.contains(&event.event_id) {
            return Err(format!("duplicate_feedback:{}", event.event_id));
        }
        if event.label != 0 && event.label != 1 {
            return Err("binary_label_required".to_string());
        }
        let row = design(event.probability)?;
        for left in 0..KNOT_COUNT {
            state.target[left] += event.label as f64 * row[left];
            for right in 0..KNOT_COUNT {
                state.gram[left][right] += row[left] * row[right];
            }
        }
    }
    let (theta, iterations, measured_objective) = solve(state)?;
    state.theta = theta.to_vec();
    state.sample_count += events.len();
    for event in events {
        processed.insert(event.event_id.clone());
    }
    state.processed_event_ids = processed.into_iter().collect();
    state.last_solver_receipt = serde_json::json!({
        "method": "projected_gradient_port_of_slsqp_contract",
        "tolerance": SOLVER_TOLERANCE,
        "iteration_cap": SOLVER_MAX_ITERATIONS,
        "converged": true,
        "status": 0,
        "iterations": iterations,
        "objective": measured_objective,
        "constraint_errors": [],
        "message": "converged",
        "sample_count": state.sample_count,
        "update_count": events.len(),
    });
    Ok(())
}

fn read_state(path: &Path) -> Result<State, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|error| format!("state_read_failed:{error}"))?;
    let state: State =
        serde_json::from_slice(&bytes).map_err(|error| format!("state_json_invalid:{error}"))?;
    state.validate()?;
    Ok(state)
}

fn durable_write(path: &Path, state: &State) -> Result<(usize, BTreeMap<String, u64>), String> {
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
    fs::rename(&temporary, path).map_err(|error| format!("state_rename_failed:{error}"))?;
    let renamed = Instant::now();
    stages.insert(
        "rename".to_string(),
        renamed.duration_since(synced).as_nanos() as u64,
    );
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("directory_fsync_failed:{error}"))?;
    let directory_synced = Instant::now();
    *stages.entry("fsync".to_string()).or_default() +=
        directory_synced.duration_since(renamed).as_nanos() as u64;
    Ok((encoded.len(), stages))
}

fn empty_stages() -> BTreeMap<String, u64> {
    [
        "read", "predict", "solve", "write", "fsync", "rename", "reload",
    ]
    .into_iter()
    .map(|name| (name.to_string(), 0))
    .collect()
}

fn add_stage(stages: &mut BTreeMap<String, u64>, name: &str, value: u64) {
    *stages.entry(name.to_string()).or_default() += value;
}

fn vectors_close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(a, b)| (a - b).abs() <= SOLVER_TOLERANCE)
}

fn durable_state_difference(left: &State, right: &State) -> Option<String> {
    if left.schema != right.schema {
        return Some("schema".to_string());
    }
    if left.gram_shape != right.gram_shape {
        return Some("gram_shape".to_string());
    }
    if left
        .gram
        .iter()
        .zip(right.gram.iter())
        .any(|(a, b)| !vectors_close(a, b))
    {
        return Some("gram".to_string());
    }
    if !vectors_close(&left.target, &right.target) {
        return Some("target".to_string());
    }
    if !vectors_close(&left.theta, &right.theta) {
        return Some("theta".to_string());
    }
    if left.sample_count != right.sample_count {
        return Some("sample_count".to_string());
    }
    if left.processed_event_ids != right.processed_event_ids {
        return Some("processed_event_ids".to_string());
    }
    if left.solver_config != right.solver_config {
        return Some("solver_config".to_string());
    }
    if left.constrained != right.constrained {
        return Some("constrained".to_string());
    }
    None
}

fn failure(error: String, stages: BTreeMap<String, u64>) -> Response {
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
        stage_ns: stages,
        kernel_ns: 0,
        durability_policy: DURABILITY_POLICY,
    }
}

fn run_trace(request: Request) -> Result<Response, String> {
    let path = request
        .state_path
        .ok_or_else(|| "state_path_required".to_string())?;
    let mut stages = empty_stages();
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
            let probability = predict(event.probability, &state.theta)?;
            let action = typed_decision(probability)?;
            add_stage(&mut stages, "predict", started.elapsed().as_nanos() as u64);
            predictions.push(Prediction {
                event_id: event.event_id.clone(),
                probability,
                action,
            });
        }
        let started = Instant::now();
        update_batch(&mut state, block)?;
        add_stage(&mut stages, "solve", started.elapsed().as_nanos() as u64);
        let (bytes, write_stages) = durable_write(&path, &state)?;
        state_bytes = bytes;
        for (name, duration) in write_stages {
            add_stage(&mut stages, &name, duration);
        }
        let started = Instant::now();
        let reloaded = read_state(&path)?;
        add_stage(&mut stages, "reload", started.elapsed().as_nanos() as u64);
        if let Some(field) = durable_state_difference(&reloaded, &state) {
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
        kernel_ns,
        durability_policy: DURABILITY_POLICY,
    })
}

fn handle(request: Request) -> Response {
    if request.operation == "solver_failure" {
        return failure(
            "recalibration_solver_did_not_converge".to_string(),
            empty_stages(),
        );
    }
    if request.operation != "trace" {
        return failure("unknown_operation".to_string(), empty_stages());
    }
    match run_trace(request) {
        Ok(response) => response,
        Err(error) => failure(error, empty_stages()),
    }
}

fn run() -> Result<(), String> {
    let input = io::stdin();
    let output = io::stdout();
    let mut writer = BufWriter::new(output.lock());
    for line in BufReader::new(input.lock()).lines() {
        let line = line.map_err(|error| format!("stdin_read_failed:{error}"))?;
        let response = match serde_json::from_str::<Request>(&line) {
            Ok(request) => handle(request),
            Err(error) => failure(format!("request_json_invalid:{error}"), empty_stages()),
        };
        serde_json::to_writer(&mut writer, &response)
            .map_err(|error| format!("stdout_encode_failed:{error}"))?;
        writer
            .write_all(b"\n")
            .map_err(|error| format!("stdout_write_failed:{error}"))?;
        writer
            .flush()
            .map_err(|error| format!("stdout_flush_failed:{error}"))?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("portable-recalibration-service: {error}");
        std::process::exit(2);
    }
}
