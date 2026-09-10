//! JSON process bridge for the fixed-cardinality sampler experiment.
//!
//! This executable keeps deployment separate from Python bindings. It reads
//! one request from standard input and writes one response to standard output.
//!
//! Spec: REQ-SAMPLER-7189-E2E, REQ-SAMPLER-7189-THROUGHPUT

use carnot_samplers::fixed_cardinality::{
    PairSwapChainOutcome, PairSwapConfig, PairSwapCore, PairSwapDraw, PairSwapReplayOutcome,
    PairSwapSeededState,
};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::time::{Duration, Instant};

#[derive(Debug, Deserialize)]
struct BridgeRequest {
    operation: String,
    config: PairSwapConfig,
    initial_state: Vec<i8>,
    #[serde(default)]
    tape: Vec<PairSwapDraw>,
    seed: Option<u64>,
    burn_in: Option<usize>,
    retained: Option<usize>,
    max_duration_s: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum BridgeResponse {
    Replay(PairSwapReplayOutcome),
    Chain(PairSwapChainOutcome),
}

fn run() -> Result<(), String> {
    let mut bytes = Vec::new();
    std::io::stdin()
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read request: {error}"))?;
    let request: BridgeRequest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid request JSON: {error}"))?;
    let config = PairSwapConfig::new(
        request.config.edges.clone(),
        request.config.fields.clone(),
        request.config.cardinality,
        request.config.beta,
    )?;
    let core = PairSwapCore::new(config);
    let response = match request.operation.as_str() {
        "replay" => BridgeResponse::Replay(core.run_replay(&request.initial_state, &request.tape)?),
        "seeded" => BridgeResponse::Chain(core.run_seeded(
            &request.initial_state,
            request.seed.ok_or_else(|| "seed is required".to_string())?,
            request.burn_in.unwrap_or(0),
            request
                .retained
                .ok_or_else(|| "retained is required".to_string())?,
        )?),
        "timed" => BridgeResponse::Chain(run_timed(&core, &request)?),
        other => return Err(format!("unknown operation: {other}")),
    };
    let encoded = serde_json::to_vec(&response)
        .map_err(|error| format!("failed to serialize response: {error}"))?;
    std::io::stdout()
        .write_all(&encoded)
        .map_err(|error| format!("failed to write response: {error}"))?;
    Ok(())
}

fn run_timed(core: &PairSwapCore, request: &BridgeRequest) -> Result<PairSwapChainOutcome, String> {
    let seconds = request
        .max_duration_s
        .ok_or_else(|| "max_duration_s is required".to_string())?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("max_duration_s must be finite and positive".to_string());
    }
    let mut state = PairSwapSeededState::new(
        request.initial_state.clone(),
        request.seed.ok_or_else(|| "seed is required".to_string())?,
    )?;
    let started = Instant::now();
    let duration = Duration::from_secs_f64(seconds);
    let mut samples = Vec::new();
    let mut energies = Vec::new();
    let mut accepted = 0;
    loop {
        let outcome = core.step_seeded(&mut state)?;
        accepted += usize::from(outcome.accepted);
        energies.push(core.energy(&state.spins)?);
        samples.push(state.spins.clone());
        if started.elapsed() >= duration {
            break;
        }
    }
    let attempted = state.transition;
    Ok(PairSwapChainOutcome {
        samples,
        energies,
        accepted,
        attempted,
        final_state: state,
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("fixed-cardinality-bridge: {error}");
        std::process::exit(2);
    }
}
