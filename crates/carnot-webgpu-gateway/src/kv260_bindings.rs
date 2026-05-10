//! KV260 q=3 Potts sampler register binding for Exp 1704.
//!
//! The KV260 RTL is exposed to Linux through a generic-UIO AXI-Lite register
//! window.  This module keeps the hardware-facing code small and auditable:
//! callers provide a Potts problem, the sampler writes the exact
//! `potts_sampler_v1.v` register map, pulses START, polls STATUS.DONE, and
//! decodes the 2-bit q=3 state output words.  Tests use the same trait through
//! an in-memory recorder, so the Python/PyO3 import path does not need a live
//! board to be safe.
//!
//! Spec: REQ-POTTS-008-3, REQ-POTTS-008-5

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::{Duration, Instant};

use memmap2::{MmapMut, MmapOptions};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub const EXPERIMENT_ID: u32 = 1704;
pub const POTTS_Q_STATES: usize = 3;
pub const MAX_KV260_SPINS: usize = 64;
pub const DEFAULT_MAX_DEGREE: usize = 32;
pub const DEFAULT_UIO_PATH: &str = "/dev/uio4";
pub const DEFAULT_UIO_WINDOW_BYTES: usize = 128 * 1024;

pub const ADDR_CONTROL: u32 = 0x0000;
pub const ADDR_STATUS: u32 = 0x0004;
pub const ADDR_SPIN_COUNT: u32 = 0x0008;
pub const ADDR_BETA_FINAL: u32 = 0x001C;
pub const ADDR_BIAS_BASE: u32 = 0x1000;
pub const ADDR_ADJ_BASE: u32 = 0x2000;
pub const ADDR_COUPL_BASE: u32 = 0x6000;
pub const ADDR_SPOUT_BASE: u32 = 0xA010;

pub const CONTROL_START_MASK: u32 = 1 << 0;
pub const CONTROL_RESET_MASK: u32 = 1 << 1;
pub const STATUS_DONE_MASK: u32 = 1 << 2;

pub const REQUIRED_ARTIFACT_FIELDS: &[&str] = &[
    "schema",
    "experiment_id",
    "vivado_available",
    "synthesis_success",
    "performance",
    "resource_utilization",
    "kv260_potts_binding_ready",
    "pyo3_binding_ready",
    "rust_binding_path",
    "python_binding_name",
    "driver_interface",
    "register_map",
    "spec_traces",
    "tests_run",
    "honest_verdict",
];

#[derive(Debug, Error)]
pub enum Kv260BindingError {
    #[error("invalid Potts problem: {0}")]
    InvalidProblem(String),
    #[error("KV260 driver I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("KV260 driver register offset 0x{offset:05x} exceeds {window_size} byte window")]
    RegisterOutOfRange { offset: u32, window_size: usize },
    #[error("KV260 sampler timed out waiting for STATUS.DONE after {polls} polls")]
    Timeout { polls: u32 },
    #[error("artifact validation failed: {0}")]
    InvalidArtifact(String),
    #[error("artifact JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub trait Kv260RegisterIo {
    fn write32(&mut self, offset: u32, value: u32) -> Result<(), Kv260BindingError>;
    fn read32(&mut self, offset: u32) -> Result<u32, Kv260BindingError>;
}

pub struct UioKv260Driver {
    device_path: PathBuf,
    window_size: usize,
    map: MmapMut,
}

impl UioKv260Driver {
    pub fn open(
        device_path: impl AsRef<Path>,
        window_size: usize,
    ) -> Result<Self, Kv260BindingError> {
        let path = device_path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let map = unsafe { MmapOptions::new().len(window_size).map_mut(&file)? };
        Ok(Self {
            device_path: path,
            window_size,
            map,
        })
    }

    pub fn device_path(&self) -> &Path {
        &self.device_path
    }

    fn check_offset(&self, offset: u32) -> Result<usize, Kv260BindingError> {
        let index = offset as usize;
        if index + std::mem::size_of::<u32>() > self.window_size {
            return Err(Kv260BindingError::RegisterOutOfRange {
                offset,
                window_size: self.window_size,
            });
        }
        Ok(index)
    }
}

impl Kv260RegisterIo for UioKv260Driver {
    fn write32(&mut self, offset: u32, value: u32) -> Result<(), Kv260BindingError> {
        let index = self.check_offset(offset)?;
        let ptr = unsafe { self.map.as_mut_ptr().add(index) as *mut u32 };
        unsafe {
            ptr::write_volatile(ptr, value.to_le());
        }
        Ok(())
    }

    fn read32(&mut self, offset: u32) -> Result<u32, Kv260BindingError> {
        let index = self.check_offset(offset)?;
        let ptr = unsafe { self.map.as_ptr().add(index) as *const u32 };
        let raw = unsafe { ptr::read_volatile(ptr) };
        Ok(u32::from_le(raw))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Kv260PottsProblem {
    pub n_spins: usize,
    pub max_degree: usize,
    pub beta_fixed: u8,
    pub biases: Vec<i8>,
    pub adjacency: Vec<i16>,
    pub couplings: Vec<i8>,
}

impl Kv260PottsProblem {
    pub fn new(
        n_spins: usize,
        max_degree: usize,
        beta_fixed: u8,
        biases: Vec<i8>,
        adjacency: Vec<i16>,
        couplings: Vec<i8>,
    ) -> Result<Self, Kv260BindingError> {
        let problem = Self {
            n_spins,
            max_degree,
            beta_fixed,
            biases,
            adjacency,
            couplings,
        };
        problem.validate()?;
        Ok(problem)
    }

    pub fn validate(&self) -> Result<(), Kv260BindingError> {
        if self.n_spins == 0 || self.n_spins > MAX_KV260_SPINS {
            return Err(Kv260BindingError::InvalidProblem(format!(
                "n_spins must be in 1..={MAX_KV260_SPINS}, got {}",
                self.n_spins
            )));
        }
        if self.max_degree == 0 || self.max_degree > DEFAULT_MAX_DEGREE {
            return Err(Kv260BindingError::InvalidProblem(format!(
                "max_degree must be in 1..={DEFAULT_MAX_DEGREE}, got {}",
                self.max_degree
            )));
        }
        let expected_biases = self.n_spins * POTTS_Q_STATES;
        if self.biases.len() != expected_biases {
            return Err(Kv260BindingError::InvalidProblem(format!(
                "bias table must contain {expected_biases} entries, got {}",
                self.biases.len()
            )));
        }
        let expected_edges = self.n_spins * self.max_degree;
        if self.adjacency.len() != expected_edges {
            return Err(Kv260BindingError::InvalidProblem(format!(
                "adjacency table must contain {expected_edges} entries, got {}",
                self.adjacency.len()
            )));
        }
        if self.couplings.len() != expected_edges {
            return Err(Kv260BindingError::InvalidProblem(format!(
                "coupling table must contain {expected_edges} entries, got {}",
                self.couplings.len()
            )));
        }
        for (index, &neighbor) in self.adjacency.iter().enumerate() {
            if neighbor != -1 && (neighbor < 0 || neighbor as usize >= self.n_spins) {
                return Err(Kv260BindingError::InvalidProblem(format!(
                    "adjacency[{index}]={neighbor} is outside -1 or 0..{}",
                    self.n_spins - 1
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Kv260PottsSample {
    pub states: Vec<u8>,
    pub status: u32,
    pub polls: u32,
    pub backend: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Kv260PottsSampler {
    pub device_path: PathBuf,
    pub window_size: usize,
    pub poll_timeout: Duration,
}

impl Default for Kv260PottsSampler {
    fn default() -> Self {
        Self {
            device_path: PathBuf::from(DEFAULT_UIO_PATH),
            window_size: DEFAULT_UIO_WINDOW_BYTES,
            poll_timeout: Duration::from_millis(50),
        }
    }
}

impl Kv260PottsSampler {
    pub fn new(device_path: impl Into<PathBuf>, poll_timeout: Duration) -> Self {
        Self {
            device_path: device_path.into(),
            window_size: DEFAULT_UIO_WINDOW_BYTES,
            poll_timeout,
        }
    }

    pub fn sample(
        &self,
        problem: &Kv260PottsProblem,
    ) -> Result<Kv260PottsSample, Kv260BindingError> {
        let mut driver = UioKv260Driver::open(&self.device_path, self.window_size)?;
        self.sample_with_io(&mut driver, problem)
    }

    pub fn sample_with_io<I: Kv260RegisterIo>(
        &self,
        io: &mut I,
        problem: &Kv260PottsProblem,
    ) -> Result<Kv260PottsSample, Kv260BindingError> {
        problem.validate()?;
        self.upload_problem(io, problem)?;
        io.write32(ADDR_CONTROL, CONTROL_START_MASK)?;

        let started = Instant::now();
        let mut polls = 0;
        let status = loop {
            polls += 1;
            let status = io.read32(ADDR_STATUS)?;
            if status & STATUS_DONE_MASK != 0 {
                break status;
            }
            if started.elapsed() >= self.poll_timeout {
                return Err(Kv260BindingError::Timeout { polls });
            }
            std::thread::yield_now();
        };

        let words = read_spin_words(io, problem.n_spins)?;
        io.write32(ADDR_CONTROL, 0)?;
        Ok(Kv260PottsSample {
            states: unpack_potts_words(&words, problem.n_spins),
            status,
            polls,
            backend: "kv260-uio".to_string(),
        })
    }

    fn upload_problem<I: Kv260RegisterIo>(
        &self,
        io: &mut I,
        problem: &Kv260PottsProblem,
    ) -> Result<(), Kv260BindingError> {
        io.write32(ADDR_CONTROL, CONTROL_RESET_MASK)?;
        io.write32(ADDR_CONTROL, 0)?;
        io.write32(ADDR_SPIN_COUNT, problem.n_spins as u32)?;
        io.write32(ADDR_BETA_FINAL, problem.beta_fixed as u32)?;

        for (index, &bias) in problem.biases.iter().enumerate() {
            io.write32(ADDR_BIAS_BASE + 4 * index as u32, bias as u8 as u32)?;
        }
        for (index, &neighbor) in problem.adjacency.iter().enumerate() {
            io.write32(ADDR_ADJ_BASE + 4 * index as u32, neighbor as u16 as u32)?;
        }
        for (index, &coupling) in problem.couplings.iter().enumerate() {
            io.write32(ADDR_COUPL_BASE + 4 * index as u32, coupling as i32 as u32)?;
        }
        Ok(())
    }
}

pub fn read_spin_words<I: Kv260RegisterIo>(
    io: &mut I,
    n_spins: usize,
) -> Result<Vec<u32>, Kv260BindingError> {
    let n_words = n_spins.div_ceil(16);
    let mut words = Vec::with_capacity(n_words);
    for word_index in 0..n_words {
        words.push(io.read32(ADDR_SPOUT_BASE + 4 * word_index as u32)?);
    }
    Ok(words)
}

pub fn unpack_potts_words(words: &[u32], n_spins: usize) -> Vec<u8> {
    let mut states = Vec::with_capacity(n_spins);
    for spin_index in 0..n_spins {
        let word = words[spin_index / 16];
        let shift = ((spin_index % 16) * 2) as u32;
        states.push(((word >> shift) & 0b11) as u8);
    }
    states
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Kv260BindingArtifact {
    pub schema: String,
    pub experiment_id: u32,
    pub vivado_available: bool,
    pub synthesis_success: bool,
    pub performance: Option<Value>,
    pub resource_utilization: Option<Value>,
    pub kv260_potts_binding_ready: bool,
    pub pyo3_binding_ready: bool,
    pub rust_binding_path: String,
    pub python_binding_name: String,
    pub driver_interface: String,
    pub register_map: BTreeMap<String, String>,
    pub spec_traces: Vec<String>,
    pub tests_run: Vec<String>,
    pub honest_verdict: String,
}

pub fn validate_artifact_json(input: &str) -> Result<Kv260BindingArtifact, Kv260BindingError> {
    let value: Value = serde_json::from_str(input)?;
    let object = value.as_object().ok_or_else(|| {
        Kv260BindingError::InvalidArtifact("top-level JSON must be an object".into())
    })?;
    for field in REQUIRED_ARTIFACT_FIELDS {
        if !object.contains_key(*field) {
            return Err(Kv260BindingError::InvalidArtifact(format!(
                "missing required field {field}"
            )));
        }
    }

    let artifact: Kv260BindingArtifact = serde_json::from_value(value)?;
    if artifact.schema != "kv260_potts_pyo3_binding_v1" {
        return Err(Kv260BindingError::InvalidArtifact(format!(
            "unexpected schema {}",
            artifact.schema
        )));
    }
    if artifact.experiment_id != EXPERIMENT_ID {
        return Err(Kv260BindingError::InvalidArtifact(format!(
            "experiment_id must be {EXPERIMENT_ID}"
        )));
    }
    if !artifact.kv260_potts_binding_ready || !artifact.pyo3_binding_ready {
        return Err(Kv260BindingError::InvalidArtifact(
            "Rust and PyO3 binding readiness must both be true".into(),
        ));
    }
    for trace in ["REQ-POTTS-008-3", "REQ-POTTS-008-4", "REQ-POTTS-008-5"] {
        if !artifact.spec_traces.iter().any(|entry| entry == trace) {
            return Err(Kv260BindingError::InvalidArtifact(format!(
                "missing spec trace {trace}"
            )));
        }
    }
    Ok(artifact)
}
