use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use carnot_webgpu_gateway::kv260_bindings::{
    validate_artifact_json, Kv260BindingError, Kv260PottsProblem, Kv260PottsSampler,
    Kv260RegisterIo, ADDR_ADJ_BASE, ADDR_BETA_FINAL, ADDR_BIAS_BASE, ADDR_CONTROL, ADDR_COUPL_BASE,
    ADDR_SPIN_COUNT, REQUIRED_ARTIFACT_FIELDS, STATUS_DONE_MASK,
};

#[derive(Default)]
struct RecordingIo {
    writes: Vec<(u32, u32)>,
    status_reads_before_done: usize,
    spin_words: BTreeMap<u32, u32>,
}

impl RecordingIo {
    fn with_spin_word(mut self, word_index: u32, word: u32) -> Self {
        self.spin_words.insert(word_index, word);
        self
    }
}

impl Kv260RegisterIo for RecordingIo {
    fn write32(&mut self, offset: u32, value: u32) -> Result<(), Kv260BindingError> {
        self.writes.push((offset, value));
        Ok(())
    }

    fn read32(&mut self, offset: u32) -> Result<u32, Kv260BindingError> {
        if offset == carnot_webgpu_gateway::kv260_bindings::ADDR_STATUS {
            if self.status_reads_before_done > 0 {
                self.status_reads_before_done -= 1;
                return Ok(0);
            }
            return Ok(STATUS_DONE_MASK);
        }
        if offset >= carnot_webgpu_gateway::kv260_bindings::ADDR_SPOUT_BASE {
            let word_index = (offset - carnot_webgpu_gateway::kv260_bindings::ADDR_SPOUT_BASE) / 4;
            return Ok(*self.spin_words.get(&word_index).unwrap_or(&0));
        }
        Ok(0)
    }
}

#[test]
fn test_kv260_binding_uploads_axi_register_sequence() {
    // REQ-POTTS-008-3: the Rust binding writes the Potts AXI-Lite register map,
    // polls STATUS.DONE, and unpacks q=3 2-bit state words.
    let problem = Kv260PottsProblem::new(
        3,
        2,
        0x40,
        vec![1, 0, -1, 0, 2, -2, 3, 0, -3],
        vec![1, -1, 2, -1, 0, -1],
        vec![7, 0, -5, 0, 4, 0],
    )
    .expect("valid small Potts problem");
    let mut io = RecordingIo::default().with_spin_word(0, 0b00_01_10);
    let sampler = Kv260PottsSampler::default();

    let sample = sampler
        .sample_with_io(&mut io, &problem)
        .expect("recorded register sample");

    assert_eq!(sample.states, vec![2, 1, 0]);
    assert_eq!(sample.backend, "kv260-uio");
    assert!(io.writes.contains(&(ADDR_CONTROL, 0x2)));
    assert!(io.writes.contains(&(ADDR_CONTROL, 0x0)));
    assert!(io.writes.contains(&(ADDR_SPIN_COUNT, 3)));
    assert!(io.writes.contains(&(ADDR_BETA_FINAL, 0x40)));
    assert!(io.writes.contains(&(ADDR_BIAS_BASE, 1)));
    assert!(io.writes.contains(&(ADDR_BIAS_BASE + 8, 0xff)));
    assert!(io.writes.contains(&(ADDR_ADJ_BASE, 1)));
    assert!(io.writes.contains(&(ADDR_ADJ_BASE + 4, 0xffff)));
    assert!(io.writes.contains(&(ADDR_COUPL_BASE, 7)));
    assert!(io.writes.contains(&(ADDR_COUPL_BASE + 8, 0xfffffffb)));
    assert!(io.writes.contains(&(ADDR_CONTROL, 0x1)));
}

#[test]
fn test_kv260_binding_rejects_invalid_problem_shapes() {
    // REQ-POTTS-008-3: malformed host-side Potts problems fail before any
    // hardware register write can occur.
    let err = Kv260PottsProblem::new(65, 1, 0x40, vec![0; 65 * 3], vec![0; 65], vec![0; 65])
        .expect_err("KV260 Potts RTL is fixed to 64 spins");
    assert!(err.to_string().contains("n_spins"));

    let err = Kv260PottsProblem::new(2, 2, 0x40, vec![0; 5], vec![0; 4], vec![0; 4])
        .expect_err("bias table must have n_spins * 3 entries");
    assert!(err.to_string().contains("bias"));
}

#[test]
fn test_exp1704_artifact_schema_fields_validate() {
    // REQ-POTTS-008-5: the Exp 1704 artifact records the Rust/PyO3 binding
    // paths, register map, spec traces, and tests that back the deliverable.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let artifact_path = repo_root.join("results/experiment_1704_kv260.json");
    let artifact = fs::read_to_string(artifact_path).expect("Exp 1704 artifact exists");
    let validated = validate_artifact_json(&artifact).expect("artifact schema validates");

    assert_eq!(validated.schema, "kv260_potts_pyo3_binding_v1");
    assert!(validated.kv260_potts_binding_ready);
    assert!(validated.pyo3_binding_ready);
    assert_eq!(
        validated.rust_binding_path,
        "crates/carnot-webgpu-gateway/src/kv260_bindings.rs"
    );
    assert_eq!(validated.python_binding_name, "RustKv260PottsSampler");
    assert!(REQUIRED_ARTIFACT_FIELDS.contains(&"register_map"));
    assert!(validated
        .spec_traces
        .contains(&"REQ-POTTS-008-3".to_string()));
}
