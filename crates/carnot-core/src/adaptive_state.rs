//! Bounded adaptive-state microkernel for Exp5859.
//!
//! Spec refs: REQ-LEARN-5859, SCENARIO-LEARN-5859-PRECONDITIONS,
//! SCENARIO-LEARN-5859-OPERATION-PARITY,
//! SCENARIO-LEARN-5859-STATE-HASH-ROUNDTRIP,
//! SCENARIO-LEARN-5859-FAIL-CLOSED.
//!
//! The kernel stores only externally validated adaptive state. It does not
//! train, infer, call model weights, or inspect labels; it just applies a small
//! versioned state ABI that Python can mirror exactly.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const ABI_VERSION: u32 = 1;
pub const STATE_SCHEMA: &str = "carnot.adaptive_state_microkernel.v1.state";
pub const CHECKPOINT_SCHEMA: &str = "carnot.adaptive_state_microkernel.v1.checkpoint";
pub const MAX_CAPACITY: u32 = 64;
pub const MAX_HISTORY_CAPACITY: u32 = 128;
pub const MAX_REPLAY_LIMIT: u32 = 64;
pub const MAX_EVENT_ID_LEN: usize = 64;
pub const MAX_REASON_LEN: usize = 32;

const QUALIFIED_CHANGES: [&str; 3] = ["addition", "supersession", "recurrence"];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveEvent {
    pub change: String,
    pub chronology_index: u32,
    pub confidence_q16: u16,
    pub event_id: String,
    pub payload_hash: String,
    pub signature_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotedEvent {
    pub change: String,
    pub chronology_index: u32,
    pub confidence_q16: u16,
    pub event_id: String,
    pub payload_hash: String,
    pub promoted_version: u32,
    pub signature_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuarantineEntry {
    pub event_id: String,
    pub reason_code: String,
    pub version_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvictedEntry {
    pub event_id: String,
    pub version_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveState {
    pub abi_version: u32,
    pub applied: Vec<AdaptiveEvent>,
    pub capacity: u32,
    pub core: Vec<AdaptiveEvent>,
    pub evicted: Vec<EvictedEntry>,
    pub history_capacity: u32,
    pub last_chronology: i64,
    pub promoted: Vec<PromotedEvent>,
    pub quarantine: Vec<QuarantineEntry>,
    pub schema: String,
    pub version_id: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdaptiveCheckpoint {
    pub abi_version: u32,
    pub active: AdaptiveState,
    pub history: Vec<AdaptiveState>,
    pub schema: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationResult {
    pub accepted: bool,
    pub code: String,
    pub state_hash: String,
    pub version_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_replay: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct AdaptiveStateKernel {
    state: AdaptiveState,
    history: BTreeMap<u32, AdaptiveState>,
}

impl AdaptiveStateKernel {
    pub fn new(capacity: u32, history_capacity: u32) -> Result<Self, String> {
        if !(1..=MAX_CAPACITY).contains(&capacity) {
            return Err("capacity must be an integer in [1, 64]".to_string());
        }
        if !(2..=MAX_HISTORY_CAPACITY).contains(&history_capacity) {
            return Err("history_capacity must be an integer in [2, 128]".to_string());
        }
        let state = base_state(capacity, history_capacity);
        let mut history = BTreeMap::new();
        history.insert(0, state.clone());
        Ok(Self { state, history })
    }

    pub fn restore(checkpoint: &[u8]) -> Result<Self, String> {
        let payload: AdaptiveCheckpoint = serde_json::from_slice(checkpoint)
            .map_err(|_| "checkpoint is not valid adaptive-state JSON".to_string())?;
        if payload.schema != CHECKPOINT_SCHEMA {
            return Err("checkpoint schema mismatch".to_string());
        }
        if payload.abi_version != ABI_VERSION {
            return Err("checkpoint ABI version mismatch".to_string());
        }
        if payload.history.is_empty() {
            return Err("checkpoint payload is incomplete".to_string());
        }
        let mut kernel = Self::new(payload.active.capacity, payload.active.history_capacity)?;
        kernel.state = payload.active.clone();
        kernel.history = payload
            .history
            .into_iter()
            .map(|state| (state.version_id, state))
            .collect();
        let active_version = kernel.state.version_id;
        let Some(active_history) = kernel.history.get(&active_version) else {
            return Err("checkpoint active version missing from history".to_string());
        };
        if active_history != &kernel.state {
            return Err("checkpoint active state differs from history".to_string());
        }
        Ok(kernel)
    }

    pub fn apply_event(&mut self, event: AdaptiveEvent) -> OperationResult {
        let before = self.canonical_state_hash();
        if let Err(code) = validate_event(&event) {
            return self.result(false, code, before, None);
        }
        if self.find_applied(&event.event_id).is_some() {
            return self.result(false, "DUPLICATE_EVENT", before, None);
        }
        if i64::from(event.chronology_index) <= self.state.last_chronology {
            return self.result(false, "OUT_OF_ORDER_EVENT", before, None);
        }
        self.state.last_chronology = i64::from(event.chronology_index);
        self.state.applied.push(event);
        self.state
            .applied
            .sort_by(|left, right| left.event_id.cmp(&right.event_id));
        self.bump_version();
        self.result(true, "OK", self.canonical_state_hash(), None)
    }

    pub fn acquire_core(&mut self, event_id: &str) -> OperationResult {
        let before = self.canonical_state_hash();
        if !valid_short_token(event_id, MAX_EVENT_ID_LEN) {
            return self.result(false, "INVALID_EVENT_ID", before, None);
        }
        let Some(event) = self.find_applied(event_id).cloned() else {
            return self.result(false, "UNKNOWN_EVENT", before, None);
        };
        if event.change != "addition" {
            return self.result(false, "UNQUALIFIED_OPERATION", before, None);
        }
        if self.find_core(event_id).is_some() {
            return self.result(false, "DUPLICATE_CORE", before, None);
        }
        self.state.core.push(event);
        self.state
            .core
            .sort_by(|left, right| left.event_id.cmp(&right.event_id));
        self.bump_version();
        self.result(true, "OK", self.canonical_state_hash(), None)
    }

    pub fn quarantine(&mut self, event_id: &str, reason_code: &str) -> OperationResult {
        let before = self.canonical_state_hash();
        if !valid_short_token(event_id, MAX_EVENT_ID_LEN) {
            return self.result(false, "INVALID_EVENT_ID", before, None);
        }
        if !valid_short_token(reason_code, MAX_REASON_LEN) {
            return self.result(false, "INVALID_REASON", before, None);
        }
        if self.find_applied(event_id).is_none() {
            return self.result(false, "UNKNOWN_EVENT", before, None);
        }
        if self
            .state
            .quarantine
            .iter()
            .any(|entry| entry.event_id == event_id)
        {
            return self.result(false, "DUPLICATE_QUARANTINE", before, None);
        }
        self.state.core.retain(|entry| entry.event_id != event_id);
        self.state.quarantine.push(QuarantineEntry {
            event_id: event_id.to_string(),
            reason_code: reason_code.to_string(),
            version_id: self.state.version_id + 1,
        });
        self.state
            .quarantine
            .sort_by(|left, right| left.event_id.cmp(&right.event_id));
        self.bump_version();
        self.result(true, "OK", self.canonical_state_hash(), None)
    }

    pub fn promote(&mut self, event_id: &str) -> OperationResult {
        let before = self.canonical_state_hash();
        if !valid_short_token(event_id, MAX_EVENT_ID_LEN) {
            return self.result(false, "INVALID_EVENT_ID", before, None);
        }
        if self
            .state
            .quarantine
            .iter()
            .any(|entry| entry.event_id == event_id)
        {
            return self.result(false, "QUARANTINED_EVENT", before, None);
        }
        let Some(event) = self.find_core(event_id).cloned() else {
            let code = if self.find_applied(event_id).is_some() {
                "NOT_IN_CORE"
            } else {
                "UNKNOWN_EVENT"
            };
            return self.result(false, code, before, None);
        };
        if self
            .state
            .promoted
            .iter()
            .any(|entry| entry.event_id == event_id)
        {
            return self.result(false, "DUPLICATE_PROMOTION", before, None);
        }
        let promoted_version = self.state.version_id + 1;
        self.state.promoted.push(PromotedEvent {
            change: event.change,
            chronology_index: event.chronology_index,
            confidence_q16: event.confidence_q16,
            event_id: event.event_id,
            payload_hash: event.payload_hash,
            promoted_version,
            signature_hash: event.signature_hash,
        });
        self.evict_if_needed(promoted_version);
        self.state
            .promoted
            .sort_by(|left, right| left.event_id.cmp(&right.event_id));
        self.bump_version();
        self.result(true, "OK", self.canonical_state_hash(), None)
    }

    pub fn select_replay(&self, limit: u32) -> OperationResult {
        let before = self.canonical_state_hash();
        if limit > MAX_REPLAY_LIMIT {
            return self.result(false, "REPLAY_LIMIT_EXCEEDED", before, None);
        }
        let mut ordered = self.state.promoted.clone();
        ordered.sort_by(|left, right| {
            right
                .confidence_q16
                .cmp(&left.confidence_q16)
                .then(left.chronology_index.cmp(&right.chronology_index))
                .then(left.event_id.cmp(&right.event_id))
        });
        let selected = ordered
            .into_iter()
            .take(limit as usize)
            .map(|entry| entry.event_id)
            .collect();
        self.result(true, "OK", before, Some(selected))
    }

    pub fn roll_back(&mut self, version_id: i64) -> OperationResult {
        let before = self.canonical_state_hash();
        if version_id < 0 {
            return self.result(false, "ROLLBACK_PAST_ROOT", before, None);
        }
        let version_id = version_id as u32;
        let Some(state) = self.history.get(&version_id).cloned() else {
            return self.result(false, "ROLLBACK_VERSION_MISSING", before, None);
        };
        self.state = state;
        self.history
            .retain(|version, _state| *version <= version_id);
        self.result(true, "OK", self.canonical_state_hash(), None)
    }

    pub fn serialize(&self) -> Vec<u8> {
        let checkpoint = AdaptiveCheckpoint {
            abi_version: ABI_VERSION,
            active: self.state.clone(),
            history: self.history.values().cloned().collect(),
            schema: CHECKPOINT_SCHEMA.to_string(),
        };
        canonical_json(&checkpoint).into_bytes()
    }

    pub fn canonical_state(&self) -> AdaptiveState {
        self.state.clone()
    }

    pub fn canonical_state_json(&self) -> String {
        canonical_json(&self.state)
    }

    pub fn canonical_state_hash(&self) -> String {
        sha256_text(&self.canonical_state_json())
    }

    pub fn version_id(&self) -> u32 {
        self.state.version_id
    }

    fn result(
        &self,
        accepted: bool,
        code: &str,
        state_hash: String,
        selected_replay: Option<Vec<String>>,
    ) -> OperationResult {
        OperationResult {
            accepted,
            code: code.to_string(),
            state_hash,
            version_id: self.state.version_id,
            selected_replay,
        }
    }

    fn find_applied(&self, event_id: &str) -> Option<&AdaptiveEvent> {
        self.state
            .applied
            .iter()
            .find(|entry| entry.event_id == event_id)
    }

    fn find_core(&self, event_id: &str) -> Option<&AdaptiveEvent> {
        self.state
            .core
            .iter()
            .find(|entry| entry.event_id == event_id)
    }

    fn bump_version(&mut self) {
        self.state.version_id += 1;
        self.history
            .insert(self.state.version_id, self.state.clone());
        let min_kept = if self.state.version_id + 2 > self.state.history_capacity {
            self.state.version_id + 2 - self.state.history_capacity
        } else {
            1
        };
        self.history
            .retain(|version, _state| *version == 0 || *version >= min_kept);
    }

    fn evict_if_needed(&mut self, version_id: u32) {
        while self.state.promoted.len() > self.state.capacity as usize {
            let victim = self
                .state
                .promoted
                .iter()
                .min_by(|left, right| {
                    left.promoted_version
                        .cmp(&right.promoted_version)
                        .then(left.event_id.cmp(&right.event_id))
                })
                .expect("promoted list is non-empty when over capacity")
                .event_id
                .clone();
            self.state.promoted.retain(|entry| entry.event_id != victim);
            self.state.evicted.push(EvictedEntry {
                event_id: victim,
                version_id,
            });
            self.state.evicted.sort_by(|left, right| {
                left.version_id
                    .cmp(&right.version_id)
                    .then(left.event_id.cmp(&right.event_id))
            });
        }
    }
}

pub fn make_event(
    event_id: &str,
    chronology_index: u32,
    change: &str,
    confidence_q16: u16,
) -> AdaptiveEvent {
    let basis = format!("{event_id}:{chronology_index}:{change}:{confidence_q16}");
    AdaptiveEvent {
        change: change.to_string(),
        chronology_index,
        confidence_q16,
        event_id: event_id.to_string(),
        payload_hash: sha256_text(&format!("payload:{basis}")),
        signature_hash: sha256_text(&format!("signature:{basis}")),
    }
}

pub fn canonical_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("adaptive-state values serialize")
}

pub fn sha256_text(value: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(value.as_bytes());
    format!("sha256:{:x}", digest.finalize())
}

fn base_state(capacity: u32, history_capacity: u32) -> AdaptiveState {
    AdaptiveState {
        abi_version: ABI_VERSION,
        applied: Vec::new(),
        capacity,
        core: Vec::new(),
        evicted: Vec::new(),
        history_capacity,
        last_chronology: -1,
        promoted: Vec::new(),
        quarantine: Vec::new(),
        schema: STATE_SCHEMA.to_string(),
        version_id: 0,
    }
}

fn validate_event(event: &AdaptiveEvent) -> Result<(), &'static str> {
    if !valid_short_token(&event.event_id, MAX_EVENT_ID_LEN) {
        return Err("INVALID_EVENT_ID");
    }
    if !QUALIFIED_CHANGES.contains(&event.change.as_str()) {
        return Err("UNQUALIFIED_OPERATION");
    }
    if !valid_hash(&event.signature_hash) || !valid_hash(&event.payload_hash) {
        return Err("INVALID_HASH");
    }
    Ok(())
}

fn valid_hash(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..].chars().all(|char| char.is_ascii_hexdigit())
        && value[7..].chars().all(|char| !char.is_ascii_uppercase())
}

fn valid_short_token(value: &str, max_len: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_len
        && value.chars().all(|char| (' '..='~').contains(&char))
}

#[cfg(test)]
mod adaptive_state_tests {
    use super::*;

    #[test]
    fn adaptive_state_trace_round_trips_and_evicts_deterministically() {
        let mut kernel = AdaptiveStateKernel::new(2, 16).unwrap();
        let event1 = make_event("evt-0001", 0, "addition", 50_000);
        let event2 = make_event("evt-0002", 1, "supersession", 1_000);
        let event3 = make_event("evt-0003", 2, "recurrence", 40_000);
        let event4 = make_event("evt-0004", 3, "addition", 60_000);
        let event5 = make_event("evt-0005", 4, "addition", 65_000);

        assert!(kernel.apply_event(event1).accepted);
        assert!(kernel.acquire_core("evt-0001").accepted);
        assert!(kernel.promote("evt-0001").accepted);
        assert!(kernel.apply_event(event2).accepted);
        assert!(kernel.quarantine("evt-0002", "superseded").accepted);
        assert!(kernel.apply_event(event3).accepted);
        assert_eq!(
            kernel.select_replay(2).selected_replay.unwrap(),
            vec!["evt-0001".to_string()]
        );
        assert!(kernel.apply_event(event4).accepted);
        assert!(kernel.acquire_core("evt-0004").accepted);
        let rollback_version = kernel.promote("evt-0004").version_id;
        assert!(kernel.apply_event(event5).accepted);
        assert!(kernel.acquire_core("evt-0005").accepted);
        assert!(kernel.promote("evt-0005").accepted);

        assert_eq!(
            kernel.select_replay(4).selected_replay.unwrap(),
            vec!["evt-0005".to_string(), "evt-0004".to_string()]
        );
        assert_eq!(kernel.canonical_state().evicted[0].event_id, "evt-0001");
        let restored = AdaptiveStateKernel::restore(&kernel.serialize()).unwrap();
        assert_eq!(
            restored.canonical_state_json(),
            kernel.canonical_state_json()
        );
        assert!(kernel.roll_back(i64::from(rollback_version)).accepted);
        assert_eq!(
            kernel.select_replay(4).selected_replay.unwrap(),
            vec!["evt-0004".to_string(), "evt-0001".to_string()]
        );
    }

    #[test]
    fn adaptive_state_invalid_inputs_fail_closed() {
        let mut kernel = AdaptiveStateKernel::new(2, 8).unwrap();
        let event = make_event("evt-0001", 0, "addition", 1);
        assert!(kernel.apply_event(event.clone()).accepted);
        let before = kernel.canonical_state_hash();

        assert_eq!(kernel.apply_event(event).code, "DUPLICATE_EVENT");
        assert_eq!(kernel.canonical_state_hash(), before);
        assert_eq!(
            kernel
                .apply_event(make_event("evt-0000", 0, "addition", 1))
                .code,
            "OUT_OF_ORDER_EVENT"
        );
        assert_eq!(kernel.canonical_state_hash(), before);
        let mut bad = make_event("evt-0002", 2, "addition", 1);
        bad.payload_hash = "sha256:nope".to_string();
        assert_eq!(kernel.apply_event(bad).code, "INVALID_HASH");
        assert_eq!(kernel.promote("evt-missing").code, "UNKNOWN_EVENT");
        assert_eq!(kernel.roll_back(-1).code, "ROLLBACK_PAST_ROOT");
        assert_eq!(
            AdaptiveStateKernel::restore(b"broken").unwrap_err(),
            "checkpoint is not valid adaptive-state JSON"
        );
    }
}
