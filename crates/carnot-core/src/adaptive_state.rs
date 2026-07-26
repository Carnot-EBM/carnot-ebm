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
use std::collections::{BTreeMap, BTreeSet};

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

pub const ABI_V2_VERSION: u32 = 2;
pub const ABI_V2_STATE_SCHEMA: &str = "carnot.adaptive_state_abi.v2.state";
pub const ABI_V2_CHECKPOINT_SCHEMA: &str = "carnot.adaptive_state_abi.v2.checkpoint";
pub const ABI_V2_OPERATION_SCHEMA: &str = "carnot.adaptive_state_abi.v2.operation";
pub const ABI_V2_MAX_ACTIVE_CAPACITY: u32 = 16;
pub const ABI_V2_MAX_QUARANTINE_CAPACITY: u32 = 32;
pub const ABI_V2_MAX_KEY_LEN: usize = 96;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiV2ActiveEntry {
    pub event_id: String,
    pub key: String,
    pub payload_hash: String,
    pub promoted_version: u32,
    pub proposal_id: String,
    pub validator_receipt_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiV2CapacityEviction {
    pub event_id: String,
    pub evicted_key: String,
    pub evicted_proposal_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiV2ClosedEntry {
    pub event_id: String,
    pub payload_hash: String,
    pub proposal_id: String,
    pub proposal_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<String>,
    pub validator_receipt_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiV2Transaction {
    pub authorized_action: Option<String>,
    pub event_id: String,
    pub key: String,
    pub payload_hash: String,
    pub proposal_id: String,
    pub proposal_kind: String,
    pub snapshot_id: String,
    pub snapshot_state_hash: String,
    pub status: String,
    pub validator_receipt_hash: Option<String>,
    pub validator_status: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiV2State {
    pub abi_version: u32,
    pub active: Vec<AbiV2ActiveEntry>,
    pub active_capacity: u32,
    pub capacity_evictions: Vec<AbiV2CapacityEviction>,
    pub quarantine: Vec<AbiV2ClosedEntry>,
    pub quarantine_capacity: u32,
    pub rejected: Vec<AbiV2ClosedEntry>,
    pub schema: String,
    pub superseded: Vec<AbiV2ActiveEntry>,
    pub transactions: BTreeMap<String, AbiV2Transaction>,
    pub version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiV2Snapshot {
    pub active: Vec<AbiV2ActiveEntry>,
    pub event_id: String,
    pub event_index: u32,
    pub readable_state_hash: String,
    pub row_prefix_checksum: String,
    pub snapshot_id: String,
    pub state_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiV2OperationResult {
    pub abi_version: u32,
    pub accepted: bool,
    pub code: String,
    pub event_id: Option<String>,
    pub operation: String,
    pub payload_hash: Option<String>,
    pub previous_state_hash: String,
    pub proposal_id: Option<String>,
    pub resulting_state_hash: String,
    pub schema: String,
    pub snapshot_id: Option<String>,
    pub status: String,
    pub validator_receipt_hash: Option<String>,
    pub version: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiV2HistoryEntry {
    pub state: AbiV2State,
    pub state_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbiV2Checkpoint {
    pub abi_version: u32,
    pub history: Vec<AbiV2HistoryEntry>,
    pub ledger: Vec<AbiV2OperationResult>,
    pub schema: String,
    pub snapshots: Vec<AbiV2Snapshot>,
    pub state: AbiV2State,
    pub state_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AbiV2ProposalIdSeed<'a> {
    pub abi_version: u32,
    pub event_id: &'a str,
    pub key: &'a str,
    pub payload_hash: &'a str,
    pub proposal_kind: &'a str,
    pub snapshot_id: &'a str,
    pub snapshot_state_hash: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AbiV2SnapshotIdSeed<'a> {
    pub abi_version: u32,
    pub event_id: &'a str,
    pub event_index: u32,
    pub ordinal: usize,
    pub row_prefix_checksum: &'a str,
    pub state_hash: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct AbiV2ReadableState<'a> {
    pub active: &'a Vec<AbiV2ActiveEntry>,
    pub capacity_evictions: &'a Vec<AbiV2CapacityEviction>,
    pub quarantine: &'a Vec<AbiV2ClosedEntry>,
    pub rejected: &'a Vec<AbiV2ClosedEntry>,
    pub superseded: &'a Vec<AbiV2ActiveEntry>,
}

#[derive(Debug, Clone)]
pub struct AdaptiveStateAbiV2Kernel {
    state: AbiV2State,
    snapshots: BTreeMap<String, AbiV2Snapshot>,
    history: BTreeMap<String, AbiV2State>,
    ledger: Vec<AbiV2OperationResult>,
    written_events: BTreeSet<String>,
    released: bool,
}

impl AdaptiveStateAbiV2Kernel {
    pub fn new(active_capacity: u32, quarantine_capacity: u32) -> Result<Self, String> {
        if !(1..=ABI_V2_MAX_ACTIVE_CAPACITY).contains(&active_capacity) {
            return Err("active_capacity must be an integer in [1, 16]".to_string());
        }
        if !(1..=ABI_V2_MAX_QUARANTINE_CAPACITY).contains(&quarantine_capacity) {
            return Err("quarantine_capacity must be an integer in [1, 32]".to_string());
        }
        let state = abi_v2_base_state(active_capacity, quarantine_capacity);
        let mut kernel = Self {
            state,
            snapshots: BTreeMap::new(),
            history: BTreeMap::new(),
            ledger: Vec::new(),
            written_events: BTreeSet::new(),
            released: false,
        };
        kernel
            .history
            .insert(kernel.canonical_state_hash(), kernel.state.clone());
        Ok(kernel)
    }

    pub fn recover(checkpoint: &[u8]) -> Result<Self, String> {
        let payload: AbiV2Checkpoint = serde_json::from_slice(checkpoint)
            .map_err(|_| "checkpoint is not valid ABI v2 JSON".to_string())?;
        if payload.schema != ABI_V2_CHECKPOINT_SCHEMA {
            return Err("checkpoint schema mismatch".to_string());
        }
        if payload.abi_version != ABI_V2_VERSION {
            return Err("checkpoint ABI version mismatch".to_string());
        }
        let mut kernel = Self::new(
            payload.state.active_capacity,
            payload.state.quarantine_capacity,
        )?;
        kernel.state = payload.state;
        if kernel.canonical_state_hash() != payload.state_hash {
            return Err("checkpoint state hash mismatch".to_string());
        }
        kernel.snapshots = payload
            .snapshots
            .into_iter()
            .map(|snapshot| (snapshot.snapshot_id.clone(), snapshot))
            .collect();
        kernel.history = payload
            .history
            .into_iter()
            .map(|entry| (entry.state_hash, entry.state))
            .collect();
        kernel.ledger = payload.ledger;
        kernel.written_events = kernel
            .state
            .active
            .iter()
            .map(|entry| entry.event_id.clone())
            .chain(
                kernel
                    .state
                    .quarantine
                    .iter()
                    .map(|entry| entry.event_id.clone()),
            )
            .chain(
                kernel
                    .state
                    .rejected
                    .iter()
                    .map(|entry| entry.event_id.clone()),
            )
            .collect();
        let Some(active_history) = kernel.history.get(&payload.state_hash) else {
            return Err("checkpoint active state missing from history".to_string());
        };
        if active_history != &kernel.state {
            return Err("checkpoint active state differs from history".to_string());
        }
        Ok(kernel)
    }

    pub fn snapshot(
        &mut self,
        event_id: &str,
        event_index: u32,
        row_prefix_checksum: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live("snapshot", Some(event_id)) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                "snapshot",
                Some(event_id),
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        if !valid_short_token(event_id, MAX_EVENT_ID_LEN) {
            return self.reject(
                "snapshot",
                Some(event_id),
                "INVALID_EVENT_ID",
                before,
                None,
                None,
                None,
                None,
            );
        }
        if !valid_hash(row_prefix_checksum) {
            return self.reject(
                "snapshot",
                Some(event_id),
                "INVALID_PREFIX_HASH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let snapshot_id = sha256_json(&AbiV2SnapshotIdSeed {
            abi_version: ABI_V2_VERSION,
            event_id,
            event_index,
            ordinal: self.snapshots.len(),
            row_prefix_checksum,
            state_hash: &before,
        });
        let snapshot = AbiV2Snapshot {
            active: self.state.active.clone(),
            event_id: event_id.to_string(),
            event_index,
            readable_state_hash: self.readable_state_hash(),
            row_prefix_checksum: row_prefix_checksum.to_string(),
            snapshot_id: snapshot_id.clone(),
            state_hash: before.clone(),
        };
        self.snapshots.insert(snapshot_id.clone(), snapshot);
        self.accept(
            "snapshot",
            Some(event_id),
            before.clone(),
            before,
            None,
            None,
            Some(snapshot_id),
            "snapshotted",
            None,
        )
    }

    pub fn lookup(
        &mut self,
        event_id: &str,
        snapshot_id: &str,
        key: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live("lookup", Some(event_id)) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                "lookup",
                Some(event_id),
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let Some(snapshot) = self.snapshots.get(snapshot_id) else {
            return self.reject(
                "lookup",
                Some(event_id),
                "STALE_SNAPSHOT",
                before,
                None,
                None,
                None,
                None,
            );
        };
        if snapshot.event_id != event_id {
            return self.reject(
                "lookup",
                Some(event_id),
                "STALE_SNAPSHOT",
                before,
                None,
                None,
                None,
                None,
            );
        }
        if self.written_events.contains(event_id) {
            return self.reject(
                "lookup",
                Some(event_id),
                "SAME_EVENT_READ_AFTER_WRITE",
                before,
                None,
                None,
                None,
                None,
            );
        }
        if !valid_short_token(key, ABI_V2_MAX_KEY_LEN) {
            return self.reject(
                "lookup",
                Some(event_id),
                "INVALID_KEY",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let hit = snapshot.active.iter().find(|entry| entry.key == key);
        let payload_hash = hit.map(|entry| entry.payload_hash.clone());
        let status = if hit.is_some() { "hit" } else { "miss" };
        self.accept(
            "lookup",
            Some(event_id),
            before.clone(),
            before,
            payload_hash,
            None,
            Some(snapshot_id.to_string()),
            status,
            None,
        )
    }

    pub fn propose(
        &mut self,
        event_id: &str,
        snapshot_id: &str,
        proposal_kind: &str,
        key: &str,
        payload_hash: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live("propose", Some(event_id)) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                "propose",
                Some(event_id),
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let Some(snapshot) = self.snapshots.get(snapshot_id) else {
            return self.reject(
                "propose",
                Some(event_id),
                "STALE_SNAPSHOT",
                before,
                None,
                None,
                None,
                None,
            );
        };
        if snapshot.event_id != event_id
            || snapshot.readable_state_hash != self.readable_state_hash()
        {
            return self.reject(
                "propose",
                Some(event_id),
                "STALE_SNAPSHOT",
                before,
                None,
                None,
                None,
                None,
            );
        }
        if !valid_short_token(proposal_kind, ABI_V2_MAX_KEY_LEN) {
            return self.reject(
                "propose",
                Some(event_id),
                "INVALID_PROPOSAL_KIND",
                before,
                None,
                None,
                None,
                None,
            );
        }
        if !valid_short_token(key, ABI_V2_MAX_KEY_LEN) {
            return self.reject(
                "propose",
                Some(event_id),
                "INVALID_KEY",
                before,
                None,
                None,
                None,
                None,
            );
        }
        if !valid_hash(payload_hash) {
            return self.reject(
                "propose",
                Some(event_id),
                "INVALID_PAYLOAD_HASH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let proposal_id = sha256_json(&AbiV2ProposalIdSeed {
            abi_version: ABI_V2_VERSION,
            event_id,
            key,
            payload_hash,
            proposal_kind,
            snapshot_id,
            snapshot_state_hash: &snapshot.state_hash,
        });
        if self.state.transactions.contains_key(&proposal_id) {
            return self.reject(
                "propose",
                Some(event_id),
                "REPLAYED_PROPOSAL",
                before,
                Some(payload_hash.to_string()),
                Some(proposal_id),
                Some(snapshot_id.to_string()),
                None,
            );
        }
        self.state.transactions.insert(
            proposal_id.clone(),
            AbiV2Transaction {
                authorized_action: None,
                event_id: event_id.to_string(),
                key: key.to_string(),
                payload_hash: payload_hash.to_string(),
                proposal_id: proposal_id.clone(),
                proposal_kind: proposal_kind.to_string(),
                snapshot_id: snapshot_id.to_string(),
                snapshot_state_hash: snapshot.state_hash.clone(),
                status: "proposed".to_string(),
                validator_receipt_hash: None,
                validator_status: None,
            },
        );
        self.bump();
        let after = self.canonical_state_hash();
        self.accept(
            "propose",
            Some(event_id),
            before,
            after,
            Some(payload_hash.to_string()),
            Some(proposal_id),
            Some(snapshot_id.to_string()),
            "proposed",
            None,
        )
    }

    pub fn commit(
        &mut self,
        event_id: &str,
        proposal_id: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live("commit", Some(event_id)) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                "commit",
                Some(event_id),
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let Some(tx) = self.transaction_for(event_id, proposal_id) else {
            return self.reject(
                "commit",
                Some(event_id),
                "INVALID_ORDER",
                before,
                None,
                Some(proposal_id.to_string()),
                None,
                None,
            );
        };
        if tx.status != "proposed" {
            return self.reject(
                "commit",
                Some(event_id),
                "REPLAYED_COMMIT",
                before,
                Some(tx.payload_hash),
                Some(proposal_id.to_string()),
                None,
                None,
            );
        }
        self.state
            .transactions
            .get_mut(proposal_id)
            .expect("transaction exists")
            .status = "committed".to_string();
        self.bump();
        let after = self.canonical_state_hash();
        let payload_hash = self.state.transactions[proposal_id].payload_hash.clone();
        self.accept(
            "commit",
            Some(event_id),
            before,
            after,
            Some(payload_hash),
            Some(proposal_id.to_string()),
            None,
            "committed",
            None,
        )
    }

    pub fn validate(
        &mut self,
        event_id: &str,
        proposal_id: &str,
        validator_receipt_hash: &str,
        validator_status: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live("validate", Some(event_id)) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                "validate",
                Some(event_id),
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let Some(tx) = self.transaction_for(event_id, proposal_id) else {
            return self.reject(
                "validate",
                Some(event_id),
                "INVALID_ORDER",
                before,
                None,
                Some(proposal_id.to_string()),
                None,
                None,
            );
        };
        if tx.status != "committed" {
            return self.reject(
                "validate",
                Some(event_id),
                "INVALID_ORDER",
                before,
                None,
                Some(proposal_id.to_string()),
                None,
                None,
            );
        }
        if !valid_hash(validator_receipt_hash) {
            return self.reject(
                "validate",
                Some(event_id),
                "INVALID_VALIDATOR_RECEIPT",
                before,
                Some(tx.payload_hash),
                Some(proposal_id.to_string()),
                None,
                None,
            );
        }
        let Some(action) = validator_status_to_action(validator_status) else {
            return self.reject(
                "validate",
                Some(event_id),
                "INVALID_VALIDATOR_STATUS",
                before,
                Some(tx.payload_hash),
                Some(proposal_id.to_string()),
                None,
                None,
            );
        };
        {
            let tx_mut = self
                .state
                .transactions
                .get_mut(proposal_id)
                .expect("transaction exists");
            tx_mut.authorized_action = Some(action.to_string());
            tx_mut.status = "validated".to_string();
            tx_mut.validator_receipt_hash = Some(validator_receipt_hash.to_string());
            tx_mut.validator_status = Some(validator_status.to_string());
        }
        self.bump();
        let after = self.canonical_state_hash();
        let payload_hash = self.state.transactions[proposal_id].payload_hash.clone();
        self.accept(
            "validate",
            Some(event_id),
            before,
            after,
            Some(payload_hash),
            Some(proposal_id.to_string()),
            None,
            "validated",
            Some(validator_receipt_hash.to_string()),
        )
    }

    pub fn supersede(
        &mut self,
        event_id: &str,
        proposal_id: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live("supersede", Some(event_id)) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                "supersede",
                Some(event_id),
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let Some(tx) = self.promotable_transaction(event_id, proposal_id) else {
            return self.reject(
                "supersede",
                Some(event_id),
                "INVALID_ORDER",
                before,
                None,
                Some(proposal_id.to_string()),
                None,
                None,
            );
        };
        let Some(existing) = self.active_for_key(&tx.key).cloned() else {
            return self.reject(
                "supersede",
                Some(event_id),
                "NO_ACTIVE_TARGET",
                before,
                Some(tx.payload_hash),
                Some(proposal_id.to_string()),
                None,
                None,
            );
        };
        self.state
            .active
            .retain(|entry| entry.proposal_id != existing.proposal_id);
        let mut closed = existing;
        closed.proposal_id = closed.proposal_id.clone();
        self.state.superseded.push(AbiV2ActiveEntry {
            event_id: closed.event_id,
            key: closed.key,
            payload_hash: closed.payload_hash,
            promoted_version: closed.promoted_version,
            proposal_id: closed.proposal_id,
            validator_receipt_hash: closed.validator_receipt_hash,
        });
        self.state
            .transactions
            .get_mut(proposal_id)
            .expect("transaction exists")
            .status = "superseded_ready".to_string();
        self.bump();
        let after = self.canonical_state_hash();
        let tx_after = self.state.transactions[proposal_id].clone();
        self.accept(
            "supersede",
            Some(event_id),
            before,
            after,
            Some(tx_after.payload_hash),
            Some(proposal_id.to_string()),
            None,
            "superseded",
            tx_after.validator_receipt_hash,
        )
    }

    pub fn promote(
        &mut self,
        event_id: &str,
        proposal_id: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live("promote", Some(event_id)) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                "promote",
                Some(event_id),
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let Some(tx) = self.promotable_transaction(event_id, proposal_id) else {
            return self.reject(
                "promote",
                Some(event_id),
                "INVALID_ORDER",
                before,
                None,
                Some(proposal_id.to_string()),
                None,
                None,
            );
        };
        if self.active_for_key(&tx.key).is_some() && tx.status != "superseded_ready" {
            return self.reject(
                "promote",
                Some(event_id),
                "SUPERSEDE_REQUIRED",
                before,
                Some(tx.payload_hash),
                Some(proposal_id.to_string()),
                None,
                None,
            );
        }
        self.state.active.push(AbiV2ActiveEntry {
            event_id: event_id.to_string(),
            key: tx.key.clone(),
            payload_hash: tx.payload_hash.clone(),
            promoted_version: self.state.version + 1,
            proposal_id: proposal_id.to_string(),
            validator_receipt_hash: tx.validator_receipt_hash.clone().unwrap_or_default(),
        });
        self.state
            .active
            .sort_by(|left, right| left.key.cmp(&right.key));
        self.state
            .transactions
            .get_mut(proposal_id)
            .expect("transaction exists")
            .status = "promoted".to_string();
        self.written_events.insert(event_id.to_string());
        self.enforce_active_capacity(event_id);
        self.bump();
        let after = self.canonical_state_hash();
        self.accept(
            "promote",
            Some(event_id),
            before,
            after,
            Some(tx.payload_hash),
            Some(proposal_id.to_string()),
            None,
            "promoted",
            tx.validator_receipt_hash,
        )
    }

    pub fn quarantine(
        &mut self,
        event_id: &str,
        proposal_id: &str,
        reason_code: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        self.close_non_active(
            "quarantine",
            event_id,
            proposal_id,
            Some(reason_code),
            expected_prior_state_hash,
        )
    }

    pub fn reject_update(
        &mut self,
        event_id: &str,
        proposal_id: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        self.close_non_active(
            "reject",
            event_id,
            proposal_id,
            None,
            expected_prior_state_hash,
        )
    }

    pub fn rollback(
        &mut self,
        event_id: &str,
        target_state_hash: &str,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live("rollback", Some(event_id)) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                "rollback",
                Some(event_id),
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let Some(state) = self.history.get(target_state_hash).cloned() else {
            return self.reject(
                "rollback",
                Some(event_id),
                "ROLLBACK_TARGET_MISSING",
                before,
                None,
                None,
                None,
                None,
            );
        };
        self.state = state;
        self.written_events = self
            .state
            .active
            .iter()
            .map(|entry| entry.event_id.clone())
            .chain(
                self.state
                    .quarantine
                    .iter()
                    .map(|entry| entry.event_id.clone()),
            )
            .chain(
                self.state
                    .rejected
                    .iter()
                    .map(|entry| entry.event_id.clone()),
            )
            .collect();
        let after = self.canonical_state_hash();
        self.accept(
            "rollback",
            Some(event_id),
            before,
            after,
            None,
            None,
            None,
            "rolled_back",
            None,
        )
    }

    pub fn partial_state_transition_probe(
        &mut self,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live("partial_state_transition_probe", None) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                "partial_state_transition_probe",
                None,
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let mut work = self.state.clone();
        work.active.push(AbiV2ActiveEntry {
            event_id: "partial".to_string(),
            key: "partial".to_string(),
            payload_hash: sha256_text("partial"),
            promoted_version: 0,
            proposal_id: "partial".to_string(),
            validator_receipt_hash: sha256_text("partial-validator"),
        });
        if work != self.state {
            return self.reject(
                "partial_state_transition_probe",
                None,
                "PARTIAL_STATE_TRANSITION_REJECTED",
                before,
                None,
                None,
                None,
                None,
            );
        }
        self.accept(
            "partial_state_transition_probe",
            None,
            before.clone(),
            before,
            None,
            None,
            None,
            "ok",
            None,
        )
    }

    pub fn release(&mut self) -> AbiV2OperationResult {
        let before = self.canonical_state_hash();
        if self.released {
            return self.reject(
                "release",
                None,
                "DOUBLE_RELEASE",
                before,
                None,
                None,
                None,
                None,
            );
        }
        self.released = true;
        self.accept(
            "release",
            None,
            before.clone(),
            before,
            None,
            None,
            None,
            "released",
            None,
        )
    }

    pub fn serialize(&self) -> Vec<u8> {
        let checkpoint = AbiV2Checkpoint {
            abi_version: ABI_V2_VERSION,
            history: self
                .history
                .iter()
                .map(|(state_hash, state)| AbiV2HistoryEntry {
                    state: state.clone(),
                    state_hash: state_hash.clone(),
                })
                .collect(),
            ledger: self.ledger.clone(),
            schema: ABI_V2_CHECKPOINT_SCHEMA.to_string(),
            snapshots: self.snapshots.values().cloned().collect(),
            state: self.state.clone(),
            state_hash: self.canonical_state_hash(),
        };
        canonical_json(&checkpoint).into_bytes()
    }

    pub fn canonical_state(&self) -> AbiV2State {
        self.state.clone()
    }

    pub fn canonical_state_json(&self) -> String {
        canonical_json(&self.state)
    }

    pub fn canonical_state_hash(&self) -> String {
        sha256_text(&self.canonical_state_json())
    }

    pub fn readable_state_hash(&self) -> String {
        sha256_json(&AbiV2ReadableState {
            active: &self.state.active,
            capacity_evictions: &self.state.capacity_evictions,
            quarantine: &self.state.quarantine,
            rejected: &self.state.rejected,
            superseded: &self.state.superseded,
        })
    }

    pub fn version(&self) -> u32 {
        self.state.version
    }

    fn close_non_active(
        &mut self,
        operation: &str,
        event_id: &str,
        proposal_id: &str,
        reason_code: Option<&str>,
        expected_prior_state_hash: &str,
    ) -> AbiV2OperationResult {
        if let Some(result) = self.ensure_live(operation, Some(event_id)) {
            return result;
        }
        let before = self.canonical_state_hash();
        if expected_prior_state_hash != before {
            return self.reject(
                operation,
                Some(event_id),
                "PRIOR_STATE_MISMATCH",
                before,
                None,
                None,
                None,
                None,
            );
        }
        let Some(tx) = self.transaction_for(event_id, proposal_id) else {
            return self.reject(
                operation,
                Some(event_id),
                "INVALID_ORDER",
                before,
                None,
                Some(proposal_id.to_string()),
                None,
                None,
            );
        };
        if tx.status != "validated" || tx.authorized_action.as_deref() != Some(operation) {
            return self.reject(
                operation,
                Some(event_id),
                "INVALID_ORDER",
                before,
                None,
                Some(proposal_id.to_string()),
                None,
                None,
            );
        }
        let mut entry = AbiV2ClosedEntry {
            event_id: event_id.to_string(),
            payload_hash: tx.payload_hash.clone(),
            proposal_id: proposal_id.to_string(),
            proposal_kind: tx.proposal_kind.clone(),
            reason_code: None,
            validator_receipt_hash: tx.validator_receipt_hash.clone().unwrap_or_default(),
        };
        let status;
        if operation == "quarantine" {
            let Some(reason) = reason_code else {
                return self.reject(
                    operation,
                    Some(event_id),
                    "INVALID_REASON",
                    before,
                    Some(tx.payload_hash),
                    Some(proposal_id.to_string()),
                    None,
                    tx.validator_receipt_hash,
                );
            };
            if !valid_short_token(reason, MAX_REASON_LEN) {
                return self.reject(
                    operation,
                    Some(event_id),
                    "INVALID_REASON",
                    before,
                    Some(tx.payload_hash),
                    Some(proposal_id.to_string()),
                    None,
                    tx.validator_receipt_hash,
                );
            }
            entry.reason_code = Some(reason.to_string());
            self.state.quarantine.push(entry);
            let len = self.state.quarantine.len();
            let cap = self.state.quarantine_capacity as usize;
            if len > cap {
                self.state.quarantine = self.state.quarantine[len - cap..].to_vec();
            }
            self.state
                .transactions
                .get_mut(proposal_id)
                .expect("transaction exists")
                .status = "quarantined".to_string();
            status = "quarantined";
        } else {
            self.state.rejected.push(entry);
            self.state
                .transactions
                .get_mut(proposal_id)
                .expect("transaction exists")
                .status = "rejected".to_string();
            status = "rejected";
        }
        self.written_events.insert(event_id.to_string());
        self.bump();
        let after = self.canonical_state_hash();
        self.accept(
            operation,
            Some(event_id),
            before,
            after,
            Some(tx.payload_hash),
            Some(proposal_id.to_string()),
            None,
            status,
            tx.validator_receipt_hash,
        )
    }

    fn transaction_for(&self, event_id: &str, proposal_id: &str) -> Option<AbiV2Transaction> {
        let tx = self.state.transactions.get(proposal_id)?;
        if tx.event_id == event_id {
            Some(tx.clone())
        } else {
            None
        }
    }

    fn promotable_transaction(
        &self,
        event_id: &str,
        proposal_id: &str,
    ) -> Option<AbiV2Transaction> {
        let tx = self.transaction_for(event_id, proposal_id)?;
        if tx.authorized_action.as_deref() != Some("promote") {
            return None;
        }
        if tx.status == "validated" || tx.status == "superseded_ready" {
            Some(tx)
        } else {
            None
        }
    }

    fn active_for_key(&self, key: &str) -> Option<&AbiV2ActiveEntry> {
        self.state.active.iter().find(|entry| entry.key == key)
    }

    fn enforce_active_capacity(&mut self, event_id: &str) {
        while self.state.active.len() > self.state.active_capacity as usize {
            let victim_index = self
                .state
                .active
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    left.promoted_version
                        .cmp(&right.promoted_version)
                        .then(left.key.cmp(&right.key))
                })
                .map(|(index, _)| index)
                .expect("active list is non-empty when over capacity");
            let victim = self.state.active.remove(victim_index);
            self.state.capacity_evictions.push(AbiV2CapacityEviction {
                event_id: event_id.to_string(),
                evicted_key: victim.key,
                evicted_proposal_id: victim.proposal_id,
            });
        }
    }

    fn ensure_live(
        &mut self,
        operation: &str,
        event_id: Option<&str>,
    ) -> Option<AbiV2OperationResult> {
        if self.released {
            let before = self.canonical_state_hash();
            Some(self.reject(
                operation,
                event_id,
                "USE_AFTER_RELEASE",
                before,
                None,
                None,
                None,
                None,
            ))
        } else {
            None
        }
    }

    fn bump(&mut self) {
        self.state.version += 1;
        self.history
            .insert(self.canonical_state_hash(), self.state.clone());
    }

    #[allow(clippy::too_many_arguments)]
    fn accept(
        &mut self,
        operation: &str,
        event_id: Option<&str>,
        before: String,
        after: String,
        payload_hash: Option<String>,
        proposal_id: Option<String>,
        snapshot_id: Option<String>,
        status: &str,
        validator_receipt_hash: Option<String>,
    ) -> AbiV2OperationResult {
        self.result(
            operation,
            event_id,
            true,
            "OK",
            before,
            after,
            payload_hash,
            proposal_id,
            snapshot_id,
            status,
            validator_receipt_hash,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reject(
        &mut self,
        operation: &str,
        event_id: Option<&str>,
        code: &str,
        before: String,
        payload_hash: Option<String>,
        proposal_id: Option<String>,
        snapshot_id: Option<String>,
        validator_receipt_hash: Option<String>,
    ) -> AbiV2OperationResult {
        self.result(
            operation,
            event_id,
            false,
            code,
            before.clone(),
            before,
            payload_hash,
            proposal_id,
            snapshot_id,
            "unchanged",
            validator_receipt_hash,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn result(
        &mut self,
        operation: &str,
        event_id: Option<&str>,
        accepted: bool,
        code: &str,
        before: String,
        after: String,
        payload_hash: Option<String>,
        proposal_id: Option<String>,
        snapshot_id: Option<String>,
        status: &str,
        validator_receipt_hash: Option<String>,
    ) -> AbiV2OperationResult {
        let receipt = AbiV2OperationResult {
            abi_version: ABI_V2_VERSION,
            accepted,
            code: code.to_string(),
            event_id: event_id.map(str::to_string),
            operation: operation.to_string(),
            payload_hash,
            previous_state_hash: before,
            proposal_id,
            resulting_state_hash: after,
            schema: ABI_V2_OPERATION_SCHEMA.to_string(),
            snapshot_id,
            status: status.to_string(),
            validator_receipt_hash,
            version: self.state.version,
        };
        self.ledger.push(receipt.clone());
        receipt
    }
}

fn abi_v2_base_state(active_capacity: u32, quarantine_capacity: u32) -> AbiV2State {
    AbiV2State {
        abi_version: ABI_V2_VERSION,
        active: Vec::new(),
        active_capacity,
        capacity_evictions: Vec::new(),
        quarantine: Vec::new(),
        quarantine_capacity,
        rejected: Vec::new(),
        schema: ABI_V2_STATE_SCHEMA.to_string(),
        superseded: Vec::new(),
        transactions: BTreeMap::new(),
        version: 0,
    }
}

fn validator_status_to_action(status: &str) -> Option<&'static str> {
    match status {
        "valid" => Some("promote"),
        "quarantine" => Some("quarantine"),
        "reject" => Some("reject"),
        _ => None,
    }
}

pub fn sha256_json<T: Serialize>(value: &T) -> String {
    sha256_text(&canonical_json(value))
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

    #[test]
    fn adaptive_state_abi_v2_ordering_recovery_and_lifetime_are_bounded() {
        let prefix = sha256_text("prefix");
        let payload = sha256_text("payload");
        let receipt = sha256_text("receipt");
        let mut kernel = AdaptiveStateAbiV2Kernel::new(2, 3).unwrap();
        let root = kernel.canonical_state_hash();
        let snapshot = kernel.snapshot("event-0001", 0, &prefix, &root);
        assert!(snapshot.accepted);
        let snapshot_id = snapshot.snapshot_id.clone().unwrap();
        let proposed = kernel.propose(
            "event-0001",
            &snapshot_id,
            "exact_outcome_fact",
            "fact::one",
            &payload,
            &kernel.canonical_state_hash(),
        );
        assert!(proposed.accepted);
        let proposal_id = proposed.proposal_id.clone().unwrap();
        assert!(
            kernel
                .commit("event-0001", &proposal_id, &kernel.canonical_state_hash())
                .accepted
        );
        assert_eq!(
            kernel
                .commit("event-0001", &proposal_id, &kernel.canonical_state_hash())
                .code,
            "REPLAYED_COMMIT"
        );
        assert!(
            kernel
                .validate(
                    "event-0001",
                    &proposal_id,
                    &receipt,
                    "valid",
                    &kernel.canonical_state_hash()
                )
                .accepted
        );
        let promoted = kernel.promote("event-0001", &proposal_id, &kernel.canonical_state_hash());
        assert!(promoted.accepted);
        let recovered = AdaptiveStateAbiV2Kernel::recover(&kernel.serialize()).unwrap();
        assert_eq!(
            recovered.canonical_state_json(),
            kernel.canonical_state_json()
        );

        assert!(kernel.release().accepted);
        assert_eq!(
            kernel
                .lookup(
                    "event-0001",
                    &snapshot_id,
                    "fact::one",
                    &kernel.canonical_state_hash()
                )
                .code,
            "USE_AFTER_RELEASE"
        );
        assert_eq!(kernel.release().code, "DOUBLE_RELEASE");
    }
}
