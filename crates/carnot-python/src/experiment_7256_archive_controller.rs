//! Persistent FIFO archive controller used by the Exp7256 prototype.
//!
//! Python owns experiment evidence and durable file publication. This object
//! owns every decision-bearing controller field between typed event calls.
//!
//! Spec: REQ-CL-7256, REQ-RUSTPY-7256

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const FAMILY_NAMES: [&str; 4] = [
    "lower_bound",
    "upper_bound",
    "modular_equals",
    "cyclic_window",
];
const DOMAIN_SIZE: usize = 33;
const FULL_MASK: u64 = (1_u64 << DOMAIN_SIZE) - 1;
const ARCHIVE_CAP: usize = 4;
const VALIDATION_WINDOW: usize = 16;
const MIN_VALIDATION_WITNESSES: usize = 8;

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn family_name(code: u8) -> Result<&'static str, &'static str> {
    FAMILY_NAMES
        .get(code as usize)
        .copied()
        .ok_or("invalid_public_event")
}

fn family_index(name: &str) -> Option<usize> {
    FAMILY_NAMES.iter().position(|candidate| *candidate == name)
}

fn accepts(family: usize, value: usize, parameter: usize) -> bool {
    match family {
        0 => value >= parameter,
        1 => value <= parameter,
        2 => value == parameter,
        3 => (value + DOMAIN_SIZE - parameter) % DOMAIN_SIZE < 8,
        _ => false,
    }
}

fn accept_mask(family: usize, value: usize) -> u64 {
    (0..DOMAIN_SIZE).fold(0_u64, |mask, parameter| {
        if accepts(family, value, parameter) {
            mask | (1_u64 << parameter)
        } else {
            mask
        }
    })
}

fn votes_for(family: usize, mask: u64) -> Vec<u8> {
    (0..DOMAIN_SIZE)
        .map(|value| (mask & accept_mask(family, value)).count_ones() as u8)
        .collect()
}

fn canonical_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let mut encoded = serde_json::to_string(&serde_json::to_value(value)?)?;
    encoded.push('\n');
    Ok(encoded)
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let encoded = canonical_json(value)?;
    Ok(format!("sha256:{:x}", Sha256::digest(encoded.as_bytes())))
}

fn valid_hash(value: &Option<String>) -> bool {
    value.as_ref().is_none_or(|text| {
        text.len() == 71
            && text.starts_with("sha256:")
            && text[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
    })
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Release {
    event_id: String,
    family_id: String,
    numeric_value: i64,
    observed_label: String,
    role: String,
    request_index: i64,
    release_index: i64,
}

impl Release {
    fn validate(&self, current_cycle: Option<i64>) -> Result<(), &'static str> {
        if self.event_id.is_empty() || family_index(&self.family_id).is_none() {
            return Err("invalid_public_event");
        }
        if self.observed_label != "accept" && self.observed_label != "reject" {
            return Err("invalid_label");
        }
        if self.role != "support" && self.role != "validation" {
            return Err("invalid_role");
        }
        if self.release_index < self.request_index {
            return Err("release_before_request");
        }
        if current_cycle.is_some_and(|cycle| self.release_index > cycle) {
            return Err("future_release");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FamilyState {
    epoch: u64,
    provenance: Vec<Release>,
    survivor_mask: u64,
    vote_counts: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ActiveState {
    families: BTreeMap<String, FamilyState>,
    parent_hash: Option<String>,
    schema: String,
    version: u64,
}

impl ActiveState {
    fn initial_from_masks(masks: &BTreeMap<String, u64>) -> Self {
        let families = FAMILY_NAMES
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let mask = *masks.get(*name).unwrap_or(&FULL_MASK);
                (
                    (*name).to_string(),
                    FamilyState {
                        epoch: 0,
                        provenance: Vec::new(),
                        survivor_mask: mask,
                        vote_counts: votes_for(index, mask),
                    },
                )
            })
            .collect();
        Self {
            families,
            parent_hash: None,
            schema: "carnot.packed_belief_state.v1".to_string(),
            version: 0,
        }
    }

    fn initial() -> Self {
        Self::initial_from_masks(&BTreeMap::new())
    }

    fn validate(&self) -> Result<(), &'static str> {
        if self.schema != "carnot.packed_belief_state.v1" || !valid_hash(&self.parent_hash) {
            return Err("invalid_active_state");
        }
        if self.families.len() != FAMILY_NAMES.len() {
            return Err("invalid_state_families");
        }
        for (index, name) in FAMILY_NAMES.iter().enumerate() {
            let family = self.families.get(*name).ok_or("invalid_state_families")?;
            if family.survivor_mask & !FULL_MASK != 0
                || family.vote_counts != votes_for(index, family.survivor_mask)
            {
                return Err("invalid_active_cache");
            }
            for release in &family.provenance {
                release.validate(None)?;
            }
        }
        Ok(())
    }

    fn state_hash(&self) -> Result<String, serde_json::Error> {
        sha256_json(self)
    }

    fn prediction(&self, family: usize, value: usize) -> (i8, f64) {
        let row = &self.families[FAMILY_NAMES[family]];
        let count = row.survivor_mask.count_ones();
        if count == 0 {
            return (-1, 0.0);
        }
        let accepted = row.vote_counts[value] as u32;
        let disagreement = accepted.min(count - accepted) as f64 / count as f64;
        (if accepted * 2 > count { 1 } else { 0 }, disagreement)
    }

    fn apply_release(&mut self, release: &Release) -> Result<(), serde_json::Error> {
        self.parent_hash = Some(self.state_hash()?);
        self.version += 1;
        let index = family_index(&release.family_id).expect("validated family");
        let row = self
            .families
            .get_mut(&release.family_id)
            .expect("validated family state");
        if release.role == "support" {
            let accepted = accept_mask(index, release.numeric_value as usize);
            let matching = if release.observed_label == "accept" {
                accepted
            } else {
                FULL_MASK ^ accepted
            };
            let survivor = row.survivor_mask & matching;
            if survivor == 0 {
                row.survivor_mask = FULL_MASK & matching;
                row.epoch += 1;
            } else {
                row.survivor_mask = survivor;
            }
        }
        row.provenance.push(release.clone());
        for (family_index, name) in FAMILY_NAMES.iter().enumerate() {
            let family = self.families.get_mut(*name).expect("complete family map");
            family.vote_counts = votes_for(family_index, family.survivor_mask);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ArchiveEntry {
    archive_id: String,
    creation_order: u64,
    state_hash: String,
    survivor_masks: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateReceipt {
    archive_id: String,
    creation_order: u64,
    nomination_order: usize,
    released_witness_count: usize,
    validation_loss: usize,
    contradiction_count: usize,
    gate_passed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct NominationReceipt {
    candidate_count: usize,
    candidates: Vec<CandidateReceipt>,
    final_gate: String,
    minimum_witnesses: usize,
    selected_archive_id: Option<String>,
    validation_window_size: usize,
}

impl NominationReceipt {
    fn empty() -> Self {
        Self {
            candidate_count: 0,
            candidates: Vec::new(),
            final_gate: "at_least_8_released_witnesses_and_zero_contradictions".to_string(),
            minimum_witnesses: MIN_VALIDATION_WITNESSES,
            selected_archive_id: None,
            validation_window_size: 0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ArchiveState {
    active: ActiveState,
    archive_cap: usize,
    archives: Vec<ArchiveEntry>,
    last_nomination_receipt: NominationReceipt,
    next_creation_order: u64,
    nomination_mode: String,
    parent_hash: Option<String>,
    release_ids: Vec<String>,
    release_window: Vec<Release>,
    schema: String,
    stale_reuse_count: u64,
    version: u64,
}

impl ArchiveState {
    fn initial(archive_cap: usize) -> Self {
        Self {
            active: ActiveState::initial(),
            archive_cap,
            archives: Vec::new(),
            last_nomination_receipt: NominationReceipt::empty(),
            next_creation_order: 0,
            nomination_mode: "validated".to_string(),
            parent_hash: None,
            release_ids: Vec::new(),
            release_window: Vec::new(),
            schema: "carnot.archived_belief_controller.v1".to_string(),
            stale_reuse_count: 0,
            version: 0,
        }
    }

    fn validate(&self) -> Result<(), &'static str> {
        if self.schema != "carnot.archived_belief_controller.v1"
            || self.archive_cap > ARCHIVE_CAP
            || self.nomination_mode != "validated"
            || !valid_hash(&self.parent_hash)
        {
            return Err("invalid_archive_contract");
        }
        self.active.validate()?;
        if self.archives.len() > self.archive_cap || self.release_window.len() > VALIDATION_WINDOW {
            return Err("invalid_archives");
        }
        let mut ids = BTreeSet::new();
        if self.release_ids.iter().any(|item| !ids.insert(item)) {
            return Err("invalid_release_ids");
        }
        let mut orders = BTreeSet::new();
        for archive in &self.archives {
            if archive.survivor_masks.len() != FAMILY_NAMES.len()
                || !orders.insert(archive.creation_order)
            {
                return Err("invalid_archive_masks");
            }
            for name in FAMILY_NAMES {
                let mask = archive
                    .survivor_masks
                    .get(name)
                    .ok_or("invalid_archive_masks")?;
                if mask & !FULL_MASK != 0 {
                    return Err("invalid_archive_masks");
                }
            }
            if sha256_json(&archive.survivor_masks).map_err(|_| "invalid_archive_hash")?
                != archive.state_hash
            {
                return Err("invalid_archive_hash");
            }
        }
        for release in &self.release_window {
            release.validate(None)?;
        }
        Ok(())
    }

    fn state_hash(&self) -> Result<String, serde_json::Error> {
        sha256_json(self)
    }

    fn append_archive(&mut self) -> Result<Option<String>, serde_json::Error> {
        if self.archive_cap == 0 {
            return Ok(None);
        }
        let masks: BTreeMap<String, u64> = FAMILY_NAMES
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    self.active.families[*name].survivor_mask,
                )
            })
            .collect();
        let state_hash = sha256_json(&masks)?;
        if let Some(existing) = self
            .archives
            .iter()
            .find(|archive| archive.state_hash == state_hash)
        {
            return Ok(Some(existing.archive_id.clone()));
        }
        let archive_id = format!("archive-{:06}", self.next_creation_order);
        self.archives.push(ArchiveEntry {
            archive_id: archive_id.clone(),
            creation_order: self.next_creation_order,
            state_hash,
            survivor_masks: masks,
        });
        self.next_creation_order += 1;
        while self.archives.len() > self.archive_cap {
            let oldest = self
                .archives
                .iter()
                .enumerate()
                .min_by_key(|(_, archive)| archive.creation_order)
                .map(|(index, _)| index)
                .expect("nonempty archive");
            self.archives.remove(oldest);
        }
        Ok(Some(archive_id))
    }

    fn nominate(&mut self) -> Option<ArchiveEntry> {
        let mut order: Vec<usize> = (0..self.archives.len()).collect();
        order.sort_by_key(|index| self.archives[*index].creation_order);
        let mut candidates = Vec::new();
        for (nomination_order, archive_index) in order.iter().enumerate() {
            let archive = &self.archives[*archive_index];
            let mut applicable = 0;
            let mut contradictions = 0;
            for witness in &self.release_window {
                let family = family_index(&witness.family_id).expect("validated family");
                let mask = archive.survivor_masks[&witness.family_id];
                if mask == 0 {
                    continue;
                }
                applicable += 1;
                let value = witness.numeric_value as usize;
                let accepted = (mask & accept_mask(family, value)).count_ones();
                let prediction = if accepted * 2 > mask.count_ones() {
                    "accept"
                } else {
                    "reject"
                };
                contradictions += usize::from(prediction != witness.observed_label);
            }
            candidates.push(CandidateReceipt {
                archive_id: archive.archive_id.clone(),
                creation_order: archive.creation_order,
                nomination_order,
                released_witness_count: applicable,
                validation_loss: contradictions,
                contradiction_count: contradictions,
                gate_passed: applicable >= MIN_VALIDATION_WITNESSES && contradictions == 0,
            });
        }
        let selected_id = candidates
            .iter()
            .filter(|row| row.gate_passed)
            .min_by_key(|row| (row.validation_loss, row.creation_order))
            .map(|row| row.archive_id.clone());
        self.last_nomination_receipt = NominationReceipt {
            candidate_count: candidates.len(),
            validation_window_size: self.release_window.len(),
            minimum_witnesses: MIN_VALIDATION_WITNESSES,
            final_gate: "at_least_8_released_witnesses_and_zero_contradictions".to_string(),
            selected_archive_id: selected_id.clone(),
            candidates,
        };
        selected_id.and_then(|wanted| {
            self.archives
                .iter()
                .position(|archive| archive.archive_id == wanted)
                .map(|index| self.archives.remove(index))
        })
    }
}

#[derive(Clone, Debug, Serialize)]
struct CommitOperation {
    event_id: String,
    active_hash_before: String,
    active_hash_after: String,
    active_contradiction: bool,
    archived_before_reset: bool,
    archived_state_id: Option<String>,
    reactivated_archive_id: Option<String>,
    prediction_frozen_before_release: bool,
    same_event_correction: bool,
    nomination_receipt: NominationReceipt,
}

#[derive(Clone, Debug, Default, Serialize)]
struct ConversionCounts {
    active_reconstruction_count: u64,
    hot_path_json_parse_count: u64,
    snapshot_json_parse_count: u64,
    snapshot_serialize_count: u64,
    typed_event_calls: u64,
}

/// Native owner for the complete V637 FIFO archive policy and packed state.
#[pyclass(name = "RustArchiveController7256")]
pub struct PyArchiveController7256 {
    state: ArchiveState,
    rollback_state: Option<ArchiveState>,
    counts: ConversionCounts,
}

#[pymethods]
impl PyArchiveController7256 {
    #[new]
    #[pyo3(signature = (archive_cap=4))]
    fn new(archive_cap: usize) -> PyResult<Self> {
        if archive_cap > ARCHIVE_CAP {
            return Err(value_error("invalid_archive_cap"));
        }
        Ok(Self {
            state: ArchiveState::initial(archive_cap),
            rollback_state: None,
            counts: ConversionCounts::default(),
        })
    }

    /// Parse only an explicit restore boundary and admit only checked state.
    fn load_snapshot(&mut self, serialized: &str) -> PyResult<()> {
        let candidate: ArchiveState = serde_json::from_str(serialized).map_err(value_error)?;
        candidate.validate().map_err(value_error)?;
        self.state = candidate;
        self.rollback_state = None;
        self.counts.snapshot_json_parse_count += 1;
        Ok(())
    }

    /// Serialize the complete controller only at an explicit snapshot boundary.
    fn snapshot_state(&mut self) -> PyResult<String> {
        self.counts.snapshot_serialize_count += 1;
        canonical_json(&self.state).map_err(value_error)
    }

    /// Return the canonical full-state hash without crossing through Python state.
    fn state_hash(&self) -> PyResult<String> {
        self.state.state_hash().map_err(value_error)
    }

    /// Evaluate one typed public event against the persistent active state.
    fn predict(&mut self, family: u8, raw_value: i64) -> PyResult<(String, f64)> {
        let family = family_name(family)
            .and_then(|name| family_index(name).ok_or("invalid_public_event"))
            .map_err(value_error)?;
        let value = raw_value.rem_euclid(DOMAIN_SIZE as i64) as usize;
        self.counts.typed_event_calls += 1;
        let (decision, disagreement) = self.state.active.prediction(family, value);
        Ok((
            match decision {
                1 => "accept",
                0 => "reject",
                _ => "abstain",
            }
            .to_string(),
            disagreement,
        ))
    }

    /// Compute typed disagreement energy without changing controller state.
    fn energy<'py>(
        &mut self,
        py: Python<'py>,
        label: i8,
        family: u8,
        raw_value: i64,
    ) -> PyResult<Bound<'py, PyDict>> {
        if label != 0 && label != 1 {
            return Err(value_error("invalid_label"));
        }
        let family = family_name(family)
            .and_then(|name| family_index(name).ok_or("invalid_public_event"))
            .map_err(value_error)?;
        let value = raw_value.rem_euclid(DOMAIN_SIZE as i64) as usize;
        self.counts.typed_event_calls += 1;
        let row = &self.state.active.families[FAMILY_NAMES[family]];
        let count = row.survivor_mask.count_ones();
        let result = PyDict::new(py);
        if count == 0 {
            result.set_item("status", "empty")?;
            result.set_item("value", None::<f64>)?;
            result.set_item("survivor_count", 0)?;
            result.set_item("disagree_count", None::<u32>)?;
            return Ok(result);
        }
        let accepted = row.vote_counts[value] as u32;
        let disagrees = if label == 1 {
            count - accepted
        } else {
            accepted
        };
        result.set_item("status", "known")?;
        result.set_item("value", disagrees as f64 / count as f64)?;
        result.set_item("survivor_count", count)?;
        result.set_item("disagree_count", disagrees)?;
        Ok(result)
    }

    /// Select one typed request by disagreement then the supplied stable rank.
    fn select_request(
        &mut self,
        families: PyReadonlyArray1<'_, u8>,
        values: PyReadonlyArray1<'_, i64>,
        tie_ranks: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<usize> {
        let families = families.as_slice()?;
        let values = values.as_slice()?;
        let tie_ranks = tie_ranks.as_slice()?;
        if families.is_empty()
            || families.len() != values.len()
            || families.len() != tie_ranks.len()
        {
            return Err(value_error("query batch lengths must match"));
        }
        let mut selected = 0;
        let mut selected_key = (f64::INFINITY, i64::MAX);
        for index in 0..families.len() {
            let family = family_name(families[index])
                .and_then(|name| family_index(name).ok_or("invalid_public_event"))
                .map_err(value_error)?;
            let value = values[index].rem_euclid(DOMAIN_SIZE as i64) as usize;
            let (_, disagreement) = self.state.active.prediction(family, value);
            let key = (-disagreement, tie_ranks[index]);
            if key < selected_key {
                selected = index;
                selected_key = key;
            }
        }
        self.counts.typed_event_calls += families.len() as u64;
        Ok(selected)
    }

    /// Apply an ordered typed release batch as one all-or-nothing transaction.
    #[allow(clippy::too_many_arguments)]
    fn commit_batch(
        &mut self,
        event_ids: Vec<String>,
        families: PyReadonlyArray1<'_, u8>,
        values: PyReadonlyArray1<'_, i64>,
        labels: PyReadonlyArray1<'_, i8>,
        roles: PyReadonlyArray1<'_, u8>,
        request_indices: PyReadonlyArray1<'_, i64>,
        release_indices: PyReadonlyArray1<'_, i64>,
        current_cycle: i64,
        expected_parent_hash: &str,
    ) -> PyResult<String> {
        let families = families.as_slice()?;
        let values = values.as_slice()?;
        let labels = labels.as_slice()?;
        let roles = roles.as_slice()?;
        let request_indices = request_indices.as_slice()?;
        let release_indices = release_indices.as_slice()?;
        let length = event_ids.len();
        if [
            families.len(),
            values.len(),
            labels.len(),
            roles.len(),
            request_indices.len(),
            release_indices.len(),
        ]
        .iter()
        .any(|candidate| *candidate != length)
        {
            return Err(value_error("update batch lengths must match"));
        }
        let parent_hash = self.state.state_hash().map_err(value_error)?;
        if expected_parent_hash != parent_hash {
            return Err(value_error("stale_parent"));
        }
        self.state.validate().map_err(value_error)?;
        let mut seen = BTreeSet::new();
        let existing: BTreeSet<&str> = self.state.release_ids.iter().map(String::as_str).collect();
        let mut releases = Vec::with_capacity(length);
        for index in 0..length {
            let family_id = family_name(families[index])
                .map_err(value_error)?
                .to_string();
            let release = Release {
                event_id: event_ids[index].clone(),
                family_id,
                numeric_value: values[index].rem_euclid(DOMAIN_SIZE as i64),
                observed_label: match labels[index] {
                    1 => "accept",
                    0 => "reject",
                    _ => return Err(value_error("invalid_label")),
                }
                .to_string(),
                role: match roles[index] {
                    1 => "support",
                    0 => "validation",
                    _ => return Err(value_error("invalid_role")),
                }
                .to_string(),
                request_index: request_indices[index],
                release_index: release_indices[index],
            };
            release.validate(Some(current_cycle)).map_err(value_error)?;
            if !seen.insert(release.event_id.clone())
                || existing.contains(release.event_id.as_str())
            {
                return Err(value_error("duplicate_release"));
            }
            releases.push(release);
        }

        let parent = self.state.clone();
        let parent_json = canonical_json(&parent).map_err(value_error)?;
        let mut candidate = parent.clone();
        let mut operations = Vec::with_capacity(length);
        for release in &releases {
            let before_hash = candidate.active.state_hash().map_err(value_error)?;
            let family = family_index(&release.family_id).expect("validated family");
            let mask = candidate.active.families[&release.family_id].survivor_mask;
            let accepted = accept_mask(family, release.numeric_value as usize);
            let matching = if release.observed_label == "accept" {
                accepted
            } else {
                FULL_MASK ^ accepted
            };
            let contradiction = mask != 0 && mask & matching == 0;
            let archived_id = if contradiction {
                candidate.append_archive().map_err(value_error)?
            } else {
                None
            };
            candidate.release_window.push(release.clone());
            if candidate.release_window.len() > VALIDATION_WINDOW {
                candidate.release_window.remove(0);
            }
            candidate
                .active
                .apply_release(release)
                .map_err(value_error)?;
            let reactivated = candidate.nominate();
            let reactivated_id = reactivated
                .as_ref()
                .map(|archive| archive.archive_id.clone());
            if let Some(archive) = reactivated {
                candidate.active = ActiveState::initial_from_masks(&archive.survivor_masks);
            }
            candidate.release_ids.push(release.event_id.clone());
            let after_hash = candidate.active.state_hash().map_err(value_error)?;
            operations.push(CommitOperation {
                event_id: release.event_id.clone(),
                active_hash_before: before_hash,
                active_hash_after: after_hash,
                active_contradiction: contradiction,
                archived_before_reset: contradiction && archived_id.is_some(),
                archived_state_id: archived_id,
                reactivated_archive_id: reactivated_id,
                prediction_frozen_before_release: true,
                same_event_correction: false,
                nomination_receipt: candidate.last_nomination_receipt.clone(),
            });
        }
        candidate.version += 1;
        candidate.parent_hash = Some(parent_hash.clone());
        candidate.validate().map_err(value_error)?;
        let new_hash = candidate.state_hash().map_err(value_error)?;
        let new_json = canonical_json(&candidate).map_err(value_error)?;
        self.rollback_state = Some(parent);
        self.state = candidate;
        self.counts.typed_event_calls += length as u64;
        serde_json::to_string(&json!({
            "parent_hash": parent_hash,
            "new_state_hash": new_hash,
            "parent_json": parent_json,
            "new_json": new_json,
            "state_version": self.state.version,
            "release_count": length,
            "release_order": event_ids,
            "operations": operations,
        }))
        .map_err(value_error)
    }

    /// Restore the exact parent retained by the latest admitted transaction.
    fn rollback(&mut self, expected_child_hash: &str, expected_parent_hash: &str) -> PyResult<()> {
        if self.state.state_hash().map_err(value_error)? != expected_child_hash {
            return Err(value_error("stale_rollback"));
        }
        let parent = self
            .rollback_state
            .take()
            .ok_or_else(|| value_error("stale_rollback"))?;
        if parent.state_hash().map_err(value_error)? != expected_parent_hash {
            self.rollback_state = Some(parent);
            return Err(value_error("rollback_parent_hash"));
        }
        self.state = parent;
        Ok(())
    }

    /// Expose counters that prove typed hot-path calls did not restore snapshots.
    fn conversion_counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let value = serde_json::to_value(&self.counts).map_err(value_error)?;
        let result = PyDict::new(py);
        if let Value::Object(fields) = value {
            for (name, count) in fields {
                result.set_item(name, count.as_u64().unwrap_or_default())?;
            }
        }
        Ok(result)
    }
}

pub fn register_experiment_7256_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyArchiveController7256>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-RUSTPY-7256: canonical native state matches the Python schema shape.
    #[test]
    fn initial_state_is_valid_and_stable() {
        let state = ArchiveState::initial(4);
        assert!(state.validate().is_ok());
        assert_eq!(state.archives.len(), 0);
        assert_eq!(state.active.families["lower_bound"].vote_counts[32], 33);
        assert_eq!(state.state_hash().expect("hash").len(), 71);
    }
}
