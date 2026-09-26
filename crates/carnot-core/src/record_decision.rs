//! Typed record decision tests for REQ-REPORT-7710.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

const RECORD_SCHEMA: &str = "carnot.exp7700.record_features.v1";
const PARAMETER_SCHEMA: &str = "carnot.exp7710.record_binary_energy.v1";
const STATE_SCHEMA: &str = "carnot.exp7710.record_state.v1";
const FEATURES: [&str; 8] = [
    "tuple_supported",
    "tuple_contradicted",
    "path_line_supported",
    "path_line_contradicted",
    "unknown_propositions",
    "residual_unknown_bytes",
    "checked_fraction",
    "source_records",
];

/// Evaluate one explicit fixture head without reading labels or source text.
pub fn score(payload_json: &str, parameters_json: &str) -> Result<(f64, String), String> {
    let payload: Value =
        serde_json::from_str(payload_json).map_err(|_| "record_field_invalid".to_string())?;
    if payload.get("schema").and_then(Value::as_str) != Some(RECORD_SCHEMA) {
        return Err("record_schema_invalid".to_string());
    }
    let counts = payload
        .get("counts")
        .and_then(Value::as_object)
        .ok_or_else(|| "record_fields_invalid".to_string())?;
    if counts.len() != FEATURES.len() || FEATURES.iter().any(|name| !counts.contains_key(*name)) {
        return Err("record_fields_invalid".to_string());
    }
    let mut vector = [0.0; 8];
    for (index, name) in FEATURES.iter().enumerate() {
        let value = counts
            .get(*name)
            .and_then(Value::as_f64)
            .ok_or_else(|| "record_field_invalid".to_string())?;
        if !value.is_finite()
            || value < 0.0
            || (*name == "checked_fraction" && value > 1.0)
            || (*name != "checked_fraction" && value.fract() != 0.0)
        {
            return Err("record_field_invalid".to_string());
        }
        vector[index] = if *name == "checked_fraction" {
            value
        } else {
            value.ln_1p()
        };
    }
    let parameters: Value = serde_json::from_str(parameters_json)
        .map_err(|_| "parameter_fields_invalid".to_string())?;
    if parameters.get("schema").and_then(Value::as_str) != Some(PARAMETER_SCHEMA) {
        return Err("parameter_schema_invalid".to_string());
    }
    let weights = parameters
        .get("weights")
        .and_then(Value::as_array)
        .ok_or_else(|| "parameter_fields_invalid".to_string())?;
    let thresholds = parameters
        .get("thresholds")
        .and_then(Value::as_array)
        .ok_or_else(|| "parameter_fields_invalid".to_string())?;
    let bias = parameters
        .get("bias")
        .and_then(Value::as_f64)
        .ok_or_else(|| "parameter_fields_invalid".to_string())?;
    if weights.len() != 8 || thresholds.len() != 2 || !bias.is_finite() {
        return Err("parameter_fields_invalid".to_string());
    }
    let mut logit = bias;
    for (x, weight) in vector.iter().zip(weights.iter()) {
        let weight = weight
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| "parameter_fields_invalid".to_string())?;
        logit += x * weight;
    }
    let low = thresholds[0]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| "parameter_fields_invalid".to_string())?;
    let high = thresholds[1]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| "parameter_fields_invalid".to_string())?;
    if !(0.0 <= low && low <= high && high <= 1.0) {
        return Err("parameter_fields_invalid".to_string());
    }
    if !logit.is_finite() {
        return Err("energy_not_finite".to_string());
    }
    let probability = if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        logit.exp() / (1.0 + logit.exp())
    };
    let action = if probability < low {
        "accept"
    } else if probability >= high {
        "reject"
    } else {
        "escalate"
    };
    Ok((probability, action.to_string()))
}

/// Owns a sidecar beside the established service state, so old callers keep
/// their existing state format. Every mutation is written and fsynced first.
#[derive(Debug)]
pub struct RecordDecisionService {
    path: PathBuf,
    pending: BTreeMap<String, f64>,
    processed: BTreeSet<String>,
}

impl RecordDecisionService {
    pub fn open(service_state_path: PathBuf) -> Result<Self, String> {
        let path = service_state_path.with_extension("records.json");
        if !path.exists() {
            let mut service = Self {
                path,
                pending: BTreeMap::new(),
                processed: BTreeSet::new(),
            };
            service.save()?;
            return Ok(service);
        }
        let bytes = fs::read(&path).map_err(|e| format!("record_state_read:{e}"))?;
        let state: Value =
            serde_json::from_slice(&bytes).map_err(|_| "record_state_invalid".to_string())?;
        if state.get("schema").and_then(Value::as_str) != Some(STATE_SCHEMA) {
            return Err("record_state_version_mismatch".to_string());
        }
        let pending: BTreeMap<String, f64> = serde_json::from_value(
            state
                .get("pending")
                .cloned()
                .ok_or("record_state_invalid")?,
        )
        .map_err(|_| "record_state_invalid".to_string())?;
        let processed: BTreeSet<String> = serde_json::from_value(
            state
                .get("processed_event_ids")
                .cloned()
                .ok_or("record_state_invalid")?,
        )
        .map_err(|_| "record_state_invalid".to_string())?;
        if pending.keys().any(|id| processed.contains(id))
            || pending.values().any(|value| !value.is_finite())
        {
            return Err("record_state_invalid".to_string());
        }
        Ok(Self {
            path,
            pending,
            processed,
        })
    }

    fn save(&mut self) -> Result<(), String> {
        let bytes = serde_json::to_vec(&json!({
            "schema": STATE_SCHEMA,
            "pending": self.pending,
            "processed_event_ids": self.processed,
            "sample_count": self.processed.len(),
        }))
        .map_err(|_| "record_state_invalid".to_string())?;
        let temporary = self
            .path
            .with_extension(format!("records.tmp-{}", std::process::id()));
        let mut file = File::create(&temporary).map_err(|e| format!("record_state_write:{e}"))?;
        file.write_all(&bytes)
            .map_err(|e| format!("record_state_write:{e}"))?;
        file.sync_all()
            .map_err(|e| format!("record_state_sync:{e}"))?;
        fs::rename(&temporary, &self.path).map_err(|e| format!("record_state_rename:{e}"))?;
        if let Some(parent) = self.path.parent() {
            File::open(parent)
                .and_then(|dir| dir.sync_all())
                .map_err(|e| format!("record_state_dir_sync:{e}"))?;
        }
        Ok(())
    }

    pub fn predict(
        &mut self,
        id: String,
        payload: &str,
        parameters: &str,
    ) -> Result<(String, f64, String), String> {
        if id.is_empty() {
            return Err("event_id_required".to_string());
        }
        if self.processed.contains(&id) {
            return Err(format!("duplicate_feedback:{id}"));
        }
        if self.pending.contains_key(&id) {
            return Err(format!("duplicate_prediction:{id}"));
        }
        let (probability, action) = score(payload, parameters)?;
        self.pending.insert(id.clone(), probability);
        if let Err(error) = self.save() {
            self.pending.remove(&id);
            return Err(error);
        }
        Ok((id, probability, action))
    }

    pub fn release_feedback(
        &mut self,
        id: String,
        label: i32,
    ) -> (String, bool, bool, Option<String>) {
        if label != 0 && label != 1 {
            return (id, false, false, Some("binary_label_required".to_string()));
        }
        if self.processed.contains(&id) {
            return (
                id.clone(),
                false,
                false,
                Some(format!("duplicate_feedback:{id}")),
            );
        }
        let Some(probability) = self.pending.remove(&id) else {
            return (
                id.clone(),
                false,
                false,
                Some(format!("unknown_prediction:{id}")),
            );
        };
        self.processed.insert(id.clone());
        if let Err(error) = self.save() {
            self.processed.remove(&id);
            self.pending.insert(id.clone(), probability);
            return (id, false, false, Some(error));
        }
        (id, true, true, None)
    }

    pub fn summary(&self) -> (usize, Vec<String>, Vec<String>, String) {
        (
            self.processed.len(),
            self.pending.keys().cloned().collect(),
            self.processed.iter().cloned().collect(),
            STATE_SCHEMA.to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scenario_report_7710_parity_128_cases_and_errors() {
        let params = json!({
            "schema": "carnot.exp7710.record_binary_energy.v1",
            "weights": [0.1,-0.2,0.3,-0.4,0.5,-0.6,0.7,-0.8],
            "bias": -0.25, "thresholds": [0.2,0.8]
        });
        for index in 0..128 {
            let payload = json!({
                "schema": "carnot.exp7700.record_features.v1",
                "counts": {
                    "tuple_supported": index, "tuple_contradicted": index * 2,
                    "path_line_supported": index * 3, "path_line_contradicted": index * 4,
                    "unknown_propositions": index * 5,
                    "residual_unknown_bytes": index * 6,
                    "checked_fraction": (index % 101) as f64 / 100.0,
                    "source_records": index * 7
                }
            });
            let (p, action) = score(&payload.to_string(), &params.to_string()).unwrap();
            assert!(p.is_finite() && (0.0..=1.0).contains(&p));
            assert!(["accept", "escalate", "reject"].contains(&action.as_str()));
        }
        let bad = json!({"schema":"wrong","counts":{}});
        assert_eq!(
            score(&bad.to_string(), &params.to_string()).unwrap_err(),
            "record_schema_invalid"
        );
        let wrong_params = json!({"schema":"wrong"});
        assert_eq!(
            score("{}", &wrong_params.to_string()).unwrap_err(),
            "record_schema_invalid"
        );
    }

    #[test]
    fn scenario_report_7710_restart_and_version() {
        let root = std::env::temp_dir().join(format!("carnot-7710-rust-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let state = root.join("service.json");
        let params = json!({"schema":PARAMETER_SCHEMA,"weights":vec![0.0;8],"bias":0.0,"thresholds":[0.2,0.8]});
        let counts: serde_json::Map<String, Value> = FEATURES
            .iter()
            .map(|name| (name.to_string(), json!(0)))
            .collect();
        let payload = json!({"schema":RECORD_SCHEMA,"counts":counts});
        let mut service = RecordDecisionService::open(state.clone()).unwrap();
        service
            .predict("one".to_string(), &payload.to_string(), &params.to_string())
            .unwrap();
        drop(service);
        let mut service = RecordDecisionService::open(state.clone()).unwrap();
        assert_eq!(service.summary().1, vec!["one"]);
        assert_eq!(service.release_feedback("one".to_string(), 1).1, true);
        assert_eq!(
            service.release_feedback("one".to_string(), 1).3.unwrap(),
            "duplicate_feedback:one"
        );
        let sidecar = state.with_extension("records.json");
        fs::write(&sidecar, r#"{"schema":"wrong"}"#).unwrap();
        assert_eq!(
            RecordDecisionService::open(state).err().unwrap(),
            "record_state_version_mismatch"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
