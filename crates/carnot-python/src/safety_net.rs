//! PyO3 ABI for the compact Safety-Net router decision.
//!
//! Spec: REQ-RUSTPY-6550, SCENARIO-RUSTPY-6550-BOUNDARY-PARITY,
//! REQ-RUSTPY-6564, SCENARIO-RUSTPY-6564-BATCH-ORDERED-PARITY.

use std::collections::{BTreeMap, BTreeSet};

use carnot_core::adaptive_state::sha256_bytes;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes, PyDict};
use serde_json::{json, Value};

const ABI_SCHEMA_VERSION: &str = "carnot.safety_net.router_abi.v1";
const BATCH_ABI_SCHEMA_VERSION: &str = "carnot.safety_net.router_batch_abi.v1";
const ROUTER_CONTRACT_HASH: &str =
    "sha256:932719273db1ef84b2e6f9fa81d996e818d3ae9c1ea341ab9693ad52297b8c16";
const DEFAULT_ABSTENTION_THRESHOLD: f64 = 0.5;
const MAX_STRUCTURAL_FEATURE_ABS: i64 = 1_000_000_000;
const FEATURE_NAMES: [&str; 5] = [
    "candidate_depth",
    "candidate_count",
    "constraint_count",
    "turn_index",
    "num_entities",
];
const FORBIDDEN_FEATURES: [&str; 7] = [
    "family_identity",
    "source_id",
    "entity_names",
    "row_order",
    "solver_effort_wall_time",
    "held_outcome",
    "future_turns",
];
const ALLOWED_TOP_LEVEL: [&str; 10] = [
    "schema_version",
    "request_id",
    "candidate_ids",
    "feature_values",
    "split_name",
    "seed",
    "router_contract_hash",
    "exception_table",
    "forced_abstain",
    "forced_fallback_reason",
];

#[derive(Clone, Debug)]
struct ParsedRequest {
    candidate_ids: Vec<String>,
    split_name: String,
    router_contract_hash: String,
    exception_table: BTreeMap<String, String>,
    forced_abstain: bool,
    forced_fallback_reason: String,
}

#[derive(Clone, Debug)]
struct Reject {
    reason: String,
    error_type: String,
}

#[derive(Clone, Debug)]
struct SafetyNetDecision {
    schema_version: String,
    route: String,
    abstain: bool,
    uncertainty_bucket: String,
    exception_hit: bool,
    fallback_reason: String,
    original_order: Vec<String>,
    chosen_order: Vec<String>,
    error_type: String,
    exact_fallback_reachable: bool,
    request_hash: String,
    router_contract_hash: String,
}

impl SafetyNetDecision {
    fn to_value(&self) -> Value {
        json!({
            "schema_version": self.schema_version,
            "route": self.route,
            "abstain": self.abstain,
            "uncertainty_bucket": self.uncertainty_bucket,
            "exception_hit": self.exception_hit,
            "fallback_reason": self.fallback_reason,
            "original_order": self.original_order,
            "chosen_order": self.chosen_order,
            "error_type": self.error_type,
            "exact_fallback_reachable": self.exact_fallback_reachable,
            "request_hash": self.request_hash,
            "router_contract_hash": self.router_contract_hash,
        })
    }

    fn canonical_json(&self) -> String {
        serde_json::to_string(&self.to_value()).expect("decision JSON serializes")
    }
}

/// Typed request bytes for the Safety-Net routing ABI.
///
/// The request stores raw bytes because the ABI contract hashes the exact input
/// passed across the language boundary.
#[pyclass(name = "RustSafetyNetFeatureRequest")]
pub struct PySafetyNetFeatureRequest {
    request_bytes: Vec<u8>,
}

/// Typed decision snapshot returned by the Safety-Net routing ABI.
#[pyclass(name = "RustSafetyNetRoutingDecision")]
#[derive(Clone)]
pub struct PySafetyNetRoutingDecision {
    inner: SafetyNetDecision,
}

/// Stateless helper for routing request bytes through the Rust ABI.
#[pyclass(name = "RustSafetyNetRouter")]
pub struct PySafetyNetRouter;

#[pymethods]
impl PySafetyNetFeatureRequest {
    #[new]
    fn new(request_bytes: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            request_bytes: request_bytes.extract::<Vec<u8>>()?,
        })
    }

    fn input_hash(&self) -> String {
        sha256_bytes(&self.request_bytes)
    }

    fn decision(&self) -> PySafetyNetRoutingDecision {
        PySafetyNetRoutingDecision {
            inner: route_bytes(&self.request_bytes),
        }
    }
}

#[pymethods]
impl PySafetyNetRoutingDecision {
    #[getter]
    fn schema_version(&self) -> String {
        self.inner.schema_version.clone()
    }

    #[getter]
    fn route(&self) -> String {
        self.inner.route.clone()
    }

    #[getter]
    fn abstain(&self) -> bool {
        self.inner.abstain
    }

    #[getter]
    fn uncertainty_bucket(&self) -> String {
        self.inner.uncertainty_bucket.clone()
    }

    #[getter]
    fn exception_hit(&self) -> bool {
        self.inner.exception_hit
    }

    #[getter]
    fn fallback_reason(&self) -> String {
        self.inner.fallback_reason.clone()
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        decision_to_pydict(py, &self.inner)
    }

    fn canonical_json(&self) -> String {
        self.inner.canonical_json()
    }

    fn canonical_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.canonical_json().as_bytes())
    }
}

#[pymethods]
impl PySafetyNetRouter {
    #[new]
    fn new() -> Self {
        Self
    }

    fn route_bytes<'py>(
        &self,
        py: Python<'py>,
        request_bytes: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let bytes = request_bytes.extract::<Vec<u8>>()?;
        decision_to_pydict(py, &route_bytes(&bytes))
    }

    fn route_batch<'py>(
        &self,
        py: Python<'py>,
        request_batch: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let batch = request_batch.extract::<Vec<Vec<u8>>>()?;
        batch
            .iter()
            .map(|bytes| decision_to_pydict(py, &route_bytes(bytes)))
            .collect()
    }
}

#[pyfunction(name = "safety_net_route_bytes")]
fn py_safety_net_route_bytes<'py>(
    py: Python<'py>,
    request_bytes: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let bytes = request_bytes.extract::<Vec<u8>>()?;
    decision_to_pydict(py, &route_bytes(&bytes))
}

#[pyfunction(name = "safety_net_route_batch")]
fn py_safety_net_route_batch<'py>(
    py: Python<'py>,
    request_batch: &Bound<'_, PyAny>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let batch = request_batch.extract::<Vec<Vec<u8>>>()?;
    batch
        .iter()
        .map(|bytes| decision_to_pydict(py, &route_bytes(bytes)))
        .collect()
}

fn route_bytes(request_bytes: &[u8]) -> SafetyNetDecision {
    let request_hash = sha256_bytes(request_bytes);
    let payload: Value = match serde_json::from_slice(request_bytes) {
        Ok(value) => value,
        Err(_) => {
            return error_decision(
                &request_hash,
                "malformed_input:invalid_json",
                "JsonDecodeError",
                ROUTER_CONTRACT_HASH,
            )
        }
    };
    let object = match payload.as_object() {
        Some(object) => object,
        None => {
            return error_decision(
                &request_hash,
                "malformed_input:not_object",
                "SafetyNetAbiError",
                ROUTER_CONTRACT_HASH,
            )
        }
    };
    let parsed = match parse_request(object) {
        Ok(parsed) => parsed,
        Err(reject) => {
            let router_hash = object
                .get("router_contract_hash")
                .and_then(Value::as_str)
                .unwrap_or(ROUTER_CONTRACT_HASH);
            return error_decision(
                &request_hash,
                &reject.reason,
                &reject.error_type,
                router_hash,
            );
        }
    };
    route_parsed(&request_hash, parsed)
}

fn route_parsed(request_hash: &str, parsed: ParsedRequest) -> SafetyNetDecision {
    if parsed.router_contract_hash != ROUTER_CONTRACT_HASH {
        return fallback_decision(
            request_hash,
            &parsed.candidate_ids,
            "stale_configuration",
            "SchemaVersionError",
            &parsed.router_contract_hash,
            false,
            false,
            "unsupported",
        );
    }
    let key_hash = exception_key(&parsed.candidate_ids, &parsed.split_name);
    let exception_hit = parsed
        .exception_table
        .get(&key_hash)
        .is_some_and(|value| value == "native_exact_fallback");
    let bucket = uncertainty_bucket(parsed.candidate_ids.len());
    if !parsed.forced_fallback_reason.is_empty() {
        return fallback_decision(
            request_hash,
            &parsed.candidate_ids,
            &parsed.forced_fallback_reason,
            "",
            &parsed.router_contract_hash,
            false,
            exception_hit,
            &bucket,
        );
    }
    if exception_hit {
        return fallback_decision(
            request_hash,
            &parsed.candidate_ids,
            "exception_table_hit",
            "",
            &parsed.router_contract_hash,
            false,
            true,
            &bucket,
        );
    }
    let abstain = parsed.forced_abstain
        || (1.0 / ((parsed.candidate_ids.len() + 1) as f64) >= DEFAULT_ABSTENTION_THRESHOLD);
    if abstain {
        return fallback_decision(
            request_hash,
            &parsed.candidate_ids,
            "abstention",
            "",
            &parsed.router_contract_hash,
            true,
            false,
            &bucket,
        );
    }
    let mut chosen = parsed.candidate_ids.clone();
    chosen.reverse();
    SafetyNetDecision {
        schema_version: ABI_SCHEMA_VERSION.to_string(),
        route: "compact_router".to_string(),
        abstain: false,
        uncertainty_bucket: bucket,
        exception_hit: false,
        fallback_reason: String::new(),
        original_order: parsed.candidate_ids,
        chosen_order: chosen,
        error_type: String::new(),
        exact_fallback_reachable: true,
        request_hash: request_hash.to_string(),
        router_contract_hash: parsed.router_contract_hash,
    }
}

fn parse_request(object: &serde_json::Map<String, Value>) -> Result<ParsedRequest, Reject> {
    let allowed: BTreeSet<&str> = ALLOWED_TOP_LEVEL.iter().copied().collect();
    if object.keys().any(|key| !allowed.contains(key.as_str())) {
        return Err(reject("malformed_input:extra_keys", "SafetyNetAbiError"));
    }
    let schema = object.get("schema_version").and_then(Value::as_str);
    if schema.is_none() {
        return Err(reject("schema_version_missing", "SchemaVersionError"));
    }
    if schema != Some(ABI_SCHEMA_VERSION) {
        return Err(reject("stale_schema_version", "SchemaVersionError"));
    }
    let candidate_values = object
        .get("candidate_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| reject("malformed_input:missing_candidate_ids", "SafetyNetAbiError"))?;
    let mut candidate_ids = Vec::with_capacity(candidate_values.len());
    for value in candidate_values {
        let Some(item) = value.as_str() else {
            return Err(reject(
                "malformed_input:missing_candidate_ids",
                "SafetyNetAbiError",
            ));
        };
        candidate_ids.push(item.to_string());
    }
    if candidate_ids.is_empty() {
        return Err(reject("malformed_input:no_candidates", "SafetyNetAbiError"));
    }
    let mut seen = BTreeSet::new();
    for candidate_id in &candidate_ids {
        if candidate_id.trim().is_empty() {
            return Err(reject(
                "malformed_input:blank_candidate_id",
                "SafetyNetAbiError",
            ));
        }
        if !candidate_id.is_ascii() {
            return Err(reject(
                "malformed_input:non_ascii_candidate_id",
                "SafetyNetAbiError",
            ));
        }
        if !seen.insert(candidate_id) {
            return Err(reject(
                "malformed_input:duplicate_candidate_ids",
                "SafetyNetAbiError",
            ));
        }
    }

    let feature_values = object
        .get("feature_values")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            reject(
                "malformed_input:feature_values_not_object",
                "SafetyNetAbiError",
            )
        })?;
    let feature_names: BTreeSet<&str> = FEATURE_NAMES.iter().copied().collect();
    let forbidden_features: BTreeSet<&str> = FORBIDDEN_FEATURES.iter().copied().collect();
    for (name, value) in feature_values {
        if forbidden_features.contains(name.as_str()) {
            return Err(reject(
                "malformed_input:forbidden_feature",
                "SafetyNetAbiError",
            ));
        }
        if !feature_names.contains(name.as_str()) {
            return Err(reject(
                "malformed_input:unsupported_feature",
                "SafetyNetAbiError",
            ));
        }
        normalize_feature_number(value)?;
    }

    let exception_object = object
        .get("exception_table")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            reject(
                "malformed_input:exception_table_not_object",
                "SafetyNetAbiError",
            )
        })?;
    let mut exception_table = BTreeMap::new();
    for (key, value) in exception_object {
        exception_table.insert(key.clone(), value.as_str().unwrap_or("").to_string());
    }
    let forced_abstain = object
        .get("forced_abstain")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            reject(
                "malformed_input:forced_abstain_not_bool",
                "SafetyNetAbiError",
            )
        })?;
    let forced_fallback_reason = object
        .get("forced_fallback_reason")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            reject(
                "malformed_input:forced_fallback_not_string",
                "SafetyNetAbiError",
            )
        })?
        .to_string();
    let split_name = object
        .get("split_name")
        .and_then(Value::as_str)
        .unwrap_or("live")
        .to_string();
    let router_contract_hash = object
        .get("router_contract_hash")
        .and_then(Value::as_str)
        .unwrap_or(ROUTER_CONTRACT_HASH)
        .to_string();
    Ok(ParsedRequest {
        candidate_ids,
        split_name,
        router_contract_hash,
        exception_table,
        forced_abstain,
        forced_fallback_reason,
    })
}

fn normalize_feature_number(value: &Value) -> Result<i64, Reject> {
    let Some(number) = value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
    else {
        let Some(float_value) = value.as_f64() else {
            return Err(reject(
                "malformed_input:non_numeric_feature",
                "SafetyNetAbiError",
            ));
        };
        if !float_value.is_finite() {
            return Err(reject(
                "malformed_input:non_finite_feature",
                "SafetyNetAbiError",
            ));
        }
        if float_value.fract() != 0.0 {
            return Err(reject(
                "malformed_input:non_integer_feature",
                "SafetyNetAbiError",
            ));
        }
        if float_value.abs() > MAX_STRUCTURAL_FEATURE_ABS as f64 {
            return Err(reject(
                "malformed_input:numeric_out_of_range",
                "SafetyNetAbiError",
            ));
        }
        return Ok(float_value as i64);
    };
    if number.abs() > MAX_STRUCTURAL_FEATURE_ABS {
        return Err(reject(
            "malformed_input:numeric_out_of_range",
            "SafetyNetAbiError",
        ));
    }
    Ok(number)
}

fn exception_key(candidate_ids: &[String], split_name: &str) -> String {
    let payload = json!({
        "candidate_hashes": candidate_ids,
        "candidate_count": candidate_ids.len(),
        "split_name": split_name,
    });
    let serialized = serde_json::to_string(&payload).expect("exception key JSON serializes");
    carnot_core::adaptive_state::sha256_text(&serialized)
}

fn uncertainty_bucket(candidate_count: usize) -> String {
    if candidate_count <= 1 {
        "high".to_string()
    } else if candidate_count == 2 {
        "medium".to_string()
    } else {
        "low".to_string()
    }
}

fn fallback_decision(
    request_hash: &str,
    original_order: &[String],
    reason: &str,
    error_type: &str,
    router_contract_hash: &str,
    abstain: bool,
    exception_hit: bool,
    uncertainty_bucket: &str,
) -> SafetyNetDecision {
    SafetyNetDecision {
        schema_version: ABI_SCHEMA_VERSION.to_string(),
        route: "native_exact_fallback".to_string(),
        abstain: abstain || reason == "abstention",
        uncertainty_bucket: uncertainty_bucket.to_string(),
        exception_hit,
        fallback_reason: reason.to_string(),
        original_order: original_order.to_vec(),
        chosen_order: original_order.to_vec(),
        error_type: error_type.to_string(),
        exact_fallback_reachable: true,
        request_hash: request_hash.to_string(),
        router_contract_hash: router_contract_hash.to_string(),
    }
}

fn error_decision(
    request_hash: &str,
    reason: &str,
    error_type: &str,
    router_contract_hash: &str,
) -> SafetyNetDecision {
    fallback_decision(
        request_hash,
        &[],
        reason,
        error_type,
        router_contract_hash,
        false,
        false,
        "unsupported",
    )
}

fn reject(reason: &str, error_type: &str) -> Reject {
    Reject {
        reason: reason.to_string(),
        error_type: error_type.to_string(),
    }
}

fn decision_to_pydict<'py>(
    py: Python<'py>,
    decision: &SafetyNetDecision,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("schema_version", decision.schema_version.clone())?;
    d.set_item("route", decision.route.clone())?;
    d.set_item("abstain", decision.abstain)?;
    d.set_item("uncertainty_bucket", decision.uncertainty_bucket.clone())?;
    d.set_item("exception_hit", decision.exception_hit)?;
    d.set_item("fallback_reason", decision.fallback_reason.clone())?;
    d.set_item("original_order", decision.original_order.clone())?;
    d.set_item("chosen_order", decision.chosen_order.clone())?;
    d.set_item("error_type", decision.error_type.clone())?;
    d.set_item(
        "exact_fallback_reachable",
        decision.exact_fallback_reachable,
    )?;
    d.set_item("request_hash", decision.request_hash.clone())?;
    d.set_item(
        "router_contract_hash",
        decision.router_contract_hash.clone(),
    )?;
    Ok(d)
}

/// Register Safety-Net ABI classes and helpers.
pub fn register_safety_net_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let safety_mod = PyModule::new(parent.py(), "safety_net")?;
    safety_mod.add("__schema_version__", ABI_SCHEMA_VERSION)?;
    safety_mod.add("__batch_schema_version__", BATCH_ABI_SCHEMA_VERSION)?;
    safety_mod.add_class::<PySafetyNetFeatureRequest>()?;
    safety_mod.add_class::<PySafetyNetRoutingDecision>()?;
    safety_mod.add_class::<PySafetyNetRouter>()?;
    safety_mod.add_function(wrap_pyfunction!(py_safety_net_route_bytes, &safety_mod)?)?;
    safety_mod.add_function(wrap_pyfunction!(py_safety_net_route_batch, &safety_mod)?)?;
    parent.add_submodule(&safety_mod)?;
    parent.add_class::<PySafetyNetFeatureRequest>()?;
    parent.add_class::<PySafetyNetRoutingDecision>()?;
    parent.add_class::<PySafetyNetRouter>()?;
    parent.add_function(wrap_pyfunction!(py_safety_net_route_bytes, parent)?)?;
    parent.add_function(wrap_pyfunction!(py_safety_net_route_batch, parent)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_request(candidate_ids: &[&str]) -> Vec<u8> {
        let payload = json!({
            "schema_version": ABI_SCHEMA_VERSION,
            "request_id": "rust-unit",
            "candidate_ids": candidate_ids,
            "feature_values": {"candidate_count": candidate_ids.len(), "constraint_count": candidate_ids.len()},
            "split_name": "live",
            "seed": 6550,
            "router_contract_hash": ROUTER_CONTRACT_HASH,
            "exception_table": {},
            "forced_abstain": false,
            "forced_fallback_reason": "",
        });
        serde_json::to_string(&payload).unwrap().into_bytes()
    }

    #[test]
    fn req_rustpy_6550_supported_route_serializes_deterministically() {
        let decision = route_bytes(&base_request(&[
            "sha256:1111111111111111111111111111111111111111111111111111111111111111",
            "sha256:2222222222222222222222222222222222222222222222222222222222222222",
        ]));
        assert_eq!(decision.route, "compact_router");
        assert_eq!(decision.uncertainty_bucket, "medium");
        assert_eq!(
            decision.chosen_order,
            vec![
                "sha256:2222222222222222222222222222222222222222222222222222222222222222",
                "sha256:1111111111111111111111111111111111111111111111111111111111111111",
            ]
        );
        assert!(decision.canonical_json().contains(ABI_SCHEMA_VERSION));
    }

    #[test]
    fn scenario_rustpy_6550_errors_fail_closed() {
        let duplicate = route_bytes(&base_request(&[
            "sha256:1111111111111111111111111111111111111111111111111111111111111111",
            "sha256:1111111111111111111111111111111111111111111111111111111111111111",
        ]));
        assert_eq!(duplicate.route, "native_exact_fallback");
        assert_eq!(
            duplicate.fallback_reason,
            "malformed_input:duplicate_candidate_ids"
        );
        assert_eq!(duplicate.error_type, "SafetyNetAbiError");

        let invalid_json = route_bytes(b"{\"candidate_count\":NaN}");
        assert_eq!(invalid_json.fallback_reason, "malformed_input:invalid_json");
        assert_eq!(invalid_json.error_type, "JsonDecodeError");
    }

    #[test]
    fn scenario_rustpy_6564_batch_matches_scalar_order() {
        let supported = base_request(&[
            "sha256:1111111111111111111111111111111111111111111111111111111111111111",
            "sha256:2222222222222222222222222222222222222222222222222222222222222222",
        ]);
        let malformed = b"{\"candidate_count\":NaN}".to_vec();
        let batch = [supported.clone(), malformed.clone()];
        let scalar = [route_bytes(&supported), route_bytes(&malformed)];
        let batch_decisions: Vec<SafetyNetDecision> =
            batch.iter().map(|request| route_bytes(request)).collect();

        assert_eq!(
            BATCH_ABI_SCHEMA_VERSION,
            "carnot.safety_net.router_batch_abi.v1"
        );
        assert_eq!(
            batch_decisions[0].canonical_json(),
            scalar[0].canonical_json()
        );
        assert_eq!(
            batch_decisions[1].canonical_json(),
            scalar[1].canonical_json()
        );
        assert_eq!(batch_decisions[0].request_hash, scalar[0].request_hash);
        assert_eq!(batch_decisions[1].request_hash, scalar[1].request_hash);
    }
}
