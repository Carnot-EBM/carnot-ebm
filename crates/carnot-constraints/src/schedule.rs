//! Exact integer schedule constraints used by the Exp7326 parity fixture.
//!
//! Pairwise separation and sliding-window capacity are deliberately kept apart
//! from floating-point `ConstraintTerm` implementations. Their evidence is
//! integer-valued, so this API preserves bit-exact energies across languages.
//!
//! Spec: REQ-VERIFY-7326 and SCENARIO-VERIFY-7326-*.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// The only accepted serialized request schema.
pub const SCHEDULE_REQUEST_SCHEMA: &str = "carnot.constraint_kernel.request.v1";

/// One named activity assigned to one integer slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleAssignment {
    /// Stable activity name used by separation terms.
    pub activity: String,
    /// Selected integer slot.
    pub slot: i64,
}

/// One serialized term. Fields not used by its `kind` stay absent.
///
/// A flat record keeps malformed or unknown kinds representable so evaluation
/// can return a deterministic fail-closed result instead of failing JSON parse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleConstraint {
    /// `pairwise_separation` or `sliding_window_capacity`.
    pub kind: String,
    /// Hash-bound atom identifier or synthetic fixture identifier.
    pub constraint_id: String,
    /// Executor version whose authority admitted this term.
    #[serde(default)]
    pub version: Option<String>,
    /// Left activity for pairwise separation.
    #[serde(default)]
    pub left: Option<String>,
    /// Right activity for pairwise separation.
    #[serde(default)]
    pub right: Option<String>,
    /// Inclusive minimum distance for pairwise separation.
    #[serde(default)]
    pub minimum: Option<i64>,
    /// Width of each half-open capacity window.
    #[serde(default)]
    pub window_size: Option<i64>,
    /// Inclusive maximum occupancy for each capacity window.
    #[serde(default)]
    pub maximum: Option<i64>,
}

/// One complete request for acquired-constraint evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleRequest {
    /// Versioned request schema.
    pub schema: String,
    /// Currently authenticated executor version.
    pub executor_version: String,
    /// Smallest allowed slot, inclusive.
    pub slot_min: i64,
    /// Largest allowed slot, inclusive.
    pub slot_max: i64,
    /// Fully supplied named schedule. Empty schedules fail closed.
    pub schedule: Vec<ScheduleAssignment>,
    /// Ordered acquired terms. Output energies preserve this order.
    pub constraints: Vec<ScheduleConstraint>,
}

/// Exact energy for one acquired term.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleTermEnergy {
    /// Input term identity.
    pub constraint_id: String,
    /// Input term kind.
    pub kind: String,
    /// Nonnegative integer violation energy.
    pub energy: u64,
    /// True exactly when this term has zero energy.
    pub satisfied: bool,
}

/// Fail-closed request result with an ordered energy decomposition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleEvaluation {
    /// True only after all request and term validation succeeds.
    pub valid_input: bool,
    /// True only for valid input whose total acquired energy is zero.
    pub feasible: bool,
    /// Always false: acquired terms are not a complete oracle language.
    pub complete_oracle_certificate: bool,
    /// Checked sum of all term energies, absent for invalid input.
    pub total_energy: Option<u64>,
    /// Ordered per-term decomposition, empty for invalid input.
    pub terms: Vec<ScheduleTermEnergy>,
    /// Stable first validation error, absent for valid input.
    pub error: Option<String>,
}

fn invalid(error: impl Into<String>) -> ScheduleEvaluation {
    ScheduleEvaluation {
        valid_input: false,
        feasible: false,
        complete_oracle_certificate: false,
        total_energy: None,
        terms: Vec::new(),
        error: Some(error.into()),
    }
}

fn required_text<'a>(
    value: &'a Option<String>,
    field: &str,
    constraint_id: &str,
) -> Result<&'a str, ScheduleEvaluation> {
    value
        .as_deref()
        .filter(|item| !item.is_empty())
        .ok_or_else(|| invalid(format!("missing_text:{constraint_id}:{field}")))
}

fn required_integer(
    value: Option<i64>,
    field: &str,
    constraint_id: &str,
) -> Result<i64, ScheduleEvaluation> {
    value.ok_or_else(|| invalid(format!("missing_integer:{constraint_id}:{field}")))
}

fn checked_square(value: u128, constraint_id: &str) -> Result<u64, ScheduleEvaluation> {
    value
        .checked_mul(value)
        .and_then(|energy| u64::try_from(energy).ok())
        .ok_or_else(|| invalid(format!("energy_overflow:{constraint_id}")))
}

fn separation_energy(
    term: &ScheduleConstraint,
    assignments: &HashMap<&str, i64>,
) -> Result<u64, ScheduleEvaluation> {
    let left = required_text(&term.left, "left", &term.constraint_id)?;
    let right = required_text(&term.right, "right", &term.constraint_id)?;
    if left == right {
        return Err(invalid(format!("identical_pair:{}", term.constraint_id)));
    }
    let minimum = required_integer(term.minimum, "minimum", &term.constraint_id)?;
    if minimum < 0 {
        return Err(invalid(format!("negative_minimum:{}", term.constraint_id)));
    }
    let left_slot = assignments
        .get(left)
        .ok_or_else(|| invalid(format!("missing_activity:{}:{left}", term.constraint_id)))?;
    let right_slot = assignments
        .get(right)
        .ok_or_else(|| invalid(format!("missing_activity:{}:{right}", term.constraint_id)))?;
    let distance = (i128::from(*left_slot) - i128::from(*right_slot)).unsigned_abs();
    let deficit = (minimum as u128).saturating_sub(distance);
    checked_square(deficit, &term.constraint_id)
}

fn capacity_energy(
    term: &ScheduleConstraint,
    schedule: &[ScheduleAssignment],
    slot_min: i64,
    slot_max: i64,
) -> Result<u64, ScheduleEvaluation> {
    let window_size = required_integer(term.window_size, "window_size", &term.constraint_id)?;
    if window_size <= 0 {
        return Err(invalid(format!(
            "nonpositive_window:{}",
            term.constraint_id
        )));
    }
    let maximum = required_integer(term.maximum, "maximum", &term.constraint_id)?;
    if maximum < 0 {
        return Err(invalid(format!("negative_maximum:{}", term.constraint_id)));
    }
    let domain_width = i128::from(slot_max) - i128::from(slot_min) + 1;
    if i128::from(window_size) > domain_width {
        return Err(invalid(format!(
            "window_exceeds_domain:{}",
            term.constraint_id
        )));
    }
    let domain_end = slot_max
        .checked_add(1)
        .ok_or_else(|| invalid(format!("window_end_overflow:{}", term.constraint_id)))?;
    let last_start = i128::from(domain_end) - i128::from(window_size);
    let mut start = i128::from(slot_min);
    let mut total = 0_u64;
    while start <= last_start {
        let end = start + i128::from(window_size);
        let occupancy = schedule
            .iter()
            .filter(|assignment| {
                let slot = i128::from(assignment.slot);
                slot >= start && slot < end
            })
            .count() as u128;
        let excess = occupancy.saturating_sub(maximum as u128);
        let energy = checked_square(excess, &term.constraint_id)?;
        total = total
            .checked_add(energy)
            .ok_or_else(|| invalid(format!("energy_overflow:{}", term.constraint_id)))?;
        start += 1;
    }
    Ok(total)
}

/// Evaluate one integer schedule without consulting a hidden executor.
///
/// The result is an acquired-language decision. Even a valid zero-energy result
/// keeps `complete_oracle_certificate=false` because acquisition can omit rules.
pub fn evaluate_schedule(request: &ScheduleRequest) -> ScheduleEvaluation {
    if request.schema != SCHEDULE_REQUEST_SCHEMA {
        return invalid("invalid_schema");
    }
    if request.executor_version.is_empty() {
        return invalid("empty_executor_version");
    }
    if request.slot_min > request.slot_max {
        return invalid("invalid_slot_domain");
    }
    if request.schedule.is_empty() {
        return invalid("empty_schedule");
    }

    let mut names = HashSet::new();
    let mut assignments = HashMap::new();
    for assignment in &request.schedule {
        if assignment.activity.is_empty() {
            return invalid("empty_activity");
        }
        if !names.insert(assignment.activity.as_str()) {
            return invalid(format!("duplicate_activity:{}", assignment.activity));
        }
        if assignment.slot < request.slot_min || assignment.slot > request.slot_max {
            return invalid(format!("slot_out_of_domain:{}", assignment.activity));
        }
        assignments.insert(assignment.activity.as_str(), assignment.slot);
    }

    let mut terms = Vec::with_capacity(request.constraints.len());
    let mut total = 0_u64;
    for term in &request.constraints {
        if term.constraint_id.is_empty() {
            return invalid("empty_constraint_id");
        }
        if term.kind != "pairwise_separation" && term.kind != "sliding_window_capacity" {
            return invalid(format!("unknown_constraint_kind:{}", term.constraint_id));
        }
        if term.version.as_deref() != Some(request.executor_version.as_str()) {
            return invalid(format!("version_mismatch:{}", term.constraint_id));
        }
        let energy = match term.kind.as_str() {
            "pairwise_separation" => match separation_energy(term, &assignments) {
                Ok(value) => value,
                Err(result) => return result,
            },
            "sliding_window_capacity" => {
                match capacity_energy(term, &request.schedule, request.slot_min, request.slot_max) {
                    Ok(value) => value,
                    Err(result) => return result,
                }
            }
            _ => unreachable!("term kind was validated above"),
        };
        total = match total.checked_add(energy) {
            Some(value) => value,
            None => return invalid(format!("total_energy_overflow:{}", term.constraint_id)),
        };
        terms.push(ScheduleTermEnergy {
            constraint_id: term.constraint_id.clone(),
            kind: term.kind.clone(),
            energy,
            satisfied: energy == 0,
        });
    }

    ScheduleEvaluation {
        valid_input: true,
        feasible: total == 0,
        complete_oracle_certificate: false,
        total_energy: Some(total),
        terms,
        error: None,
    }
}

/// Evaluate an ordered batch while preserving request order.
pub fn evaluate_schedule_batch(requests: &[ScheduleRequest]) -> Vec<ScheduleEvaluation> {
    requests.iter().map(evaluate_schedule).collect()
}
