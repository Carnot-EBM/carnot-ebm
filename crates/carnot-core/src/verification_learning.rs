use std::collections::HashMap;

/// A constraint for the Verification Learning proxy.
#[derive(Debug, Clone)]
pub struct VlConstraint {
    pub c_type: Option<String>,
    pub c_value: String,
}

/// Verification Learning (VL) proxy for continuous self-learning.
/// Implements a constraint-based loss function natively.
#[derive(Debug, Clone, Default)]
pub struct VerificationLearningProxy {
    pub constraints: Vec<VlConstraint>,
}

/// Unlabelled data item.
#[derive(Debug, Clone)]
pub struct UnlabelledData {
    pub id: String,
    pub text: String,
}

impl VerificationLearningProxy {
    pub fn new(constraints: Vec<VlConstraint>) -> Self {
        Self { constraints }
    }

    pub fn score_constraint_satisfaction(&self, unlabelled_data: &[UnlabelledData]) -> HashMap<String, f64> {
        let mut scores = HashMap::new();

        for item in unlabelled_data {
            if self.constraints.is_empty() {
                scores.insert(item.id.clone(), 1.0);
                continue;
            }

            let mut satisfied_count = 0;
            for constraint in &self.constraints {
                let c_type = constraint.c_type.as_deref().unwrap_or("");
                let c_value = &constraint.c_value;

                if c_type == "must_contain" {
                    if item.text.contains(c_value) {
                        satisfied_count += 1;
                    }
                } else if c_type == "must_not_contain" {
                    if !item.text.contains(c_value) {
                        satisfied_count += 1;
                    }
                } else {
                    // Unknown constraint type, assume satisfied for the proxy
                    satisfied_count += 1;
                }
            }

            let score = satisfied_count as f64 / self.constraints.len() as f64;
            scores.insert(item.id.clone(), score);
        }

        scores
    }

    pub fn compute_proxy_loss(&self, unlabelled_data: &[UnlabelledData]) -> f64 {
        if unlabelled_data.is_empty() {
            return 0.0;
        }

        let scores = self.score_constraint_satisfaction(unlabelled_data);
        let sum: f64 = scores.values().sum();
        1.0 - (sum / scores.len() as f64)
    }
}
