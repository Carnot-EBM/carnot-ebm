//! Model state serialization via safetensors.
//!
//! Spec: REQ-CORE-004

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array1;
use safetensors::tensor::{SafeTensors, TensorView};

use crate::{CarnotError, Float, ModelMetadata};

/// Save parameters to a safetensors file.
///
/// Spec: REQ-CORE-004
pub fn save_parameters(
    path: &Path,
    parameters: &HashMap<String, Array1<Float>>,
) -> Result<(), CarnotError> {
    let mut data_buffers: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();
    for (name, arr) in parameters {
        let slice = arr.as_slice().unwrap();
        let bytes: &[u8] = bytemuck::cast_slice(slice);
        data_buffers.push((name.clone(), bytes.to_vec(), vec![arr.len()]));
    }

    let views: Vec<(String, TensorView<'_>)> = data_buffers
        .iter()
        .map(|(name, bytes, shape)| {
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .map_err(|e| CarnotError::Serialization(e.to_string()))
                .unwrap();
            (name.clone(), view)
        })
        .collect();

    safetensors::tensor::serialize_to_file(
        views.iter().map(|(name, view)| (name.as_str(), view)),
        &None,
        path,
    )
    .map_err(|e| CarnotError::Serialization(e.to_string()))?;

    Ok(())
}

/// Load parameters from a safetensors file.
///
/// Spec: REQ-CORE-004
pub fn load_parameters(path: &Path) -> Result<HashMap<String, Array1<Float>>, CarnotError> {
    let data = std::fs::read(path).map_err(|e| CarnotError::Serialization(e.to_string()))?;
    let tensors =
        SafeTensors::deserialize(&data).map_err(|e| CarnotError::Serialization(e.to_string()))?;

    let mut parameters = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let float_data: &[Float] = bytemuck::cast_slice(tensor.data());
        let arr = Array1::from_vec(float_data.to_vec());
        parameters.insert(name.to_string(), arr);
    }
    Ok(parameters)
}

/// Save training metadata as JSON sidecar.
pub fn save_metadata(path: &Path, metadata: &ModelMetadata) -> Result<(), CarnotError> {
    let json = serde_json::to_string_pretty(metadata)
        .map_err(|e| CarnotError::Serialization(e.to_string()))?;
    std::fs::write(path, json).map_err(|e| CarnotError::Serialization(e.to_string()))?;
    Ok(())
}

/// Load training metadata from JSON sidecar.
pub fn load_metadata(path: &Path) -> Result<ModelMetadata, CarnotError> {
    let json =
        std::fs::read_to_string(path).map_err(|e| CarnotError::Serialization(e.to_string()))?;
    serde_json::from_str(&json).map_err(|e| CarnotError::Serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_save_load_parameters_roundtrip() {
        // SCENARIO-CORE-004
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");

        let mut params = HashMap::new();
        params.insert("weight".to_string(), array![1.0, 2.0, 3.0]);
        params.insert("bias".to_string(), array![0.1, 0.2]);

        save_parameters(&path, &params).unwrap();
        let loaded = load_parameters(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        for (name, arr) in &params {
            let loaded_arr = &loaded[name];
            assert_eq!(arr.len(), loaded_arr.len());
            for (a, b) in arr.iter().zip(loaded_arr.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_save_load_metadata_roundtrip() {
        // SCENARIO-CORE-004
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metadata.json");

        let metadata = ModelMetadata {
            step: 42,
            loss_history: vec![1.0, 0.5, 0.25],
        };

        save_metadata(&path, &metadata).unwrap();
        let loaded = load_metadata(&path).unwrap();

        assert_eq!(loaded.step, 42);
        assert_eq!(loaded.loss_history.len(), 3);
    }
}
