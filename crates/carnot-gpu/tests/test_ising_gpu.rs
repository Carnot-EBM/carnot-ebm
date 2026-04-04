//! Integration tests for GPU-accelerated Ising model.

use carnot_gpu::{GpuContext, IsingGpu};
use ndarray::Array2;

#[test]
fn test_gpu_context_creation() {
    let ctx = GpuContext::new();
    match ctx {
        Ok(ctx) => {
            println!("GPU detected: {}", ctx.device_name());
            println!("Backend: {:?}", ctx.backend());
        }
        Err(e) => {
            println!("No GPU available (OK in CI): {}", e);
        }
    }
}

#[test]
fn test_ising_energy_batch() {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("Skipping GPU test — no GPU available");
            return;
        }
    };

    let dim = 4;
    // Simple coupling: identity matrix (E = -0.5 * ||x||^2 when bias=0)
    let mut coupling = vec![0.0f32; dim * dim];
    for i in 0..dim {
        coupling[i * dim + i] = 1.0;
    }
    let bias = vec![0.0f32; dim];

    let ising = IsingGpu::new(&ctx, &coupling, &bias, dim);
    assert_eq!(ising.dim(), dim);

    // Batch of 3 configurations
    let inputs = vec![
        // x = [1, 0, 0, 0] -> E = -0.5 * 1 = -0.5
        1.0, 0.0, 0.0, 0.0,
        // x = [1, 1, 0, 0] -> E = -0.5 * 2 = -1.0
        1.0, 1.0, 0.0, 0.0,
        // x = [0, 0, 0, 0] -> E = 0.0
        0.0, 0.0, 0.0, 0.0,
    ];

    let energies = ising.energy_batch(&ctx, &inputs).expect("GPU compute failed");
    assert_eq!(energies.len(), 3);

    println!("GPU energies: {:?}", energies);

    // Check values (with tolerance for GPU floating-point)
    assert!((energies[0] - (-0.5)).abs() < 0.01, "Expected -0.5, got {}", energies[0]);
    assert!((energies[1] - (-1.0)).abs() < 0.01, "Expected -1.0, got {}", energies[1]);
    assert!((energies[2] - 0.0).abs() < 0.01, "Expected 0.0, got {}", energies[2]);
}

#[test]
fn test_ising_gpu_matches_cpu() {
    let ctx = match GpuContext::new() {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("Skipping GPU test — no GPU available");
            return;
        }
    };

    let dim = 8;
    // Random-ish coupling matrix (symmetric)
    let mut coupling = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let val = ((i * 7 + j * 13) % 10) as f32 / 10.0 - 0.5;
            coupling[i * dim + j] = val;
            coupling[j * dim + i] = val;
        }
    }
    let bias: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 0.4).collect();

    let ising = IsingGpu::new(&ctx, &coupling, &bias, dim);

    // Test batch
    let batch_size = 10;
    let inputs: Vec<f32> = (0..batch_size * dim)
        .map(|i| ((i * 17 + 3) % 20) as f32 / 20.0)
        .collect();

    let gpu_energies = ising.energy_batch(&ctx, &inputs).expect("GPU failed");

    // Compute on CPU for comparison
    use ndarray::{Array1, ArrayView1};
    let j = Array2::from_shape_vec((dim, dim), coupling.clone()).unwrap();
    let b = Array1::from_vec(bias.clone());

    for i in 0..batch_size {
        let x = ArrayView1::from(&inputs[i * dim..(i + 1) * dim]);
        let jx = j.dot(&x);
        let cpu_energy = -0.5 * x.dot(&jx) - b.dot(&x);

        assert!(
            (gpu_energies[i] - cpu_energy).abs() < 0.01,
            "Sample {}: GPU={}, CPU={}",
            i,
            gpu_energies[i],
            cpu_energy
        );
    }

    println!("GPU matches CPU on {} samples (dim={})", batch_size, dim);
}
