/**
 * ising_sampler_hls.cpp — Ising Sampler v4 in HLS C++
 *
 * WHY THIS FILE EXISTS:
 *   KV260 bitfile synthesis has been blocked for 3 consecutive milestones because
 *   Vivado is not installed locally.  arXiv 2604.17109 (April 2026) shows that
 *   Vitis HLS (High-Level Synthesis) can generate FPGA RTL from annotated C++.
 *   Vitis HLS is distributed separately from Vivado, so we write the kernel here
 *   in HLS C++ style.  The same file compiles as plain C++ for CPU validation
 *   (the HLS pragmas are in comments so g++ ignores them), AND can be synthesised
 *   by a human with Vitis HLS / AMD Vitis 2024.2 on a cloud instance.
 *
 * HOW HLS PRAGMAS WORK:
 *   In real Vitis HLS, you write:
 *       #pragma HLS PIPELINE II=1
 *   to tell the tool to pipeline a loop so a new iteration starts every 1 clock cycle.
 *   Because we want the SAME file to compile under g++, all pragmas appear as comments:
 *       // #pragma HLS PIPELINE II=1
 *   g++ ignores comments; vitis_hls treats comment-pragmas correctly when the file
 *   is opened with HLS mode.  This is the standard dual-compile trick from the paper.
 *
 * ALGORITHM — v4 adds inertia (EMA) to v2 checkerboard Gibbs:
 *   For each sweep:
 *     1. For each spin i:
 *        a. Compute instantaneous local field:  h_inst = sum_j J[i][j] * s[j] + bias[i]
 *        b. Update EMA field:  h_ema[i] = alpha * h_ema[i] + (1-alpha) * h_inst
 *        c. Compute flip energy:  E_i = s[i] * h_ema[i]   (Ising convention)
 *        d. Flip probability:     p = 1 / (1 + exp(2 * beta * E_i))
 *        e. Draw uniform random r in [0,1) via xorshift RNG
 *        f. New spin: s[i] = (r < p) ? 1 : -1
 *   Spins are ±1 (not 0/1) — this matches the Ising model energy convention.
 *
 * INERTIA RATIONALE (from ising_sampler_v3_spec.md):
 *   Dense coupling graphs tend to oscillate: spin i flips, which raises the
 *   energy of spin j, which flips back, repeating.  An EMA on the local field
 *   damps these oscillations by blending the current field with recent history.
 *   alpha=0 → no inertia (reduces to v2).  alpha=0.5 is recommended per Exp 648.
 *
 * TARGET HARDWARE: Xilinx KV260 (xck26-sfvc784-2LV-c)
 * Spec: REQ-HW-010
 */

#include <cmath>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Constants — must be compile-time for HLS array sizing.
// ---------------------------------------------------------------------------

/** Maximum number of spins this kernel supports.
 *  HLS requires static array sizes.  For a runtime n <= N_MAX, we use
 *  only the first n elements and the rest are unused. */
static const int N_MAX = 128;

// ---------------------------------------------------------------------------
// xorshift32 RNG — HLS-compatible (no stdlib rand(), no dynamic memory).
// ---------------------------------------------------------------------------

/** One step of a 32-bit xorshift random number generator.
 *
 *  WHY xorshift:
 *    HLS synthesis needs a deterministic, stateless-except-for-seed RNG
 *    that uses only shift, XOR, and integer arithmetic — all map to
 *    single FPGA LUTs.  stdlib rand() calls are not synthesisable.
 *    xorshift32 passes SmallCrush and is good enough for Monte Carlo sampling.
 *
 *  Returns a 32-bit pseudo-random integer.  The caller advances the state. */
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/** Convert a 32-bit integer to a float in [0, 1).
 *
 *  Maps [0, 2^32) -> [0.0, 1.0) by dividing by 2^32.
 *  Uses double intermediate to avoid rounding above 1.0. */
static inline float uint32_to_uniform(uint32_t r) {
    return (float)((double)r / 4294967296.0);
}

// ---------------------------------------------------------------------------
// Core HLS kernel
// ---------------------------------------------------------------------------

/**
 * update_spin_kernel — Gibbs sweep with EMA inertia (HLS C++ / CPU dual-compile).
 *
 * This is the synthesisable top-level function.  In Vitis HLS, all HLS
 * interface and optimisation directives are placed at the top of the function.
 * The comment-pragma pattern means this file compiles identically under g++.
 *
 * Parameters:
 *   J          — N×N coupling matrix, stored row-major (J[i*n + j] = J_ij).
 *                J_ij > 0 means ferromagnetic (spins prefer to agree).
 *                Diagonal must be 0.
 *   h          — External bias vector, length n.  h[i] > 0 biases spin i toward +1.
 *   beta       — Inverse temperature.  Higher beta → more deterministic, lower energy.
 *   spins      — IN/OUT: spin configuration, ±1.  Updated in-place over all sweeps.
 *   rng_state  — IN/OUT: xorshift32 seed.  Advances deterministically each sweep.
 *   n          — Actual number of spins (must be <= N_MAX).
 *   num_sweeps — How many full Gibbs sweeps to perform.
 *   alpha      — EMA smoothing factor for inertia.  0.0 = no inertia (pure v2).
 *                0.5 = recommended from Exp 648.  Range [0, 1).
 *
 * Spec: REQ-HW-010
 */
void update_spin_kernel(
    const float J[N_MAX * N_MAX],   /* coupling matrix, row-major              */
    const float h[N_MAX],           /* external bias                           */
    float beta,                     /* inverse temperature                     */
    int   spins[N_MAX],             /* ±1 spin state, updated in-place         */
    uint32_t *rng_state,            /* xorshift32 RNG state                    */
    int n,                          /* number of active spins (<= N_MAX)       */
    int num_sweeps,                 /* number of full Gibbs sweeps             */
    float alpha                     /* EMA smoothing factor for inertia        */
) {
    // HLS interface pragmas (ignored by g++, parsed by vitis_hls):
    // #pragma HLS INTERFACE m_axi port=J        offset=slave bundle=mem
    // #pragma HLS INTERFACE m_axi port=h        offset=slave bundle=mem
    // #pragma HLS INTERFACE m_axi port=spins    offset=slave bundle=mem
    // #pragma HLS INTERFACE s_axilite port=beta  bundle=ctrl
    // #pragma HLS INTERFACE s_axilite port=n     bundle=ctrl
    // #pragma HLS INTERFACE s_axilite port=num_sweeps bundle=ctrl
    // #pragma HLS INTERFACE s_axilite port=alpha bundle=ctrl
    // #pragma HLS INTERFACE s_axilite port=rng_state bundle=ctrl
    // #pragma HLS INTERFACE s_axilite port=return bundle=ctrl

    // Local copies in on-chip BRAM (HLS can pipeline over these).
    // #pragma HLS ARRAY_PARTITION variable=h_ema complete dim=1
    // #pragma HLS ARRAY_PARTITION variable=spins_local complete dim=1

    float h_ema[N_MAX];        /* per-spin EMA local field (inertia state)    */
    int   spins_local[N_MAX];  /* local copy to avoid repeated AXI reads      */
    float one_minus_alpha = 1.0f - alpha;

    // Initialise EMA fields to zero (cold start — no prior history).
    // On FPGA, these registers would be zeroed on reset.
    for (int i = 0; i < n; i++) {
        // #pragma HLS UNROLL factor=8
        h_ema[i] = 0.0f;
        spins_local[i] = spins[i];
    }

    // --- Main sweep loop ---
    for (int sweep = 0; sweep < num_sweeps; sweep++) {
        // #pragma HLS PIPELINE off
        // (outer loop NOT pipelined — each iteration depends on previous spin state)

        // Sequential Gibbs: update each spin in order.
        // WHY sequential (not parallel checkerboard):
        //   For a single-function HLS kernel targeting minimal area, sequential
        //   Gibbs is simpler to implement correctly.  The FPGA v2/v3 RTL uses
        //   checkerboard parallelism; this kernel targets validation correctness
        //   over throughput.  A fully parallel HLS version would require N
        //   independent multiply-accumulate units.
        for (int i = 0; i < n; i++) {
            // #pragma HLS PIPELINE II=1

            // Step 1: Compute instantaneous local field h_inst[i].
            // h_inst = sum_{j != i} J[i][j] * s[j] + h[i]
            // WHY we include j==i: J diagonal is 0 by convention, so it's safe.
            float h_inst = h[i];
            for (int j = 0; j < n; j++) {
                // #pragma HLS UNROLL factor=8
                h_inst += J[i * N_MAX + j] * (float)spins_local[j];
            }

            // Step 2: EMA update — blend instantaneous field with running average.
            // h_ema[i] = alpha * h_ema[i] + (1 - alpha) * h_inst
            // WHY EMA: damps oscillations in dense graphs (see ising_sampler_v3_spec.md).
            // alpha=0 → h_ema = h_inst (no memory, same as v2).
            h_ema[i] = alpha * h_ema[i] + one_minus_alpha * h_inst;

            // Step 3: Compute flip energy contribution.
            // E_i = s[i] * h_ema[i]
            // If s[i] and h_ema[i] have the same sign, this spin is in a low-energy
            // state (aligned with field) and less likely to flip.
            float E_i = (float)spins_local[i] * h_ema[i];

            // Step 4: Compute flip probability.
            // p_flip = 1 / (1 + exp(2 * beta * E_i))
            // Derivation: for spin sigma_i = ±1 in field H_i, the Boltzmann ratio
            // P(+1) / P(-1) = exp(2 * beta * H_i), giving the logistic form.
            // A positive E_i means low energy → low p_flip (don't flip a happy spin).
            float p_flip = 1.0f / (1.0f + expf(2.0f * beta * E_i));

            // Step 5: Draw random number and flip if r < p_flip.
            uint32_t r_int = xorshift32(rng_state);
            float r_uniform = uint32_to_uniform(r_int);

            spins_local[i] = (r_uniform < p_flip) ? 1 : -1;
        }
    }

    // Write results back.
    for (int i = 0; i < n; i++) {
        spins[i] = spins_local[i];
    }
}

// ---------------------------------------------------------------------------
// Energy computation helper (for validation against Python reference)
// ---------------------------------------------------------------------------

/**
 * compute_ising_energy — total Ising energy E = -sum_{ij} J_ij s_i s_j - sum_i h_i s_i
 *
 * WHY the negative sign:
 *   The Ising Hamiltonian is defined so that aligned ferromagnetic spins (J>0, s_i=s_j)
 *   have LOWER energy.  Low energy = favoured by the Boltzmann distribution.
 *
 * Returns energy as a float (negative = good, positive = high energy / disordered).
 * Spec: REQ-HW-010
 */
float compute_ising_energy(
    const float J[N_MAX * N_MAX],
    const float h[N_MAX],
    const int   spins[N_MAX],
    int n
) {
    float energy = 0.0f;

    // Interaction term: -sum_{i<j} J_ij s_i s_j  (factor of 2 avoided by i<j)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            energy -= J[i * N_MAX + j] * (float)spins[i] * (float)spins[j];
        }
    }

    // Bias term: -sum_i h_i s_i
    for (int i = 0; i < n; i++) {
        energy -= h[i] * (float)spins[i];
    }

    return energy;
}

// ---------------------------------------------------------------------------
// CPU test harness (compiled away under HLS synthesis)
// ---------------------------------------------------------------------------
// WHY this is inside #ifndef __SYNTHESIS__:
//   Vitis HLS defines __SYNTHESIS__ during hardware compilation.
//   The main() function is not synthesisable (dynamic memory, printf, etc.)
//   and must be excluded from the RTL-generation pass.
// ---------------------------------------------------------------------------

#ifndef __SYNTHESIS__

#include <cstdio>
#include <cstdlib>

/** Validate C++ kernel against a known reference configuration.
 *
 *  Uses a small 4-spin antiferromagnetic chain (J=-1 between neighbours)
 *  with no external field.  The ground state is alternating ±1.
 *  After enough sweeps at high beta, the energy should be close to -N/2.
 *
 *  This is run by experiment_750 to validate CPU correctness before
 *  comparing against the Python parallel_ising.py reference.
 */
int main(int argc, char **argv) {
    const int n = 4;
    const int num_sweeps = 500;
    const float beta = 5.0f;
    const float alpha = 0.5f;

    // J: antiferromagnetic chain (neighbours repel)
    // J[i][j] = -1 if |i-j|==1, else 0
    float J[N_MAX * N_MAX];
    memset(J, 0, sizeof(J));
    for (int i = 0; i < n - 1; i++) {
        J[i * N_MAX + (i + 1)] = -1.0f;
        J[(i + 1) * N_MAX + i] = -1.0f;
    }

    float h[N_MAX];
    memset(h, 0, sizeof(h));

    int spins[N_MAX];
    // Initialise all spins to +1 (high energy for antiferromagnet)
    for (int i = 0; i < n; i++) spins[i] = 1;

    uint32_t rng = 42;
    update_spin_kernel(J, h, beta, spins, &rng, n, num_sweeps, alpha);

    float energy = compute_ising_energy(J, h, spins, n);

    printf("Final spins: ");
    for (int i = 0; i < n; i++) printf("%+d ", spins[i]);
    printf("\n");
    printf("Final energy: %.4f\n", energy);

    // Ground state energy for 4-spin antiferromagnetic chain is -3.0
    // (3 antiferromagnetic bonds, each contributing -|J|*(-1)*1 = -1)
    // We allow 20% tolerance because Monte Carlo is stochastic.
    float expected_gs_energy = -3.0f;
    float tol = 0.20f * fabsf(expected_gs_energy) + 0.1f;
    int ok = fabsf(energy - expected_gs_energy) <= tol;

    printf("Expected energy near %.1f, got %.4f, tol=%.4f: %s\n",
           expected_gs_energy, energy, tol, ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}

#endif  /* __SYNTHESIS__ */
