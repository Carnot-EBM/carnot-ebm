import re

with open("hardware/kv260/ising_sampler_v3.v", "r") as f:
    code = f.read()

old_str = """        // --- Steps 2a-2f: EMA update + Metropolis for each spin ---
        for (i = 0; i < N; i = i + 1) begin : spin_update

            // --- 2a: Extract h_in[i] from packed bus ---
            // WHY packed bus: Verilog module ports cannot be arrays,
            // so N fields of FIELD_WIDTH bits each are concatenated.
            // The slice below reverses this for spin i.
            reg signed [FIELD_WIDTH-1:0] h_in_i;
            h_in_i = $signed(h_in[(i+1)*FIELD_WIDTH-1 -: FIELD_WIDTH]);

            // --- 2b: EMA update (v3 CORE ADDITION) ---
            // h_ema[i] <= (EMA_ALPHA_NUM * h_ema[i] + EMA_BETA_NUM * h_in[i])
            //             / EMA_ALPHA_DEN
            // With defaults (7/8): h_ema = (7*h_ema + 1*h_in) >> 3
            // Division by power-of-two synthesises as arithmetic right shift.
            // Uses extended width to avoid overflow before shifting:
            //   (7 * h_ema) can be up to 7 * 2^17 = 2^20, needing 21 bits.
            reg signed [FIELD_WIDTH+3:0] ema_wide;
            ema_wide = ($signed(EMA_ALPHA_NUM) * $signed(h_ema[i])
                      + $signed(EMA_BETA_NUM)  * $signed(h_in_i))
                      >>> EMA_SHIFT;
            // Saturate back to FIELD_WIDTH to avoid wrap-around.
            if (ema_wide > $signed({{4{1'b0}}, {(FIELD_WIDTH-1){1'b1}}}))
                h_ema[i] <= {1'b0, {(FIELD_WIDTH-1){1'b1}}};  // MAX_POS
            else if (ema_wide < $signed({1'b1, {(FIELD_WIDTH-1){1'b0}}, {4{1'b0}}}))
                h_ema[i] <= {1'b1, {(FIELD_WIDTH-1){1'b0}}};  // MIN_NEG
            else
                h_ema[i] <= ema_wide[FIELD_WIDTH-1:0];

            // --- 2c: Neighbour coupling (ring topology: left + right neighbours) ---
            // WHY ring: simplest topology that exercises coupling logic, matches
            // 1D periodic Ising benchmark. Replace with full adjacency matrix
            // or RAM-based neighbour list for production constraint graphs.
            //
            // Spin value convention: s[i]=1 => physical spin +1; s[i]=0 => -1.
            // Represented as signed: spin_val = s[i] ? +1 : -1.
            // Scaled by FIELD_WIDTH fixed-point: +1 => +(2^(FIELD_WIDTH/2))
            //   but for simplicity here we use +1 and -1 directly in integer.
            reg signed [FIELD_WIDTH-1:0] left_spin_val;
            reg signed [FIELD_WIDTH-1:0] right_spin_val;
            reg signed [FIELD_WIDTH-1:0] h_nbr;

            left_spin_val  = s[(i + N - 1) % N] ? $signed({{(FIELD_WIDTH-1){1'b0}}, 1'b1})
                                                 : $signed({1'b1, {(FIELD_WIDTH-1){1'b0}}});
            right_spin_val = s[(i + 1) % N]      ? $signed({{(FIELD_WIDTH-1){1'b0}}, 1'b1})
                                                 : $signed({1'b1, {(FIELD_WIDTH-1){1'b0}}});
            // Note: modulo in generate params must use localparams or literals.
            // In simulation, modulo on integers works; synthesis requires static.
            // For production, replace with generate/genvar ring addressing.
            h_nbr = $signed(J_INT) * (left_spin_val + right_spin_val);

            // --- 2d: Total effective field using EMA (v3: use h_ema, not h_inst) ---
            // v2 used h_eff computed from h_inst (current neighbours only).
            // v3 uses h_ema[i] (smoothed) + current neighbour coupling.
            // This is the "flip probability uses h_ema" change from the spec.
            reg signed [FIELD_WIDTH:0] h_total;
            h_total = $signed(h_ema[i]) + $signed(h_nbr);

            // --- 2e: Metropolis energy change ---
            // deltaE = 2 * s_signed * h_total
            // Physical interpretation: if s_signed and h_total have the same sign,
            // the spin is already aligned with the field — flipping would INCREASE
            // energy (deltaE > 0), so only accept with thermal probability.
            // If opposite signs, flipping lowers energy (deltaE <= 0), always accept.
            reg signed [FIELD_WIDTH+1:0] s_signed;
            reg signed [FIELD_WIDTH+3:0] deltaE;

            s_signed = s[i] ? $signed({{(FIELD_WIDTH){1'b0}}, 1'b1})
                            : $signed({1'b1, {(FIELD_WIDTH+1){1'b0}}});
            deltaE = 2 * s_signed * $signed(h_total);"""

new_str = """        // --- Steps 2a-2f: EMA update + Metropolis for each spin ---
        for (i = 0; i < N; i = i + 1) begin : spin_update

            // Declarations moved to top for Verilog-2005 compatibility:
            reg signed [FIELD_WIDTH-1:0] h_in_i;
            reg signed [FIELD_WIDTH+3:0] ema_wide;
            reg signed [FIELD_WIDTH-1:0] left_spin_val;
            reg signed [FIELD_WIDTH-1:0] right_spin_val;
            reg signed [FIELD_WIDTH-1:0] h_nbr;
            reg signed [FIELD_WIDTH:0] h_total;
            reg signed [FIELD_WIDTH+1:0] s_signed;
            reg signed [FIELD_WIDTH+3:0] deltaE;

            // --- 2a: Extract h_in[i] from packed bus ---
            // WHY packed bus: Verilog module ports cannot be arrays,
            // so N fields of FIELD_WIDTH bits each are concatenated.
            // The slice below reverses this for spin i.
            h_in_i = $signed(h_in[(i+1)*FIELD_WIDTH-1 -: FIELD_WIDTH]);

            // --- 2b: EMA update (v3 CORE ADDITION) ---
            // h_ema[i] <= (EMA_ALPHA_NUM * h_ema[i] + EMA_BETA_NUM * h_in[i])
            //             / EMA_ALPHA_DEN
            // With defaults (7/8): h_ema = (7*h_ema + 1*h_in) >> 3
            // Division by power-of-two synthesises as arithmetic right shift.
            // Uses extended width to avoid overflow before shifting:
            //   (7 * h_ema) can be up to 7 * 2^17 = 2^20, needing 21 bits.
            ema_wide = ($signed(EMA_ALPHA_NUM) * $signed(h_ema[i])
                      + $signed(EMA_BETA_NUM)  * $signed(h_in_i))
                      >>> EMA_SHIFT;
            // Saturate back to FIELD_WIDTH to avoid wrap-around.
            if (ema_wide > $signed({{4{1'b0}}, {(FIELD_WIDTH-1){1'b1}}}))
                h_ema[i] <= {1'b0, {(FIELD_WIDTH-1){1'b1}}};  // MAX_POS
            else if (ema_wide < $signed({1'b1, {(FIELD_WIDTH-1){1'b0}}, {4{1'b0}}}))
                h_ema[i] <= {1'b1, {(FIELD_WIDTH-1){1'b0}}};  // MIN_NEG
            else
                h_ema[i] <= ema_wide[FIELD_WIDTH-1:0];

            // --- 2c: Neighbour coupling (ring topology: left + right neighbours) ---
            // WHY ring: simplest topology that exercises coupling logic, matches
            // 1D periodic Ising benchmark. Replace with full adjacency matrix
            // or RAM-based neighbour list for production constraint graphs.
            //
            // Spin value convention: s[i]=1 => physical spin +1; s[i]=0 => -1.
            // Represented as signed: spin_val = s[i] ? +1 : -1.
            // Scaled by FIELD_WIDTH fixed-point: +1 => +(2^(FIELD_WIDTH/2))
            //   but for simplicity here we use +1 and -1 directly in integer.
            left_spin_val  = s[(i + N - 1) % N] ? $signed({{(FIELD_WIDTH-1){1'b0}}, 1'b1})
                                                 : $signed({1'b1, {(FIELD_WIDTH-1){1'b0}}});
            right_spin_val = s[(i + 1) % N]      ? $signed({{(FIELD_WIDTH-1){1'b0}}, 1'b1})
                                                 : $signed({1'b1, {(FIELD_WIDTH-1){1'b0}}});
            // Note: modulo in generate params must use localparams or literals.
            // In simulation, modulo on integers works; synthesis requires static.
            // For production, replace with generate/genvar ring addressing.
            h_nbr = $signed(J_INT) * (left_spin_val + right_spin_val);

            // --- 2d: Total effective field using EMA (v3: use h_ema, not h_inst) ---
            // v2 used h_eff computed from h_inst (current neighbours only).
            // v3 uses h_ema[i] (smoothed) + current neighbour coupling.
            // This is the "flip probability uses h_ema" change from the spec.
            h_total = $signed(h_ema[i]) + $signed(h_nbr);

            // --- 2e: Metropolis energy change ---
            // deltaE = 2 * s_signed * h_total
            // Physical interpretation: if s_signed and h_total have the same sign,
            // the spin is already aligned with the field — flipping would INCREASE
            // energy (deltaE > 0), so only accept with thermal probability.
            // If opposite signs, flipping lowers energy (deltaE <= 0), always accept.
            s_signed = s[i] ? $signed({{(FIELD_WIDTH){1'b0}}, 1'b1})
                            : $signed({1'b1, {(FIELD_WIDTH+1){1'b0}}});
            deltaE = 2 * s_signed * $signed(h_total);"""

if old_str in code:
    with open("hardware/kv260/ising_sampler_v3.v", "w") as f:
        f.write(code.replace(old_str, new_str))
    print("Success")
else:
    print("Error: Old string not found")

