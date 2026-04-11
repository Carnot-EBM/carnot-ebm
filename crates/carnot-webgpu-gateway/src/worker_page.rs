//! Generates the static HTML+JS worker page served to browsers.

/// The WGSL compute shader for Ising energy (same as carnot-gpu).
pub const ISING_SHADER: &str = include_str!("../../carnot-gpu/src/shaders/ising_energy.wgsl");

/// Generate the HTML page that browsers visit to become compute workers.
/// The WebSocket URL is auto-detected from window.location.host in the browser.
pub fn worker_html() -> String {
    String::from(
        r#"<!DOCTYPE html>
<html>
<head>
<title>Carnot WebGPU Worker</title>
<style>
body { font-family: monospace; background: #111; color: #0f0; padding: 2em; }
h1 { color: #0f0; }
#status { padding: 1em; border: 1px solid #0f0; margin: 1em 0; }
#log { height: 400px; overflow-y: auto; border: 1px solid #333; padding: 0.5em; font-size: 12px; }
.error { color: #f00; }
.result { color: #0ff; }
</style>
</head>
<body>
<h1>Carnot WebGPU Compute Worker</h1>
<div id="status">Initializing...</div>
<div id="log"></div>

<script>
// Auto-detect WebSocket URL from current page location so remote browsers work
const WS_URL = (window.location.protocol === "https:" ? "wss://" : "ws://") + window.location.host + "/ws";
let ws = null;
let device = null;
let workerId = "unknown";

function log(msg, cls) {
    const div = document.getElementById('log');
    const line = document.createElement('div');
    line.textContent = `[${new Date().toISOString().substr(11,8)}] ${msg}`;
    if (cls) line.className = cls;
    div.appendChild(line);
    div.scrollTop = div.scrollHeight;
}

function setStatus(msg) {
    document.getElementById('status').textContent = msg;
}

async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported in this browser");
    }
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance"
    });
    if (!adapter) throw new Error("No WebGPU adapter found");

    // adapter.info is the modern API; requestAdapterInfo() is deprecated
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    log(`GPU: ${info.device || info.description || info.vendor || "unknown"}`);

    device = await adapter.requestDevice();
    return info;
}

async function computeIsingEnergy(shader, coupling, bias, inputs, dim, batchSize) {
    const start = performance.now();

    // Create buffers
    const couplingBuf = device.createBuffer({
        size: coupling.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(couplingBuf, 0, new Float32Array(coupling));

    const biasBuf = device.createBuffer({
        size: bias.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(biasBuf, 0, new Float32Array(bias));

    const inputBuf = device.createBuffer({
        size: inputs.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(inputBuf, 0, new Float32Array(inputs));

    const outputBuf = device.createBuffer({
        size: batchSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const paramsBuf = device.createBuffer({
        size: 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([dim, batchSize]));

    const stagingBuf = device.createBuffer({
        size: batchSize * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    // Create pipeline
    const shaderModule = device.createShaderModule({ code: shader });
    const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: shaderModule, entryPoint: "main" },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: couplingBuf } },
            { binding: 1, resource: { buffer: biasBuf } },
            { binding: 2, resource: { buffer: inputBuf } },
            { binding: 3, resource: { buffer: outputBuf } },
            { binding: 4, resource: { buffer: paramsBuf } },
        ],
    });

    // Dispatch
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(batchSize / 64));
    pass.end();

    commandEncoder.copyBufferToBuffer(outputBuf, 0, stagingBuf, 0, batchSize * 4);
    device.queue.submit([commandEncoder.finish()]);

    // Read back
    await stagingBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuf.getMappedRange().slice(0));
    stagingBuf.unmap();

    // Cleanup
    couplingBuf.destroy();
    biasBuf.destroy();
    inputBuf.destroy();
    outputBuf.destroy();
    paramsBuf.destroy();
    stagingBuf.destroy();

    const elapsed = performance.now() - start;
    return { energies: Array.from(result), computeMs: elapsed };
}

function connect() {
    setStatus("Connecting to gateway...");
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        log("Connected to gateway");
    };

    ws.onmessage = async (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === "welcome") {
            workerId = msg.worker_id;
            log(`Assigned worker ID: ${workerId}`);
            // Send ready with GPU info
            ws.send(JSON.stringify({
                type: "ready",
                gpu_name: "WebGPU Browser Worker"
            }));
            setStatus(`Connected as ${workerId.substr(0,8)}... — waiting for work`);
        }

        else if (msg.type === "compute") {
            setStatus(`Computing work ${msg.work_id.substr(0,8)}... (dim=${msg.dim}, batch=${msg.batch_size})`);
            log(`Received work: dim=${msg.dim}, batch=${msg.batch_size}`);

            try {
                const result = await computeIsingEnergy(
                    msg.shader, msg.coupling, msg.bias, msg.inputs,
                    msg.dim, msg.batch_size
                );
                ws.send(JSON.stringify({
                    type: "result",
                    work_id: msg.work_id,
                    energies: result.energies,
                    compute_ms: result.computeMs,
                }));
                log(`Completed: ${result.energies.length} energies in ${result.computeMs.toFixed(1)}ms`, 'result');
                setStatus(`Idle — completed ${msg.work_id.substr(0,8)}...`);
            } catch (e) {
                ws.send(JSON.stringify({
                    type: "error",
                    work_id: msg.work_id,
                    message: e.toString(),
                }));
                log(`Error: ${e}`, 'error');
                setStatus("Error — waiting for next work");
            }

            // Ready for more work
            ws.send(JSON.stringify({ type: "ready", gpu_name: "WebGPU Browser Worker" }));
        }

        else if (msg.type === "idle") {
            setStatus("No work available — waiting...");
            // Poll again in 5 seconds
            setTimeout(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: "ready", gpu_name: "WebGPU Browser Worker" }));
                }
            }, 5000);
        }
    };

    ws.onclose = () => {
        log("Disconnected from gateway");
        setStatus("Disconnected — reconnecting in 5s...");
        setTimeout(connect, 5000);
    };

    ws.onerror = (e) => {
        log(`WebSocket error: ${e}`, 'error');
    };
}

// Initialize
(async () => {
    try {
        const gpuInfo = await initWebGPU();
        log("WebGPU initialized successfully");
        connect();
    } catch (e) {
        log(`Fatal: ${e}`, 'error');
        setStatus(`Error: ${e.message}`);
    }
})();
</script>
</body>
</html>"#,
    )
}
