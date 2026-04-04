//! Carnot WebGPU Gateway Server
//!
//! Starts an HTTP+WebSocket server that:
//! 1. Serves a WebGPU worker page to browsers at /
//! 2. Accepts WebSocket connections from workers at /ws
//! 3. Exposes /health and /workers API endpoints
//!
//! Usage:
//!   cargo run -p carnot-webgpu-gateway --bin gateway
//!   # Then open http://localhost:3000 in a WebGPU-capable browser

use carnot_webgpu_gateway::gateway::{build_router, GatewayState};
use carnot_webgpu_gateway::worker_page::worker_html;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let addr = format!("{}:{}", host, port);

    // Generate the worker HTML page
    let ws_url = format!("ws://localhost:{}/ws", port);
    let html = worker_html(&ws_url);

    // Write to static dir
    let static_dir = std::env::temp_dir().join("carnot-gateway-static");
    std::fs::create_dir_all(&static_dir).unwrap();
    std::fs::write(static_dir.join("index.html"), &html).unwrap();

    let state = Arc::new(GatewayState::new());
    let app = build_router(state, static_dir.to_str().unwrap());

    println!("==============================================");
    println!("  Carnot WebGPU Distributed Compute Gateway");
    println!("==============================================");
    println!();
    println!("  Server:  http://{}", addr);
    println!("  Worker page:  http://localhost:{}/", port);
    println!("  WebSocket:    ws://localhost:{}/ws", port);
    println!("  Health:       http://localhost:{}/health", port);
    println!("  Workers:      http://localhost:{}/workers", port);
    println!();
    println!("  Open the worker page in a WebGPU-capable browser");
    println!("  (Chrome 113+, Firefox 130+, Edge 113+) to contribute GPU.");
    println!();

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    tracing::info!("Gateway listening on {}", addr);
    axum::serve(listener, app).await.unwrap();
}
