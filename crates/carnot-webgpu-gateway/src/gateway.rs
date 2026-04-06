//! Gateway server: distributes work to browser WebGPU workers.

use crate::protocol::{ServerMessage, WorkerMessage};
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Router,
};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

/// Connected worker state.
#[derive(Debug, Clone)]
pub struct Worker {
    pub id: String,
    pub gpu_name: String,
    pub busy: bool,
}

/// Work unit pending assignment.
#[derive(Debug, Clone)]
pub struct WorkUnit {
    pub id: String,
    pub message: ServerMessage,
}

/// Shared gateway state.
pub struct GatewayState {
    pub workers: RwLock<HashMap<String, Worker>>,
    pub work_queue: Mutex<Vec<WorkUnit>>,
    pub results: Mutex<HashMap<String, Vec<f32>>>,
}

impl GatewayState {
    pub fn new() -> Self {
        Self {
            workers: RwLock::new(HashMap::new()),
            work_queue: Mutex::new(Vec::new()),
            results: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for GatewayState {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the axum router for the gateway.
pub fn build_router(state: Arc<GatewayState>, static_dir: &str) -> Router {
    Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(health))
        .route("/workers", get(list_workers))
        .nest_service("/", ServeDir::new(static_dir))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    axum::Json(serde_json::json!({"status": "ok"}))
}

async fn list_workers(State(state): State<Arc<GatewayState>>) -> impl IntoResponse {
    let workers = state.workers.read().await;
    let list: Vec<&Worker> = workers.values().collect();
    axum::Json(serde_json::json!({
        "workers": list.len(),
        "names": list.iter().map(|w| &w.gpu_name).collect::<Vec<_>>(),
    }))
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<GatewayState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_worker(socket, state))
}

async fn handle_worker(socket: WebSocket, state: Arc<GatewayState>) {
    let (mut sender, mut receiver) = socket.split();

    let worker_id = uuid::Uuid::new_v4().to_string();
    tracing::info!("Worker {} connected", worker_id);

    // Send welcome
    let welcome = ServerMessage::Welcome {
        worker_id: worker_id.clone(),
    };
    if sender
        .send(Message::Text(serde_json::to_string(&welcome).unwrap()))
        .await
        .is_err()
    {
        return;
    }

    // Main loop: receive messages from worker
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                let Ok(worker_msg) = serde_json::from_str::<WorkerMessage>(&text) else {
                    tracing::warn!("Invalid message from {}: {}", worker_id, text);
                    continue;
                };

                match worker_msg {
                    WorkerMessage::Ready { gpu_name } => {
                        tracing::info!("Worker {} ready: {}", worker_id, gpu_name);
                        state.workers.write().await.insert(
                            worker_id.clone(),
                            Worker {
                                id: worker_id.clone(),
                                gpu_name,
                                busy: false,
                            },
                        );

                        // Try to assign work
                        let mut queue = state.work_queue.lock().await;
                        if let Some(work) = queue.pop() {
                            if let Ok(json) = serde_json::to_string(&work.message) {
                                let _ = sender.send(Message::Text(json)).await;
                                // Store the sender for result delivery
                                // (simplified — in production, use a proper work tracker)
                            }
                        } else {
                            let idle = ServerMessage::Idle;
                            let _ = sender
                                .send(Message::Text(
                                    serde_json::to_string(&idle).unwrap(),
                                ))
                                .await;
                        }
                    }

                    WorkerMessage::Result {
                        work_id,
                        energies,
                        compute_ms,
                    } => {
                        tracing::info!(
                            "Worker {} completed work {}: {} energies in {:.1}ms",
                            worker_id,
                            work_id,
                            energies.len(),
                            compute_ms
                        );
                        state.results.lock().await.insert(work_id, energies);

                        // Mark worker as not busy
                        if let Some(w) = state.workers.write().await.get_mut(&worker_id) {
                            w.busy = false;
                        }
                    }

                    WorkerMessage::Error { work_id, message } => {
                        tracing::error!(
                            "Worker {} error on {}: {}",
                            worker_id,
                            work_id,
                            message
                        );
                    }

                    WorkerMessage::Heartbeat => {
                        // Worker is alive, nothing to do
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    // Worker disconnected
    state.workers.write().await.remove(&worker_id);
    tracing::info!("Worker {} disconnected", worker_id);
}
