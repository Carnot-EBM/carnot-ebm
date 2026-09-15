//! Newline-delimited service fixture for REQ-VERIFY-7326 parity and cost checks.

use std::io::{self, BufRead, BufWriter, Write};

use carnot_constraints::{evaluate_schedule_batch, ScheduleRequest};
use serde::Deserialize;

#[derive(Deserialize)]
struct ServiceRequest {
    operation: String,
    #[serde(default)]
    requests: Vec<ScheduleRequest>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        let response = match serde_json::from_str::<ServiceRequest>(&line) {
            Ok(message) if message.operation == "ping" => serde_json::json!({"kind": "ready"}),
            Ok(message) if message.operation == "evaluate" => {
                serde_json::json!({"results": evaluate_schedule_batch(&message.requests)})
            }
            Ok(message) if message.operation == "shutdown" => {
                serde_json::to_writer(&mut stdout, &serde_json::json!({"kind": "shutdown"}))?;
                writeln!(&mut stdout)?;
                stdout.flush()?;
                break;
            }
            Ok(_) => serde_json::json!({"error": "unknown_operation"}),
            Err(_) => serde_json::json!({"error": "invalid_request"}),
        };
        serde_json::to_writer(&mut stdout, &response)?;
        writeln!(&mut stdout)?;
        stdout.flush()?;
    }
    Ok(())
}
