//! JSON-lines caller for the shared durable recalibration core.
//!
//! Spec: REQ-REPORT-7626 and SCENARIO-REPORT-7626-PARITY.

fn main() {
    if let Err(error) = carnot_core::portable_recalibration::run_stdio() {
        eprintln!("portable-recalibration-service: {error}");
        std::process::exit(2);
    }
}
