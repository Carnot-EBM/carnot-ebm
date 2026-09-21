# Run the vLLM log-probability probe on the scored server

Status: implemented locally; disabled by default; scored evidence pending.

Requirement: REQ-INFRA-7090.

The private probe received two T4 GPUs. It could not measure the Blackwell
server used by scoring. This story moves the probe onto the scored server.

The probe reuses the live loopback HTTP server. It does not manage that
server. It does not change its launch arguments. The scored path stays
unchanged while `RUN_LOGPROB_PROBE` is false.

The probe keeps the 45 REQ-INFRA-7089 fixtures and result fields. It records
the real server launch argv. It has a 240-second default budget. Failures
produce a `blocked_*` verdict and do not stop game play.

The module ships in the `carnot-agent-code` dataset. That dataset must be
re-versioned before an enabled run. The changed submission kernel must also be
pushed as a new kernel version. No dataset or kernel was uploaded here.

Local tests use fake HTTP servers for success, rejection, and timeout cases.
The Blackwell result remains pending until the operator enables the constant
and runs a scored submission.
