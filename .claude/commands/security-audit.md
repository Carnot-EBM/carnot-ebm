Run the security auditor agent to scan the codebase for vulnerabilities and leaked secrets.

Read .claude/agents/security-auditor.md for instructions, then:

1. Scan for leaked secrets (API keys, tokens, private keys, passwords)
2. Verify SOPS compliance (no unencrypted secret files)
3. Check Rust safety (no unsafe/unwrap in library code)
4. Check Python safety (no eval/exec outside sandbox, no pickle)
5. Verify sandbox import blocklist is comprehensive
6. Check serialization uses only safetensors (not pickle)
7. Run cargo audit / pip-audit if available
8. Report verdict: CLEAN / WARNING / CRITICAL
