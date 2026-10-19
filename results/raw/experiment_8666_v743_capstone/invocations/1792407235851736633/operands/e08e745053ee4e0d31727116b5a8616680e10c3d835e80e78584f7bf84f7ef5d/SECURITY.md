# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in Carnot, please report it responsibly. **Do not open a public issue.**

### Preferred methods

1. **GitHub Security Advisories**: Use the "Report a vulnerability" button on the [Security tab](https://github.com/Carnot-EBM/carnot-ebm/security/advisories) of this repository.
2. **Email**: Send details to security@carnot-ebm.org.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Any potential mitigations you have identified

### What to expect

- **Acknowledgement**: Within 48 hours of your report.
- **Assessment**: We will evaluate the severity and impact within 7 days.
- **Fix timeline**: Critical vulnerabilities will be patched as quickly as possible, typically within 14 days. Lower-severity issues will be addressed in the next scheduled release.
- **Disclosure**: We follow coordinated disclosure. We will work with you on an appropriate disclosure timeline and credit you in the advisory unless you prefer to remain anonymous.

## Sandbox Security Model

Carnot's autoresearch pipeline executes untrusted code in sandboxed environments:

- **Production**: Firecracker microVMs provide strong isolation for autonomous research workloads.
- **Development/CI**: gVisor (via Docker's `runsc` runtime) provides container-level sandboxing.

If you discover a sandbox escape or isolation bypass, please treat it as a critical vulnerability and report it through the channels above.
