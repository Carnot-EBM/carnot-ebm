"""Regression probes for `.gitleaks.toml`.

Spec coverage: REQ-SEC-002

Origin: 2026-07-31 security audit re-baseline.

A secret-scanner config is a guard, and this project's own QA-Layer Authenticity
Discipline says the question to ask a guard is inverted: not "is this logic
right" but "name a real input this is SUPPOSED to catch and DOES NOT."  These
tests are that question, executed.

Two findings came out of writing them, both of which would otherwise have
shipped silently:

1. gitleaks 8.30.1 has NO rule for Anthropic (`sk-ant-`) or OpenAI project
   (`sk-proj-`) keys, and `generic-api-key` does not catch either.  In a repo
   developed with Claude Code that declares an `openai` extra, those were the
   two likeliest credentials to leak and the scanner was blind to both.
   `.gitleaks.toml` now defines custom rules; `test_anthropic_key_is_detected`
   and `test_openai_project_key_is_detected` hold that line.

2. An allowlist scoping benign value shapes to the artifact trees silently
   applied REPO-WIDE, because gitleaks ORs allowlist conditions and the
   documented fix (`matchCondition = "AND"`) no-ops in this build with no
   warning.  That would have blinded the scanner to a leaked Kaggle key, which
   is a 32-character lowercase hex string -- a credential this repo actually
   holds.  `test_hex_credential_in_source_tree_is_detected` is the probe that
   caught it and the one that must keep passing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from carnot.paths import repo_root

CONFIG = repo_root() / ".gitleaks.toml"

pytestmark = pytest.mark.skipif(
    shutil.which("gitleaks") is None,
    reason="gitleaks binary not installed",
)


def _scan(tmp_path: Path, files: dict[str, str]) -> dict[tuple[str, int], str]:
    """Write `files` into tmp_path, scan with the repo config, return findings.

    Returns {(relative_path, line_number): rule_id}.
    """
    for name, body in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    # POSITIVE CONTROL. Every scan carries a known-detectable secret, and the
    # helper asserts it was found. Without this, a scan that silently does
    # nothing returns {} and every "must be suppressed" assertion passes
    # vacuously -- which is exactly what happened on the first run of this file:
    # `gitleaks detect --source <path>` is a legacy form that this build accepts
    # and IGNORES, so it scanned nothing, and two suppression tests went green
    # on an empty result. The correct invocation is `gitleaks dir <path>`.
    control = "ghp_A9fK2mQ7zX4vB1nR6tY8wE3sD5gH0jL7pC2x"
    (tmp_path / "_positive_control.py").write_text(f'k = "{control}"\n', encoding="utf-8")

    config_copy = tmp_path / ".gitleaks.toml"
    shutil.copy(CONFIG, config_copy)
    report = tmp_path / "report.json"
    subprocess.run(
        [
            "gitleaks",
            "dir",
            str(tmp_path),
            "--config",
            str(config_copy),
            "--redact",
            "--log-level",
            "error",
            "--report-format",
            "json",
            "--report-path",
            str(report),
        ],
        capture_output=True,
        timeout=300,
        check=False,  # exit 1 simply means "leaks found", which is the normal case here
    )
    assert report.exists(), "gitleaks produced no report -- the scan did not run"

    def _rel(raw: str) -> str:
        """`gitleaks dir` reports absolute paths; the assertions use relative."""
        path = Path(raw)
        try:
            return str(path.relative_to(tmp_path))
        except ValueError:
            return raw

    findings = {
        (_rel(f["File"]), f["StartLine"]): f["RuleID"] for f in json.loads(report.read_text())
    }
    assert ("_positive_control.py", 1) in findings, (
        "positive control not detected -- the scan is not working, so any "
        "'suppressed' assertion in this test would be meaningless"
    )
    findings.pop(("_positive_control.py", 1))
    return findings


class TestRealSecretsAreDetected:
    """REQ-SEC-002: the allowlists must not blind the scanner to real credentials."""

    def test_anthropic_key_is_detected(self, tmp_path: Path) -> None:
        """gitleaks 8.30.1 ships no Anthropic rule; .gitleaks.toml adds one."""
        hits = _scan(
            tmp_path,
            {
                "probe.py": (
                    'k = "sk-ant-api03-9fK2mQ7zX4vB1nR6tY8wE3sD5gH0jL7pC2xA9fK2mQ7zX4vB1nR-abcdAA"\n'
                )
            },
        )
        assert ("probe.py", 1) in hits, "Anthropic API key not detected"

    def test_openai_project_key_is_detected(self, tmp_path: Path) -> None:
        """Also absent from the stock ruleset."""
        hits = _scan(
            tmp_path,
            {
                "probe.py": (
                    'k = "sk-proj-9fK2mQ7zX4vB1nR6tY8wE3sD5gH0jL7pC2xA9fK2mQ7zX4vB1nR6tY8wE3sD"\n'
                )
            },
        )
        assert ("probe.py", 1) in hits, "OpenAI project key not detected"

    def test_hex_credential_in_source_tree_is_detected(self, tmp_path: Path) -> None:
        """The probe that rejected an over-broad allowlist. Do not weaken it.

        A Kaggle API key is 32 lowercase hex characters, and this repo holds
        Kaggle credentials. An allowlist suppressing that shape to de-noise the
        artifact trees leaked repo-wide (gitleaks ORs allowlist conditions and
        `matchCondition = "AND"` no-ops here), so this exact input stopped being
        reported. If this test fails, a value-shape allowlist has been
        reintroduced without path scoping that actually works.
        """
        # `secret = ` matters: gitleaks' generic-api-key rule requires a keyword
        # (key/secret/token/api) near the value, so a bare `k = "..."` is not
        # detected by the stock rule either and would make this probe test
        # nothing. That is a property of the rule, not of the allowlists.
        hits = _scan(
            tmp_path,
            {"python/secret.py": 'secret = "9f8e7d6c5b4a39281706f5e4d3c2b1a0"\n'},
        )
        assert ("python/secret.py", 1) in hits, (
            "a 32-hex credential in a source tree is no longer reported -- an "
            "allowlist is over-suppressing; see .gitleaks.toml's ATTEMPTED AND "
            "REJECTED note"
        )

    def test_private_key_is_detected(self, tmp_path: Path) -> None:
        hits = _scan(
            tmp_path,
            {
                "results/leak.pem": (
                    "-----BEGIN RSA PRIVATE KEY-----\n"
                    "MIIEowIBAAKCAQEA3Tz2mr7SZiAMfQyuvBjM9Oi9hwXnTuxx\n"
                    "-----END RSA PRIVATE KEY-----\n"
                )
            },
        )
        assert any(f == "results/leak.pem" for f, _ in hits), "private key not detected"


class TestBenignClassesAreSuppressed:
    """REQ-SEC-002: the 378-finding cache-key class stays quiet."""

    def test_detector_cache_content_hashes_are_allowlisted(self, tmp_path: Path) -> None:
        """These files are wholly cache data: {"key": "<md5-length hex>", ...}.

        The literal field name "key" beside a high-entropy hex value is the
        textbook generic-api-key false positive, and it accounted for 378 of the
        544 findings in the full-history scan.
        """
        hits = _scan(
            tmp_path,
            {
                "results/_cache_detector_ar_length.jsonl": '{"key": "d41d8cd98f00b204e9800998ecf8427e", "y": 1, "ar": 3.27}\n',
                "results/_detector_items_manifest.jsonl": '{"key": "d41d8cd98f00b204e9800998ecf8427e", "y": 0, "text": "x"}\n',
            },
        )
        assert hits == {}, f"detector cache files should be allowlisted, got {hits}"

    def test_direct_baseline_blob_keys_are_allowlisted(self, tmp_path: Path) -> None:
        """REQ-SEC-002: a HuggingFace cache blob key is a content hash, not a secret.

        The direct-baseline artifacts record model provenance as
        `blob_key: <sha256>`, which is the same value they store as
        `trusted_sha256`. Whether it trips the entropy floor is luck: Exp6605's
        key clears it and Exp6607's does not, from one code path on one day.
        """

        body = (
            '{"cache_provenance": {"blob_key": '
            '"34c746b1d50ab813e29cd46c4796e3f43c741901a582f93a67b55b9fc9687b35"}}\n'
        )
        hits = _scan(
            tmp_path,
            {
                "results/experiment_6607_gemma4_26b_direct_headroom.json": body,
                "results/experiment_6605_qwen36_direct_headroom.json": body,
            },
        )
        assert hits == {}, f"direct-baseline blob keys should be allowlisted, got {hits}"

    def test_direct_baseline_allowlist_does_not_hide_a_real_key(self, tmp_path: Path) -> None:
        """REQ-SEC-002: the allowlist exempts one rule, not the whole file.

        A path-scoped allowlist is only safe while a real provider key landing
        in the same file still trips its own rule.
        """

        hits = _scan(
            tmp_path,
            {
                "results/experiment_6607_gemma4_26b_direct_headroom.json": (
                    '{"note": "ghp_A9fK2mQ7zX4vB1nR6tY8wE3sD5gH0jL7pC2x"}\n'
                )
            },
        )
        assert any(
            name == "results/experiment_6607_gemma4_26b_direct_headroom.json" for name, _ in hits
        ), f"a real token inside an allowlisted artifact must still be reported, got {hits}"

    def test_sk_ant_test_fixture_is_not_a_false_positive(self, tmp_path: Path) -> None:
        """`sk-ant-access-secret` is a real fixture in test_agent_plan_usage.py.

        The custom Anthropic rule's length floor exists to clear it. An earlier
        floor of {80,} was too tight and missed a real key; {32,} catches real
        keys while leaving this 6-character body alone.
        """
        hits = _scan(tmp_path, {"python/fixture.py": 'assert t == "sk-ant-access-secret"\n'})
        assert hits == {}, f"test fixture became a false positive: {hits}"
