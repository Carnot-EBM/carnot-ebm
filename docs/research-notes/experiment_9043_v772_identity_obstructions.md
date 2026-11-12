# Exp9043: retained source identity obstructions

Date: 2026-11-12. This audit explains Exp9029's retained metadata. It makes no
human-pilot or verifier-performance claim. It performs no network request,
annotation reduction or model load. Production parsers and closure gates stay fixed.

| Exclusive cause | Release identities | Historical identities | Total |
| --- | ---: | ---: | ---: |
| Malformed or missing receipt | 0 | 0 | 0 |
| Alias or parser omission | 0 | 0 | 0 |
| Missing page | 0 | 206 | 206 |
| Existing page without QID | 2 | 7 | 9 |
| Unresolved other | 0 | 0 | 0 |
| Total | 2 | 213 | 215 |

The two release identities are the Persian template
`الگو:میزان سواد در استان‌های ایران (۱۴۰۰)` and list
`فهرست پارک‌های ملی ایران`. Their retained local pageids are 492654 and 533231.
Their namespaces are 10 and 0. Neither page has a `wikibase_item` property in
the retained response. A local pageid does not establish cross-language identity.

All 98 original requests remain in the evidence, including the original HTTP429.
The final observations for all 215 unresolved identities have HTTP200 status.
Successful transport therefore does not resolve these obstructions. The audit
retains each request URL and hash, every original response hash, all observations,
title aliases, markers, namespace, pageid presence and original parser result.
Raw evidence includes copies of all response bodies and primitive per-unit rows.
No prompt, generated response or annotation value is printed.

The [Pageprops API](https://www.mediawiki.org/wiki/API:Pageprops) returns page
properties. This audit tests for the requested property in each actual response.
It does not infer a QID from a successful request or a local page identifier.

No retained QID was lost by the original parser. No production patch is proposed.
A private converted-title control demonstrates how a real omission would be
identified. That control is not a measurement. The artifact records the parser's
callers and the evidence needed before proposing a repair.

Each category needs additional evidence before source eligibility can change:

- Missing receipts need complete original request and response bytes with
  independently authenticated hashes and the matching author release revision.
- Parser omissions need a valid retained QID and a proven alias link. A private
  regression must precede any proposed production change. All joins must then close.
- Missing pages need author-confirmed original URLs and revisions, with move or
  deletion history and authenticated cross-language identity for every exposure.
- Existing pages without QIDs need author-confirmed source revisions and authenticated
  sitelinks or a complete equivalent source mapping. Such mapping is evidence for
  investigation; it does not waive the existing QID closure criterion.
- Other failures need the specific API error, invalid title, ambiguous alias or
  invalid QID resolved through authenticated author metadata.

Replaying identical retained bytes cannot change any category. An identical live
request can differ only if remote metadata or transport changes. A paced retry alone
does not supply the missing author provenance or cross-language exposure evidence.
There is no diagnosed parser repair to fund or schedule from this audit.

The independent historical authority is the preserved V771 design and queue, bound
to full-task digest `a353be3a3351cc47a11d5c93a3a36a057653d9851e17da6896e33a7737e9ce1c`.
The audit does not read a successor roadmap as historical authority.

`identity_diagnosis_ready_score=1` means complete classification and passing current
scoped checks. `source_identity_ready_score=0` preserves Exp9029's strict result:
two release groups and 213 historical identities remain unresolved. Failed broad-suite
history remains failed under `repository_health`; it is not a current scoped pass.

Reopening requires independently authenticated evidence for every unresolved identity,
followed by a complete replay under the original strict criterion. Preserve Exp9029's
primary and rejected evidence. Check a proposed replacement candidate in private
storage. The following criterion currently fails for the original primary:

```bash
candidate=results/experiment_9029_v771_source_identity_resolution.json
PYTHONUNBUFFERED=1 .venv/bin/python -u \
  scripts/experiments/experiment_9029_v771_source_identity_resolution.py \
  --cold-replay "$candidate" && .venv/bin/python - "$candidate" <<'PY'
import json
import sys
from pathlib import Path
value = json.loads(Path(sys.argv[1]).read_bytes())
gates = {g['check']: g for g in value['gate_check_summary']}
expected = {'release_identity_closure': 0, 'historical_identity_closure': 0,
            'unknown_exposure': [],
            'all_fourteen_languages': ['ar', 'ca', 'cs', 'de', 'en', 'es', 'eu',
                                       'fa', 'fi', 'fr', 'hi', 'it', 'sv', 'zh']}
assert value['required_checks_passed'] is True
assert value['source_identity_ready_score'] == 1
assert all(gates[k]['passed'] and gates[k]['expected'] == v
           and gates[k]['observed'] == v for k, v in expected.items())
assert value['unresolved_request_count'] == 0
PY
```

The actual Exp9043 primary and its bound validator sidecar provide the measured
coverage and current validation receipts. No downstream dependency, external
publication or activation follows from diagnosis readiness.
