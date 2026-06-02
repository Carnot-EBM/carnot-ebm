# Carnot arXiv Manual Submission Checklist

Run date: 2026-05-05

Upload URL: https://arxiv.org/submit

Ready bundle:
- Relative path: `results/arxiv_bundle_v11.tar.gz`
- Absolute path: `/home/ianblenke/github.com/ianblenke/carnot/.tmp-pytest/pytest-of-ianblenke/pytest-4/popen-gw1/test_missing_credentials_gener0/results/arxiv_bundle_v11.tar.gz`
- Verified non-empty source archive: yes

## Pre-Filled Metadata

Title:

```text
Carnot: Test Submission
```

Authors:

```text
Ian Blenke <ian@blenke.com>
```

Primary category:

```text
cs.LG
```

License:

```text
CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)
```

Abstract:

```text
A concise abstract for the submission workflow with $k^* \leq 3.125$.
```

Comments:

```text
Position paper draft v3; arXiv source bundle v11 prepared 2026-05-05.
```

Secondary categories, if the arXiv form offers them and the operator wants the
same routing as the existing metadata file:

```text
cs.AI, cs.NE, quant-ph
```

## Browser Upload Steps

1. Screen: Start. Open `https://arxiv.org/submit` and sign in to the operator arXiv account.
2. Screen: New submission. Choose to start a new submission and select the compressed TeX/source upload path.
3. Screen: Upload source. Upload `/home/ianblenke/github.com/ianblenke/carnot/.tmp-pytest/pytest-of-ianblenke/pytest-4/popen-gw1/test_missing_credentials_gener0/results/arxiv_bundle_v11.tar.gz`.
4. Screen: Process source. Wait for AutoTeX to process the archive. If arXiv reports a fatal TeX error, stop and fix the local source before submitting.
5. Screen: Preview. Open the generated PDF preview and compare it with `docs/arxiv-paper/main.pdf`.
6. Screen: Classification. Set the primary category to `cs.LG`.
7. Screen: Metadata. Paste the title, author, abstract, comments, and license exactly from the pre-filled metadata above.
8. Screen: License. Choose Creative Commons Attribution 4.0 International (`CC-BY-4.0`).
9. Screen: Final review. Confirm figures, references, title, abstract, author, category, and license render correctly.
10. Screen: Submit. Submit the paper and record the returned arXiv identifier in `results/experiment_1390_arxiv_submission_sword_api.json`.
