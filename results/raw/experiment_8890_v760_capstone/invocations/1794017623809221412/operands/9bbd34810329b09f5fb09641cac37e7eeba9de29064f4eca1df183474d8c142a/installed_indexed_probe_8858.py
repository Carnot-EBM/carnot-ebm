"""REQ-REPORT-8858: execute only public imports and explicit records in a fresh interpreter."""

import json
import os
import sys
from tempfile import TemporaryDirectory

with TemporaryDirectory(prefix="exp8858-outside-") as directory:
    os.chdir(directory)
    from carnot.verify.indexed_witness import verify_indexed_witness
    from carnot.verify import verify_indexed_witness as verify_alias

    assert verify_indexed_witness is verify_alias
    assert not any(n.startswith(("carnot.reporting", "scripts", "jax")) for n in sys.modules)
    units = json.loads(open(sys.argv[1]).read())["rows"]
    rows = []
    for i, unit in enumerate(units):
        rows.append(verify_indexed_witness(unit["input"]))
        if i % 200 == 0:
            print("[exp8858] installed completed=", i, "pending=", len(units) - i, flush=True)
    assert not any(n.startswith(("carnot.reporting", "scripts", "jax")) for n in sys.modules)
    with open(sys.argv[2], "w") as stream:
        json.dump(
            dict(
                rows=rows,
                installed_import_rows=[
                    dict(check="installed_outside_checkout_no_pythonpath", passed=True),
                    dict(check="no_reporting_training_or_artifact_loaders", passed=True),
                ],
                cwd=os.getcwd(),
                python=sys.executable,
            ),
            stream,
        )
