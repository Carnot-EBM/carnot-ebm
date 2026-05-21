#!/bin/bash

# Save current branch/commit
CURRENT_HEAD=$(git rev-parse HEAD)

while read commit; do
    echo "Testing commit $commit"
    git checkout $commit -- python/carnot/pipeline/
    
    # Check if pipeline is importable
    .venv/bin/python -c "import sys; sys.path.insert(0,'python'); from carnot.pipeline.verify_repair import VerifyRepairPipeline; print('ok')" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "COMMIT $commit NOT IMPORTABLE"
        continue
    fi
    
    # Run test
    OUT=$(.venv/bin/python bisect_test.py 20 42 2>/dev/null)
    # Extract delta
    DELTA=$(echo "$OUT" | grep "DELTA:" | cut -d':' -f2)
    echo "COMMIT $commit DELTA=$DELTA"
    
    # We are looking for first commit where delta > 0.05
    # Since we are iterating most recent to oldest, the first one with > 0.05 is the last working commit.
    # The regression commit is the one BEFORE it in our loop (i.e. more recent, the one right above it).
    
    # Wait, we can do the logic manually. Let's just print the results.
done < commits.txt

# Restore pipeline dir to HEAD
git checkout HEAD -- python/carnot/pipeline/
