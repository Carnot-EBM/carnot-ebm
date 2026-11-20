"""Reflink-dedupe identical large files under a root (XFS/btrfs). Dry run unless --apply.

Keeps each duplicate's own mode and mtime. Skips files under 1 MB or newer than 24 h.
Usage: python3 -I scripts/dedupe_reflink.py <root> [--apply]
"""
import hashlib, os, subprocess, sys, time, json
ROOT = sys.argv[1]; APPLY = "--apply" in sys.argv
MIN = 1 << 20; AGE = 24 * 3600; now = time.time()
by_size = {}
for d, _, fs in os.walk(ROOT):
    for f in fs:
        p = os.path.join(d, f)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if not os.path.isfile(p) or os.path.islink(p) or st.st_size < MIN or now - st.st_mtime < AGE:
            continue
        by_size.setdefault(st.st_size, []).append((p, st.st_ino, st.st_dev))
def h(p):
    s = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            s.update(b)
    return s.hexdigest()
saved = n = 0
for size, items in by_size.items():
    if len(items) < 2: continue
    groups = {}
    for p, ino, dev in items:
        groups.setdefault(h(p), []).append((p, ino))
    for digest, g in groups.items():
        if len(g) < 2: continue
        keep = g[0]
        for p, ino in g[1:]:
            if ino == keep[1]: continue
            n += 1; saved += size
            if APPLY:
                tmp = p + ".dedupe.tmp"
                st = os.stat(p)
                if h(p) != digest: continue
                subprocess.run(["cp", "--reflink=always", "--preserve=all", keep[0], tmp], check=True)
                os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
                os.chmod(tmp, st.st_mode & 0o7777)  # keep each copy's own mode (exec bit)
                os.replace(tmp, p)
print(json.dumps({"apply": APPLY, "files": n, "bytes": saved, "gib": round(saved / 2**30, 1)}))
