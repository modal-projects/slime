"""All-turns snapshot store: Modal measurement bench (design doc Q1/Q6 + stress).

Answers, with the PRODUCTION adapter code paths (backends/modal_snapshot.py):

  1. Correctness in-cloud (GNU rsync): staged per-turn dirs match the live
     workspace exactly across 75 turns of agent-like mutations — including the
     two hazards the local suite found (same-size same-mtime edits → needs
     --checksum; chmod-only changes → GNU rsync --link-dest must break the link).
  2. Q1 hardlink survival: does the staging root's hardlink dedup survive
     snapshot_directory → mount_image → restore? (physical du vs logical size,
     before and after the round-trip)
  3. Latency: per-turn rsync cost; snapshot_directory at ~25 / ~100 / ~400 MiB;
     restore (mount + cp) cost; image deletion cost.
  4. Stress: 10 sequential snapshots (p50/p95) + 4 concurrent sandboxes
     snapshotting simultaneously.

Run (CPU sandboxes only, ~$0.2):

    MODAL_ENVIRONMENT=junlin-dev uv run --with modal modal run \
        agentic_rl/profiles/retro_snapshot_bench/bench.py

Writes results/bench-<UTC date>.json next to this file.
"""

from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agentic_rl.retro.backends.modal_snapshot import (  # noqa: E402
    delete_snapshot,
    restore_directory,
    snapshot_sandbox,
)
from agentic_rl.retro.turns import staging_command  # noqa: E402

app = modal.App("retro-snapshot-bench")
IMAGE = modal.Image.debian_slim().apt_install("rsync", "zstd")

TURNS = 75
SAMPLE_TURNS = 10  # per-turn dirs fully re-verified against the live workspace

# The in-sandbox driver replicates turns.staging_command's rsync invocation; this
# assertion makes any flag drift break the bench loudly instead of silently.
_CMD = staging_command("/SRC", "/ROOT", 1, prev_turn_index=0)
assert "rsync -a --checksum --link-dest=" in _CMD, _CMD

_DRIVER = r'''
import hashlib, json, os, random, statistics, subprocess, time

APP, ROOT = "/app", "/staging"
TURNS = int(os.environ["BENCH_TURNS"])
SAMPLES = int(os.environ["BENCH_SAMPLES"])
rng = random.Random(20260827)


def sh(cmd):
    subprocess.run(["bash", "-lc", cmd], check=True, capture_output=True, text=True)


def tree_digest(root):
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            mode = oct(os.stat(path).st_mode & 0o777)
            entries.append(f"{rel}\x00{digest}\x00{mode}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def usage(root):
    logical = 0
    seen = set()
    physical = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            st = os.stat(os.path.join(dirpath, name))
            logical += st.st_size
            if st.st_ino not in seen:
                seen.add(st.st_ino)
                physical += st.st_size
    return {"logical_bytes": logical, "physical_bytes": physical}


# ---- build a ~10 MiB agent-like workspace: 180 text + 20 binary files ----
os.makedirs(APP)
os.makedirs(ROOT)
paths = []
for i in range(180):
    p = os.path.join(APP, f"src/mod_{i:03d}.py" if i % 3 else f"src/pkg{i % 7}/mod_{i:03d}.py")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("".join(f"line {j} of file {i}: {rng.random()}\n" for j in range(rng.randint(300, 900))))
    paths.append(p)
for i in range(20):
    p = os.path.join(APP, f"assets/blob_{i:02d}.bin")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(rng.randbytes(64 * 1024))
    paths.append(p)

expected = {}   # sampled turn -> live workspace digest at that turn
rsync_s = []
prev = None
chmod_turns = []
for t in range(TURNS):
    # agent-like mutations: 5 SAME-SIZE rewrites with the mtime RESTORED to its
    # pre-write value — the exact rsync quick-check hazard; only --checksum can
    # see these — plus 1 in-place append, 1 new ~100 KiB file, periodic
    # delete / chmod-only turns.
    live = [p for p in paths if os.path.exists(p) and not p.endswith(".bin")]
    rewritten = set(rng.sample(live, 5))
    for p in rewritten:
        st = os.stat(p)
        with open(p, "w") as f:
            body = f"# turn {t}\n"
            f.write(body + "x" * max(0, st.st_size - len(body)))
        os.utime(p, (st.st_atime, st.st_mtime))
    with open(rng.choice(live), "a") as f:
        f.write(f"appended at turn {t}\n")
    newp = os.path.join(APP, f"out/turn_{t:03d}.dat")
    os.makedirs(os.path.dirname(newp), exist_ok=True)
    with open(newp, "wb") as f:
        f.write(rng.randbytes(100 * 1024))
    paths.append(newp)
    if t % 10 == 9:
        victim = rng.choice([p for p in live if os.path.getsize(p) < 40_000])
        os.remove(victim)
    if t % 15 == 14:  # chmod-only turn: this file must have NO content change
        candidates = [p for p in live if os.path.exists(p) and p not in rewritten]
        target = rng.choice(candidates)
        os.chmod(target, 0o755)
        chmod_turns.append({"turn": t, "path": os.path.relpath(target, APP)})

    dest = f"{ROOT}/{t:04d}"
    link = f" --link-dest={ROOT}/{prev:04d}/" if prev is not None else ""
    started = time.perf_counter()
    sh(f"mkdir -p {dest} && rsync -a --checksum{link} {APP}/ {dest}/")
    rsync_s.append(time.perf_counter() - started)
    prev = t
    if t % (TURNS // SAMPLES) == 0 or t == TURNS - 1:
        expected[t] = tree_digest(APP)

# ---- correctness: sampled staged turns == live workspace at that turn ----
mismatches = [t for t, digest in expected.items() if tree_digest(f"{ROOT}/{t:04d}") != digest]

# ---- chmod-only propagation (GNU rsync --link-dest must not link across it) ----
chmod_ok = all(
    (os.stat(f"{ROOT}/{c['turn']:04d}/{c['path']}").st_mode & 0o777) == 0o755
    for c in chmod_turns
)

# manifests travel with the image for restore-side verification
os.makedirs(f"{ROOT}/manifests", exist_ok=True)
with open(f"{ROOT}/manifests/expected.json", "w") as f:
    json.dump(expected, f)

report = {
    "turns": TURNS,
    "rsync_seconds": {
        "p50": statistics.median(rsync_s),
        "p95": sorted(rsync_s)[int(0.95 * len(rsync_s))],
        "max": max(rsync_s),
        "total": sum(rsync_s),
    },
    "app_usage": usage(APP),
    "staging_usage": usage(ROOT),
    "sampled_turns": sorted(expected),
    "staged_mismatches": mismatches,
    "chmod_turns": len(chmod_turns),
    "chmod_only_propagated": chmod_ok,
}
print("BENCH_JSON " + json.dumps(report))
'''

_VERIFIER = r'''
import hashlib, json, os

ROOT = os.environ["BENCH_RESTORED_ROOT"]


def tree_digest(root):
    entries = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, root)
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            mode = oct(os.stat(path).st_mode & 0o777)
            entries.append(f"{rel}\x00{digest}\x00{mode}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def usage(root):
    logical = 0
    seen = set()
    physical = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            st = os.stat(os.path.join(dirpath, name))
            logical += st.st_size
            if st.st_ino not in seen:
                seen.add(st.st_ino)
                physical += st.st_size
    return {"logical_bytes": logical, "physical_bytes": physical}


with open(f"{ROOT}/manifests/expected.json") as f:
    expected = json.load(f)
mismatches = [t for t, digest in expected.items() if tree_digest(f"{ROOT}/{int(t):04d}") != digest]
print("BENCH_JSON " + json.dumps({"restored_mismatches": mismatches, "restored_usage": usage(ROOT)}))
'''


def _exec(sb, cmd: str, timeout: int = 1800, env: dict[str, str] | None = None) -> str:
    exports = "".join(f"export {k}={v}; " for k, v in (env or {}).items())
    process = sb.exec("bash", "-lc", exports + cmd, timeout=timeout)
    stdout = process.stdout.read()
    stderr = process.stderr.read()
    if process.wait() != 0:
        raise RuntimeError(f"sandbox exec failed: {cmd[:120]} …\n{(stderr or stdout)[-1500:]}")
    return stdout


def _exec_json(sb, script: str, env: dict[str, str], timeout: int = 1800) -> dict:
    _exec(sb, f"cat > /bench_script.py <<'PYEOF'\n{script}\nPYEOF", timeout=60)
    out = _exec(sb, "python3 /bench_script.py", timeout=timeout, env=env)
    for line in out.splitlines():
        if line.startswith("BENCH_JSON "):
            return json.loads(line[len("BENCH_JSON ") :])
    raise RuntimeError(f"driver produced no BENCH_JSON:\n{out[-1500:]}")


def _snapshot(sb, path: str, label: str, timings: list, ids: list[str]) -> str:
    result = snapshot_sandbox(sb, path=path, timeout_seconds=300)
    timings.append(
        {
            "label": label,
            "seconds": round(result.latency_seconds, 3),
            "physical_bytes": result.estimated_bytes,
            "files": result.estimated_files,
            "snapshot_id": result.snapshot_id,
        }
    )
    ids.append(result.snapshot_id)
    return result.snapshot_id


def _make_sandbox():
    return modal.Sandbox.create(
        "sleep", "infinity", app=app, image=IMAGE, cpu=4.0, memory=4096, timeout=3600
    )


@app.local_entrypoint()
def main():
    report: dict = {"utc": datetime.now(timezone.utc).isoformat(), "turns": TURNS}
    snapshot_timings: list = []
    snapshot_ids: list[str] = []
    sandboxes = []
    try:
        # ---- phase 1: 75-turn staged episode + in-cloud correctness ----
        print("phase 1: 75-turn staging episode (GNU rsync) …")
        sb_a = _make_sandbox()
        sandboxes.append(sb_a)
        report["driver"] = _exec_json(
            sb_a, _DRIVER, env={"BENCH_TURNS": str(TURNS), "BENCH_SAMPLES": str(SAMPLE_TURNS)}
        )
        assert report["driver"]["staged_mismatches"] == [], report["driver"]
        assert report["driver"]["chmod_only_propagated"] is True, report["driver"]
        print(f"  staged state exact at {len(report['driver']['sampled_turns'])} sampled turns; "
              f"chmod-only propagated; rsync p50 {report['driver']['rsync_seconds']['p50']*1e3:.0f} ms")

        # ---- phase 2: snapshot the staging root (the ONE image per trajectory) ----
        print("phase 2: snapshot_directory latency + size curve …")
        main_id = _snapshot(sb_a, "/staging", "staging-root", snapshot_timings, snapshot_ids)
        for label, mib in (("pad-100MiB", 100), ("pad-400MiB", 400)):
            _exec(sb_a, f"mkdir -p /{label} && head -c {mib}M /dev/urandom | split -b 4M - /{label}/blob_")
            _snapshot(sb_a, f"/{label}", label, snapshot_timings, snapshot_ids)

        # ---- phase 3: sequential stress (10 snapshots of the same root) ----
        print("phase 3: 10 sequential snapshots …")
        sequential = []
        for _ in range(10):
            _snapshot(sb_a, "/staging", "sequential", snapshot_timings, snapshot_ids)
            sequential.append(snapshot_timings[-1]["seconds"])
        report["sequential_seconds"] = {
            "p50": statistics.median(sequential),
            "p95": sorted(sequential)[int(0.95 * len(sequential))],
        }

        # ---- phase 4: restore round-trip + Q1 hardlink survival ----
        print("phase 4: restore round-trip in a second sandbox …")
        sb_b = _make_sandbox()
        sandboxes.append(sb_b)
        restore = restore_directory(sb_b, main_id, target_path="/restored")
        report["restore_seconds"] = round(restore.latency_seconds, 3)
        report["verifier"] = _exec_json(sb_b, _VERIFIER, env={"BENCH_RESTORED_ROOT": "/restored"})
        assert report["verifier"]["restored_mismatches"] == [], report["verifier"]
        # single-turn branch restore cost (what a branch sibling actually pays)
        started = time.perf_counter()
        _exec(sb_b, "rm -rf /app_branch && mkdir -p /app_branch && cp -a /restored/0040/. /app_branch/")
        report["branch_copy_seconds"] = round(time.perf_counter() - started, 3)

        stage_use = report["driver"]["staging_usage"]
        rest_use = report["verifier"]["restored_usage"]
        report["q1_hardlinks"] = {
            "staging_dedup_x": round(stage_use["logical_bytes"] / max(1, stage_use["physical_bytes"]), 2),
            "restored_dedup_x": round(rest_use["logical_bytes"] / max(1, rest_use["physical_bytes"]), 2),
            "survived": rest_use["physical_bytes"] < rest_use["logical_bytes"] / 3,
        }
        print(f"  restored exact; dedup staging {report['q1_hardlinks']['staging_dedup_x']}x → "
              f"restored {report['q1_hardlinks']['restored_dedup_x']}x")

        # ---- phase 4b: tarball artifact — dedup that survives ANY pipeline ----
        # tar stores hardlinks as link entries and compresses, so the trajectory
        # artifact is one opaque file the image layer cannot expand.
        print("phase 4b: tarball round-trip (gzip vs zstd) …")
        tarballs = {}
        for label, make, ext in (
            ("gzip", "tar -czf", "tgz"),
            ("zstd", "tar --zstd -cf", "tzst"),
        ):
            started = time.perf_counter()
            _exec(sb_a, f"mkdir -p /snap && {make} /snap/staging.{ext} -C / staging")
            seconds = round(time.perf_counter() - started, 3)
            size = int(_exec(sb_a, f"stat -c %s /snap/staging.{ext}").strip())
            tarballs[label] = {"pack_seconds": seconds, "bytes": size}
        report["tarballs"] = tarballs
        snap_tar = _snapshot(sb_a, "/snap", "tarball-dir", snapshot_timings, snapshot_ids)

        restore_tar = restore_directory(sb_b, snap_tar, target_path="/snap2")
        started = time.perf_counter()
        _exec(sb_b, "mkdir -p /rt && tar -xf /snap2/staging.tzst -C /rt")
        unpack_seconds = round(time.perf_counter() - started, 3)
        report["tarball_verifier"] = _exec_json(
            sb_b, _VERIFIER, env={"BENCH_RESTORED_ROOT": "/rt/staging"}
        )
        assert report["tarball_verifier"]["restored_mismatches"] == [], report["tarball_verifier"]
        started = time.perf_counter()
        _exec(sb_b, "rm -rf /app_branch2 && mkdir -p /app_branch2 && cp -a /rt/staging/0040/. /app_branch2/")
        tar_use = report["tarball_verifier"]["restored_usage"]
        report["tarball_roundtrip"] = {
            "restore_seconds": round(restore_tar.latency_seconds, 3),
            "unpack_seconds": unpack_seconds,
            "branch_copy_seconds": round(time.perf_counter() - started, 3),
            "extracted_dedup_x": round(tar_use["logical_bytes"] / max(1, tar_use["physical_bytes"]), 2),
        }
        print(f"  tarball exact; gz {tarballs['gzip']['bytes']/2**20:.1f} MiB / "
              f"zstd {tarballs['zstd']['bytes']/2**20:.1f} MiB; extracted dedup "
              f"{report['tarball_roundtrip']['extracted_dedup_x']}x")

        # ---- phase 5: concurrent snapshot stress (4 sandboxes at once) ----
        print("phase 5: 4 concurrent sandboxes snapshotting …")
        concurrent: list = []
        errors: list = []

        def one(index: int):
            try:
                sb = _make_sandbox()
                sandboxes.append(sb)
                _exec(sb, "mkdir -p /w && head -c 25M /dev/urandom | split -b 1M - /w/blob_")
                result = snapshot_sandbox(sb, path="/w", timeout_seconds=300)
                snapshot_ids.append(result.snapshot_id)
                concurrent.append(round(result.latency_seconds, 3))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"worker {index}: {exc}")

        threads = [threading.Thread(target=one, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors
        report["concurrent_seconds"] = sorted(concurrent)

        # ---- cleanup: production GC path, timed ----
        print(f"cleanup: deleting {len(snapshot_ids)} images via delete_snapshot …")
        started = time.perf_counter()
        for snapshot_id in snapshot_ids:
            delete_snapshot(snapshot_id)
        report["gc_seconds_total"] = round(time.perf_counter() - started, 3)
        report["gc_images"] = len(snapshot_ids)
        report["snapshots"] = snapshot_timings
    finally:
        for sb in sandboxes:
            try:
                sb.terminate()
            except Exception:  # noqa: BLE001
                pass

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"bench-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")
