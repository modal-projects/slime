# retro_snapshot_bench — all-turns snapshot store measurements

Companion to the "All-Turns Snapshot Store" design (2026-08-26) and
`tests/test_agent/test_allturns_verification.py` (the CPU-side correctness
suite). This bench answers the design's open questions on real Modal
infrastructure with the PRODUCTION adapter code paths
(`agentic_rl/retro/backends/modal_snapshot.py`).

```bash
MODAL_ENVIRONMENT=junlin-dev uv run --with modal modal run \
    agentic_rl/profiles/retro_snapshot_bench/bench.py
```

CPU sandboxes only (~$0.2/run); all created images are deleted at the end.

## Verdicts (results/bench-20260827-060804.json, 75-turn episode, ~12.5 MiB workspace)

| Question | Verdict |
|---|---|
| Staged turn dirs exact? | ✅ 0 mismatches at 10 sampled turns — including forced same-size edits with the mtime restored (the rsync quick-check hazard; `--checksum` is load-bearing) |
| chmod-only turns propagate? | ✅ on GNU rsync (`--link-dest` refuses to link across a perms change). macOS openrsync links anyway — linux-only guarantee, which is what production runs |
| **Q1: hardlinks survive `snapshot_directory` → mount?** | ❌ **No.** 23.1× dedup in the source sandbox (672 MiB logical → 29 MiB) expands back to 672 MiB physical after restore |
| Tarball artifact instead? | ✅ `tar` stores hardlinks as link entries: round-trip exact, extracted dedup 23.09× (identical to source), artifact **11.1 MiB gzip / 10.9 MiB zstd** |
| Per-turn staging cost | rsync `--checksum --link-dest`: **p50 71–77 ms**, p95 ~81 ms → ~5.8 s per 75-turn episode (<1% of episode wall clock) |
| Snapshot latency | ~1.6–4 s, flat in size (29 MiB → 400 MiB); sequential p50 2.4 s, p95 2.6 s; 4 concurrent sandboxes: 0.8–4 s |
| Restore cost | dir-of-tarball image: mount+cp 0.9 s + `tar -x` 0.5 s + single-turn `cp -a` 0.13 s ≈ **1.5 s** (vs 5.8–9.5 s restoring the expanded staging dir) |
| GC | `image_delete` ~60 ms/image |

## Consequences for the implementation

1. The per-trajectory artifact is a **compressed tarball of the rsync-staged
   root** (gzip for tool-availability; zstd if present), snapshotted (or later
   volume-uploaded — the `SnapshotBackend` axis) as one object. Never snapshot
   the staging directory raw: the image pipeline expands hardlinks (~23× size).
2. `--checksum` stays mandatory in the staging rsync
   (`agentic_rl/retro/turns.py:staging_command`); the quick-check corruption it
   prevents was reproduced both locally and in-cloud.
3. Task images must be probed for `rsync`/`tar`/`gzip` at episode start
   (they are converter-built app images, not ours); fallback = ship a static
   rsync or a python3 stager into the sandbox at capture setup.
