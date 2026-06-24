/**
 * Bun server for the SWE rollout-dump dashboard.
 *
 * Serves the bundled frontend (public/index.html) plus a JSON API over the
 * slime-checkpoints volume mounted at DUMP_ROOT:
 *
 *   GET /api/runs                          run dirs (+ legacy loose dumps)
 *   GET /api/rollouts?run=<name>           .pt files inside one run
 *   GET /api/rollout?run=<name>&file=<f>   converted dump (JSON view-model)
 *   GET /api/run-summary?run=<n>&budget=<k>  per-dump stats; converts up to k
 *                                            missing summaries (newest first)
 *                                            and reports how many remain so the
 *                                            frontend can poll until 0
 *
 * .pt -> JSON conversion shells out to convert.py (torch pickles aren't
 * JS-parseable), cached by file mtime+size+converter version. Summaries are
 * tiny sidecar JSONs: derived in-process from an already-converted full view
 * when possible, else via `convert.py --summary` (skips turn parsing).
 */

import index from "./public/index.html";
import { mkdirSync, readdirSync, renameSync, statSync } from "node:fs";
import { join } from "node:path";

// Bun emits HTML-import chunk URLs relative to process.cwd() (e.g. cwd=/root
// yields src="/../../root/chunk-x.js"). Pin cwd to public/ so asset URLs stay
// root-relative regardless of launch dir.
process.chdir(new URL("./public/", import.meta.url).pathname);

const DUMP_ROOT = process.env.DUMP_ROOT ?? "/vol/swe_rollout_dumps";
const PORT = Number(process.env.PORT ?? 3000);
const CACHE_DIR = process.env.CACHE_DIR ?? "/tmp/rollout-json-cache";
const PYTHON_BIN = process.env.PYTHON_BIN ?? "python3";
const CONVERT_PY = new URL("./convert.py", import.meta.url).pathname;

// Pseudo-run for legacy dumps sitting directly in DUMP_ROOT (pre run-tag).
const ROOT_RUN = "(root)";

// Bump when convert.py's output shape changes so stale caches re-convert.
const CONVERT_VERSION = "v8"; // v8: read env diagnostics (is_solved/timing/...) from metadata.agentic

mkdirSync(CACHE_DIR, { recursive: true });

type RunInfo = { name: string; fileCount: number; latestMtime: number | null };
type RolloutFile = { file: string; sizeBytes: number; mtime: number };
type BucketStats = {
  n: number;
  solved: number;
  aborted: number;
  truncated: number;
  mean_reward: number | null;
};
type DumpSummary = BucketStats & {
  rollout_id?: number | string;
  datasets: Record<string, BucketStats> | null;
  instances: {
    id: string;
    dataset: string | null;
    reward: number | null;
    solved: boolean;
    status: string | null;
  }[];
};

function safeName(name: string): boolean {
  return name.length > 0 && !name.includes("/") && !name.includes("..") && !name.startsWith(".");
}

function runDir(run: string): string {
  return run === ROOT_RUN ? DUMP_ROOT : join(DUMP_ROOT, run);
}

function listPtFiles(dir: string): RolloutFile[] {
  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return [];
  }
  const files: RolloutFile[] = [];
  for (const name of entries) {
    if (!name.endsWith(".pt")) continue;
    try {
      const st = statSync(join(dir, name));
      if (st.isFile()) files.push({ file: name, sizeBytes: st.size, mtime: st.mtimeMs });
    } catch {
      // file vanished between readdir and stat (volume reload) -- skip
    }
  }
  // eval dumps after train dumps, numeric id order within
  const key = (f: string) => {
    const m = f.match(/rollout_(eval_)?(\d+)\.pt$/);
    if (!m) return [2, 0, f] as const;
    return [m[1] ? 1 : 0, Number(m[2]), f] as const;
  };
  files.sort((a, b) => {
    const ka = key(a.file);
    const kb = key(b.file);
    return ka[0] - kb[0] || ka[1] - kb[1] || (ka[2] < kb[2] ? -1 : 1);
  });
  return files;
}

function listRuns(): RunInfo[] {
  let entries: string[];
  try {
    entries = readdirSync(DUMP_ROOT);
  } catch (e) {
    throw new Error(`cannot read DUMP_ROOT ${DUMP_ROOT}: ${e}`);
  }
  const runs: RunInfo[] = [];
  let rootFiles = 0;
  let rootLatest: number | null = null;
  for (const name of entries) {
    let st;
    try {
      st = statSync(join(DUMP_ROOT, name));
    } catch {
      continue;
    }
    if (st.isDirectory()) {
      const files = listPtFiles(join(DUMP_ROOT, name));
      runs.push({
        name,
        fileCount: files.length,
        latestMtime: files.length ? Math.max(...files.map((f) => f.mtime)) : null,
      });
    } else if (name.endsWith(".pt")) {
      rootFiles += 1;
      rootLatest = Math.max(rootLatest ?? 0, st.mtimeMs);
    }
  }
  if (rootFiles > 0) runs.push({ name: ROOT_RUN, fileCount: rootFiles, latestMtime: rootLatest });
  runs.sort((a, b) => (b.latestMtime ?? 0) - (a.latestMtime ?? 0));
  return runs;
}

// train rollout_12.pt -> {kind:"train", step:12}; rollout_eval_20.pt -> eval.
function fileKind(file: string): { kind: "train" | "eval"; step: number } | null {
  const m = file.match(/^rollout_(eval_)?(\d+)\.pt$/);
  if (!m) return null;
  return { kind: m[1] ? "eval" : "train", step: Number(m[2]) };
}

function cachePathFor(run: string, file: string, st: { mtimeMs: number; size: number }, suffix = ""): string {
  const key = `${run.replaceAll("/", "_")}__${file}__${Math.round(st.mtimeMs)}_${st.size}_${CONVERT_VERSION}${suffix}.json`;
  return join(CACHE_DIR, key);
}

// Dedup concurrent conversions of the same dump (e.g. two browser tabs).
const inflight = new Map<string, Promise<string>>();

// Spawn convert.py and atomically move its output into the cache.
function runConvert(ptPath: string, cachePath: string, summary: boolean): Promise<string> {
  const existing = inflight.get(cachePath);
  if (existing) return existing;

  const promise = (async () => {
    const tmpPath = `${cachePath}.tmp-${process.pid}-${Date.now()}`;
    const args = [PYTHON_BIN, CONVERT_PY, ...(summary ? ["--summary"] : []), ptPath, tmpPath];
    const proc = Bun.spawn(args, { stdout: "pipe", stderr: "pipe" });
    const exitCode = await proc.exited;
    if (exitCode !== 0) {
      const stderr = await new Response(proc.stderr).text();
      throw new Error(`convert.py exited ${exitCode}: ${stderr.slice(-2000)}`);
    }
    renameSync(tmpPath, cachePath);
    return cachePath;
  })().finally(() => inflight.delete(cachePath));
  inflight.set(cachePath, promise);
  return promise;
}

async function convertedJsonPath(run: string, file: string): Promise<string> {
  const ptPath = join(runDir(run), file);
  const st = statSync(ptPath); // throws -> 404 upstream
  const cachePath = cachePathFor(run, file, st);
  if (await Bun.file(cachePath).exists()) return cachePath;
  return runConvert(ptPath, cachePath, false);
}

// ---------------------------------------------------------------------------
// per-dump summaries (run landing page)
// ---------------------------------------------------------------------------

// Mirror of convert.py summarize() over the full view-model JSON — derives a
// summary without re-loading the .pt for an already-converted dump.
function bucketStats(rows: { status?: string | null; reward?: unknown; solved?: unknown }[]): BucketStats {
  const rewards = rows.map((r) => Number(r.reward)).filter((x) => Number.isFinite(x));
  return {
    n: rows.length,
    solved: rows.filter((r) => r.solved).length,
    aborted: rows.filter((r) => r.status === "aborted").length,
    truncated: rows.filter((r) => r.status === "truncated").length,
    mean_reward: rewards.length ? rewards.reduce((a, b) => a + b, 0) / rewards.length : null,
  };
}

function summarizeView(view: {
  rollout_id?: number | string;
  samples: {
    status?: string | null;
    reward?: unknown;
    is_solved?: unknown;
    instance_id?: string | null;
    eval_dataset?: string | null;
  }[];
}): DumpSummary {
  const rows = view.samples.map((s) => ({
    status: s.status,
    reward: s.reward,
    solved: s.is_solved,
    instance: s.instance_id,
    dataset: s.eval_dataset ?? null,
  }));
  const names = [...new Set(rows.map((r) => r.dataset).filter((d): d is string => !!d))].sort();
  return {
    rollout_id: view.rollout_id,
    ...bucketStats(rows),
    datasets: names.length
      ? Object.fromEntries(names.map((n) => [n, bucketStats(rows.filter((r) => r.dataset === n))]))
      : null,
    instances: rows
      .filter((r) => r.instance)
      .map((r) => ({
        id: r.instance!,
        dataset: r.dataset,
        reward: Number.isFinite(Number(r.reward)) ? Number(r.reward) : null,
        solved: !!r.solved,
        status: r.status ?? null,
      })),
  };
}

// Summary for one dump: sidecar cache -> derive from full JSON -> (if allowed)
// convert.py --summary. Returns null when missing and conversion not allowed.
async function dumpSummary(run: string, file: string, allowConvert: boolean): Promise<DumpSummary | null> {
  const ptPath = join(runDir(run), file);
  const st = statSync(ptPath);
  const sidecar = cachePathFor(run, file, st, "__summary");
  if (await Bun.file(sidecar).exists()) return (await Bun.file(sidecar).json()) as DumpSummary;

  const fullPath = cachePathFor(run, file, st);
  if (await Bun.file(fullPath).exists()) {
    const summary = summarizeView(await Bun.file(fullPath).json());
    await Bun.write(sidecar, JSON.stringify(summary));
    return summary;
  }

  if (!allowConvert) return null;
  await runConvert(ptPath, sidecar, true);
  return (await Bun.file(sidecar).json()) as DumpSummary;
}

async function runSummary(run: string, budget: number) {
  const files = listPtFiles(runDir(run));
  // Cached/derived summaries are cheap — always collect; convert missing ones
  // newest-first within budget so an in-flight run's tail fills in first.
  const entries = await Promise.all(
    files.map(async (f) => {
      let summary: DumpSummary | null = null;
      try {
        summary = await dumpSummary(run, f.file, false);
      } catch (e) {
        console.error(`summary failed for ${run}/${f.file}:`, e);
      }
      return { ...f, ...(fileKind(f.file) ?? { kind: "train" as const, step: -1 }), summary };
    }),
  );
  const missing = entries.filter((e) => e.summary === null).sort((a, b) => b.mtime - a.mtime);
  await Promise.all(
    missing.slice(0, Math.max(0, budget)).map(async (e) => {
      try {
        e.summary = await dumpSummary(run, e.file, true);
      } catch (err) {
        console.error(`summary conversion failed for ${run}/${e.file}:`, err);
      }
    }),
  );
  const pending = entries.filter((e) => e.summary === null).length;
  return { run, files: entries, pending };
}

function jsonError(message: string, status: number): Response {
  return Response.json({ error: message }, { status });
}

async function gzippedJson(path: string, req: Request): Promise<Response> {
  const body = await Bun.file(path).arrayBuffer();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if ((req.headers.get("accept-encoding") ?? "").includes("gzip")) {
    headers["Content-Encoding"] = "gzip";
    return new Response(Bun.gzipSync(new Uint8Array(body)), { headers });
  }
  return new Response(body, { headers });
}

const server = Bun.serve({
  port: PORT,
  hostname: "0.0.0.0",
  // Big-dump conversions take tens of seconds; default 10s would cut the
  // connection mid-convert.
  idleTimeout: 240,
  development: process.env.NODE_ENV !== "production",
  routes: {
    "/": index,

    "/api/runs": () => {
      try {
        return Response.json({ dumpRoot: DUMP_ROOT, runs: listRuns() });
      } catch (e) {
        return jsonError(String(e), 500);
      }
    },

    "/api/rollouts": (req) => {
      const run = new URL(req.url).searchParams.get("run") ?? "";
      if (run !== ROOT_RUN && !safeName(run)) return jsonError("bad run name", 400);
      return Response.json({ run, files: listPtFiles(runDir(run)) });
    },

    "/api/run-summary": async (req) => {
      const params = new URL(req.url).searchParams;
      const run = params.get("run") ?? "";
      if (run !== ROOT_RUN && !safeName(run)) return jsonError("bad run name", 400);
      const budget = Math.min(16, Math.max(0, Number(params.get("budget") ?? 4) || 0));
      try {
        return Response.json(await runSummary(run, budget));
      } catch (e) {
        return jsonError(String(e), 500);
      }
    },

    "/api/rollout": async (req) => {
      const params = new URL(req.url).searchParams;
      const run = params.get("run") ?? "";
      const file = params.get("file") ?? "";
      if (run !== ROOT_RUN && !safeName(run)) return jsonError("bad run name", 400);
      if (!safeName(file) || !file.endsWith(".pt")) return jsonError("bad file name", 400);
      try {
        const jsonPath = await convertedJsonPath(run, file);
        return await gzippedJson(jsonPath, req);
      } catch (e: any) {
        if (e?.code === "ENOENT") return jsonError("dump not found", 404);
        console.error(`conversion failed for ${run}/${file}:`, e);
        return jsonError(String(e), 500);
      }
    },
  },
});

console.log(`swe-rollout dashboard on http://${server.hostname}:${server.port} (dumps: ${DUMP_ROOT})`);
