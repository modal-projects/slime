/**
 * DOM tests for the eval-aware views: dataset-grouped eval overview, run
 * landing page (training chart + eval matrix), and instance history chips.
 *
 * Fixtures (convert.py-driven generator; see README):
 *   EVAL_FIXTURE        converted eval dump   (default /tmp/api_rollout_eval.json)
 *   RUN_SUMMARY_FIXTURE /api/run-summary body (default /tmp/run_summary.json)
 *
 *   bun test async_rl_research/dashboard/test/
 */

import { test, expect } from "bun:test";
import { Window } from "happy-dom";

const EVAL_FIXTURE = process.env.EVAL_FIXTURE ?? "/tmp/api_rollout_eval.json";
const SUMMARY_FIXTURE = process.env.RUN_SUMMARY_FIXTURE ?? "/tmp/run_summary.json";
const fixturesExist =
  (await Bun.file(EVAL_FIXTURE).exists()) && (await Bun.file(SUMMARY_FIXTURE).exists());

const RUN = "20260609-192345";
const EVAL_FILE = "rollout_eval_20.pt";

// app.ts wires listeners on import; the unique query string defeats Bun's
// module cache so each case renders fresh into its own DOM.
let bootCount = 0;

async function boot(hash: string) {
  const rollout = await Bun.file(EVAL_FIXTURE).json();
  const runSummary = await Bun.file(SUMMARY_FIXTURE).json();
  const win = new Window({ url: `http://localhost/${hash}` });
  win.document.body.innerHTML = `
    <div id="app">
      <aside id="sidebar">
        <div class="sidebar-header"><h1>SWE rollouts</h1><button id="refresh-btn">R</button></div>
        <div id="dump-root" class="dump-root"></div>
        <nav id="run-list"></nav>
      </aside>
      <main id="main"><div id="content" class="placeholder"></div></main>
    </div>`;

  const g = globalThis as Record<string, unknown>;
  g.window = win;
  g.document = win.document;
  g.location = win.location;
  g.fetch = async (url: string) => {
    if (url.startsWith("/api/runs"))
      return Response.json({
        dumpRoot: "/tmp/dump_root",
        runs: [{ name: RUN, fileCount: 5, latestMtime: Date.now() }],
      });
    if (url.startsWith("/api/rollouts"))
      return Response.json({
        run: RUN,
        files: runSummary.files.map((f: { file: string; sizeBytes: number; mtime: number }) => ({
          file: f.file,
          sizeBytes: f.sizeBytes,
          mtime: f.mtime,
        })),
      });
    if (url.startsWith("/api/run-summary")) return Response.json(runSummary);
    if (url.startsWith("/api/rollout")) return Response.json(rollout);
    return new Response("not found", { status: 404 });
  };

  await import(`../public/app.ts?case=eval${bootCount++}-${hash}`);
  await new Promise((r) => setTimeout(r, 100));
  return win.document;
}

test.skipIf(!fixturesExist)("eval overview groups by dataset", async () => {
  const document = await boot(`#${RUN}/${EVAL_FILE}`);
  const content = document.getElementById("content")!;
  const titles = [...content.querySelectorAll(".group-title")].map((n) => n.textContent);
  expect(titles).toContain("usaco_50");
  expect(titles).toContain("swebench_verified_50");
  // chips are labeled by instance, not GRPO group index
  expect(content.textContent).toContain("usaco_50-task-0");
  // per-dataset score line
  expect(content.textContent).toContain("1/3 solved");
});

test.skipIf(!fixturesExist)("run page renders training chart and eval matrix", async () => {
  const document = await boot(`#${RUN}`);
  const content = document.getElementById("content")!;
  expect(content.querySelector(".train-chart")).not.toBeNull();
  expect(content.querySelector(".eval-matrix")).not.toBeNull();
  const headers = [...content.querySelectorAll(".eval-matrix th")].map((n) => n.textContent);
  expect(headers).toContain("usaco_50");
  expect(headers).toContain("swebench_verified_50");
  // one row per eval step
  expect(content.textContent).toContain("@ 20");
  expect(content.textContent).toContain("@ 40");
});

test.skipIf(!fixturesExist)("sample view shows dataset and instance history", async () => {
  const document = await boot(`#${RUN}/${EVAL_FILE}/0`);
  const content = document.getElementById("content")!;
  expect(content.textContent).toContain("eval dataset");
  expect(content.textContent).toContain("usaco_50");
  const history = content.querySelector(".instance-history")!;
  expect(history).not.toBeNull();
  // fixture flips this instance to solved at step 40
  expect(history.textContent).toContain("@20");
  expect(history.textContent).toContain("@40");
});

test.skipIf(!fixturesExist)("sidebar splits train and eval lanes", async () => {
  const document = await boot(`#${RUN}/${EVAL_FILE}`);
  const lanes = [...document.querySelectorAll(".lane-header")].map((n) => n.textContent);
  expect(lanes.some((t) => t?.startsWith("train"))).toBe(true);
  expect(lanes.some((t) => t?.startsWith("eval"))).toBe(true);
  expect(document.getElementById("run-list")!.textContent).toContain("eval @ 20");
});
