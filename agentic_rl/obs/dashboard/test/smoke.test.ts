/**
 * DOM smoke test for the dashboard frontend, run against a real converted
 * dump so render paths see production-shaped data.
 *
 * Fixture: ROLLOUT_FIXTURE = a convert.py output JSON (default
 * /tmp/api_rollout.json); the test skips when it's missing.
 *
 *   bun test async_rl_research/dashboard/test/
 */

import { test, expect } from "bun:test";
import { Window } from "happy-dom";

const FIXTURE = process.env.ROLLOUT_FIXTURE ?? "/tmp/api_rollout.json";
const fixtureExists = await Bun.file(FIXTURE).exists();

const RUN = "20260609-192345";
const FILE = "rollout_1.pt";

async function boot(hash: string) {
  const rollout = await Bun.file(FIXTURE).json();
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
        runs: [{ name: RUN, fileCount: 1, latestMtime: Date.now() }],
      });
    if (url.startsWith("/api/rollouts"))
      return Response.json({ run: RUN, files: [{ file: FILE, sizeBytes: 1, mtime: Date.now() }] });
    if (url.startsWith("/api/rollout")) return Response.json(rollout);
    return new Response("not found", { status: 404 });
  };

  // app.ts wires listeners + kicks off the initial render on import; the query
  // string defeats Bun's module cache between cases.
  await import(`../public/app.ts?case=${hash}`);
  await new Promise((r) => setTimeout(r, 100));
  return win.document;
}

test.skipIf(!fixtureExists)("overview renders groups and summary", async () => {
  const document = await boot(`#${RUN}/${FILE}`);
  const content = document.getElementById("content")!;
  expect(content.querySelectorAll(".group").length).toBeGreaterThan(0);
  expect(content.textContent).toContain("python__mypy-11241");
  expect(content.querySelectorAll(".chip").length).toBeGreaterThanOrEqual(8);
  expect(content.textContent).toContain("samples");
  // sidebar populated too
  expect(document.getElementById("run-list")!.textContent).toContain(RUN);
});

test.skipIf(!fixtureExists)("sample view renders conversation turns", async () => {
  const document = await boot(`#${RUN}/${FILE}/0`);
  const content = document.getElementById("content")!;
  expect(content.querySelectorAll(".turn").length).toBeGreaterThan(10);
  expect(content.querySelectorAll(".tool-call").length).toBeGreaterThan(0);
  expect(content.querySelectorAll(".terminal").length).toBeGreaterThan(0);
  // the env's bash output is shown as a terminal prompt line ($ <cmd>)
  expect(content.querySelectorAll(".terminal-cmd").length).toBeGreaterThan(0);
  expect(content.textContent).toContain("status");
  expect(content.querySelector(".meta-grid")).not.toBeNull();
});

test.skipIf(!fixtureExists)("aborted sample renders placeholder instead of turns", async () => {
  const rollout = await Bun.file(FIXTURE).json();
  const abortedIdx = rollout.samples.findIndex((s: { status: string }) => s.status === "aborted");
  if (abortedIdx === -1) return; // dump has no aborted samples; nothing to assert
  const document = await boot(`#${RUN}/${FILE}/${abortedIdx}`);
  const content = document.getElementById("content")!;
  expect(content.querySelectorAll(".turn").length).toBe(0);
  expect(content.textContent).toContain("abort");
});

test.skipIf(!fixtureExists)("sample surfaces the in-sandbox agent error log", async () => {
  const rollout = await Bun.file(FIXTURE).json();
  const idx = rollout.samples.findIndex((s: { agent_tail: string | null }) => s.agent_tail);
  if (idx === -1) return; // fixture captured no agent failure tails
  const document = await boot(`#${RUN}/${FILE}/${idx}`);
  const content = document.getElementById("content")!;
  // the failure tail renders as its own highlighted block, open by default
  expect(content.querySelector("pre.agent-tail")).not.toBeNull();
  expect(content.textContent).toContain("agent error log");
  // nonzero exit is surfaced both in the block header and a meta cell
  expect(content.textContent).toContain("agent exit");
});
