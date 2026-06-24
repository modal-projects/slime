/**
 * Frontend for the SWE rollout-dump dashboard.
 *
 * Hash routes (deep-linkable):
 *   #<run>             run landing page (training curve + eval matrix)
 *   #<run>/<file>      rollout overview (groups of samples)
 *   #<run>/<file>/<i>  conversation view for samples[i]
 */

type RunInfo = { name: string; fileCount: number; latestMtime: number | null };
type RolloutFile = { file: string; sizeBytes: number; mtime: number };
type BucketStats = {
  n: number;
  solved: number;
  aborted: number;
  truncated: number;
  mean_reward: number | null;
};
type InstanceResult = {
  id: string;
  dataset: string | null;
  reward: number | null;
  solved: boolean;
  status: string | null;
};
type DumpSummary = BucketStats & {
  rollout_id?: number | string;
  datasets: Record<string, BucketStats> | null;
  instances: InstanceResult[];
};
type SummaryEntry = RolloutFile & {
  kind: "train" | "eval";
  step: number;
  summary: DumpSummary | null;
};
type RunSummary = { run: string; files: SummaryEntry[]; pending: number };

type ToolCall = { name?: string; arguments?: Record<string, unknown>; raw?: string };
type ToolResponse = { returncode?: number | null; output?: string; raw?: string; command?: string };
type Turn = {
  role: string;
  text?: string;
  think?: string[];
  tool_calls?: ToolCall[];
  tool_responses?: ToolResponse[];
  tok?: number;
  trained?: number;
};
type Span = {
  name: string;
  start_ts: number | null;
  end_ts: number | null;
  duration_sec: number | null;
  attrs?: Record<string, unknown> | null;
};
type TimingPhase = {
  name: string;
  seconds: number;
  kind: string;
};
type SampleView = {
  index: number | null;
  group_index: number | null;
  rollout_id: number | null;
  session_id: string | null;
  status: string;
  reward: number | null;
  eval_dataset: string | null;
  instance_id: string | null;
  repo: string | null;
  is_solved: boolean | null;
  applied_cleanly: boolean | null;
  abort_reason: string | null;
  finish_reason: string | null;
  agent_exit_code: number | null;
  agent_tail: string | null;
  elapsed_sec: number | null;
  segment_idx: number | null;
  num_segments: number | null;
  image: string | null;
  workdir: string | null;
  base_commit: string | null;
  problem_statement: string | null;
  full_prompt: string | null;
  n_tokens: number;
  response_length: number | null;
  trained_tokens: number;
  mean_logprob: number | null;
  weight_versions: string[];
  n_turns: number;
  gen_s: number | null;
  overhead_sec: number | null;
  recorded_turns: number | null;
  non_generation_time: number | null;
  timing_phases: TimingPhase[] | null;
  dockerfile: string | null;
  task_path: string | null;
  agent_timeout_sec: number | null;
  verifier_timeout_sec: number | null;
  boot_timeout_sec: number | null;
  cpus: number | null;
  memory_mb: number | null;
  segment_kind: string | null;
  harbor_steps_completed: number | null;
  harbor_steps_total: number | null;
  harbor_step_results: { name: string | null; reward: number | null }[] | null;
  remove_sample: boolean | null;
  label: string | null;
  turns: Turn[];
  spans: Span[];
};
type RolloutView = { rollout_id: number | string; n_samples: number; samples: SampleView[] };

const runListEl = document.getElementById("run-list")!;
const contentEl = document.getElementById("content")!;
const dumpRootEl = document.getElementById("dump-root")!;
const refreshBtn = document.getElementById("refresh-btn")! as HTMLButtonElement;

const rolloutCache = new Map<string, RolloutView>();

// ---------------------------------------------------------------------------
// tiny DOM helpers
// ---------------------------------------------------------------------------
function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  cls?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtBytes(n: number): string {
  if (n > 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n > 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${(n / 1e3).toFixed(0)} KB`;
}

function fmtTime(ms: number | null): string {
  return ms ? new Date(ms).toLocaleString() : "";
}

function fmtDur(sec: number | null | undefined): string {
  if (sec == null) return "—";
  if (sec >= 90) return `${Math.floor(sec / 60)}m${Math.round(sec % 60)}s`;
  return `${sec.toFixed(1)}s`;
}

function fmtHours(sec: number): string {
  return sec >= 3600 ? `${(sec / 3600).toFixed(1)}h` : `${Math.round(sec / 60)}m`;
}

function pct(frac: number | null): string {
  return frac == null ? "—" : `${Math.round(frac * 100)}%`;
}

function median(xs: number[]): number | null {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function fmtTokens(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = ((await res.json()) as { error?: string }).error ?? detail;
    } catch {}
    throw new Error(`${url}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// routing
// ---------------------------------------------------------------------------
function navigate(run: string, file?: string, sampleIdx?: number): void {
  const parts = [encodeURIComponent(run)];
  if (file !== undefined) parts.push(encodeURIComponent(file));
  if (sampleIdx !== undefined) parts.push(String(sampleIdx));
  location.hash = parts.join("/");
}

function parseHash(): { run: string; file?: string; sampleIdx?: number } | null {
  const h = location.hash.replace(/^#/, "");
  if (!h) return null;
  const parts = h.split("/");
  const run = decodeURIComponent(parts[0]);
  if (!run) return null;
  if (parts.length < 2) return { run };
  const file = decodeURIComponent(parts[1]);
  const sampleIdx = parts.length > 2 ? Number(parts[2]) : undefined;
  return { run, file, sampleIdx: Number.isFinite(sampleIdx) ? sampleIdx : undefined };
}

// train rollout_12.pt -> {kind:"train", step:12}; rollout_eval_20.pt -> eval.
function fileKind(file: string): { kind: "train" | "eval"; step: number } | null {
  const m = file.match(/^rollout_(eval_)?(\d+)\.pt$/);
  if (!m) return null;
  return { kind: m[1] ? "eval" : "train", step: Number(m[2]) };
}

// ---------------------------------------------------------------------------
// sidebar
// ---------------------------------------------------------------------------
async function loadSidebar(): Promise<void> {
  runListEl.textContent = "";
  runListEl.append(el("div", "placeholder", "loading runs…"));
  try {
    const { dumpRoot, runs } = await fetchJson<{ dumpRoot: string; runs: RunInfo[] }>("/api/runs");
    dumpRootEl.textContent = dumpRoot;
    runListEl.textContent = "";
    if (runs.length === 0) {
      runListEl.append(el("div", "placeholder", "no rollout dumps found"));
      return;
    }
    const current = parseHash();
    for (const run of runs) {
      runListEl.append(renderRun(run, run.name === current?.run));
    }
  } catch (e) {
    runListEl.textContent = "";
    runListEl.append(el("div", "error", String(e)));
  }
}

function renderRun(run: RunInfo, expand: boolean): HTMLElement {
  const box = el("div", "run");
  const header = el("div", "run-header");
  const name = el("span", "run-name", run.name);
  name.title = "open run page";
  header.append(name);
  header.append(el("span", "run-meta", `${run.fileCount} · ${fmtTime(run.latestMtime)}`));
  box.append(header);

  const filesBox = el("div", "rollout-files");
  let loaded = false;
  const renderLane = (label: string, files: RolloutFile[]) => {
    if (files.length === 0) return;
    filesBox.append(el("div", "lane-header", `${label} (${files.length})`));
    for (const f of files) {
      const kind = fileKind(f.file);
      const item = el("div", `rollout-file${kind?.kind === "eval" ? " eval" : ""}`);
      item.dataset.run = run.name;
      item.dataset.file = f.file;
      const display =
        kind?.kind === "eval" ? `eval @ ${kind.step}` : f.file.replace(/\.pt$/, "");
      item.append(el("span", "", display));
      item.append(el("span", "size", fmtBytes(f.sizeBytes)));
      item.onclick = () => navigate(run.name, f.file);
      filesBox.append(item);
    }
  };
  const load = async (forceOpen: boolean) => {
    if (loaded) {
      if (!forceOpen) {
        filesBox.textContent = "";
        loaded = false;
      }
      return;
    }
    const { files } = await fetchJson<{ files: RolloutFile[] }>(
      `/api/rollouts?run=${encodeURIComponent(run.name)}`,
    );
    filesBox.textContent = "";
    renderLane("train", files.filter((f) => fileKind(f.file)?.kind !== "eval"));
    renderLane("eval", files.filter((f) => fileKind(f.file)?.kind === "eval"));
    loaded = true;
    markActiveFile();
  };
  // Run name opens the landing page (and expands); the meta area toggles only.
  name.onclick = (e) => {
    e.stopPropagation();
    navigate(run.name);
    void load(true);
  };
  header.onclick = () => void load(false);
  box.append(filesBox);
  if (expand) void load(true);
  return box;
}

function markActiveFile(): void {
  const current = parseHash();
  for (const node of document.querySelectorAll<HTMLElement>(".rollout-file")) {
    node.classList.toggle(
      "active",
      node.dataset.run === current?.run && node.dataset.file === current?.file,
    );
  }
}

// ---------------------------------------------------------------------------
// run landing page (training curve + eval matrix)
// ---------------------------------------------------------------------------
const runSummaryCache = new Map<string, RunSummary>();
// Bumped on every route change so an in-flight summary poll for an
// already-left page stops touching the DOM.
let renderGeneration = 0;

async function fetchRunSummary(run: string, budget: number): Promise<RunSummary> {
  const rs = await fetchJson<RunSummary>(
    `/api/run-summary?run=${encodeURIComponent(run)}&budget=${budget}`,
  );
  runSummaryCache.set(run, rs);
  return rs;
}

function solveRate(b: BucketStats): number | null {
  return b.n > 0 ? b.solved / b.n : null;
}

// Two-series sparkline: solve rate + mean reward over the training dumps.
function renderTrainChart(run: string, entries: SummaryEntry[]): HTMLElement {
  const pts = entries
    .filter((e) => e.summary)
    .map((e) => ({
      step: e.step,
      file: e.file,
      solve: solveRate(e.summary!),
      reward: e.summary!.mean_reward,
    }));
  const box = el("div", "chart-box");
  if (pts.length === 0) {
    box.append(el("div", "placeholder", "no summarized training dumps yet"));
    return box;
  }
  const W = 720;
  const H = 120;
  const PAD = { l: 36, r: 10, t: 8, b: 18 };
  const xs = pts.map((p) => p.step);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const vals = pts.flatMap((p) => [p.solve, p.reward]).filter((v): v is number => v != null);
  const yMin = Math.min(0, ...vals);
  const yMax = Math.max(1, ...vals);
  const x = (s: number) =>
    PAD.l + (xMax === xMin ? 0.5 : (s - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "train-chart");

  const axis = (v: number, label: string) => {
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", String(PAD.l));
    line.setAttribute("x2", String(W - PAD.r));
    line.setAttribute("y1", String(y(v)));
    line.setAttribute("y2", String(y(v)));
    line.setAttribute("class", "gridline");
    svg.append(line);
    const text = document.createElementNS(svgNS, "text");
    text.setAttribute("x", "2");
    text.setAttribute("y", String(y(v) + 4));
    text.setAttribute("class", "axis-label");
    text.textContent = label;
    svg.append(text);
  };
  axis(0, "0");
  axis(1, "1");

  const series = (key: "solve" | "reward", cls: string) => {
    const have = pts.filter((p) => p[key] != null);
    if (have.length === 0) return;
    const poly = document.createElementNS(svgNS, "polyline");
    poly.setAttribute("points", have.map((p) => `${x(p.step)},${y(p[key]!)}`).join(" "));
    poly.setAttribute("class", `series ${cls}`);
    svg.append(poly);
    for (const p of have) {
      const c = document.createElementNS(svgNS, "circle");
      c.setAttribute("cx", String(x(p.step)));
      c.setAttribute("cy", String(y(p[key]!)));
      c.setAttribute("r", "3.5");
      c.setAttribute("class", `dot ${cls}`);
      const title = document.createElementNS(svgNS, "title");
      title.textContent = `step ${p.step}: solve ${p.solve == null ? "—" : (p.solve * 100).toFixed(0) + "%"}, mean reward ${p.reward == null ? "—" : p.reward.toFixed(3)}`;
      c.append(title);
      c.addEventListener("click", () => navigate(run, p.file));
      svg.append(c);
    }
  };
  series("solve", "solve");
  series("reward", "reward");

  const legend = el("div", "chart-legend");
  legend.append(el("span", "legend-item solve", "● solve rate"));
  legend.append(el("span", "legend-item reward", "● mean reward"));
  box.append(svg, legend);
  return box;
}

// Rows = eval step, columns = datasets. Cells are solve fractions.
function renderEvalMatrix(run: string, entries: SummaryEntry[]): HTMLElement {
  const box = el("div");
  const summarized = entries.filter((e) => e.summary);
  if (summarized.length === 0) {
    box.append(el("div", "placeholder", "no summarized eval dumps yet"));
    return box;
  }
  const UNTAGGED = "(untagged)";
  const datasets = [
    ...new Set(
      summarized.flatMap((e) => (e.summary!.datasets ? Object.keys(e.summary!.datasets) : [UNTAGGED])),
    ),
  ].sort();

  const table = el("table", "eval-matrix");
  const head = el("tr");
  head.append(el("th", "", "step"));
  for (const d of datasets) head.append(el("th", "", d));
  table.append(head);

  for (const e of [...summarized].sort((a, b) => a.step - b.step)) {
    const row = el("tr");
    const stepCell = el("td", "step-cell", `@ ${e.step}`);
    stepCell.onclick = () => navigate(run, e.file);
    row.append(stepCell);
    const buckets = e.summary!.datasets ?? { [UNTAGGED]: e.summary! };
    for (const d of datasets) {
      const b = buckets[d];
      const cell = el("td", "cell");
      if (b) {
        const rate = solveRate(b);
        cell.append(el("b", "", `${b.solved}/${b.n}`));
        if (rate != null) {
          cell.append(el("span", "pct", ` ${(rate * 100).toFixed(0)}%`));
          cell.style.background = `color-mix(in srgb, var(--green) ${Math.round(rate * 28)}%, transparent)`;
        }
        cell.title = `${d} @ ${e.step}: mean reward ${b.mean_reward?.toFixed(3) ?? "—"}, aborted ${b.aborted}, truncated ${b.truncated}`;
        cell.onclick = () => navigate(run, e.file);
      } else {
        cell.textContent = "—";
      }
      row.append(cell);
    }
    table.append(row);
  }
  box.append(table);
  return box;
}

function renderRunPage(run: string, rs: RunSummary): void {
  contentEl.className = "view";
  contentEl.textContent = "";
  contentEl.append(el("h2", "", run));

  const train = rs.files.filter((e) => e.kind === "train");
  const evals = rs.files.filter((e) => e.kind === "eval");
  contentEl.append(
    el("div", "subtitle", `${train.length} training dumps · ${evals.length} eval dumps`),
  );

  if (rs.pending > 0) {
    contentEl.append(
      el("div", "pending-note", `summarizing ${rs.pending} more dump(s) in the background…`),
    );
  }

  contentEl.append(el("h3", "section-title", "training"));
  contentEl.append(renderTrainChart(run, train));

  contentEl.append(el("h3", "section-title", "eval"));
  contentEl.append(renderEvalMatrix(run, evals));
}

async function showRunPage(run: string): Promise<void> {
  const gen = ++renderGeneration;
  const cached = runSummaryCache.get(run);
  if (cached) renderRunPage(run, cached);
  else {
    contentEl.className = "loading";
    contentEl.textContent = `summarizing dumps for ${run} …`;
  }
  try {
    let rs = await fetchRunSummary(run, 4);
    while (gen === renderGeneration) {
      renderRunPage(run, rs);
      if (rs.pending === 0) break;
      await new Promise((resolve) => setTimeout(resolve, 2500));
      if (gen !== renderGeneration) break;
      rs = await fetchRunSummary(run, 4);
    }
  } catch (e) {
    if (gen !== renderGeneration) return;
    contentEl.className = "error";
    contentEl.textContent = String(e);
  }
}

// ---------------------------------------------------------------------------
// rollout overview (groups)
// ---------------------------------------------------------------------------
async function loadRollout(run: string, file: string): Promise<RolloutView> {
  const key = `${run}/${file}`;
  const cached = rolloutCache.get(key);
  if (cached) return cached;
  const view = await fetchJson<RolloutView>(
    `/api/rollout?run=${encodeURIComponent(run)}&file=${encodeURIComponent(file)}`,
  );
  rolloutCache.set(key, view);
  return view;
}

function chipClass(s: SampleView): string {
  if (s.status === "aborted") return "chip aborted";
  if (s.status === "truncated") return "chip truncated";
  if (s.is_solved) return "chip solved";
  return "chip failed";
}

function chipLabel(s: SampleView): string {
  if (s.status === "aborted") return "✕ abort";
  if (s.is_solved) return `✓ ${s.reward ?? ""}`;
  if (s.status === "truncated") return "✂ len";
  return `${s.reward ?? "·"}`;
}

// Rollout-health profile: how rollouts terminate, how that predicts solving,
// and where wall-clock goes. Built from the full sample list the overview
// already holds — no summary-schema change needed.
function renderHealthPanel(samples: SampleView[]): HTMLElement {
  const box = el("div", "health-panel");
  const completed = samples.filter((s) => s.status !== "aborted");
  const aborted = samples.filter((s) => s.status === "aborted");
  if (samples.length === 0) return box;

  const turnsOf = (s: SampleView) => s.recorded_turns ?? s.n_turns ?? 0;
  const solveRateOf = (arr: SampleView[]): number | null =>
    arr.length ? arr.filter((s) => s.is_solved).length / arr.length : null;

  const lengthB = completed.filter((s) => s.finish_reason === "length");
  const stopB = completed.filter((s) => s.finish_reason === "stop");
  const otherB = completed.filter(
    (s) => s.finish_reason !== "length" && s.finish_reason !== "stop",
  );
  const maxTurns = completed.length ? Math.max(...completed.map(turnsOf)) : 0;
  const atCap = completed.filter((s) => maxTurns > 0 && turnsOf(s) >= maxTurns);
  // Only call it a "cap" when the top turn-count recurs (a real ceiling, not
  // just the single longest run); then split the natural-stop bucket by it.
  const isCap = maxTurns > 0 && atCap.length >= 2;

  // --- solve rate by how the rollout ended ---
  type Bucket = { label: string; cls: string; arr: SampleView[] };
  const buckets: Bucket[] = [];
  if (isCap) {
    buckets.push({ label: "finished early", cls: "good", arr: stopB.filter((s) => turnsOf(s) < maxTurns) });
    buckets.push({ label: `stop @ ${maxTurns}-turn cap`, cls: "mid", arr: stopB.filter((s) => turnsOf(s) >= maxTurns) });
  } else {
    buckets.push({ label: "natural stop", cls: "good", arr: stopB });
  }
  buckets.push({ label: "length-truncated", cls: "warn", arr: lengthB });
  if (otherB.length) buckets.push({ label: "other finish", cls: "mid", arr: otherB });
  buckets.push({ label: "aborted", cls: "bad", arr: aborted });

  const termSec = el("div", "health-section");
  termSec.append(el("div", "health-title", "solve rate by termination"));
  const bars = el("div", "term-bars");
  for (const b of buckets.filter((b) => b.arr.length > 0)) {
    const rate = solveRateOf(b.arr);
    const row = el("div", "term-row");
    row.append(el("div", "term-label", b.label));
    const track = el("div", "term-track");
    const fill = el("div", `term-fill ${b.cls}`);
    fill.style.width = `${Math.round((rate ?? 0) * 100)}%`;
    track.append(fill);
    row.append(track, el("div", "term-meta", `${pct(rate)} · n=${b.arr.length}`));
    bars.append(row);
  }
  termSec.append(bars);
  box.append(termSec);

  // --- wall-clock split: generation vs in-sandbox overhead ---
  const withGen = completed.filter((s) => s.gen_s != null);
  const genTotal = withGen.reduce((a, s) => a + (s.gen_s ?? 0), 0);
  const ovTotal = completed.reduce((a, s) => a + (s.overhead_sec ?? 0), 0);
  if (genTotal + ovTotal > 0) {
    const genFrac = genTotal / (genTotal + ovTotal);
    const ovPerTurn = median(
      completed
        .filter((s) => s.overhead_sec != null && turnsOf(s) > 0)
        .map((s) => (s.overhead_sec as number) / turnsOf(s)),
    );
    const perfSec = el("div", "health-section");
    perfSec.append(el("div", "health-title", "wall-clock split"));
    const split = el("div", "split-bar");
    const g = el("div", "split-seg gen", `generation ${pct(genFrac)}`);
    g.style.width = `${Math.round(genFrac * 100)}%`;
    const o = el("div", "split-seg ov", `sandbox + grading ${pct(1 - genFrac)}`);
    o.style.width = `${Math.round((1 - genFrac) * 100)}%`;
    split.append(g, o);
    perfSec.append(split);
    perfSec.append(
      el(
        "div",
        "health-note",
        `${fmtHours(genTotal)} generating · ${fmtHours(ovTotal)} in sandbox` +
          (ovPerTurn != null ? ` · ≈${ovPerTurn.toFixed(1)}s overhead/turn` : ""),
      ),
    );
    box.append(perfSec);
  }

  // --- abort-reason rollup ---
  if (aborted.length) {
    const reasons = new Map<string, number>();
    for (const s of aborted) {
      const r = s.abort_reason ?? "unknown";
      reasons.set(r, (reasons.get(r) ?? 0) + 1);
    }
    const abortSec = el("div", "health-section");
    abortSec.append(el("div", "health-title", `aborts (${aborted.length})`));
    const list = el("div", "abort-list");
    for (const [r, n] of [...reasons.entries()].sort((a, b) => b[1] - a[1])) {
      list.append(el("span", "abort-chip", `${n}× ${r}`));
    }
    abortSec.append(list);
    box.append(abortSec);
  }

  return box;
}

function renderOverview(run: string, file: string, view: RolloutView): void {
  const isEval = fileKind(file)?.kind === "eval";
  contentEl.className = "view";
  contentEl.textContent = "";

  const back = el("span", "back-link", `← ${run}`);
  back.onclick = () => navigate(run);
  contentEl.append(back);

  contentEl.append(el("h2", "", `${run} / ${file}`));
  contentEl.append(
    el("div", "subtitle", `rollout_id=${view.rollout_id}${isEval ? " · eval" : ""}`),
  );

  const samples = view.samples;
  const solved = samples.filter((s) => s.is_solved).length;
  const aborted = samples.filter((s) => s.status === "aborted").length;
  const truncated = samples.filter((s) => s.status === "truncated").length;
  const rewards = samples.map((s) => s.reward ?? 0);
  const meanReward = rewards.length ? rewards.reduce((a, b) => a + b, 0) / rewards.length : 0;
  const elapsed = samples.map((s) => s.elapsed_sec).filter((x): x is number => x != null);
  const maxElapsed = elapsed.length ? Math.max(...elapsed) : null;
  const nLength = samples.filter((s) => s.finish_reason === "length").length;
  const turnsArr = samples
    .filter((s) => s.status !== "aborted")
    .map((s) => s.recorded_turns ?? s.n_turns ?? 0);
  const meanTurns = turnsArr.length
    ? turnsArr.reduce((a, b) => a + b, 0) / turnsArr.length
    : null;

  const bar = el("div", "summary-bar");
  const stat = (label: string, value: string) => {
    const s = el("span", "stat");
    s.append(el("b", "", value), document.createTextNode(` ${label}`));
    return s;
  };
  bar.append(
    stat("samples", String(samples.length)),
    stat("solved", `${solved}`),
    stat("aborted", `${aborted}`),
    stat("truncated", `${truncated}`),
    stat("length-capped", nLength ? `${nLength} · ${Math.round((100 * nLength) / samples.length)}%` : "0"),
    stat("mean turns", meanTurns == null ? "—" : meanTurns.toFixed(0)),
    stat("mean reward", meanReward.toFixed(3)),
    stat("max elapsed", fmtDur(maxElapsed)),
  );
  contentEl.append(bar);
  contentEl.append(renderHealthPanel(samples));

  // filters
  const filters = el("div", "filters");
  const search = el("input");
  search.placeholder = "filter by instance id…";
  const statusSel = el("select");
  for (const opt of ["all", "solved", "unsolved", "aborted", "truncated"]) {
    const o = el("option", "", opt);
    o.value = opt;
    statusSel.append(o);
  }
  filters.append(search, statusSel);
  contentEl.append(filters);

  // Eval dumps flatten many benchmarks (swe_gym_lite, swebench_verified,
  // tblite, …) into one list; without a way to focus one, they read as a
  // single wall. A benchmark tab bar (built once sortedGroups exists below)
  // scopes the view to one dataset at a time. "all" keeps the combined view.
  let selectedDataset = "all";
  const benchTabs = el("div", "bench-tabs");
  if (isEval) contentEl.append(benchTabs);

  const groupsBox = el("div");
  contentEl.append(groupsBox);

  // Train dumps group by GRPO group (one prompt's siblings); eval dumps group
  // by dataset (group_index is meaningless there; the dump flattens all
  // datasets into one list, attributed via metadata.eval_dataset).
  const groups = new Map<string, SampleView[]>();
  const groupKey = (s: SampleView) =>
    isEval ? (s.eval_dataset ?? "(untagged)") : String(s.group_index ?? -1);
  samples.forEach((s) => {
    const g = groupKey(s);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g)!.push(s);
  });
  const sortedGroups = [...groups.entries()].sort((a, b) =>
    isEval ? (a[0] < b[0] ? -1 : 1) : Number(a[0]) - Number(b[0]),
  );

  const renderGroups = () => {
    groupsBox.textContent = "";
    const q = search.value.trim().toLowerCase();
    const want = statusSel.value;
    for (const [gid, members] of sortedGroups) {
      if (isEval && selectedDataset !== "all" && gid !== selectedDataset) continue;
      const inst = members[0]?.instance_id ?? "?";
      // Train: group shares one instance, filter at group level. Eval: group is
      // a dataset of many instances, filter per sample.
      if (!isEval && q && !inst.toLowerCase().includes(q)) continue;
      const visible = members.filter((s) => {
        if (isEval && q && !(s.instance_id ?? "").toLowerCase().includes(q)) return false;
        if (want === "solved") return !!s.is_solved;
        if (want === "unsolved") return !s.is_solved && s.status !== "aborted";
        if (want === "aborted") return s.status === "aborted";
        if (want === "truncated") return s.status === "truncated";
        return true;
      });
      if (visible.length === 0) continue;

      const g = el("div", "group");
      const gh = el("div", "group-header");
      const title = el("span", "group-title", isEval ? gid : inst);
      if (!isEval && members[0]?.repo) title.append(el("span", "repo", members[0].repo));
      gh.append(title);
      const nSolved = members.filter((s) => s.is_solved).length;
      const rewards = members.map((s) => s.reward ?? 0);
      const mean = rewards.length ? rewards.reduce((a, b) => a + b, 0) / rewards.length : 0;
      gh.append(
        el(
          "span",
          "group-score",
          isEval
            ? `${nSolved}/${members.length} solved · mean reward ${mean.toFixed(3)}`
            : `group ${gid} · ${nSolved}/${members.length} solved`,
        ),
      );
      g.append(gh);

      const chips = el("div", "sample-chips");
      for (const s of visible) {
        const idx = samples.indexOf(s);
        const label = isEval
          ? `${s.instance_id ?? `#${s.index ?? idx}`} ${chipLabel(s)}`
          : `#${s.index ?? idx} ${chipLabel(s)}`;
        const chip = el("button", chipClass(s), label);
        chip.title = `status=${s.status} reward=${s.reward} turns=${s.n_turns} tokens=${s.n_tokens}`;
        chip.onclick = () => navigate(run, file, idx);
        chips.append(chip);
      }
      g.append(chips);
      groupsBox.append(g);
    }
    if (groupsBox.childElementCount === 0) {
      groupsBox.append(el("div", "placeholder", "no samples match the filter"));
    }
  };
  // One tab per benchmark (+ "all"), each labelled with its solved/total so
  // you can compare benchmarks at a glance and click to scope the list.
  const renderBenchTabs = () => {
    if (!isEval) return;
    benchTabs.textContent = "";
    const mkTab = (key: string, label: string) => {
      const t = el("button", `bench-tab${selectedDataset === key ? " active" : ""}`, label);
      t.onclick = () => {
        selectedDataset = key;
        renderBenchTabs();
        renderGroups();
      };
      benchTabs.append(t);
    };
    const totalSolved = samples.filter((s) => s.is_solved).length;
    mkTab("all", `all · ${totalSolved}/${samples.length}`);
    for (const [gid, members] of sortedGroups) {
      const nSolved = members.filter((s) => s.is_solved).length;
      mkTab(gid, `${gid} · ${nSolved}/${members.length}`);
    }
  };

  search.oninput = renderGroups;
  statusSel.onchange = renderGroups;
  renderBenchTabs();
  renderGroups();
}

// ---------------------------------------------------------------------------
// sample conversation view
// ---------------------------------------------------------------------------
const CLIP_CHARS = 1500;

function clippedPre(text: string, cls: string): HTMLElement {
  const wrap = el("div");
  const pre = el("pre", cls);
  if (text.length <= CLIP_CHARS) {
    pre.textContent = text;
    wrap.append(pre);
    return wrap;
  }
  pre.textContent = text.slice(0, CLIP_CHARS);
  const note = el("div", "clipped-note", `… show ${text.length - CLIP_CHARS} more chars`);
  note.onclick = () => {
    pre.textContent = text;
    note.remove();
  };
  wrap.append(pre, note);
  return wrap;
}

// Render tool responses (the "user" turn fed back to the agent) as a terminal
// pane: titlebar with exit code, the bash command as a prompt line (paired in
// convert.py), then the captured stdout/stderr.
function renderTerminal(responses: ToolResponse[]): HTMLElement {
  const term = el("div", "terminal");

  const bar = el("div", "terminal-titlebar");
  const dots = el("div", "terminal-dots");
  for (const c of ["red", "yellow", "green"]) dots.append(el("span", `terminal-dot ${c}`));
  bar.append(dots, el("span", "terminal-title", "bash"));
  const rc = responses.find((r) => r.returncode != null)?.returncode;
  if (rc != null) {
    const badge = el("span", `rc-badge${rc === 0 ? "" : " nonzero"}`, `exit ${rc}`);
    bar.append(badge);
  }
  term.append(bar);

  const bodyWrap = el("div", "terminal-body");
  for (const resp of responses) {
    if (resp.command) {
      const line = el("pre", "terminal-cmd");
      line.append(el("span", "terminal-sigil", "$ "), document.createTextNode(resp.command));
      bodyWrap.append(line);
    }
    const out = resp.output ?? resp.raw ?? "";
    if (out.trim()) bodyWrap.append(clippedPre(out, "terminal-out"));
  }
  term.append(bodyWrap);
  return term;
}

function renderTurn(turn: Turn, idx: number, cumTokens?: number): HTMLElement {
  const box = el("div", `turn ${turn.role}`);
  const header = el("div", "turn-header");
  header.append(el("span", "th-idx", `${idx}`), el("span", "th-role", turn.role), el("span", "th-spacer"));
  if (turn.tok != null) {
    const tok = turn.tok;
    const trained = turn.trained ?? 0;
    const frac = tok > 0 ? trained / tok : 0;
    header.append(el("span", "th-tok", `${tok} tok`));
    const bar = el("div", "mask-bar");
    bar.title = `trained ${trained}/${tok} tokens (${Math.round(frac * 100)}%)`;
    const fill = el("div", "mask-fill");
    fill.style.width = `${Math.round(frac * 100)}%`;
    bar.append(fill);
    header.append(bar, el("span", "th-trained", `${Math.round(frac * 100)}%`));
    // An assistant turn with no trained tokens = fully masked output (drift /
    // re-render mismatch) — the signal worth catching, so flag it.
    if (turn.role === "assistant" && trained === 0) header.append(el("span", "th-flag", "masked"));
  }
  if (cumTokens != null) header.append(el("span", "th-cum", `Σ ${fmtTokens(cumTokens)}`));
  const rc = (turn.tool_responses ?? []).find((r) => r.returncode != null)?.returncode;
  if (rc != null) header.append(el("span", `th-rc${rc === 0 ? "" : " nonzero"}`, `exit ${rc}`));
  box.append(header);
  const body = el("div", "turn-body");

  for (const think of turn.think ?? []) {
    const d = el("details", "think") as HTMLDetailsElement;
    d.append(el("summary", "", `thinking (${think.length} chars)`));
    d.append(clippedPre(think, ""));
    body.append(d);
  }

  if (turn.text) body.append(clippedPre(turn.text, "turn-text"));

  for (const call of turn.tool_calls ?? []) {
    const c = el("div", "tool-call");
    const cmd = call.arguments?.["command"];
    if (call.name === "bash" && typeof cmd === "string") {
      c.append(el("div", "tool-name", "$ bash"));
      c.append(clippedPre(cmd, ""));
    } else {
      c.append(el("div", "tool-name", call.name ?? "tool_call"));
      c.append(clippedPre(call.raw ?? JSON.stringify(call.arguments, null, 2), ""));
    }
    body.append(c);
  }

  if (turn.tool_responses?.length) {
    body.append(renderTerminal(turn.tool_responses));
  }

  box.append(body);
  return box;
}

function metaCell(grid: HTMLElement, key: string, value: string, cls = ""): void {
  const cell = el("div");
  cell.append(el("div", "k", key));
  cell.append(el("div", `v ${cls}`.trim(), value));
  grid.append(cell);
}

function renderTimingBar(s: SampleView): HTMLElement | null {
  const phases = (s.timing_phases ?? []).filter((p) => Number.isFinite(p.seconds) && p.seconds > 0);
  if (!phases.length) return null;
  const total = phases.reduce((sum, p) => sum + p.seconds, 0);
  if (total <= 0) return null;

  const card = el("div", "timing-card");
  const header = el("div", "timing-header");
  header.append(el("span", "", "wall-clock split"), el("span", "", fmtDur(total)));
  card.append(header);

  const bar = el("div", "timing-bar");
  for (const phase of phases) {
    const frac = phase.seconds / total;
    const seg = el("div", `timing-seg ${phase.kind}`);
    seg.style.width = `${frac * 100}%`;
    seg.title = `${phase.name}: ${fmtDur(phase.seconds)} (${pct(frac)})`;
    if (frac >= 0.1) seg.textContent = `${phase.name} ${fmtDur(phase.seconds)}`;
    bar.append(seg);
  }
  card.append(bar);

  const legend = el("div", "timing-legend");
  for (const phase of phases) {
    const item = el("span", "timing-legend-item");
    item.append(el("span", `timing-dot ${phase.kind}`), document.createTextNode(`${phase.name} ${fmtDur(phase.seconds)}`));
    legend.append(item);
  }
  card.append(legend);
  return card;
}

async function renderInstanceHistory(
  box: HTMLElement,
  run: string,
  currentFile: string,
  instanceId: string,
): Promise<void> {
  let rs = runSummaryCache.get(run);
  if (!rs) {
    try {
      rs = await fetchRunSummary(run, 0); // cached summaries only — no conversions
    } catch {
      return;
    }
  }
  if (!box.isConnected) return; // user navigated away mid-fetch
  const hits = rs.files
    .filter((e) => e.kind === "eval" && e.summary)
    .map((e) => ({
      e,
      r: e.summary!.instances.find((i) => i.id === instanceId),
    }))
    .filter((h): h is { e: SummaryEntry; r: InstanceResult } => !!h.r)
    .sort((a, b) => a.e.step - b.e.step);
  if (hits.length === 0) return;

  box.append(el("span", "history-label", "across evals:"));
  for (const { e, r } of hits) {
    const cls = r.status === "aborted" ? "aborted" : r.solved ? "solved" : "failed";
    const chip = el(
      "button",
      `chip ${cls}${e.file === currentFile ? " current" : ""}`,
      `@${e.step} ${r.solved ? "✓" : "✗"} ${r.reward?.toFixed(2) ?? "·"}`,
    );
    chip.title = `${e.file}: status=${r.status} reward=${r.reward}`;
    chip.onclick = () => navigate(run, e.file);
    box.append(chip);
  }
}

function renderSample(run: string, file: string, view: RolloutView, sampleIdx: number): void {
  const s = view.samples[sampleIdx];
  contentEl.className = "view";
  contentEl.textContent = "";

  const back = el("span", "back-link", `← ${run} / ${file}`);
  back.onclick = () => navigate(run, file);
  contentEl.append(back);

  contentEl.append(el("h2", "", `${s.instance_id ?? "sample"} · #${s.index ?? sampleIdx}`));
  const sib = view.samples
    .map((x, i) => ({ x, i }))
    .filter(({ x }) => x.group_index === s.group_index);
  const nav = el("div", "subtitle");
  nav.append(document.createTextNode(`group ${s.group_index} siblings: `));
  for (const { x, i } of sib) {
    const a = el("span", i === sampleIdx ? "chip solved" : chipClass(x), `#${x.index ?? i}`);
    (a.style as CSSStyleDeclaration).cursor = "pointer";
    a.style.marginRight = "4px";
    if (i === sampleIdx) a.style.borderColor = "var(--accent)";
    a.onclick = () => navigate(run, file, i);
    nav.append(a);
  }
  contentEl.append(nav);

  const timingBar = renderTimingBar(s);
  if (timingBar) contentEl.append(timingBar);

  const grid = el("div", "meta-grid");
  metaCell(grid, "status", s.status, s.status === "completed" ? "good" : s.status === "aborted" ? "warn" : "bad");
  if (s.eval_dataset) metaCell(grid, "eval dataset", s.eval_dataset);
  metaCell(grid, "reward", String(s.reward ?? "—"), (s.reward ?? 0) > 0 ? "good" : "");
  metaCell(grid, "solved", String(s.is_solved ?? "—"), s.is_solved ? "good" : "bad");
  if (s.applied_cleanly != null)
    metaCell(grid, "applied cleanly", String(s.applied_cleanly), s.applied_cleanly === false ? "bad" : "");
  if (s.abort_reason) metaCell(grid, "abort reason", s.abort_reason, "warn");
  if (s.agent_exit_code != null && s.agent_exit_code !== 0)
    metaCell(grid, "agent exit", String(s.agent_exit_code), "bad");
  metaCell(grid, "finish reason", s.finish_reason ?? "—", s.finish_reason === "length" ? "warn" : "");
  metaCell(grid, "elapsed", fmtDur(s.elapsed_sec));
  if (s.gen_s != null) metaCell(grid, "gen time", fmtDur(s.gen_s));
  if (s.overhead_sec != null) metaCell(grid, "env overhead", fmtDur(s.overhead_sec));
  if (s.gen_s != null && s.elapsed_sec)
    metaCell(grid, "gen fraction", pct(s.gen_s / s.elapsed_sec), s.gen_s / s.elapsed_sec < 0.5 ? "warn" : "");
  // recorded_turns = agent-loop turns (the capped count); n_turns = rendered
  // message turns (assistant + tool-response, ~2x). Show both, unambiguously.
  if (s.recorded_turns != null) metaCell(grid, "agent turns", String(s.recorded_turns));
  metaCell(grid, "message turns", String(s.n_turns));
  if (s.gen_s != null && s.recorded_turns) metaCell(grid, "avg gen/turn", fmtDur(s.gen_s / s.recorded_turns));
  metaCell(grid, "tokens (prompt+resp)", String(s.n_tokens));
  metaCell(grid, "response tokens", String(s.response_length ?? "—"));
  metaCell(grid, "trained tokens", String(s.trained_tokens));
  if (s.mean_logprob != null) metaCell(grid, "mean logprob", s.mean_logprob.toFixed(4));
  if (s.weight_versions.length) metaCell(grid, "weight versions", s.weight_versions.join(", "));
  metaCell(grid, "session", s.session_id ?? "—");
  if (s.base_commit) metaCell(grid, "base commit", s.base_commit.slice(0, 12));
  if (s.workdir) metaCell(grid, "workdir", s.workdir);
  if (s.image) metaCell(grid, "image", s.image.replace(/^docker\.io\//, ""));
  else if (s.dockerfile) metaCell(grid, "dockerfile", s.dockerfile);
  if (s.cpus != null || s.memory_mb != null)
    metaCell(grid, "sandbox", `${s.cpus ?? "?"} cpu · ${s.memory_mb ?? "?"} MB`);
  if (s.boot_timeout_sec != null) metaCell(grid, "boot budget", fmtDur(s.boot_timeout_sec));
  if (s.agent_timeout_sec != null) metaCell(grid, "agent budget", fmtDur(s.agent_timeout_sec));
  if (s.verifier_timeout_sec != null) metaCell(grid, "verifier timeout", fmtDur(s.verifier_timeout_sec));
  if (s.harbor_steps_total != null)
    metaCell(grid, "harbor steps", `${s.harbor_steps_completed ?? "?"}/${s.harbor_steps_total}`);
  if (s.segment_kind) metaCell(grid, "segment kind", s.segment_kind);
  if (s.task_path) metaCell(grid, "task path", s.task_path);
  if (s.remove_sample) metaCell(grid, "remove sample", "true", "warn");
  for (const span of s.spans) {
    metaCell(grid, `span ${span.name}`, fmtDur(span.duration_sec));
  }
  contentEl.append(grid);

  // The in-sandbox agent process's failure tail (agent_tail; empty on a clean
  // exit). Open by default — it's the "why" behind an abort / zero-turn rollout
  // (e.g. a pydantic_core ImportError that yields adapter_session_empty).
  if (s.agent_tail) {
    const d = el("details", "block") as HTMLDetailsElement;
    d.open = true;
    d.append(
      el("summary", "", `agent error log — exit ${s.agent_exit_code ?? "?"} (${s.agent_tail.length} chars)`),
    );
    d.append(clippedPre(s.agent_tail, "agent-tail"));
    contentEl.append(d);
  }

  // Per-step grading breakdown matters only for multi-step graders; for a
  // single step it's just the overall reward, so skip it.
  if (s.harbor_step_results && s.harbor_step_results.length > 1) {
    const d = el("details", "block") as HTMLDetailsElement;
    d.append(el("summary", "", `grading steps (${s.harbor_step_results.length})`));
    const list = el("div");
    list.style.padding = "4px 14px 10px";
    for (const st of s.harbor_step_results) {
      list.append(el("div", st.reward ? "v good" : "v", `${st.name ?? "(step)"} — ${st.reward ?? "—"}`));
    }
    d.append(list);
    contentEl.append(d);
  }

  // This instance's solved/reward at every eval step — spot instances that flip
  // solved<->unsolved as training progresses. Async from cached summaries
  // (budget=0 never triggers .pt conversions; visit the run page to build).
  if (s.instance_id) {
    const historyBox = el("div", "instance-history");
    contentEl.append(historyBox);
    void renderInstanceHistory(historyBox, run, file, s.instance_id);
  }

  if (s.full_prompt) {
    const d = el("details", "block") as HTMLDetailsElement;
    d.append(el("summary", "", `full prompt — first turn (${s.full_prompt.length} chars)`));
    d.append(clippedPre(s.full_prompt, ""));
    contentEl.append(d);
  }

  if (s.problem_statement) {
    const d = el("details", "block") as HTMLDetailsElement;
    d.append(el("summary", "", `problem statement (${s.problem_statement.length} chars)`));
    d.append(clippedPre(s.problem_statement, ""));
    contentEl.append(d);
  }

  if (s.turns.length === 0) {
    contentEl.append(
      el(
        "div",
        "placeholder",
        s.abort_reason
          ? `no trajectory recorded — aborted (${s.abort_reason})`
          : "no trajectory recorded",
      ),
    );
    return;
  }

  // Turn minimap: one cell per turn, width ∝ tokens (so the strip doubles as a
  // context-growth timeline — wide gray blocks are big tool outputs, thin green
  // ones are the model's turns), colored by training signal: green = trained
  // assistant, gray = masked context, red = assistant turn with 0 trained
  // tokens (drift). Click a cell to scroll to that turn.
  const totalTok = s.turns.reduce((a, t) => a + (t.tok ?? 0), 0);
  const turnEls: HTMLElement[] = [];
  const minimap = el("div", "turn-minimap");
  s.turns.forEach((t, i) => {
    const tok = t.tok ?? 0;
    const trained = t.trained ?? 0;
    const cell = el("button", "mm-cell");
    cell.style.flexGrow = String(totalTok > 0 ? tok : 1);
    cell.classList.add(t.role === "assistant" ? (trained > 0 ? "asst" : "drift") : "ctx");
    const rc = (t.tool_responses ?? []).find((r) => r.returncode != null)?.returncode;
    if (rc != null && rc !== 0) cell.classList.add("err");
    cell.title = `turn ${i} · ${t.role} · ${tok} tok · ${trained} trained${rc != null ? ` · exit ${rc}` : ""}`;
    cell.onclick = () => turnEls[i]?.scrollIntoView({ behavior: "smooth", block: "start" });
    minimap.append(cell);
  });
  const mmWrap = el("div", "minimap-wrap");
  const legend = el("div", "minimap-legend");
  legend.append(
    el("span", "mm-key asst", "■ trained"),
    el("span", "mm-key ctx", "■ context"),
    el("span", "mm-key drift", "■ masked assistant"),
    el("span", "mm-grow", `${s.turns.length} turns · ${fmtTokens(totalTok)} response tokens`),
  );
  mmWrap.append(minimap, legend);
  contentEl.append(mmWrap);

  let cum = 0;
  for (let i = 0; i < s.turns.length; i++) {
    cum += s.turns[i].tok ?? 0;
    const box = renderTurn(s.turns[i], i, totalTok > 0 ? cum : undefined);
    box.id = `turn-${i}`;
    turnEls.push(box);
    contentEl.append(box);
  }
}

// ---------------------------------------------------------------------------
// route dispatch
// ---------------------------------------------------------------------------
async function render(): Promise<void> {
  markActiveFile();
  const route = parseHash();
  if (!route) {
    renderGeneration++; // cancel any run-page summary polling
    contentEl.className = "placeholder";
    contentEl.textContent = "Select a run or rollout dump on the left.";
    return;
  }
  if (route.file === undefined) {
    void showRunPage(route.run); // manages renderGeneration itself
    return;
  }
  renderGeneration++;
  contentEl.className = "loading";
  contentEl.textContent = `loading ${route.run}/${route.file} … (first open converts the .pt, may take a moment)`;
  try {
    const view = await loadRollout(route.run, route.file);
    if (route.sampleIdx !== undefined && view.samples[route.sampleIdx]) {
      renderSample(route.run, route.file, view, route.sampleIdx);
      window.scrollTo(0, 0);
    } else {
      renderOverview(route.run, route.file, view);
    }
  } catch (e) {
    contentEl.className = "error";
    contentEl.textContent = String(e);
  }
}

refreshBtn.onclick = () => void loadSidebar();
window.addEventListener("hashchange", () => void render());

void loadSidebar();
void render();
