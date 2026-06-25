#!/usr/bin/env python3
"""Profile the Frontier-CS verifier server (Node + go-judge) under oracle load.

Boots the SAME production judge that ``FrontierCsEnv`` uses
(``environment/verifier_server/autostart.ensure_started`` -> a ``vm_runtime`` Modal
Sandbox mounting ``slime-data`` at ``/data`` with ``problemsRoot=/data/frontier_cs/problems``),
then drives it with C++ submissions over its HTTP API and records latency cleanly.

Because only problem 0 ships a verified ``reference.cpp``, the "oracle" for each task
is CALIBRATED: we submit the strongest available candidate solutions (reference*.cpp
first, then the strongest model solutions) and keep the highest-scoring / first-AC one
as that task's oracle (recorded with its score). Candidate sources + per-problem
metadata (``config.yaml``: type / time-limit / n_cases) are read from the LOCAL
algorithmic mirror; the judge itself grades against slime-data's identical problems
(pids are intersected with the judge's ``GET /problems`` to be safe).

Phases (each writes its artifacts so a crash never loses prior work):
    boot       build image + boot the vm_runtime judge sandbox; persist url+sandbox_id+nproc
    stratify   pick ~N problems stratified by worst-case grade cost (n_cases x time_limit)
    calibrate  per problem, find the best/AC candidate -> the task's oracle
    phase1     per task, 10 trials of its oracle at concurrency=1 -> per-task p50/p90 eval time
    phase2     concurrency sweep (50/100/200): fire N oracle submits at once -> latency under load
    report     aggregate everything into REPORT.md
    run        stratify -> calibrate -> phase1 -> phase2 -> report (reads boot's judge.json)
    teardown   terminate the judge sandbox

HTTP: the judge accepts a JSON ``POST /submit {pid,lang,code} -> {sid}`` and
``GET /result/<sid> -> {status, score(0..100), passed, cases[...]}``; we use stdlib
urllib only (no third-party deps in the --no-dev venv).

    uv run --no-dev python agentic_rl/profiles/judge_profile.py boot     --run-dir RD
    uv run --no-dev python agentic_rl/profiles/judge_profile.py smoke    --run-dir RD
    uv run --no-dev python agentic_rl/profiles/judge_profile.py run      --run-dir RD
    uv run --no-dev python agentic_rl/profiles/judge_profile.py teardown --run-dir RD
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# repo root on sys.path so `agentic_rl.environment...` imports regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Local algorithmic mirror: problem configs + candidate solutions (NOT what the judge
# grades against — the judge reads slime-data — but the pids/configs are identical).
ALG_ROOT = Path(os.environ.get("FRONTIER_CS_ALG_ROOT", "/Users/junlin/Documents/Research/Misc/Frontier-CS/algorithmic"))
PROBLEMS_DIR = ALG_ROOT / "problems"
SOLUTIONS_DIR = ALG_ROOT / "solutions"

# Candidate priority for oracle calibration: reference* first, then strongest models.
# Within a model we try the base file then _1.._4. `.FAILED` files have no .cpp ext so
# the *.cpp glob already drops them; we still guard on the name.
MODEL_RANK = [
    "gpt5.2", "gemini3pro", "gpt5.1", "gpt5", "deepseekreasoner", "gemini2.5pro",
    "grok4fastreasoning", "claude4.5sonnet", "claude4.1opus", "trinitylargethinking",
    "gpt5_high", "gpt5_medium", "gpt5_low",
]
CALIBRATION_BUDGET = 3  # max candidates tried per problem before taking the best seen
AC_SCORE = 99.5         # near-perfect: treat as "oracle found", stop trying more candidates
                        # (Frontier-CS problems are often optimization/partial-credit, so a
                        #  full 100 / all-cases-AC is frequently unattainable even for refs)

log = logging.getLogger("judge_profile")


# ── HTTP helpers (stdlib only) ────────────────────────────────────────────────
def _req(url: str, *, data: bytes | None = None, headers: dict | None = None, timeout: float = 30.0):
    r = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.status, resp.read()


def submit(judge_url: str, pid: str, code: str, timeout: float = 60.0) -> int:
    body = json.dumps({"pid": str(pid), "lang": "cpp", "code": code}).encode()
    status, raw = _req(f"{judge_url}/submit", data=body, headers={"Content-Type": "application/json"}, timeout=timeout)
    if status != 200:
        raise RuntimeError(f"submit HTTP {status}: {raw[:200]!r}")
    return int(json.loads(raw)["sid"])


def fetch_full(judge_url: str, sid: int, timeout: float = 30.0) -> dict:
    status, raw = _req(f"{judge_url}/result/{sid}", timeout=timeout)
    return json.loads(raw) if status == 200 else {"status": f"http_{status}"}


def grade_once(judge_url: str, pid: str, code: str, *, cap: float, poll: float = 0.5) -> dict:
    """Submit `code` for `pid`, poll to completion. Returns a flat record with the
    client-observed timings. `cap` is the max seconds to wait for a verdict.

    NB: the judge's getResult() is consume-once — the FIRST read of a done/error
    result deletes it from cache (then a racy disk fallback). So we poll the FULL
    result and capture it on the first done/error (matching the production
    evaluate.py); never read a finished sid twice.
    """
    rec = {"pid": str(pid), "ok": False, "status": "", "score": None, "passed": None,
           "cases_total": 0, "cases_passed": 0, "t_submit": None, "t_grade": None, "t_total": None, "error": ""}
    t0 = time.monotonic()
    try:
        sid = submit(judge_url, pid, code)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"submit:{type(e).__name__}:{e}"[:300]
        rec["status"] = "submit_failed"
        rec["t_total"] = round(time.monotonic() - t0, 3)
        return rec
    t_sid = time.monotonic()
    rec["t_submit"] = round(t_sid - t0, 3)
    rec["sid"] = sid
    deadline = t_sid + cap
    final, final_status = None, "timeout"
    while time.monotonic() < deadline:
        try:
            full = fetch_full(judge_url, sid)
        except Exception:  # noqa: BLE001
            full = {}
        st = full.get("status", "")
        if st in ("done", "error"):
            final, final_status = full, st
            break
        time.sleep(poll)
    t_done = time.monotonic()
    rec["t_grade"] = round(t_done - t_sid, 3)
    rec["t_total"] = round(t_done - t0, 3)
    rec["status"] = final_status
    if final_status == "done" and final is not None:
        cases = final.get("cases") or []
        rec["ok"] = True
        rec["score"] = float(final.get("score") or 0.0)
        rec["passed"] = bool(final.get("passed")) or rec["score"] >= 100.0 - 1e-6
        rec["cases_total"] = len(cases) if isinstance(cases, list) else 0
        rec["cases_passed"] = sum(1 for c in cases if isinstance(c, dict) and (c.get("scoreRatio") or 0) >= 1.0)
    elif final_status == "error" and final is not None:
        rec["error"] = str(final.get("message") or final.get("error") or "judge error")[:300]
    return rec


# ── local problem metadata + candidate solutions ─────────────────────────────
def parse_config(pid: str) -> dict:
    """Minimal config.yaml parse (type / time seconds / total n_cases). Avoids a
    PyYAML dep — the schema is flat and stable."""
    cfg = {"pid": str(pid), "type": "default", "time_sec": 1.0, "n_cases": 0}
    p = PROBLEMS_DIR / str(pid) / "config.yaml"
    if not p.is_file():
        return cfg
    for line in p.read_text(errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("type:"):
            cfg["type"] = s.split(":", 1)[1].strip()
        elif s.startswith("time:"):
            v = s.split(":", 1)[1].strip().rstrip("s").strip()
            try:
                cfg["time_sec"] = float(v)
            except ValueError:
                pass
        elif "n_cases:" in s:
            v = s.split("n_cases:", 1)[1].split("#", 1)[0].strip()
            try:
                cfg["n_cases"] += int(v)
            except ValueError:
                pass
    cfg["worst_case_sec"] = round(cfg["n_cases"] * cfg["time_sec"], 1)
    return cfg


def candidate_solutions(pid: str) -> list[tuple[str, Path]]:
    """Ordered (name, path) candidate solutions for a pid: reference* first, then
    strongest models (base file then _1.._4)."""
    d = SOLUTIONS_DIR / str(pid)
    if not d.is_dir():
        return []
    cpps = {f.stem: f for f in d.glob("*.cpp") if "FAILED" not in f.name}
    ordered: list[tuple[str, Path]] = []
    seen: set[str] = set()
    # references
    for stem in sorted(cpps):
        if "reference" in stem.lower() and stem not in seen:
            ordered.append((stem, cpps[stem])); seen.add(stem)
    # models in rank order, base then numbered variants
    for model in MODEL_RANK:
        for suffix in ["", "_1", "_2", "_3", "_4"]:
            stem = f"{model}{suffix}"
            if stem in cpps and stem not in seen:
                ordered.append((stem, cpps[stem])); seen.add(stem)
    # anything else left
    for stem in sorted(cpps):
        if stem not in seen:
            ordered.append((stem, cpps[stem])); seen.add(stem)
    return ordered


# ── small utilities ───────────────────────────────────────────────────────────
def pct(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in [0,100]); None for empty input."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return round(xs[0], 3)
    k = (len(xs) - 1) * (q / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path):
    return json.loads(path.read_text())


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log")],
    )


# ── phases ─────────────────────────────────────────────────────────────────────
def phase_boot(args) -> None:
    rd = Path(args.run_dir)
    os.environ.setdefault("MODAL_ENVIRONMENT", args.modal_env)
    os.environ["FRONTIER_CS_JUDGE_CPU"] = str(args.judge_cpu)
    os.environ["FRONTIER_CS_JUDGE_MEMORY_MB"] = str(args.judge_mem)
    os.environ.pop("FRONTIER_CS_JUDGE_URL", None)  # force a fresh boot

    from agentic_rl.environment.verifier_server import autostart

    log.info("booting judge sandbox (cpu=%s mem=%sMB env=%s)…", args.judge_cpu, args.judge_mem, args.modal_env)
    t0 = time.monotonic()
    url = autostart.ensure_started()
    boot_sec = round(time.monotonic() - t0, 1)
    sb = autostart._JUDGE_SANDBOX
    sandbox_id = getattr(sb, "object_id", "") or ""

    nproc = workers_line = ""
    try:  # the real worker count (go-judge -parallelism / JUDGE_WORKERS default to nproc)
        proc = sb.exec("nproc")
        nproc = (proc.stdout.read() or "").strip()
    except Exception as e:  # noqa: BLE001
        log.warning("nproc probe failed: %s", e)
    # confirm problems visible to the judge
    try:
        status, raw = _req(f"{url}/problems", timeout=30)
        judge_pids = [str(p.get("id") or p.get("pid") or p) if isinstance(p, dict) else str(p)
                      for p in (json.loads(raw).get("problems") or [])]
    except Exception as e:  # noqa: BLE001
        log.warning("/problems probe failed: %s", e)
        judge_pids = []

    info = {"url": url, "sandbox_id": sandbox_id, "cpu": args.judge_cpu, "mem_mb": args.judge_mem,
            "modal_env": args.modal_env, "boot_sec": boot_sec, "nproc": nproc,
            "n_judge_problems": len(judge_pids), "judge_pids": sorted(judge_pids)}
    write_json(rd / "judge.json", info)
    log.info("judge ready url=%s sandbox=%s boot=%ss nproc=%s problems=%d",
             url, sandbox_id[:12], boot_sec, nproc, len(judge_pids))
    log.info("NOTE: go-judge parallelism / JUDGE_WORKERS default to nproc=%s -> that is the judge's real worker count", nproc)


def _judge(rd: Path) -> tuple[str, dict]:
    info = load_json(rd / "judge.json")
    return info["url"], info


def phase_stratify(args) -> None:
    rd = Path(args.run_dir)
    _, info = _judge(rd)
    judge_pids = set(info.get("judge_pids") or [])
    # candidate pool: has local config + has candidate solutions + visible to judge
    pool = []
    for d in sorted(SOLUTIONS_DIR.iterdir()):
        if not d.is_dir():
            continue
        pid = d.name
        if judge_pids and pid not in judge_pids:
            continue
        if not candidate_solutions(pid):
            continue
        cfg = parse_config(pid)
        if cfg["n_cases"] == 0:
            continue
        pool.append(cfg)
    pool.sort(key=lambda c: c["worst_case_sec"])
    n = min(args.subset_size, len(pool))
    # stratified pick: split sorted pool into n buckets, take the median of each
    import random
    rng = random.Random(args.seed)
    chosen, bsz = [], len(pool) / n
    for i in range(n):
        lo, hi = int(i * bsz), int((i + 1) * bsz)
        bucket = pool[lo:max(hi, lo + 1)]
        chosen.append(rng.choice(bucket))
    # always include the verified-reference problems if present
    for must in ("0", "263"):
        if must in judge_pids and not any(c["pid"] == must for c in chosen):
            cfg = parse_config(must)
            if cfg["n_cases"]:
                chosen.append(cfg)
    # dedupe, keep stable order by worst_case
    uniq = {c["pid"]: c for c in chosen}
    chosen = sorted(uniq.values(), key=lambda c: c["worst_case_sec"])
    write_json(rd / "stratification.json", {
        "subset_size": args.subset_size, "seed": args.seed, "pool_size": len(pool),
        "chosen": chosen,
        "n_interactive": sum(1 for c in chosen if c["type"] == "interactive"),
    })
    log.info("stratified %d/%d problems (%d interactive); worst_case_sec range %.0f..%.0f",
             len(chosen), len(pool), sum(1 for c in chosen if c["type"] == "interactive"),
             chosen[0]["worst_case_sec"], chosen[-1]["worst_case_sec"])


def _grade_cap(cfg: dict, *, floor: float, mult: float) -> float:
    """Generous per-submission verdict cap = worst_case x mult, bounded below by floor."""
    return max(floor, cfg["worst_case_sec"] * mult + 30.0)


def phase_calibrate(args) -> None:
    rd = Path(args.run_dir)
    url, _ = _judge(rd)
    chosen = load_json(rd / "stratification.json")["chosen"]
    only = set(args.only.split(",")) if args.only else None
    cal_log = rd / "calibration.jsonl"
    oracles = {}
    if (rd / "oracles.json").exists():
        oracles = load_json(rd / "oracles.json")
    for cfg in chosen:
        pid = cfg["pid"]
        if only and pid not in only:
            continue
        if pid in oracles and not args.force:
            continue
        cands = candidate_solutions(pid)[:CALIBRATION_BUDGET]
        cap = _grade_cap(cfg, floor=120.0, mult=1.5)
        best = None
        for name, path in cands:
            code = path.read_text(errors="ignore")
            rec = grade_once(url, pid, code, cap=cap)
            rec.update({"phase": "calibrate", "candidate": name, "n_cases": cfg["n_cases"],
                        "time_sec": cfg["time_sec"], "type": cfg["type"]})
            append_jsonl(cal_log, rec)
            log.info("[calib] pid=%s cand=%s status=%s score=%s cases=%s/%s t_grade=%.1fs",
                     pid, name, rec["status"], rec["score"], rec["cases_passed"], rec["cases_total"], rec["t_grade"] or 0)
            score = rec["score"] if rec["score"] is not None else -1
            if best is None or score > best["score"]:
                best = {"candidate": name, "score": score, "passed": rec["passed"],
                        "cases_total": rec["cases_total"], "cases_passed": rec["cases_passed"],
                        "t_grade": rec["t_grade"], "code_path": str(path)}
            if rec["passed"] or (rec["score"] or 0) >= AC_SCORE:
                break
        if best is not None:
            best.update({"pid": pid, "n_cases": cfg["n_cases"], "time_sec": cfg["time_sec"],
                         "type": cfg["type"], "worst_case_sec": cfg["worst_case_sec"]})
            oracles[pid] = best
            write_json(rd / "oracles.json", oracles)
    n_ac = sum(1 for o in oracles.values() if o["passed"])
    log.info("calibrated %d oracles (%d AC / %d partial)", len(oracles), n_ac, len(oracles) - n_ac)


def phase1(args) -> None:
    """Per-task clean latency: TRIALS sequential (concurrency=1) submissions of each
    task's oracle. Sequential so numbers are uncontended per-task eval times."""
    rd = Path(args.run_dir)
    url, _ = _judge(rd)
    oracles = load_json(rd / "oracles.json")
    trials_log = rd / "phase1_trials.jsonl"
    summary = []
    for pid, orc in sorted(oracles.items(), key=lambda kv: kv[1]["worst_case_sec"]):
        code = Path(orc["code_path"]).read_text(errors="ignore")
        cap = _grade_cap(orc, floor=120.0, mult=2.0)
        grades = []
        for t in range(args.trials):
            rec = grade_once(url, pid, code, cap=cap)
            rec.update({"phase": "phase1", "trial": t, "candidate": orc["candidate"]})
            append_jsonl(trials_log, rec)
            grades.append(rec["t_grade"] if rec["status"] == "done" else None)
        ok = [g for g in grades if g is not None]
        row = {"pid": pid, "type": orc["type"], "n_cases": orc["n_cases"], "time_sec": orc["time_sec"],
               "oracle": orc["candidate"], "oracle_score": orc["score"], "oracle_passed": orc["passed"],
               "trials": args.trials, "n_ok": len(ok),
               "p50": pct(ok, 50), "p90": pct(ok, 90), "mean": round(statistics.mean(ok), 3) if ok else None,
               "min": round(min(ok), 3) if ok else None, "max": round(max(ok), 3) if ok else None,
               "std": round(statistics.pstdev(ok), 3) if len(ok) > 1 else 0.0}
        summary.append(row)
        write_json(rd / "phase1_summary.json", summary)
        log.info("[phase1] pid=%s oracle=%s p50=%.2fs p90=%.2fs (n=%d/%d) cases=%d tl=%.0fs",
                 pid, orc["candidate"], row["p50"] or 0, row["p90"] or 0, len(ok), args.trials,
                 orc["n_cases"], orc["time_sec"])
    _write_phase1_csv(rd, summary)


def _phase2_pool(oracles: dict, max_grade: float) -> list[dict]:
    """Oracles to sample under load: those whose calibrated grade <= max_grade (so
    latency reflects QUEUEING, not a few slow solutions); fall back to the fastest if
    too few clear the bar. Sorted by worst_case for stable seeded sampling — identical
    across cpu levels when the same oracles.json is reused, so the workload is held
    constant and only the worker count varies."""
    graded = [o for o in oracles.values() if o.get("t_grade")]
    fast = [o for o in graded if o["t_grade"] <= max_grade]
    pool = fast if len(fast) >= 8 else sorted(graded, key=lambda o: o["t_grade"])[:max(8, len(graded) // 2)]
    pool = pool or list(oracles.values())
    return sorted(pool, key=lambda o: o["worst_case_sec"])


def phase_warm(args) -> None:
    """Pre-compile/cache each phase2-pool problem's checker/interactor so the sweep runs
    WARM — matching the cpu=8 baseline (where phase2 followed calibration+phase1). Without
    this a fresh judge pays the cold per-problem compile tax mid-sweep and the queueing
    numbers aren't comparable."""
    rd = Path(args.run_dir)
    url, _ = _judge(rd)
    oracles = load_json(rd / "oracles.json")
    pool = _phase2_pool(oracles, args.phase2_max_grade)
    log.info("[warm] warming %d pool problems (sequential)…", len(pool))
    for o in pool:
        code = Path(o["code_path"]).read_text(errors="ignore")
        rec = grade_once(url, o["pid"], code, cap=_grade_cap(o, floor=150.0, mult=2.0))
        log.info("[warm] pid=%s status=%s t_grade=%.1fs", o["pid"], rec["status"], rec["t_grade"] or 0)


def phase2(args) -> None:
    """Concurrency sweep: for each level N, fire N oracle submissions ~simultaneously
    (sampled with replacement from the calibrated oracle pool) and measure latency +
    throughput + errors as they contend for the judge's nproc workers."""
    rd = Path(args.run_dir)
    url, info = _judge(rd)
    oracles = load_json(rd / "oracles.json")
    pool = _phase2_pool(oracles, args.phase2_max_grade)
    log.info("[phase2] oracle pool: %d fast (<=%.0fs grade) of %d calibrated",
             len(pool), args.phase2_max_grade, len(oracles))
    import random
    rng = random.Random(args.seed)
    reqs_log = rd / "phase2_requests.jsonl"
    summary = []
    levels = [int(x) for x in args.concurrency.split(",")]
    for level in levels:
        picks = [rng.choice(pool) for _ in range(level)]
        # per-request cap (generous under load) so one slow problem can't pin the whole level
        # cap must tolerate contention: ~level/nproc x base grade time; floor 900s covers
        # fast oracles at level 200 (e.g. 15s grade x 200/8 ~ 375s).
        codes = [(o["pid"], Path(o["code_path"]).read_text(errors="ignore"), o,
                  _grade_cap(o, floor=900.0, mult=4.0)) for o in picks]
        log.info("[phase2] level=%d firing %d concurrent submits (judge nproc=%s)…",
                 level, level, info.get("nproc"))
        results: list[dict] = []
        t_start = time.monotonic()
        with ThreadPoolExecutor(max_workers=level) as ex:
            futs = {ex.submit(grade_once, url, pid, code, cap=cap): o for pid, code, o, cap in codes}
            for fut in as_completed(futs):
                o = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:  # noqa: BLE001
                    rec = {"pid": o["pid"], "status": "exception", "error": str(e)[:200], "t_total": None}
                rec.update({"phase": "phase2", "level": level, "oracle": o["candidate"]})
                append_jsonl(reqs_log, rec)
                results.append(rec)
        drain = round(time.monotonic() - t_start, 2)
        lat = [r["t_total"] for r in results if r.get("status") == "done"]
        n_done = len(lat)
        n_err = sum(1 for r in results if r.get("status") not in ("done",))
        row = {"level": level, "n": level, "n_done": n_done, "n_error": n_err,
               "error_rate": round(n_err / level, 3),
               "drain_sec": drain, "throughput_per_s": round(n_done / drain, 2) if drain else None,
               "lat_p50": pct(lat, 50), "lat_p90": pct(lat, 90), "lat_p99": pct(lat, 99),
               "lat_min": round(min(lat), 2) if lat else None, "lat_max": round(max(lat), 2) if lat else None,
               "lat_mean": round(statistics.mean(lat), 2) if lat else None}
        summary.append(row)
        write_json(rd / "phase2_summary.json", summary)
        log.info("[phase2] level=%d done=%d err=%d drain=%.1fs thru=%.2f/s lat p50=%.1fs p90=%.1fs p99=%.1fs max=%.1fs",
                 level, n_done, n_err, drain, row["throughput_per_s"] or 0,
                 row["lat_p50"] or 0, row["lat_p90"] or 0, row["lat_p99"] or 0, row["lat_max"] or 0)
        # clear judge in-memory submission cache between levels
        try:
            _req(f"{url}/submissions/reset", data=b"{}", headers={"Content-Type": "application/json"}, timeout=30)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    _write_phase2_csv(rd, summary)


def _write_phase1_csv(rd: Path, rows: list[dict]) -> None:
    cols = ["pid", "type", "n_cases", "time_sec", "oracle", "oracle_score", "oracle_passed",
            "n_ok", "trials", "p50", "p90", "mean", "min", "max", "std"]
    _write_csv(rd / "phase1_summary.csv", cols, rows)


def _write_phase2_csv(rd: Path, rows: list[dict]) -> None:
    cols = ["level", "n_done", "n_error", "error_rate", "drain_sec", "throughput_per_s",
            "lat_p50", "lat_p90", "lat_p99", "lat_min", "lat_max", "lat_mean"]
    _write_csv(rd / "phase2_summary.csv", cols, rows)


def _write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    import csv
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def phase_report(args) -> None:
    rd = Path(args.run_dir)
    info = load_json(rd / "judge.json")
    strat = load_json(rd / "stratification.json") if (rd / "stratification.json").exists() else {}
    p1 = load_json(rd / "phase1_summary.json") if (rd / "phase1_summary.json").exists() else []
    p2 = load_json(rd / "phase2_summary.json") if (rd / "phase2_summary.json").exists() else []
    oracles = load_json(rd / "oracles.json") if (rd / "oracles.json").exists() else {}
    n_ac = sum(1 for o in oracles.values() if o["passed"])
    L = []
    L.append("# Frontier-CS judge profiling report\n")
    L.append(f"- Judge: `{info['url']}` (Modal vm_runtime sandbox `{info['sandbox_id'][:12]}`, "
             f"cpu={info['cpu']}, mem={info['mem_mb']}MB, **nproc={info.get('nproc')}** = real go-judge worker count)")
    L.append(f"- Boot: {info.get('boot_sec')}s · problems visible to judge: {info.get('n_judge_problems')}")
    L.append(f"- Subset: {len(p1)} problems profiled (stratified by n_cases × time-limit; "
             f"{strat.get('n_interactive', '?')} interactive) · oracle calibration: {n_ac} AC / {len(oracles) - n_ac} partial\n")
    L.append("> Oracle = highest-scoring calibrated candidate per task (reference*.cpp where present, else strongest model solution). "
             "Phase-1 latencies are uncontended (concurrency=1); phase-2 fires AC oracles concurrently so latency reflects judge queueing, "
             "not solution slowness — real training mixes in slower TLE submissions.\n")

    if p2:
        L.append("## Concurrency sweep (end-to-end submit→verdict latency)\n")
        L.append("| concurrency | done | errors | drain (s) | throughput (/s) | p50 (s) | p90 (s) | p99 (s) | max (s) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in p2:
            L.append(f"| {r['level']} | {r['n_done']} | {r['n_error']} | {r['drain_sec']} | "
                     f"{r['throughput_per_s']} | {r['lat_p50']} | {r['lat_p90']} | {r['lat_p99']} | {r['lat_max']} |")
        L.append("")

    if p1:
        oks = [r for r in p1 if r["p50"] is not None]
        allp50 = [r["p50"] for r in oks]
        allp90 = [r["p90"] for r in oks]
        L.append("## Per-task oracle eval time (concurrency=1, 10 trials)\n")
        L.append(f"Across {len(oks)} tasks: median-of-p50 = {pct(allp50,50)}s, "
                 f"p90-of-p50 = {pct(allp50,90)}s, max p90 = {max(allp90) if allp90 else None}s\n")
        L.append("| pid | type | cases | TL(s) | oracle | score | p50 (s) | p90 (s) | mean | min | max |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(p1, key=lambda x: (x["p90"] or 0)):
            L.append(f"| {r['pid']} | {r['type']} | {r['n_cases']} | {r['time_sec']} | {r['oracle']} | "
                     f"{r['oracle_score']} | {r['p50']} | {r['p90']} | {r['mean']} | {r['min']} | {r['max']} |")
        L.append("")
    L.append("## Artifacts\n")
    L.append("- `judge.json` · `stratification.json` · `oracles.json`")
    L.append("- `calibration.jsonl` · `phase1_trials.jsonl` · `phase2_requests.jsonl`")
    L.append("- `phase1_summary.{json,csv}` · `phase2_summary.{json,csv}`")
    (rd / "REPORT.md").write_text("\n".join(L))
    log.info("wrote %s", rd / "REPORT.md")


def phase_smoke(args) -> None:
    """Quick end-to-end validation on a couple problems before the long run."""
    rd = Path(args.run_dir)
    url, info = _judge(rd)
    judge_pids = set(info.get("judge_pids") or [])
    test_pids = [p for p in ("0", "263") if (not judge_pids) or p in judge_pids][: args.smoke_n] or ["0"]
    for pid in test_pids:
        cfg = parse_config(pid)
        cands = candidate_solutions(pid)
        if not cands:
            log.warning("[smoke] pid=%s no candidates", pid); continue
        name, path = cands[0]
        cap = _grade_cap(cfg, floor=120.0, mult=2.0)
        rec = grade_once(url, pid, path.read_text(errors="ignore"), cap=cap)
        log.info("[smoke] pid=%s cand=%s -> status=%s score=%s cases=%s/%s t_submit=%.2fs t_grade=%.2fs err=%s",
                 pid, name, rec["status"], rec["score"], rec["cases_passed"], rec["cases_total"],
                 rec["t_submit"] or 0, rec["t_grade"] or 0, rec["error"])


def phase_diag(args) -> None:
    """Probe the judge sandbox's resource limits + summarize non-done phase2 records
    (with the go-judge error body, once judge_engine.js surfaces it). Run after a
    reproduction burst to pin WHY submissions 500."""
    rd = Path(args.run_dir)
    info = load_json(rd / "judge.json")
    os.environ.setdefault("MODAL_ENVIRONMENT", info.get("modal_env", args.modal_env))
    import modal
    probe = (
        "echo nproc=$(nproc); "
        "echo pid_max=$(cat /proc/sys/kernel/pid_max 2>/dev/null); "
        "echo ulimit_u=$(bash -c 'ulimit -u'); echo ulimit_n=$(bash -c 'ulimit -n'); "
        "echo cg_pids_max=$(cat /sys/fs/cgroup/pids.max 2>/dev/null); "
        "echo cg_pids_cur=$(cat /sys/fs/cgroup/pids.current 2>/dev/null); "
        "echo cg_mem_max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null); "
        "echo cg_mem_cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null); "
        "echo loadavg=$(cat /proc/loadavg)"
    )
    try:
        p = modal.Sandbox.from_id(info["sandbox_id"]).exec("bash", "-lc", probe)
        log.info("[diag] sandbox limits:\n%s", (p.stdout.read() or "").strip())
    except Exception as e:  # noqa: BLE001
        log.warning("[diag] sandbox probe failed: %s", e)
    rl = rd / "phase2_requests.jsonl"
    if rl.exists():
        from collections import Counter
        recs = [json.loads(l) for l in rl.read_text().splitlines() if l.strip()]
        nd = [r for r in recs if r.get("status") != "done"]
        log.info("[diag] %d/%d phase2 records non-done", len(nd), len(recs))
        for (st, er, pid), n in Counter((r.get("status"), (r.get("error") or "")[:200], r.get("pid")) for r in nd).most_common():
            log.info("[diag]   x%d pid=%s status=%s err=%s", n, pid, st, er)


def phase_teardown(args) -> None:
    rd = Path(args.run_dir)
    info = load_json(rd / "judge.json")
    os.environ.setdefault("MODAL_ENVIRONMENT", info.get("modal_env", args.modal_env))
    import modal
    try:
        modal.Sandbox.from_id(info["sandbox_id"]).terminate()
        log.info("terminated judge sandbox %s", info["sandbox_id"][:12])
    except Exception as e:  # noqa: BLE001
        log.warning("teardown failed (may already be gone): %s", e)


def phase_run(args) -> None:
    phase_stratify(args)
    phase_calibrate(args)
    phase1(args)
    phase2(args)
    phase_report(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["boot", "stratify", "calibrate", "phase1", "phase2",
                                       "report", "smoke", "warm", "diag", "run", "teardown"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--modal-env", default="junlin-dev")
    ap.add_argument("--judge-cpu", type=int, default=32)
    ap.add_argument("--judge-mem", type=int, default=65536)
    ap.add_argument("--subset-size", type=int, default=40)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--concurrency", default="50,100,200")
    ap.add_argument("--phase2-max-grade", type=float, default=30.0,
                    help="phase2 samples only oracles whose calibrated grade <= this (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-n", type=int, default=2)
    ap.add_argument("--only", default="", help="comma pids to (re)calibrate only")
    ap.add_argument("--force", action="store_true", help="recompute even if artifact exists")
    args = ap.parse_args()

    setup_logging(Path(args.run_dir))
    {"boot": phase_boot, "stratify": phase_stratify, "calibrate": phase_calibrate,
     "phase1": phase1, "phase2": phase2, "report": phase_report, "smoke": phase_smoke,
     "warm": phase_warm, "diag": phase_diag, "run": phase_run, "teardown": phase_teardown}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
