"""Rebuild the staleness-tab series in frontier-cs-retro-replay.html.

Merges every fault-tolerance retry inside each W&B run *group* (a single run id
shows "failed/crashed" with stale history while training continues in a newer
retry), then prints one JSON blob of per-step series. The report samples it
every five updates onto a 7-step trailing mean and embeds the result as
``var stal = {...}`` in the page script.

Deliberately omits ``behavior_lag/rejected_groups`` and
``dynamic_sampling/completed_groups``: a since-fixed requeue bug recycled
lag-rejected groups in the rlag4/rlag8 images, inflating both counters.

    uv run --with wandb python agentic_rl/progress/retro-replay/refresh_staleness_series.py
"""

import wandb, json, statistics as st
api = wandb.Api(timeout=300)
PROJ = "junlinwang/Modal"

def merge_group(prefix, exclude, keys):
    runs = [r for r in api.runs(PROJ) if (r.name or "").startswith(prefix)
            and not any(e in (r.name or "") for e in exclude)]
    runs.sort(key=lambda r: r.created_at)
    m = {}
    for r in runs:
        for row in r.scan_history(keys=["rollout/step"]+keys, page_size=500):
            s = row.get("rollout/step")
            if s is None: continue
            for k in keys:
                if row.get(k) is not None: m.setdefault(int(s), {})[k] = row[k]
    return m

def run_series(rid, keys):
    r = api.run(PROJ+"/"+rid); m = {}
    for row in r.scan_history(keys=["rollout/step"]+keys, page_size=500):
        s = row.get("rollout/step")
        if s is None: continue
        for k in keys:
            if row.get(k) is not None: m.setdefault(int(s), {})[k] = row[k]
    return m

RK = "agentic/outcome/reward_final/mean"
# Pre-filter fresh-leg reward: mean over every scored group BEFORE the DAPO
# nonzero-std drop (RK is a post-filter batch mean and is survivorship-biased;
# see the report's 2026-08-24 metric-basis audit).
PK = "dynamic_sampling/raw_reward_all"
AK = ["agentic/budget_hit_frac","async/fresh/behavior_lag/max","async/retro/behavior_lag/max",
      "perf/step_time","agentic/decode_tok_per_s/mean","agentic/solved_frac","agentic/elapsed_sec/p50"]

src = {}
src["oldP50_reward"] = run_series("p50canon85", [RK, "agentic/solved_frac"])
src["oldP50_agentic"] = merge_group("qwen3.6-27b-frontier-cs-retro-a-final-p50",
                                    ["rlag","canonical","p50-v2","rdepth"], AK+[RK, PK])
for tag, grp in [("rlag4","qwen3.6-27b-frontier-cs-retro-a-final-p50-rlag4"),
                 ("rlag8","qwen3.6-27b-frontier-cs-retro-a-final-p50-rlag8"),
                 ("p50_v2","qwen3.6-27b-frontier-cs-retro-a-final-p50-v2-20260821-010000"),
                 ("rdepth2","qwen3.6-27b-frontier-cs-retro-a-final-p50-rdepth2"),
                 ("rdepth4","qwen3.6-27b-frontier-cs-retro-a-final-p50-rdepth4-20260821-020000")]:
    src[tag] = merge_group(grp, [], [RK, PK]+AK+["agentic/solved_frac"])

def trailing(m, key, win=7, upto=84):
    out = []
    vals = {s: m[s][key] for s in m if key in m[s]}
    for s in range(0, upto+1):
        w = [vals[t] for t in range(max(0,s-win+1), s+1) if t in vals]
        out.append(round(st.mean(w), 5) if w else None)
    return out

D = {}
D["steps"] = list(range(0, 85))
D["reward"] = {
  "oldP50": trailing(src["oldP50_reward"], RK),
  "rlag4":  trailing(src["rlag4"], RK),
  "rlag8":  trailing(src["rlag8"], RK),
}
D["budget"] = {
  "oldP50": trailing(src["oldP50_agentic"], "agentic/budget_hit_frac"),
  "rlag4":  trailing(src["rlag4"], "agentic/budget_hit_frac"),
  "rlag8":  trailing(src["rlag8"], "agentic/budget_hit_frac"),
}
D["lag"] = {}
for tag, key in [("rlag4","async/fresh/behavior_lag/max"),("rlag8","async/fresh/behavior_lag/max")]:
    D["lag"]["fresh_"+tag] = trailing(src[tag], key, win=1)
for tag in ("rlag4","rlag8"):
    D["lag"]["retro_"+tag] = trailing(src[tag], "async/retro/behavior_lag/max", win=1)
D["solved"] = {
  "oldP50": trailing(src["oldP50_reward"], "agentic/solved_frac"),
  "rlag4":  trailing(src["rlag4"], "agentic/solved_frac"),
  "rlag8":  trailing(src["rlag8"], "agentic/solved_frac"),
}
# Post- and pre-filter reward for all six runs (the report's paired audit charts;
# embedded at the 18 stalSteps sample points [0,5,...,80,84]).
SIX = [("oldP50", "oldP50_agentic"), ("rlag4", "rlag4"), ("rlag8", "rlag8"),
       ("p50_v2", "p50_v2"), ("rdepth2", "rdepth2"), ("rdepth4", "rdepth4")]
D["rewardPost6"] = {tag: trailing(src[key], RK) for tag, key in SIX}
D["rewardPre6"] = {tag: trailing(src[key], PK) for tag, key in SIX}
# cumulative pure compute hours, per arm, using own step_time (median-filled)
D["compute"] = {}
for tag, m in [("oldP50", src["oldP50_agentic"]), ("rlag4", src["rlag4"]), ("rlag8", src["rlag8"])]:
    sts = [m[s]["perf/step_time"] for s in sorted(m) if "perf/step_time" in m[s]]
    med = st.median(sts) if sts else 0
    cum, tot = [], 0.0
    for s in range(0, 85):
        tot += (m.get(s, {}).get("perf/step_time") or med)
        cum.append(round(tot/3600, 3))
    D["compute"][tag] = {"cum_h": cum, "median_step_s": round(med,1), "n_step_time": len(sts)}
# headline scalars over last 15 steps
D["kpi"] = {}
for tag, rm, am in [("oldP50", src["oldP50_reward"], src["oldP50_agentic"]),
                    ("rlag4", src["rlag4"], src["rlag4"]), ("rlag8", src["rlag8"], src["rlag8"])]:
    def w(m, k, lo=70, hi=84):
        xs = [m[s][k] for s in sorted(m) if lo<=s<=hi and k in m[s]]
        return round(st.mean(xs),4) if xs else None
    D["kpi"][tag] = {
      "reward_70_84": w(rm, RK), "solved_70_84": w(rm, "agentic/solved_frac"),
      "budget_hit": w(am, "agentic/budget_hit_frac"), "decode": w(am, "agentic/decode_tok_per_s/mean"),
      "step_s": w(am, "perf/step_time"), "fresh_lag_max": w(am, "async/fresh/behavior_lag/max"),
      "retro_lag_max": w(am, "async/retro/behavior_lag/max"), "ep_p50": w(am, "agentic/elapsed_sec/p50"),
    }
print(json.dumps(D))
