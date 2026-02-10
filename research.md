# PEFT (LoRA/DoRA) Holistic Audit Since `yoyovlss` (`58ee7489`)

## 1) Scope and Method

This report treats `58ee7489` as the baseline (the revision you referred to as `yoyovlss`) and audits all PEFT-relevant behavior added after that point across:

- `slime`
- `~/Megatron-LM`
- `~/Megatron-Bridge`
- runtime patch surface in `slime/docker/patch/latest/megatron.patch`

I traced:

1. model construction and PEFT injection,
2. checkpoint load path,
3. LoRA/DoRA merge path used during rollout weight sync,
4. TP/PP/EP/ETP + MoE-specific utility behavior,
5. CI/test coverage and blind spots.

---

## 2) What Changed Since Baseline

### 2.1 Commits since `58ee7489`

1. `c2330e88` `initial lora support`
2. `a92f1e46` `lora weight sync: use tensor identity matching and precompute TP-aware deltas`
3. `451dd527` `add DoRA support to LoRA weight sync and model initialization`
4. `9873870a` `refactor: make lora + dora ci-test friendly`

### 2.2 Files changed in committed PEFT work

- `slime/backends/megatron_utils/model_provider.py`
- `slime/backends/megatron_utils/model.py`
- `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py`
- `slime/backends/megatron_utils/ci_utils.py`
- `slime/utils/arguments.py`

### 2.3 Additional in-progress PEFT-adjacent working-tree edits

- `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `slime/backends/sglang_utils/sglang_engine.py`
- `tests/test_qwen3_0.6B_peft_parallel_matrix.py`
- `tests/utils/test_peft_exact_math.py`

---

## 3) Executive Diagnosis: What’s Misguided in Current Design

The core problem is architectural: we implemented PEFT in Slime as a local wrapper/merge layer around Bridge, instead of integrating with Bridge’s canonical PEFT lifecycle.

Concretely, current Slime PEFT behavior diverges from Bridge in three critical places:

1. **PEFT lifecycle ordering** (inject/freeze before checkpoint load, without Bridge resume filtering).
2. **Checkpoint semantics** (no adapter-only filtering + strictness choreography used by Bridge PEFT resume).
3. **Adapter export merge path** (custom tensor-id/shape-heuristic merge path that duplicates Bridge conversion internals and can drift for topology/MoE/canonical variants).

That mismatch is exactly why the `_io.BytesIO has no len()` failure appears in PEFT runs and not in non-PEFT control runs.

---

## 4) The `_io.BytesIO` Failure: Full Causal Chain

## 4.1 Runtime evidence

PEFT matrix runs fail with the same signature:

- `.logs/peft-matrix/dp2_tp1_pp1_uv.log:2598`
- `.logs/peft-matrix/dp1_tp2_pp1_uv.log:2594`
- `.logs/peft-matrix/dp1_tp1_pp2_uv.log:2399`

All fail with:

- `TypeError: object of type '_io.BytesIO' has no len()`
- at `_replace_sharded_keys_with_state_dict_keys`

Call stack (example):

- `.logs/peft-matrix/dp2_tp1_pp1_uv.log:2580`
- `.logs/peft-matrix/dp2_tp1_pp1_uv.log:2595`
- `.logs/peft-matrix/dp2_tp1_pp1_uv.log:2598`

PEFT logs simultaneously show adapter keys missing from checkpoint:

- `.logs/peft-matrix/dp2_tp1_pp1_uv.log:2287`
- `.logs/peft-matrix/dp1_tp2_pp1_uv.log:2282`
- `.logs/peft-matrix/dp1_tp1_pp2_uv.log:2214`

Non-PEFT control loads checkpoint successfully:

- `.logs/peft-investigation/nopeft_dp2_tp1_pp1.log:2252`

## 4.2 Code path

Slime load path:

- `slime/backends/megatron_utils/model.py:834`
- `slime/backends/megatron_utils/model.py:837`
- `slime/backends/megatron_utils/checkpoint.py:97`
- `slime/backends/megatron_utils/checkpoint.py:107`

Megatron dist-ckpt path:

- `/Users/jm/Megatron-LM/megatron/training/checkpointing.py:1130`
- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/serialization.py:142`
- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:770`
- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:796`
- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:817`
- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:380`

The key type assumption:

- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:381`
- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:388`

expects list-like values, but `mcore_to_pyt_state_dict` can produce raw `io.BytesIO` for object entries:

- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:320`
- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:333`

## 4.3 Why PEFT triggers this

From baseline onwards, Slime injects PEFT during provider construction:

- `slime/backends/megatron_utils/model_provider.py:100`
- `slime/backends/megatron_utils/model_provider.py:103`
- `slime/backends/megatron_utils/model_provider.py:355`

So checkpoint loading now happens against an adapter-augmented state template, while base checkpoint has no adapter keys.

## 4.4 Why this turns into BytesIO `len()` crash

The runtime Megatron patch in this repo modifies load planner behavior:

- `docker/patch/latest/megatron.patch:18`
- `docker/patch/latest/megatron.patch:30`
- `docker/patch/latest/megatron.patch:48`

Specifically it:

- skips missing model keys instead of raising,
- sets `allow_partial_load=True`.

PyTorch planner behavior confirms skipped keys under non-strict/partial load:

- `.logs/peft-investigation/torch_dcp_planner_source.log:47`
- `.logs/peft-investigation/torch_dcp_planner_source.log:92`
- `.logs/peft-investigation/torch_dcp_load_plan_source.log:57`
- `.logs/peft-investigation/strict_false_skip_minimal.log:40`

So some object placeholders remain raw `BytesIO` and later hit `len(...)` assumption.

Minimal reproduction artifact confirms exact type error:

- `.logs/peft-investigation/bytesio_replace_minimal.log:40`

## 4.5 Attribution

This is not random infra noise.

- **Trigger:** Slime PEFT lifecycle change (introduced in `c2330e88`).
- **Failure mode:** exposed by patched Megatron partial-load behavior (`allow_partial_load=True`).
- **Net:** PEFT-induced + loader-contract mismatch.

---

## 5) Where Current Slime PEFT Diverges from Bridge Canonical Semantics

## 5.1 Transform/freeze lifecycle divergence

Current Slime path:

- constructs PEFT config,
- applies `peft_map(..., peft_config.transform)` directly,
- manually freezes via name-substring matching.

Relevant lines:

- `slime/backends/megatron_utils/model_provider.py:335`
- `slime/backends/megatron_utils/model_provider.py:355`
- `slime/backends/megatron_utils/model_provider.py:265`
- `slime/backends/megatron_utils/model_provider.py:274`

Bridge canonical path uses `PEFT.__call__` with freeze/transform/mode/recompute hooks:

- `/Users/jm/Megatron-Bridge/src/megatron/bridge/peft/base.py:83`
- `/Users/jm/Megatron-Bridge/src/megatron/bridge/peft/base.py:98`
- `/Users/jm/Megatron-Bridge/src/megatron/bridge/peft/base.py:103`

Implication: Slime bypasses parts of the PEFT contract (notably `maybe_enable_recompute_inputs_grad` and canonical freeze semantics).

## 5.2 Missing `set_params_to_save` + adapter filter usage

Bridge applies:

- `peft.set_params_to_save(...)`
- adapter-only filter on save/load-resume.

Relevant Bridge lines:

- `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/setup.py:419`
- `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/setup.py:420`
- `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py:1517`
- `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py:1518`
- `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py:1566`
- `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py:1746`

Slime does not integrate this in its load path.

## 5.3 Checkpoint ordering divergence

Bridge PEFT pre-wrap hook explicitly loads pretrained/base checkpoint first, then transforms:

- `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/setup.py:375`
- `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/setup.py:387`
- `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/setup.py:401`

Slime currently transforms before load.

---

## 6) Rough Edges Inventory (Prioritized)

## P0 (must fix first)

### P0.1 PEFT load/ckpt contract mismatch (root cause of BytesIO crash)

Evidence:

- `slime/backends/megatron_utils/model_provider.py:103`
- `slime/backends/megatron_utils/model.py:837`
- `docker/patch/latest/megatron.patch:48`
- `.logs/peft-matrix/dp2_tp1_pp1_uv.log:2598`

Impact:

- deterministic init-time failure on tested TP/PP/DP matrix.

### P0.2 Slime bypasses Bridge PEFT resume/load semantics

Evidence:

- Slime: direct `_load_checkpoint_megatron` pass-through in `slime/backends/megatron_utils/checkpoint.py:107`
- Bridge: PEFT resume filtering and `load_strict=False` logic in `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py:1517` and `/Users/jm/Megatron-Bridge/src/megatron/bridge/training/checkpointing.py:1566`

Impact:

- fragile/non-canonical behavior for base->PEFT init and PEFT resume.

## P1 (likely to break MoE/topology generalization)

### P1.1 Custom adapter merge path duplicates Bridge internals

Evidence:

- custom map/delta/merge path in `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py:27`, `:160`, `:343`
- Bridge already has adapter task materialization + merge in `/Users/jm/Megatron-Bridge/src/megatron/bridge/models/conversion/model_bridge.py:947`, `:971`, `:1002`

Impact:

- risk of semantic drift for fused QKV/FC1, canonical adapters, MoE shards, and future Bridge changes.

### P1.2 Canonical LoRA compatibility gap in Slime merge discovery

Evidence:

- Slime discovery expects direct `ParallelLinearAdapter` child (`adapter_child` heuristic) in `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py:85`
- Canonical LoRA uses `ModuleDict` adapter composition in `/Users/jm/Megatron-Bridge/src/megatron/bridge/peft/canonical_lora.py:33`

Impact:

- canonical adapter layers may not be discovered/merged correctly in Slime custom merge path.

### P1.3 DoRA + MoE expert handling risk

Evidence:

- Bridge LoRA marks experts via `is_expert_linear(...)` in `/Users/jm/Megatron-Bridge/src/megatron/bridge/peft/lora.py:137`
- Bridge DoRA transform does **not** pass `is_expert` when building adapter in `/Users/jm/Megatron-Bridge/src/megatron/bridge/peft/dora.py:93` and `:97`
- Slime expert gather choice depends on adapter `is_expert` in `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py:268` and `:311`

Impact:

- DoRA on expert layers can route TP gathers through wrong group semantics.

Note: this is primarily a dependency-level issue, but it directly affects Slime’s MoE DoRA behavior.

### P1.4 Bridge itself flags uncertain expert adapter replica logic

Evidence:

- explicit TODO in `/Users/jm/Megatron-Bridge/src/megatron/bridge/peft/utils.py:671`

Impact:

- MoE checkpoint/replica metadata semantics still have known uncertainty upstream.

## P2 (correctness/operability rough edges)

### P2.1 Manual freeze by substring is brittle

Evidence:

- `slime/backends/megatron_utils/model_provider.py:274`
- `slime/backends/megatron_utils/model_provider.py:286`

Impact:

- relies on naming conventions; easy to silently mis-freeze as wrapper internals evolve.

### P2.2 `canonical_lora` exposed, but CLI defaults are LoRA-style targets

Evidence:

- Slime default target modules: `linear_qkv linear_proj linear_fc1 linear_fc2` in `slime/utils/arguments.py:254`
- Canonical LoRA forbids `linear_qkv`/`linear_fc1` targets in `/Users/jm/Megatron-Bridge/src/megatron/bridge/peft/canonical_lora.py:236`

Impact:

- selecting `--peft-type canonical_lora` with defaults is invalid/confusing.

### P2.3 Strict CI checks are overfitted to current small matrix

Evidence:

- adapter count hard floor and strict grad semantics in `slime/backends/megatron_utils/ci_utils.py:152`, `:353`, `:364`

Impact:

- false positives on different model sizes/topologies/warmup behavior.

### P2.4 Coverage gap for EP/ETP/MoE

Evidence:

- current matrix test only uses `ep=1, etp=1` in `tests/test_qwen3_0.6B_peft_parallel_matrix.py:12`

Impact:

- no direct proof that expert parallel combinations behave correctly.

### P2.5 Transport hash CI only validates colocated path

Evidence:

- warning in `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py:117`

Impact:

- distributed rollout transport path remains unchecked for payload-integrity invariants.

---

## 7) Utility/Call Trace of Slime-Introduced Paths

## 7.1 Model build + PEFT inject

- `slime/backends/megatron_utils/model.py:109`
- `slime/backends/megatron_utils/model_provider.py:83`
- `slime/backends/megatron_utils/model_provider.py:103`
- `slime/backends/megatron_utils/model_provider.py:355`

## 7.2 Checkpoint load

- `slime/backends/megatron_utils/model.py:837`
- `slime/backends/megatron_utils/checkpoint.py:107`
- `/Users/jm/Megatron-LM/megatron/training/checkpointing.py:1130`
- `/Users/jm/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py:817`

## 7.3 Weight sync / conversion

- `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py:138`
- `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py:114`
- `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py:160`
- `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py:343`

---

## 8) Recommended Remediation Plan (ordered)

## Phase A (unblocker)

1. **Align base-load + PEFT-inject lifecycle with Bridge.**
   - Either load base checkpoint before PEFT transform (Bridge pre-wrap semantics), or
   - apply adapter-filtered resume/load semantics before dist-ckpt load.
2. **Stop relying on partial-load + skipped adapter keys as normal control flow.**
   - If patch behavior remains, ensure object-key postprocessing cannot pass raw `BytesIO` into `_replace_sharded_keys_with_state_dict_keys`.

## Phase B (stabilize semantics)

1. Replace direct `peft_map(...transform)` path with canonical PEFT entrypoint (`PEFT.__call__`) and `set_params_to_save` integration.
2. Integrate Bridge’s PEFT checkpoint filtering (`apply_peft_adapter_filter_to_state_dict`) for save/resume.

## Phase C (remove semantic drift)

1. Refactor Slime custom adapter merge path to reuse Bridge adapter conversion tasks/materialization where possible.
2. Keep Slime-specific augmentation only where Bridge lacks functionality (primarily DoRA export merge), in a narrow extension seam.

## Phase D (MoE + topology hardening)

1. Add matrix beyond current coverage:
   - `(dp, tp, pp, ep, etp)` cases with `ep>1` and `etp>1`.
2. Add GLM4.7 MoE-specific PEFT matrix for both LoRA and DoRA.
3. Gate DoRA+MoE behind explicit validation until expert-group semantics are confirmed.
4. Add canonical-lora argument validation/default remapping in Slime parser.

## Phase E (CI robustness)

1. Make strict CI checks topology/model-size aware (remove hardcoded adapter-count assumptions).
2. Keep exact merge checks, but make gradient expectations configurable by optimizer step schedule and cold-start conditions.
3. Extend payload hash validation to distributed rollout path (or add equivalent invariants).

---

## 9) Answer to the central question

"What is misguided about the current LoRA/DoRA implementation?"

The implementation is trying to do PEFT as a thin local patch over Megatron/Bridge internals (manual transform, manual freeze, manual merge, manual strict CI) while the real correctness boundaries live in Bridge’s integrated PEFT lifecycle + checkpoint semantics.

That mismatch causes:

- immediate hard failures (BytesIO bug),
- latent topology/MoE fragility,
- high maintenance burden because Slime duplicates behavior Bridge already owns.

The fastest path to a robust dp/ep/pp/tp + MoE implementation is to re-center on Bridge semantics and keep Slime custom logic only where Bridge does not yet support needed behavior.
