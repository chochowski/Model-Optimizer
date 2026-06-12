# MCore Minitron — `modelopt.torch.prune.plugins.mcore_minitron` Skill

How the `mcore_minitron` pruning plugin works in the Model Optimizer repo
(`modelopt/torch/prune/plugins/mcore_minitron.py`).

> Minitron is an **activation-magnitude** importance-based pruning algorithm for
> NVIDIA Megatron-Core / NeMo GPT and Mamba models (including hybrid models).
> Paper: [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/pdf/2407.14679).
> The plugin implements both the **conversion** of an MCore model into a dynamic
> search space and the **search** that picks the pruned sub-network.

No prior agent transcripts discuss this module; this document was produced
directly from the source at
`modelopt/torch/prune/plugins/mcore_minitron.py`.

---

## 1. Mental model

Minitron pruning in modelopt is a **two-phase mode** plugged into the generic
`modelopt.torch.prune.prune(model, mode, constraints, dummy_input, config)`
entrypoint (`modelopt/torch/prune/pruning.py`):

1. **Convert** the MCore model in-place into a `DynamicModule` search space
   (every pruneable hyperparameter becomes a configurable `Hparam` with choices).
2. **Search** runs a forward loop to collect activations, turns those
   activations into per-hparam **importance scores**, sorts weights by
   importance, and then sets `active` choices on the hparams — physically
   trimming the weights (width pruning) and/or dropping decoder layers (depth
   pruning).

The plugin does **not** implement the dynamic modules themselves — those live in
`modelopt.torch.nas.plugins.megatron` (imported names prefixed with `_Dynamic`).
`mcore_minitron` is the *handler* that orchestrates them.

### Supported hparams (`SUPPORTED_HPARAMS`, lines 83–99)

Width pruning:
- `hidden_size`
- `ffn_hidden_size` (MLP)
- `num_attention_heads` (Attention)
- `mamba_num_heads`, `mamba_head_dim` (Mamba)
- `moe_ffn_hidden_size`, `moe_shared_expert_intermediate_size`, `num_moe_experts` (MoE)

Depth pruning:
- `num_layers`

### Supported constraints

Exactly one of:
- **`export_config`** — dict of explicit target values for any subset of
  `SUPPORTED_HPARAMS`. Deterministic: no grid search, just set and trim.
- **`params`** — upper bound on parameter count (int/float). Triggers a
  grid-search over pruned architectures plus `score_func` validation of the
  top-k candidates. Requires a `score_func` (e.g. MMLU).

`num_query_groups` was removed in 0.41; it's accepted for backward-compat but
ignored (lines 237–243).

---

## 2. Public API surface

`__all__` (lines 101–109):

| Name | Purpose |
|---|---|
| `SUPPORTED_HPARAMS` | Set of hparam names the searcher understands. |
| `MCoreMinitronConfig` | Pydantic config class built via `create_model` with per-model-type divisors. |
| `MCoreMinitronModeDescriptor` | Registers the `"mcore_minitron"` mode in `NASModeRegistry` and `PruneModeRegistry`. |
| `MCoreMinitronSearcher` | The `BaseSearcher` subclass that runs importance collection + search + prune. |
| `drop_mcore_language_model_layers` | Utility to remove 1-indexed decoder layers from an MCore model under TP/PP. |
| `get_mcore_minitron_config` | Factory that returns a config with caller-specified divisors. |
| `get_mcore_param_count` | Count reduced params across TP+PP for `GPTModel` / `MambaModel` (DynamicModule-aware). |

### Mode registration and transitions

`MCoreMinitronModeDescriptor` (lines 736–777):

- `name = "mcore_minitron"`
- `config_class = MCoreMinitronConfig`
- `search_algorithm = MCoreMinitronSearcher`
- `convert = convert_mcore_minitron`
- `restore = restore_mcore_minitron`
- `next_modes = {"export_nas", "kd_loss", "quantize", "sparse_magnitude", "sparse_gpt"}`
- `export_mode = "export_nas"`

Registered in **two** registries via the `@NASModeRegistry.register_mode` and
`@PruneModeRegistry.register_mode` decorators, so the mode is usable from both
`mtn.convert(...)` and `mtp.prune(...)`.

---

## 3. Phase 1 — Conversion (`convert_mcore_minitron`)

Entry: `convert_mcore_minitron(model, config) -> (model, metadata)` (lines 713–726).

Flow:

1. `_convert_model_to_dynamic_space(model, config)` (lines 697–710):
   - Creates a `DynamicSpace(model)`.
   - Restricts `_should_be_converted` to top-level `GPTModel` / `MambaModel`
     (keys of `SUPPORTED_MODELS` from `modelopt.torch.nas.plugins.megatron`).
   - Calls `dynamic_space.convert_to_dynamic(config.model_dump(), DMRegistry)` —
     this is where `_DynamicMCoreLanguageModel`, `_DynamicTransformerLayer`,
     `_DynamicSelfAttention`, `_DynamicMLP`, `_DynamicSequentialMLP`,
     `_DynamicMambaLayer`, `_DynamicMambaMixer`, `_DynamicMoELayer` get wrapped
     around the MCore modules.
   - Asserts the space ended up configurable, otherwise raises `ApplyModeError`.
2. Stores `get_subnet_config(model)` in `metadata["subnet_config"]` so the
   current hparam choices are persisted.

`restore_mcore_minitron` is intentionally a no-op (lines 729–733) — re-running
`convert` on restore would force TP=1 and break parallelism.

### `MCoreMinitronConfig`

Built dynamically with `pydantic.create_model` via
`get_kwargs_for_create_model_with_rules` (lines 637–664). Default divisors:

- GPT:
  - `hidden_size_divisor=256`
  - `ffn_hidden_size_divisor=512`
  - `num_moe_experts_divisor=8`
  - `num_layers_divisor=2`
- Mamba (only when `HAS_MAMBA`): additionally
  - `mamba_head_dim_divisor=8`

Divisors control the discrete choices each hparam exposes. Override via
`get_mcore_minitron_config(...)` (lines 667–694) which walks the config dict
and replaces any key named `{hparam}_divisor`.

---

## 4. Phase 2 — Search (`MCoreMinitronSearcher`)

`MCoreMinitronSearcher(BaseSearcher)` (lines 173–573).

### Additional search config (`default_search_config`, lines 191–204)

- `max_iter_data_loader: 1024`
- `skip_sorting: False` — if True, reuse existing ordering, do not re-sort.
- `scores_path: None` — alias for `checkpoint`; lets you persist/reuse
  activations + layer scores across re-prunes with different export configs.
- Grid-search-only:
  - `max_width_pruning: 0.40` (only top `1 - 0.40` width choices considered)
  - `max_depth_pruning: 0.20`
  - `hparams_to_skip: None`
  - `top_k: 10`

### State dict (`default_state_dict`, lines 206–214)

Persisted across runs via `save_search_checkpoint`:

- `activations_per_rank: list[dict[str, Tensor]]` — one dict per PP rank.
- `layer_scores: dict[int, Tensor]` — 1-indexed.
- `sorted_layers: list[int] | None` — layers sorted most-important first.
- `top_k_candidates_per_constraint: dict[float, list[CandidateSubnet]]`.

`sanitize_search_config` (lines 216–222) maps `scores_path` → `checkpoint` and
forces `verbose=True` on all ranks.

### `before_search` (lines 224–271)

- Enforces exactly one constraint from `{"export_config", "params"}`.
- For `export_config`:
  - Strips legacy `num_query_groups`.
  - Validates keys are within `SUPPORTED_HPARAMS`.
  - Sets `self.hps_to_sort = set(export_config.keys())` — **only sort what you
    prune** (e.g. depth-only prune skips width sorting).
- For `params`:
  - Requires a `score_func` (`self.has_score`).
  - Sets `self.hps_to_sort = SUPPORTED_HPARAMS` (sort everything).
- For every `named_hparams`:
  - Asserts `hp_name in SUPPORTED_HPARAMS` (safety-net against plumbing errors).
  - If pinned by `export_config`, asserts chosen value is in `hp.choices`.
  - Calls `hp.reset_choices()` (needed so `ConcatHparam`s refresh after `modify`).
- Asserts `self.model` is an unwrapped `_DynamicMCoreLanguageModel`.

### `run_search` (lines 273–331)

Main entry when `mtn.search` is invoked. Pseudo-code:

```python
registry = ImportanceEstimatorRegistry(self.model)  # registers hooks + importance fns

if checkpoint has layer_scores and activations:
    registry.set_activations_and_layer_scores(...)
elif not skip_sorting:
    self.model.eval()
    with torch.no_grad():
        self.forward_loop(self.model)      # <-- user-provided; runs ~max_iter_data_loader batches
    self.activations_per_rank, self.layer_scores = \
        registry.get_activations_and_layer_scores()
    self.save_search_checkpoint(verbose=True)

if not skip_sorting:
    sort_parameters(self.model, self.hps_to_sort, verbose=True)  # permute weights
registry.cleanup()   # remove hooks and temp attributes

# Depth ordering (only meaningful if layer_scores collected)
self.sorted_layers = (
    [layer for layer, _ in sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)]
    if layer_scores else None
)

export_config = (
    self.search_best_arch_by_params()  if "params" in constraints
    else constraints["export_config"]
)

self._prune(export_config, prune_depth=True)

# Mamba-only: rewrite hybrid_override_pattern to match kept layers
```

Key invariants:

- `sorted_layers` is guaranteed to be a permutation of `range(1, num_layers+1)`.
- Pruning `num_layers` without `layer_scores` is explicitly disallowed.

### `_prune` (lines 333–381)

Homogeneous pruning — sets `hp.active = export_config[name]` on every matching
configurable hparam, then:

1. If `prune_depth` and `num_layers` was reduced: call
   `drop_mcore_language_model_layers(self.model, layers_to_drop=sorted_layers[new_num_layers:])`.
2. Back-fill config fields that may be `None` on the source model so the pruned
   config is self-contained:
   - `kv_channels = hidden_size // num_attention_heads`
   - `num_query_groups = num_attention_heads`
   - `moe_ffn_hidden_size = ffn_hidden_size` (when MoE is active)
3. Write `export_config` into `model.config` via `setattr`.
4. Reinitialize the **MoE token dispatcher** on the first `_DynamicMoELayer`
   (`m._export_reinit_token_dispatcher()`).

### `search_best_arch_by_params` (lines 383–505)

Run only when the constraint is `params`. Stages:

1. **Collect per-hparam choices** from every PP rank and merge them:
   ```python
   hp_choices = {name: hp.choices for ...}
   dist.DistributedProcessGroup.get_dist_syncd_obj(hp_choices, pp_group, op=merge_dicts)
   ```
2. **Generate search-space combos** via
   `MCoreMinitronSearcher._generate_search_space_combos(...)`:
   - Drops `hparams_to_skip`.
   - Keeps only the **top** `1 - max_width_pruning` (width) or
     `1 - max_depth_pruning` (depth) choices; skips single-choice hparams.
   - `product(*choices)` yields a list of candidate `ss_config` dicts.
3. **Filter by params constraint**:
   - For each `ss_config`, call `self._prune(ss_config, prune_depth=False)`,
     compute `_param_num_dynamic(self.model, layer_numbers_to_count=...)` (using
     `sorted_layers[:num_layers]` when `num_layers` shrinks), keep those
     ≤ `max_params`, then `sample(self.model, sample_func=max)` to restore the
     max subnet before the next iteration.
   - Cache top-k by param count: `self.top_k_candidates_per_constraint[max_params]`.
4. **(Optional) KD step** — *not* performed automatically. A verbose log
   (lines 457–466) instructs users to export each top-k candidate and KD-tune
   on ~2B tokens before scoring, per the paper.
5. **Validate top-k** with `self.eval_score(...)` (the `score_func`):
   - For each candidate, materialize with `self._prune(ss_config, prune_depth=True)`,
     score, reassemble decoder layers (revert layer drop), restore max subnet.
   - Save checkpoint after each score.
6. **Return the best `ss_config`** (max by score). `dist.barrier()` before
   selection for deterministic rank ordering.

### `_generate_search_space_combos` (lines 507–573)

Static helper. Example:

```python
search_space = {"hidden_size": [1024, 2048, 3072, 4096], "num_layers": [1, 2, ..., 32]}
# with max_width_pruning=0.40, max_depth_pruning=0.20:
# hidden_size -> keep top 60% => [2048, 3072, 4096]
# num_layers  -> keep top 80% => last 80% of sorted list
# result = cartesian product of filtered lists
```

Hparams with a single choice are dropped (nothing to search).

---

## 5. Dropping decoder layers under TP/PP — `drop_mcore_language_model_layers`

Lines 112–163. Key behaviors:

- Accepts 1-indexed `layers_to_drop`. Asserts minimum ≥ 1.
- If `model` is a wrapper (e.g. `Float16Module`), walks `named_modules` and
  picks the first `GPTModel` / `MambaModel`.
- Each PP rank computes how many of its local layers survive, then
  `all_gather_into_tensor(..., group=pp_group)` to learn the global remaining
  count and an offset used to **reindex `layer.layer_number`** starting from
  `sum(remaining_on_earlier_ranks) + 1`.
- Rebuilds `model.decoder.layers` as `nn.ModuleList(kept_layers)` and updates
  `model.config.num_layers = new_num_layers`.
- Side effect: iterating `named_modules` sets `model` to the inner GPT/Mamba
  model, so later accesses are on the unwrapped module.

---

## 6. Parameter counting — `get_mcore_param_count` / `_param_num[_dynamic]`

Lines 576–634.

- `get_mcore_param_count(model)` dispatches on `isinstance(model, DynamicModule)`.
- `_param_num` (static) — sums `p.numel()` over `named_parameters`, skipping
  `output_layer.weight` if `share_embeddings_and_output_weights`.
  All-reduces across PP and TP groups.
- `_param_num_dynamic(model, *, layer_numbers_to_count=None)` — needed for
  dynamic modules because `model.parameters()` ignores `active_slice`, so it
  manually walks `named_parameters()` and uses `getattr(submodule, param_name)`
  on the live tensor view. When `layer_numbers_to_count` is given, only those
  layer numbers contribute (used for pre-drop param estimation during grid
  search).

---

## 7. Importance estimation — `ImportanceEstimatorRegistry`

Lines 780–925. Constructed once per `run_search`; walks `model.modules()` and
dispatches to per-module registration functions:

| Module type | Registration function | Hparam(s) it scores |
|---|---|---|
| `_DynamicMCoreLanguageModel` | `_register_hidden_size_importance` | `hidden_size` |
| `_DynamicTransformerLayer`, `_DynamicMambaLayer` | `_register_depth_cosine_importance` | (`num_layers` via `layer._scores`) |
| `_DynamicSelfAttention` | `_register_self_attention_importance` | `num_attention_heads` (ranking) |
| `_DynamicMLP` | `_register_mlp_importance` | `module.hparam_name` (e.g. `ffn_hidden_size`, `moe_ffn_hidden_size`, `moe_shared_expert_intermediate_size`) |
| `_DynamicSequentialMLP` | `_register_sequential_mlp_importance` | `num_local_experts` (→ `num_moe_experts`) |
| `_DynamicMambaMixer` | `_register_mamba_mixer_importance` | `mamba_num_heads`, `mamba_head_dim` (rankings) |

Registry API:

- `register_hook(module, fn, hook_type="forward"|"forward_pre", **kw)` — stores
  handle in `self._hooks` for later cleanup.
- `register_importance(dynamic_module, hparam_name, importance_fn, importance_is_order=False)` —
  thin wrapper around `hp.register_importance(fn)`. When
  `importance_is_order=True` the hparam treats the returned tensor as a
  **ranking** (argsort result), not raw magnitudes; trimming is then done
  group-aware downstream (e.g. inside `NumAttentionHeadsHp`, `MambaNumHeadsHp`).
- `cleanup()` — removes all hooks (but leaves `_activations` / `_scores` temp
  attributes in place so they can be persisted).
- `get_layer_scores()` — asserts every layer has `_scores > 0`, gathers
  `{layer_number: score}` across PP, asserts coverage `[1..max_num_layers]`.
- `get_activations_and_layer_scores()` — `dist.allgather` local `_activations`
  dicts across PP, returns `(activations_per_rank, layer_scores)`.
- `set_activations_and_layer_scores(...)` — restore from checkpoint without
  running the forward loop; requires matching PP size.

### Hook-level details (all collect activations in **float32** for numerical safety)

- **Hidden size** (lines 929–990): forward-hooks `input_layernorm`,
  `pre_mlp_layernorm`, or Mamba `norm`. Computes
  `abs().mean(seq)` → `pow(2).sum(batch)` → per-hook `_activations[id]`;
  final importance = `stack(pow(0.5)).sum(dim=0)` and all-reduced over PP.
- **Depth** (lines 993–1022): forward-hook on the whole Transformer / Mamba
  layer, `with_kwargs=True`. Uses `1 - cosine_similarity(input, output).mean()`
  as layer importance — **higher means "this layer changes the hidden state
  more, so keep it"**. `reduce_from_tensor_model_parallel_region` across TP
  before accumulating.
- **Self-attention heads** (lines 1025–1086): hook on `linear_proj` input
  (which is the concatenated query projection of shape `[seq, batch,
  query_projection_size]`). Reshapes scores to `(max_nheads, kv_channels)`,
  takes L2 per head, then produces a **group-aware ranking** so heads are sorted
  *within each query group*. `importance_is_order=True`.
- **MLP / MoE experts' MLP** (lines 1089–1123): hook on `linear_fc2` input
  (post-activation, pre-output-projection), `[seq, batch, ffn_hidden_size]`
  (or `[tokens, ffn_hidden_size]` for sparse experts — expanded via `input[:, None, :]`).
  Importance = L2 norm per ffn channel.
- **MoE `SequentialMLP` (num_experts)** (lines 1126–1180): hook on the
  `SequentialMLP` module itself. Splits the output by `tokens_per_expert`,
  accumulates `l2_norm` and `sample_counts` per expert; importance =
  `l2_scores / (sample_counts + 1e-8)`. Used to rank and drop experts.
- **Mamba mixer** (lines 1183–1268): hook on `in_proj` output (shape `[seq,
  batch, d_inner]`, activations summed-squared over batch). From a slice
  corresponding to `x` (indices `[max_d_inner, 2*max_d_inner)`), derives two
  rankings:
  - **head dims**: L2 over heads → argsort → top `target_headdim`.
  - **heads**: within each group, L2 over (selected head dims) → argsort →
    flatten with group offsets.
  Both registered with `importance_is_order=True`. Per the paper's ablations,
  `x` is the best projection to rank on (comment lines 1219–1220).

### Temp attributes the hooks attach

Used for serialization / cleanup:

- `_DynamicMCoreLanguageModel._activations: dict[int id(layernorm_submodule) -> Tensor]`
- `_DynamicTransformerLayer._scores: float`, `_DynamicMambaLayer._scores: float`
- `_DynamicSelfAttention._activations: Tensor | None`
- `_DynamicMLP._activations: Tensor | None`
- `_DynamicSequentialMLP._activations: {"expert_l2_scores": Tensor, "expert_sample_counts": Tensor}`
- `_DynamicMambaMixer._activations: Tensor | None`

They're registered with `module._register_temp_attribute(...)` so the
DynamicModule layer handles them correctly during `sample`/`export`.

---

## 8. End-to-end call graph

```
modelopt.torch.prune.prune(model, "mcore_minitron", constraints, dummy_input, config)
  └── mtn.convert(model, "mcore_minitron", registry=PruneModeRegistry)
        └── MCoreMinitronModeDescriptor.convert -> convert_mcore_minitron
              └── _convert_model_to_dynamic_space -> DynamicSpace.convert_to_dynamic(DMRegistry)
                    └── _DynamicMCoreLanguageModel / _DynamicTransformerLayer / ... wrapping

  └── mtn.search(model, constraints, dummy_input, config)
        └── MCoreMinitronSearcher
              ├── sanitize_search_config
              ├── before_search    -> validate constraints, reset choices
              └── run_search
                   ├── ImportanceEstimatorRegistry(model)     # registers hooks
                   ├── forward_loop(model)                   # user-supplied, no_grad + eval
                   ├── get_activations_and_layer_scores()    # all-gather over PP
                   ├── save_search_checkpoint()
                   ├── sort_parameters(model, hps_to_sort)   # permutes weights by importance
                   ├── cleanup hooks
                   ├── sort_layers_by_score -> sorted_layers
                   ├── export_config := constraints["export_config"]
                   │                  or search_best_arch_by_params()
                   │        ├── _generate_search_space_combos
                   │        ├── grid-search filter by _param_num_dynamic
                   │        └── for each top-k: _prune(prune_depth=True) + eval_score
                   └── _prune(export_config, prune_depth=True)
                         ├── hp.active = export_config[name]
                         ├── drop_mcore_language_model_layers
                         ├── update model.config defaults (kv_channels, num_query_groups, moe_ffn_hidden_size)
                         └── _export_reinit_token_dispatcher (MoE)
```

After search, the model is at its pruned architecture in-place. Users typically
chain `next_modes` (`export_nas`, `kd_loss`, `quantize`, `sparse_*`) to export
the subnet and continue optimization / training.

---

## 9. Typical usage

### Deterministic prune via `export_config`

```python
import modelopt.torch.prune as mtp

pruned_model, state = mtp.prune(
    model,
    mode="mcore_minitron",
    constraints={
        "export_config": {
            "ffn_hidden_size": 8192,
            "num_attention_heads": 16,
            "hidden_size": 3072,
            "num_layers": 24,
        },
    },
    dummy_input=dummy,
    config={
        "forward_loop": forward_loop,   # needed when width hparams are pruned
        "scores_path": "/path/to/cache.pt",
    },
)
```

Only the hparams listed in `export_config` are sorted (see `before_search`).
`num_layers` requires running the forward loop so layer cosine scores exist.

### `params`-constrained grid search

```python
pruned_model, state = mtp.prune(
    model,
    mode="mcore_minitron",
    constraints={"params": 4e9},    # 4B params upper bound
    dummy_input=dummy,
    config={
        "forward_loop": forward_loop,
        "score_func": lambda m: run_mmlu(m),
        "max_width_pruning": 0.4,
        "max_depth_pruning": 0.2,
        "top_k": 10,
        "hparams_to_skip": ["num_moe_experts"],
        "scores_path": "/path/to/cache.pt",
    },
)
```

### Reusing cached scores

Set `config["scores_path"]` (aliased to `checkpoint`); if the file exists,
`run_search` skips the forward loop and calls
`set_activations_and_layer_scores(...)`. Useful for iterating on different
`export_config`s without paying the forward-loop cost each time.

---

## 10. Constraints, limitations, and gotchas

- **Exactly one constraint**: either `export_config` or `params`. Both together
  (or any other key) will fail in `before_search`.
- **Unwrapped MCore model required**: `self.model` must be a
  `_DynamicMCoreLanguageModel` after conversion — wrappers like
  `Float16Module` should be unwrapped before calling `prune`.
  `drop_mcore_language_model_layers` can still unwrap at depth-drop time, but
  the searcher's assertion at line 269 does not.
- **TP=1 for activation gathering**: comments at lines 942, 1038, 1100, 1196
  note that `gather_from_tensor_model_parallel_region` is currently a no-op
  since activations are only collected at TP=1.
- **`num_query_groups` is frozen** (since 0.41). It's silently dropped from
  `export_config`, but values other than the model's own trigger a
  `ValueError`.
- **Depth pruning requires `layer_scores`**: if you prune only width via
  `export_config`, no forward loop is needed *unless* any width hparam is
  listed. Attempting to prune `num_layers` without scores raises.
- **KD after search is manual**: the searcher prints a message instructing
  users to KD the top-k candidates before relying on `score_func` results (see
  lines 457–466).
- **`restore` is a no-op**: re-running `convert_mcore_minitron` during restore
  would force TP=1. Persisted metadata (`subnet_config`) is what describes the
  pruned subnet.
- **Score aggregation is summation, not mean**: comments at lines 951–953 and
  1015 note that scores are accumulated as sums across the forward loop for
  simplicity. This is fine for ranking but not a calibrated magnitude.

---

## 11. Quick file index

| Topic | Path | Lines |
|---|---|---|
| High-level `prune()` API | `modelopt/torch/prune/pruning.py` | 31–210 |
| Plugin source | `modelopt/torch/prune/plugins/mcore_minitron.py` | full file |
| `SUPPORTED_HPARAMS` | `…/mcore_minitron.py` | 83–99 |
| `drop_mcore_language_model_layers` | `…/mcore_minitron.py` | 112–163 |
| `MCoreMinitronSearcher` | `…/mcore_minitron.py` | 173–573 |
| `before_search` | `…/mcore_minitron.py` | 224–271 |
| `run_search` | `…/mcore_minitron.py` | 273–331 |
| `_prune` | `…/mcore_minitron.py` | 333–381 |
| `search_best_arch_by_params` | `…/mcore_minitron.py` | 383–505 |
| `_generate_search_space_combos` | `…/mcore_minitron.py` | 507–573 |
| Parameter counting | `…/mcore_minitron.py` | 576–634 |
| `MCoreMinitronConfig` | `…/mcore_minitron.py` | 637–694 |
| `convert_mcore_minitron` / `restore_mcore_minitron` | `…/mcore_minitron.py` | 697–733 |
| `MCoreMinitronModeDescriptor` | `…/mcore_minitron.py` | 736–777 |
| `ImportanceEstimatorRegistry` | `…/mcore_minitron.py` | 780–925 |
| Hidden-size hook | `…/mcore_minitron.py` | 929–990 |
| Depth (cosine) hook | `…/mcore_minitron.py` | 993–1022 |
| Self-attention head ranking | `…/mcore_minitron.py` | 1025–1086 |
| MLP (ffn_hidden_size) importance | `…/mcore_minitron.py` | 1089–1123 |
| MoE expert importance | `…/mcore_minitron.py` | 1126–1180 |
| Mamba mixer rankings | `…/mcore_minitron.py` | 1183–1268 |
| Dynamic module impls | `modelopt/torch/nas/plugins/megatron.py` | referenced |

---

## 12. Guidance for future agents

- **Treat `mcore_minitron` as a Minitron-style *handler***. The actual per-
  module dynamic wrappers (`_Dynamic*`) live in
  `modelopt.torch.nas.plugins.megatron` — look there first when behavior
  depends on how a specific module exposes its hparams, slicing, or export.
- **The ordering of the search matters**: activations → sort weights → decide
  `sorted_layers` → choose `export_config` → prune (width then depth) →
  patch `model.config` defaults → re-init MoE token dispatcher. Don't skip
  steps when extending the algorithm.
- **Two "importance" kinds**: magnitude-style tensors (`_activations`-derived)
  for width, and **ranking-style** tensors (`importance_is_order=True`) for
  heads and Mamba params. New hparams that need group-aware trimming should
  use the ranking style and implement group-aware `Hp` classes (see
  `NumAttentionHeadsHp`, `MambaNumHeadsHp`).
- **Use `scores_path`** when iterating on different `export_config`s during
  development — the forward loop is the expensive step.
- **Depth scoring is `1 - cos`** and *aggregated as a sum across batches*.
  When interpreting `layer_scores`, higher = more important = keep.
- **Don't add new top-level constraints** without extending `before_search`.
  It's hard-coded to the two-constraint universe.
