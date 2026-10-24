# Semantic update pass for the KernelBlaster knowledgebase

Before this change, the knowledgebase update (`GPUOptimizationDatabase.update_optimization_result`)
only did arithmetic. For each (state, technique) pair it incremented `usage_count` and kept running
averages of speedup and improvement. Descriptions, states and characteristics were static text, so
the store kept no record of why a technique worked or failed, or when it applies.

This change adds a semantic layer next to the numbers:

```
attempt ──► OutcomeRecord ──► OutcomeLog (JSONL, append-only)          knowledge/outcomes.py
               │                    │
               │ observe()          │ every N new records: one cheap-model call
               ▼                    ▼
        "already tried"      SemanticUpdater.reflect()                  knowledge/reflect.py
          tallies               │  JSON edits → validate_edit() → apply_edit()
               │                ▼
               └────────►  ClaimStore (claims.json)                     knowledge/claims.py
                                 │ relevant(target, arch, op_class, state, technique, problem, slot)
                                 ▼
             plan / codegen / fix prompts (database.py builders, opt_ncu_rl_optimized)
```

The numbers are also finer now. Stats are kept per (state, technique, op_class, arch), with a
fallback to the coarse key when the fine key has too few samples. See "Finer numerical stats" below.

Modules:

| file | role |
|---|---|
| `knowledge/outcomes.py` | one record per attempt (base commit, unchanged) |
| `knowledge/claims.py` | `Claim`, `ClaimStore`: merging, scoping, threshold, decay, dedupe, rendering, "already tried" |
| `knowledge/reflect.py` | `SemanticUpdater`: prompt, edit schema, validation, cadence, consolidation |
| `knowledge/semantic.py` | `SemanticKnowledge`: optional facade the single-file agents hold (log + store + updater + run context) |
| `agents/database.py` | fine stats; optional `knowledge_notes` / `stats_context` on the prompt builders |
| `agents/opt_ncu_rl_optimized.py` | records outcomes, injects knowledge, triggers reflection, all behind `knowledge=None` |

## Public API

```python
from kernelblaster.knowledge.claims import Claim, ClaimStore
store = ClaimStore(path)
store.relevant(*, target, arch=None, op_class=None, state=None, technique=None,
               problem_id=None, slot=None, k=12, impl_id=None, kinds=None,
               include_ineligible=False) -> list[Claim]
store.render(claims) -> str
store.already_tried(problem_id, slot=None) -> str
store.observe(outcome_record)          # deterministic tally, no LLM
store.save()

from kernelblaster.knowledge.reflect import SemanticUpdater
up = SemanticUpdater(outcome_log, store, *, model, every_n=8, logger=None,
                     max_batch=24, max_edits=12, max_prompt_claims=40, consolidate_every=10,
                     known_techniques=None, known_states=None, trace_dir=None)
await up.maybe_reflect(*, force=False) -> list[dict]   # applied edits
await up.reflect(records) -> list[dict]
await up.consolidate() -> list[dict]
```

The model is only a name and goes through `kernelblaster.agents.utils.generate_code_retry`.
Tests use `stub:<name>` and dev runs use `claude-code:sonnet`.

## Claim schema

| field | meaning |
|---|---|
| `kind` | `applicability`, `pitfall`, `interaction`, `parameter`, `revise_description`, `state_split`, `impl_note` |
| `text` | the lesson, ≤ 400 chars |
| `target` | backend; always set; retrieval never crosses it |
| `arch`, `op_class`, `state`, `technique` | optional scope; `None` = general |
| `problem_id`, `slot`, `impl_id` | impl-note scope (problem-local) |
| `data` | kind-specific payload (below) |
| `support`, `contradict` | counts of **distinct** outcome records |
| `support_ids`, `contradict_ids` | the cited record ids (last 64 kept) |
| `status` | `active` / `retired` |
| `created_epoch`, `last_epoch` | reflection epochs, used for decay |
| `source`, `id`, `created_ts`, `updated_ts` | bookkeeping |

`data` by kind:

| kind | data |
|---|---|
| applicability | `{"effect": "helps" \| "no_effect" \| "hurts"}` |
| pitfall | `{"symptom": str, "fix": str?}` |
| interaction | `{"other": technique, "relation": "works_after" \| "works_before" \| "undoes" \| "conflicts" \| "enables"}` (subject = `scope.technique`) |
| parameter | `{"params": {name: scalar}}` (≤ 8 keys) |
| revise_description | none (the text *is* the description) |
| state_split | `{"proposed_state": snake_case, "criterion": str}` (a proposal only) |
| impl_note | `{"note_type": "why_fast" \| "dead_end"}` |

`claims.json` is `{"version": 1, "meta": {epoch, reflect_cursor, reflections}, "claims": [...], "tried": {...}}`.
Unknown claim fields are ignored when loading, and invalid claims are dropped.

## Edit schema (what the model returns)

```json
{"edits": [
  {"kind": "applicability|pitfall|interaction|parameter|revise_description|state_split|impl_note",
   "text": "...", "evidence": ["<record id>", ...], "counter_evidence": ["<record id>"],
   "scope": {"arch": "sm_89|*", "op_class": "gemm|*", "technique": "...|*", "state": "...|*",
             "problem_id": "...", "slot": "...", "impl_id": "..."},
   "data": {"effect": "helps"}},
  {"kind": "support|contradict", "claim_id": "c_...", "evidence": ["<record id>"]}
]}
```

`validate_edit` rejects an edit (the reason is logged and kept in `last_rejected`) when any of these hold:

- the edit is not an object, or its kind is unknown;
- `evidence` is empty, is not a list of strings, or cites an id that is not in the outcome log (the
  same applies to `counter_evidence`);
- the evidence spans more than one target, or `scope.target` disagrees with the evidence;
- `support`/`contradict` names an unknown or retired claim, or a claim of a different target;
- a scope value is not allowed:
  - `arch` must be the arch of a cited record;
  - `op_class` must be in `OP_CLASSES`;
  - `technique` must be a known technique or appear in the log;
  - `state` must be a known state or appear in the cited records;
  - `problem_id`, `slot` and `impl_id` must come from the cited records;
- the kind-specific `data` is malformed (see the table above);
- the evidence does not fit the claim:
  - `helps` needs at least one improved record;
  - `no_effect` and `hurts` cannot cite only improved records;
  - `pitfall` needs a compile, wrong-output or runtime failure;
  - `parameter` and `why_fast` need an improved record;
  - `state_split` needs at least two records with different outcomes;
  - `interaction` needs two distinct techniques;
- more than `max_edits` edits are already applied.

Scope resolution: if the model leaves a field out (or sets it to null), the field takes the common
value of the cited records, or `None` if the records differ. `"*"` widens it to general. Pitfalls,
revised descriptions and impl notes don't inherit `state`. Problem, slot and impl fields are set
only on impl notes.

A `data` object nested under the kind name (`{"applicability": {"effect": ...}}`) is unwrapped.
Sonnet produced that shape on the first live run.

## Merge, threshold and decay rules

| constant | default | rule |
|---|---|---|
| `MIN_SUPPORT` | 2 | a claim enters prompts only if `support >= min_support(kind)` **and** `support > contradict` |
| `MIN_SUPPORT_BY_KIND` | impl_note 1, pitfall 1 | a single observed dead end, or a cited failure with its fix, is enough |
| `AGE_OUT_EPOCHS` | 12 | an active claim that is *not eligible* and untouched for 12 reflections is retired |
| `HALF_LIFE_EPOCHS` | 30 | ranking = `(support − contradict) · 0.5^(age/30)`; eligible claims only lose rank, they never retire from age alone |
| `PURGE_RETIRED_EPOCHS` | 100 | retired claims are dropped from the file |
| `MAX_ACTIVE_CLAIMS` | 2000 | lowest-ranked are retired beyond this |
| `MAX_CITED_IDS` | 64 | cited ids kept per claim (the counts keep growing) |
| `SIM_THRESHOLD` / `JACCARD_THRESHOLD` | 0.85 / 0.8 | near-duplicate test on normalised text |
| `FINE_STATS_MIN_SAMPLES` (database) | 3 | fine-key stats are used only with at least 3 samples |

- **Evidence.** Support and contradiction counts are of distinct record ids. Citing the same record
  again changes nothing, and a record already counted on one side of a claim is ignored on the
  other side.
- **Contradictions.** A `contradict` edit raises the claim's contradiction count. An opposite claim
  (for example "hurts" against "helps") is a separate claim in the same scope. Both stay in the
  store with their counts and are rendered together, so a planner sees `(+3/-1) helps…` next to
  `(+2/-0) hurts…`. Nothing is overwritten.
- **Near-duplicates.** Two claims merge only if all of these hold:
  - they have the same `dedupe_key`: kind, every scope field, and polarity (applicability effect,
    interaction relation and other technique, parameter values, proposed state, note type);
  - their normalised texts are similar;
  - they contain the same set of numbers, so "tile 64" and "tile 128" never merge.

  Merging happens on insert: an `add` that matches an existing claim becomes support for it. It also
  happens in `ClaimStore.dedupe()`. A merge takes the union of the evidence, and the absorbed claim
  is retired with `data.merged_into`.
- **LLM consolidation.** `consolidate()` runs `dedupe()` first. If the largest target has at least 6
  active claims, one model call then proposes `{"keep", "absorb", "text"}` merges. A merge is applied
  only when every absorbed claim is active and has the same `dedupe_key` as the kept one. The
  updater runs it every `consolidate_every=10` reflections.

## Retrieval and rendering

`relevant()` rules:

- `target` must match exactly.
- For `arch`, `op_class`, `state` and `technique`: a field that is unset on the claim matches
  anything; a field that is set must equal the query value when the query gives one.
- `problem_id` and `slot` must always match, so impl notes never leave their problem or slot.
- Results are ordered by how many query fields matched, then decayed rank, then support. Only
  eligible claims are returned unless `include_ineligible=True`; the reflection prompt uses that
  flag so weak claims can collect support.

`render()` writes one line per claim:

```
- (+2/-0) [helps | tensor_core_utilization | gemm | sm_89 | compute_throughput_saturated] wmma tensor core usage helps fp16 GEMM when compute-saturated, ...
- (+1/-0) [pitfall | tensor_core_utilization | gemm | sm_89] wmma 16x16x16 K loop assuming K%16==0 gives wrong output ... FIX: zero-pad the K tail tile ...
```

`already_tried(problem_id, slot)` combines two sources:

- deterministic tallies from `observe()`, for example
  `- tensor_core_utilization: 2 tries (compile_error 1, wrong_output 1) -- no improvement yet; last error: "..."`;
- eligible `impl_note` claims for that problem and slot.

## Prompt injection points

Everything is opt-in. With no knowledge configured the prompts are byte-identical to before, and
tests check this.

| where | what | how |
|---|---|---|
| `generate_optimization_plan(..., knowledge_notes=, stats_context=)` | claims for (target, arch, op_class, state) plus "already tried" plus measured fine stats | legacy path: a `LEARNED KNOWLEDGE` section appended to the prompt; tiered path: added to the **user** message, so the cached system prompt is unchanged |
| `build_codegen_messages(..., knowledge_notes=)` | claims for the chosen technique (pitfalls, parameters, interactions, revised description) plus "already tried" | user message, `LEARNED NOTES FOR THIS TECHNIQUE` |
| `build_fix_messages(..., knowledge_notes=)` | pitfall claims for the technique | user message, `KNOWN PITFALLS` |
| `OptimizedRLNCUAgent(knowledge=...)` | builds the notes via `SemanticKnowledge.plan_notes` / `codegen_notes` and pitfalls | `knowledge=None` and `KB_KNOWLEDGE_DIR` unset means the old behaviour |

The agent records an `OutcomeRecord` for every step, including codegen, compile, verify and runtime
failures (`classify_error`). It passes `op_class` and `arch` to `update_optimization_result`, calls
`maybe_reflect()` at the end of each rollout, and calls `maybe_reflect(force=True)` once at the end
of the run.

Env-only enablement for `scripts/run_opt_ncu_rl_optimized.py`:

- `KB_KNOWLEDGE_DIR`: directory for `outcomes.jsonl`, `claims.json` and `reflect_traces/`;
- `KB_REFLECT_MODEL`: the reflection model, for example `claude-code:sonnet`; leave it unset to use
  existing claims without reflecting;
- `KB_REFLECT_EVERY`: the cadence, default 8;
- `KB_OP_CLASS`: default `other`.

`analyze_performance_state` and `match_state_against_database` were deliberately left untouched.
State classification should depend on the profile, not on past lessons, and state-split proposals
are for humans to review.

## Finer numerical stats

`update_optimization_result(state, technique, actual_improvement, ..., op_class=None, arch=None, step_speedup=None)`
still updates every legacy field on the `OptimizationEntry`. It also records one per-step speedup
sample (`step_speedup`, or inferred as `1/(1 − improvement/100)`) into `fine_stats`:

- `"state|technique|*|*"`: the coarse per-step aggregate, always written;
- `"state|technique|op_class|arch"`: written when either value is given.

Each entry holds `n`, `sum_log_speedup`, `geomean_speedup`, `mean_improvement`, `n_improved`,
`best_speedup`, `worst_speedup` and `last_updated`.

`get_speedup_estimate(state, technique, op_class, arch)` returns `(speedup, n, source)` and falls
back in this order:

1. `fine`, if the fine key has at least 3 samples;
2. `coarse`, the per-step (state, technique) aggregate;
3. `legacy`, the entry's `predicted_speedup`;
4. `prior`, 1.0.

When `stats_context` is given, the plan prompt lists fine keys with enough samples:
`<technique> in <state>: 1.82x geomean per step over 5 tries (4 improved)`.

The JSON stays back-compatible. `fine_stats` is a new top-level key, and old snapshots without it
load with `{}`. Old readers already use `data.get(...)`, so they ignore it. `OptimizationEntry` is
unchanged.

## Cost bounds

- One model call per reflection, and reflections run only after `every_n=8` new records.
  `maybe_reflect` returns at once if a pass is already running, so parallel rollouts never trigger
  concurrent calls.
- Per call the prompt holds at most `max_batch=24` records (older unreflected ones are skipped and
  logged; their numbers and tallies are still kept), at most `max_prompt_claims=40` claims, and
  errors cut to 300 chars. At most `max_edits=12` edits are applied.
- Consolidation adds one call every 10 reflections, with at most 60 claims.
- Measured on the live smoke test with `claude-code:sonnet`, 10 records and 1 seed claim: about
  8.2k input and 1.5k output tokens, roughly $0.037 per reflection.
- For a typical CUDA run (50 rollouts × 5 steps ≈ 250 records) that is about 30 reflections, around
  $1, compared with 250+ codegen calls.

## Design decisions

1. **Edits are structured JSON validated against the log, not free text.** *Alternative:* let the
   model rewrite technique descriptions or a markdown knowledge file directly. *Why:* a rewrite
   cannot be audited, can quietly delete knowledge, and cannot be counted. Required citations plus
   status consistency checks (e.g. "helps" must cite an improved record) are a cheap guard against
   hallucinated lessons.
2. **Counts are of distinct record ids.** *Alternative:* add one per edit. *Why:* the same records
   show up in several reflections, through the existing claims in the prompt and through `force`
   flushes. Counting edits would inflate support with no new evidence.
3. **Time is measured in reflection epochs.** *Alternatives:* wall-clock time, or record count.
   *Why:* runs are bursty and machines differ. Epochs grow with the amount of evidence actually
   reviewed, so "untouched for 12 epochs" means "about 100 records went by and none were relevant".
4. **Only weak claims age out; strong claims just lose rank.** *Alternative:* decay everything.
   *Why:* this is what the requirement asked for. A well-supported lesson ("include mma.h for
   wmma") should not disappear because nobody tried wmma lately.
5. **Scope defaults to the cited records' common value; `"*"` widens it.** *Alternative:* default
   to general and let the model narrow it. *Why:* leaking a lesson across arch or target is the
   costly mistake (CUDA advice in a RISC-V prompt). A claim that is too narrow still shows up when
   the query leaves that field unset, because unset query fields match anything.
6. **Problem, slot and impl scope only for impl notes, matched strictly.** *Why:* "already tried"
   is local knowledge. General lessons should not carry a problem id that would hide them from
   other problems.
7. **"Already tried" tallies are deterministic and immediate, separate from LLM notes.**
   *Alternative:* wait for reflection. *Why:* parallel rollouts need dead-end information within
   seconds. The tally is free; the impl_note claims add the *why* later.
8. **Threshold of 1 for impl notes and pitfalls, 2 for the rest.** *Why:* a pitfall must cite a real
   failure, its fix is cheap to try, and every pitfall in the smoke batch was seen exactly once. With a
   threshold of 2 they would almost never reach the fix prompt. Applicability and parameter claims
   steer search, so they need repeated evidence.
9. **Contradictions as counts plus side-by-side opposite claims, never overwritten.**
   *Alternative:* replace the claim with the newer finding. *Why:* outcomes are noisy (±2% neutral
   band, shape effects). Showing both with counts lets the planner weigh them. The eligibility rule
   `support > contradict` drops the losing one automatically.
10. **Dedupe is cheap normalisation plus similarity, gated by an exact scope and polarity key and
    equal numbers.** *Alternative:* embeddings. *Why:* no embedding dependency or cost. The gate
    removes the dangerous merges (helps vs hurts, tile 64 vs 128). Paraphrases the cheap test misses
    are left to the optional LLM `consolidate()`, which may only merge claims that already share a
    dedupe key.
11. **State splits are proposal claims only.** *Why:* this is what the design asked for. Changing
    the state table changes the bandit arms and the plan prompts globally, so a human should do it.
12. **Fine stats use per-step geomean speedup, and the coarse fallback is a per-step aggregate in
    the same table.** *Alternative:* fall back to the legacy `predicted_speedup`. *Why:* the legacy
    value is measured against the entry's *first* baseline (`initial_elapsed_cycles`, set once per
    entry even across problems), so it means something different. Mixing the two in one fallback
    chain would compare unlike numbers. The legacy value stays as the third fallback, and all legacy
    fields are still updated.
13. **Knowledge goes in user messages, not system prompts.** *Why:* the tiered path depends on
    byte-stable system prompts for prefix caching.
14. **The knowledge object is a facade (`SemanticKnowledge`) and also builds from env.** *Why:* the
    runner script is outside this change's file ownership. Env enablement makes it usable without
    editing the script, and `knowledge=None` keeps behaviour identical.
15. **The cursor lives in `claims.json` and moves forward before the model call.** *Why:* after a
    restart, records are not reflected again. Concurrent callers cannot pick up the same batch. A
    failed call loses one batch of reflection, but the records stay in the log.

## Deviations from the agreed design

- `relevant()` has extra optional kwargs: `impl_id`, `kinds`, `include_ineligible`.
  `maybe_reflect()` has an optional `force`. `SemanticUpdater` has optional bounds and the
  `trace_dir`, `known_techniques` and `known_states` kwargs. No required signature changed.
- `ClaimStore.observe(record)` was added to feed the deterministic "already tried" tallies.
- Pitfalls have a support threshold of 1 (decision 8).
- `build_fix_messages` also gets an optional `knowledge_notes` (pitfalls). The design listed only
  the plan and codegen prompts, but this is where pitfalls pay off.
- `outcomes.py` and `knowledge/__init__.py` are unchanged, and `opt_rl_optimized_base.py` did not
  need changes.

## Live smoke test (claude-code:sonnet)

The test used 10 hand-written records across gemm and reduction on sm_89, with improved, regressed,
neutral, compile_error, wrong_output and runtime_error outcomes, plus one seeded weak claim.

- **Call 1:** the edits were sensible, but only 1 of 9 was accepted. The model nested `data` under
  the kind name, because the schema in the prompt was ambiguous. The prompt now shows flat per-kind
  `data` shapes and an example, and the validator unwraps the nested form. Re-validating call 1's
  raw response offline: 9 of 9 valid.
- **Call 2, with the fixed prompt:** 9 of 9 valid edits.
  - 1 `support` on the seeded claim;
  - 4 `applicability` claims (wmma helps compute-saturated gemm; extra smem staging on a wmma kernel
    hurts; cp.async double-buffering helps large-K gemm; shuffle has no effect on a tiny strided
    reduced dim);
  - 3 `pitfall` claims with fixes (missing `<mma.h>`, the K%16 tail, a misaligned float4
    reinterpret_cast);
  - 1 `interaction` (tensor cores, then cp.async tiling).

Artifacts are under `/scratch/dima/BlasterMerge/tmp/kb_semantic_scratch/` (`smoke_reflect.py`,
`smoke_call{1,2}/traces/*.json` with the prompt and raw response, `applied.json`, `claims.json`).

## Open issues

- **Several processes sharing one `claims.json`:** the last writer wins. Writes are atomic
  (tmp file plus rename), but nothing merges concurrent writes. Within one process the store is
  lock-protected. If the multi-slot flow runs separate processes, add a file lock with
  reload-and-merge on save, or a single writer.
- **`OutcomeLog.since()` reads the whole JSONL file** on every `maybe_reflect`. That is fine at
  thousands of records; past that, add a byte-offset cursor.
- **No `parameter` or `revise_description` edits came up in the smoke test.** Their validation is
  unit-tested only. Revised descriptions are injected as claims in the codegen prompt; they do not
  replace the static description in `get_technique_description`.
- **`op_class` defaults to `other` for KernelBench single-file problems.** Set `KB_OP_CLASS`, or
  pass a `SemanticKnowledge` with the right class, until problems carry a tag.
- **The RISC-V and OpenCL optimized agents are not wired up yet.** The pieces are
  backend-agnostic; only the CUDA flow was in scope.
