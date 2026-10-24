# Knowledge architecture v2: cross-target, mechanism-aware, hypothesis-driven

Status: design (Phase 2 of the multi-slot work). Implementation follows the order in
"Plan". Companion docs: `docs/semantic_update_pass.md` (claims + reflection, v1),
`docs/multislot_optimization.md`, ModelBlaster `docs/kernelblaster_multislot.md`.

## 1. Why

What the knowledge looks like today (v1):

| store | what it is | problem |
|---|---|---|
| technique DB (`agents/database.py`, `optimization_database.json`) | 16 named states → lists of technique names with rolling speedups | a state is a name + prose; an LLM matches a kernel to one; below 0.6 confidence it creates a new state and **copies the nearest state's techniques ×0.8**. States have no measurable definition, can't be tested, merged or reasoned about; they proliferate as near-duplicates. Techniques are name strings: no mechanism, preconditions or predicted effect. CUDA-centric. |
| claims (`knowledge/claims.py`) | semantic lessons with support/contradiction counts | scoped by exact tags (target, arch, op_class, state, technique): a lesson matches all tags or doesn't transfer at all. "Tiling helps when reuse is high" learned on sm_75 never reaches Adreno. |
| roofline / NCU state (`agents/opt_multislot.py`, `knowledge/ncu.py`) | a third state system, measured or estimated | disconnected from the DB's states. |
| implementation store (`knowledge/impl_store.py`) | verified code by op/signature/arch/shape bucket | knows code, not the concepts that explain it. |
| bandit priors | technique mean over every other key | flat: no notion of "close" keys (same family, parent state). |

Goals: (a) **cross-target learning** — share what transfers (tiling, memory-bound) and keep
what doesn't (cp.async, Adreno image paths, VLEN-specific strip-mining) where it belongs;
(b) **states the system can reason about** — defined by measurements, organised in a
hierarchy, proposed from evidence and validated, never spawned by copying; (c) **proactive,
creative optimisation** — explicit hypotheses with mechanisms and predictions, tested on a
budget, promoted or retired on evidence.

## 2. Abstraction levels

Every piece of knowledge lives at one level of a fixed three-level hierarchy and applies to
everything below it:

```
universal                      reuse, parallelism exposure, latency hiding, fusion, precision, vectorisation
└── family                     simt_gpu (CUDA, OpenCL/Adreno), vector_cpu (RVV), systolic (Gemmini)
    └── target                 cuda:sm_75, cuda:sm_89, opencl:adreno_740, rvv:vlen256, ...
```

The levels are fixed in code; **families and targets are data** (`targets.yaml`): name,
family, peaks (GFLOP/s, GB/s), machine width (SMs × warp, subgroups, VLEN), on-chip memory
sizes, profiler (ncu / adreno / mcycle / roofline). Adding a target is a data change; the
LLM does not invent families.

## 3. Canonical metrics (the vocabulary states are written in)

A per-target **metric adapter** turns whatever profiler exists into one vector of canonical
metrics, each with `value`, `source` (`measured` | `estimated`) and provenance:

| metric | meaning | CUDA (NCU) | estimated (any target) |
|---|---|---|---|
| `bw_util` | % of peak DRAM bandwidth | dram__throughput | bytes / time / peak |
| `compute_util` | % of peak compute | sm__throughput | flops / time / peak |
| `mem_util` | % of the busiest memory unit (L1/L2/DRAM) | Memory Throughput | = bw_util |
| `arith_intensity` | FLOP / byte (DRAM) | from counters | manifest bytes, flops |
| `parallel_fill` | work items / (machine width × waves needed) | achieved occupancy, waves/SM | grid size / width |
| `occupancy` | achieved / theoretical | Occupancy section | — |
| `launch_frac` | share of slot time that is launch/sync overhead | per-launch duration vs gap | time < latency floor |
| `l1_hit`, `l2_hit` | cache hit rates | MemoryWorkloadAnalysis | — |
| `reg_pressure` | registers per thread vs limit | LaunchStats | — |
| `divergence` | branch efficiency (when available) | — | — |

Missing metrics are absent (not zero). Predicates over an absent metric evaluate to
"unknown", never true.

## 4. States: predicates in a hierarchy

```yaml
- id: memory_bound            level: universal  parent: null
  predicate: {all: [{bw_util: ">=", v: 50}, {compute_util: "<", v: 40}]}
  mechanism: "time is set by bytes moved; fewer/cheaper bytes is the lever"
- id: dram_streaming          parent: memory_bound
  predicate: {all: [{l2_hit: "<", v: 50}]}
- id: cache_thrashing         parent: memory_bound
  predicate: {all: [{l1_hit: "<", v: 40}, {arith_intensity: ">=", v: 4}]}   # reuse exists but is lost
- id: latency_bound           parent: null
  predicate: {any: [{parallel_fill: "<", v: 0.3}, {launch_frac: ">=", v: 0.5}]}
- id: low_parallelism         parent: latency_bound  ...
- id: launch_overhead         parent: latency_bound  ...
```

* **Classification is deterministic**: evaluate predicates top-down, return the deepest
  matching chain (e.g. `memory_bound › cache_thrashing`). No LLM matcher. A metric vector
  that fits no child is classified at the parent; one that fits no root is `unexplained`
  (feeds state discovery).
* Predicates use a tiny JSON DSL (`all`/`any`/`not`, comparisons on canonical metrics,
  `op_class in [...]`), so they can be validated, stored and diffed.
* Target-level states are allowed (e.g. `adreno:texture_path_unused`) but must refine a
  family/universal parent.
* A state never owns a copy of technique values. Values come from statistics (§7), shrunk
  toward the parent.

**State discovery** (replaces "NEW_STATE_NEEDED → copy techniques ×0.8"):

1. Triggers: (a) within one state, a technique's rewards split into two clearly different
   groups (≥ 6 outcomes, difference in mean reward ≥ 0.1, the reflection pass already flags
   these as `state_split`); (b) ≥ 5 `unexplained` slot profiles.
2. The LLM gets the cluster: metric vectors (with sources), the two outcome groups, code
   diffs, and the parent's predicate. It must return a **predicate** (in the DSL) that
   separates the groups, the parent it refines, a mechanism, and predicted technique effects.
3. Validation (no LLM): the predicate must evaluate (no unknowns) on ≥ 80 % of the cluster,
   separate held-out outcomes better than the parent (reward variance within the two sides
   reduced by ≥ 30 %), and both sides need ≥ 3 outcomes. Otherwise the proposal is recorded
   as rejected with the reason (the LLM sees it next time).
4. Admitted as `probationary`; becomes `core` after ≥ 10 more classified outcomes keep the
   separation. Retired if separation collapses. Every change lands in a review queue
   (`knowledge/review.jsonl`) for humans; probation runs without waiting for review.

## 5. Techniques: mechanisms with predictions

```yaml
- id: tiling_fast_memory       level: universal
  mechanism: "stage a reused block in fast memory so each DRAM element is read once per tile"
  preconditions: {all: [{arith_intensity: ">=", v: 2}, {op_class: in, v: [gemm, conv, attention]}]}
  predicts: {bw_util: down, l1_hit: up, reg_pressure: up, occupancy: down_or_flat}
  risks: ["register spills at large tiles", "bank conflicts"]
  params: {tile_m: [16, 32, 64, 128], tile_n: ..., tile_k: ...}
  relations: {stacks_before: [register_blocking, double_buffering], conflicts: []}
  realizations:
    - scope: family:simt_gpu      how: "__shared__ (CUDA) / __local (OpenCL) tiles, barrier per k-step"
    - scope: target:cuda:sm_80+   how: "cp.async + multi-stage pipeline"      params_prior: {stages: 3}
    - scope: family:vector_cpu    how: "L1-resident blocks, strip-mined at VLEN"
```

* The **prediction** is what makes reasoning possible: after an attempt, the observed
  `metrics_delta` is compared with `predicts` (sign agreement on measured metrics) and the
  outcome is labelled `mechanism: confirmed | contradicted | unknown`. A speedup with a
  contradicted mechanism is a lucky or mislabelled edit; a slowdown with a confirmed mechanism
  says the lever is real but something else (spills, occupancy) dominated — that is what
  reflection writes claims about.
* **Realizations** carry the target-specific how-to and parameter priors, and link to
  implementation-store exemplars. The prompt shows the realization for the slot's target
  (falling back to its family), so CUDA spellings never reach an Adreno prompt.
* Seeded from the existing 16 states × ~80 techniques, the multi-slot `TECHNIQUES` catalogue,
  and the CUDA / OpenCL / RISC-V database footers — mapped onto mechanisms; duplicates merged.

## 6. Hypotheses: proactive, creative proposals

A hypothesis has a technique's shape (mechanism, preconditions, predicts, realization for the
target) plus `status: proposed → testing → promoted | retired`, a test budget and its
results. Generated (cheap model, structured JSON, deduplicated against techniques by mechanism
similarity) from:

| source | trigger | example |
|---|---|---|
| gap | a state where every applicable technique's estimate ≤ 0 | "memory_bound elementwise on Adreno: use image2d reads" |
| port | a realization exists for another family/target, not this one | CUDA warp-shuffle reduction → OpenCL subgroup reduce |
| combine | two techniques with complementary predicts that haven't been stacked | tiling + vectorised global loads |
| profiler rules | NCU / vendor rule hints with high estimated speedup | NCU "LaunchConfiguration 99 %" → split-K for small-M GEMM |
| code mining | a verified winner whose diff matches no known technique | name and describe what it did |
| graph level | ModelBlaster partition evidence | fusion / layout / split proposals (kb-proposal) |

The bandit reserves an **exploration budget** (default 15 % of steps) for hypotheses whose
preconditions hold on the chosen slot. Promotion: wins (reward > 0) on ≥ 2 problems, mean
reward's lower confidence bound > 0, mechanism confirmed in ≥ 50 % of measured attempts →
becomes a technique at the lowest level that covers its evidence. Retired after its budget
with no win. Every hypothesis outcome is evidence the reflection pass can use.

## 7. Statistics: shrinkage up the hierarchy

Reward statistics are kept per (technique or hypothesis, scope key) where the scope key is
(target, op_class, state) and its ancestors (family, …, universal, parent states). The
estimate used by the bandit and shown in prompts:

    est(k) = (n_k · mean_k + κ · est(parent(k))) / (n_k + κ)        κ = 3 by default

so a new target starts from its family's (then universal) evidence and moves to its own as
data arrives. UCB uses `est` with an exploration term on the local count. This replaces the
v1 flat cross-key prior.

## 8. Claims: levels and lifting

Claims keep their v1 machinery (citations, support/contradiction, thresholds, decay) and gain
`level` + `scope` and links to technique/state/hypothesis ids. Reflection gets two new edit
kinds:

* `lift`: move a claim up (target → family needs consistent support on ≥ 2 targets of the
  family; family → universal on ≥ 2 families) — validated against the cited outcomes'
  targets, not taken on the LLM's word;
* `mechanism`: an explanation tying a technique's predicted effects to observed deltas (the
  material for refining `predicts` and `risks`).

## 9. Store and retrieval

One SQLite database (`knowledge.db`, WAL mode: concurrent readers, serialised writers — this
also removes v1's single-writer `claims.json` limit) with tables: `targets`, `states`,
`techniques`, `realizations`, `hypotheses`, `claims`, `outcomes` (append-only, with canonical
metric vectors before/after and the state chain), `stats`, `review`. The implementation store
stays content-addressed on disk; its records gain `technique_ids`. Ontology seeds and
`targets.yaml` live under `data/kernelblaster/knowledge/` and are loaded idempotently.

Retrieval for a slot builds a budgeted prompt section, most specific first:

1. the slot's state chain (with the measured numbers that put it there);
2. techniques applicable to that chain whose preconditions hold, ranked by `est`, each with
   its target realization and the level of its evidence ("general GPU lesson", "sm_89 only");
3. eligible claims along the target and state chains, labelled by level;
4. hypotheses under test for this state (when the bandit picks one, its full spec);
5. "already tried" for this problem/slot.

## 10. Integration

* `MultiSlotAgent`: metric adapter → classify → select (techniques + hypotheses, hierarchical
  UCB, exploration budget) → prompt from retrieval → outcome with metric vectors, state chain,
  technique/hypothesis id, mechanism check.
* `SemanticUpdater`: v1 edits + `lift`, `mechanism`; plus two new passes on a slower
  cadence — state discovery (§4) and hypothesis generation (§6).
* The legacy single-file flows (`opt_ncu_rl_optimized`) keep working: their state analysis is
  mapped onto the new states through the adapter (NCU metrics → canonical), and the old JSON
  DB is read once to seed the ontology.
* ModelBlaster proposals become graph-level hypotheses with the same lifecycle.

## 11. Plan

1. Metric adapters (CUDA NCU, roofline estimate; OpenCL/RVV adapters as stubs with the same
   interface) + predicate states + deterministic classifier; the seed state hierarchy.
2. Technique ontology (mechanisms, preconditions, predicts, realizations) seeded from v1; the
   mechanism check on outcomes.
3. Hierarchical statistics in the bandit and retrieval.
4. Claim levels + `lift` / `mechanism` edits in reflection; SQLite store.
5. Hypothesis loop: generation sources, exploration budget, promotion/retirement.
6. Evidence-driven state discovery with validation and the review queue.
7. Evaluation (below).

## 12. Evaluation

* Offline: classify all recorded outcomes; check predicates are total enough (unknown rate),
  that seed states separate technique effects, and that the mechanism check agrees with
  speedups more often than chance.
* Online, small (Haiku): the system responds sensibly — states assigned, hypotheses proposed
  and tested, at least one state proposal validated or rejected with a reason.
* Transfer: learn on CUDA sm_75 Level1–3 subsets, then optimise the same problems on sm_89
  (L40S via `--evaluator ssh`) and OpenCL (when that target lands) with knowledge on vs off;
  metric: time/LLM calls to first win and final speedup at a fixed budget; ≥ 3 seeds each
  (single runs vary too much to conclude anything — see the ResNet18 runs).
* Budget: prototyping/testing ≈ $200 of Haiku/Sonnet; per-run token accounting
  (`llm_usage`), conservative price estimates for Bedrock (which returns tokens only).

## 13. Decisions (proposed defaults, for review)

| decision | default | alternative | why |
|---|---|---|---|
| hierarchy | fixed universal/family/target; families + targets as data | open, LLM-proposed families | families are a hardware fact; letting the LLM invent them adds noise for no gain |
| promotion gate | automatic probation, human review queue | human approval before use | keeps the loop moving; review catches bad generalisations |
| state definition | predicates over canonical metrics | LLM-described profiles | testable, deterministic, mergeable |
| value inheritance | statistical shrinkage to parent | copying technique lists into new states | no duplicated, drifting entries; evidence flows both ways |
| store | SQLite (WAL) | JSON files | concurrency, queries across entities, one source of truth |
