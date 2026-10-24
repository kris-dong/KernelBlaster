# Knowledge architecture v2: cross-target, mechanism-aware, hypothesis-driven

Status: steps 1-6 of "Plan" are implemented in `src/kernelblaster/knowledge/v2/` behind a switch
(`run_multislot.py --knowledge v2`, `KB_KNOWLEDGE=v2` for the single-file flows) and evaluated;
v1 stays the default (the evaluation shows no transfer benefit yet). Results and the deviations
from this design are in section 14. Companion docs: `docs/semantic_update_pass.md` (claims + reflection, v1),
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
| seed format (impl.) | JSON seeds | YAML | no PyYAML dependency (section 14) |
| `occupancy` (impl.) | achieved occupancy, % of hardware max | achieved / theoretical | what the profilers report directly (section 14) |
| seed roots (impl.) | five ordered roots, total over (mem_util, compute_util) | the design's partial roots | nothing falls between roots; estimates rarely reach 50 % (section 14) |
| stats path (impl.) | linear: target -> family -> universal at the leaf, then the state chain, then op class `*` | a lattice with several parents | one parent per node keeps the shrinkage recursion simple (section 14) |
| graph hypotheses (impl.) | keyed by ModelBlaster `graph_hints/v1` ids, decisions synced from MB | a KB-side lifecycle | MB already builds and decides graph changes (section 14) |
| metric adapters (impl.) | chosen by the agent's `TargetProfile.profiler` module; profile peaks fill `targets.json` gaps | a separate target switch | one place decides per-target behaviour (section 14) |

## 14. Implementation (steps 1-6) and evaluation

### Where things are

| step | module (`knowledge/v2/`) | seed data (`data/kernelblaster/knowledge/`) |
|---|---|---|
| 1 levels, targets | `targets.py` | `targets.json` |
| 1 canonical metrics, adapters | `metrics.py`: `NCUAdapter` (per-slot `ncu.summarize`), `NCUTextAdapter` (single-file text reports), `roofline_vector` (any target), `AdrenoAdapter` / `MCycleAdapter` stubs | |
| 1 predicate DSL | `predicates.py` (`all`/`any`/`not`, metric comparisons incl. `ref` to another metric, `op_class`/`dtype`/`family`/`target in`) | |
| 1 states + classifier | `states.py` | `states.json` (15 seed states, `legacy_map` for the 16 technique-DB and the v1 roofline names) |
| 2 ontology + mechanism check | `ontology.py` (`seed_legacy` reads the old JSON DB and the three footers once) | `techniques.json` (27 curated, 84 v1/DB/footer names folded in as aliases, 21 imported as `slot=false`) |
| 3 hierarchical stats | `stats.py` | |
| 4 store, claims, reflection | `store.py` (`KnowledgeDB`, `SQLiteOutcomeLog`, `SQLiteClaimStore`), `reflect_v2.py` (`lift`, `mechanism`, `SemanticUpdaterV2`); `claims.Claim` gains `level`, `family`, `hypothesis_id`, kind `mechanism` | |
| 5 hypotheses | `hypotheses.py` (`HypothesisBook`, `validate_spec`, graph hypotheses) | |
| 6 state discovery | `discovery.py` | |
| facade | `knowledge.py` (`KnowledgeV2`: vector, classify, select, retrieve, record, maybe_upkeep) | |

Integration: `MultiSlotAgent(knowledge_v2=...)` (the bandit, roofline state and claims lookup are
replaced; nothing changes when it is unset); `run_multislot.py --knowledge v2 --upkeep-model M
[--upkeep-every 8 --explore-frac 0.15]`; `SemanticKnowledge(version="v2")` / `KB_KNOWLEDGE=v2`
for `opt_ncu_rl_optimized` (its NCU text reports are passed to `record()` and classified through
`NCUTextAdapter`; the technique-DB state is kept as `legacy_state` and the DB keeps working as
before). `scripts/eval_knowledge_v2.py` runs the offline checks on v1 logs or v2 databases.
Tests: `tests/test_knowledge_v2.py`.

### Deviations (smallest change the code forced; see also the Decisions table)

| design | implemented | why |
|---|---|---|
| `targets.yaml`, YAML seeds | JSON (`targets.json`, `states.json`, `techniques.json`) | no PyYAML in the environment; JSON loads the same and is diffable |
| `occupancy` = achieved / theoretical | achieved occupancy, % of the hardware maximum | it is what NCU and the single-file report give directly; the theoretical limit is reflected in `reg_pressure` and the `register_limited` / `stalled_low_occupancy` states |
| seed roots `memory_bound` (bw >= 50, compute < 40), `latency_bound` | five ordered roots: `latency_bound`, `balanced_bound`, `memory_bound` (`mem_util >= 50`, compute below it), `compute_bound`, `underutilized` (both < 50) | with both utilisations present the roots are total (nothing is unexplained); roofline estimates rarely reach 50 % of peak, so the old 15 %-threshold roofline names map to `underutilized > underutilized_memory|compute` |
| stats ancestors "(family, ..., universal, parent states)" | one linear path: target -> family -> universal at the leaf state, then the state chain at universal, then op class `*` (fp16 adds a most specific node) | the shrinkage recursion needs one parent per node; a parent node includes its children's data (empirical-Bayes approximation) |
| `realizations` linked to impl-store exemplars | realizations carry `how` / `params_prior`; impl-store records keep `techniques` (now ontology ids) | the store's retrieval by op/shape already supplies exemplars; no schema change needed |
| claims "level + scope" | `level` + `family` on the v1 claim; target-level claims keep v1's backend + optional arch scope | v1 claims load unchanged as target-level |
| promotion "mechanism confirmed in >= 50 % of measured attempts" | same; with no measured attempt the gate is skipped and recorded ("mechanism unmeasured") | roofline-only targets could never promote otherwise |
| hypothesis generation "cheap model" | the upkeep model is configurable (`--upkeep-model`); runs below used Sonnet for generation and state discovery, Haiku for codegen and reflection | design reasoning quality matters most there |
| graph-level hypotheses with the same lifecycle | kb-proposals become `kind=graph` hypotheses keyed by ModelBlaster's `graph_hints/v1` ids (`fuse:<kind>`, `split:<slot>`); MB's accepted / rejected decisions are synced into promoted / retired (`sync_graph_hints`) | MB's hint layer already owns building and deciding graph changes; KB does not duplicate it |
| "already tried" tally in the claim store | derived from the outcomes table | one source of truth, safe across processes |

### Evaluation

**Offline** (`scripts/eval_knowledge_v2.py`, 142 recorded v1 outcomes from seven earlier runs on
sm_75 / sm_89; roofline vectors rebuilt from the problem manifests and the measured slot times):
every record classifies (unknown rate 0 %; 126 at depth 2, 16 at depth 1). The seed thresholds
put most roofline-estimated slots under `underutilized` (the old 15 %-of-peak names
`compute_bound` / `underutilized_compute` both land in `underutilized > underutilized_compute`).
Do states separate technique effects (eta^2 of log2 speedup by state, within technique,
permutation p over 2000 shuffles): v1 roofline state 0.62 (p < 0.001), v2 leaf 0.61 (p < 0.001), v2
root 0.19 (p = 0.07), op class alone 0.68. Within technique x op class (does the state add
anything beyond the op class): v1 0.27 (p = 0.04, 87 outcomes), v2 leaf 0.37 (p = 0.05, 36
outcomes: fewer groups have two v2 states). So the seed leaves are about as informative as v1's
roofline names on estimated data, and neither is much better than the op class. The old logs
have only 4 records with NCU deltas (2 decided), too few for the mechanism check.

On the v2 database of the L40S NCU run (B1 below; 90 outcomes, 43 with measured deltas):
mechanism check decided on 20 (23 ties / no predicted metric measured), agreement with the
outcome 0.70 vs 0.485 by chance; P(improved | confirmed) = 6/7 = 0.86, P(improved | contradicted)
= 5/13 = 0.38, one-sided Fisher p = 0.058. Suggestive, not conclusive at this size. The
contradictions are informative: on naive seed kernels `thread_coarsening` / `loop_unrolling_and_ilp`
win while `compute_util` (SM throughput) *falls* (the naive kernel was busy issuing redundant
instructions), against their `predicts`; reflection wrote validated `mechanism` claims saying
exactly that (see below) -- material for refining `predicts`.

**Online** (Bedrock Haiku codegen + reflection, Sonnet for the upkeep passes; problems exported
by ModelBlaster: Level2 1_Conv2D_ReLU_BiasAdd, 12_Gemm_Multiply_LeakyReLU, 17_Conv2d_InstanceNorm_
Divide, 99_Matmul_GELU_Softmax; 3 rollouts x 5 steps per problem):

| run | where | what the system did | tokens (in/out) | est. $ |
|---|---|---|---|---|
| A1 | sm_75 local (counters unavailable: roofline vectors) | 45 outcomes classified (`underutilized_compute` 30, `memory_bound` 9, `underutilized_memory` 6); 10 hypotheses proposed (port / combine triggers), 2 tested and **promoted** (target level, mechanism unmeasured: `cuda_streaming_cs_stores_for_elementwise`, `inline_affine_index_instead_of_index_tensor_load`; wins on 2 problems, small rewards); state discovery: `underutilized_compute_near_idle` (`compute_util < 5 and mem_util < 0.5`) **admitted** as probationary (variance reduced 95 % on the held-out third), `underutilized_compute_low_util` **rejected** ("sides 2/4, need >= 3 each"), one cluster declined by the model ("groups interleave on every metric") | codegen 198k/224k, reflection 24k/11k, upkeep 25k/13k | 1.66 |
| B1 | L40S sm_89 via ssh, `--ncu all` (copy of A1's DB) | measured vectors: `balanced_bound` (naive conv), `compute_bound > matrix_compute`, `memory_bound > cache_thrashing`, `on_chip_memory_bound`, `low_parallelism`; 11 hypotheses from **profiler** (NCU rule) and **code-mining** triggers (e.g. `gemm_address_compaction_to_reach_48_regs_for_5_cta_residency`, `stream_k_persistent_gemm_with_l2_swizzled_order`); 10 validated `mechanism` claims (3 rejected: "observed down but the cited deltas are mostly flat") | codegen 220k/201k, reflection 37k/20k, upkeep 29k/16k | 1.68 |
| lift | reflection pass on a copy of B1's DB, mixed sm_75 + sm_89 batch | with deterministic lift candidates in the prompt, one claim **lifted** target -> `simt_gpu` family, validated on its cited records (support on cuda:sm_75 and cuda:sm_89); a universal lift in the unit tests is rejected ("needs >= 2 families"). Without the candidates (B1 live) the model never saw the sm_75 claims and proposed no lift; fixed in 1d7a1e4 | 24k/9k (2 calls) | 0.07 |
| A2 | sm_75 local, seed 1, continues A1 | 45 more outcomes; one discovery cluster declined by the model with reasons; knowledge snapshot for the transfer (90 outcomes, 23 claims, 15 hypotheses: 2 promoted, 2 testing, 11 proposed, 1 probationary state) | codegen 236k/208k, reflection 33k/15k, upkeep 17k/6k | 1.52 |

| A3 | sm_75 local, seed 2, after the unit fix (012e5ac) | the review pass **retired** 10 stored hypotheses with the reason ("preconditions in the wrong unit: bw_util is in percent (0-100) but compared with 0.85"); 2 new ones with correct units; `underutilized_compute_gemm_high_ai` (`arith_intensity > 400`) **admitted** (variance -92 %, judged on the full cluster: < 4 held out); a re-proposal of an existing state rejected ("already exists"), two clusters declined by the model | codegen 248k/203k, reflection 30k/15k, upkeep 13k/4k | 1.47 |
| transfer x 9 | L40S sm_89, `--ncu best`, 3 x 4 steps per problem | see below; in the knowledge-on runs reflection **lifted 9 claims** target -> `simt_gpu` family on sm_75 + sm_89 support and **rejected 5 lifts** with the per-target counts ("needs consistent support on >= 2 of its targets, has ['cuda:sm_89'] (support {sm_75: 3, sm_89: 5}, contradict {sm_75: 3})") | ~1.1 per run | 9.9 |

Across the online runs: states were assigned from metrics on every step (roofline on sm_75,
NCU-measured on sm_89); hypotheses were proposed from five sources (gap triggers never fired:
no state had >= 4 locally tried techniques all <= 0), tested under the exploration budget,
promoted (2) and retired (10, plus budget retirements in the unit tests); state proposals were
admitted (2, both still probationary) and rejected with reasons (6); lifts happened only with
two-target evidence.

On the NCU-measured data (the L40S run plus the three knowledge-off transfer runs: 198
outcomes, 85 with measured deltas), the seed states do separate technique effects: eta^2 0.38
(p = 0.04) for leaves and 0.48 (p = 0.005) within technique x op class, against 0.14 (p = 0.24)
for the op class alone. On roofline-only sm_75 data (135 outcomes) they do not (leaf 0.16,
p = 0.37; op class 0.31, p = 0.02). The mechanism check on the 33 decided measured records:
agreement 0.70 vs 0.51 by chance, P(improved | confirmed) = 15/17 = 0.88 vs 8/16 = 0.50 for
contradicted, one-sided Fisher p = 0.02.

**Transfer** (learn on sm_75: A1-A3, 135 outcomes, 3 Level2 problems; then the same problems on
the L40S, knowledge on = a copy of that database with an empty implementation store, off = an
empty database; both v2, Haiku codegen + reflection, no upkeep model, 3 seeds each). Final
partition speedup over the exported seed (geometric mean, log s.d.):

| problem | off | on (round 1) | on (round 2, after the unit fix) |
|---|---|---|---|
| 1_Conv2D_ReLU_BiasAdd | 35.7x (0.20) | 30.4x (0.25) | 27.3x (0.10) |
| 12_Gemm_Multiply_LeakyReLU | 45.8x (0.08) | 48.0x (0.14) | 46.8x (0.10) |
| 17_Conv2d_InstanceNorm_Divide | 36.6x (0.23) | 31.8x (0.21) | 34.2x (0.04) |

LLM calls to the first win (> 1 % of the partition): 1 in every run for problems 1 and 12 (the
first rewrite of a naive seed always wins), 3-6 (off) vs 2-5 (on) for problem 17. Pooled over
problems (log speedup centred per problem), on / off = 0.92 (round 1, permutation p = 0.32) and
0.90 (round 2, p = 0.14). **Conclusion: no measurable benefit from sm_75 knowledge on sm_89 at
this scale; the point estimate is slightly negative and within the seed-to-seed spread.** Round 1
tested no hypothesis at all (every learned hypothesis had fraction thresholds on percent
metrics, so its preconditions were False on measured slots -- the bug fixed in 012e5ac); in
round 2 hypotheses took 0-3 of 12 steps, mostly on small elementwise slots (rewards <= 0.002).
Caveats: the seeds are naive kernels, so a 25-50x first-step gain dominates the final number and
leaves little room for priors; off-arm runs ran one at a time on GPU 3, on-arm runs in parallel on
GPUs 2 and 3 (shared local compile CPU); 12 steps per problem.

LLM spend (ledger `who=knowledge-v2`): $16.34 of the $150 allowance (Haiku codegen and reflection
$15.51, Sonnet upkeep in A1-A3 and B1 $0.83).

### Open issues / next steps

* Predictions: `thread_coarsening` / `loop_unrolling_and_ilp` predict `compute_util` up, but on
  naive kernels it falls with a win; the validated `mechanism` claims say so. Close the loop:
  refine `predicts` (or condition them on the state) from eligible mechanism claims.
* Gap triggers never fired and port triggers dominate hypothesis generation (footer techniques
  without a CUDA realization); give gap/profiler/code-mining triggers priority and cap port.
* Promotion with tiny rewards (mean 0.005) passes "LCB > 0" on two or three attempts; consider a
  minimum effect size or >= 3 attempts.
* Seed thresholds at 50 % of peak put most roofline-estimated slots under `underutilized`;
  state discovery admitted refinements but both are still probationary (probation needs 10 more
  decidable outcomes of the trigger technique).
* The transfer needs a harder setting to be informative: stronger seeds (e.g. ModelBlaster's
  kernel-cache winners), more steps, more problems/op classes, and an OpenCL target.
* Graph hypotheses are recorded from kb-proposals and synced from ModelBlaster's graph_hints
  decisions, but nothing runs `sync_graph_hints` automatically yet.

A v1 bug surfaced on the way: `parse_json_response` cut JSON at a ```` ```cuda ```` mentioned
inside a claim text and silently dropped whole reflection passes (fixed, tested).
