# Multi-slot optimisation of ModelBlaster partitions

ModelBlaster exports a graph partition as a problem with N kernel *slots* (one per C entry
point it calls); KernelBlaster optimises it one slot per step and keeps what it learns for
other kernels and problems. The end-to-end design, the problem format (`mb-kb-problem/1`) and
the full list of design decisions are in ModelBlaster's
`docs/kernelblaster_multislot.md`; this page covers the KernelBlaster side.

## Pieces

| file | role |
|---|---|
| `data/sources/base.py` | `Problem.slots` (+ `slot_artifacts`, `is_multislot`); single-file problems are one `kernel` slot |
| `data/sources/modelblaster_source.py` | `get_source("modelblaster", root=...)`: finds exported problem dirs |
| `src/kernelblaster/knowledge/slots.py` | `SlotProblem` (manifest, assembly, neighbours, fusion-candidate pairs), `parse_driver_output`, `LocalEvaluator` (nvcc), `ServerEvaluator` (compile + exec servers), OpenCL evaluators (below) |
| `src/kernelblaster/agents/opt_multislot.py` | `MultiSlotAgent`, roofline `slot_state`, `TechniqueBandit`, prompts, `propose_fusions` |
| `src/kernelblaster/knowledge/impl_store.py` | implementation knowledgebase: verified sources, retrieval, winners export |
| `src/kernelblaster/knowledge/{outcomes,claims,reflect,semantic}.py` | outcome records and the semantic update pass (`docs/semantic_update_pass.md`) |
| `src/kernelblaster/agents/utils/agentic_llm.py` | `claude-code:<model>` (Claude Code CLI, no key) and `stub:<name>` (tests) providers |
| `scripts/run_multislot.py` | runner |
| `scripts/time_problems.py` | check + time problems with the local / ssh evaluators outside the agent loop (ModelBlaster's model result, hint re-measurement, graph-optimiser profiles and final verification), paired with an incumbent; on one GPU of a required model |

## Running

```bash
# problems exported by ModelBlaster: python -m modelblaster.pipeline.kb_export IR_DIR OUT
python scripts/run_multislot.py OUT --out runs/x --knowledge-dir knowledge \
    --model claude-code:sonnet --reflect-model claude-code:haiku \
    --arch 75 --rollouts 3 --steps 5 --export-winners runs/x/winners
# through the compile/exec servers instead of local nvcc:
KERNELBLASTER_ALLOW_SHARED_GPU=1 python scripts/run_multislot.py ... --evaluator server --gpu titanrtx
```

`--knowledge-dir` holds `impl/` (implementation store), `outcomes.jsonl` and `claims.json`;
reuse it across runs so later problems start from earlier winners and lessons. Outputs per
problem: `result.json` (baseline/final times, per-slot results, every step), `best/slots/*.cu`,
`proposals.json`.

## Agent loop

Per step: sample a slot (time share × stall decay × exploration bonus) → classify its roofline
state from the manifest's bytes/FLOPs and the measured time → pick a technique (UCB1 keyed by
op class and state, untried arms seeded from the technique's mean elsewhere) → the LLM rewrites
that slot only, with neighbours' signatures, the C reference, shapes, store examples, claims
and "already tried" notes → assemble, build, run the candidate in the persistent runtime
(partition + isolated-op checks, the partition replayed as a CUDA graph, paired with the
rollout's current state) with a bounded fix loop → reward = fraction of the end-to-end
partition time saved → outcome record, store record, claim tally. Parallel rollouts share a
per-slot composite best that is re-verified at the end. After the run, `propose_fusions` and
`propose_splits` write `kb-proposal/1` for ModelBlaster. The reward and the runtime are
described in the next section.

## End-to-end reward and the persistent runtime

### Reward

A step is accepted and rewarded on the **whole partition's forward pass as deployed**, not on
the changed slot's own time.

- **Deployed time.** The driver's `MB_E2E_US` is the median CUDA-graph replay of the whole
  partition when it is capturable, else the eager launch sequence. `EvalResult.deployed_us`
  falls back to the graph time, then the eager time, for drivers that predate the line.
- **Pairing.** The persistent host times the candidate and the rollout's current state in
  the same process, rep by rep, alternately (`incumbent=`, driver `ref=`). The gain is
  `1 − median(candidate / incumbent)` (`e2e_gain`). Both sides use the same method: both graph
  replays when both capture, else both eager.
- **Unpaired evaluators.** The one-process path and the server evaluator compare against the
  incumbent's time from an earlier run.
- **Accepting a gain.**
  - A gain of `--e2e-accept-pct` (default 1 % of the partition) or more is re-measured with
    every check, paired against the *composite* best.
  - It joins the composite if it holds there too. The reward is the smaller of the two gains.
  - Smaller moves are neutral. The rollout keeps a candidate unless it is more than half the
    threshold slower.
- **Status** of outcome records (improved / neutral / regressed) is the same threshold on the
  end-to-end gain.
- **Not graph-capturable is a defect.** A verified candidate that makes a capturable
  partition fall back to eager launches (a host sync, an allocation or a host copy per call)
  goes back to the fix loop with the capture error. If it stays uncapturable, it is compared
  eager to eager, and the step records `graph_capture`.
- **The old reward stays available.** `--reward slot` (`KERNELBLASTER_MULTISLOT_REWARD=slot`)
  keeps the previous rule: slot time saved over the sum of slot times, accepted at
  `confirm_pct`. In the end-to-end mode every step records what that rule would have decided
  on the same runs (`context.e2e.slot_rule`).

### Attribution and the launch signal

Slot times are diagnostics. They choose the slot to work on, feed the prompt, the roofline and
v2 metrics, and the proposals.

- **Kernel time per op.** The driver measures it with CUPTI activity records, for ops run back
  to back as in the partition (`MB_ATTR_SLOT_US` / `MB_ATTR_OP_US`, with launches per slot).
  `EvalResult.slot_time()` prefers it to the event-timed `MB_SLOT_US`, which charged tiny
  slots 2–3× their kernel time (ModelBlaster `docs/gpu_harness_perf.md`).
- **Slot share** for sampling is a slot's kernel time plus its launches' share of the
  deployed overhead (`slot_weights`).
- **Overhead.** `MB_LAUNCH_OVERHEAD_US` is deployed or eager time minus kernel time
  (`partition_runtime`: `launch_overhead_frac`, `graph_saving_frac`, `launch_bound`).
- **When the partition or slot is launch-bound** (eager overhead ≥ 15 %, a graph saving ≥ 5 %,
  more than 1.5 launches per call, or kernels under 5 µs):
  - the prompt says that fewer launches count as much as faster kernels;
  - the v1 bandit also offers `kernel_fusion_in_slot` and `persistent_kernel`;
  - knowledge v2 gets the measured `launch_frac` and launch count, so its `launch_overhead`
    state and `persistent_kernel` technique apply.
- **Prompt context.** Every prompt states the deployed time, kernel time, launches and
  overhead, and that the slot must stay graph-capturable.
- **Proposals.** `propose_fusions` estimates a fusion as the consumer's kernel time plus its
  launch overhead. Before, it used the event-timed consumer time: the ResNet18 conv+bn
  proposal fell from 13 % to 5.5 %, and the measured fusion saved 4.2 %.

### Records

| record | new fields | compatibility |
|---|---|---|
| `OutcomeRecord` | `timing_basis` (`e2e_graph`, `e2e_eager` or `slot_events`; `None` = written before), `e2e_before`, `e2e_after`, `partition_effect` | `metric_*` stay the slot's times; readers drop unknown fields (`OutcomeRecord.from_json`) |
| v2 outcome | the reward is the end-to-end gain, the same unit as before (a fraction of the partition); `extra.e2e` holds the decision | unchanged schema |
| `result.json` | `reward`, `baseline_e2e_us`, `final_e2e_us`, `e2e_speedup`, `final_e2e_ratio` (final vs seeds, paired), `runtime` (baseline/final `partition_runtime`, evaluator host stats), `attr_op_us`, per slot `baseline_attr_us` / `final_attr_us` / `launches`; steps `reward_basis`, `e2e_before`, `e2e_after` | old fields unchanged (`speedup` is still eager) |
| `summary.json` | `reward`, `baseline_e2e_us`, `final_e2e_us`, `e2e_speedup`; `load_balance.cuda.runtime` | |
| `result.json` | `gpu`: the GPU the final (else the baseline) numbers were timed on: `name` (model), `uuid`, `host`, `label`, `index`, `arch` (`EvalResult.gpu`) | absent in older runs; ModelBlaster then cannot match the model's timing to it |

### The hardware a number belongs to

- **Recorded device.** Every `LocalEvaluator` / `SSHEvaluator` result records the device it
  was leased (`EvalResult.gpu`), and `result.json` keeps the final one (`gpu`).
- **Pinning a model.** `gpu_model="NVIDIA L40S"` (`--gpu-model`) keeps only the selected GPUs
  of that model; names compare normalised, so `"l40s"` matches. With none of them, `pool()`
  raises `NoDevice`, naming the models it found. A run never falls back to another model.
- **Capping the pool.** `max_devices=N` keeps at most N devices; 1 puts every run on one GPU.
- **`scripts/time_problems.py`** times a list of jobs with these evaluators: problem, sources,
  optional incumbent, reps.
  - The jobs run in rounds (rep k of every job, then rep k + 1) or job by job.
  - On the persistent host an incumbent is paired in the same request. With one process per
    run it is evaluated right after the candidate.
  - Output (`kernelblaster.time_problems/v1`): every `EvalResult`, the device of each, the pool
    and the host statistics.
- **Who uses it.** ModelBlaster's `kb_model_run` (the whole-model result) and, with
  `--ssh-host`, every other GPU step of `kb_graph_opt`: the per-round whole-model profiles
  (per-op times, CUPTI attribution and launch overhead from the `EvalResult` of a full run,
  `reps` = the profile runs), the hint re-measurement and the final model-level verification
  (one full run: the driver checks every output and op against the references linked into the
  problem). All of them run on one GPU model -- the one the KernelBlaster runs recorded, or the
  first profile's -- so the plan, the decisions, the check and the model number share the
  hardware (ModelBlaster `docs/graph_optimizer.md`, "Where the GPU steps run", and
  `docs/gpu_harness_perf.md`, "The model's result").

### Persistent runtime (`knowledge/runtime_host.py`)

`LocalEvaluator` and `SSHEvaluator` run candidates in a persistent driver host per (problem,
GPU) when the driver has one (`--runtime auto`, the default; `host`; `process` for one
process per run).

- **Host.** The problem's `driver.cpp` built with `-DMB_HOST`. It keeps the CUDA context,
  buffers and reference data.
- **Candidate.** A shared object of the slot units + tail (`-shared -fPIC`, per-thread
  default stream, cached by content), dlopen'ed for one request.
- **Protocol.** One tab-separated `EVAL` line per request. The reply is the one-process
  driver's output for the same `MB_*` variables, then `MB_HOST_DONE`.
- **GPU lock.** The host takes the GPU's flock (`resources.gpu_lock_file`) for each request
  only. An idle host does not block other runs on its GPU.
- **SSH.** The host runs as `ssh host ./kbhost_<sha>` over the multiplexed connection. Its
  binary is copied once per problem and host. Shared objects (about 1 MB) are copied per
  candidate, and at most 48 are kept per host. A `data.bin` too big to link is copied once per
  host and data sha1, and the host is started with `MB_DATA_PATH` pointing at the copy.
- **Failures** are runtime errors of the candidate, and the next request starts a new host:
  - a crash ends the stream;
  - a hang is killed at the timeout;
  - a sticky CUDA error is reported by the host itself (`poisoned=1`).
- **Recycling.** The host exits by itself after 500 requests, after 1 GiB of leaked device
  memory, or when its stdin closes.
- **Fallback.** A host that cannot be built or started falls back to one process per run.
- **Server path.** `ServerEvaluator` (KernelBlaster's exec server) stays one process per run.
  It parses the same lines, eager only, without attribution: the server builds the driver
  without CUPTI's headers.

Measurements and design: ModelBlaster `docs/gpu_harness_perf.md`; bench script
`scripts/bench_runtime.py`.

## Proposals

- `fuse`: a producer -> consumer pair where the consumer is elementwise and the only reader of
  the producer's output; gain = the consumer instances' time. Producers or consumers may already
  be fused ops, so chains grow over rounds.
- `split`: a slot whose instances fall into several shape buckets and whose per-op times
  (`MB_OP_US`) show the one kernel much less efficient (fraction of the binding roofline peak) on
  some groups than on its best; gain = every group at the best group's efficiency. ModelBlaster
  gives each group its own slot (`<slot>__v<k>`), seeded with the current kernel.

ModelBlaster turns proposals into graph hints, builds original and variant problems of the
affected ops, has them optimised with the same budget and decides on re-measured times
(ModelBlaster's `docs/graph_optimizer.md`). `--plugin FILE` runs a python file first, e.g. to
register `stub:` models when ModelBlaster drives this runner as a subprocess.

## Tests

`tests/test_multislot_agent.py` runs the whole agent on the LeNet fixture
(`tests/fixtures/mb_lenet`, exported by ModelBlaster) with a scripted `stub:` LLM: a real
float4 improvement, no-op rewrites, and a broken candidate repaired by the fix loop; and reuse
of a stored winner by a second problem. Needs nvcc and a GPU (skipped otherwise).
`tests/test_multislot_units.py` covers proposals (fusion, split), the roofline state, bandit
transfer and driver-output parsing without a GPU. `tests/test_runtime_host.py` covers the host
protocol and crash recovery against a scripted fake host, the end-to-end reward decision
(a fewer-launches win the slot rule rejects, a slot-only gain it would take) with a scripted
evaluator, record compatibility, and, with a GPU, the real host on the LeNet fixture: the same
checks and errors as the one-process driver, and recovery from a segfault, an illegal address
and a hang.

## NCU

`--ncu best|all` profiles slots with NCU (`knowledge/ncu.py`): one run over `regex:mbslot_`
with the driver in `MB_MODE=profile`, rows attributed by the demangled namespace. The measured
bound and metrics replace the roofline state in prompts and the bandit key; outcome records get
`metrics_delta`. Needs counter access; otherwise NCU is disabled after the first failure.
`--evaluator ssh --ssh-host user@host --remote-dir DIR [--cuda-visible-devices auto|2,3]` builds
here and runs the self-contained binary on remote GPUs (e.g. L40S with counter access);
`--ssh-host` repeats for several hosts, and by default each host's idle GPUs are used.
Problems whose `data.bin` is read at run time (above 1 GiB, e.g. SwinV2) are supported:
`data.bin` is copied to `DIR/data_<sha12>/` once per host before a GPU is leased, and removed by
`close()` unless `KERNELBLASTER_SSH_KEEP_DATA=1` (see *Load balancing* in `docs/fast_paths.md`).
Problems run `--parallel-problems` at a time (default: the number of GPUs in the pool) over one
shared implementation store, outcome log and claims; every timed run and NCU pass leases its
own GPU. See *Load balancing* in `docs/fast_paths.md`.

## OpenCL / Qualcomm Adreno

Problems with manifest `target: "opencl"` (ModelBlaster `kb_export --target opencl`; design in
ModelBlaster's `docs/opencl_adreno.md`) have C++11 slots that hold OpenCL C programs as raw
strings plus the host launcher, on ModelBlaster's OpenCL runtime (SVM pointers, `mb_launch`,
cl_event timing). What is target-specific on this side:

| piece | OpenCL |
|---|---|
| `opt_multislot.TargetProfile` | system prompt (`agents/multislot_opencl.py`: slot format, runtime API, OpenCL C 1.2 / no fp64 rules, Adreno guidance), ```` ```cpp ```` fence, technique catalogue (vectorised 128-bit I/O, `__local` tiling, cooperative reductions, register tiling, work-group size, compile-time constants, mad/native math -- what the Adreno RL flow found), roofline peaks per device (`adreno650` ...; `titanrtx` for the local stand-in), arch label = device name, profiler |
| records | `ImplStore`, `OutcomeRecord` and claims keyed by `(target, arch)`: CUDA and OpenCL winners and lessons never mix |
| evaluators (`knowledge/slots.py`) | `OpenCLLocalEvaluator` (g++ + this host's OpenCL GPU), `AdrenoEvaluator` (scp one `main.cpp`, build with the board's g++, run, `rm -rf`; data.bin copied once per problem, `close()` removes the remote dir), `OpenCLServerEvaluator` (OpenCL compile server -- `compile_opencl.py` now builds `.cpp` mains with g++ -- and Adreno exec server, data.bin as a kernel file). Slots that call `clEnqueue*`/`clFinish` directly are rejected (all work must go through the timed `mb_launch`) |
| profile (`knowledge/opencl_profile.py`) | `--ncu best|all` runs the driver's `MB_MODE=profile` and parses per-kernel cl_event times and launch counts, `CL_KERNEL_WORK_GROUP_SIZE` (falls with register pressure on Adreno), private/local memory; hints and `latency_bound` (many short launches) go into prompts and outcome records (same interface as `knowledge/ncu.py`) |

```bash
# on the board (RB5 over ssh; default host $KERNELBLASTER_ADRENO_BOARD_HOST)
python scripts/run_multislot.py OUT --out runs/cl --evaluator board --board-host root@rb5 --arch adreno650 \
    --model us.anthropic.claude-haiku-5-5 --ncu best
# functional stand-in: any local OpenCL 2.0 GPU with coarse-grained SVM (times are not Adreno's)
python scripts/run_multislot.py OUT --out runs/cl --evaluator local --arch titanrtx ...
```

`tests/test_multislot_opencl.py`: units (profiles, prompts, peaks, store separation, the direct-
enqueue guard, the server evaluator's plumbing with faked servers) everywhere; with g++ and an
OpenCL GPU, the OpenCL LeNet fixture (`tests/fixtures/mb_lenet_opencl`) baseline, a wrong slot
attributed, the profile run, and the scripted agent end to end; `KB_OPENCL_BOARD_HOST=user@host`
(`KB_OPENCL_BOARD_DIR`) runs the board evaluator against a real host.

## Model heterogeneity

`agents/model_router.py` decides which model makes each LLM call, in the same way as the legacy optimized-RL flow's `MODEL_*` tiers.

| role | used for | set by |
|---|---|---|
| `codegen_simple` | mechanical techniques (coalescing, vectorising, launch config) | `--model-simple`, `MODEL_CODEGEN_SIMPLE` |
| `codegen_hard` | structural techniques (tiling, implicit GEMM, tensor cores, fusion, flash) and escalations | `--model-hard`, `MODEL_CODEGEN_HARD` |
| `fix` | compile, runtime and mismatch repairs | `--fix-model`, `MODEL_FIX` (default: `codegen_simple`) |
| `reflect` | semantic update pass | `--reflect-model` |
| `upkeep` | v2 hypothesis generation and state discovery | `--upkeep-model` |

- **Defaults and precedence.** A role with no setting falls back to `--model`. Flags win over `--models-config FILE.json` (`{"models": {role: model}, "policy": "..."}`), which wins over the environment variables.
- **Policy.** `--hetero-policy` (or `KB_HETERO_POLICY`; default `category,fix-escalate`) is a comma-separated list of rules. The first rule that fires sends codegen to `codegen_hard`:
  - `category`: the technique is structural. A v2 technique's `params.tier` takes precedence over the name patterns.
  - `stall:N`: the slot has had N or more attempts since its last best.
  - `share:X`: the slot is at least a fraction X of the partition time.
  - `explore`: the step tests a v2 hypothesis.
  - `fix-escalate`: the last fix attempt uses `codegen_hard`.
  - `always` / `never`.
- **What the run records.** Each step's `context` in `result.json` records `model`, `model_role` and `model_why` (plus `fix_models`). `llm_usage_by_model` breaks calls, tokens and estimated USD down by role and model. The estimates come from `cost_tracker`'s price table, which `COST_PRICE_*` overrides.
- **Example.** `--model us.anthropic.claude-haiku-5-5 --model-hard us.anthropic.claude-sonnet-5-5 --hetero-policy category,stall:3,fix-escalate` on the ResNet block (sm_75, 2×5 steps) gave 25× in 12 calls. The 4 Sonnet calls cost $0.47; the 8 Haiku calls (6 codegen, 2 fix) cost $0.15.
- **Output length.** Long structural rewrites from the hard model can exceed Bedrock's default 8192-token output limit. A truncated reply has no code block and is counted as a compile error. Set `BEDROCK_MAX_TOKENS=16384` for the hard model.

## Telemetry

Every run records per-LLM-call tokens (input, cache, output), stop reasons / truncation, latency and cost by role and model, and a wall-clock breakdown of the agent loop. The breakdown covers prompt, LLM wait, compile queue vs nvcc, GPU-lease wait, driver phases, NCU, knowledge and I/O.

- **Where it is written.** Per step in `result.json` (`steps[].context.telemetry`), per problem (`telemetry`), per run (`summary.json`), and as events in `OUT/telemetry.jsonl`.
- **Reports.** `scripts/telemetry_report.py OUT` prints the tables.
- **Switching it off.** `KB_TELEMETRY=0`.

See [telemetry.md](telemetry.md).

## LLM relay

To make the provider calls on another machine (VPN, keys kept there), run `kernelblaster.servers.llm_server` on
that machine and set `KB_LLM_SERVER_URL` (plus `KB_LLM_SERVER_TOKEN_FILE`) here. Typically this goes through an
`ssh -L` tunnel. `run_multislot.py` needs no flags, and accounting and telemetry are unchanged. Provider settings
such as `BEDROCK_MAX_TOKENS` are then read on the server. See [llm_server.md](llm_server.md).

## Notes

- `KERNELBLASTER_ALLOW_SHARED_GPU=1` lets the exec server start when other processes use the
  GPU (single-GPU dev boxes); timings are noisier, which is why wins are re-measured.
- `GPUType.TITAN_RTX` (sm_75) was added for the dev box.
