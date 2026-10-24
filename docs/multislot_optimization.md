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
and "already tried" notes → assemble, build, run the driver (partition + isolated-op checks,
per-slot times) with a bounded fix loop → reward = slot time saved / partition time → outcome
record, store record, claim tally. Parallel rollouts share a per-slot composite best that is
re-verified at the end. After the run, `propose_fusions` and `propose_splits` write
`kb-proposal/1` for ModelBlaster.

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
transfer and driver-output parsing without a GPU.

## NCU

`--ncu best|all` profiles slots with NCU (`knowledge/ncu.py`): one run over `regex:mbslot_`
with the driver in `MB_MODE=profile`, rows attributed by the demangled namespace. The measured
bound and metrics replace the roofline state in prompts and the bandit key; outcome records get
`metrics_delta`. Needs counter access; otherwise NCU is disabled after the first failure.
`--evaluator ssh --ssh-host user@host --remote-dir DIR --cuda-visible-devices N` builds here and
runs the self-contained binary on a remote GPU (e.g. an L40S with counter access).

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

## Notes

- `KERNELBLASTER_ALLOW_SHARED_GPU=1` lets the exec server start when other processes use the
  GPU (single-GPU dev boxes); timings are noisier, which is why wins are re-measured.
- `GPUType.TITAN_RTX` (sm_75) was added for the dev box.
