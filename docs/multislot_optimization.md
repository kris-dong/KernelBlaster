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
| `src/kernelblaster/knowledge/slots.py` | `SlotProblem` (manifest, assembly, neighbours, fusion-candidate pairs), `parse_driver_output`, `LocalEvaluator` (nvcc), `ServerEvaluator` (compile + exec servers) |
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

## Notes

- `KERNELBLASTER_ALLOW_SHARED_GPU=1` lets the exec server start when other processes use the
  GPU (single-GPU dev boxes); timings are noisier, which is why wins are re-measured.
- `GPUType.TITAN_RTX` (sm_75) was added for the dev box.
