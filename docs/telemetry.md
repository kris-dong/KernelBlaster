# Telemetry: tokens and agent-loop time

Every multi-slot run records two things:

- what each LLM call cost: input, output and cache tokens, stop reason / truncation, latency, retries, USD;
- where the agent loop's wall clock goes: prompt building, waiting for the LLM, compile queue vs nvcc, GPU-lease
  wait, driver phases, NCU, knowledge, I/O.

The data lands in three places:

| where | what |
|---|---|
| `OUT/telemetry.jsonl` | one event per span and per LLM call (one file per `run_multislot.py` run) |
| `OUT/<problem>/result.json` | `telemetry`: the problem's breakdown, LLM totals, derived metrics; each step's `context.telemetry`: that step's breakdown |
| `OUT/summary.json` | a `{"telemetry": ...}` entry for the whole run; `telemetry` (wall, LLM totals, derived) on each problem's entry |

`scripts/telemetry_report.py OUT [OUT2 ...]` prints the tables. Arguments are files or directories, which it
searches for `telemetry.jsonl`; a ModelBlaster `kb_graph_opt` work dir works too. Flags: `--by-problem`, `--top N`,
`--json`.

`KB_TELEMETRY=0` turns all of it off. The spans become a shared no-op and nothing is written, and the run is
otherwise unchanged: `llm_usage`, `llm_usage_by_model` and `StepLog.seconds` are kept as before.
`KB_TELEMETRY_FILE` redirects the event file and `KB_TELEMETRY_RUN_ID` names the run.

The code is in `src/kernelblaster/telemetry/` (stdlib only):

- `core.py`: spans, collectors, the JSONL sink;
- `llm.py`: LLM-call records;
- `evaluator.py`: the evaluator boundary;
- `report.py`: categories, derived metrics, the CLI.

## Spans

```python
from kernelblaster import telemetry as tel

with tel.span("build", slot=sid):          # timed region; attributes are inherited by spans below
    ...
with tel.attrs(role="fix"):                # attributes only, nothing recorded
    await generate_code_retry(...)         # -> span llm.fix + one llm event
@tel.traced("reflect", role="reflect")     # decorator form (sync or async)
tel.lap("prompt.build")                    # sequential phases of a laps=True span, one line each
tel.count("win")                           # counter in every collector above
```

- **Concurrency.** The current span is a `contextvars` variable. Tasks started by `asyncio.gather` / `create_task`
  / `to_thread` inherit their parent span when they are created and never see their siblings' spans. Concurrent
  problems, rollouts and unit compiles therefore nest correctly without any locking in the agent. A worker thread
  from a plain `ThreadPoolExecutor` does not inherit the span; use `with tel.within(parent):`.
- **Collectors.** A span opened with `collect=True` aggregates everything that closes below it. The run
  (`kb.run`), each problem (`problem`) and each step (`step`) collect. A collector reports:
  - `time_s`: wall-clock attribution. Each instant of the collector's wall time goes to the innermost span(s) open
    at that instant, split evenly when tasks run concurrently, so it sums to `wall_s` exactly. The collector's own
    name gets the time no child covers, which is the untracked remainder.
  - `busy_s`: exclusive time per span name, summed over concurrent tasks ("agent-seconds").
  - `llm`: the LLM totals, and the same per (role, model).
  - `counters`, `evals` (evaluate() calls per purpose: count, seconds, failures) and `gpu_busy_s` (the union of
    GPU-holding spans per device).

## LLM calls

`query.generate_code`, the provider dispatcher, is wrapped once (`llm_traced`). That covers every caller: codegen
and fix in the agent, reflection, v2 upkeep (discovery, hypotheses) and the legacy agents. The role comes from the
enclosing attributes: `codegen_simple` / `codegen_hard` / `fix` from the agent's `_llm`, `reflect`, `upkeep`,
`discovery` and `hypothesis` from the knowledge passes, and `other` when none is set. One record per call:

| field | meaning |
|---|---|
| `input_tokens` | every prompt token, cached or not |
| `uncached_input_tokens`, `cache_read_input_tokens`, `cache_write_input_tokens` | where the provider reports them (Bedrock `cacheRead/WriteInputTokens`, Anthropic / Claude CLI `cache_*`, OpenAI `prompt_tokens_details.cached_tokens`) |
| `output_tokens` | |
| `stop_reason`, `truncated` | truncated = stop reason `max_tokens` / `length`, or, with no stop reason, output tokens at the request's `max_tokens`. Each truncation also logs a warning |
| `latency_s`, `server_latency_s`, `ttft_s` | time to the full response; the provider's own latency (Bedrock `metrics.latencyMs`, Claude CLI `duration_api_ms`); time to first token when the OpenAI path streams |
| `provider_retries`, `errors`, `backoff_s` | boto3's internal retries (`RetryAttempts`); failed attempts; `generate_code_retry`'s back-off sleeps (span `llm.backoff`) |
| `cost_usd`, `est_usd` | cost reported by the provider (Claude CLI); estimate from `cost_tracker`'s price table, with cache reads at 0.1× and cache writes at 1.25× the input price when the table has no cache row |
| `prompt_chars`, `output_chars`, context | sizes, plus problem / rollout / step / slot |
| `relay_queue_s`, `relay_network_s` | relayed calls only ([llm_server.md](llm_server.md)): waiting for a server slot; round trip minus server time. The server's throttling back-off is counted in `backoff_s` |

- **Where the extra fields come from.** Providers put them under `usage["telemetry"]` (a dict), so code that sums
  the numeric usage fields (reflection, v2 upkeep) is unaffected.
- **Scripting usage in tests.** A `stub:` model can return `{"text": ..., "usage": {...}}` to script token counts
  and stop reasons.
- **No double counting.** The agent's `llm_usage` and `llm_usage_by_model` keep their keys and meaning (provider
  `prompt_tokens` / `input_tokens`). Telemetry's per-role numbers are a separate structure. For the agent's own
  calls the two agree (`test_telemetry_e2e.py`). Telemetry additionally covers reflection and upkeep, and counts
  Bedrock cache tokens in `input_tokens`.

## Agent loop

| span | where |
|---|---|
| `kb.run` (collects, laps) | `run_multislot.py`, the whole run; laps `setup.knowledge` (stores, knowledge v2), `setup.problems`, `problems`, `finish` (catch-up reflection / upkeep, summary) |
| `problem` (collects) | `MultiSlotAgent.run` |
| `kb.time_problems` (collects) | `scripts/time_problems.py`, the whole call (`--telemetry FILE`); its summary is in the output's `telemetry` (ModelBlaster's `mb.kb_time` span wraps the call) |
| `init`, `proposals`, `io.result` | `initialize`, `propose_fusions` / `propose_splits`, `_write_result` |
| `step` (collects, laps) | `MultiSlotAgent._step`, the wrapper around `_step_body` |
| laps `knowledge.select`, `knowledge.lookup`, `prompt.build`, `candidate`, `knowledge.record` | one-line `tel.lap(...)` in `_step_body`: slot / technique choice; store retrieval + claims; prompt; extract → evaluate → fix loop → confirm; bandit / store / outcome / v2 record + NCU of a new best |
| `extract` | `extract_code` |
| `llm.<role>`, `llm.backoff` | `query.generate_code`, `generate_code_retry` |
| `reflect`, `upkeep`, `upkeep.discovery`, `upkeep.hypothesis` | `SemanticUpdater(V2).maybe_reflect`, `KnowledgeV2.maybe_upkeep` / `discover_states` / `generate_hypotheses` |
| `eval` | every evaluator's `evaluate` / `profile` (below) |
| `build.queue` | `AdaptiveLimiter.acquire`: waiting for a compile slot |
| `build`, `build.nvcc` | an evaluator's build; each compiler / linker process (`LocalEvaluator._nvcc`, `FastCudaBuilder._run`) |
| `gpu.queue` | `GPUPool.acquire`: waiting for a GPU lease |
| `run`, `run.exec` / `ncu.exec`, `run.stage` | timed run incl. infra retries; the process (under `profile()`: the NCU process); ssh upload |
| `driver.<phase>` / `ncu.driver.<phase>` | children of `run.exec` built from the driver's `MB_PHASE_MS` lines (`start`, `cuda_init`, `alloc`, `check_partition`, `check_ops`, `graph_build`, `time_partition`, `time_ops`), laid end to end at the end of the process. The rest of `run.exec` is process spawn, the cross-process GPU lock wait (`flock`) and teardown |
| `io.workdir` | creating work dirs |

Counters:

- `step` and `win` (a new slot best);
- `eval.<purpose>`;
- `build.cache_hit` / `build.cache_miss` (`BuildCache.get_or_build`, which covers the incremental multi-slot build
  and the compile server's fast path);
- `driver_ms.<phase>`.

### Evaluator boundary

`Evaluator.__init_subclass__` calls `telemetry.evaluator.instrument_class`. It wraps the methods in `STAGES` that
the class defines itself:

| methods | span |
|---|---|
| `evaluate`, `profile` | `eval` |
| `_build`, `_build_units`, `_compile`, `_compile_units` | `build` |
| `_nvcc` | `build.nvcc` |
| `_timed_run` | `run` |
| `_run`, `_exec_staged` | `run.exec` |
| `_stage`, `_send`, `_remote_data` | `run.stage` |
| `_workdir` | `io.workdir` |

- **Same-stage calls.** A call of the same stage nested in another (`super()._build`, `_build` → `_build_units`)
  produces one span.
- **Purpose.** The `eval` purpose comes from the agent's tag: `_f0` is a candidate, `_fN` a fix, then `_confirm`,
  `baseline`, `store_seeds` (as `seed`), `final*` and `profile`. An enclosing `tel.attrs(purpose=...)` wins.
- **New evaluators.** An evaluator added later (e.g. one talking to a persistent driver host) is covered at
  `evaluate` / `profile` with no glue. It gets the finer stages by using these method names or by declaring
  `TELEMETRY_STAGES = {"_method": "span name"}` on the class. If its output carries `MB_PHASE_MS` lines through a
  `run.exec` method, the driver phases follow.
- **Persistent driver host** (`knowledge/runtime_host.py`). `LocalEvaluator` / `SSHEvaluator` declare
  `_build_so` and `_host_exe` (`build`), `_host_request` (`run`: lease and infra retries) and `_stage_file`
  (`run.stage`). The host client records:
  - `driver.host_start`, with `restart=` and the counters `host.start` / `host.restart`;
  - each request as `run.exec` (GPU busy), with the driver's phases as children: `driver.load`, the checks,
    `driver.graph_build` (graph capture) and `driver.time_partition` (graph and eager replay), attribution;
  - the failures as `host.crash` / `host.timeout` / `host.poisoned`.

## Categories and derived metrics

`report.category` maps span names onto categories:

| category | what |
|---|---|
| `llm_wait` | waiting for LLM responses |
| `llm_backoff` | retry back-off sleeps |
| `build_queue` | waiting for a compile slot |
| `gpu_queue` | waiting for a GPU lease |
| `compile` | nvcc, link, build-cache lookups |
| `profile_ncu` | NCU runs |
| `checks` | driver correctness checks |
| `timing` | driver timing loops |
| `driver_startup` | process start, CUDA init, allocation |
| `process_overhead` | spawn, GPU lock wait, teardown |
| `transfer` | copies to a remote host |
| `setup` | run start-up: knowledge stores, loading problems |
| `run_other`, `eval_overhead`, `knowledge`, `io`, `prompt`, `extract`, `agent` | the rest of the loop |

`derived` (in `result.json`, `summary.json` and the report) gives:

- `steps_per_hour`, `candidates_per_min` (candidate + fix evaluations);
- `llm_wait_frac` of wall, `gpu_idle_frac` (1 − GPU-busy / (devices × wall)), `compile_cache_hit_rate`;
- `usd` and `usd_is_estimate`, `usd_per_win`, `tokens_per_win`, `truncation_rate`, `mean_llm_latency_s`;
- the time by category in seconds and in % of wall.

## Events (`telemetry.jsonl`)

```
{"t":"meta","ts":...,"pid":...,"run":...}
{"t":"span","id":12,"parent":9,"name":"build.nvcc","ts":<epoch s>,"dur":0.41,"self":0.41,"attrs":{"problem":..,"rollout":0,"step":3,"slot":"conv2d",...},"run":...}
{"t":"llm","span":15,"role":"fix","model":"...","input_tokens":..,"output_tokens":..,"stop_reason":"end_turn","truncated":0,"latency_s":..,"est_usd":..,"problem":..,"step":3,"slot":"conv2d","ts":..,"run":..}
{"t":"llm_backoff",...} {"t":"count","name":"win","n":1,...}
```

- **Writes.** Lines are buffered and flushed every 2 s, every 256 lines and at exit. A killed run loses at most
  that, and the report skips a torn last line.
- **Rebuilding a run.** `report.summarize_events` rebuilds a run's collector summary from its events, so the
  JSONL alone gives the same tables as `summary.json`.
- **Resumed runs.** Each segment of a resumed run appends to the same file as its own run (`<id>.s<n>`) after a
  `{"t":"resume",...}` event; the multi-slot agent flushes the sink with every checkpoint and every journaled LLM
  response. `step` counts completed steps (a step redone after a kill counts once) and responses reused from the
  journal count as `llm.replayed`, so the report's combined totals equal an uninterrupted run's (plus any call
  that was in flight at a kill). See [multislot_optimization.md](multislot_optimization.md#resume).

## Overhead

Measured on this box (Python 3.11, TITAN RTX host):

| operation | cost |
|---|---|
| one span, open + close, no event file | 2.1 µs |
| one span, writing its event | 7.3 µs |
| one span, `KB_TELEMETRY=0` | 0.2 µs |
| one LLM record (normalise, aggregate, write) | 39 µs |
| a collector summary of 20 000 spans | 35 ms |

- **Per step.** The scripted LeNet run (2×6 steps) wrote 33 spans per step plus counters and LLM records: about
  4 ms of telemetry in a 6–9 s run, i.e. 0.05 %.
- **End to end.** Three alternating runs each way gave 6.0 / 6.6 / 9.0 s with telemetry on and 6.8 / 9.1 / 8.2 s
  with `KB_TELEMETRY=0`, all within the noise of the shared GPU.

`test_span_overhead_is_small` guards the per-span cost.

## Live runs (Haiku 5.5 on Bedrock, local TITAN RTX sm_75, GPU shared with another user)

### Two problems in parallel

`run_multislot.py` on the ResNet basic block and SqueezeNet fire-module partitions:

- **Settings.** `--parallel-problems 2`, 2 rollouts × 4 steps each, `--knowledge v2 --upkeep-model`, a reflect
  model, `--ncu best`.
- **Result.** 23.7× and 6.3×, 16 steps, 30 LLM calls, est. $0.72, 300 s wall.

| category | s | % wall |
|---|---|---|
| llm_wait (codegen 117 s, fix 31 s, reflect 17 s, hypothesis 15 s) | 179.8 | 59.9 |
| compile (nvcc) | 60.7 | 20.2 |
| agent (run start-up: knowledge-v2 open + seeding, problem loading; now the `setup.*` laps) | 42.4 | 14.1 |
| process_overhead (spawn, GPU lock wait) | 7.6 | 2.5 |
| gpu_queue | 2.6 | 0.9 |
| profile_ncu | 2.0 | 0.7 |
| driver_startup (CUDA init) | 1.7 | 0.6 |
| io, eval_overhead, timing, checks, knowledge | 3.5 | 1.2 |

Derived metrics for this run:

- **Throughput.** 192 steps/h, 3.8 candidate evaluations/min, build-cache hit rate 74 %.
- **GPU.** Idle 92 % of wall.
- **Cost.** $0.10 and 35 k tokens per accepted win.
- **Prompt caching.** None: 0 cache-read tokens.
- **Truncation.** No output was truncated at `BEDROCK_MAX_TOKENS=16384`. Codegen replies average 5.7 k output
  tokens at 24 s mean latency.

### One graph-optimiser round

ModelBlaster `kb_graph_opt`, one round on the ResNet basic block: 8.7×, 9 LLM calls, est. $0.27. Of 202 s of
ModelBlaster wall:

| ModelBlaster span | s |
|---|---|
| `mb.kb_wait` | 164 |
| `mb.profile` | 21 |
| `mb.verify` | 16 |

Inside the 158 s KernelBlaster run: llm_wait 69 %, compile 25 %, everything else under 2 % each.

### Takeaways

- The loop is LLM-latency-bound first and nvcc-bound second.
- GPU work (timing, checks, NCU) is a few percent.
- Knowledge-v2 start-up is a fixed ~10–40 s per KernelBlaster process, which `kb_graph_opt` pays on every run.

## Tests

- **`tests/test_telemetry.py`:**
  - nesting, exclusive time, wall attribution summing to wall;
  - concurrent rollouts (asyncio + `to_thread`) keeping their own spans;
  - the no-op when disabled;
  - usage normalisation (OpenAI, Bedrock, Claude CLI) and truncation detection;
  - stub-model token aggregation per role / model and the JSONL events;
  - back-off and failed attempts;
  - Bedrock cache tokens / stop reason / server latency / boto3 retries (fake client);
  - evaluator boundary spans, purposes, driver phases and `super()` calls;
  - categories, the report CLI (several runs, a torn line), span overhead.
- **`tests/test_telemetry_e2e.py` (GPU):**
  - the agent on the LeNet fixture with a scripted LLM: every step's breakdown sums to the step's wall time and
    matches `StepLog.seconds`; result.json totals match `llm_usage` (no double counting); the JSONL rebuilds the
    same counts;
  - the same flow with telemetry disabled.
