# Trace store: replaying and auditing a multi-slot run

`telemetry.jsonl` (docs/telemetry.md) says *where the time and tokens went*. The trace
store says *what happened*: every prompt, every raw response, every candidate source
(winning and losing), each fix attempt, the build commands with their compiler output,
the raw driver output, the raw NCU CSV, and each step's decision. With it you can replay
or audit any step without rerunning anything.

`scripts/run_multislot.py` writes it to `OUT/trace/` by default (`--trace full`). The
ModelBlaster graph optimiser (`pipeline/kb_graph_opt.py`) writes its own steps to
`WORK/trace/` and passes `--trace` on to each KernelBlaster run.

**Traces contain prompts and generated code.** Treat a trace directory like the source
tree. Secrets are scrubbed before anything is stored (see [Secrets](#secrets)).

## Levels

| `--trace` / `KB_TRACE` | stores |
|---|---|
| `full` (default for run_multislot and kb_graph_opt) | records plus texts (gzip, content-addressed, deduplicated) |
| `meta` | records with the hash and size of each text; nothing goes to `blobs/` |
| `off` (default for library use) | nothing; the directory is not created |

Library code, such as tests or legacy agents, records only after
`trace.configure(root, level=...)`. `KB_TRACE_DIR` overrides the root.

### Size and overhead

LeNet fixture, scripted LLM, 2 rollouts × 4 steps:

- 70 records.
- 33 blobs: 27 KB stored for 162 KB of text, with 81 dedup hits.
- 0.04 s spent in the store, out of a 15 s run.

Prompts share most of their text: the system prompt, the reference, and the unchanged
slot sources are each stored once.

A live Haiku run is about 1–2 MB per problem with 10–20 steps. The size is dominated by
the user prompts and by the driver output of every evaluation.

Caps:

- `KB_TRACE_BLOB_MAX` (default 4 MiB per text): keeps the head (2/3) and tail (1/3),
  with a `...[trace: N bytes cut]...` marker in between.
- `KB_TRACE_MAX_MB` (default 2048, compressed): when this is reached, the store logs one
  warning and keeps only hashes and sizes from then on (`"dropped": true` in the ref).

## Layout

```
OUT/trace/index.jsonl            one JSON record per line (append-only)
OUT/trace/blobs/ab/<sha256>.gz   stored texts, gzip, named by the sha256 of the stored bytes
```

A text field holds a *ref*, which takes one of four forms:

| ref | when |
|---|---|
| `{"sha", "n"[, "trunc"]}` | a blob (`n` = stored bytes, `trunc` = bytes cut) |
| `{"text"}` | a text of 200 bytes or less, stored inline |
| `{"sha", "n", "meta": true}` | at level `meta` |
| `{"sha", "n", "dropped": true}` | over the total cap |

Blobs are plain gzip files, so `zcat` works on them.

## Records

Every record has:

- `t`: the kind.
- `id`: a stable id.
- `ts` and `pid`.
- The context fields: `run`, `problem`, `rollout`, `step`, `slot`, `technique`,
  `attempt`, `eval`, `eval_occ`, `purpose`.
- `tel_span`: the id of the telemetry span the record was made in, which links it to
  `telemetry.jsonl`.

Ids are built from `<run_id>/<problem>/r<R>/s<S>/...`:

- `run_id` is the agent's run id, which is kept across a resume.
- `attempt` is 0 for the codegen candidate and n for fix n.

A record written again under the same id, for example when a step is replayed after a
kill, *replaces* the earlier one in the viewer.

| `t` | written by | holds |
|---|---|---|
| `run`, `close` | `configure`, `close` | run id, scrubbed argv, level, caps; at close, the store's stats (records, blobs, bytes in / stored, dedup hits, seconds) |
| `llm` | `query.generate_code` (all roles: codegen, fix, reflect, upkeep, legacy agents; direct or relayed) | `request_id`, the same id the LLM relay server logs; `role`, `model`, `n`; `messages` (role + ref each, so each message is deduplicated on its own); `prompt_sha`; `response` (raw text, always kept, also when it has no code block); `usage` (tokens, cache, `stop_reason`, `max_tokens`, `truncated`, cost); relay timings; `ok` / `error`; `call` (1, 2, ... for retries of the same id) |
| `llm_replay` | `opt_multislot._llm` on resume | a response taken from the resume journal instead of a new call: journal key, `replay_of` (the original `llm` record id) |
| `step_input` | `opt_multislot` step | slot, technique, state, and the slot's source the step starts from |
| `candidate` | `extract_code` | per attempt: the extracted `code` (ref, sha, lines) or `no_code` with the reason; `request_id` of the response it came from |
| `eval` | every evaluator's `evaluate` / `profile` | `tag`, `purpose` (baseline / seed / candidate / fix / confirm / final / profile), `attempt`, `evaluator`, options (`only_slots`, `screen`, incumbent hashes); `sources` (ref per slot) and `composed_sha` (the assembled TU); `result` (ok, stage, times, ratio, checks, errors, graph skip, GPU); `log`; `occ`, which is unique per call |
| `build` | `LocalEvaluator._nvcc` / `_build`, the compile server, OpenCL `g++` | `cmd` (scrubbed), `rc`, `output` (compiler stdout+stderr), seconds. Build-cache hits run no compiler, so they have no record |
| `driver` | `parse_driver_output` (process, persistent host, server, ssh) | the raw run output, with every `MB_*` line |
| `ncu` | `_ncu_result` | the raw `ncu --csv` output. NCU writes no `.ncu-rep`; the CSV is the whole capture |
| `step` | after each step | `decision` (`merged` / `not_merged` / `rejected`) and `why`; `status`, `reward`, `fix_attempts`; `steps_index` (index in `result.json` `steps[]`) and `result_json`; `outcome_id` (outcome log / knowledge-v2 row); `tel_span` (the telemetry `step` span); the full StepLog |
| `step_error` | a step that raised | the error and its traceback |

`build`, `driver` and `ncu` records carry `eval` (the id of their evaluation) and
`eval_occ` (the `occ` of the call they belong to). On a resumed run, children always
join the latest occurrence of their evaluation.

### ModelBlaster records (`WORK/trace`, `pipeline/kb_trace.py`)

| `t` | holds |
|---|---|
| `mb.profile` | an IR's `profile.json` (per-op times, CUPTI attribution, device) and manifest; its builds and driver runs are `build` / `driver` records with `mb_stage=profile` |
| `mb.plan` | the inputs (every profile, the options, how many items were excluded) and `plan.json` |
| `mb.export` | the exported problems and their manifests |
| `mb.kb_run` | the KernelBlaster command (scrubbed), exit code, log, `summary.json`, each problem's result, and `kb_trace` (the run's own store) |
| `mb.ingest` | `winners.json` and the cache report |
| `mb.hint` | a hint decision: the original / variant IR graphs and manifests, KernelBlaster's run dirs, re-measured times (each re-timing run's raw output is a `driver` record with `mb_stage=remeasure` and `hint`), gain, decision, why |
| `mb.kb_time` | a `time_problems.py` run: command, jobs, result, log |
| `mb.verify`, `mb.model_run` | final verification (result, log, graph, manifest) and the model's timing |

These records carry `mb_round` (from the telemetry `mb.round` span) and `mb_stage`.

## Viewer

```
python scripts/trace_view.py OUT --list                     # one row per step
python scripts/trace_view.py OUT --failed --diff            # steps that failed or needed fixes
python scripts/trace_view.py OUT --problem P --rollout 0 --step 2 [--diff] [--full]
python scripts/trace_view.py OUT --llm 1c73bca7             # one LLM call (request id prefix), in full
python scripts/trace_view.py OUT --export DIR               # plain files per step
python scripts/trace_view.py OUT --html trace.html          # one self-contained HTML file
python scripts/trace_view.py MB_WORK --mb                   # ModelBlaster's records
python scripts/trace_view.py MB_WORK --list                 # every KernelBlaster run below the work dir
```

A step's story contains:

- The slot and technique.
- Per attempt:
  - The LLM call: model, request id, tokens, stop reason.
  - The prompt. It is abridged unless you pass `--full`.
  - The raw response.
  - The candidate source. With `--diff` it is shown against the source it replaced.
  - The evaluation result and the errors.
  - The build commands, with compiler output for the failing ones.
  - The driver `MB_*` lines.
- The confirm and profile runs.
- The decision, with its `result.json`, outcome and telemetry links.

`--export` writes these files under `DIR/<problem>/r<R>_s<S>/`:

- `00_input.cu`
- `a<N>_prompt.md`, `a<N>_response.md`, `a<N>_candidate.cu`, `a<N>_llm.json`
- `eval_<tag>.json`, with `eval_<tag>_build<k>.txt` and `eval_<tag>_driver<k>.txt`
- `step.json` and `story.txt`

The library is `kernelblaster.trace.view`, with `Trace`, `Trace.steps()`, `render_step`,
`export` and `export_html`.

## Secrets

Before any text is stored, `core.Scrubber` replaces:

- The values of environment variables whose names contain KEY, TOKEN, SECRET, PASSW,
  BEARER, CREDENTIAL, AUTH or SESSION. This covers `AWS_BEARER_TOKEN_BEDROCK`, API keys
  and `KB_LLM_SERVER_TOKEN`. The environment is re-read when it changes.
- The contents of `KB_LLM_SERVER_TOKEN_FILE`.
- Values in the `AKIA…`, `sk-…`, `ABSK…` and `Bearer …` formats.

Replaced values become `***`. This applies to prompts, responses, sources, command lines
and outputs.

Environments recorded with a run, such as driver variables, have the values of
secret-named keys redacted.

The scrubber cannot recognise a secret that is in none of these places. Do not paste
keys into prompts or sources.

## Resume

Trace ids come from the persisted run id, problem, rollout, step and attempt. As a
result:

- A step replayed after a kill re-records under the same ids. The viewer shows the
  latest copy and counts the duplicates in `Trace.dupes`.
- A response taken from the resume journal is not stored again. The step gets an
  `llm_replay` record that points to the original `llm` record, which stays in the index
  from the earlier segment.

The trace store is separate from the resume journal (`agents/journal.py`) and is never
read back by the agent.

## Hook points (for new code)

| hook | where |
|---|---|
| `trace.llm_recorded` | decorator on `query.generate_code` |
| `trace.instrument_evaluator(cls)` | `Evaluator.__init_subclass__`, before the telemetry instrumentation |
| `trace.tool(name, cmd, rc, output)` | compiler invocations |
| `trace.output(kind, text)` | raw run output |
| `trace.scope(**ctx)`, `trace.set_attrs(**ctx)` | context (a `contextvars` dict, so it is per asyncio task) |
| `trace.step_input`, `trace.attempt`, `trace.on_extract`, `trace.link`, `trace.step_done` | the multi-slot agent's step |
| `trace.record(kind, rid=..., **fields)`, `trace.put(text)` | anything else |

The package is stdlib only, so ModelBlaster loads it from the checkout.
