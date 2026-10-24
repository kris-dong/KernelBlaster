# Fast compile / exec / profile paths

Per-candidate cost of KernelBlaster's evaluation loop, for the native single-file CUDA flow
(`agents/opt_ncu_rl_optimized.py`) and the multi-slot ModelBlaster flow
(`knowledge/slots.py`, `agents/opt_multislot.py`), and what was done about it. Every
optimisation keeps the program that gets built and the numbers the agents read the same, and
each one falls back to the old path when it cannot apply.

## Measuring

```bash
KERNELBLASTER_ALLOW_SHARED_GPU=1 python scripts/bench_fast_paths.py \
    --native data/kernelbench-cuda/level1/019_ReLU data/kernelbench-cuda/level2/009_Matmul_Subtract_Multiply_ReLU \
    --multislot tests/fixtures/mb_lenet OUT/mb_9_ResNet18 OUT/mb_30_SwinTransformerV2 \
    --reps 3 --out bench.json            # KERNELBLASTER_FAST_COMPILE=0 for the CMake path
```

Each rep is a new candidate (the kernel, or one slot in round robin, gets a unique comment),
so caches only hit what a real RL step can hit. `same` rows evaluate the previous candidate
again, which is what the multi-slot confirm run and NCU build do. Native rows run the agent's
own `_gather_perf_metrics` against the compile and exec servers. The phases come from its
`perf_span` records, and the compile breakdown comes from the compile server's
`[stage] finished ... elapsed=` log lines. The multi-slot problems came from
`python -m modelblaster.pipeline.kb_export IR OUT` on
`gpu_eval/level3/ir16/{9_ResNet18,30_SwinTransformerV2}`.

## Before / after

TITAN RTX (sm_75), shared with another long-running GPU job, so run times are noisy and
compile times are the reliable signal. Medians in seconds, N=3 (native after: N=4). This box
has no profiler-counter permission, so the native NCU details pass fails after about 1.5 s in
both columns.

| path | before | after |
|---|---|---|
| native L1 `019_ReLU`, one candidate end to end | **23.2** | **8.6** |
| &nbsp;&nbsp;compile + verify run (`compile_run`) | 15.0 (configure 3.9 + make 10.5) | 3.2 (kernel 1.3 + link 0.4, driver cached) |
| &nbsp;&nbsp;kernel-name listing (`ncu -k <invalid>`) | 1.3 | 0 (from the nsys trace) |
| &nbsp;&nbsp;nsys timing pass | 5.2 | 3.5 |
| native L2 `009_Matmul_Subtract_Multiply_ReLU`, end to end | **23.6** | **9.4** |
| multi-slot LeNet (4 slots), server, new candidate | **9.45** (compile 8.7) | **1.82** (compile 1.35) |
| &nbsp;&nbsp;same candidate again (confirm / NCU build) | 10.4 | 0.63 (binary cached) |
| multi-slot LeNet, local nvcc, new / same | 1.84 / 1.61 | 1.40 / 0.35 |
| multi-slot ResNet18 (7 slots, 68 ops, 110 MB linked data), server, new | **13.0** (compile 9.8, exec 3.2) | **4.16** (compile 1.6, exec 2.6) |
| &nbsp;&nbsp;same candidate again | 13.6 | 2.37 |
| multi-slot ResNet18, local nvcc, new / same | 4.51 / 4.70 | 3.46 / 2.28 |
| multi-slot SwinV2 (13 slots, 339 ops), server compile, new | 13.4 | 1.32 |
| multi-slot SwinV2, local build, new | 1.98 | 1.05 |

The SwinV2 driver run (22–52 s on this shared GPU, before and after) dominates that problem's
time and is not touched here; see *Remaining bottlenecks*.

The profiler passes were measured on an idle L40S (firesim2, sm_89) with counter access. These
are medians of 3 single runs, in seconds:

| run | `019_ReLU` | `009_Matmul...` | LeNet `MB_MODE=profile` | ResNet18 `MB_MODE=profile` |
|---|---|---|---|---|
| plain binary | 0.92 | 1.07 | 0.30 | 1.77 |
| `ncu -k <invalid>` (kernel listing, now skipped) | 2.23 | 2.59 | – | – |
| native NCU details (bottleneck kernel) | 3.04 | 3.40 | – | – |
| … with `-c 1` | 3.04 | 3.42 | – | – |
| nsys, default options | 7.20 | 7.51 | – | – |
| nsys, `--trace=cuda --sample=none --cpuctxsw=none --backtrace=none` | 4.54 | 4.71 | – | – |
| multi-slot NCU, every slot | – | – | 5.07 | 25.4 |
| … one slot (`relu`) | – | – | 2.96 | 6.93 |
| … two slots (`conv2d`, `relu`) | – | – | 3.59 | 13.2 |
| … every slot, `--cache-control none` | – | – | 7.71 | 21.1 |

For the native flow on the L40S, this works out to about 3 s for the listing and nsys changes
plus about 12 s for the compile, per candidate.

## What changed

### 1. Compile server: CMake configured once, objects cached (`servers/cuda_fast.py`)

The CMake path ran `cmake` and then `make` in a CMake project. With `persistent_artifacts`,
which the RL agents use, each job got a fresh project. So every candidate paid for a configure
and a full rebuild of the unchanged driver, about 10 s of LibTorch headers in the host TU.
`FastCudaBuilder` instead configures `cuda_env` once per SM version
(`toolchain_<sm>/`, no build). It reads the exact commands back from `compile_commands.json`
and `link.txt`, with the per-job paths templated, and runs them directly:

| product | cache key |
|---|---|
| driver object | stage, command, `main.cpp`, `cuda_model.cuh`, absolute inputs (`.incbin` data: path/size/mtime) |
| kernel object | stage, command, `cuda_model.cu`, `cuda_model.cuh` |
| binary | link command, the object keys |

The files are still split and written into the same work or persistent directory as before.
The kernel is compiled from that path, so `-lineinfo` source paths are as before.
`servers/utils/build_cache.py` holds the cache. It is content-addressed, entries are written
to a temp name and `os.replace`d, and an `asyncio.Lock` per key makes concurrent workers wait
for one build instead of duplicating it. It lives in `<artifacts>/cuda_env/build_cache`, so it
lasts as long as the server; `KERNELBLASTER_BUILD_CACHE_DIR` points it elsewhere. Products are
hard-linked out of the cache. The CMake fallback never writes through such a link: it removes
`build_<sm>/main` first and reconfigures any build dir that has no `Makefile`.

*Torch-free drivers*, such as ModelBlaster's: if neither the driver nor the kernel mentions
`torch/ATen/c10/pybind11/Python.h` or `torch::/at::/c10::`, the host TU is built without the
`torch/torch.h` include that the legacy split forces into it. The binary is linked against the
CUDA runtime only (static cudart, the same as `make`). If such a driver fails to compile this
way, for example because it relied on torch.h's transitive includes, that driver is built with
LibTorch from then on. Kernel and link errors are the candidate's own and are reported as is.

*Falls back to CMake* when the toolchain cannot be derived (`ToolchainError`).
`KERNELBLASTER_FAST_COMPILE=0` turns the fast path off.

### 2. Multi-slot: one TU per slot (`knowledge/slots.py`, `/compile_units`)

`SlotProblem.units(sources)` produces the same program as `assemble()`, split into one TU per
slot plus the walker tail:

- each slot TU is `walker_head.cu` plus `wrap_slot(...)`;
- the tail TU is the head, plus `namespace mbslot_<id> { <manifest signature>; }` for every
  entry, plus `walker_tail.cu`.

Kernels keep their `mbslot_<id>::` names, so NCU attribution is unchanged. `LocalEvaluator`
compiles each unit and the driver to objects keyed by (flags, text, absolute inputs). It
caches the linked binary too, using the same nvcc flags with compile and link done as separate
steps. `ServerEvaluator` writes the units under content-addressed names and posts them to the
compile server's `POST /compile_units` (`FastCudaBuilder.build_units`, which uses the
CMake-derived nvcc command). A new candidate therefore recompiles one slot. The confirm run and
the NCU build of an unchanged candidate only copy the cached binary.

*Falls back to the assembled TU* if any unit or the link fails. That covers a slot whose entry
no longer matches its manifest signature, a slot that relies on an include another slot
hoisted, and duplicate host globals. The assembled build is also the one that reports compile
errors. A server without the endpoint (404/405) turns units off for that evaluator, and
`KERNELBLASTER_SLOT_UNITS=0` turns them off everywhere. `assemble()` and ModelBlaster's
`kb_export.assemble` are unchanged and still equivalent.

### 3. Exec by path (`exec_server.py`, `strategies/local_exec.py`, `agents/utils/commands.py`)

Each run used to upload the binary, and the server wrote it to a temp file with `fsync` before
running it. That cost about 0.7 s for ResNet18's 115 MB binary on every run, nsys and NCU runs
included. `/gpu/binary` now also accepts `binary_path`: the local strategy runs the binary in
place, and other strategies read it on the server. Clients send the path only when the exec
server is `127.0.0.1` or `localhost`. They fall back to the upload, with its historical retry
and error semantics, if the server rejects the path or the request fails.
`KERNELBLASTER_EXEC_BY_PATH=0` turns this off.

### 4. Native profiling (`agents/opt_ncu_rl_optimized.py`, `agents/utils/parsing.py`)

- The launched kernels come from the nsys kernel table, which the reward pass reads anyway.
  `find_kernel_names_trace` takes their base names (what NCU lists under
  "Available Kernels") and intersects them with the source's kernels, so the extra
  `ncu -k <invalid>` run is gone. It remains the fallback when the trace has no rows. The
  failure messages and their order are the same as before.
- nsys traces CUDA only and turns off CPU sampling, context-switch and backtrace collection.
  The query reads only `CUPTI_ACTIVITY_KIND_KERNEL`, and the solution-kernel span measured
  1568–1600 ns with either option set. `KERNELBLASTER_NSYS_ARGS` overrides these options.

### 5. Multi-slot NCU: only uncached slots (`agents/opt_multislot.py`, `knowledge/ncu.py`)

The agent already caches NCU summaries per (slot, code). `_profile` now filters ncu to the
slots whose current code has no cached profile, using
`ncu_command(slots=[...])` → `-k regex:'mbslot_(a|b)::'`, and returns the cached summaries for
the rest. The cache contents and the prompts are unchanged. The baseline still profiles every
slot, and each later best only profiles its own slot: 25 s becomes 7–13 s on ResNet18.
`Evaluator.profile` gained an optional `slots` argument.

## Design decisions

| decision | alternatives | why |
|---|---|---|
| Derive the commands from a once-configured CMake project | hand-written nvcc/g++ command lines; keep CMake with a shared configured build dir | gives the same flags, includes and libs as `make`, including whatever LibTorch's CMake config adds, without maintaining a copy; a shared CMake dir still rebuilds the driver and serialises jobs |
| Cache objects by content, not by worker or job directory | ccache; per-worker incremental `make` | needs no extra tool; correct when concurrent workers compile the same driver; `make` mtime logic is defeated by rewriting the sources for every job |
| Torch-free flavour only on a lexical "no torch" check, with a driver-only retry | always link LibTorch; try plain for every driver | linking LibTorch into a binary that does not use it costs link and load time on every run; retrying on kernel errors would double the cost of every failing candidate |
| Per-slot TUs with manifest-signature declarations, falling back to the assembled TU | `-rdc` device linking; caching the assembled TU only | the manifest signature is the slot ABI; whole-program objects keep codegen identical; fallback keeps every candidate the old path accepted |
| Assembled TU stays authoritative for errors | report the failing unit's errors | the error text and line numbers the LLM sees stay the same; the cost is one extra slot compile (~0.7–1.2 s) on failing candidates |
| Exec by path only for localhost servers | shared-filesystem flag; always upload | a remote exec server cannot see the compile host's paths; localhost is the deployment the runners use |
| Kernel names from the nsys trace, with NCU listing as fallback | cache names per source hash; keep the listing run | the trace already holds the answer, so no second run is needed; the fallback covers traces without rows |
| Lean nsys options by default | keep the defaults | the CUDA kernel table is all the agent reads, and the span measures the same |
| Not adopted: `ncu -c 1`, `--cache-control none`, batching exec requests | – | `-c 1` saves nothing (each candidate launches the bottleneck kernel once); `--cache-control none` changes the cache-dependent metrics and was not faster on ResNet18; after exec by path the per-request overhead is about 0.1 s, so batching has little left to save |

## Load balancing

Builds use the CPUs that are actually free on a shared host, and timed runs spread over every
idle GPU, local or on remote hosts. The code is in `servers/utils/resources.py`.

### What is detected

| quantity | how |
|---|---|
| usable CPUs | `sched_getaffinity`, capped by the cgroup quota (v2 `cpu.max` on every ancestor of the process's cgroup, or v1 `cpu.cfs_quota_us`) |
| free CPUs | usable CPUs minus other users' load: the 1-minute load average minus the compiler processes this process is running itself |
| memory headroom | `MemAvailable`, minus 2 GB, divided by a per-process estimate (1.5 GB for nvcc, 2.5 GB for a LibTorch host TU; `KERNELBLASTER_NVCC_MEM_MB` overrides) |
| GPUs | `nvidia-smi` locally or over ssh: index, UUID, name, memory, utilisation, and the compute processes that are not ours |

An `AdaptiveLimiter` caps concurrent compiler processes at `free CPUs × 0.75`, bounded by
the memory headroom, in `[1, 16]`. It re-evaluates the cap every 5 s, both on acquire and
for waiters. When the box gets busy it stops admitting new jobs; running jobs are never
interrupted. When the box goes idle it scales back up. `KERNELBLASTER_CPU_SHARE` changes the
0.75 and `KERNELBLASTER_MAX_BUILD_JOBS` the 16.

GPU selection (`select_gpus`):

- `auto` takes the idle GPUs: no foreign compute process and utilisation ≤ 10 %
  (`KERNELBLASTER_GPU_IDLE_UTIL`).
- `KERNELBLASTER_ALLOW_SHARED_GPU=1` takes every GPU.
- When no GPU is idle, auto takes the least busy one, which matches the old one-shared-GPU
  behaviour, and logs that the GPU is shared.
- An explicit list (nvidia-smi indices or UUID prefixes) or `all` is always used as given.
- Locally, an externally set `CUDA_VISIBLE_DEVICES` bounds what auto and `all` may pick.

Runs get `CUDA_VISIBLE_DEVICES=<UUID>`, so CUDA's device ordering cannot remap them. Other
processes are never signalled or touched.

The `GPUPool` hands out exclusive (host, GPU) leases, so each device has at most one timed run
or NCU pass at a time:

- It picks the free device used least recently.
- After 3 consecutive infrastructure failures (no CUDA device, ssh or copy failure; not
  compile errors, wrong output or crashes), a device sits out for 5 minutes. The evaluator
  retries that run on another device or host, so the candidate is not charged for it.
- Every run also holds an `flock` on `/dev/shm/kernelblaster_gpu_<uuid>.lock` on the host
  that runs it (`KERNELBLASTER_GPU_LOCK_DIR`; `KERNELBLASTER_GPU_FLOCK=0` turns it off). Two
  KernelBlaster processes that both see a GPU as "idle" between each other's short runs
  therefore still take turns on it. Processes running code from before this change do not
  take the lock.

### Defaults and overrides

| component | default now | before | override |
|---|---|---|---|
| `LocalEvaluator` nvcc processes | adaptive | 4 unit jobs + 2 assembled builds | `build_jobs=` / `max_parallel=`, `--compile-jobs N`, `KERNELBLASTER_MULTISLOT_COMPILE_JOBS` |
| `LocalEvaluator` GPUs | pool over the auto GPUs | one `asyncio.Lock`, `CUDA_VISIBLE_DEVICES` untouched | `gpus=`, `--gpus auto\|all\|0,2`, `pool=` |
| `SSHEvaluator` | every `--ssh-host`, each with its auto (idle) GPUs | one host, one fixed `CUDA_VISIBLE_DEVICES` string | `--cuda-visible-devices` once for all hosts or once per host; `{host: spec}` / `{host: dir}` in the API |
| `run_multislot.py` problems | `--parallel-problems` = GPUs in the pool | sequential | `--parallel-problems N` |
| exec server (local strategy) | one worker per auto GPU (UUIDs); the pre-flight check is scoped to them | one worker on GPU 0, global check | `KERNELBLASTER_GPU_SERVER_GPU_IDS`, `KERNELBLASTER_GPU_SERVER_NUM_WORKERS` |
| compile server workers | `--num-workers auto`: `min(16, CPUs/2, MemAvailable/3 GB)` workers; adaptive admission of `free CPUs × 0.75 / 2` jobs | physical cores − 1 (`management.py`), 4 (CLI) | `--num-workers N`, `KERNELBLASTER_COMPILE_SERVER_NUM_WORKERS`, `KERNELBLASTER_MAX_COMPILE_WORKERS` |
| `FastCudaBuilder` processes | adaptive | 8 | `KERNELBLASTER_FAST_COMPILE_JOBS` |

`SSHEvaluator` transport:

- It picks the host with the most free GPUs (minus runs already on their way there) and
  copies the binary there *before* leasing a GPU, so no GPU waits on a copy.
- ssh connections are multiplexed (`ControlMaster`; `KERNELBLASTER_SSH_MUX=0` turns it off).
- When both ends have rsync, the binary goes as a delta against the problem's previous binary
  on that host, kept as `.kb_basis_<problem>`; `close()` removes it. Candidates differ in one
  slot while the linked `data.bin` dominates the size, so a 117 MB ResNet binary matches
  completely and sends almost no literal data. `KERNELBLASTER_SSH_RSYNC=0` falls back to scp.
- Hosts that are unreachable at start-up are skipped. The remote directory is created.

Several problems in one event loop (`--parallel-problems`):

- The implementation store, outcome log, claims and knowledge-v2 writes are synchronous, so
  they cannot interleave.
- `SemanticUpdater` and the knowledge-v2 upkeep already run one pass at a time and skip
  overlapping calls. A problem's final *forced* reflection now waits for a running pass
  instead of skipping it.
- After the last problem, the runner runs one more forced reflection or upkeep pass for any
  records that came in while another pass ran.
- Each problem keeps its own `OUT/<problem>/` and its own bandit and per-problem lock.
  Duplicate problem names are rejected.

### Timing comparability

Within a problem, every speedup compares two runs: before against after, and the confirm
re-measurement against the first run. These may now land on different GPUs. **Decision:** the
pool pins each problem (the lease group is the problem directory) to the GPU *model* of its
first lease, which is the baseline run. All later runs of that problem, including confirm and
NCU runs, go only to GPUs of that model, on any host.

- Same-model GPUs are treated as interchangeable. Their run-to-run spread is the same order
  as the noise that the confirm re-measurement already guards against.
- With a mixed pool, problems still spread over all models, but no problem's numbers mix
  models.
- The store keys records by arch label, so a mixed pool should only mix GPUs that share
  `--arch`. `SSHEvaluator` builds one binary for one SM.

Alternatives considered: requiring a homogeneous pool per evaluator, which wastes the other
GPUs; and pinning a problem to a single device, which serialises the problem's rollouts on
one GPU and gives no load balancing.

### Measured

`scripts/bench_loadbalance.py` runs an LLM-free scripted candidate stream. Each candidate
changes one slot of LeNet, `p_1_Conv2D_ReLU_BiasAdd` or `prob_resnet` (round robin), so it
compiles one slot TU, links and runs, as an RL step does. `--concurrency` candidates are in
flight at once, which corresponds to rollouts × parallel problems. "before" is `feat/multislot`
@eee08f1 driven through the same script with run_multislot's old defaults (`--compile-jobs 3`,
one ssh host, a fixed `CUDA_VISIBLE_DEVICES`).

Candidates per minute; medians, with the range of N runs of 30 candidates each.

*Remote: firesim2, 4× L40S.* GPUs 0 and 1 had other users' resident processes, so only 2
and 3 were idle. Builds ran on this box (sm_89) and runs went over ssh.

| setup | concurrency | GPUs | cands/min | N |
|---|---|---|---|---|
| before: `SSHEvaluator`, scp per run with the GPU held during the copy | 3 | 1 | 16.1 (15.9–16.2) | 2 |
| after, 1 GPU | 3 | 1 | 31.4 (30.7–32.0) | 2 |
| after, 1 GPU, scp instead of rsync | 3 | 1 | 33.0 (32.3–33.7) | 2 |
| after, 1 GPU | 6 | 1 | 35.5 (34.3–36.8) | 2 |
| after, 2 GPUs (`--cuda-visible-devices 2,3`) | 6 | 2 | **58.5** (51.9–65.1) | 2 |

- Before → after on one GPU is 2×. It comes from copying before leasing the GPU and from
  ssh multiplexing, not from rsync: at 3 in flight the link keeps up with scp as well.
- 1 → 2 GPUs at the same concurrency is 1.65×, with runs split 15/18 over the two GPUs.
- Four GPUs could not be measured: GPUs 0 and 1 stayed occupied, and auto correctly left
  them out.

*Exec server on firesim2.* `exec_server` ran there and the compile server here, with
`ServerEvaluator` going through an ssh tunnel. The problems were LeNet and
`p_1_Conv2D_ReLU_BiasAdd`, whose binaries are uploaded per run.

| exec server GPUs | concurrency 3 | concurrency 6 |
|---|---|---|
| `KERNELBLASTER_GPU_SERVER_GPU_IDS=<GPU 2>` | 80.7 | 104.7 |
| default (auto picked GPUs 2 and 3; the old default would have been GPU 0, which is busy) | 82.4 | **132.2** (1.26×) |

*Local: this box, 48 CPUs (load 17–26 from other users), 1 TITAN RTX.* One GPU means one
run at a time either way.

| setup | GPU state | concurrency 3 | concurrency 6 |
|---|---|---|---|
| before (`--compile-jobs 3`) | idle | 56.5 (51.8–68.0), N=3 | 74.5 (70.4–78.7), N=2 |
| after (adaptive, peak 6 nvcc) | idle | **84.3** (83.7–85.8), N=3 | 78.9 (72.5–85.4), N=2 |
| before | shared with a 100 %-utilisation job | 43.0 (42.6–43.5), N=2 | 50.6 (36.0–65.1), N=2 |
| after | shared with a 100 %-utilisation job | 34.6 (31.5–37.7), N=2 | 51.9 (39.7–64.2), N=2 |

- With the GPU to itself, the adaptive build limit gives +50 % at 3 in flight.
- At 6 in flight both versions are GPU-bound.
- With another user's job saturating the GPU, run times roughly double and vary run to run.
  The after-c3 rows came out lower, but per-phase probes showed the same nvcc and run
  times, and the spread is as large as the gap, so this is noise from the shared GPU.

## Remaining bottlenecks

- **Multi-slot driver runtime** on whole-model partitions (addressed; ModelBlaster
  `docs/gpu_harness_perf.md`): SwinV2 took 21–26 s per run on an L40S with naive seed kernels,
  17 s of it in the isolated-op checks (every weight re-uploaded per op). The new driver takes
  3.8 s for a full run. An agent step now evaluates with `MB_ONLY_SLOTS=<slot>` and
  `MB_SCREEN=1` (`knowledge/slots.py` `step_eval_options`), 1.2–1.7 s; confirm and final runs
  stay full.
- **nvcc fixed cost** of about 0.7 s per TU with plain flags, or 1.2–1.4 s with the
  CMake-derived flags (LibTorch include paths, `-Xcudafe` options). This is now the floor of a
  native or multi-slot compile.
- **First driver build per problem and server** is still about 10 s (LibTorch headers).
  `KERNELBLASTER_BUILD_CACHE_DIR` on a persistent path removes it across server restarts, but
  only while the toolchain is unchanged, because the keys hold commands, not compiler
  versions. A precompiled `torch/torch.h` header is the next step.
- **Native verify run** (about 0.9 s) is separate from the nsys run. The nsys run could also
  check `passed`.
- **Native NCU details** take about 3 s on an L40S per candidate. They are already limited to
  one kernel and cached by code hash.
- Failing multi-slot candidates pay one slot compile before the assembled build.

## Tests

`tests/test_fast_paths.py` covers:

- cache build-once under concurrency, no caching of failed builds, and `.incbin` data
  invalidation;
- toolchain templating, torch detection, and the torch-free split;
- fast and CMake builds giving identical driver output for a LibTorch driver (`019_ReLU`) and
  a torch-free one (LeNet). The second candidate rebuilds only the kernel and the link, and the
  plain binary does not link LibTorch;
- units against the assembled TU, invalidation on a slot change and a data change, fallback,
  and concurrent incremental builds;
- nsys kernel names, the NCU slot filter and the skipping of cached slots, and exec by path.

The build tests need nvcc and a GPU (CMake and LibTorch for the comparison test).

`tests/test_loadbalance.py` covers the load balancing without multiple GPUs:

- cgroup v1/v2 quota parsing, and the free-CPU, memory and job-count math;
- the adaptive limiter scaling up and down, including waiters re-probing without a release;
- nvidia-smi inventory parsing and GPU selection (auto, shared, explicit, `CUDA_VISIBLE_DEVICES`);
- the pool on fake devices: at most one run per device, LRU spread, GPU-model pinning,
  quarantine, waiting while every device is quarantined, host picking with reservations, and
  reuse across event loops;
- the cross-process `flock`;
- multi-host `SSHEvaluator` dispatch through a fake ssh/scp/rsync: every idle GPU is used, no
  device ever runs two binaries at once, retries skip a host that went down, unreachable hosts
  are skipped, and the scp fallback works;
- the exec-server GPU ids, compile-server workers and `FastCudaBuilder` jobs;
- a forced reflection waiting for a running pass;
- with a GPU: concurrent `LocalEvaluator` runs through the pool, and `run_multislot.py
  --parallel-problems 2` on two LeNet copies.
