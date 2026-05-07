# Adreno OpenCL Profiling Methodology

> Reference doc for cross-machine performance comparisons. Hand this to anyone running QNN / TFLite / ONNX-Runtime benchmarks against KernelBlaster's OpenCL/Adreno results.

---

## Headline

What we report as "kernel time" in our OpenCL/Adreno flow is **only the GPU dispatch interval measured by `cl_event` profiling counters** — not end-to-end op latency. Any framework-level profiling that includes graph dispatch, buffer marshalling, or sync setup will measure a fundamentally larger quantity, and a direct comparison is invalid for sub-millisecond ops.

For the existing QNN data set, **only problems where QNN time ≥ 200 ms AND our_best ≥ 10 ms produce defensible apples-to-apples ratios.** The rest reflect framework-overhead floors, not real GPU performance differences. See the [Outlier Filter](#suggested-outlier-filter-for-the-current-data-set) section at the bottom for specifics.

---

## What our profiler actually measures

For each kernel, the LLM-generated host driver (`driver.c`, compiled on-device with `gcc … -lOpenCL`) does this when invoked with `--profile`:

```c
// 1. SETUP — NOT TIMED
ctx, queue (with CL_QUEUE_PROFILING_ENABLE), program, kernel, buffers
clEnqueueWriteBuffer(...)   // host → device input copies
clBuildProgram(...)         // OpenCL JIT compile

// 2. WARMUP — NOT TIMED (one full dispatch + sync, results discarded)
clEnqueueNDRangeKernel(queue, kernel, ..., NULL);
clFinish(queue);

// 3. THE TIMED RUN
cl_event event;
clEnqueueNDRangeKernel(queue, kernel, ..., &event);
clFinish(queue);

clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, ...);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, ...);
double exec_ms = (end - start) / 1e6;        // <-- THIS is what we report
printf("[PROFILE] %s: %.3f ms\n", kernel_name, exec_ms);

// 4. TEARDOWN — NOT TIMED
clEnqueueReadBuffer(...)    // device → host output copy
```

Per the OpenCL spec, the `(end - start)` interval is the time the kernel is actually running on the GPU — it does NOT include queue submission, the host driver's dispatch path, or any framework overhead above the OpenCL host runtime.

## Inclusion / exclusion table

| Cost | Included? | Notes |
|---|---|---|
| Host buffer alloc (`clCreateBuffer`) | ❌ | one-time setup |
| Input H→D (`clEnqueueWriteBuffer`) | ❌ | data marshalling |
| `clBuildProgram` JIT | ❌ | one-time per kernel |
| `clEnqueueNDRangeKernel` queue submission latency | ❌ | event START is when kernel actually begins on GPU |
| **Kernel execution on GPU** | ✅ | THE measurement |
| Implicit warm-up (one prior dispatch) | ❌ | discarded |
| Output D→H (`clEnqueueReadBuffer`) | ❌ | result transfer |
| Process startup, OpenCL ICD load, context creation | ❌ | one-time |

## How the harness drives this

1. Harness sends `kernel.cl` + `driver.c` to the Adreno board via SCP.
2. Board-side: `gcc -o main driver.c -lOpenCL` then `./main --profile`.
3. Driver does the steps above and prints `[PROFILE] <kernel_name>: <ms> ms` to stdout.
4. Harness parses with `r"\[PROFILE\]\s+(\S+):\s+([0-9]+(?:\.[0-9]+)?)\s*ms"` (`opt_opencl_rl.py:46`).
5. If a problem has multiple `[PROFILE]` lines, the harness **sums them** (`get_total_kernel_time_ms`).

**Number of measurement runs:** one timed run per `--profile` invocation, after one warmup. The driver does not loop internally; the harness can re-invoke for repetition but each invocation is a fresh process.

---

## Why a naive QNN comparison breaks down

A typical QNN benchmark wraps each model forward pass like:

```python
qnn_runtime = qnn.create_session(model_path, backend="adreno")
for _ in range(N):
    t0 = time.perf_counter()
    out = qnn_runtime.run(inputs)        # ← what gets timed
    t1 = time.perf_counter()
```

`qnn_runtime.run(inputs)` includes:
- QNN graph scheduler & Op dispatch
- Tensor metadata processing (shape, layout, type validation)
- Internal buffer commit + DMA setup
- The actual GPU kernel
- Sync barrier
- Output unpacking

For a 128-element hinge loss, the kernel itself is ~6 µs but everything around it routinely runs **50–65 ms**. That's why every tiny pointwise op in QNN's reported numbers clusters at 53–66 ms regardless of arithmetic content — it's a constant per-call framework floor, not GPU time. Same explains the ~30 ms floor on tiny reductions.

## Three ways to fix the comparison on the QNN side

### Option A — QNN per-op profiling counters (best)

```cpp
QnnGraph_setConfig(graph, QNN_GRAPH_CONFIG_OPTION_PROFILE);
// run...
QnnProfile_getEvents(...);  // returns per-op GPU execution times
```

Use the per-op event GPU time, not the wall-clock `run()`. Apples-to-apples with our `cl_event` numbers.

### Option B — Loop and amortize

```python
WARMUP = 10
N = 1000
for _ in range(WARMUP): qnn_runtime.run(inputs)        # warmup, not timed
t0 = time.perf_counter()
for _ in range(N): qnn_runtime.run(inputs)
t1 = time.perf_counter()
per_call_ms = (t1 - t0) * 1000 / N
```

Per-call overhead amortizes. **Caveat**: if QNN holds inputs by reference (zero-copy), a tight loop may overstate cache locality. To match our methodology, also loop our binary internally over the same N.

### Option C — Symmetric `cl_event` instrumentation

Add `cl_event` profiling counters around the actual GPU dispatch inside QNN. Most mobile inference runtimes have hooks for this. Most defensible because both numbers come from identical hardware counters.

---

## Suggested outlier filter for the current data set

Looking at our measured QNN data (`level1_qnn_qualcomm_results.json`):

- Pure pointwise activations (#25–32): cluster at **53–66 ms** regardless of math content
- Single-axis reductions (#47–50): cluster at **30–62 ms**
- Index reductions (#51–52): cluster at **25–27 ms**

These are **plateaus**, not real compute times — they expose QNN's per-call framework overhead floor at ~25–65 ms. Any reported QNN time below ~100 ms is almost certainly dominated by this floor.

### Recommended filter

To produce a believable speedup table from the existing QNN data without re-measuring, drop any problem failing **both** of:

```
QNN_ms      >= 200    # ≥ 4× the framework-overhead floor
our_best_ms >= 10     # ≥ 100× our launch-latency floor
```

This leaves the comparison on problems where actual GPU compute dominates on both sides.

### What survives the filter (current dataset)

| # | Problem | QNN (ms) | our best (ms) | Speedup |
|---:|---|---:|---:|---:|
| 1 | Square_matmul | 1126 | 614 | **1.8×** |
| 2 | Standard_matmul | 1432 | 540 | **2.7×** |
| 3 | Batched_matmul | 1049 | 147 | **7.2×** |
| 6 | Matmul_large_K | 1567 | 542 | **2.9×** |

Geomean over these 4: **~3.0×**, median **2.8×**. These are the headline-quality numbers.

### What gets dropped and why

| # | Problem | QNN (ms) | our best (ms) | Why dropped |
|---:|---|---:|---:|---|
| 4 | Matrix_vector_mul | 338 | 12.3 | our_best below 10 ms threshold; QNN borderline |
| 23 | Softmax | 455 | 4.10 | our_best below 10 ms; QNN includes overhead floor |
| 24 | LogSoftmax | 67 | 3.51 | both fail thresholds |
| 25–32 | pointwise activations | 53–66 | 0.06 | QNN at framework floor; our_best is 6 µs |
| 38 | L1Norm | 135 | 4.85 | QNN below 200 ms threshold |
| 47–50 | dim reductions | 30–62 | 0.10 | QNN at framework floor |
| 51–52 | argmax/argmin | 25–27 | 0.13 | QNN at framework floor |
| 100 | HingeLoss | 51 | 0.006 | QNN at framework floor; reported 8587× is meaningless |

### Looser alternative filter

If 4 problems is too few, the looser cut

```
QNN_ms      >= 100    # ≥ 2× the framework floor
our_best_ms >=  1     # ≥ 10× our launch-latency floor
```

adds problems 4, 23, 24, 38 — five-to-nine problems total. Numbers are noisier but still meaningfully driven by GPU compute on both sides. Speedups for those are 19–111×, which is large but not absurd given QNN's overhead is still substantial relative to its compute.

### Don't quote at all without re-measurement

For #25–32, #47–53, and #100, the speedup ratios are **predominantly framework-overhead-on-QNN-vs-our-GPU-time** — the ~50 ms QNN floor on these problems is essentially constant, while our compute really is ~6 µs to 0.5 ms. Reporting "1000× speedup over QNN" on a HardSigmoid kernel is technically what the numbers show, but defensibly indistinguishable from "our timing methodology excludes the overhead theirs includes." Skip these in any external comparison.

---

## TL;DR for the QNN team on the other machine

> Our 0.006 ms for HingeLoss is the OpenCL `cl_event` GPU kernel time only, with one warmup run preceding it; we don't include buffer alloc, JIT compile, host↔device transfer, or framework dispatch. To match it, please report QNN per-op GPU time (via the QNN profiling API) rather than wall-clock `run()`, OR loop & amortize over ≥1000 calls.

---

*Generated 2026-04-30 from analysis of `out/kernelbench-opencl/opencl_rl/gpt-5-mini-2025-08-07/` and `level1_qnn_qualcomm_results.json`. Profiler logic at `data/benchmark-opencl/L1/<p>/driver.c` (LLM-generated, follows the harness convention enforced by `scripts/kgen_step_*.py`).*
