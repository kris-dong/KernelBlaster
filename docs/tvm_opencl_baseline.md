# TVM-OpenCL baseline on Qualcomm Adreno

An optimization-based compiler baseline for the KernelBench Level1 OpenCL
problems (`data/benchmark-opencl/L1`, 52 ops): Apache TVM's Adreno OpenCL
pipeline, timed on the board's GPU, compared with the QNN vendor baseline
(`level1_qnn_qualcomm_results_fixed.json`) and KernelBlaster's OpenCL results.
Scripts: `scripts/tvm_baseline/`.

| script | what |
|---|---|
| `run_tvm_opencl_baseline.py [NUM ...]` | the sweep: every L1 op (or the listed numbers) through the strategy cascade below; results merged into `level1_tvm_opencl_results.json` |
| `recover_tvm_failures.py NUM ...` | second-round recovery for listed failures (`max_num_threads` caps, scalar baking, dynamo ONNX exporter, exact decompositions); merges into the same JSON |
| `tvm_rpc_probe.py` | one 512x512 fp16 matmul compiled on the host, run on the board: checks the toolchain, the tunnel and the board runtime |
| `compare_baselines.py` | per-problem CSV `opencl_baseline_comparison.csv`, `opencl_baseline_summary.json`, plots |
| `plot_kb_speedup_vs_baselines.py` | KernelBlaster speedup over QNN and TVM in one plot (TVM fallback bars hatched) |
| `board_build_runtime.sh` | on the board: TVM runtime + C++ RPC server build |
| `board_start_rpc.sh` | on the board: start `tvm_rpc` on the board's localhost |
| `tvm_paths.py` | paths and the RPC endpoint, all overridable from the environment |

## Setup

Compile on the x86 host, run on the Adreno GPU over RPC (the QAIRT `libtvm.so`
on the board is an x86 cross-compiler, unusable on the device).

**Board** (RB5 / QRB5165, Ubuntu aarch64, `libOpenCL.so`):

1. `scripts/tvm_baseline/board_build_runtime.sh` builds TVM v0.25.0 runtime-only
   with `USE_OPENCL=ON`, `USE_CPP_RPC=ON` into `/root/tvm/build`.
2. `scripts/tvm_baseline/board_start_rpc.sh` starts `tvm_rpc` bound to
   **127.0.0.1:9191 only**: an unauthenticated TVM RPC server executes any
   uploaded module, so it must never listen on a network interface. It preloads
   `libtvm_runtime_opencl.so` so the Adreno OpenCL device registers.

**Host** (x86 Linux):

- Python with `apache-tvm` 0.25 (Relax, `tvm.relax.backend.adreno`), `onnx`,
  `torch`, `numpy`, `matplotlib`.
- An aarch64 cross gcc to link the board-side `.so`:
  `mamba create -y -p <KB>/.xgcc gcc_linux-aarch64 gxx_linux-aarch64`
  (default `TVM_XCC=<KB>/.xgcc/bin/aarch64-conda-linux-gnu-gcc`).
- KernelBench level1 reference models in `data/KernelBench/KernelBench/level1`
  (a KernelBench checkout; or set `TVM_REF_DIR`).
- The tunnel: `ssh -N -L 9191:127.0.0.1:9191 root@<board>`.

Then `python scripts/tvm_baseline/tvm_rpc_probe.py`, the sweep, recovery for the
listed failures, and the two analysis scripts. Outputs go to
`out/tvm_baseline/{data,plots}` (`TVM_OUT_DIR`); `compare_baselines.py` also
reads KernelBlaster's `out/opencl_speedup_summary.csv` (`TVM_KB_SUMMARY`).

| variable | default |
|---|---|
| `KB_ROOT` | the checkout holding `scripts/tvm_baseline` |
| `TVM_L1_DIR` | `data/benchmark-opencl/L1` |
| `TVM_REF_DIR` | `data/KernelBench/KernelBench/level1` |
| `TVM_XCC` | `.xgcc/bin/aarch64-conda-linux-gnu-gcc` |
| `TVM_RPC_HOST`, `TVM_RPC_PORT` | `127.0.0.1`, `9191` |
| `TVM_OUT_DIR` | `out/tvm_baseline` |
| `TVM_QNN_JSON` | `level1_qnn_qualcomm_results_fixed.json` |
| `TVM_KB_SUMMARY` | `out/opencl_speedup_summary.csv` |

## Method

Per op: KernelBench reference model → fp16 → ONNX (opset 17) →
`relax.frontend.onnx.from_onnx` → `tvm.relax.backend.adreno.get_default_pipeline`
(Adreno OpenCL schedules, texture memory) → `relax.build` for
`opencl -device=adreno`, host `llvm -mtriple=aarch64-linux-gnu` → cross-linked
`.so` → RPC upload → `relax.VirtualMachine` on the Adreno GPU → `time_evaluator`.

**Timing.** `time_evaluator` warms up, then loops `number` calls per `repeat` with
a device sync; `avg_ms_gpu_kernel` is the minimum over repeats of the per-call
mean: amortized on-device latency, the QNN "loop and amortize" methodology of
`docs/adreno_opencl_profiling_methodology.md`, and comparable to KernelBlaster's
`cl_event` kernel time to within the VM's dispatch overhead. The result schema
matches the QNN JSON (`avg_ms_gpu_kernel`, `avg_ms_wallclock`, `status`, `stage`,
`note`). Headline ratios use the compute-bound filter of the methodology doc
(TVM ≥ 10 ms and KernelBlaster ≥ 1 ms; TVM ≥ 10 ms and QNN ≥ 200 ms).

**Strategy cascade.** TVM runs its default Adreno pipeline with no per-op
autotuning (an out-of-the-box optimizing compiler). Ops it cannot compile go
through fallbacks, and every result records the `path` that produced it so a
fallback number is never reported as the optimized one (`tvm_optimized=False`
in the comparison CSV, hatched in the plot):

1. `adreno/float16`: the optimized Adreno pipeline (primary);
2. `gpu_generic/float16`: generic GPU schedules (valid work-group sizes);
3. `gpu_generic/float32`: fixes fp16 constant dtype mismatches;
4. `recover_tvm_failures.py`: `max_num_threads` 128/64/32, python scalars baked
   into constants, the dynamo ONNX exporter, exact decompositions
   (cross-entropy → log_softmax + gather + mean).

`TVM_FALLBACK_ONLY=1` skips the optimized path when re-running known failures.

**Known hard limits.** Buffers over Adreno's 256 MiB single-allocation limit
(the large matmul shapes), `CL_INVALID_WORK_GROUP_SIZE` on 16384-wide reductions
even at `max_num_threads=32`, and TVM 0.25 scheduler errors on some loss and
reduction fusions.
