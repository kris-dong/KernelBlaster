"""A2 capabilities in KernelBlaster: the recurrent op class (menu, roofline state, prompt notes, technique seeds for
existing stores, opt-in base-kind matching) and multi-turn codegen with the experiment tool (protocol, budget,
restrictions, journal keys, default off). GPU-free; the nvcc test runs only where nvcc and a GPU exist."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kernelblaster.agents import experiment_tool as xt  # noqa: E402
from kernelblaster.agents import hw_profile  # noqa: E402
from kernelblaster.agents import opt_multislot as om  # noqa: E402


def _gru_problem(d: Path, op_class: str):
    from kernelblaster.knowledge.slots import SlotProblem
    sig = ("void kernel_gru_seq_f16(const __half *x, const __half *h0, const __half *w_ih, const __half *w_hh, "
           "const __half *b_ih, const __half *b_hh, __half *y, __half *hn, int B, int T, int I, int H, int reverse)")
    man = {"format": "mb-kb-problem/1", "name": "t", "dtype": "fp16", "target": "cuda",
           "tolerance": {"slot_rel": 2e-3}, "driver": "driver.cpp", "data": "data.bin",
           "assembly": {"head": "h.cu", "tail": "t.cu", "namespace_prefix": "mbslot_", "order": ["gru"]},
           "slots": [{"id": "gru", "op": "gru_seq_f16", "ops": ["gru_seq_f16"], "op_class": op_class,
                      "entry": "kernel_gru_seq_f16", "signature": sig, "file": "slots/gru.cu",
                      "reference_file": "slots/gru.ref.c", "semantics": "GRU", "seed_source": "hand",
                      "instances": [{"op_name": "g", "op": "gru_seq_f16", "index": 0,
                                     "shape": {"B": 10, "T": 512, "I": 512, "H": 256, "reverse": 0}}]}],
           "ops": [{"index": 0, "name": "g", "op": "gru_seq_f16", "slot": "gru", "inputs": [0], "outputs": [1]}],
           "buffers": [{"name": "x", "kind": "input", "dtype": "f16", "numel": 10 * 512 * 512, "alias": None},
                       {"name": "y", "kind": "output", "dtype": "f16", "numel": 10 * 512 * 256, "alias": None}],
           "inputs": [0], "outputs": [1]}
    (d / "slots").mkdir(parents=True, exist_ok=True)
    (d / "slots" / "gru.cu").write_text("// seed\n")
    (d / "slots" / "gru.ref.c").write_text("// ref\n")
    for f in ("h.cu", "t.cu", "driver.cpp", "data.bin"):
        (d / f).write_text("")
    (d / "manifest.json").write_text(json.dumps(man))
    return SlotProblem(d)


class RecurrentClassTest(unittest.TestCase):
    def test_v1_menu(self):
        gemm = [t for t, _ in om.techniques_for("gemm", om.TECHNIQUES, arch="89", dtype="fp16")]
        rec = [t for t, _ in om.techniques_for("recurrent", om.TECHNIQUES, arch="89", dtype="fp16")]
        self.assertIn("split_k_or_stream_k", gemm)
        self.assertNotIn("split_k_or_stream_k", rec)
        self.assertEqual(rec[:3], ["whh_mma_fragments_in_registers", "persistent_recurrence_state_on_chip",
                                   "recurrence_input_projection_hoisting"])
        self.assertEqual(hw_profile.tensor_core_techniques("89", "fp32", "recurrent"), [])

    def test_roofline_state_and_prompt(self):
        with tempfile.TemporaryDirectory() as d:
            p = _gru_problem(Path(d) / "g", "gemm")
            st, nums = om.slot_state(p, "gru", 2774.9, "89", (91600.0, 864.0))
            self.assertNotEqual(st, "latency_bound")
            self.assertNotIn("us_per_step", nums)
            p = _gru_problem(Path(d) / "r", "recurrent")
            st, nums = om.slot_state(p, "gru", 2774.9, "89", (91600.0, 864.0))
            self.assertEqual(st, "latency_bound")
            self.assertEqual(nums["serial_steps"], 512)
            self.assertAlmostEqual(nums["us_per_step"], 2774.9 / 512, places=2)
            msgs = om.build_codegen_messages(problem=p, sid="gru", source="//", technique="t", arch="89",
                                             technique_desc="d", state=st, state_numbers=nums, slot_us=1.0,
                                             partition_us=2.0, neighbours=[], examples=[], adapt=None,
                                             knowledge_text="", already_tried="", best_note="")
            self.assertIn("Recurrent op notes", msgs[1]["content"])
            self.assertIn("142 SMs", msgs[1]["content"])
            p = _gru_problem(Path(d) / "g2", "gemm")
            msgs = om.build_codegen_messages(problem=p, sid="gru", source="//", technique="t", arch="89",
                                             technique_desc="d", state=st, state_numbers=nums, slot_us=1.0,
                                             partition_us=2.0, neighbours=[], examples=[], adapt=None,
                                             knowledge_text="", already_tried="", best_note="")
            self.assertNotIn("Recurrent op notes", msgs[1]["content"])
            self.assertNotIn("Whole-model context", msgs[1]["content"])

    def test_v2_candidates_and_old_store(self):
        from kernelblaster.knowledge.v2.knowledge import KnowledgeV2
        with tempfile.TemporaryDirectory() as d:
            kv = KnowledgeV2(Path(d), seed_mined=False)
            kv.registry.ensure("cuda:sm_89", peaks=(91600.0, 864.0))
            ids = lambda c: [t.id for t in kv.candidates(target_key="cuda:sm_89", op_class=c, dtype="fp16",  # noqa
                                                         chain=["latency_bound"], vec={}, op_kind="gru_seq_f16")[0]]
            rec, gemm = ids("recurrent"), ids("gemm")
            self.assertIn("persistent_recurrence_state_on_chip", rec)
            self.assertNotIn("split_k_parallelism", rec)
            self.assertIn("split_k_parallelism", gemm)
            self.assertNotIn("persistent_recurrence_state_on_chip", gemm)
            # a store seeded before 45_recurrent.json existed gets the recurrent-only techniques, nothing else
            with kv.db.tx() as c:
                c.execute("DELETE FROM techniques WHERE id IN ('step_sync_minimization', 'flash_style_tiling')")
            kv2 = KnowledgeV2(Path(d), seed_mined=False)
            have = {t["id"] for t in kv2.db.techniques()}
            self.assertIn("step_sync_minimization", have)
            self.assertNotIn("flash_style_tiling", have)               # other classes' seeds are not re-added

    def test_kind_match_opt_in(self):
        from kernelblaster.knowledge.v2.ontology import _kind_match
        self.assertFalse(_kind_match("gru_seq_f16", ["gru_seq"]))
        os.environ["KERNELBLASTER_KIND_MATCH"] = "base"
        try:
            self.assertTrue(_kind_match("gru_seq_f16", ["gru_seq"]))
            self.assertTrue(_kind_match("fused__conv2d_f16__relu_f16", ["conv2d"]))
        finally:
            del os.environ["KERNELBLASTER_KIND_MATCH"]
        self.assertTrue(_kind_match("fused__par__gru_seq_f16__gru_seq_f16", ["gru_seq_f16"]))


class FakeTool:
    arch = "89"

    def __init__(self):
        self.calls = []

    async def run(self, code, left):
        self.calls.append(code)
        return {"ok": True, "stage": "run", "exit": 0, "out": f"probe {len(self.calls)}: 640 cycles", "wall_s": 0.1}


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if sys.version_info < (3, 10) else asyncio.run(coro)


EXP = "```experiment\n#include <cstdio>\nint main(){printf(\"x\");}\n```"
FINAL = "```cuda\nvoid kernel_x() {}\n```"


class MultiTurnTest(unittest.TestCase):
    def _llm(self, replies, seen):
        async def llm(conv, key, extra=None):
            seen.append((key, [m["content"] for m in conv]))
            return replies.pop(0)
        return llm

    def test_default_off(self):
        os.environ.pop(xt.ENV, None)
        self.assertFalse(xt.enabled())

    def test_experiment_then_answer(self):
        seen, tool = [], FakeTool()
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        text, log = _run(xt.converse(self._llm([EXP, FINAL], seen), msgs, tool, {"turns": 3, "runs": 4, "seconds": 60},
                                     key="r0s1c0"))
        self.assertEqual(text, FINAL)
        self.assertEqual([k for k, _ in seen], ["r0s1c0", "r0s1c0t1"])
        self.assertIn("Experiment tool", seen[0][1][-1])
        self.assertIn("probe 1: 640 cycles", seen[1][1][-1])
        self.assertEqual((log.turns, log.runs, log.forced_final), (2, 1, False))
        self.assertEqual(msgs[1]["content"], "u")                       # caller's messages untouched

    def test_single_shot_answer_costs_one_call(self):
        seen = []
        text, log = _run(xt.converse(self._llm([FINAL], seen), [{"role": "user", "content": "u"}], FakeTool(),
                                     {"turns": 3, "runs": 4, "seconds": 60}, key="k"))
        self.assertEqual((log.turns, log.runs), (1, 0))

    def test_budget_forces_answer(self):
        seen, tool = [], FakeTool()
        text, log = _run(xt.converse(self._llm([EXP + EXP, EXP, FINAL], seen), [{"role": "user", "content": "u"}],
                                     tool, {"turns": 2, "runs": 1, "seconds": 60}, key="k"))
        self.assertEqual(text, FINAL)
        self.assertEqual(len(tool.calls), 1)                            # runs budget: the second block never ran
        self.assertTrue(log.forced_final)
        self.assertEqual([k for k, _ in seen], ["k", "kt1", "ktf"])

    def test_restrictions(self):
        self.assertIsNone(xt.refused("#include <cstdio>\nint main(){return 0;}"))
        for bad in ('int main(){system("ls");}', "#include <unistd.h>\nint main(){}", "int main(){FILE*f=fopen(\"x\",\"w\");}",
                    "#include <fstream>\nint main(){}", "__global__ void k(){}", "x" * (xt.MAX_SRC + 1) + "int main(){}"):
            self.assertIsNotNone(xt.refused(bad), bad[:40])

    def test_budget_env(self):
        os.environ[xt.ENV_BUDGET] = "2,3,45"
        try:
            self.assertEqual(xt.budget(), {"turns": 2, "runs": 3, "seconds": 45.0})
        finally:
            del os.environ[xt.ENV_BUDGET]

    def test_model_context_text(self):
        man = {"model_context": {"n_ops": 35, "model_us": 9316.0, "kinds": {"gru_seq_f16": {"count": 12, "share": 0.99}},
                                 "top_ops": [{"name": "gru_l1", "op": "gru_seq_f16", "us": 555.0, "share": 0.06,
                                              "in_this_problem": True}]}}
        t = xt.model_context_text(man)
        self.assertIn("Whole-model context", t)
        self.assertIn("[in this problem]", t)
        self.assertEqual(xt.model_context_text({}), "")


@unittest.skipUnless(shutil.which("nvcc") and shutil.which("nvidia-smi"), "needs nvcc + a GPU")
class RealToolTest(unittest.TestCase):
    def test_compile_and_run(self):
        from kernelblaster.knowledge.slots import LocalEvaluator
        Ev = lambda: LocalEvaluator(arch="89", gpus="all")  # noqa: E731 -- the run's own evaluator and GPU pool

        class P:
            root = Path(tempfile.mkdtemp())

        with tempfile.TemporaryDirectory() as d:
            tool = xt.ExperimentTool(Ev(), P(), "89", Path(d))
            code = ('#include <cstdio>\n__global__ void k(long long *o){long long t=clock64(); o[0]=clock64()-t;}\n'
                    'int main(){long long *o; cudaMalloc(&o,8); k<<<1,1>>>(o); long long h; '
                    'cudaMemcpy(&h,o,8,cudaMemcpyDeviceToHost); printf("cycles %lld\\n", h); return 0;}\n')
            r = _run(tool.run(code, 100.0))
            self.assertTrue(r["ok"], r)
            self.assertIn("cycles", r["out"])
            r = _run(tool.run("int main(){ syntax error }", 100.0))
            self.assertEqual(r["stage"], "compile")


if __name__ == "__main__":
    unittest.main()
