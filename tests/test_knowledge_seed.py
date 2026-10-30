"""The comprehensive v2 seed: every technique / state / scope parses and is consistent, aliases
are unique, mined data is valid, and retrieval stays within a bounded prompt budget."""
import json
import re

import pytest

from kernelblaster.knowledge.outcomes import OP_CLASSES
from kernelblaster.knowledge.v2 import KnowledgeV2
from kernelblaster.knowledge.v2 import predicates as P
from kernelblaster.knowledge.v2.metrics import CANONICAL
from kernelblaster.knowledge.v2.ontology import Ontology, Technique, load_seed_techniques, scope_of
from kernelblaster.knowledge.v2.states import StateHierarchy, load_seed
from kernelblaster.knowledge.v2.targets import DATA_DIR, TargetRegistry

REG = TargetRegistry()
RAW = load_seed_techniques()
ONTO = Ontology.seed(REG)
SAMPLE_VECTORS = [{}, {k: 50.0 for k in CANONICAL}, {k: 0.5 for k in CANONICAL},
                  {"mem_util": 80.0, "compute_util": 10.0, "arith_intensity": 0.2, "parallel_fill": 1.0}]
CTXS = [{"op_class": oc, "dtype": dt, "family": "simt_gpu", "target": "cuda:sm_89"}
        for oc in OP_CLASSES for dt in ("fp32", "fp16", "bf16")]


def _valid_scope(scope: str) -> bool:
    if scope == "universal":
        return True
    kind, _, rest = scope.partition(":")
    if kind in ("node", "family"):
        return rest in REG.nodes
    if kind != "target":
        return False
    if rest in REG.targets:
        return True
    m = re.fullmatch(r"([a-z0-9_]+)(?::sm_(\d+)\+)?", rest)
    return bool(m) and m.group(1) in REG.backends


def test_tree_and_targets():
    assert "universal" in REG.nodes and all(REG.ancestors(n)[-1] == "universal" for n in REG.nodes)
    for key, t in REG.targets.items():
        assert t.node in REG.nodes, key
        assert t.peak_gflops and t.peak_gbps and t.limits, f"{key}: peaks and limits"
    for b, n in REG.backends.items():
        assert n in REG.nodes, b
    assert REG.chain("rvv:vlen512")[:3] == ["rvv:vlen512", "rvv", "vector_cpu"]
    assert REG.chain("gemmini:default")[1:3] == ["gemmini", "systolic"]


def test_every_technique_parses_and_scopes_exist():
    assert len(RAW) >= 60
    for d in RAW:
        t = Technique.from_dict(d)
        Ontology([t], REG)                                      # validates predicates and predicts
        sid = REG.scope_id(t.level, t.scope)
        assert sid == "universal" or sid in REG.nodes or sid in REG.targets, (t.id, sid)
        assert t.realizations, f"{t.id}: no realization"
        for r in t.realizations:
            assert _valid_scope(r["scope"]), (t.id, r["scope"])
            assert len(r["how"]) >= 20, (t.id, r["scope"])
            if r["scope"] != "universal" and r["scope"].startswith("node:"):
                # a realization below the technique's own scope is fine, one outside it never applies
                n = scope_of(r["scope"])
                assert sid == "universal" or sid in REG.ancestors(n) or n in REG.ancestors(sid) if sid in REG.nodes \
                    else True, (t.id, r["scope"])
        for oc in t.op_classes:
            assert oc in OP_CLASSES, (t.id, oc)
        assert set(t.states) <= set(StateHierarchy.seed(REG).states), (t.id, t.states)
        assert t.prior and 0 <= t.prior["mean"] <= 0.5 and 0 < t.prior["n"] <= 2, f"{t.id}: seed prior, low pseudo-count"
        assert t.mechanism and len(t.mechanism) > 30, t.id
        for pre in (t.preconditions,):
            for vals in SAMPLE_VECTORS:
                for ctx in CTXS:
                    assert P.evaluate(pre, vals, ctx) in (True, False, None)
        assert not P.unit_problems(t.preconditions), (t.id, t.preconditions)


def test_no_duplicate_names_or_aliases():
    ids = [d["id"] for d in RAW]
    assert len(ids) == len(set(ids))
    owner = {}
    for d in RAW:
        for a in [d["id"]] + d.get("aliases", []):
            k = a.lower()
            assert k not in owner or owner[k] == d["id"], f"{a!r} used by {owner.get(k)} and {d['id']}"
            owner[k] = d["id"]


def test_states_seed_consistent():
    states, legacy = load_seed()
    h = StateHierarchy(states, REG, legacy)
    for s in states:
        assert s.parent is None or s.parent in h.states
        assert REG.scope_id(s.level, s.scope) in REG.nodes or s.level == "target"
        for vals in SAMPLE_VECTORS:
            assert P.evaluate(s.predicate, vals, CTXS[0]) in (True, False, None)
        assert s.mechanism
    assert set(legacy.values()) <= set(h.states)
    for new in ("tensor_core_eligible", "smem_bank_conflicted", "uncoalesced_access", "register_limited_occupancy",
                "launch_overhead", "streaming_low_reuse"):
        assert new in h.states
    v = {"mem_util": {"v": 10, "src": "measured"}, "compute_util": {"v": 20, "src": "measured"},
         "arith_intensity": {"v": 40, "src": "estimated"}, "parallel_fill": {"v": 1, "src": "estimated"},
         "launch_frac": {"v": 0.1, "src": "estimated"}}
    assert h.classify(v, target="cuda:sm_75", op_class="gemm").chain[-1] == "tensor_core_eligible"


def test_target_specific_realizations():
    real = lambda tid, tk: (ONTO.realization(ONTO.get(tid), tk) or {}).get("how", "")   # noqa: E731
    assert "cp.async" in real("cp_async_multistage_pipeline", "cuda:sm_80")
    assert real("cp_async_multistage_pipeline", "cuda:sm_75") == ""               # no cp.async on Turing
    assert "fp16 only" in real("tensor_core_mma", "cuda:sm_75")
    assert "tf32" in real("tensor_core_mma", "cuda:sm_89")
    assert "wgmma" in real("tensor_core_mma", "cuda:sm_90")
    assert real("tma_wgmma_pipeline", "cuda:sm_89") == "" and "TMA" in real("tma_wgmma_pipeline", "cuda:sm_90")
    assert "image2d_t" in real("image2d_texture_reads", "opencl:adreno650")
    assert "64 or 128" in real("launch_configuration_tuning", "opencl:adreno740")
    assert "vsetvl" in real("rvv_strip_mining_lmul", "rvv:vlen128") and "LMUL" in real(
        "vectorized_memory_access", "rvv:vlen512")
    assert "scratchpad" in real("scratchpad_tiling", "gemmini:default")
    # CUDA spellings never reach OpenCL / RVV prompts
    for t in ONTO.techniques.values():
        for tk in ("opencl:adreno740", "rvv:vlen256"):
            how = (ONTO.realization(t, tk) or {}).get("how", "")
            assert not re.search(r"__shfl|__syncthreads|__shared__|cp\.async|wmma|__ldg", how), (t.id, tk, how)


def test_coverage_per_op_class():
    ctx = lambda oc: {"op_class": oc, "dtype": "fp32", "family": "simt_gpu", "target": "cuda:sm_89"}   # noqa: E731
    for oc in OP_CLASSES:
        n = len(ONTO.candidates(target_key="cuda:sm_89", op_class=oc, chain=[], values={}, ctx=ctx(oc)))
        assert n >= 6, (oc, n)
    kinds = {"conv2d_dw": "depthwise_conv_spatial_tiling", "conv_transpose2d": "transposed_conv_output_gather",
             "lstm_seq": "rnn_input_projection_hoisting", "cumsum": "blocked_scan", "softmax": "online_softmax",
             "layer_norm": "welford_online_statistics", "maxpool2d": "pooling_window_reuse",
             "index_select": "gather_contiguous_inner", "sdpa": "flash_style_tiling", "topk": "topk_register_selection"}
    oc_of = {"conv2d_dw": "conv", "conv_transpose2d": "conv", "lstm_seq": "gemm", "cumsum": "scan",
             "softmax": "reduction", "layer_norm": "reduction", "maxpool2d": "reduction",
             "index_select": "data_movement", "sdpa": "attention", "topk": "reduction"}
    for kind, tid in kinds.items():
        got = {t.id for t in ONTO.candidates(target_key="cuda:sm_89", op_class=oc_of[kind], chain=[], values={},
                                             ctx=ctx(oc_of[kind]), op_kind=kind)}
        assert tid in got, (kind, tid)
    # op-kind gates keep specialised techniques away from other kinds
    got = {t.id for t in ONTO.candidates(target_key="cuda:sm_89", op_class="conv", chain=[], values={},
                                         ctx=ctx("conv"), op_kind="conv2d")}
    assert "depthwise_conv_spatial_tiling" not in got and "implicit_gemm" in got
    fused = {t.id for t in ONTO.candidates(target_key="cuda:sm_89", op_class="conv", chain=[], values={},
                                           ctx=ctx("conv"), op_kind="fused__conv2d_dw__relu")}
    assert "depthwise_conv_spatial_tiling" in fused


def test_mined_seed_valid():
    fp = DATA_DIR / "mined.json"
    d = json.loads(fp.read_text())
    assert d["outcomes"] and d["claims"]
    for o in d["outcomes"]:
        assert o["item"] in ONTO.techniques and o["op_class"] in OP_CLASSES and isinstance(o["reward"], float)
    for c in d["claims"]:
        assert c["technique"] in ONTO.techniques and c["effect"] in ("helps", "hurts")
        assert c["scope"] in REG.nodes or c["scope"] in REG.targets
        assert len(c["targets"]) == 1 or c["scope"] == REG.lca(c["targets"])
        assert len(c["evidence"]) >= 2


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    return KnowledgeV2(tmp_path_factory.mktemp("seeded"))


def test_seeded_db_and_prompt_budget(seeded):
    kv = seeded
    assert len(kv.claims.active()) >= 20 and kv.stats.data
    assert any(c.source == "mined" and c.level == "node" for c in kv.claims.active())
    budget = 3500
    worst = 0
    for tk in ("cuda:sm_75", "cuda:sm_89", "cuda:sm_90", "opencl:adreno650", "rvv:vlen256", "gemmini:default"):
        for oc in OP_CLASSES:
            for vals in SAMPLE_VECTORS[1:]:
                vec = {k: {"v": v, "src": "measured"} for k, v in vals.items()}
                cls = kv.classify(vec, target_key=tk, op_class=oc)
                txt = kv.retrieve(target_key=tk, op_class=oc, dtype="fp32", cls=cls, vec=vec, problem_id="p", slot="s")
                worst = max(worst, len(txt))
                techs, _ = kv.candidates(target_key=tk, op_class=oc, dtype="fp32", chain=cls.chain, vec=vec)
                for t in techs:
                    assert len(kv.technique_text(t, tk)) < 1200, t.id
    assert worst <= budget, worst
    # the seeded priors break ties among untried arms: the highest prior wins the first pick
    import random
    vec = {"mem_util": {"v": 10, "src": "estimated"}, "compute_util": {"v": 20, "src": "estimated"},
           "arith_intensity": {"v": 40, "src": "estimated"}}
    cls = kv.classify(vec, target_key="gemmini:default", op_class="gemm")
    ch = kv.select(target_key="gemmini:default", op_class="gemm", dtype="fp32", cls=cls, vec=vec,
                   rng=random.Random(0))
    assert ch.item_id in ("scratchpad_tiling", "shared_memory_tiling", "register_blocking", "implicit_gemm")


def test_seed_cli(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts" / "seed_knowledge.py"
    out = subprocess.run([sys.executable, str(script), str(tmp_path / "k"), "--fresh", "--report"],
                         capture_output=True, text=True, check=True).stdout
    rep = json.loads(out)
    assert rep["techniques"] >= 60 and rep["claims"] >= 20 and rep["coverage"]["conv"]
    again = json.loads(subprocess.run([sys.executable, str(script), str(tmp_path / "k")], capture_output=True,
                                      text=True, check=True).stdout)
    assert again["claims"] == rep["claims"] and again["stats_rows"] == rep["stats_rows"]     # idempotent
