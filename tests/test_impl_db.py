# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A5 implementation database: pools / routing / retention / similarity retrieval / tunables / links (no GPU)."""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from kernelblaster.knowledge.impl_links import ImplLinks
from kernelblaster.knowledge.impl_pool import (Pool, SourceIndex, rank_kernels, shape_distance, shape_key,
                                               struct_info)
from kernelblaster.knowledge.impl_store import ImplStore, shape_bucket, signature_sha
from kernelblaster.knowledge.tunables import best_settings, contract, extract_tunables, sweep_plan, variant_source

SIG = "void k(const half* x, half* y, int N, int C)"


def conv(c, hw=28, k=3):
    return {"N": 10, "IC": c, "IH": hw, "IW": hw, "OC": 32, "OH": hw, "OW": hw, "KH": k, "KW": k, "SH": 1, "SW": 1}


def add(store, code, shapes, us, *, op="conv2d_f16", problem="p1", sp=2.0, parent=None, techniques=(), rid=None):
    return store.add(code, target="cuda", arch="sm_89", op=op, op_class="conv", signature_sha=signature_sha(SIG),
                     shapes=shapes, buckets=sorted({shape_bucket("conv", s) for s in shapes}), slot_us=us,
                     seed_us=us * sp, speedup_vs_seed=sp, problem_id=problem, dtype="fp16", parent_id=parent,
                     techniques=list(techniques), **({"id": rid} if rid else {}))


# ---------------------------------------------------------------- migration / compatibility

def test_old_index_and_unknown_keys_load(tmp_path):
    s = ImplStore(tmp_path)
    r = add(s, "k1", [conv(64)], 10.0)
    line = json.loads((tmp_path / "index.jsonl").read_text())
    line["a_future_field"] = {"x": 1}                    # a newer writer's extra key must not break readers
    (tmp_path / "index.jsonl").write_text(json.dumps(line) + "\n")
    assert ImplStore(tmp_path).records()[0].id == r.id
    assert not (tmp_path / "pool").exists()                 # nothing new is written by reading


def test_retrieve_default_is_the_unchanged_tiers(tmp_path):
    s = ImplStore(tmp_path)
    add(s, "own", [conv(64)], 10.0, problem="me")
    add(s, "other", [conv(64)], 8.0, problem="p2")
    add(s, "near", [conv(96)], 9.0, problem="p3")
    kw = dict(target="cuda", arch="sm_89", op="conv2d_f16", op_class="conv", signature=SIG, shapes=[conv(64)],
              exclude_problem="me", dtype="fp16")
    a, b = s.retrieve(**kw), s._retrieve_tiers(**kw)
    assert set(a) == {"seed", "adapt", "examples"} and a["seed"].id == b["seed"].id
    c = s.retrieve(**kw, similar_k=2)
    assert c["seed"].id == a["seed"].id and len(c["similar"]) == 2   # the tiers win; similar is extra


def test_similar_fills_adapt_only_when_the_tiers_are_empty(tmp_path):
    s = ImplStore(tmp_path)
    far = add(s, "far", [conv(64, hw=7, k=1)], 5.0, problem="p2", sp=1.03)    # no shared bucket, low speedup
    kw = dict(target="cuda", arch="sm_89", op="conv2d_f16", op_class="conv", signature=SIG,
              shapes=[conv(512, hw=56)], dtype="fp16")
    assert s.retrieve(**kw)["adapt"] is None
    r = s.retrieve(**kw, similar_k=3)
    assert r["adapt"].id == far.id and r["adapt_from"] == "similar"


# ---------------------------------------------------------------- pool, routing, pareto

def matrix_pool():
    p = Pool("conv2d_f16", "abc", "sm_89", "fp16")
    s1, s2, s3 = conv(64), conv(128), conv(256, hw=14)
    t = {"A": (10, 30, 50), "B": (20, 12, 40), "C": (25, 25, 60)}
    for k, row in t.items():
        p.add_kernel(k * 40)
        for sh, us in zip((s1, s2, s3), row):
            p.observe(k * 16, sh, us=us)
    return p, (s1, s2, s3)


def test_pool_route_measured_choices_and_incumbent_guard():
    p, (s1, s2, s3) = matrix_pool()
    r = p.route([{"op_name": "o1", "shape": s1}, {"op_name": "o2", "shape": s2}, {"op_name": "o3", "shape": s3}])
    assert [c["choice"]["kernel"][0] for c in r["choices"]] == ["A", "B", "B"]
    assert r["sum_us"] == 10 + 12 + 40 and not r["unvalidated"]
    assert all(c["choice"]["measured"] and not c["choice"]["guard"]["needs_check"] for c in r["choices"])
    # an incumbent faster than min_gain keeps its instance
    r2 = p.route([{"op_name": "o3", "shape": s3}], incumbent={"o3": 41.0}, min_gain=0.05)
    assert r2["choices"][0]["choice"] is None and r2["sum_us"] == 41.0
    # a failure on a shape removes the kernel there (conservative guard), and shows in failed_on
    p.observe("B" * 16, s3, ok=False, err="invalid argument")
    c3 = p.best_for(s3)
    assert c3["kernel"][0] == "A" and p.best_for(s2)["guard"]["n_failed"] == 1


def test_pool_predicts_unmeasured_shapes_and_marks_them():
    p, (s1, s2, s3) = matrix_pool()
    q = conv(160)                                           # between s1 and s2, never timed
    ch = p.best_for(q)
    assert ch is not None and not ch["measured"] and ch["guard"]["needs_check"] and ch["us"] is None
    assert p.best_for(q, measured_only=True) is None
    r = p.route([q])
    assert r["unvalidated"] == [shape_key(q)]


def test_pareto_and_oracle():
    p, _ = matrix_pool()
    assert p.pareto() == sorted(["A" * 16, "B" * 16])       # C is dominated by A on every shape
    assert p.oracle() == 10 + 12 + 40
    m = p.matrix()
    assert len(m["kernels"]) == 3 and len(m["shapes"]) == 3


def test_retention_usefulness_then_diversity():
    p = Pool("conv2d_f16", "abc", "sm_89", "fp16")
    shapes = [conv(c) for c in (64, 128, 256, 512)]
    code = {"A": "for(i) { a[i] = wmma(b); }", "B": "for(j) { c[j] = mma_sync(d); }",
            "C": "for(i) { a[i] = wmma(b); } // copy of A", "D": "__shared__ x; cp.async; ldmatrix; persistent loop"}
    times = {"A": [10, 50, 50, 50], "B": [50, 10, 50, 50], "C": [11, 51, 51, 51], "D": [60, 60, 60, 60]}
    for k, ts in times.items():
        kid = p.add_kernel(k * 40, code=code[k])
        for s, us in zip(shapes, ts):
            p.observe(kid, s, us=us)
    rep = p.retain(3, useful_frac=0.67)                       # 2 useful slots, 1 diversity slot
    kept = {k[0] for k in rep["kept"]}
    assert {"A", "B"} <= kept and len(kept) == 3
    assert "D" in kept and "C" not in kept                    # the distinct kernel, not A's near copy
    assert set(p.kernels) == {k for k in rep["kept"]}
    p2 = Pool("x", "y", "z")
    for k in "ABCDE":
        p2.add_kernel(k * 40)
    assert "E" * 16 in p2.retain(2, protect=["E" * 16])["kept"]


def test_rank_kernels_prefers_the_nearest_shapes_winner():
    p = Pool("conv2d_f16", "abc", "sm_89", "fp16")
    for k, c in (("A", 64), ("B", 128), ("C", 512)):
        p.add_kernel(k * 40, record={"id": k, "shapes": [conv(c)], "slot_us": 10.0, "code_sha": k * 40})
    r = rank_kernels(p, conv(96))
    assert r[0]["kernel"][0] in "AB" and r[-1]["kernel"][0] == "C"
    assert shape_distance(conv(64), conv(64)) == 0 < shape_distance(conv(64), conv(64, k=1))


def test_retrieve_similar_excludes_own_problem_and_returns_records(tmp_path):
    s = ImplStore(tmp_path)
    add(s, "mine", [conv(64)], 1.0, problem="me")
    o = add(s, "other", [conv(64)], 9.0, problem="p2")
    add(s, "far", [conv(512, hw=7)], 1.0, problem="p3")
    r = s.retrieve_similar(arch="sm_89", op="conv2d_f16", signature=SIG, shapes=[conv(64)], k=2,
                           exclude_problem="me", dtype="fp16")
    assert r[0]["record"].id == o.id and all(x["record"].problem_id != "me" for x in r)


def test_observe_persists_cells_into_the_pool(tmp_path):
    s = ImplStore(tmp_path)
    r = add(s, "k1", [conv(64)], 10.0)
    sha = signature_sha(SIG)
    s.observe("conv2d_f16", sha, "sm_89", "fp16", r.code_sha, conv(128), us=7.0, config=2)
    s.observe("conv2d_f16", sha, "sm_89", "fp16", None, conv(256), ok=False, err="bad", code="k2 source")
    p = s.pool("conv2d_f16", sha, "sm_89", "fp16")
    kid = r.code_sha[:16]
    assert p.cell(kid, shape_key(conv(128)))["us"] == 7.0 and p.tuned_table(kid)[shape_key(conv(128))]["config"] == 2
    assert len(p.kernels) == 2 and any(not c["ok"] for row in p.obs.values() for c in row.values())
    assert s.pool("conv2d_f16", SIG, "sm_89", "fp16").key == p.key          # full signature accepted too


def test_struct_info_and_structural_retrieval(tmp_path):
    assert struct_info("fused__conv2d_f16__batchnorm2d_f16__add_f16__a") == {
        "kind": "fused", "stages": ["conv2d_f16", "batchnorm2d_f16", "add_f16"],
        "composition": "conv2d+batchnorm2d+add"}
    assert struct_info("fused__par__gru_seq_f16__gru_seq_f16")["kind"] == "sibling"
    assert struct_info("attention_f16")["kind"] == "attention"
    assert struct_info("bmm_strided_f16")["kind"] == "layout"
    s = ImplStore(tmp_path)
    fop = "fused__conv2d_f16__relu_f16"
    seed = add(s, "composite seed", [{"a": conv(64), "b": {"n": 1000}}], 20.0, op=fop, problem="p2", sp=1.0)
    win = add(s, "fused winner", [{"a": conv(64), "b": {"n": 1000}}], 10.0, op=fop, problem="p2", parent=seed.id)
    add(s, "stage conv", [conv(64)], 8.0, op="conv2d_f16", problem="p3")
    r = s.retrieve_structural(arch="sm_89", op=fop, signature=SIG, shapes=[{"a": conv(64), "b": {"n": 1000}}],
                              dtype="fp16")
    assert r["info"]["kind"] == "fused" and r["same"][0]["record"].id == win.id
    assert [x.id for x in r["lineage"]] == [seed.id, win.id]                 # composite-seed lineage
    assert r["stages"]["conv2d_f16"][0].op == "conv2d_f16"


# ---------------------------------------------------------------- tunables

FAMILY = """#define MB_NUM_CONFIGS 3
#define BM 64
// #define BN 999
constexpr int TILE_K = 32;
static int mb_choose_config(const int* s) {
#ifdef MB_FORCE_CONFIG
  return MB_FORCE_CONFIG;
#endif
#ifdef MB_TUNED_TABLE
  { int c = mb_tuned_lookup(s); if (c >= 0) return c; }
#endif
  return 0; }
__global__ void __launch_bounds__(256, 2) k() {}
"""


def test_tunables_contract_params_and_variants():
    t = extract_tunables(FAMILY)
    assert t["contract"]["configs"] == 3 and t["contract"]["sweepable"] and t["contract"]["table_hook"]
    assert t["params"] == {"BM": 64, "TILE_K": 32} and t["launch_bounds"] == [256, 2]
    v = variant_source(FAMILY, {"BM": 128, "TILE_K": 16, "MB_FORCE_CONFIG": 2})
    assert v.startswith("#define MB_FORCE_CONFIG 2\n") and "#define BM 128" in v and "TILE_K = 16" in v
    assert "// #define BN 999" in v
    with pytest.raises(KeyError):
        variant_source(FAMILY, {"BN": 1})                     # only in a comment: not a tunable
    plan = sweep_plan(t, 10)
    assert {"MB_FORCE_CONFIG": 0} in plan and {"BM": 128} in plan and {"BM": 64} not in plan
    assert contract("__global__ void k() {}") is None
    assert best_settings([{"skey": "a", "config": 1, "us": 5, "ok": True}, {"skey": "a", "config": 2, "us": 4,
                                                                             "ok": True},
                          {"skey": "a", "config": 3, "us": 1, "ok": False}]) == {"a": {"config": 2, "us": 4}}


def test_pool_records_tunables_of_kernels_with_code():
    p = Pool("conv2d_f16", "abc", "sm_89", "fp16")
    kid = p.add_kernel("f" * 40, code=FAMILY)
    assert p.kernels[kid]["tunables"]["contract"]["configs"] == 3
    assert p.add_kernel("e" * 40, code="__global__ void k() {}") and p.kernels["e" * 16]["tunables"] is None


# ---------------------------------------------------------------- knowledge links

def test_impl_links_claims_negatives_render_and_penalty():
    outs = [("o1", {"impl_id": "I1", "technique": "tile_shape_retune", "status": "improved", "speedup": 1.5,
                    "state_before": "s0"}, {"hypothesis_id": "h_1"}),
            ("o2", {"parent_impl_id": "I1", "technique": "split_k", "status": "regressed", "speedup": 0.8,
                    "shape_bucket": "conv:x", "op_kind": "conv2d_f16"}, {}),
            ("o3", {"parent_impl_id": "I1", "technique": "cp_async", "status": "compile_error",
                    "error": "ptxas error", "op_kind": "conv2d_f16"}, {})]
    claims = [{"id": "c1", "kind": "applicability", "text": "tile retune helps 3x3 conv", "support": 3,
               "contradict": 0, "support_ids": ["o1"], "status": "active", "data": {"effect": "helps"}},
              {"id": "c2", "kind": "pitfall", "text": "split_k hurts small-K conv", "support": 2, "contradict": 0,
               "support_ids": ["o2"], "status": "active"}]
    L = ImplLinks(outs, claims)
    x = L.links("I1")
    assert x["produced_by"][0]["technique"] == "tile_shape_retune" and x["hypotheses"] == ["h_1"]
    assert x["claims"][0]["id"] == "c1" and x["children"] == {"regressed": 1, "compile_error": 1}
    assert {n["technique"] for n in x["negatives"]} == {"split_k", "cp_async"}
    assert L.negatives(op_kind="conv2d_f16")[0]["why"] == "split_k hurts small-K conv"
    txt = L.render(["I1"], labels={"I1": "adapt"}, max_chars=10_000)
    assert "impl I1 (adapt)" in txt and "split_k hurts" in txt and "did not work" in txt
    assert len(L.render(["I1"], max_chars=120)) <= 120
    assert L.penalty({"records": ["I1"]}) == pytest.approx(0.05)
    assert L.render(["nothing"]) == ""
