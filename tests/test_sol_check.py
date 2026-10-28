"""The driver's SOL-ExecBench check lines (ModelBlaster pipeline/sol_tolerance.py) in KernelBlaster's result.

A gated export prints ``MB_SOL`` lines (matched ratio of SOL's per-element rule against SOL's own reference outputs and / or the
golden); ``parse_driver_output`` keeps them in ``EvalResult.sol`` and ``error_text`` tells a candidate that the flow's whole-tensor
bound passed but SOL's rule did not (a single-pass TF32 winner), so it can be fixed. An ungated run parses exactly as before.
"""
from kernelblaster.knowledge.slots import parse_driver_output

SOL_FAIL = """MB_SOL add ref=sol matched=0.986120 required=0.99 max_abs_err=0.0487 max_rel_err=3.2 max_excess=0.027 nonfinite=0 allzero=0 cap_exceeded=0 n=4718592 atol=0.021 rtol=1e-05 ok=0 gate=1
MB_SOL add ref=golden matched=0.994800 required=0.99 max_abs_err=0.0416 max_rel_err=2.9 max_excess=0.02 nonfinite=0 allzero=0 cap_exceeded=0 n=4718592 atol=0.0105 rtol=5e-06 ok=1 gate=0
MB_CHECK output add max_abs_err=0.0487 bound=0.0521 ok=0
MB_SLOT_CHECK convnd rel_err=0.0008 ok=1
failed
"""


def test_sol_lines_are_parsed():
    r = parse_driver_output(SOL_FAIL)
    assert not r.ok
    d = r.sol["add"]
    assert d["gate"] == "sol"
    assert abs(d["sol"]["matched_ratio"] - 0.98612) < 1e-9
    assert d["sol"]["ok"] is False and d["golden"]["ok"] is True
    assert d["sol"]["atol"] == 0.021
    assert r.outputs["add"]["ok"] is False             # the combined verdict the driver printed


def test_error_text_names_sols_rule_when_the_flow_bound_passed():
    txt = parse_driver_output(SOL_FAIL).error_text()
    assert "SOL-ExecBench" in txt and "0.98612" in txt and "0.99000" in txt
    assert "TF32" in txt


def test_error_text_unchanged_when_the_flow_bound_failed():
    log = SOL_FAIL.replace("max_abs_err=0.0487 bound=0.0521", "max_abs_err=0.09 bound=0.0521")
    txt = parse_driver_output(log).error_text()
    assert "max_abs_err=0.09 > bound=0.0521" in txt
    assert "SOL-ExecBench" not in txt


def test_ungated_run_is_what_it_was():
    r = parse_driver_output("MB_CHECK output add max_abs_err=0 bound=1 ok=1\nMB_SLOT_CHECK convnd rel_err=0 ok=1\npassed\n")
    assert r.ok and r.sol == {}
    assert r.error_text() == r.log[-3000:] or r.error_text() == ""


def test_failed_check_of_any_pass_stands():
    first = SOL_FAIL.splitlines()[0]
    ok_again = first.replace("ok=0", "ok=1").replace("0.986120", "1.000000")
    r = parse_driver_output(first + "\n" + ok_again + "\nfailed\n")
    assert r.sol["add"]["sol"]["ok"] is False
