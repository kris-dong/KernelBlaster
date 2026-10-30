"""run_multislot.py --plugin for the beam-search tests (tests/test_beam.py).

The resume plugin (stub LLM ``stub:resume``, fake evaluator, ``$KB_FAULT`` fault injection), with an
evaluator that also reports an end-to-end time (partition time + a fixed launch overhead), so the
default ``--reward e2e`` decides on it.
"""
import pathlib
import runpy

_R = runpy.run_path(str(pathlib.Path(__file__).with_name("resume_plugin.py")), run_name="kb_plugin_resume")


class E2EEvaluator(_R["FakeEvaluator"]):
    async def evaluate(self, problem, sources, tag, **kw):
        res = await super().evaluate(problem, sources, tag, **kw)
        if res.ok:
            res.e2e_us, res.e2e_basis = round(res.partition_us + 3.0, 6), "eager"
        return res


def kb_evaluator(target, args, out):
    return E2EEvaluator()
