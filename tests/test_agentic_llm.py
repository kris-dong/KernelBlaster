import asyncio

from kernelblaster.agents.utils import agentic_llm
from kernelblaster.agents.utils.query import generate_code


def test_stub_provider_routes_and_counts():
    seen = []
    agentic_llm.register_stub("echo", lambda msgs: (seen.append(msgs), "```cpp\nint x;\n```")[1])
    try:
        r = asyncio.run(generate_code([{"role": "system", "content": "s"},
                                       {"role": "user", "content": "u"}], n_tasks=2, model="stub:echo"))
    finally:
        agentic_llm.unregister_stub("echo")
    assert r.generations == ["```cpp\nint x;\n```"] * 2 and len(seen) == 2


def test_async_stub_and_message_split():
    async def fn(msgs):
        return msgs[-1]["content"].upper()
    agentic_llm.register_stub("up", fn)
    try:
        r = asyncio.run(generate_code([{"role": "user", "content": "hi"}], model="stub:up"))
    finally:
        agentic_llm.unregister_stub("up")
    assert r.response == "HI"
    sys_, body = agentic_llm._split_messages([{"role": "system", "content": "S"},
                                              {"role": "user", "content": "a"},
                                              {"role": "assistant", "content": "b"}])
    assert sys_ == "S" and "[USER]\na" in body and "[ASSISTANT]\nb" in body
