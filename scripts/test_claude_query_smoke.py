#!/usr/bin/env python3
"""Smoke-test Claude via KernelBlaster ``query`` (direct API or Amazon Bedrock).

Run from the repo root::

    python scripts/test_claude_query_smoke.py
    python scripts/test_claude_query_smoke.py --retry

Environment (see query.py): ANTHROPIC_API_KEY and/or AWS keys, MODEL or
ANTHROPIC_SMOKE_MODEL, optional ANTHROPIC_BEDROCK=1.

Exit codes: 0 success, 1 failure, 2 skipped (no client configured).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


async def _main_async(args: argparse.Namespace) -> int:
    _ensure_repo_root_on_path()
    from src.kernelblaster.agents.utils.claude_smoke_agent import (
        ClaudeQuerySmokeAgent,
        assert_smoke_response,
        claude_query_configured,
    )

    if not claude_query_configured():
        print(
            "SKIP: No Anthropic/Bedrock client (set ANTHROPIC_API_KEY or AWS credentials).",
            file=sys.stderr,
        )
        return 2

    agent = ClaudeQuerySmokeAgent(model=args.model or None)
    model = agent.resolved_model()
    print(f"Using model: {model!r}")

    if args.retry:
        resp = await agent.call_with_retry(max_retries=args.max_retries)
    else:
        resp = await agent.call_once(n_tasks=args.n_tasks)

    print("--- raw first generation ---")
    print(resp.response)
    print("--- usage ---")
    print(resp.usage)

    assert_smoke_response(resp)
    print("OK: smoke assertion passed.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=None,
        help="Override model id (else ANTHROPIC_SMOKE_MODEL, MODEL, or defaults).",
    )
    p.add_argument(
        "--retry",
        action="store_true",
        help="Use generate_code_retry instead of a single generate_code call.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Only with --retry: max attempts (default 3).",
    )
    p.add_argument(
        "--n-tasks",
        type=int,
        default=1,
        help="Number of completions (default 1).",
    )
    args = p.parse_args()

    try:
        code = asyncio.run(_main_async(args))
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(code)


if __name__ == "__main__":
    main()
