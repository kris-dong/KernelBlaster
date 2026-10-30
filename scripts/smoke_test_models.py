"""Standalone smoke test for the optimised RL flow's configured models.

Designed to be run as a SUBPROCESS from ``scripts/run_opt_ncu_rl.sh``
*before* the main runner. Living in its own process means the shell's
``timeout`` command can OS-kill it cleanly when the underlying Anthropic /
Bedrock SDK does blocking I/O that doesn't honour ``asyncio.wait_for``.

Usage
-----

    python scripts/smoke_test_models.py model1 model2 ...        # explicit
    python scripts/smoke_test_models.py --from-env               # MODEL_*
    timeout 60 python scripts/smoke_test_models.py --from-env    # bounded

Exit codes
----------

    0 — all probed models replied (strict echo or non-empty plausible reply)
    1 — at least one model failed (timeout, error, empty)
    2 — argument / config problem (no models given)
   124 — overall timeout from the shell's ``timeout`` (probe was killed mid-call)

The script prints one human-readable + machine-greppable line per model,
prefixed with ``[smoke]``::

    [smoke] PASS anthropic.claude-opus-4-6-v1   (1.4s)  OK strict (10 chars)
    [smoke] FAIL anthropic.claude-sonnet-4-6   (15.0s) timeout 15s
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# SSL cert bootstrap (mirrors run_opt_ncu_rl_optimized.py).
if not os.environ.get("SSL_CERT_FILE") and not os.environ.get("REQUESTS_CA_BUNDLE"):
    try:
        import certifi
        _cert = certifi.where()
        if _cert and Path(_cert).exists():
            os.environ["SSL_CERT_FILE"] = _cert
            os.environ.setdefault("REQUESTS_CA_BUNDLE", _cert)
    except Exception:
        pass

from src.kernelblaster.agents.utils import generate_code_retry  # noqa: E402


_PROMPT = (
    "Reply with EXACTLY this token sequence and nothing else: SMOKE_OK_42. "
    "Do not add quotes, punctuation, code blocks, or commentary."
)


async def probe(model: str, timeout_s: int) -> tuple[bool, str]:
    """Probe one model. Returns ``(ok, info_string)``."""
    try:
        resp = await asyncio.wait_for(
            generate_code_retry(
                messages=[{"role": "user", "content": _PROMPT}],
                model=model,
                max_retries=1,
            ),
            timeout=timeout_s,
        )
        text = (resp.generations[0] if resp.generations else "").strip()
        usage = getattr(resp, "usage", None) or {}
        if "SMOKE_OK_42" in text:
            return True, f"OK strict ({len(text)} chars, usage={usage})"
        if len(text) >= 5 and any(c.isalnum() for c in text):
            return True, f"OK loose: {text[:40]!r}"
        return False, f"empty / unparseable: {text[:80]!r}"
    except asyncio.TimeoutError:
        return False, f"timeout after {timeout_s}s"
    except Exception as e:  # pragma: no cover — surface anything provider-side
        return False, f"{type(e).__name__}: {e}"


async def main_async(models: list[str], timeout_s: int) -> bool:
    # Dedupe while preserving order so the user can predict probe order.
    seen: dict[str, None] = {}
    for m in models:
        if m and m not in seen:
            seen[m] = None
    deduped = list(seen.keys())

    print(
        f"[smoke] Testing {len(deduped)} unique model(s) sequentially "
        f"(per-model timeout {timeout_s}s).",
        flush=True,
    )

    all_ok = True
    for m in deduped:
        print(f"[smoke] → probing {m}", flush=True)
        t0 = time.time()
        ok, info = await probe(m, timeout_s)
        elapsed = time.time() - t0
        marker = "PASS" if ok else "FAIL"
        print(f"[smoke] {marker} {m}   ({elapsed:.1f}s)  {info}", flush=True)
        if not ok:
            all_ok = False
    return all_ok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help="Model IDs to probe.")
    p.add_argument(
        "--from-env",
        action="store_true",
        help="Pick up models from MODEL_PLAN / MODEL_CODEGEN_SIMPLE / "
             "MODEL_CODEGEN_HARD / MODEL_FIX env vars (in addition to "
             "any positional args).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("SMOKE_TEST_TIMEOUT", "30")),
        help="Per-model timeout in seconds (default: $SMOKE_TEST_TIMEOUT or 30).",
    )
    args = p.parse_args()

    models = list(args.models)
    if args.from_env:
        for v in ("MODEL_PLAN", "MODEL_CODEGEN_SIMPLE", "MODEL_CODEGEN_HARD", "MODEL_FIX"):
            m = os.environ.get(v)
            if m:
                models.append(m)

    if not models:
        print(
            "[smoke] no models given — pass positional args or --from-env "
            "(with the MODEL_* env vars exported)",
            file=sys.stderr,
        )
        return 2

    ok = asyncio.run(main_async(models, args.timeout))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
