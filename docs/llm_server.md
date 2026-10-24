# LLM relay server

The agents' LLM calls can be made on another machine. Use this when the provider API is only reachable from that
machine (VPN, network policy) or when the API keys should only live there.

```
agent machine (KB / MB, no keys)                API machine (keys, VPN)
generate_code ──HTTP──> 127.0.0.1:18080 ══ssh -L══> 127.0.0.1:18081  llm_server ──> Bedrock / OpenAI / ...
```

Every LLM call in KernelBlaster goes through `query.generate_code`. That covers the multi-slot agent's codegen and
fix calls, reflection, knowledge-v2 upkeep, the legacy agents, and ModelBlaster's graph optimiser (which runs KB).
When `KB_LLM_SERVER_URL` is set, `generate_code` sends the request to the relay server. The server runs the normal
provider dispatch (`generate_code_direct`), so Bedrock, OpenAI, perflab, local models and `claude-code:` all work
there unchanged. `stub:` models (tests) always run in the client process.

| piece | file |
|---|---|
| server | `src/kernelblaster/servers/llm_server.py` |
| client | `src/kernelblaster/agents/utils/llm_relay.py` (called from `query.generate_code`) |
| tests | `tests/test_llm_server.py` |

## Setup

### On the API machine

```bash
cd KernelBlaster
export AWS_BEARER_TOKEN_BEDROCK=...        # or OPENAI_API_KEY, AWS creds, ...; only in this environment
export BEDROCK_MAX_TOKENS=16384            # provider knobs are read here, not on the agent machine
PYTHONPATH=src python -m kernelblaster.servers.llm_server --write-token ~/.kb_llm_token   # once, mode 0600
PYTHONPATH=src python -m kernelblaster.servers.llm_server --port 18081 --token-file ~/.kb_llm_token \
    --max-concurrency 8 --allow-models 'us\.anthropic\.claude-.*' \
    --ledger ~/llm_ledger.jsonl --budget-usd 20 --log-file ~/llm_server.log
```

Copy the token file to the agent machine over a secure channel (`scp`), with mode 0600.

| flag | default | |
|---|---|---|
| `--host` | `127.0.0.1` | bind address. A non-loopback host needs a token |
| `--port` | 18081 | |
| `--token-file F` / env `KB_LLM_SERVER_TOKEN` | | shared secret, at least 16 characters |
| `--no-auth` | off | run without a token. Loopback only, and any local user can then use the server |
| `--write-token F` | | write a new random token to F (0600) and exit. The token is never printed |
| `--max-concurrency K` | 8 | concurrent provider calls; the rest wait in a queue served round robin per client |
| `--max-queue N` | 256 | queued requests before `503 queue_full` (the client retries) |
| `--allow-models RE` | all | full-match regex of the model ids the server will call (`403 model_not_allowed` otherwise) |
| `--ledger PATH` | | one JSONL row per provider call (format below) |
| `--ledger-who W` | `llm-server` | the rows' `who`; also the budget's scope |
| `--budget-usd X` | | refuse new calls (`402 budget_exceeded`) once the ledger's `who=W` rows add up to X |
| `--idempotency-ttl S` | 600 | how long a result is kept for a retried `request_id` |
| `--throttle-retries N` | 4 | server-side back-off retries on provider throttling (1, 2, 4, … s, capped at 60 s) |
| `--max-request-mb M` | 8 | request body limit (`413`) |
| `--log-file F` | stderr only | structured request log (JSON lines) |
| `--log-messages` | off | also log prompts and generations |
| `--ssl-certfile`, `--ssl-keyfile` | | serve HTTPS directly (see the direct option below) |

### On the agent machine: ssh tunnel (recommended)

```bash
ssh -N -L 18080:127.0.0.1:18081 user@api-machine          # keep running; autossh below
export KB_LLM_SERVER_URL=http://127.0.0.1:18080
export KB_LLM_SERVER_TOKEN_FILE=~/.kb_llm_token
python scripts/run_multislot.py ... --model us.anthropic.claude-haiku-5-5     # unchanged otherwise
```

To restart the tunnel automatically when it drops:

```bash
autossh -M 0 -f -N -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
    -L 18080:127.0.0.1:18081 user@api-machine
```

If only the API machine can open ssh connections (for example, it sits behind the VPN), reverse the tunnel. Run
this on the API machine: `ssh -N -R 18080:127.0.0.1:18081 user@agent-machine`. The agent side then uses
`http://127.0.0.1:18080` in the same way.

### Direct HTTPS with a token

When a tunnel is impractical, bind a reachable address with TLS:

```bash
python -m kernelblaster.servers.llm_server --host 0.0.0.0 --port 18443 --token-file ~/.kb_llm_token \
    --ssl-certfile cert.pem --ssl-keyfile key.pem
# agent: KB_LLM_SERVER_URL=https://api-machine:18443  (KB_LLM_SERVER_CA_FILE=ca.pem for a private CA)
```

The server refuses a non-loopback bind without a token. Without TLS it warns, because the token would then cross
the network in clear text.

### Client environment

| variable | |
|---|---|
| `KB_LLM_SERVER_URL` | turns the relay on (`http://127.0.0.1:18080`, `https://host:port`) |
| `KB_LLM_SERVER_TOKEN` / `KB_LLM_SERVER_TOKEN_FILE` | the shared secret |
| `KB_LLM_SERVER_FALLBACK=direct` | call the providers from this process if the server is unreachable. Off by default. This needs local keys. After a fallback the client goes direct for 60 s before it tries the server again. A bad token or an exhausted budget never falls back |
| `KB_LLM_SERVER_TIMEOUT_S` | per-attempt timeout (default 1800, the provider timeout) |
| `KB_LLM_SERVER_RETRIES` | attempts on connection errors and 5xx, all with the same `request_id` (default 6, back-off 1, 2, 4, … 30 s) |
| `KB_LLM_SERVER_CLIENT` | client name for fairness and logs (default `host:pid`) |
| `KB_LLM_SERVER_CA_FILE` | CA bundle for an HTTPS server with a private certificate |

ModelBlaster passes the environment through to KernelBlaster, so the same variables relay its graph optimiser's
calls. Nothing in `run_multislot.py` or MB needs a flag.

## Protocol

- `POST /v1/generate`: `{messages, n_tasks, model, role, client, run_id, request_id}`. The response holds
  `generations`, `usage` (the provider's usage dict, including its `telemetry` extras), `model`, `num_tasks` and
  `elapsed_time`, plus `server: {queue_s, provider_s, backoff_s, throttle_retries, total_s, usd, replayed}`.
- `GET /v1/health`: providers configured (booleans), allowed models, concurrency (active, queued, per client),
  budget (spent, remaining) and the idempotency cache size. No secrets.
- `GET /v1/stats`: requests, ok, errors, replays, rejections, tokens, USD, provider and queue seconds, per client
  and per model.

| status | `error` | client behaviour |
|---|---|---|
| 401 | `unauthorized` | `RelayAuthError`, not retried |
| 402 | `budget_exceeded` | `RelayBudgetExceeded`, not retried |
| 403 | `model_not_allowed` | `RelayModelNotAllowed`, not retried |
| 400 / 409 / 413 | `bad_request` / `request_id_conflict` / `too_large` | `RelayFatalError`, not retried |
| 429 | `rate_limited` (throttled even after the server's back-off) | `generate_code_retry`'s rate-limit back-off |
| 502 | `provider_error` (`message` carries the provider's error type and text, with key values scrubbed) | `generate_code_retry`, as for a local provider error |
| 503 / other 5xx / connection error / timeout | `queue_full`, ... | retried by the relay with the same `request_id`, then `RelayUnavailable` (or the fallback) |

`generate_code_retry` re-raises the `RelayFatalError` subclasses immediately, so a run stops at once on a bad token or
an exhausted budget.

### Idempotency

The client sends a fresh `request_id` per `generate_code` call and reuses it on its own retries. The server keeps
the result for `--idempotency-ttl` seconds. A retry after a dropped connection gets the stored result
(`server.replayed`), or waits for the call still in flight; it is not billed twice. The provider call runs as its
own task, so a client that disconnects does not cancel it.

A failed call is not stored, so a retry runs it again. A `request_id` reused with a different payload gets `409`.

## Accounting and telemetry

The client builds the `LLMResponse` from the server's reply, and the same `usage` dict the provider returned
locally. Therefore:

- `llm_usage` and `llm_usage_by_model` are unchanged.
- `telemetry.jsonl` gets the same record a direct call would produce: tokens incl. cache read/write, stop reason,
  truncation, the provider's own latency, provider retries and cost.
- Each call is recorded once, on the client. The server does not write telemetry.

The relay adds to the record:

| field | |
|---|---|
| `relay_queue_s` | time waiting for a server slot |
| `relay_network_s` | round trip minus server time (the tunnel and serialisation) |
| `backoff_s`, `backoffs` | the server's throttling back-off, counted like `generate_code_retry`'s |
| `relay_replayed` | the result came from the idempotency cache |

`latency_s` is still the full time the caller waited. `scripts/telemetry_report.py` prints the relay's queue and
network totals.

### Ledger and budget

Ledger rows use the shared `llm_budget` ledger format, one per provider call:

```json
{"ts": "2026-10-24T15:02:11", "who": "llm-server", "what": "llm-server role=codegen_simple client=host:123 run=...",
 "model": "us.anthropic.claude-haiku-5-5", "calls": 1, "prompt_tokens": 5210, "completion_tokens": 1630,
 "cache_read_input_tokens": 0, "cache_write_input_tokens": 0, "est_usd": 0.01336, "request_id": "..."}
```

- **Tokens and cost.** `prompt_tokens` counts all input tokens, cached ones included. `est_usd` is the provider's
  reported cost if there is one, else the `cost_tracker` price estimate. It is the same number as the client's
  telemetry `cost_usd` or `est_usd`.
- **Budget scope.** The budget counts every row with `who = --ledger-who`, including rows from earlier server runs,
  so a restart does not reset it. For a fresh budget, use a new `--ledger-who` or a separate ledger file.
- **Overshoot.** The check happens before each call. Calls already running can therefore overshoot the budget by
  up to `--max-concurrency` calls.

## Security model

- **Keys stay on the API machine.** Only the server's environment holds provider keys. The client needs only the
  URL and the token.
- **Shared-secret bearer token.** The server compares it in constant time and checks it before it reads the request
  body. The token is never logged or returned. `--write-token` creates the file with mode 0600, and the server warns
  when a token file is group- or world-readable.
- **Loopback by default.** The intended transport is an ssh tunnel, which gives encryption and host authentication.
  A token is still required on loopback (unless `--no-auth`), because on a shared machine every local user can
  connect to 127.0.0.1. The same holds for the tunnel's local port on the agent machine.
- **Non-loopback binds** need a token. Use TLS for them.
- **Nothing secret in responses or logs.** Provider error messages are scrubbed of the values of
  `AWS_BEARER_TOKEN_BEDROCK`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `OPENAI_API_KEY`, `PERFLAB_KEY`,
  `ANTHROPIC_API_KEY` and the token before they are returned. `/v1/health` reports providers as booleans.
- **Request log.** It records ids, client, role, model, tokens, cost and timings. It has no prompts or generations
  unless `--log-messages`.
- **Limits on what the server will do.** `--allow-models` limits which models are called. The request size limit,
  the queue limit and the budget bound how much a client can spend or load the server.

## Troubleshooting

| symptom | check |
|---|---|
| `RelayUnavailable: ... unreachable (ClientConnectorError ...)` | Is the server running (`curl -H "Authorization: Bearer $(cat ~/.kb_llm_token)" http://127.0.0.1:18080/v1/health`)? Is the tunnel up (`ss -ltn \| grep 18080` on the agent machine)? Is the server bound to the port the tunnel forwards to? |
| `RelayAuthError: HTTP 401` | the client token differs from the server's (`KB_LLM_SERVER_TOKEN` takes precedence over `..._TOKEN_FILE`); a stray newline is stripped on both sides |
| `RelayModelNotAllowed` | `--allow-models` on the server; the regex must match the whole model id |
| `RelayBudgetExceeded` | `curl .../v1/health` shows `budget`; raise `--budget-usd` or start a new `--ledger-who` |
| `provider_error ... no bedrock-runtime client is configured` | the key is not in the server's environment (`/v1/health` → `providers.bedrock: false`) |
| outputs truncated at 8192 tokens | `BEDROCK_MAX_TOKENS` is read by the server; set it there |
| `llm_server: refusing to bind non-loopback host` | add `--token-file`, or bind 127.0.0.1 and use a tunnel |
| `llm_server: no token configured` | `--token-file`, `KB_LLM_SERVER_TOKEN`, or `--no-auth` (loopback only) |
| long queue waits (`relay_queue_s` in the telemetry report) | raise `--max-concurrency` (watch provider throttling: `throttle` events in the server log) |
| requests hang with an ssh tunnel | add `-o ServerAliveInterval=30` (or autossh). The client retries dropped connections with the same `request_id`, so nothing is billed twice |
