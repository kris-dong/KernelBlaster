#!/bin/bash
# Live watcher for an in-progress / completed scripts/run_opt_ncu_rl_optimized.py
# run. Reads the file-only progress + cost outputs (no logger spam) and
# renders them as a periodically-refreshed dashboard, or tails the event
# stream as it grows.
#
# Modes:
#   ./scripts/watch_opt_ncu_rl.sh              # auto-detect newest run, dashboard
#   ./scripts/watch_opt_ncu_rl.sh <out_root>   # specific run dir (the one printed
#                                              # at startup as "Output root")
#   ./scripts/watch_opt_ncu_rl.sh -e <out_root>  # tail progress.jsonl events
#   ./scripts/watch_opt_ncu_rl.sh -c <out_root>  # tail cost_live.jsonl
#
# With no out_root the watcher walks ``out/kernelbench-cuda/`` and picks the
# subdirectory whose ``progress.json`` has the most recent mtime. Override the
# search root with ``OPT_RL_OUT_ROOT_BASE`` if you've moved outputs elsewhere.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SEARCH_BASE="${OPT_RL_OUT_ROOT_BASE:-$ROOT_DIR/out/kernelbench-cuda}"

REFRESH_S="${REFRESH_S:-3}"
MODE="dashboard"

usage() {
    cat <<USAGE
Usage: $0 [-e | -c] [<out_root>]

  (no flag)   live dashboard (totals + per-problem table) — refreshes every
              \$REFRESH_S seconds (default 3)
  -e          tail progress.jsonl events as they arrive
  -c          tail cost_live.jsonl ticks as they arrive
  <out_root>  the run directory printed by the runner as "Output root".
              If omitted, the most recently-active run under
              \$OPT_RL_OUT_ROOT_BASE (default: out/kernelbench-cuda) is picked.

Hotkeys in dashboard mode: Ctrl-C to exit.
USAGE
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help) usage ;;
        -e|--events) MODE="events"; shift ;;
        -c|--cost-events) MODE="cost_events"; shift ;;
        --refresh) REFRESH_S="$2"; shift 2 ;;
        --) shift; break ;;
        -*) echo "unknown flag: $1" >&2; usage ;;
        *) break ;;
    esac
done

OUT_ROOT="${1:-}"

if [ -z "$OUT_ROOT" ]; then
    if [ ! -d "$SEARCH_BASE" ]; then
        echo "No <out_root> given and search base $SEARCH_BASE doesn't exist." >&2
        exit 2
    fi
    # Pick the most recently-touched progress.json.
    OUT_ROOT="$(
        find "$SEARCH_BASE" -name 'progress.json' -printf '%T@ %h\n' 2>/dev/null \
        | sort -nr | head -n1 | awk '{print $2}'
    )"
    if [ -z "$OUT_ROOT" ]; then
        echo "No progress.json found under $SEARCH_BASE." >&2
        echo "Pass an explicit <out_root> or wait for the runner to start." >&2
        exit 2
    fi
    echo "→ Auto-detected newest run: $OUT_ROOT" >&2
fi

PROGRESS_JSON="$OUT_ROOT/progress.json"
PROGRESS_JSONL="$OUT_ROOT/progress.jsonl"
COST_JSON="$OUT_ROOT/cost_live.json"
COST_JSONL="$OUT_ROOT/cost_live.jsonl"

if ! command -v jq >/dev/null 2>&1; then
    echo "jq not found in PATH — install it for nicely-formatted output." >&2
    exit 3
fi

case "$MODE" in
    events)
        if [ ! -e "$PROGRESS_JSONL" ]; then
            echo "Waiting for $PROGRESS_JSONL to be created…"
            while [ ! -e "$PROGRESS_JSONL" ]; do sleep 1; done
        fi
        echo "Tailing events from $PROGRESS_JSONL (Ctrl-C to stop)"
        # Compact one-liner per event for parsing/grepping:
        #   <local-time>  <type>  <problem>  [step=<n> traj=<n> tech=<name> cycles=<n> impr=<n>%]
        tail -n +1 -F "$PROGRESS_JSONL" 2>/dev/null \
        | jq -r --unbuffered '
            (.ts | strftime("%H:%M:%S")) as $t
            | if .type == "problem_started" then
                "\($t)  START   \(.problem_id)  init_cycles=\(.init_cycles)"
              elif .type == "step_done" then
                "\($t)  STEP    \(.problem_id)  traj=\(.traj_idx) step=\(.step_idx) tech=\(.technique) cycles=\(.cycles) impr=\(.improvement_pct | (if . == null then "n/a" else (. * 100 | round | . / 100 | tostring) + "%" end))"
              elif .type == "problem_finished" then
                "\($t)  DONE    \(.problem_id)  status=\(.status) final_cycles=\(.final_cycles)"
              else
                "\($t)  \(.type)  \(. | tostring)"
              end
          '
        ;;

    cost_events)
        if [ ! -e "$COST_JSONL" ]; then
            echo "Waiting for $COST_JSONL to be created…"
            while [ ! -e "$COST_JSONL" ]; do sleep 1; done
        fi
        echo "Tailing cost ticks from $COST_JSONL (Ctrl-C to stop)"
        tail -n +1 -F "$COST_JSONL" 2>/dev/null \
        | jq -r --unbuffered '
            (.ts | strftime("%H:%M:%S")) as $t
            | "\($t)  calls=\(.totals.calls)  in=\(.totals.input_tokens)  out=\(.totals.output_tokens)  cost=$\(.totals.cost_usd | . * 1000 | round / 1000)"
          '
        ;;

    dashboard)
        # Loop until interrupted, redrawing the whole screen each tick.
        trap 'tput cnorm; exit 0' INT TERM
        tput civis 2>/dev/null || true   # hide cursor for a less twitchy redraw
        while true; do
            clear
            echo "════════════════════════════════════════════════════════════════════════"
            echo "  Optimised RL CUDA flow — watcher"
            echo "  out_root:  $OUT_ROOT"
            echo "  refresh:   ${REFRESH_S}s   (Ctrl-C to exit)"
            echo "════════════════════════════════════════════════════════════════════════"

            if [ -e "$PROGRESS_JSON" ]; then
                jq -r '
                    (.ts | strftime("%Y-%m-%d %H:%M:%S")) as $t
                    | "  Last update: \($t)",
                      "",
                      "  PROBLEMS (totals):",
                      "    total      : \(.totals.problems_total)",
                      "    running    : \(.totals.problems_running)",
                      "    succeeded  : \(.totals.problems_succeeded)",
                      "    failed     : \(.totals.problems_failed)",
                      ""
                ' "$PROGRESS_JSON" 2>/dev/null || echo "  (progress.json not yet parseable)"

                # Per-problem one-liner table
                echo "  PER-PROBLEM:"
                printf "    %-50s  %-9s  %12s  %12s  %8s  %5s\n" "id" "status" "init_cycles" "best_cycles" "improv%" "steps"
                printf "    %-50s  %-9s  %12s  %12s  %8s  %5s\n" "$(printf '%.0s-' {1..50})" "$(printf '%.0s-' {1..9})" "------------" "------------" "--------" "-----"
                jq -r '
                    .problems
                    | to_entries
                    | sort_by(.key)
                    | map([
                        .key,
                        (.value.status // "?"),
                        (.value.init_cycles | tostring),
                        (.value.best_cycles | tostring),
                        (if .value.improvement_pct == null then "n/a" else (.value.improvement_pct * 100 | round / 100 | tostring) + "%" end),
                        (.value.step_count | tostring)
                      ])
                    | .[]
                    | @tsv
                ' "$PROGRESS_JSON" 2>/dev/null \
                | awk -F'\t' '{ printf "    %-50s  %-9s  %12s  %12s  %8s  %5s\n", $1, $2, $3, $4, $5, $6 }'
                echo
            else
                echo "  (waiting for $PROGRESS_JSON to appear…)"
                echo
            fi

            if [ -e "$COST_JSON" ]; then
                jq -r '
                    "  COST (live):",
                    "    calls      : \(.totals.calls)",
                    "    input tok  : \(.totals.input_tokens)",
                    "    output tok : \(.totals.output_tokens)",
                    "    cost USD   : $\(.totals.cost_usd | . * 1000 | round / 1000)",
                    "    elapsed s  : \(.totals.elapsed_s | round)",
                    ""
                ' "$COST_JSON" 2>/dev/null

                # Per-role breakdown (one-liner table)
                echo "  COST by role:"
                jq -r '
                    .by_role
                    | to_entries
                    | sort_by(.key)
                    | map([
                        .key,
                        (.value.calls | tostring),
                        (.value.input_tokens | tostring),
                        (.value.output_tokens | tostring),
                        (.value.cost_usd | . * 1000 | round / 1000 | tostring)
                      ])
                    | .[]
                    | @tsv
                ' "$COST_JSON" 2>/dev/null \
                | awk -F'\t' 'BEGIN { printf "    %-22s  %5s  %12s  %12s  %10s\n", "role", "calls", "in_tok", "out_tok", "cost_usd" } { printf "    %-22s  %5s  %12s  %12s  $%9s\n", $1, $2, $3, $4, $5 }'
                echo
                echo "  COST by model:"
                jq -r '
                    .by_model
                    | to_entries
                    | sort_by(.key)
                    | map([
                        .key,
                        (.value.calls | tostring),
                        (.value.input_tokens | tostring),
                        (.value.output_tokens | tostring),
                        (.value.cost_usd | . * 1000 | round / 1000 | tostring)
                      ])
                    | .[]
                    | @tsv
                ' "$COST_JSON" 2>/dev/null \
                | awk -F'\t' 'BEGIN { printf "    %-32s  %5s  %12s  %12s  %10s\n", "model", "calls", "in_tok", "out_tok", "cost_usd" } { printf "    %-32s  %5s  %12s  %12s  $%9s\n", $1, $2, $3, $4, $5 }'
            else
                echo "  (cost_live.json not yet present — cost tracking may be disabled)"
            fi

            sleep "$REFRESH_S"
        done
        ;;
esac
