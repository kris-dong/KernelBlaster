#!/bin/bash
# Ncurses-style live dashboard for an in-progress run of
# scripts/run_opt_ncu_rl_optimized.py.
#
# Reads the file-only telemetry (no log scraping) — progress.json,
# progress.jsonl, cost_live.json — and renders a refreshing TUI with:
#   • Run summary (counts, problem progress %)
#   • Live token / cost meter (totals + by role)
#   • Active problems with per-problem progress bars
#   • Recent events tail (last 10)
#   • Global performance leaders (best, median, top technique)
#
# Usage:
#   ./scripts/watch_opt_ncu_rl_tui.sh                       # auto-detect newest run
#   ./scripts/watch_opt_ncu_rl_tui.sh <out_root>            # specific run
#   REFRESH_S=1 ./scripts/watch_opt_ncu_rl_tui.sh           # faster refresh
#   NUM_ITERATIONS=20 MAX_STEPS=5 ./scripts/watch_opt_ncu_rl_tui.sh
#                                                           # adjust % math when
#                                                           # the runner uses
#                                                           # non-default knobs
#
# Hotkey: Ctrl-C to exit cleanly.

# Intentionally NOT using ``set -u`` / ``set -e`` / ``set -o pipefail``:
# this is a long-running display loop that polls files which may be
# transiently empty / partially written / not yet present. Any single
# render-helper hiccup must not exit the loop. We use ``trap`` to keep
# Ctrl-C clean.
set +e +u +o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SEARCH_BASE="${OPT_RL_OUT_ROOT_BASE:-$ROOT_DIR/out/kernelbench-cuda}"

REFRESH_S="${REFRESH_S:-2}"
# Defaults match scripts/run_opt_ncu_rl.sh — override if the runner uses
# different knobs and you want accurate per-problem percentages.
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
MAX_STEPS="${MAX_STEPS:-5}"
TOTAL_STEPS_PER_PROBLEM=$(( NUM_ITERATIONS * MAX_STEPS ))

OUT_ROOT="${1:-}"

if ! command -v jq >/dev/null 2>&1; then
    echo "jq not found in PATH — install it for the TUI to render." >&2
    exit 3
fi

if [ -z "$OUT_ROOT" ]; then
    if [ ! -d "$SEARCH_BASE" ]; then
        echo "No <out_root> given and search base $SEARCH_BASE doesn't exist." >&2
        exit 2
    fi
    OUT_ROOT="$(
        find "$SEARCH_BASE" -name 'progress.json' -printf '%T@ %h\n' 2>/dev/null \
        | sort -nr | head -n1 | awk '{print $2}'
    )"
    if [ -z "$OUT_ROOT" ]; then
        echo "No progress.json found under $SEARCH_BASE — pass a path or wait." >&2
        exit 2
    fi
fi

PROGRESS_JSON="$OUT_ROOT/progress.json"
PROGRESS_JSONL="$OUT_ROOT/progress.jsonl"
COST_JSON="$OUT_ROOT/cost_live.json"

# ── Colours / Box drawing ───────────────────────────────────────────
C_RESET=$'\033[0m'
C_BOLD=$'\033[1m'
C_DIM=$'\033[2m'
C_RED=$'\033[31m'
C_GREEN=$'\033[32m'
C_YELLOW=$'\033[33m'
C_BLUE=$'\033[34m'
C_MAGENTA=$'\033[35m'
C_CYAN=$'\033[36m'
C_WHITE=$'\033[97m'

# ── Cursor management ───────────────────────────────────────────────
cleanup() {
    # Prevent re-entry when our own ``exit 0`` re-triggers any handler.
    trap - INT TERM
    tput cnorm 2>/dev/null || true
    printf '%s' "$C_RESET"
    tput cup "$(tput lines 2>/dev/null || echo 24)" 0 2>/dev/null || true
    echo
    exit 0
}
# Only trap signals, NOT EXIT — an EXIT trap on a script that has any
# transient nonzero exit (e.g. from a failed jq parse on a half-written
# file) would terminate the watcher after one frame.
trap cleanup INT TERM

tput civis 2>/dev/null || true   # hide cursor for less twitch on redraw

# ── Helper: render a fixed-width progress bar ───────────────────────
# usage: bar PCT WIDTH FILLED_COLOR EMPTY_COLOR
bar() {
    local pct=${1:-0} width=${2:-20} cf=${3:-$C_GREEN} ce=${4:-$C_DIM}
    # Clamp 0..100 (and tolerate floats from jq).
    pct=${pct%.*}
    [ -z "$pct" ] && pct=0
    [ "$pct" -lt 0 ] && pct=0
    [ "$pct" -gt 100 ] && pct=100
    local filled=$(( pct * width / 100 ))
    local empty=$(( width - filled ))
    local fchars=""
    local echars=""
    if [ "$filled" -gt 0 ]; then
        fchars=$(printf '█%.0s' $(seq 1 $filled))
    fi
    if [ "$empty" -gt 0 ]; then
        echars=$(printf '░%.0s' $(seq 1 $empty))
    fi
    printf "%s%s%s%s%s" "$cf" "$fchars" "$ce" "$echars" "$C_RESET"
}

# ── Helper: print a horizontal divider that fits the terminal ───────
divider() {
    local w
    w=$(tput cols 2>/dev/null || echo 80)
    printf "%s" "$C_DIM"
    printf '─%.0s' $(seq 1 $w)
    printf "%s\n" "$C_RESET"
}

# Compact human-readable cycles: 1234567 → 1.23M
human_cycles() {
    local n="${1:-0}"
    [ -z "$n" ] && n=0
    if [ "$n" = "null" ]; then echo "—"; return; fi
    if [ "$n" -ge 1000000000 ] 2>/dev/null; then
        awk -v n="$n" 'BEGIN { printf "%.2fG", n/1e9 }'
    elif [ "$n" -ge 1000000 ] 2>/dev/null; then
        awk -v n="$n" 'BEGIN { printf "%.2fM", n/1e6 }'
    elif [ "$n" -ge 1000 ] 2>/dev/null; then
        awk -v n="$n" 'BEGIN { printf "%.1fk", n/1e3 }'
    else
        echo "$n"
    fi
}

# Fallback init lookup: when progress.json reports init_cycles=null (the writer
# is called via problem_started() *before* the agent profiles init.cu and is
# never re-notified once init succeeds), the canonical baseline is still
# written to disk by ``initialize()`` at:
#   <OUT_ROOT>/<problem_id>/opt_ncu_rl_optimized/ncu/0_init_ncu_log.txt
# Format is one line: "Elapsed Cycles: <N>". Extracting the integer here keeps
# the TUI honest without requiring a writer change. Returns "null" if neither
# the file is missing nor parseable; the caller treats that the same as a
# null progress.json field (renders as —).
read_init_marker() {
    local out_root="$1"
    local problem_id="$2"
    local f="$out_root/$problem_id/opt_ncu_rl_optimized/ncu/0_init_ncu_log.txt"
    if [ ! -f "$f" ]; then echo "null"; return; fi
    local val
    val=$(grep -oE "Elapsed Cycles:[[:space:]]+[0-9]+" "$f" 2>/dev/null \
          | head -n1 | grep -oE "[0-9]+$")
    if [ -z "$val" ] || [ "$val" = "0" ]; then
        echo "null"
    else
        echo "$val"
    fi
}

# Compute integer percent ((init - best) / init * 100), to two decimal places,
# rendered as a string. Returns "0" on missing / invalid inputs (matches the
# default progress.json improvement_pct semantic).
compute_improvement_pct() {
    local init="$1" best="$2"
    if [ -z "$init" ] || [ "$init" = "null" ] || [ "$init" = "0" ]; then echo "0"; return; fi
    if [ -z "$best" ] || [ "$best" = "null" ]; then echo "0"; return; fi
    awk -v i="$init" -v b="$best" \
        'BEGIN { p = (i - b) / i * 100; printf "%.2f", p }'
}

# ── Render passes ───────────────────────────────────────────────────

render_header() {
    local ts tick
    ts="$(date +"%Y-%m-%d %H:%M:%S")"
    # Visual heartbeat that proves the loop is iterating (alternates ●/○).
    if [ $(( FRAME % 2 )) -eq 0 ]; then tick="●"; else tick="○"; fi
    printf "%s%s%-58s%s  %s[#%04d %s]%s  %s%s%s\n" \
        "$C_BOLD" "$C_CYAN" \
        " KernelBlaster RL — Optimisation Dashboard" \
        "$C_RESET" "$C_GREEN" "$FRAME" "$tick" "$C_RESET" \
        "$C_DIM" "$ts" "$C_RESET"
    printf " %srun:%s %s%s%s   %srefresh:%s ${REFRESH_S}s   %sCtrl-C to exit%s\n" \
        "$C_DIM" "$C_RESET" "$C_WHITE" "$OUT_ROOT" "$C_RESET" \
        "$C_DIM" "$C_RESET" "$C_DIM" "$C_RESET"
    divider
}

render_overview() {
    local p_total p_run p_ok p_fail p_nb
    if [ -f "$PROGRESS_JSON" ]; then
        read -r p_total p_run p_ok p_fail p_nb <<< "$(
            jq -r '.totals | "\(.problems_total) \(.problems_running) \(.problems_succeeded) \(.problems_failed) \(.problems_no_baseline // 0)"' \
                "$PROGRESS_JSON" 2>/dev/null
        )"
    fi
    p_total=${p_total:-0} p_run=${p_run:-0} p_ok=${p_ok:-0} p_fail=${p_fail:-0} p_nb=${p_nb:-0}

    local pct=0
    if [ "$p_total" -gt 0 ]; then
        pct=$(( (p_ok + p_fail + p_nb) * 100 / p_total ))
    fi

    local calls in_tok out_tok cost elapsed
    if [ -f "$COST_JSON" ]; then
        read -r calls in_tok out_tok cost elapsed <<< "$(
            jq -r '.totals | "\(.calls) \(.input_tokens) \(.output_tokens) \(.cost_usd) \(.elapsed_s)"' \
                "$COST_JSON" 2>/dev/null
        )"
    fi
    calls=${calls:-0} in_tok=${in_tok:-0} out_tok=${out_tok:-0}
    cost=${cost:-0} elapsed=${elapsed:-0}

    local elapsed_h=$(( ${elapsed%.*} / 3600 ))
    local elapsed_m=$(( (${elapsed%.*} % 3600) / 60 ))
    local elapsed_s=$(( ${elapsed%.*} % 60 ))
    local elapsed_str
    if [ "$elapsed_h" -gt 0 ]; then
        elapsed_str=$(printf "%dh %dm %ds" "$elapsed_h" "$elapsed_m" "$elapsed_s")
    elif [ "$elapsed_m" -gt 0 ]; then
        elapsed_str=$(printf "%dm %ds" "$elapsed_m" "$elapsed_s")
    else
        elapsed_str=$(printf "%ds" "$elapsed_s")
    fi

    printf " %sPROBLEMS%s                                    %sTOKENS / COST%s\n" \
        "$C_BOLD" "$C_RESET" "$C_BOLD" "$C_RESET"
    printf "  total      %s%4d%s                          calls       %s%6d%s\n" \
        "$C_WHITE" "$p_total" "$C_RESET" "$C_WHITE" "$calls" "$C_RESET"
    printf "  %s✓ done     %s%4d%s    %s    in tokens   %s%10d%s\n" \
        "$C_GREEN" "$C_BOLD" "$p_ok" "$C_RESET" "$(bar "$pct" 18 "$C_GREEN")" \
        "$C_WHITE" "$in_tok" "$C_RESET"
    printf "  %s⚙ running  %s%4d%s                          out tokens  %s%10d%s\n" \
        "$C_YELLOW" "$C_BOLD" "$p_run" "$C_RESET" "$C_WHITE" "$out_tok" "$C_RESET"
    printf "  %s✗ failed   %s%4d%s                          %scost USD%s    %s\$%6.3f%s\n" \
        "$C_RED" "$C_BOLD" "$p_fail" "$C_RESET" "$C_BOLD" "$C_RESET" \
        "$C_GREEN" "$cost" "$C_RESET"
    printf "  %s∅ no_base  %s%4d%s                          elapsed     %s%s%s\n" \
        "$C_YELLOW" "$C_BOLD" "$p_nb" "$C_RESET" \
        "$C_WHITE" "$elapsed_str" "$C_RESET"
    divider
}

render_active_problems() {
    printf " %sACTIVE PROBLEMS (running / interrupted)%s\n" "$C_BOLD" "$C_RESET"
    if [ ! -f "$PROGRESS_JSON" ]; then
        printf "   %s(waiting for progress.json)%s\n" "$C_DIM" "$C_RESET"
        divider
        return
    fi

    local active
    active=$(
        jq -r --argjson tot "$TOTAL_STEPS_PER_PROBLEM" '
            .problems
            | to_entries
            | map(select(.value.status == "running" or .value.status == "interrupted"))
            | sort_by(.value.improvement_pct // 0) | reverse
            | map([
                .key,
                (.value.status // "?"),
                (.value.step_count | tostring),
                ((.value.step_count // 0) * 100 / $tot | floor | tostring),
                (.value.init_cycles // null | tostring),
                (.value.best_cycles // null | tostring),
                ((.value.improvement_pct // 0) * 100 | floor | . / 100 | tostring),
                ((.value.trajectories | keys | length) | tostring)
              ])
            | .[]
            | @tsv
        ' "$PROGRESS_JSON" 2>/dev/null
    )
    if [ -z "$active" ]; then
        printf "   %s(no active problems — run not started, all done, or all failed)%s\n" \
            "$C_DIM" "$C_RESET"
        divider
        return
    fi

    while IFS=$'\t' read -r pid status steps pct init_cyc best_cyc impr ntraj; do
        # Truncate problem id to fit
        local short="${pid: -55}"
        local stat_color="$C_YELLOW"
        [ "$status" = "interrupted" ] && stat_color="$C_RED"

        # Recover init_cycles from disk when progress.json hasn't been told yet.
        # The writer's problem_started() is called before the agent profiles
        # init.cu, so init_cyc is null until the run finishes. Read the marker
        # file initialize() writes once profiling succeeds.
        if [ "$init_cyc" = "null" ] || [ "$init_cyc" = "0" ] || [ -z "$init_cyc" ]; then
            init_cyc=$(read_init_marker "$OUT_ROOT" "$pid")
        fi

        # If we recovered init from disk, recompute improvement_pct on the fly
        # (progress.json's improvement_pct is also stale in this case because it
        # depends on init_cycles being set in the snapshot).
        if [ "$init_cyc" != "null" ] && [ "$best_cyc" != "null" ] && [ -n "$best_cyc" ]; then
            impr=$(compute_improvement_pct "$init_cyc" "$best_cyc")
        fi

        local impr_color="$C_GREEN"
        # Negative improvement (regression) is red; positive (or zero) is green.
        case "$impr" in
            -*) impr_color="$C_RED" ;;
        esac

        local init_h best_h
        init_h=$(human_cycles "$init_cyc")
        best_h=$(human_cycles "$best_cyc")

        printf "   %s%-55s%s  %s%-11s%s  trajs=%-3s steps=%-4s %s\n" \
            "$C_WHITE" "$short" "$C_RESET" \
            "$stat_color" "$status" "$C_RESET" \
            "$ntraj" "$steps" \
            "$(bar "$pct" 14 "$C_CYAN")"
        printf "      %sinit:%s %-8s  %sbest:%s %s%-8s%s   %simprovement:%s %s%6s%%%s\n" \
            "$C_DIM" "$C_RESET" "$init_h" \
            "$C_DIM" "$C_RESET" "$C_BOLD" "$best_h" "$C_RESET" \
            "$C_DIM" "$C_RESET" "$impr_color" "$impr" "$C_RESET"
    done <<< "$active"
    divider
}

render_recent_events() {
    printf " %sRECENT EVENTS%s (last 10)\n" "$C_BOLD" "$C_RESET"
    if [ ! -f "$PROGRESS_JSONL" ] || [ ! -s "$PROGRESS_JSONL" ]; then
        printf "   %s(progress.jsonl is empty)%s\n" "$C_DIM" "$C_RESET"
        divider
        return
    fi
    tail -n 30 "$PROGRESS_JSONL" 2>/dev/null \
    | jq -r '
        (.ts | strftime("%H:%M:%S")) as $t
        | if .type == "problem_started" then
            "\($t)\tSTART\t\(.problem_id)\tinit_cycles=\(.init_cycles // "?")"
          elif .type == "step_done" then
            ((.improvement_pct // 0) * 100 | round / 100) as $impr
            | "\($t)\tSTEP\t\(.problem_id)\ttraj=\(.traj_idx) step=\(.step_idx) tech=\(.technique // "?") cycles=\(.cycles // "?") impr=\($impr)%"
          elif .type == "problem_finished" then
            "\($t)\tDONE\t\(.problem_id)\tstatus=\(.status) final=\(.final_cycles // "?")"
          else
            "\($t)\t\(.type)\t\(.problem_id // "")\t"
          end
    ' 2>/dev/null \
    | tail -n 10 \
    | while IFS=$'\t' read -r ts kind pid rest; do
        local color="$C_DIM"
        case "$kind" in
            START) color="$C_BLUE" ;;
            STEP)  color="$C_YELLOW" ;;
            DONE)  color="$C_GREEN" ;;
        esac
        # Color regressions in step lines red.
        local rest_colored="$rest"
        case "$rest" in
            *"impr=-"*) rest_colored="${C_RED}${rest}${C_RESET}" ;;
        esac
        printf "   %s%s  %s%-5s%s  %s%-50s%s  %s\n" \
            "$C_DIM" "$ts" \
            "$color" "$kind" "$C_RESET" \
            "$C_WHITE" "${pid: -50}" "$C_RESET" \
            "$rest_colored"
    done
    divider
}

render_perf_leaders() {
    printf " %sPERFORMANCE LEADERS%s\n" "$C_BOLD" "$C_RESET"

    if [ ! -f "$PROGRESS_JSON" ]; then
        printf "   %s(waiting for data)%s\n" "$C_DIM" "$C_RESET"
        divider
        return
    fi

    # Best improvement so far across all completed problems.
    local best_line
    best_line=$(
        jq -r '
            .problems
            | to_entries
            | map(select(.value.status == "success" and .value.improvement_pct != null))
            | sort_by(.value.improvement_pct) | reverse
            | .[0]
            | if . == null then "—\t0\t0\t0"
              else "\(.key)\t\(.value.init_cycles)\t\(.value.best_cycles)\t\(((.value.improvement_pct // 0) * 100 | round / 100))"
              end
        ' "$PROGRESS_JSON" 2>/dev/null
    )
    IFS=$'\t' read -r bp bi bb bimpr <<< "$best_line"

    # Median improvement across all completed problems.
    local median
    median=$(
        jq -r '
            [ .problems | to_entries[]
              | select(.value.status == "success" and .value.improvement_pct != null)
              | .value.improvement_pct ]
            | if length == 0 then 0
              else (sort | if length % 2 == 1 then .[length/2|floor]
                          else (.[length/2 - 1] + .[length/2]) / 2 end)
              end
            | (. * 100 | round / 100)
        ' "$PROGRESS_JSON" 2>/dev/null
    )

    # Top technique by win count (counts step_done events with positive improvement).
    local top_tech
    if [ -f "$PROGRESS_JSONL" ] && [ -s "$PROGRESS_JSONL" ]; then
        top_tech=$(
            jq -r 'select(.type == "step_done" and (.improvement_pct // 0) > 0) | .technique // "?"' \
                "$PROGRESS_JSONL" 2>/dev/null \
            | sort | uniq -c | sort -rn | head -n1 \
            | awk '{ printf "%s (%d wins)", $2, $1 }'
        )
    fi
    [ -z "$top_tech" ] && top_tech="—"

    local bi_h bb_h
    bi_h=$(human_cycles "$bi")
    bb_h=$(human_cycles "$bb")

    printf "   best improvement   %s%s%-50s%s  %s%6s%%%s  (%s → %s)\n" \
        "$C_WHITE" "$C_BOLD" "${bp: -50}" "$C_RESET" \
        "$C_GREEN" "$bimpr" "$C_RESET" \
        "$bi_h" "$bb_h"
    printf "   median improvement %s%6s%%%s\n" \
        "$C_GREEN" "$median" "$C_RESET"
    printf "   top technique      %s%s%s\n" \
        "$C_MAGENTA" "$top_tech" "$C_RESET"
    divider
}

# ── Main loop ───────────────────────────────────────────────────────
FRAME=0
while true; do
    FRAME=$((FRAME + 1))
    # Each render call wraps in `|| true` so a transient failure (file
    # half-written, jq parse error, missing field, terminal resize race) does
    # not exit the loop. The FRAME counter + ●/○ tick in the header confirms
    # the loop is iterating.
    {
        printf '\033[H\033[2J' || true   # clear + home
        render_header           || true
        render_overview         || true
        render_active_problems  || true
        render_perf_leaders     || true
        render_recent_events    || true
    } 2>/dev/null

    sleep "$REFRESH_S" || break   # break on Ctrl-C interrupting sleep
done

# If we somehow fall out of the loop (broken sleep, etc.), restore the
# terminal cleanly.
cleanup
