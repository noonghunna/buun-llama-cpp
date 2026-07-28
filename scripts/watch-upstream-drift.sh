#!/usr/bin/env bash
set -euo pipefail

MARKER='<!-- buun-sync-watcher:sync-due -->'
TITLE='[sync-watcher] buun master sync due'

repo_slug=${SYNC_WATCHER_REPO:-noonghunna/buun-llama-cpp}
buun_url=${SYNC_WATCHER_BUUN_URL:-https://github.com/spiritbuun/buun-llama-cpp.git}
fork_url=${SYNC_WATCHER_FORK_URL:-https://github.com/noonghunna/buun-llama-cpp.git}
mainline_url=${SYNC_WATCHER_MAINLINE_URL:-https://github.com/ggml-org/llama.cpp.git}
buun_ref=${SYNC_WATCHER_BUUN_REF:-refs/sync-watcher/buun/master}
fork_ref=${SYNC_WATCHER_FORK_REF:-refs/sync-watcher/fork/master}
mainline_ref=${SYNC_WATCHER_MAINLINE_REF:-refs/sync-watcher/mainline/master}
git_bin=${SYNC_WATCHER_GIT_BIN:-git}
gh_bin=${SYNC_WATCHER_GH_BIN:-gh}
do_fetch=1
check_mainline=1
dry_run=0

usage() {
    cat <<'USAGE'
Usage: scripts/watch-upstream-drift.sh [options]

Watcher-only comparison of spiritbuun/master and the public fork's master.
It updates one marker-keyed GitHub issue when a sync is due and closes that
same issue when the fork catches up. It never checks out, rebases, or pushes.

Options:
  --repo OWNER/REPO  Rolling issue repository (default: noonghunna/buun-llama-cpp)
  --no-mainline      Skip the informational ggml-org -> buun comparison
  --no-fetch         Compare existing SYNC_WATCHER_*_REF refs (tests/offline use)
  --dry-run          Print planned issue actions without mutating GitHub
  -h, --help         Show this help

Example off-hours cron entry (authentication is supplied by gh):
  17 3 * * 0 cd /path/to/buun-llama-cpp && scripts/watch-upstream-drift.sh
USAGE
}

while (($#)); do
    case "$1" in
        --repo)
            [[ $# -ge 2 ]] || { echo "--repo requires OWNER/REPO" >&2; exit 2; }
            repo_slug=$2
            shift 2
            ;;
        --no-mainline)
            check_mainline=0
            shift
            ;;
        --no-fetch)
            do_fetch=0
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for command in "$git_bin" "$gh_bin" jq base64; do
    command -v "$command" >/dev/null || { echo "required command not found: $command" >&2; exit 2; }
done

repo_root=$("$git_bin" rev-parse --show-toplevel)
cd "$repo_root"

fetch_master() {
    local url=$1
    local ref=$2
    "$git_bin" fetch --quiet --no-tags "$url" "+refs/heads/master:$ref"
}

if ((do_fetch)); then
    fetch_master "$buun_url" "$buun_ref"
    fetch_master "$fork_url" "$fork_ref"
fi

"$git_bin" rev-parse --verify "${buun_ref}^{commit}" >/dev/null
"$git_bin" rev-parse --verify "${fork_ref}^{commit}" >/dev/null

mainline_note='not checked'
if ((check_mainline)); then
    mainline_ready=1
    if ((do_fetch)) && ! fetch_master "$mainline_url" "$mainline_ref"; then
        mainline_ready=0
    fi
    if ((mainline_ready)) && "$git_bin" rev-parse --verify "${mainline_ref}^{commit}" >/dev/null 2>&1; then
        mainline_behind=$("$git_bin" rev-list --count "${buun_ref}..${mainline_ref}")
        mainline_note="ggml-org master has ${mainline_behind} commit(s) not in buun master"
    else
        mainline_note='mainline comparison unavailable (fetch/ref failure)'
    fi
fi

behind=$("$git_bin" rev-list --count "${fork_ref}..${buun_ref}")
mapfile -t changed_files < <("$git_bin" diff --name-only "${fork_ref}..${buun_ref}")

tripwire_line() {
    local label=$1
    local file
    local -a matches=()
    for file in "${changed_files[@]}"; do
        case "$label:$file" in
            mmvq:*/mmvq.cu|mmvq:mmvq.cu)
                matches+=("$file")
                ;;
            ggml-cpu:*/ggml-cpu.c|ggml-cpu:ggml-cpu.c)
                matches+=("$file")
                ;;
            ggml-backend:*/ggml-backend.cpp|ggml-backend:ggml-backend.cpp)
                matches+=("$file")
                ;;
            arg:*/arg.cpp|arg:arg.cpp)
                matches+=("$file")
                ;;
            dflash:*dflash*)
                matches+=("$file")
                ;;
        esac
    done

    if ((${#matches[@]} == 0)); then
        printf -- '- [ ] `%s`: clear' "$label"
        return
    fi

    local joined=''
    for file in "${matches[@]}"; do
        [[ -z "$joined" ]] || joined+=', '
        joined+="\`$file\`"
    done
    printf -- '- [x] `%s`: %s' "$label" "$joined"
}

body="${MARKER}

Fork master is **${behind} commit(s) behind** **spiritbuun/buun-llama-cpp** master.

## New buun commits"
while IFS=$'\t' read -r hash subject; do
    [[ -n "$hash" ]] || continue
    body+=$'\n'
    body+="- \`${hash}\` ${subject}"
done < <("$git_bin" log --reverse --format='%h%x09%s' "${fork_ref}..${buun_ref}")

body+=$'\n\n## Hand-inspection tripwires\n'
body+="$(tripwire_line mmvq)"$'\n'
body+="$(tripwire_line ggml-cpu)"$'\n'
body+="$(tripwire_line ggml-backend)"$'\n'
body+="$(tripwire_line arg)"$'\n'
body+="$(tripwire_line dflash)"
body+=$'\n\n## Mainline context\n\n- '
body+="$mainline_note"
body+=$'\n\nThis is a watcher-only report. Follow pinned issue #10 for the human-run sync and post-rebase device gate.'

# Keep issue discovery marker-based rather than title-only so renames do not
# produce duplicates. gh emits each selected issue as one base64 record.
mapfile -t encoded_issues < <(
    "$gh_bin" issue list --repo "$repo_slug" --state all --limit 100 \
        --json number,state,title,body \
        --jq ".[] | select((.body // \"\") | contains(\"${MARKER}\")) | @base64"
)

declare -a numbers=() states=() titles=() bodies=()
for encoded in "${encoded_issues[@]}"; do
    decoded=$(printf '%s' "$encoded" | base64 --decode)
    numbers+=("$(jq -r '.number' <<<"$decoded")")
    states+=("$(jq -r '.state' <<<"$decoded")")
    titles+=("$(jq -r '.title' <<<"$decoded")")
    bodies+=("$(jq -r '.body // ""' <<<"$decoded")")
done

run_gh() {
    if ((dry_run)); then
        return 0
    fi
    "$gh_bin" "$@" >/dev/null
}

if ((behind > 0)); then
    primary=-1
    for i in "${!numbers[@]}"; do
        if [[ ${states[i]} == OPEN ]]; then
            primary=$i
            break
        fi
    done
    if ((primary < 0 && ${#numbers[@]} > 0)); then
        primary=0
    fi

    if ((primary < 0)); then
        echo "sync due: creating rolling issue (${behind} commits)"
        run_gh issue create --repo "$repo_slug" --title "$TITLE" --body "$body"
    else
        number=${numbers[primary]}
        if [[ ${titles[primary]} != "$TITLE" || ${bodies[primary]} != "$body" ]]; then
            echo "sync due: updating rolling issue #${number} (${behind} commits)"
            run_gh issue edit "$number" --repo "$repo_slug" --title "$TITLE" --body "$body"
        else
            echo "sync due: rolling issue #${number} already current (${behind} commits)"
        fi
        if [[ ${states[primary]} != OPEN ]]; then
            echo "sync due: reopening rolling issue #${number}"
            run_gh issue reopen "$number" --repo "$repo_slug"
        fi
    fi

    for i in "${!numbers[@]}"; do
        ((i == primary)) && continue
        if [[ ${states[i]} == OPEN ]]; then
            echo "dedup: closing extra rolling issue #${numbers[i]}"
            run_gh issue close "${numbers[i]}" --repo "$repo_slug"
        fi
    done
else
    closed_any=0
    for i in "${!numbers[@]}"; do
        if [[ ${states[i]} == OPEN ]]; then
            echo "fork caught up: closing rolling issue #${numbers[i]}"
            run_gh issue close "${numbers[i]}" --repo "$repo_slug"
            closed_any=1
        fi
    done
    if ((closed_any == 0)); then
        echo 'fork caught up: no open rolling issue'
    fi
fi
