#!/usr/bin/env bash
set -euo pipefail

source_root=$(cd "$(dirname "$0")/../.." && pwd)
watcher="$source_root/scripts/watch-upstream-drift.sh"
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/sync-watcher-test.XXXXXX")
trap 'rm -rf -- "$tmpdir"' EXIT

fixture="$tmpdir/repo"
state="$tmpdir/issues.json"
fake_gh="$tmpdir/gh"
mkdir -p "$fixture"

git -C "$fixture" init -q
git -C "$fixture" config user.name 'sync watcher test'
git -C "$fixture" config user.email 'sync-watcher@example.invalid'
printf 'base\n' > "$fixture/README.md"
git -C "$fixture" add README.md
git -C "$fixture" commit -q -m 'fixture base'
git -C "$fixture" update-ref refs/test/fork HEAD

mkdir -p \
    "$fixture/ggml/src/ggml-cuda" \
    "$fixture/ggml/src/ggml-cpu" \
    "$fixture/ggml/src" \
    "$fixture/common" \
    "$fixture/src/models"
printf 'mmvq\n' > "$fixture/ggml/src/ggml-cuda/mmvq.cu"
printf 'cpu\n' > "$fixture/ggml/src/ggml-cpu/ggml-cpu.c"
printf 'backend\n' > "$fixture/ggml/src/ggml-backend.cpp"
printf 'arg\n' > "$fixture/common/arg.cpp"
printf 'dflash\n' > "$fixture/src/models/dflash.cpp"
git -C "$fixture" add .
git -C "$fixture" commit -q -m 'upstream drift fixture'
git -C "$fixture" update-ref refs/test/buun HEAD

printf '{"next":1,"mutations":0,"issues":[]}\n' > "$state"
cat > "$fake_gh" <<'FAKE'
#!/usr/bin/env bash
set -euo pipefail
state=${FAKE_GH_STATE:?}
[[ ${1:-} == issue ]] || exit 2
action=${2:-}
shift 2

write_state() {
    local next=$1
    mv "$next" "$state"
}

case "$action" in
    list)
        jq -r '.issues[] | @base64' "$state"
        ;;
    create)
        title=''
        body=''
        while (($#)); do
            case "$1" in
                --title) title=$2; shift 2 ;;
                --body) body=$2; shift 2 ;;
                *) shift ;;
            esac
        done
        next=$(mktemp "${TMPDIR:-/tmp}/fake-gh.XXXXXX")
        jq --arg title "$title" --arg body "$body" '
            .issues += [{number:.next,state:"OPEN",title:$title,body:$body}] |
            .next += 1 | .mutations += 1' "$state" > "$next"
        write_state "$next"
        echo 'https://example.invalid/issues/1'
        ;;
    edit)
        number=$1
        shift
        title=''
        body=''
        while (($#)); do
            case "$1" in
                --title) title=$2; shift 2 ;;
                --body) body=$2; shift 2 ;;
                *) shift ;;
            esac
        done
        next=$(mktemp "${TMPDIR:-/tmp}/fake-gh.XXXXXX")
        jq --argjson number "$number" --arg title "$title" --arg body "$body" '
            (.issues[] | select(.number == $number) | .title) = $title |
            (.issues[] | select(.number == $number) | .body) = $body |
            .mutations += 1' "$state" > "$next"
        write_state "$next"
        ;;
    close|reopen)
        number=$1
        wanted=CLOSED
        [[ $action == reopen ]] && wanted=OPEN
        next=$(mktemp "${TMPDIR:-/tmp}/fake-gh.XXXXXX")
        jq --argjson number "$number" --arg wanted "$wanted" '
            (.issues[] | select(.number == $number) | .state) = $wanted |
            .mutations += 1' "$state" > "$next"
        write_state "$next"
        ;;
    *)
        exit 2
        ;;
esac
FAKE
chmod +x "$fake_gh"

run_watcher() {
    (
        cd "$fixture"
        FAKE_GH_STATE="$state" \
        SYNC_WATCHER_GH_BIN="$fake_gh" \
        SYNC_WATCHER_BUUN_REF=refs/test/buun \
        SYNC_WATCHER_FORK_REF=refs/test/fork \
        "$watcher" --no-fetch --no-mainline
    )
}

run_watcher
jq -e '
    .mutations == 1 and
    (.issues | length) == 1 and
    .issues[0].state == "OPEN" and
    (.issues[0].body | contains("upstream drift fixture")) and
    (.issues[0].body | contains("ggml/src/ggml-cuda/mmvq.cu")) and
    (.issues[0].body | contains("ggml/src/ggml-cpu/ggml-cpu.c")) and
    (.issues[0].body | contains("ggml/src/ggml-backend.cpp")) and
    (.issues[0].body | contains("common/arg.cpp")) and
    (.issues[0].body | contains("src/models/dflash.cpp"))' "$state" >/dev/null

run_watcher
jq -e '.mutations == 1 and (.issues | length) == 1' "$state" >/dev/null

git -C "$fixture" update-ref refs/test/fork refs/test/buun
run_watcher
jq -e '.mutations == 2 and .issues[0].state == "CLOSED"' "$state" >/dev/null

run_watcher
jq -e '.mutations == 2 and .issues[0].state == "CLOSED"' "$state" >/dev/null

printf 'sync-watcher acceptance: OK\n'
