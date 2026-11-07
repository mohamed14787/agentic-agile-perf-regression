#!/usr/bin/env zsh
# run_oracle.sh — system-level oracle: load-test a senior and a junior workspace with the
# SAME workload, compare response-time distributions (Wilcoxon + Cliff's Delta, the same
# stats already used for the local Kieker signal), set the row's system_verdict.
#
# Deviations from the paper, disclosed:
#   - Docker Compose substitutes for Podman (functionally equivalent for our purposes).
#   - The JMeter script's load-generation engine (BlazeMeter's Concurrency Thread Group
#     plugin) is swapped for stock JMeter ThreadGroup + ConstantThroughputTimer
#     (profiling/oracle/adapt_jmx.py) — the plugin isn't installed and homebrew's jmeter
#     formula ships no CLI installer; the HTTP test SCENARIO (same requests/regex
#     extractors) is preserved byte-for-byte, only the load engine changed.
#   - Single machine: load generator and app share the same CPU cores. Every row from this
#     script carries oracle.isolation="single-machine" for that reason.
#   - Reduced run length. The paper discards a 5-minute warm-up and retains 15 minutes of
#     steady state, repeated 3 times per version. A full sweep at that length is not feasible
#     here (see the warmup notes below — webui startup alone has been observed to need 50-80
#     minutes on this machine), so the recommended setting is warmup 180s + 600s steady x 3
#     repeats. Shorter runs mean wider confidence intervals, not biased ones.
#
# INTERLEAVED MEASUREMENT (2026-07-27). This script used to run the senior track to completion
# and then the junior track. That is exactly the confound already diagnosed and fixed for the
# Kieker local signal: two measurements taken in sequence differ by whatever the machine was
# doing at the time, and on this hardware that offset reached -30.5% — large enough to invert
# the sign of a real regression (see profiling/compare_kieker.py's module docstring). With
# REPEATS > 1 the MEASUREMENT phases now alternate S,J,S,J,S,J so that each senior sample sits
# adjacent in time to a junior sample and slow machine drift cancels instead of loading onto
# one track. Maven packaging and the docker build still happen once per track, since they cost
# minutes and cannot influence a later measurement.
#
# Usage: zsh scripts/run_oracle.sh <senior_workspace> <junior_workspace> <ticket_id> \
#          [threads] [duration_s] [warmup_s] [rps] [repeats]
#
# Paper-like invocation:
#   zsh scripts/run_oracle.sh <sen> <jun> PERF-x 5 600 180 50 3

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
source .venv/bin/activate
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk}"

SENIOR_WS="$1"; JUNIOR_WS="$2"; TICKET="$3"
THREADS="${4:-5}"
DURATION="${5:-30}"
WARMUP="${6:-5}"
RPS="${7:-5}"
REPEATS="${8:-${REPEATS:-1}}"

if [[ -z "$SENIOR_WS" || -z "$JUNIOR_WS" || -z "$TICKET" ]]; then
  echo "usage: run_oracle.sh <senior_workspace> <junior_workspace> <ticket_id> [threads] [duration_s] [warmup_s] [rps]"
  exit 1
fi

JMX="profiling/oracle/teastore_adapted.jmx"

# ── host memory telemetry ────────────────────────────────────────────────────
# Two runs (2026-08-05 15:13 and 16:33) died mid-JMeter with no error, no exception, no shell
# diagnostic: both logs stop on a clean 30s summariser tick at 0.00% errors and dead-on target
# throughput. Post-mortem ruled out every workload-side cause — the colima VM survived both
# (uptime 4d, zero OOM kills in its dmesg, no swap inside), no JetsamEvent report exists for
# that date although the mechanism demonstrably writes them on this machine, and Amphetamine
# held PreventUserIdleSystemSleep across both windows. What is left is host memory: colima was
# configured `memory: 12` on a 16 GB Mac while JMeter runs on the HOST (not in a container),
# and lima#2789 (open) means a vmType:vz guest never returns memory to macOS — it ratchets to
# its configured ceiling and holds it until a colima restart. The VM is now sized 8 GB, but a
# silent kill must never again be undiagnosable, so record the memory state at every phase
# boundary. MEMLOG is per-run and append-only; it is the artifact to read first if a run
# vanishes again.
MEMLOG="${MEMLOG:-logs/oracle_mem_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$MEMLOG")"
_mem_snapshot() {
  local tag="$1"
  {
    # Page size is 16 KiB on Apple silicon, not the 4 KiB every vm_stat one-liner on the web
    # assumes; hardcoding 4096 here would understate free memory by 4x.
    local psize free_mb swap_line vm_rss_mb
    psize=$(sysctl -n hw.pagesize)
    free_mb=$(vm_stat | awk -v p="$psize" '/Pages free/ {gsub(/\./,"",$3); printf "%.0f", $3*p/1048576}')
    swap_line=$(sysctl -n vm.swapusage)
    # The guest's host-side footprint: this is the number lima#2789 says never comes back down.
    vm_rss_mb=$(ps -Ao rss,comm | awk '/Virtualization.VirtualMachine/ {s+=$1} END {printf "%.0f", s/1024}')
    printf '%s\t%s\tfree=%sMB\tvmRSS=%sMB\tswap[%s]\n' \
      "$(date '+%Y-%m-%d %H:%M:%S')" "$tag" "${free_mb:-?}" "${vm_rss_mb:-0}" "$swap_line"
  } >> "$MEMLOG" 2>&1 || true
}
echo "=== host memory telemetry -> $MEMLOG ==="
_mem_snapshot "run-start"

# ── the feature endpoint ─────────────────────────────────────────────────────
# WITHOUT THIS THE RUN MEASURES NOTHING. The JMeter scenario is TeaStore's stock browse path
# (Home, Login, LoginAction, Category, Product, Logout). Every ticket confines the feature to
# the persistence service and every injected inefficiency lands in a NEW repository method, so
# the browse path cannot reach the changed code. Four rows were measured that way and all four
# returned system_verdict="none" — a verdict about an unmodified code path, not about the
# change. The workload now also drives the feature's own endpoint, resolved from the frozen
# ticket and cross-checked against the @Path annotations in BOTH workspaces.
#
# FEATURE_ID defaults to the junior workspace's basename minus the -junior suffix, which is how
# every workspace in teastore-dataset/ is named. Set ORACLE_FEATURE_PATH to override the
# resolution entirely, or ORACLE_FEATURE_PATH="" to run the browse-only workload deliberately.
FEATURE_ID="${FEATURE_ID:-$(basename "$JUNIOR_WS" | sed 's/-junior$//')}"
PERSISTENCE_PORT="${ORACLE_PERSISTENCE_PORT:-8081}"
if [[ -z "${ORACLE_FEATURE_PATH+x}" ]]; then
  echo "=== resolving feature endpoint for '$FEATURE_ID' ==="
  ORACLE_FEATURE_PATH="$(.venv/bin/python scripts/feature_endpoint.py "$FEATURE_ID" \
      --workspace "$SENIOR_WS")" || {
    echo "!!! could not resolve the feature endpoint for '$FEATURE_ID' against $SENIOR_WS."
    echo "!!! Measuring without it would repeat the defect this check exists to prevent:"
    echo "!!! the workload would never touch the changed code and would report a verdict anyway."
    exit 1
  }
  # The junior must serve the SAME endpoint. A renamed endpoint is the API-name divergence that
  # already cost feature 13 a full re-run; catching it here costs seconds instead of hours.
  .venv/bin/python scripts/feature_endpoint.py "$FEATURE_ID" --workspace "$JUNIOR_WS" >/dev/null || {
    echo "!!! $JUNIOR_WS does not serve the endpoint its ticket promises — aborting."
    exit 1
  }
fi
if [[ -n "$ORACLE_FEATURE_PATH" ]]; then
  echo "    feature endpoint: http://localhost:${PERSISTENCE_PORT}${ORACLE_FEATURE_PATH}"
else
  echo "    feature endpoint: (none — browse-only workload, verdict will not reflect the change)"
fi

MODULES=(services/tools.descartes.teastore.registry
         services/tools.descartes.teastore.persistence
         services/tools.descartes.teastore.auth
         services/tools.descartes.teastore.image
         services/tools.descartes.teastore.recommender
         services/tools.descartes.teastore.webui)

_package_track() {
  local ws="$1"; local label="$2"
  echo "=== [$label] packaging WARs for docker build ($ws) ==="
  local plarg=$(IFS=,; echo "${MODULES[*]}")
  # `clean package` (not just `package`): this project directory is under active iCloud
  # Desktop sync, confirmed live to race fast incremental rebuilds and leave "Foo N.class"
  # conflict-copy duplicates in target/ that corrupt the docker build's tar context
  # ("missed writing N bytes" / "unexpected EOF"). A full clean before each package reduces
  # (does not eliminate) the collision surface by not leaving stale classes for iCloud to
  # race against. Root cause is environmental (iCloud sync), not fixable from this script.
  # maven.test.skip=true (not just -DskipTests): the oracle only needs the packaged WARs to
  # boot and serve load, not passing unit tests. -DskipTests alone still COMPILES test
  # sources, and a live junior-workspace run hit a real "cannot find symbol" test-compile
  # break in services/tools.descartes.teastore.image — irrelevant to a load test, but fatal
  # to `package` with -DskipTests. maven.test.skip=true skips compiling tests altogether.
  (cd "$ws" && mvn -q clean package -pl "$plarg" -am -Dmaven.test.skip=true \
      -Dspring-javaformat.validate.skip=true -Dcheckstyle.skip=true \
      -Dnohttp.checkstyle.validate.skip=true)

  echo "=== [$label] generating docker-compose override ==="
  local compose="$ws/docker-compose.oracle.yaml"
  .venv/bin/python scripts/oracle_compose.py "$ws" "$compose"

  echo "=== [$label] docker compose build (images only; measurement brings them up) ==="
  docker-compose -f "$compose" build
}

# Measurement phase for one track, one repeat. Split out of the packaging above so that
# repeats can alternate between tracks without paying for a rebuild each time.
_measure_track() {
  local ws="$1"; local label="$2"; local rep="$3"
  local compose="$ws/docker-compose.oracle.yaml"

  echo "=== [$label r$rep] docker compose up ==="
  _mem_snapshot "[$label r$rep] before-up"
  docker-compose -f "$compose" up -d
  _mem_snapshot "[$label r$rep] after-up"

  echo "=== [$label r$rep] warmup: waiting for webui, then one throwaway request ==="
  # -m 5 is required, not optional: without a per-request timeout, a single slow/hung
  # backend call (confirmed live under amd64-on-arm64 QEMU emulation — Jersey client
  # TimeoutExceptions in webui/recommender logs, JVM at 80% CPU for minutes with no new
  # log output) blocks curl indefinitely, so the "tries" counter never advances and this
  # loop silently never times out at all.
  # WARMUP_MAX_TRIES default raised from 90 (~10-15 min) to 600 (~50-80 min): observed live
  # that host-level I/O contention (Spotlight/Time Machine reacting to the docker image
  # layers/WARs a build just wrote) can starve JVM startup inside the container well past the
  # old ceiling even though webui is not actually hung — just slow. Overridable via env var.
  #
  # READINESS IS NOT "THE INDEX PAGE RETURNS 200" (2026-07-27). The index renders before the
  # backend services have registered with the registry, so a gate on it alone releases JMeter
  # against a half-started system. Measured live on a 60s smoke run: the senior track produced
  # 829 samples over a 60s span at 9.4% errors while the junior track produced 395 over 32s at
  # 20% errors (incl. SocketTimeoutException) — the two tracks did not receive the same load,
  # which makes the comparison meaningless no matter how good the statistics downstream are.
  # The gate therefore exercises the pages the JMeter scenario actually drives: `category`
  # needs persistence + image, `login` needs auth. Only when all three serve 200 is the stack
  # genuinely ready.
  #
  # The feature endpoint is part of the gate for the same reason. It is served by persistence
  # on its own published port, so it can 404 (wrong path), 500 (a junior whose query throws) or
  # simply not be up while all three webui pages are green. Any of those turns every Feature
  # sample into a failed sample, which parse_jtl drops — leaving a verdict computed from browse
  # traffic only and indistinguishable from a healthy run. Gating on it makes that state a loud
  # abort instead of a quiet null.
  # WHICH probe failed is the diagnosis, so record it instead of collapsing every failure into
  # one "not ready". FEATURE_PROBE_TIMEOUT is separate from the webui probes' 8s because a
  # feature endpoint is allowed to be genuinely slow — that is what several of these pairs are
  # measuring — while a webui page that needs more than 8s is not slow, it is broken.
  local last_probe=""
  local booted=0
  _all_endpoints_ready() {
    local base="http://localhost:8080/tools.descartes.teastore.webui"
    last_probe="webui /"
    curl -sf -m 8 -o /dev/null "$base/" 2>/dev/null || return 1
    booted=1
    last_probe="webui /category"
    curl -sf -m 8 -o /dev/null "$base/category?category=2&page=1" 2>/dev/null || return 1
    last_probe="webui /login"
    curl -sf -m 8 -o /dev/null "$base/login" 2>/dev/null || return 1
    if [[ -n "$ORACLE_FEATURE_PATH" ]]; then
      last_probe="feature ${ORACLE_FEATURE_PATH}"
      curl -sf -m "${ORACLE_FEATURE_PROBE_TIMEOUT:-30}" -o /dev/null \
        "http://localhost:${PERSISTENCE_PORT}${ORACLE_FEATURE_PATH}" 2>/dev/null || return 1
    fi
    return 0
  }
  _dump_warmup_diagnostics() {
    local reason="$1"
    local diag="$ws/oracle_warmup_failure_logs.txt"
    echo "=== [$label r$rep] dumping container logs to $diag before teardown ==="
    : > "$diag"
    echo "--- reason: $reason (last failing probe: ${last_probe:-none}) ---" >> "$diag"
    if [[ -n "$ORACLE_FEATURE_PATH" ]]; then
      echo "--- feature endpoint probe (http://localhost:${PERSISTENCE_PORT}${ORACLE_FEATURE_PATH}) ---" >> "$diag"
      # `|| true` IS the fix, not defensive noise. With `set -e` a curl that exits 28 (timeout)
      # killed the whole script HERE — before the container logs below, before `down`, and
      # before the caller could report the pair as failed. Measured live on copurchase: the
      # diagnostic file held two lines, the six services that would have named the cause were
      # never read, and the stack stayed up for seven hours because the teardown never ran.
      curl -s -m "${ORACLE_FEATURE_PROBE_TIMEOUT:-30}" -o /dev/null \
        -w 'http_code=%{http_code} time_total=%{time_total}s\n' \
        "http://localhost:${PERSISTENCE_PORT}${ORACLE_FEATURE_PATH}" >> "$diag" 2>&1 || true
    fi
    for svc in webui registry persistence auth image recommender db; do
      echo "--- $svc ---" >> "$diag"
      docker-compose -f "$compose" logs --no-color --tail=200 "$svc" >> "$diag" 2>&1 || true
    done
    echo "--- docker ps -a ---" >> "$diag"
    docker ps -a >> "$diag" 2>&1 || true
    docker-compose -f "$compose" down || true
  }
  local tries=0
  local booted_at=-1
  local max_tries="${WARMUP_MAX_TRIES:-600}"
  # Two budgets, because "still starting" and "started but broken" are different failures and
  # only the first one deserves 30 minutes. Once webui's index has served a 200 the JVMs are up;
  # anything still failing after that is functional — copurchase's junior overrode the datasource
  # to a driver it does not ship, so persistence answered 500 forever while the loop patiently
  # waited out the full startup budget. Post-boot failures now abort in ~5 minutes with the
  # failing probe named.
  local post_boot_tries="${WARMUP_POST_BOOT_TRIES:-100}"
  until _all_endpoints_ready; do
    tries=$((tries + 1))
    if (( booted && booted_at < 0 )); then
      booted_at=$tries
      echo "=== [$label r$rep] webui is up after ~$((tries * 3))s; remaining probes have a ${post_boot_tries}-try budget ==="
    fi
    if (( booted_at >= 0 && tries - booted_at > post_boot_tries )); then
      echo "!!! [$label r$rep] webui booted but '${last_probe}' never became healthy within"
      echo "!!! $((post_boot_tries * 3))s of boot — this is a functional failure, not a slow start."
      _dump_warmup_diagnostics "post-boot probe failure"
      return 1
    fi
    if [[ $tries -gt $max_tries ]]; then
      echo "!!! [$label r$rep] stack never became fully reachable after $max_tries tries — aborting this track"
      _dump_warmup_diagnostics "startup timeout"
      return 1
    fi
    sleep 3
  done
  echo "=== [$label r$rep] all endpoints ready after ~$((tries * 3))s; settling before load ==="
  # A short settle after first-response: the first hit on each page triggers JSP compilation
  # and lazy service-client initialisation, which would otherwise land inside the measurement.
  sleep "${SETTLE_SECONDS:-15}"

  # JMeter's scheduler duration INCLUDES the ramp-up, and the JMX maps our `warmup` onto
  # ThreadGroup.ramp_time. Passing DURATION straight through therefore bought only
  # (DURATION - WARMUP) seconds at full concurrency — a nominal "600s steady state" was really
  # 420s, and the discrepancy grew with the warm-up. Pass ramp + steady so DURATION means what
  # the flag says: seconds of steady state AFTER the ramp, which is also the window
  # --discard-warmup-s leaves behind.
  local total=$(( DURATION + WARMUP ))
  echo "=== [$label r$rep] running JMeter (threads=$THREADS steady=${DURATION}s ramp/warmup=${WARMUP}s total=${total}s rps=$RPS) ==="
  # Per-repeat JTL so the repeats accumulate instead of overwriting each other; _concat_jtl
  # merges them into the canonical oracle_results.jtl that system_oracle_stats.py reads.
  local jtl="$ws/oracle_results_r${rep}.jtl"
  rm -f "$jtl"
  _mem_snapshot "[$label r$rep] before-jmeter"
  # Cap the load generator's heap. JMeter's launcher defaults to a 1 GB heap and will grow the
  # JVM's footprint well past that with metaspace and the JTL write buffers; on a host that is
  # already sharing 16 GB with an 8 GB guest, an unbounded generator is the one process most
  # likely to push the system over. 1 GB is ample for a 15-thread CLI run with no listeners.
  HEAP="${ORACLE_JMETER_HEAP:--Xms512m -Xmx1g}" \
  jmeter -n -t "$JMX" -Jhost=localhost -Jport=8080 \
    -Jthread="$THREADS" -Jduration="$total" -Jwarmup="$WARMUP" -Jrps="$RPS" \
    -Jfeaturehost=localhost -Jfeatureport="$PERSISTENCE_PORT" \
    -Jfeaturepath="$ORACLE_FEATURE_PATH" \
    -l "$jtl"

  # ACHIEVED-vs-TARGET throughput check. ConstantThroughputTimer can only PAUSE threads, never
  # accelerate them, so a run that cannot reach its target silently produces a valid-looking JTL
  # at the wrong load level — and load level is the independent variable of the whole experiment.
  # Nothing downstream would notice. Compute it from the JTL's own timestamps and say so.
  .venv/bin/python - "$jtl" "$RPS" "$WARMUP" <<'PY' || true
import csv, sys, collections
jtl, target, warmup = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
try:
    with open(jtl, newline="", encoding="utf-8", errors="replace") as fh:
        rows = [(r["label"], int(r["timeStamp"])) for r in csv.DictReader(fh)
                if r.get("timeStamp", "").isdigit()]
except (OSError, KeyError):
    sys.exit(0)
if len(rows) < 2:
    sys.exit(0)
# THREE VALID COUNTS, and the target refers to exactly one of them. A TransactionController
# writes a synthetic PARENT row alongside its child rows, so:
#   all rows            78.6/s  - double-counts each transaction
#   JMeter samples      50.0/s  - parent counted, children not  <-- what ConstantThroughputTimer targets
#   real HTTP requests  64.3/s  - children counted, parent not  <-- what the server actually receives
# Verified against JMeter's own summariser, which printed "50.0/s" for the run these numbers
# come from. Comparing the target against either other count produces a phantom OFF-TARGET
# (+57% and +29% respectively were both reported here before this was reconciled).
# Children are detected structurally: "L-0"/"L-1" alongside an "L" label.
labels = {l for l, _ in rows}
children = {l for l in labels
            if l.rsplit("-", 1)[0] in labels and l.rsplit("-", 1)[-1].isdigit()}
parents = {l.rsplit("-", 1)[0] for l in children}
t0 = min(t for _, t in rows)
cut = t0 + warmup * 1000                      # same window system_oracle_stats.py analyses
steady = [(l, t) for l, t in rows if t >= cut]
if not steady:
    sys.exit(0)
span = (max(t for _, t in steady) - cut) / 1000.0
if span <= 0:
    sys.exit(0)
per_label = collections.Counter(l for l, _ in steady)
achieved = sum(v for k, v in per_label.items() if k not in children) / span   # JMeter accounting
http_rate = sum(v for k, v in per_label.items() if k not in parents) / span   # server-side load
loops = max(per_label.values()) / span if per_label else 0.0
drift = (achieved - target) / target * 100.0
flag = "OK" if abs(drift) <= 10 else "OFF-TARGET"
print(f"    throughput: target {target:.1f}/s, achieved {achieved:.1f}/s ({drift:+.1f}%) "
      f"over {span:.0f}s steady — {flag}   [{loops:.2f} loops/s, "
      f"{http_rate:.1f} HTTP req/s at the server]")
PY

  _mem_snapshot "[$label r$rep] after-jmeter"

  echo "=== [$label r$rep] capturing container CPU stats ==="
  docker stats --no-stream > "$ws/oracle_docker_stats_r${rep}.txt" 2>&1 || true

  echo "=== [$label r$rep] tearing down ==="
  docker-compose -f "$compose" down
  # Post-teardown is the diagnostic sample that matters: with every container stopped, any vmRSS
  # that stays high is the lima#2789 ratchet rather than live workload, and a monotonic climb
  # across phases is the signature to look for if a run dies again.
  _mem_snapshot "[$label r$rep] after-down"
}

# Merge per-repeat JTLs into one CSV. system_oracle_stats.parse_jtl uses csv.DictReader, so
# exactly one header row must survive: keep the first file's header, skip it in the rest.
_concat_jtl() {
  local ws="$1"
  local out="$ws/oracle_results.jtl"
  local first=1 n=0
  : > "$out"
  for f in "$ws"/oracle_results_r*.jtl(N); do
    [[ -f "$f" ]] || continue
    if [[ $first == 1 ]]; then
      cat "$f" >> "$out"
      first=0
    else
      tail -n +2 "$f" >> "$out"
    fi
    n=$((n + 1))
  done
  echo "=== merged $n repeat(s) -> $out ==="
}

REUSE_SENIOR=0
if [[ -n "$SKIP_SENIOR" && -f "$SENIOR_WS/oracle_results.jtl" ]]; then
  REUSE_SENIOR=1
  echo "=== [senior] SKIP_SENIOR set and $SENIOR_WS/oracle_results.jtl exists — reusing prior senior results ==="
  echo "!!! WARNING: reused senior results were measured at a different time, which"
  echo "!!! REINTRODUCES the run-order/drift confound that interleaving exists to remove."
  echo "!!! The resulting verdict is not drift-controlled. Do not use it as ground truth."
fi

echo
echo "########## packaging both tracks once (build cost does not vary per repeat) ##########"
[[ $REUSE_SENIOR == 1 ]] || _package_track "$SENIOR_WS" "senior"
_package_track "$JUNIOR_WS" "junior"

# (N) is zsh's NULL_GLOB qualifier. Without it an unmatched glob is a FATAL error in zsh (not
# an empty expansion as in bash), so with `set -e` the very first run of a fresh workspace —
# which by definition has no prior per-repeat JTLs — aborts the whole script.
[[ $REUSE_SENIOR == 1 ]] || rm -f "$SENIOR_WS"/oracle_results_r*.jtl(N)
rm -f "$JUNIOR_WS"/oracle_results_r*.jtl(N)

# ALTERNATE THE ORDER, not just the repeats. Running senior-then-junior in every repeat leaves
# a residual bias: the track measured second always benefits from whatever the first one warmed
# up. Measured live across the 4-pair subset, the machine warmed substantially over a session
# (volatility's senior track: 9.96 -> 8.88 -> 6.87 ms mean across its three repeats), so a
# fixed within-repeat order systematically favours the second track. Swapping the order on even
# repeats makes each track run first and second equally often, so the effect cancels in the
# pooled result instead of accumulating onto one side.
for rep in $(seq 1 "$REPEATS"); do
  echo
  if (( rep % 2 == 1 )); then
    echo "########## repeat $rep/$REPEATS — interleaved senior -> junior ##########"
    [[ $REUSE_SENIOR == 1 ]] || _measure_track "$SENIOR_WS" "senior" "$rep"
    _measure_track "$JUNIOR_WS" "junior" "$rep"
  else
    echo "########## repeat $rep/$REPEATS — interleaved junior -> senior (order swapped) ##########"
    _measure_track "$JUNIOR_WS" "junior" "$rep"
    [[ $REUSE_SENIOR == 1 ]] || _measure_track "$SENIOR_WS" "senior" "$rep"
  fi
done

[[ $REUSE_SENIOR == 1 ]] || _concat_jtl "$SENIOR_WS"
_concat_jtl "$JUNIOR_WS"

echo "=== computing system-level verdict ==="
STATS_OUT="$JUNIOR_WS/oracle_stats.json"
# Discard the ramp-up window: JMeter maps our `warmup` onto ThreadGroup.ramp_time, which still
# RECORDS samples, so without this the cold-JVM ramp is averaged into the steady state (the
# paper excludes its warm-up period explicitly, §IV-C).
#
# The path below is the MERGED jtl, but system_oracle_stats now discovers the per-repeat
# `oracle_results_r*.jtl` beside it and applies the cutoff to each repeat separately. That
# matters because every repeat restarts the stack and so has its own cold ramp: with one cutoff
# over the merged file, repeat 1's ramp was discarded and repeats 2-3's were kept — measured on
# reorder, 16.4% of the Feature samples were ramp, and the pooled MPD read +15.7% instead of
# +11.7%. The inline throughput check below was always per-repeat; the two paths now agree.
.venv/bin/python profiling/system_oracle_stats.py \
  "$SENIOR_WS/oracle_results.jtl" "$JUNIOR_WS/oracle_results.jtl" \
  --discard-warmup-s "$WARMUP" --target-rps "$RPS" --out "$STATS_OUT"

echo
echo "=== DONE — ticket=$TICKET stats written to $STATS_OUT ==="
echo "oracle.isolation: single-machine (load generator and app share CPU cores — disclosed deviation)"
echo "  the generator CANNOT be pinned away from the app: macOS exposes no CPU affinity API for"
echo "  user processes (no taskset), and shrinking the colima VM to reserve host cores would"
echo "  change the application's capacity mid-corpus. The deficit is instead MEASURED per row —"
echo "  see load_generator.worst_drift_pct in the stats; sessions A-C never fell below -0.7%."

# ---------------------------------------------------------------------------------------------
# RECLAIM THIS PAIR'S BUILD IMAGES.
#
# `docker-compose down` stops containers but keeps the images the build produced, and a pair
# leaves ~12 of them (one per service per track). Measured over sessions A-D: ~3.4 GB of host
# space per pair, and after 19 pairs the host had 3.5 GB left of 228 GB — close enough to the
# edge that the NEXT session would have died mid-run. An ENOSPC here does not announce itself;
# it surfaces as a generic container failure that reads like a model or harness fault.
#
# Only this pair's own images are removed, matched on the two workspace basenames, so the
# protected bases (`descartesresearch/teastore-base`, `teastore-db`) and every other pair are
# untouched by construction rather than by a filter that has to be got right. The images are
# pure build outputs — `docker-compose build` regenerates them from workspace source, and the
# shared base layers they sit on are what makes that fast, so removing them costs nothing on a
# re-run and nothing at all on a fresh pair.
#
# THE TRIM IS THE PART THAT MATTERS. Freeing space inside the guest does not give it back to
# the host: the VM's disk image is sparse and only shrinks when the guest issues TRIM. Measured
# 2026-08-22 — pruning 14 GB inside the VM moved host free space from 3.5 GB to 3.3 GB (i.e.
# nowhere); `fstrim` then took it to 23 GB. This is the same host-side ratchet as lima#2789
# noted at the memory snapshots above.
#
# Set ORACLE_KEEP_IMAGES=1 to skip, e.g. when debugging a stack and you want the built images
# to inspect. Never allowed to fail the run: the measurement is already written and committed
# to disk by this point, and losing a completed pair to a cleanup error would be absurd.
if [[ -z "${ORACLE_KEEP_IMAGES:-}" ]]; then
  echo
  echo "=== reclaiming build images for this pair ==="
  for ws in "$SENIOR_WS" "$JUNIOR_WS"; do
    proj="$(basename "$ws")"
    imgs=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep "^${proj}-" || true)
    if [[ -n "$imgs" ]]; then
      echo "$imgs" | xargs -r docker rmi -f >/dev/null 2>&1 || true
      echo "    removed $(echo "$imgs" | wc -l | tr -d ' ') image(s) for $proj"
    fi
  done
  docker volume prune -f >/dev/null 2>&1 || true
  # Guest-to-host reclaim. Only meaningful under colima; a no-op elsewhere, never fatal.
  if command -v colima >/dev/null 2>&1; then
    colima ssh -- sudo fstrim /mnt/lima-colima >/dev/null 2>&1 || true
  fi
  echo "    host free now: $(df -h / | awk 'NR==2{print $4}')"
fi
