#!/bin/zsh
# ─────────────────────────────────────────────────────────────────────────────
# kieker_profile.sh — local/component-level performance profiling of TeaStore,
# replicating the LOCAL measurement of:
#   "Early Detection of Performance Regressions by Bridging Local Performance Data
#    and Architectural Models" (Liao et al., ICSE 2025).
#
# THE PAPER'S "30 ITERATIONS" ARE IN-SESSION, NOT 30 SEPARATE TEST RUNS. Checked
# directly against their own replication package (Zenodo 10.5281/zenodo.13135986,
# Component_perf_res/Lx_<intensity>/kieker.dat): each file carries exactly ONE Kieker
# metadata ($0) record and per-operation call counts that are exact multiples of 30
# (e.g. 150 calls = 30 reps x 5 calls/rep). So they repeat the component test 30 times
# INSIDE one Kieker session (one JVM), not across 30 `mvn test` invocations. This
# script matches that: ITERATIONS defaults to 1 (one Kieker session), and the
# perf-exercise workload below repeats each catalog-wide query 30 times inside it
# (teastore.perf.repeats, default 30) — see FeaturePerfExerciseTest's javadoc for the
# one disclosed deviation (their repeats are cold, re-running test setup; the
# exerciser's are warm, repeating only the query).
#
# Running 30 separate invocations instead (ITERATIONS=30) is available but not the
# paper's protocol, and re-introduces the ~30% cross-invocation drift documented in
# compare_kieker.py — the reason this project's normalization existed in the first
# place. In-session repetition removes that confound rather than correcting for it.
#
# Additive / aside: touches no pipeline code and no pom files. The AspectJ agent is
# injected purely through Surefire's -DargLine (the persistence pom sets no argLine).
#
# Usage:
#   scripts/kieker_profile.sh <workspace> [iterations] [module-rel-path] [out-root]
# Env overrides:
#   JAVA_HOME, AJ_WEAVER (aspectjweaver jar), KIEKER_PROPS, AOP_XML, MVN, PERF_repeats
# Examples:
#   scripts/kieker_profile.sh teastore-lift              # 1 session, 30 in-session reps
#   PERF_repeats=100 scripts/kieker_profile.sh teastore-lift
#   scripts/kieker_profile.sh teastore-lift 5             # paper-UNFAITHFUL: 5 separate runs
# ─────────────────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

WS="${1:?usage: kieker_profile.sh <workspace> [iterations] [module] [out-root]}"
ITERATIONS="${2:-1}"
MODULE="${3:-services/tools.descartes.teastore.persistence}"
OUT_ROOT="${4:-$ROOT/profiling/runs/$(basename "$WS")_$(date +%Y%m%d_%H%M%S)}"

# ── toolchain resolution ─────────────────────────────────────────────────────
MVN="${MVN:-$(command -v mvn)}"
: "${JAVA_HOME:=/opt/homebrew/opt/openjdk}"     # the JDK mvn uses in this env (OpenJDK 25)
export JAVA_HOME
AJ_WEAVER="${AJ_WEAVER:-$HOME/.m2/repository/net/kieker-monitoring/kieker/2.0.0/kieker-2.0.0-aspectj.jar}"
KIEKER_PROPS="${KIEKER_PROPS:-$ROOT/profiling/kieker.local.properties}"
# Our aop.xml = TeaStore's exact instrumentation + JUnit-test-class excludes (so LTW doesn't
# break test invocation). It measures the same PRODUCTION methods the paper does.
AOP_XML="${AOP_XML:-profiling/aop.xml}"
case "$AOP_XML" in /*) AOP_ABS="$AOP_XML";; *) AOP_ABS="$ROOT/$AOP_XML";; esac

# ── perf-exercise workload ───────────────────────────────────────────────────
# The functional tests size their fixtures for readable assertions (four products in one
# category, and so on). Timing a catalog-wide query against those row counts measures
# EntityManager setup, not the query: a deliberately injected N+1 in getPriceOutliers
# measured 14527us senior vs 14536us junior — a 0.06% difference produced by data too
# small for the query's shape to matter. FeaturePerfExerciseTest (persistence test scope)
# generates a representative catalog and drives every catalog-wide query over it, but only
# when this property is set, so an ordinary `mvn test` — including the frozen-contract run
# that decides equivalence — is untouched. Sizes are overridable so a workspace that needs
# a bigger or smaller workload does not need a code change.
PERF_EXERCISE="${PERF_EXERCISE:-1}"
# The in-session repeat count the paper's protocol needs (see header). Always resolved and
# passed explicitly — not left to FeaturePerfExerciseTest's own default — so this script
# and meta.json (written below, read by parse_kieker.py's block-bootstrap labeling) agree
# on the exact number without parsing Java source.
PERF_REPEATS="${PERF_repeats:-30}"
PERF_PROPS=()
if [ "$PERF_EXERCISE" = "1" ]; then
  PERF_PROPS=(-Dteastore.perf.exercise=true -Dteastore.perf.repeats="$PERF_REPEATS")
  # `only` restricts the sweep to ONE query (PERF_only=<methodName>). Required for any profile
  # feeding the paper's propagation step: the full sweep's relative call counts describe the
  # sweep, not a request mix, and one expensive unrelated query then dominates station demand.
  # Measured on reorder — getNearDuplicatePairs took 83.9% of inclusive time, pushing the feature
  # under test down to 0.238% of demand, so a 22x local regression predicted +0.03 ms system-wide.
  for v in only categories productsPerCategory users maxOrdersPerUser \
           priceChangesPerProduct stockMovementsPerProduct; do
    eval "override=\${PERF_${v}:-}"
    [ -n "$override" ] && PERF_PROPS+=("-Dteastore.perf.$v=$override")
  done
fi

echo "═══════════════════════════════════════════════════════════════════"
echo " Kieker component profiling (paper: Liao et al. ICSE'25, local signal)"
echo "   workspace   : $WS"
echo "   module      : $MODULE"
echo "   iterations  : $ITERATIONS  (JVM sessions — paper protocol is 1)"
echo "   in-session repeats: $PERF_REPEATS  (paper protocol is 30)"
echo "   out         : $OUT_ROOT"
echo "   JAVA_HOME   : $JAVA_HOME"
echo "   weaver      : $AJ_WEAVER"
echo "   aop.xml     : $AOP_ABS"
echo "   perf-exercise: ${PERF_EXERCISE} ${PERF_PROPS[*]}"
echo "═══════════════════════════════════════════════════════════════════"

[ -d "$WS/$MODULE" ]    || { echo "✗ module not found: $WS/$MODULE"; exit 1; }
[ -f "$AJ_WEAVER" ]     || { echo "✗ aspectjweaver jar not found (set AJ_WEAVER)"; exit 1; }
[ -f "$AOP_ABS" ]       || { echo "✗ aop.xml not found (set AOP_XML)"; exit 1; }
[ -f "$KIEKER_PROPS" ]  || { echo "✗ kieker props not found: $KIEKER_PROPS"; exit 1; }

mkdir -p "$OUT_ROOT"
cp "$AOP_ABS" "$OUT_ROOT/aop.xml.used"
cp "$KIEKER_PROPS" "$OUT_ROOT/kieker.properties.used"
# Read by parse_kieker.py to rank-rebucket a single-session profile into `repeats` blocks
# for the paper_detect.py bootstrap, instead of one undifferentiated block. Only meaningful
# when the perf-exercise workload actually repeated the queries in-session.
#
# v2 (2026-08-15) additionally records what the run WAS, so scripts/measurement_gate.py can
# check it afterwards. v1 wrote only repeats+protocol, which left two claims unverifiable from
# the artifacts: that the workload was isolated (`only`, the F6 requirement), and that the
# profile is not older than the measurement code (`harness_fingerprint`). Do not drop these —
# a profile that cannot be audited is one that gets silently trusted.
if [ "$PERF_EXERCISE" = "1" ]; then
  HARNESS_FP="$(python3 -c 'import sys;sys.path.insert(0,sys.argv[1]);
import measurement_gate;print(measurement_gate.harness_fingerprint())' \
    "$PWD/scripts" 2>/dev/null || echo "unavailable")"
  printf '{"meta_version": 2, "repeats": %s, "protocol": "in-session", "only": %s, "iters": %s, "harness_fingerprint": "%s", "timestamp": "%s"}\n' \
    "$PERF_REPEATS" \
    "$([ -n "${PERF_only:-}" ] && printf '"%s"' "$PERF_only" || echo null)" \
    "$ITERATIONS" "$HARNESS_FP" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT_ROOT/meta.json"
fi

# ── TEST SCOPE: profile ONE test class, so the run is ONE JVM ────────────────────────────
# TeaStore's persistence pom sets <reuseForks>false</reuseForks> (inherited verbatim from
# upstream DescartesResearch/TeaStore), which forks a FRESH JVM per test class. Profiling the
# whole suite therefore produced ~26-51 Kieker sessions, every one of them JIT-COLD: each
# repository method was measured interpreted or barely C1-compiled, in the handful of calls an
# assertion test makes, and the parser then pooled 26-51 independent warm-up curves into one
# distribution. That is the most likely source of the control noise floors seen in the dataset
# (bundles' controls reach 542.9%) — cold first-call timings have exactly that heavy right tail.
#
# It CANNOT be fixed with -DreuseForks=true: an explicit <reuseForks> in the pom overrides the
# user property. Verified 2026-08-05 — 3 test classes still produced 3 sessions with the flag,
# with -Dsurefire.reuseForks=true, and without either.
#
# Scoping to the single perf-exercise class sidesteps it entirely and lands exactly on the
# paper's protocol. Verified on the trunk: 1 kieker session, 1 $0 metadata record, 17 operations
# at exactly 30 calls each (= teastore.perf.repeats, warm and in-session) — the same shape as
# Liao et al.'s Component_perf_res kieker.dat files.
#
# Set KIEKER_TEST_SCOPE to another class, or KIEKER_TEST_SCOPE="" to profile the whole suite
# (the old, cold-JVM behaviour — kept because a non-perf-exercise workspace has no such class).
TEST_SCOPE=()
if [ "$PERF_EXERCISE" = "1" ]; then
  _scope="${KIEKER_TEST_SCOPE-FeaturePerfExerciseTest}"
  if [ -n "$_scope" ]; then
    # -Dtest applies across the whole reactor, and -am drags in `entities`, which has tests but
    # none matching. Both opt-outs are needed: surefire 2.x honours failIfNoTests, 3.x renamed it
    # to surefire.failIfNoSpecifiedTests and hard-fails the build without it (persistence pins
    # 2.22.2, entities resolves 3.5.4 — so this reactor genuinely uses both).
    TEST_SCOPE=(-Dtest="$_scope" -DfailIfNoTests=false -Dsurefire.failIfNoSpecifiedTests=false)
    echo "  test scope: $_scope (one fork → one warm Kieker session)"
  fi
fi

# ── 0. build module + deps once (deps into ~/.m2 so per-iter test runs are fast) ──
echo "── [0/$ITERATIONS] clean + build module + deps ──"
"$MVN" -q -pl "$MODULE" -am -f "$WS/pom.xml" clean install -DskipTests \
  >"$OUT_ROOT/build.log" 2>&1 || { echo "✗ dependency build failed — see $OUT_ROOT/build.log"; tail -20 "$OUT_ROOT/build.log"; exit 1; }
echo "  ✓ deps installed"

# ── iterate: run the test suite under Kieker LTW, collect per-method timing ───
ok=0
for i in $(seq 1 "$ITERATIONS"); do
  ITER_DIR="$OUT_ROOT/iter_$(printf '%02d' "$i")"
  mkdir -p "$ITER_DIR"
  # AspectJ LTW + Kieker argLine — built by the SHARED single-source-of-truth helper
  # (profiling/kieker_args.py), so this harness and dual_track.py never drift.
  ARGLINE="$(AJ_WEAVER="$AJ_WEAVER" AOP_XML="$AOP_ABS" KIEKER_PROPS="$KIEKER_PROPS" \
      python3 "$ROOT/profiling/kieker_args.py" "$ITER_DIR")"

  echo "── [$i/$ITERATIONS] mvn test under Kieker → $ITER_DIR ──"
  # -am: build sibling modules from the reactor (TeaStore's ${teastoreversion} parent ref
  # only resolves in-reactor, not standalone from ~/.m2). Only 'entities' has a test, and it
  # is weave-excluded, so the Kieker output stays persistence-only.
  if "$MVN" -q -pl "$MODULE" -am -f "$WS/pom.xml" \
        test -DargLine="$ARGLINE" "${TEST_SCOPE[@]}" "${PERF_PROPS[@]}" \
        >"$ITER_DIR/mvn.log" 2>&1; then
    n=$(find "$ITER_DIR" -name '*.dat' 2>/dev/null | wc -l | tr -d ' ')
    echo "   ✓ iteration $i done — $n kieker .dat file(s)"
    ok=$((ok+1))
  else
    echo "   ✗ iteration $i FAILED — see $ITER_DIR/mvn.log"; tail -15 "$ITER_DIR/mvn.log"
  fi
done

echo "═══════════════════════════════════════════════════════════════════"
echo " $ok/$ITERATIONS iterations produced output → $OUT_ROOT"
echo " Next: python3 profiling/parse_kieker.py $OUT_ROOT"
echo "═══════════════════════════════════════════════════════════════════"
