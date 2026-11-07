#!/usr/bin/env zsh
# lib_trunk_chain.sh — decision-logic functions shared by run_teastore_chained_history.sh.
#
# Pulled into its own file (no top-level side effects — only function definitions) so it
# can be `source`d safely by the no-LLM test suite (agile_team_cline/tests/test_trunk_chain.py)
# without executing the batch script's cd/.env/main-loop side effects.
#
# Depends on the caller having set: DUAL_DATASET, TRUNK_DIR, HIST_DIR.

# ── interpreter resolution ────────────────────────────────────────────────────────────────
# Every decision function below is a Python heredoc, and all of them were written as bare
# `python`. There is no `python` on a modern Homebrew/macOS PATH — only `python3` — and the
# failure is SILENT IN THE WORST DIRECTION: the shell writes "command not found" to stderr and
# the function exits non-zero, which each caller reads as a routine "no". So `_feature_already_done`
# answers "not done" (harmless, just re-runs), `_feature_landable` answers "not landable" (the
# trunk never advances), and `_prune_unexercised_row` deletes nothing at all — which is why six
# unexercised rows survived a pruning batch in production. Nothing in the chain announces that
# its entire decision layer is inert.
#
# `$0` is captured HERE, at source time, and not inside `_py`: zsh sets FUNCTION_ARGZERO by
# default, so `$0` within a function is the function's own name, not the file's path.
_TRUNK_LIB_DIR="${0:A:h}"

# Echo the interpreter to run the heredocs with; non-zero and loud if there is none.
# Order matters: the venv first, since the dataset helpers import the project's own modules.
_py() {
  local c
  for c in "$TRUNK_PY" "$_TRUNK_LIB_DIR/../.venv/bin/python" python3 python; do
    [[ -n "$c" ]] && command -v "$c" >/dev/null 2>&1 && { echo "$c"; return 0; }
  done
  echo "lib_trunk_chain.sh: FATAL — no Python interpreter (tried \$TRUNK_PY, .venv/bin/python," \
       "python3, python). Every chain decision below depends on one; refusing to guess." >&2
  return 1
}

# ── resume: skip a feature only when it already produced a VALID, COMPLETE row ──
# A row is a complete measurement (→ done, skip) when it was admitted, OR the feature
# was actually exercised (feature_exercised is True), OR it predates the field entirely
# (key absent → legacy / kieker-off back-compat). A row whose feature_exercised is present
# but null(None)/False is an INCOMPLETE measurement and must NOT count as done — leaving it
# to be re-run is the whole point, otherwise the invalid row silently blocks reprocessing
# AND (via _feature_landable below) an unexercised feature would still seed the trunk.
#   • False  = profiled but the feature code was never hit (tests landed outside the module).
#   • null   = a track (usually the JUNIOR) failed to COMPILE, so its profiler recorded
#              nothing and the top-level signal collapsed to null — NOT False. This is the
#              common real failure (the 2026-07-21 reorder trap): keying on `is not False`
#              read null as done and skipped the feature forever. None/False both re-run now.
# We deliberately do NOT require `admitted`: a fully-measured but non-equivalent row (both
# tracks profiled, feature_exercised=True, admitted=False — a real regression) is complete data.
_feature_already_done() {
  local fid="$1"
  [[ -f "$DUAL_DATASET" ]] || return 1
  "$(_py)" - "$DUAL_DATASET" "$fid" <<'PY'
import json, sys
ds, fid = sys.argv[1], sys.argv[2]
valid = False
with open(ds) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("feature_id") == fid:
            # last row for this id wins; complete iff admitted, exercised-AND-green, or
            # pre-field row. The green-junior conjunct is HIGH 5 of the 2026-07-26 audit:
            # `fe is True` alone accepted a row whose JUNIOR FAILED its tests, which is an
            # invalid sample by this pipeline's own definition — and because the row still
            # counted as done, the feature was skipped forever and could never get a valid
            # pair. Live instance: volatility row 13 (induced, junior.tests_pass=False,
            # admitted=False). reorder escaped the same trap only via a manual prune on
            # 2026-07-21. A non-equivalent row whose junior is merely SLOWER still passes
            # its tests, so genuine regression samples are unaffected.
            fe = row.get("feature_exercised", "__absent__")
            jr_pass = (row.get("junior") or {}).get("tests_pass") is True
            valid = ((row.get("admitted") is True)
                     or (fe is True and jr_pass)
                     or (fe == "__absent__"))
sys.exit(0 if valid else 1)
PY
}

# ── drop any stale UNMEASURED rows for a feature_id so a re-run appends a clean row
# instead of accumulating a duplicate beside a row that holds no data. Called just before
# a fresh (non-skip) run. Rewrites the dataset in place.
#
# Unmeasured == feature_exercised is present but NOT True (null on a junior COMPILE
# failure, which is the common case; False on an unexercised run), and the row was not
# admitted. The old `is False`-only guard never matched the nulls, so they were never
# cleaned.
#
# This predicate is DELIBERATELY NARROWER than _feature_already_done's. The two answer
# different questions and must not share a rule:
#
#   _feature_already_done  — "is this feature finished?"   A red junior is not a valid
#                            pair, so the feature is re-run. (HIGH 5, 2026-07-26.)
#   _prune_unexercised_row — "does this row hold data?"     A row with
#                            feature_exercised=True was measured; both tracks profiled and
#                            the numbers are real even though the junior broke equivalence.
#                            It is KEPT as provenance of the attempt.
#
# They were exact negations until 2026-08-16, which meant a re-run silently DELETED its own
# measured history. The 6 rows of that shape in the corpus (churn x2, reorder, rfm x3 — all
# junior.tests_pass=False, admitted=False) survive only because those retries went through
# scripts/rerun_chained_junior.sh, which never prunes. Keeping them is now intentional
# rather than accidental. Safe because the canonical reader (dataset_view.py) is
# last-row-wins, so a retained superseded row changes no derived result.
#
# Admitted rows and pre-field legacy rows are ALWAYS kept. ──
_prune_unexercised_row() {
  local fid="$1"
  [[ -f "$DUAL_DATASET" ]] || return 0
  "$(_py)" - "$DUAL_DATASET" "$fid" <<'PY'
import json, os, sys
ds, fid = sys.argv[1], sys.argv[2]
kept, dropped = [], 0
with open(ds) as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        try:
            row = json.loads(s)
        except Exception:
            kept.append(line.rstrip("\n")); continue
        fe = row.get("feature_exercised", "__absent__")
        unmeasured = (fe != "__absent__" and fe is not True
                      and row.get("admitted") is not True)
        if row.get("feature_id") == fid and unmeasured:
            dropped += 1
            continue
        kept.append(line.rstrip("\n"))
if dropped:
    tmp = ds + ".tmp"
    with open(tmp, "w") as f:
        for l in kept:
            f.write(l + "\n")
    os.replace(tmp, ds)
    print(f"  🧹 pruned {dropped} stale unexercised row(s) for {fid} before re-run")
PY
}

# ── quota check: an infra-abort stops the batch instead of burning more features ──
_infra_aborted() {
  local out_prefix="$1"
  local ws_prefix="${WORKSPACE_ROOT:+${WORKSPACE_ROOT}/}${out_prefix}"
  if [[ -f "${ws_prefix}-senior/INFRA_ABORT.json" || -f "${ws_prefix}-junior/INFRA_ABORT.json" ]]; then
    return 0
  fi
  [[ -f "$DUAL_DATASET" ]] || return 1
  "$(_py)" - "$DUAL_DATASET" "$out_prefix" <<'PY'
import json, sys
ds, fid = sys.argv[1], sys.argv[2]
status = None
with open(ds) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("feature_id") == fid:
            status = row.get("status")
sys.exit(0 if status == "infra_abort" else 1)
PY
}

# ── trunk-advance gate: senior's OWN tests must be green (independent of junior
# equivalence) — mirrors a real team merging on green tests, not mere compilation ──
_senior_tests_pass() {
  local out_prefix="$1"
  [[ -f "$DUAL_DATASET" ]] || return 1
  "$(_py)" - "$DUAL_DATASET" "$out_prefix" <<'PY'
import json, sys
ds, fid = sys.argv[1], sys.argv[2]
ok = False
with open(ds) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("feature_id") == fid:
            ok = bool(row.get("senior", {}).get("tests_pass"))
sys.exit(0 if ok else 1)
PY
}

# ── the profiler must actually have exercised the feature. Blocks ONLY on a positive
# finding it wasn't (feature_exercised is exactly False); None/missing (unprofiled)
# does not block, mirroring dual_track._admit_row's semantics. ──
_feature_exercised_ok() {
  local fid="$1"
  [[ -f "$DUAL_DATASET" ]] || return 1
  "$(_py)" - "$DUAL_DATASET" "$fid" <<'PY'
import json, sys
ds, fid = sys.argv[1], sys.argv[2]
blocked = False
with open(ds) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("feature_id") == fid:
            blocked = row.get("feature_exercised") is False
sys.exit(1 if blocked else 0)
PY
}

# ── SENIOR-SPECIFIC exercised gate (2026-07-18). The top-level `feature_exercised`
# is `senior_exercised AND junior_exercised` and collapses to None whenever the junior
# fails to COMPILE (its profiler records nothing) — regardless of whether the SENIOR
# was measured. Since only the senior lands on the trunk, gating the advance on that
# conflated field could not tell "senior exercised, junior broke" (advance is correct)
# apart from "senior NOT exercised, junior broke" (must NOT advance): both read None
# and both passed. This reads `senior.feature_exercised` (persisted by dual_track.py)
# and blocks ONLY on a positive False; None/missing (unprofiled, or an old pre-2026-07-18
# row) does not block, preserving the existing None-passes semantics. ──
_senior_feature_exercised_ok() {
  local fid="$1"
  [[ -f "$DUAL_DATASET" ]] || return 1
  "$(_py)" - "$DUAL_DATASET" "$fid" <<'PY'
import json, sys
ds, fid = sys.argv[1], sys.argv[2]
blocked = False
with open(ds) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("feature_id") == fid:
            blocked = row.get("senior", {}).get("feature_exercised") is False
sys.exit(1 if blocked else 0)
PY
}

# ── trunk-advance / chain-parent gate: senior's own tests green AND the SENIOR feature
# was actually measured. This couples trunk state to dataset validity so an UNEXERCISED
# feature (tests_pass=True on baseline tests alone, feature never profiled) can no
# longer silently become the chain baseline for every downstream feature — and, unlike
# the old conflated gate, a junior that merely fails to compile no longer masks whether
# the senior (the code that lands) was genuinely exercised. ──
_feature_landable() {
  local fid="$1"
  _senior_tests_pass "$fid" && _senior_feature_exercised_ok "$fid"
}

# ── idempotent trunk advance: safe to call on EVERY iteration (fresh run or
# resume-skip). Keyed off the test-point snapshot's existence, so a batch that
# dies between "row written" and "trunk advanced" self-heals on the next
# invocation instead of silently orphaning that feature's code out of the chain. ──
# ── idempotent, authoritative chain manifest (separate from the per-row chain
# block in the dataset). Upserts one test-point entry keyed on sequence; records the
# AUTHORITATIVE landed_on_trunk + snapshot (known only here, after the advance). ──
_update_chain_manifest() {
  local idx="$1" prefix="$2" parent="$3" snap_basename="$4"
  local manifest="${CHAIN_MANIFEST:-chain_manifest.json}"
  "$(_py)" - "$manifest" "$idx" "$prefix" "$parent" "$snap_basename" \
           "${TRUNK_DIR:-}" "${DUAL_DATASET:-}" "${HIST_DIR:-}" "${WORKSPACE_ROOT:-}" <<'PY'
import datetime, json, os, sys
manifest, idx, prefix, parent, snap, trunk, dataset, histdir, ws_root = sys.argv[1:10]
idx = int(idx)
try:
    with open(manifest) as f:
        doc = json.load(f)
except Exception:
    doc = {}
doc.setdefault("trunk_dir", trunk)
doc.setdefault("dataset", dataset)
if os.getenv("TEASTORE_BASELINE_GENESIS"):
    doc.setdefault("genesis_baseline", os.getenv("TEASTORE_BASELINE_GENESIS"))
senior_out_dir = os.path.join(ws_root, f"{prefix}-senior") if ws_root else f"{prefix}-senior"
# `advanced_at` records WHEN this feature actually landed on the trunk, so it must survive
# re-upserts. This function is called unconditionally from _ensure_trunk_advanced's
# already-advanced early-return, so overwriting it with now() on every resume destroys the
# provenance it exists to record (HIGH 8, docs/PIPELINE-AUDIT-2026-07-26.md: a single
# reconcile pass during the feature-11 run collapsed ten real landing times onto
# 2026-07-22T15:18:17/18). Preserve any existing value; only stamp it when absent.
prior = next((tp for tp in doc.get("test_points", []) if tp.get("sequence") == idx), {})
entry = {
    "sequence": idx,
    "feature_id": prefix,
    "parent": parent,
    "senior_out_dir": senior_out_dir,
    "landed_on_trunk": True,
    "snapshot": os.path.join(histdir, snap) if histdir else snap,
    "advanced_at": prior.get("advanced_at")
                   or datetime.datetime.now().isoformat(timespec="seconds"),
}
tps = [tp for tp in doc.get("test_points", []) if tp.get("sequence") != idx]
tps.append(entry)
tps.sort(key=lambda tp: tp.get("sequence", 0))
doc["test_points"] = tps
tmp = manifest + ".tmp"
with open(tmp, "w") as f:
    json.dump(doc, f, indent=2)
os.replace(tmp, manifest)
PY
}

_ensure_trunk_advanced() {
  local idx="$1" out_prefix="$2" parent="${3:-genesis}"
  local tp; tp=$(printf "tp-%02d-%s" "$idx" "$out_prefix")
  local snap="${HIST_DIR}/${tp}.tar.gz"

  if [[ -f "$snap" ]]; then
    _update_chain_manifest "$idx" "$out_prefix" "$parent" "${tp}.tar.gz"
    return 0   # already advanced for this test point — idempotent no-op (manifest reconciled)
  fi

  if _feature_landable "$out_prefix"; then
    echo "  🌳 advancing trunk for ${out_prefix} (senior tests green + feature exercised) ..."
    local advance_script="${TRUNK_CHAIN_LIB_DIR:-$_TRUNK_LIB_DIR}/advance_trunk.sh"
    local senior_ws="${WORKSPACE_ROOT:+${WORKSPACE_ROOT}/}${out_prefix}-senior"
    if zsh "$advance_script" "$TRUNK_DIR" "$senior_ws"; then
      tar czf "$snap" -C "$TRUNK_DIR" .
      _update_chain_manifest "$idx" "$out_prefix" "$parent" "${tp}.tar.gz"
      echo "  ── test point recorded: ${tp}"
    else
      echo "  ✗ advance_trunk.sh refused/failed — trunk left at its previous state (see error above)"
    fi
  else
    echo "  ⏸️  trunk NOT advanced for ${out_prefix} (senior tests not green, feature not"
    echo "      exercised by the profiled suite, or no row yet) —"
    echo "      next feature chains off the same (unchanged) trunk state"
  fi
}
