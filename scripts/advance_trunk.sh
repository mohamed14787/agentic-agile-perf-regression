#!/usr/bin/env zsh
# advance_trunk.sh <trunk_dir> <accepted_senior_workspace> — merge an accepted feature's
# senior output into the persistent chained-history trunk (agile_team_cline/dual_track.py
# stays untouched; this is pure orchestration, called by run_teastore_chained_history.sh).
#
# Unlike restore_workspace_sources.sh (which only fills in MISSING files, never overwrites),
# this does a FULL mirror: the senior workspace is authoritative for this test point — it
# was itself staged as a full copy of the trunk's previous state, so it's normally a strict
# evolution of the trunk minus harness bookkeeping files (excluded below).
#
# Deletion guard: this repo has a documented history of workspace-scale source deletions
# (Jun-28 event — 36/38 workspaces gutted, whole baseline trees removed by an unattributed
# process). If that ever hits a senior workspace, a naive `rsync --delete` mirror would
# propagate the destruction into the trunk PERMANENTLY, corrupting every future feature in
# the chain. So: dry-run first, count pending deletions, abort loudly above the threshold.
# Legitimate feature work essentially never deletes baseline files; a deleter event deletes
# hundreds.
#
# Usage: zsh scripts/advance_trunk.sh teastore-trunk teastore-copurchase-v3-senior
# Env:   ADVANCE_MAX_DELETES=10   ADVANCE_FORCE=0

set -e
TRUNK_DIR="${1:?usage: $0 <trunk_dir> <accepted_senior_workspace>}"
SENIOR_WS="${2:?usage: $0 <trunk_dir> <accepted_senior_workspace>}"
MAX_DELETES="${ADVANCE_MAX_DELETES:-10}"
FORCE="${ADVANCE_FORCE:-0}"

[[ -d "$SENIOR_WS" ]] || { echo "✗ senior workspace not found: $SENIOR_WS"; exit 1; }
mkdir -p "$TRUNK_DIR"

EXCLUDES=(
  --exclude target --exclude .git
  --exclude contract.json --exclude 'profile/'
  --exclude src_snapshot.tar.gz --exclude INFRA_ABORT.json
  --exclude token_usage.jsonl --exclude '*.log'
  --exclude pom_snapshot.json
  # agile-team scratch/bookkeeping files — feature-specific, must NOT bleed into the
  # trunk (a leaked milestone_checkpoint.json/sprint_backlog.json makes the next
  # feature's pipeline think its own milestones are already done and skip real work —
  # see 2026-07-14 teastore-hist-dupes incident: admitted=False, feature_exercised=False)
  --exclude blackboard.json --exclude sprint_backlog.json
  --exclude sprint_summary.json --exclude milestone_checkpoint.json
  --exclude conventions.json --exclude vf_failures.json
  --exclude test_report.json --exclude baseline_src.json
  --exclude workspace_mapping.json --exclude '*_cline.jsonl'
  --exclude camel_token_usage.jsonl
  # TeaStore is fully modular — all real source lives under a module (services/…, etc.).
  # A bare repo-ROOT src/ tree is therefore always junk: it's where a coder agent
  # mistakenly wrote test files that no module compiles (2026-07-14 dupes/copurchase
  # test-path bug). Anchored (leading slash) so it excludes ONLY root src/, never a
  # module's own services/*/src/. Belt-and-suspenders alongside the sprint.py path fix.
  --exclude '/src'
)

echo "checking for pending deletions ($SENIOR_WS -> $TRUNK_DIR) ..."
DELETIONS=$(rsync -a -n --delete --itemize-changes "${EXCLUDES[@]}" \
    "$SENIOR_WS"/ "$TRUNK_DIR"/ | grep '^\*deleting' | grep -v '/$' || true)
N_DELETIONS=$(echo -n "$DELETIONS" | grep -c '^\*deleting' || true)

if [[ "$N_DELETIONS" -gt "$MAX_DELETES" && "$FORCE" != "1" ]]; then
  echo "✗ REFUSING to advance trunk: ${N_DELETIONS} file(s) would be deleted"
  echo "  (threshold ADVANCE_MAX_DELETES=${MAX_DELETES}) — this looks like a workspace-scale"
  echo "  source deletion, not normal feature work. First 20 doomed files:"
  echo "$DELETIONS" | head -20 | sed 's/^/    /'
  echo "  If this is genuinely intended, re-run with ADVANCE_FORCE=1."
  exit 1
fi

if [[ "$N_DELETIONS" -gt "$MAX_DELETES" ]]; then
  echo "advancing trunk (${N_DELETIONS} deletion(s), ABOVE guard threshold, ADVANCE_FORCE=1 override) ..."
else
  echo "advancing trunk (${N_DELETIONS} deletion(s), within guard) ..."
fi
N=$(rsync -a --delete "${EXCLUDES[@]}" --out-format='changed: %n' \
    "$SENIOR_WS"/ "$TRUNK_DIR"/ | grep -v '/$' | tee /dev/stderr | wc -l | tr -d ' ')
echo "── trunk advanced: ${N} file(s) changed"
