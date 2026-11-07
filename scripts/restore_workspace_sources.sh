#!/usr/bin/env zsh
# restore_workspace_sources.sh <workspace> — restore MISSING source files from the
# workspace's own src_snapshot.tar.gz (written by dual_track.stage_workspace at staging
# time, agile_team_cline/dual_track.py _write_src_snapshot). Complements the pom-only
# guard (agile_team_cline/sprint.py _restore_missing_poms) — the Jun-28 deleter event
# removed whole baseline PACKAGES, not just poms.
#
# NEVER overwrites an existing file (rsync --ignore-existing) — feature edits already on
# disk are always preserved, exactly like the pom guard and the manual `rsync
# --ignore-existing` recovery used on 2026-07-02.
#
# Usage: zsh scripts/restore_workspace_sources.sh teastore-lift

set -e
WS="${1:?usage: $0 <workspace>}"
SNAP="$WS/src_snapshot.tar.gz"

[[ -d "$WS" ]] || { echo "✗ workspace not found: $WS"; exit 1; }
[[ -f "$SNAP" ]] || {
  echo "✗ no snapshot at $SNAP (workspace predates the snapshot guard, or was staged"
  echo "  with DUAL_SRC_SNAPSHOT=0)"; exit 1;
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "extracting $SNAP ..."
tar -xzf "$SNAP" -C "$TMP"

echo "restoring files missing from $WS (never overwrites files already on disk) ..."
# --ignore-existing does NOT skip directories (only files) — rsync still emits an
# --out-format line for every directory it traverses even when nothing in it changed, so
# a raw line count wildly overstates the restore. Filter to actual FILE entries (no
# trailing '/') for both the printed lines and the final count.
N=$(rsync -a --ignore-existing --out-format='restored: %n' "$TMP"/ "$WS"/ \
    | grep -v '/$' | tee /dev/stderr | wc -l | tr -d ' ')
echo "── ${N} file(s) restored"
