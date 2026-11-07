#!/bin/zsh
# Pre-run mitigation for the pom.xml deleter.
#
# ROOT CAUSE (proven 2026-06-19 via pom_watch instrumentation): cline auto-starts a
# persistent `cline-hub-daemon --cwd <dir>` per working directory that OUTLIVES the
# session. A stale daemon whose --cwd is a parent of the run workspace (e.g. the project
# root) reconciles/prunes the whole tree when cline is next invoked, deleting a batch of
# module pom.xml files early in the run (caught at 20:28:12, "no cline session yet", only
# hub-daemons alive; pipeline hygiene ran 20 min later → pipeline ruled out by timing).
#
# The deleter-agnostic pom-guard (_restore_missing_poms before every compile) already
# makes runs CORRECT (0 'Child module does not exist' aborts in the instrumented run).
# This script is the PREVENT-AT-SOURCE complement: kill stale hub daemons before a run so
# no cross-tree reconciler is watching the workspace. Run it before staging a fresh run.
#
# Usage:
#   zsh kill_cline_daemons.sh          # list, then kill all cline-hub-daemons
#   zsh kill_cline_daemons.sh --dry    # list only, kill nothing

DRY=0
[[ "$1" == "--dry" ]] && DRY=1

echo "## current cline-hub-daemon processes ##"
ps axo pid,etime,command | grep "cline-hub-daemon" | grep -v grep | sed -E 's#(--cwd [^ ]+).*#\1#'
COUNT=$(pgrep -f "cline-hub-daemon" | wc -l | tr -d ' ')
echo "found: ${COUNT} daemon(s)"

if [[ $DRY -eq 1 ]]; then
  echo "(--dry: nothing killed)"
  exit 0
fi

if [[ "$COUNT" -gt 0 ]]; then
  pkill -f "cline-hub-daemon"
  sleep 1
  REMAIN=$(pgrep -f "cline-hub-daemon" | wc -l | tr -d ' ')
  echo "killed; remaining: ${REMAIN} (cline will respawn a workspace-scoped daemon on next invocation)"
fi
