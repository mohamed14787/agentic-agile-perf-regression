from datetime import datetime
from agile_team_cline.shared.blackboard import Blackboard


class ScrumMaster:
    def __init__(self, blackboard: Blackboard):
        self.bb = blackboard

    def init_sprint(self, feature: str, stack: str) -> None:
        import os
        # Carry tracked state forward across milestones — losing sprint_num/
        # failed_tasks breaks tech_debt numbering and failure history.
        prior = self.bb.read("blackboard") or {}
        if not isinstance(prior, dict):
            prior = {}
        prior_files = list(prior.get("generated_files", []))
        seen = set(prior_files)

        # BROWNFIELD: If this is a brownfield project (existing workspace),
        # pre-populate with discovered files so baseline tracking is correct
        workspace_mapping = self.bb.read("workspace_mapping") or {}
        if workspace_mapping.get("is_brownfield") and len(prior_files) == 0:
            # First milestone in brownfield: add all existing files
            existing = workspace_mapping.get("existing_files", [])
            prior_files.extend(existing)
            seen.update(existing)
            print(f"   📊 Pre-existing files: {len(existing)}")
            print(f"      Baseline for new features: file #{len(existing)+1}")

        # Refresh from disk in case files were created out-of-band, but only
        # source/config types and skip blackboard JSON sidecars.
        path_dir = self.bb.output_dir
        skip_names = {"blackboard.json", "contract.json", "sprint_backlog.json",
                      "sprint_summary.json", "test_report.json", "workspace_mapping.json"}
        if path_dir.exists():
            for root, _, files in os.walk(path_dir):
                for fname in files:
                    if not fname.endswith((".java", ".properties", ".xml", ".yml", ".yaml", ".sql")):
                        continue
                    rel_path = os.path.relpath(os.path.join(root, fname), path_dir)
                    if fname in skip_names or rel_path in skip_names or fname in seen or rel_path in seen:
                        continue
                    prior_files.append(rel_path)
                    seen.add(rel_path)

        self.bb.write("blackboard", {
            "feature": feature,
            "stack": stack,
            "sprint_num": prior.get("sprint_num", 0),
            "sprint_backlog": [],
            "generated_files": prior_files,
            # Baseline so consumers can distinguish files generated THIS milestone
            # from carryover. _run_dev_sprint uses this to scope prior-file context.
            "milestone_file_baseline": len(prior_files),
            "blockers": prior.get("blockers", []),
            "failed_tasks": prior.get("failed_tasks", []),
            "started_at": prior.get("started_at") or datetime.utcnow().isoformat(),
        })

        # Persist a source snapshot (relpath -> [mtime, size]) of the PRE-development
        # workspace so dev-sprint skip-guards can tell an untouched pre-existing /
        # carryover file from one THIS run actually wrote or modified. Without it,
        # brownfield "add a method to ExistingFile.java" tasks false-skip — their target
        # already exists on disk (generated_files is pre-seeded with the whole tree above)
        # — and the sprint reports green while writing nothing (TeaStore perf runs #33-35).
        self.bb.write("baseline_src", self.bb.snapshot_src())

    def set_backlog(self, backlog: list) -> None:
        # Only update the backlog. sprint_num is owned by advance_sprint —
        # writing it here resets the counter when called mid-pipeline for sprint 2+.
        self.bb.update("blackboard", {"sprint_backlog": backlog})

    def tasks_for_sprint(self, sprint_num: int) -> list:
        bb = self.bb.read("blackboard")
        return [t for t in bb.get("sprint_backlog", []) if t.get("sprint") == sprint_num]

    def mark_task(self, task_id: str, status: str) -> None:
        bb = self.bb.read("blackboard")
        for task in bb.get("sprint_backlog", []):
            if task["id"] == task_id:
                task["status"] = status
        self.bb.write("blackboard", bb)

    def advance_sprint(self, sprint_num: int) -> None:
        self.bb.update("blackboard", {"sprint_num": sprint_num})

    def standup(self, sprint_num: int) -> None:
        bb = self.bb.read("blackboard")
        backlog = bb.get("sprint_backlog", [])
        max_sprint = max([t.get("sprint", 3) for t in backlog] + [3])
        if sprint_num > max_sprint:
            tasks = [t for t in backlog if t.get("status") in ["in_progress", "done"]]
        else:
            tasks = [t for t in backlog if t.get("sprint") == sprint_num]
        done = [t["id"] for t in tasks if t.get("status") == "done"]
        in_progress = [t["id"] for t in tasks if t.get("status") == "in_progress"]
        failed = bb.get("failed_tasks", [])
        files = bb.get("generated_files", [])
        print(f"\n{'='*50}")
        print(f"  STANDUP — Sprint {sprint_num}")
        print(f"  Done     : {done or 'none'}")
        print(f"  In Prog  : {in_progress or 'none'}")
        print(f"  Failed   : {failed or 'none'}")
        print(f"  Files    : {len(files)} generated")
        print(f"{'='*50}\n")
