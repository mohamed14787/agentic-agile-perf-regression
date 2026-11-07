import json
from pathlib import Path
from typing import Any


class Blackboard:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{name}.json"

    def read(self, name: str) -> Any:
        p = self._path(name)
        if not p.exists():
            return {}
        with open(p) as f:
            return json.load(f)

    def write(self, name: str, data: Any) -> None:
        with open(self._path(name), "w") as f:
            json.dump(data, f, indent=2)

    def update(self, name: str, updates: dict) -> None:
        current = self.read(name)
        if isinstance(current, dict):
            current.update(updates)
        else:
            current = updates
        self.write(name, current)

    def resolve_path(self, filename: str, content: str = "") -> str:
        import os
        # Avoid path-doubling if filename is already a resolved relative path
        normalized = filename.replace("\\", "/")
        if "src/main/" in normalized or "src/test/" in normalized:
            return normalized

        if filename == "pom.xml":
            return filename
        elif filename in ["application.properties", "application.yml", "schema.sql", "data.sql"]:
            return os.path.join("src", "main", "resources", filename)
        elif filename.endswith(".java"):
            package_name = ""
            if content:
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("package ") and stripped.endswith(";"):
                        package_name = stripped[8:-1].strip()
                        break
            if not package_name:
                package_name = "com.example"
            package_path = package_name.replace(".", "/")
            filename_lower = filename.lower()
            if filename_lower.endswith("test.java") or filename.endswith("IT.java") or "test" in filename_lower:
                return os.path.join("src", "test", "java", package_path, filename)
            else:
                return os.path.join("src", "main", "java", package_path, filename)
        return filename

    def write_file(self, filename: str, content: str) -> str:
        resolved_path = self.resolve_path(filename, content)
        full_path = self.output_dir / resolved_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return resolved_path

    def record_file(self, filename: str) -> None:
        bb = self.read("blackboard")
        gen_files = bb.setdefault("generated_files", [])
        if filename not in gen_files:
            gen_files.append(filename)
        self.write("blackboard", bb)

    def record_failure(self, task_id: str) -> None:
        bb = self.read("blackboard")
        bb.setdefault("failed_tasks", []).append(task_id)
        self.write("blackboard", bb)

    def clear_failure(self, task_id: str) -> None:
        """Drop a task from failed_tasks once its file exists, so stale failures don't
        perpetuate re-planning/standup noise across sprints and refinement rounds."""
        bb = self.read("blackboard")
        failed = bb.get("failed_tasks")
        if failed and task_id in failed:
            bb["failed_tasks"] = [t for t in failed if t != task_id]
            self.write("blackboard", bb)

    # ── Cline-path file tracking ──────────────────────────────────────────
    # The agentic coder writes files itself, so we detect what it produced by
    # diffing a snapshot of the project tree instead of capturing write_file()'s
    # return path. Tracked set = src/** plus pom.xml (exactly what resolve_path
    # emits), which excludes the JSON blackboard state and Maven's target/ output.
    def snapshot_src(self) -> dict:
        """Map tracked relpath -> (mtime, size) for the current project tree."""
        import os
        root = str(self.output_dir)
        snap = {}
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
            for fn in files:
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace("\\", "/")
                # Track src files under ANY module (reactor: <module>/src/main/...), not just
                # a single root src/ tree — otherwise new files in a module are invisible to
                # generated_files → find_existing false-fails completion (TeaStore run #7).
                norm = "/" + rel
                if (rel.startswith("src/") or "/src/main/" in norm or "/src/test/" in norm
                        or os.path.basename(rel) == "pom.xml"):
                    try:
                        st = os.stat(full)
                        snap[rel] = (st.st_mtime, st.st_size)
                    except OSError:
                        pass
        return snap

    def record_changed_since(self, before: dict) -> list:
        """Append (deduped, sorted) every new/modified tracked file vs `before` to
        generated_files via record_file, preserving its order/format."""
        after = self.snapshot_src()
        changed = sorted(rel for rel, meta in after.items() if before.get(rel) != meta)
        for rel in changed:
            self.record_file(rel)
        return changed

    def find_existing(self, filename: str) -> str:
        """Return a recorded generated_files relpath matching `filename` (by exact
        relpath or basename) that exists on disk, else "". Exact-relpath matches are
        checked across ALL entries before any basename match — basename alone is
        ambiguous for files that exist at multiple paths (e.g. db/h2/schema.sql vs
        db/mysql/schema.sql vs target/classes/db/h2/schema.sql)."""
        import os
        target = filename.replace("\\", "/")
        base = os.path.basename(target)
        bb = self.read("blackboard")
        entries = bb.get("generated_files", [])
        for rel in entries:
            if rel == target and (self.output_dir / rel).exists():
                return rel
        for rel in entries:
            if os.path.basename(rel) == base and (self.output_dir / rel).exists():
                return rel
        return ""
