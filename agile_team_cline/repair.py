"""Deterministic, compiler-error-driven semantic repair for generated Java projects.

Builds a project-wide symbol table, runs `mvn`, and applies minimal guaranteed-correct
edits to reconcile each error against the table — instead of regenerating whole files via
the LLM. Scope (v1, safe high-value): intra-project import correction + missing repository
derived-finders, plus the existing source hygiene (javax→jakarta, single/zero main class,
bootstrap name-align, test-source dedup) which lives in sprint.py. Fuzzy accessor/return-
type drift is intentionally left to the LLM refinement loop.
"""
import json
import os
import re
import subprocess
import zipfile

# --- regex fallbacks (used when javalang can't parse a broken/partial file) ----------
_PKG_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)
_TYPE_RE = re.compile(
    r"\bpublic\s+(?:final\s+|abstract\s+|sealed\s+|non-sealed\s+)*"
    r"(class|interface|enum|record)\s+([A-Za-z_]\w*)")
_REPO_RE = re.compile(
    r"extends\s+(?:JpaRepository|CrudRepository|PagingAndSortingRepository|"
    r"JpaSpecificationExecutor)\s*<\s*([A-Za-z_]\w*)\s*,\s*([A-Za-z_][\w.]*)\s*>")


def _render_type(tp) -> str:
    """Best-effort render of a javalang type node to a Java type string."""
    if tp is None:
        return "void"
    try:
        name = getattr(tp, "name", None) or "Object"
        args = getattr(tp, "arguments", None)
        if args:
            inner = ", ".join(_render_type(a.type) if getattr(a, "type", None) else "?" for a in args)
            name = f"{name}<{inner}>"
        dims = getattr(tp, "dimensions", None)
        if dims:
            name += "[]" * len(dims)
        return name
    except Exception:
        return "Object"


def _members(t) -> list:
    """Public member signatures (methods, constructors, public fields) of a
    javalang type declaration — interface methods are implicitly public."""
    out = []
    is_iface = type(t).__name__ == "InterfaceDeclaration"
    for m in (getattr(t, "body", None) or []):
        cls = type(m).__name__
        mods = getattr(m, "modifiers", set()) or set()
        try:
            if cls == "MethodDeclaration":
                if not is_iface and "public" not in mods:
                    continue
                params = ", ".join(_render_type(p.type) for p in (m.parameters or []))
                out.append(f"{_render_type(m.return_type)} {m.name}({params})")
            elif cls == "ConstructorDeclaration":
                if "public" not in mods:
                    continue
                params = ", ".join(_render_type(p.type) for p in (m.parameters or []))
                out.append(f"{m.name}({params})")
            elif cls == "FieldDeclaration":
                if "public" not in mods:
                    continue
                for d in m.declarators:
                    out.append(f"{_render_type(m.type)} {d.name}")
        except Exception:
            continue
    return out


def _id_field(t) -> str:
    """Name of the @Id-annotated field of a JPA entity, or None. Used to drive the
    deterministic test-setter repair (generated ids have no setter)."""
    for m in (getattr(t, "body", None) or []):
        if type(m).__name__ != "FieldDeclaration":
            continue
        anns = {getattr(a, "name", "") for a in (getattr(m, "annotations", None) or [])}
        if "Id" in anns or "EmbeddedId" in anns:
            for d in (m.declarators or []):
                return d.name
    return None


def _parse_java_file(src: str):
    """Return (package, [(name, kind, is_main, members, id_field)], repo_or_None).
    Prefers javalang (full member signatures + @Id field); falls back to regex
    declarations (members empty) when the source doesn't parse (common mid-repair)."""
    pkg = ""
    types = []
    repo = None
    try:
        import javalang
        cu = javalang.parse.parse(src)
        pkg = cu.package.name if cu.package else ""
        for t in cu.types:
            kind = type(t).__name__.replace("Declaration", "").lower()
            anns = {a.name for a in (t.annotations or [])}
            types.append((t.name, kind, "SpringBootApplication" in anns, _members(t), _id_field(t)))
    except Exception:
        m = _PKG_RE.search(src)
        pkg = m.group(1) if m else ""
        for kind, name in _TYPE_RE.findall(src):
            types.append((name, kind, "@SpringBootApplication" in src, [], None))
    # Repo entity/id: regex is more robust than walking javalang's generic AST.
    rm = _REPO_RE.search(src)
    if rm:
        repo = (rm.group(1), rm.group(2).split(".")[-1])
    return pkg, types, repo


def build_symbol_table(output_dir: str, generated_files: list) -> dict:
    """Project-wide index: types by FQN, types by simple name (for import
    resolution + interface injection), and repositories → {entity, id}. Each type
    info carries `members` (public signatures) or, when javalang couldn't parse,
    a regex `summary` fallback.

    Generated files get a full entry (members, id_field, summary). The rest of
    src/main/java — legacy/brownfield classes the agent never wrote — get a
    lightweight package/fqn-only entry, just enough for import resolution
    (e.g. resolving `Pet` to `org.springframework.samples.petclinic.owner.Pet`)."""
    table = {"by_simple": {}, "repos": {}}
    seen_rel = set()

    def _index(rel, full):
        if rel in seen_rel or not rel.endswith(".java"):
            return
        seen_rel.add(rel)
        path = os.path.join(output_dir, rel)
        if not os.path.exists(path):
            return
        try:
            src = open(path).read()
        except Exception:
            return
        pkg, types, repo = _parse_java_file(src)
        for name, kind, is_main, members, id_field in types:
            if full:
                has_set_id = any(re.match(r"\S+\s+setId\(", m) for m in members)
                info = {"package": pkg, "file": rel, "kind": kind, "is_main": is_main,
                        "fqn": f"{pkg}.{name}" if pkg else name, "members": members,
                        "summary": "", "id_field": id_field, "has_set_id": has_set_id}
                if not members:
                    # javalang couldn't parse → keep a regex signature summary so the
                    # interface block still shows something useful.
                    try:
                        from agile_team_cline.sprint import _get_structural_summary
                        info["summary"] = _get_structural_summary(output_dir, rel)
                    except Exception:
                        pass
            else:
                info = {"package": pkg, "file": rel, "kind": kind, "is_main": is_main,
                        "fqn": f"{pkg}.{name}" if pkg else name, "members": [],
                        "summary": "", "id_field": id_field, "has_set_id": False}
            table["by_simple"].setdefault(name, []).append(info)
            if full and repo and name == types[0][0]:
                table["repos"][name] = {"entity": repo[0], "id": repo[1],
                                        "file": rel, "package": pkg}

    for rel in generated_files:
        _index(rel, full=True)

    src_main = os.path.join(output_dir, "src", "main", "java")
    if os.path.isdir(src_main):
        for dirpath, _dirs, filenames in os.walk(src_main):
            for fn in filenames:
                if fn.endswith(".java"):
                    _index(os.path.relpath(os.path.join(dirpath, fn), output_dir), full=False)

    return table


def render_interfaces(table: dict, exclude_simple: str = None) -> str:
    """Authoritative interface block: every project type's package + public member
    signatures (or regex summary fallback), except the one being generated."""
    blocks = []
    for name in sorted(table["by_simple"]):
        if exclude_simple and name == exclude_simple:
            continue
        for info in table["by_simple"][name]:
            header = f"--- {info['fqn']} ({info['kind']}) ---"
            if info["members"]:
                body = "\n".join(f"  {m}" for m in info["members"])
            elif info["summary"]:
                body = "\n".join(f"  {ln.strip()}" for ln in info["summary"].splitlines() if ln.strip())
            else:
                body = "  (no public members)"
            blocks.append(f"{header}\n{body}")
    return "\n".join(blocks)


def render_test_directives(table: dict) -> str:
    """Positive, data-specific rules for the test author, derived from the symbol
    table. Converts 'X is absent from the interface list' (which LLMs infer
    poorly) into explicit 'do this instead' guidance for the drift patterns the
    deterministic repair can't safely auto-fix (object accessors, invented CRUD)."""
    lines = []
    for name in sorted(table["by_simple"]):
        info = table["by_simple"][name][0]
        if info.get("kind") == "class" and info.get("id_field") and not info.get("has_set_id"):
            lines.append(
                f'- `{name}` has an auto-generated id and NO `setId(...)`: never call '
                f'`{name}.setId(...)`. Construct it via its real constructor; if a test needs a '
                f'fixed id use `ReflectionTestUtils.setField(obj, "{info["id_field"]}", value)`.')
    lines.append(
        "- Entities expose scalar FK accessors (e.g. `getOwnerId()`/`setOwnerId(UUID)`), NOT "
        "object accessors (`getOwner()`/`setOwner(...)`). Use the scalar ids plus the "
        "repositories shown in the interface block to reach related data.")
    lines.append(
        "- Call ONLY the constructors, service methods, repository finders, and DTO fields that "
        "appear in the interface block above. Do NOT invent CRUD methods (e.g. `createX`/"
        "`updateX`), derived finders, or accessors that are not listed.")
    return "\n".join(lines)


def resolve_imports(src: str, table: dict) -> list:
    """Project files referenced by this source's intra-project imports (used to
    inject full dependency source during refinement)."""
    out = []
    seen = set()
    for m in re.finditer(r"import\s+([\w.]+)\.(\w+)\s*;", src):
        simple = m.group(2)
        for info in table["by_simple"].get(simple, []):
            if info["file"] not in seen:
                seen.add(info["file"])
                out.append(info["file"])
    return out


# --- structured maven error parsing --------------------------------------------------
_FILE_ERR_RE = re.compile(
    r"\[ERROR\]\s+(?:.*[/\\])?([^/\\]+\.java)(?::\[(\d+),(\d+)\]|:(\d+):)?\s+(.*)")
_BAD_PKG_RE = re.compile(r"package\s+([\w.]+)\s+does not exist")
_SYM_METHOD_RE = re.compile(r"symbol:\s*method\s+(\w+)\(([^)]*)\)")
_SYM_CLASS_RE = re.compile(r"symbol:\s*class\s+(\w+)")
_LOC_TYPE_RE = re.compile(r"location:.*?\bof type ([\w.]+)")
_IN_TYPE_RE = re.compile(r"\bin (?:class|interface) ([\w.]+)")
_LOC_DECL_RE = re.compile(r"location:\s*(?:class|interface)\s+([\w.]+)")
_GOAL_FAIL = "Failed to execute goal"


def parse_maven_errors(stdout: str) -> list:
    """THE single low-level parser for `mvn` output — the one place that knows how
    javac/maven-compiler `[ERROR]` lines, their indented continuation sub-lines,
    and the cross-file owner types an error references are laid out. Both the
    deterministic repair pass and the QA issue transformer (sprint._parse_maven_issues)
    build on these records, so the regexes and drift heuristic never diverge.

    Each record: {file, line, col, headline, message, kind, bad_package, method,
    param_types, owner, owners}, where `headline` is the first-line message,
    `message` the full block (headline + continuations), `owner` the single
    referenced type the repair edits act on, and `owners` the SET of all referenced
    types (used for drift interface-injection). kind ∈ {missing_package_import,
    missing_repo_method, missing_symbol, goal_failure, other}."""
    records = []
    seen = set()
    lines = stdout.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if "[ERROR]" not in line:
            i += 1
            continue
        m = _FILE_ERR_RE.search(line)
        if not m:
            # Non-file [ERROR] line — capture maven goal failures (e.g. the
            # compiler plugin's final summary) as a pom-level record.
            cleaned = line.replace("[ERROR]", "").strip()
            if cleaned and _GOAL_FAIL in cleaned:
                key = ("pom.xml", "goal_failure", cleaned)
                if key not in seen:
                    seen.add(key)
                    records.append({"file": "pom.xml", "line": "", "col": "",
                                    "headline": cleaned, "message": cleaned,
                                    "kind": "goal_failure", "bad_package": "",
                                    "method": "", "param_types": [], "owner": "",
                                    "owners": set()})
            i += 1
            continue
        fname, msg = m.group(1), m.group(5).strip()
        col_line = m.group(2) or m.group(4) or ""
        col = m.group(3) or ""
        # Collect the indented sub-lines (symbol:/location:/required:) of this error.
        block = msg
        owner = ""
        owners = set()
        method = ""
        params = []
        missing_class = ""
        # The owner type(s) this error references and that live in ANOTHER file —
        # the consumer is calling a member the referenced type does not declare
        # (the canonical cross-file signature drift). `owner` is the single type the
        # repair edits target; `owners` is the full set fed back to the LLM as an
        # interface block so the caller is steered to conform rather than inventing.
        im = _IN_TYPE_RE.search(msg)
        if im:
            owner = im.group(1).split(".")[-1]
            owners.add(owner)
        # Scan continuation lines (symbol:/location:/required:/...). In the primary
        # javac diagnostic these are indented and carry NO [ERROR] prefix; in the
        # maven-compiler-plugin summary they repeat WITH [ERROR]. Both end at a blank
        # line or the next file-error line.
        j = i + 1
        while j < n and not _FILE_ERR_RE.search(lines[j]):
            sub = lines[j]
            if not sub.strip():
                break
            block += " " + sub.strip()
            sm = _SYM_METHOD_RE.search(sub)
            if sm:
                method = sm.group(1)
                params = [p.strip() for p in sm.group(2).split(",") if p.strip()]
            lt = _LOC_TYPE_RE.search(sub)
            if lt:
                owner = lt.group(1).split(".")[-1]
                owners.add(owner)
            ld = _LOC_DECL_RE.search(sub)
            if ld:
                owners.add(ld.group(1).split(".")[-1])
            sc = _SYM_CLASS_RE.search(sub)
            if sc:
                missing_class = sc.group(1)
            j += 1

        rec = {"file": fname, "line": col_line, "col": col, "headline": msg,
               "message": block, "kind": "other", "bad_package": "",
               "method": method, "param_types": params, "owner": owner,
               "owners": owners, "missing_class": missing_class}
        bp = _BAD_PKG_RE.search(block)
        if bp:
            rec["kind"] = "missing_package_import"
            rec["bad_package"] = bp.group(1)
        elif method and owner.endswith("Repository"):
            rec["kind"] = "missing_repo_method"
        elif "cannot find symbol" in block:
            rec["kind"] = "missing_symbol"

        key = (fname, col_line, col, rec["kind"], rec["bad_package"], method, owner, missing_class)
        if key not in seen:
            seen.add(key)
            records.append(rec)
        i = j
    return records


# --- deterministic edits -------------------------------------------------------------
def _ensure_imports(src: str, fqns: set) -> str:
    """Insert `import <fqn>;` lines for any fqn not already imported."""
    to_add = [f for f in fqns if f and f"import {f};" not in src]
    if not to_add:
        return src
    lines = src.splitlines(keepends=True)
    insert_at = 0
    for idx, line in enumerate(lines):
        s = line.strip()
        if s.startswith("package "):
            insert_at = idx + 1
        elif s.startswith("import "):
            insert_at = idx + 1
    block = "".join(f"import {f};\n" for f in sorted(to_add))
    if insert_at and not lines[insert_at - 1].endswith("\n"):
        block = "\n" + block
    return "".join(lines[:insert_at]) + block + "".join(lines[insert_at:])


_M2_RAW_INDEX = None  # simple -> {package: {jar_path, ...}}, scanned once per process
_CLASSPATH_INDEX_CACHE = {}  # output_dir -> simple -> {package, ...} (version-disambiguated)


def _m2_raw_index() -> dict:
    """Scan ~/.m2/repository once for simple-class-name -> {package: jar paths}.
    `mvn dependency:build-classpath` needs plugin jars that may not be cached
    offline, so we read the local repo directly — it's the same jars either way."""
    global _M2_RAW_INDEX
    if _M2_RAW_INDEX is not None:
        return _M2_RAW_INDEX
    import glob
    index = {}
    m2 = os.path.expanduser("~/.m2/repository")
    for jar in glob.glob(os.path.join(m2, "**", "*.jar"), recursive=True):
        try:
            with zipfile.ZipFile(jar) as zf:
                for name in zf.namelist():
                    if not name.endswith(".class") or "$" in name or name == "module-info.class":
                        continue
                    parts = name[:-len(".class")].split("/")
                    simple = parts[-1]
                    pkg = ".".join(parts[:-1])
                    index.setdefault(simple, {}).setdefault(pkg, set()).add(jar)
        except Exception:
            continue
    _M2_RAW_INDEX = index
    return index


def _project_version_hint(output_dir: str) -> str:
    """The project's spring-boot-starter-parent version (or own <parent> version),
    used to disambiguate a simple class name that exists in multiple cached jar
    versions in ~/.m2 (e.g. spring-boot 3.2.x leftovers from other projects vs
    this project's actual 4.0.3 — different packages for the same class name)."""
    try:
        pom = open(os.path.join(output_dir, "pom.xml")).read()
    except Exception:
        return ""
    m = re.search(r"<parent>.*?<version>([\w.\-]+)</version>", pom, re.S)
    return m.group(1) if m else ""


def _classpath_simple_index(output_dir: str) -> dict:
    """Map simple class name -> set of packages, derived from ~/.m2/repository.
    Last-resort import-resolution source for library types (e.g. `EntityScan`)
    that build_symbol_table can't see, since it only indexes project source
    files. When a name is ambiguous across jar versions, narrow to jars matching
    the project's own dependency version (see _project_version_hint)."""
    if output_dir in _CLASSPATH_INDEX_CACHE:
        return _CLASSPATH_INDEX_CACHE[output_dir]
    raw = _m2_raw_index()
    version = _project_version_hint(output_dir)
    index = {}
    for simple, pkgs in raw.items():
        if len(pkgs) == 1 or not version:
            index[simple] = set(pkgs)
            continue
        narrowed = {pkg for pkg, jars in pkgs.items() if any(f"/{version}/" in j for j in jars)}
        index[simple] = narrowed if len(narrowed) == 1 else set(pkgs)
    _CLASSPATH_INDEX_CACHE[output_dir] = index
    return index


def _fix_import(output_dir: str, rec: dict, table: dict) -> str:
    """Rewrite imports referencing a non-existent package to the real package of
    that type (only when the simple name resolves unambiguously). Tries project
    sources first, then the dependency-jar classpath index."""
    path = os.path.join(output_dir, _find_rel(output_dir, rec["file"]))
    if not os.path.exists(path):
        return ""
    src = open(path).read()
    bad = rec["bad_package"]
    pat = re.compile(rf"import\s+{re.escape(bad)}\.(\w+)\s*;")
    changed = False
    jar_index = None
    for simple in pat.findall(src):
        cands = [c for c in table["by_simple"].get(simple, []) if c["package"] and c["package"] != bad]
        pkgs = {c["package"] for c in cands}
        if len(pkgs) != 1:
            if jar_index is None:
                jar_index = _classpath_simple_index(output_dir)
            jar_pkgs = {p for p in jar_index.get(simple, set()) if p != bad}
            if len(jar_pkgs) == 1:
                pkgs = jar_pkgs
        if len(pkgs) == 1:
            real = next(iter(pkgs))
            src = src.replace(f"{bad}.{simple}", f"{real}.{simple}")
            changed = True
    if changed:
        open(path, "w").write(src)
        return f"import {bad}.* → real package in {rec['file']}"
    return ""


def _add_missing_class_import(output_dir: str, rec: dict, table: dict) -> str:
    """For `cannot find symbol: class Y` where Y is simply unimported (no bad
    `import` statement to rewrite — the import line is just absent). Resolve Y's
    package from the workspace symbol table (legacy/brownfield classes included,
    see build_symbol_table) or, failing that, the classpath jar index, and add
    the import if it resolves unambiguously."""
    simple = rec.get("missing_class")
    if not simple:
        return ""
    rel = _find_rel(output_dir, rec["file"])
    path = os.path.join(output_dir, rel)
    if not os.path.exists(path):
        return ""
    src = open(path).read()
    if re.search(rf"import\s+[\w.]+\.{simple}\s*;", src) or re.search(rf"\bclass\s+{simple}\b", src):
        return ""
    cands = {c["package"] for c in table["by_simple"].get(simple, []) if c.get("package")}
    if len(cands) != 1:
        jar_pkgs = _classpath_simple_index(output_dir).get(simple, set())
        if len(jar_pkgs) == 1:
            cands = jar_pkgs
    if len(cands) != 1:
        return ""
    pkg = next(iter(cands))
    new_src = _ensure_imports(src, {f"{pkg}.{simple}"})
    if new_src == src:
        return ""
    open(path, "w").write(new_src)
    return f"import {pkg}.{simple} added to {rec['file']}"


def _add_repo_finder(output_dir: str, rec: dict, table: dict) -> str:
    """Add a Spring Data derived-query signature the caller needs to the repository
    interface (Spring auto-implements it). Producer-gains-method is the correct
    direction for repos, complementing the consumer-conforms drift heuristic."""
    repo = table["repos"].get(rec["owner"])
    method = rec["method"]
    if not repo or not method:
        return ""
    path = os.path.join(output_dir, repo["file"])
    if not os.path.exists(path):
        return ""
    src = open(path).read()
    if re.search(rf"\b\w[\w<>,. ]*\s+{re.escape(method)}\s*\(", src):
        return ""  # already declared
    entity = repo["entity"]
    if method.startswith(("countBy",)):
        ret = "long"
    elif method.startswith(("existsBy",)):
        ret = "boolean"
    elif method.startswith(("deleteBy", "removeBy")):
        ret = "void"
    elif method in ("findById", "getById") or method.startswith(("findOneBy", "getOneBy")):
        ret = f"Optional<{entity}>"
    else:  # findBy / getBy / readBy / queryBy / streamBy ...
        ret = f"List<{entity}>"
    params = ", ".join(f"{pt.split('.')[-1]} arg{k}" for k, pt in enumerate(rec["param_types"]))
    sig = f"\n    {ret} {method}({params});\n"
    idx = src.rfind("}")
    if idx < 0:
        return ""
    src = src[:idx] + sig + src[idx:]
    needed = set()
    if "List<" in ret:
        needed.add("java.util.List")
    if "Optional<" in ret:
        needed.add("java.util.Optional")
    for pt in rec["param_types"]:
        if "." in pt and not pt.startswith("java.lang."):
            needed.add(pt)
    ent = table["by_simple"].get(entity, [])
    if ent and ent[0]["package"] and ent[0]["package"] != repo["package"]:
        needed.add(f"{ent[0]['package']}.{entity}")
    src = _ensure_imports(src, needed)
    open(path, "w").write(src)
    return f"added {repo['file'].split('/')[-1]}.{method}() -> {ret}"


_REFLECTION_UTILS = "org.springframework.test.util.ReflectionTestUtils"


def _rewrite_setid_calls(src: str, var: str, id_field: str) -> tuple:
    """Rewrite `<var>.setId(<expr>)` → `ReflectionTestUtils.setField(<var>,
    "<id_field>", <expr>)`, matching balanced parens in <expr>. Returns
    (new_src, count)."""
    out = []
    i = 0
    n = len(src)
    needle = f"{var}.setId("
    count = 0
    while i < n:
        k = src.find(needle, i)
        if k < 0:
            out.append(src[i:])
            break
        # Match the balanced argument list starting at the '(' of setId(.
        p = k + len(needle)  # first char after '('
        depth = 1
        j = p
        while j < n and depth:
            c = src[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            j += 1
        if depth != 0:  # unbalanced — bail, leave the rest untouched
            out.append(src[i:])
            break
        expr = src[p:j - 1].strip()
        out.append(src[i:k])
        out.append(f'ReflectionTestUtils.setField({var}, "{id_field}", {expr})')
        count += 1
        i = j
    return "".join(out), count


def _fix_test_id_setter(output_dir: str, rec: dict, table: dict) -> str:
    """Test code calling `setId(...)` on an entity whose id is generated (no
    setter) → rewrite to `ReflectionTestUtils.setField(obj, "<id>", value)`. The
    only mechanically safe cure for the dominant test↔entity drift; scoped to test
    sources, idempotent (a rewritten file has no `.setId(` left to match)."""
    if rec.get("method") != "setId":
        return ""
    entity = rec.get("owner")
    infos = table["by_simple"].get(entity, [])
    info = infos[0] if infos else None
    if not info or not info.get("id_field") or info.get("has_set_id"):
        return ""
    rel = _find_rel(output_dir, rec["file"])
    low = rel.lower()
    if "test" not in low and not rel.endswith(("Test.java", "Tests.java", "IT.java")):
        return ""
    path = os.path.join(output_dir, rel)
    if not os.path.exists(path):
        return ""
    src = open(path).read()
    id_field = info["id_field"]
    # Variables of this entity type in the file: `Entity var` decls and
    # `var = new Entity(` instantiations.
    var_names = set(re.findall(rf"\b{re.escape(entity)}\s+(\w+)\s*[=;)]", src))
    var_names |= set(re.findall(rf"\b(\w+)\s*=\s*new\s+{re.escape(entity)}\s*\(", src))
    total = 0
    for v in var_names:
        src, c = _rewrite_setid_calls(src, v, id_field)
        total += c
    if not total:
        return ""
    src = _ensure_imports(src, {_REFLECTION_UTILS})
    open(path, "w").write(src)
    return f"rewrote {total} {entity}.setId() → ReflectionTestUtils.setField in {rec['file']}"


# Map package prefixes → (groupId, artifactId) for Spring Boot starters and
# common libs the LLM imports but the architect omits from pom.xml.
_PKG_TO_MAVEN_DEP = {
    "org.springframework.security":  ("org.springframework.boot",  "spring-boot-starter-security"),
    "org.springframework.batch":     ("org.springframework.boot",  "spring-boot-starter-batch"),
    "org.springframework.amqp":      ("org.springframework.boot",  "spring-boot-starter-amqp"),
    "org.springframework.mail":      ("org.springframework.boot",  "spring-boot-starter-mail"),
    "org.thymeleaf":                 ("org.springframework.boot",  "spring-boot-starter-thymeleaf"),
    "io.micrometer":                 ("org.springframework.boot",  "spring-boot-starter-actuator"),
    "org.springdoc":                 ("org.springdoc",             "springdoc-openapi-starter-webmvc-ui"),
    "io.swagger.v3":                 ("org.springdoc",             "springdoc-openapi-starter-webmvc-ui"),
}


def _fix_missing_maven_dep(output_dir: str, records: list) -> str:
    """Add missing Maven dependencies to pom.xml inferred from package-not-found errors.

    Only adds each artifact once per call; subsequent repair iterations are
    idempotent because we check `artifact_id not in pom` before inserting.
    """
    pom_path = os.path.join(output_dir, "pom.xml")
    if not os.path.exists(pom_path):
        return ""
    pom = open(pom_path).read()
    added = []
    for rec in records:
        if rec["kind"] != "missing_package_import":
            continue
        pkg = rec["bad_package"]
        for prefix, (group_id, artifact_id) in _PKG_TO_MAVEN_DEP.items():
            if pkg.startswith(prefix) and artifact_id not in pom:
                dep_xml = (
                    f"\n        <dependency>"
                    f"\n            <groupId>{group_id}</groupId>"
                    f"\n            <artifactId>{artifact_id}</artifactId>"
                    f"\n        </dependency>"
                )
                pom = pom.replace("</dependencies>", dep_xml + "\n    </dependencies>", 1)
                added.append(artifact_id)
                break
    if added:
        open(pom_path, "w").write(pom)
        return f"pom.xml: added {', '.join(added)}"
    return ""


# Hallucinated Spring Boot test import FQNs the model reliably re-emits, mapped to
# the real package. `@MockBean` lives in ...test.mock.mockito, never ...test.mock.bean.
# A wrong package is a `does not exist` compile error the LLM does not self-correct,
# so rewrite it deterministically before the build.
_BAD_TEST_IMPORTS = {
    "org.springframework.boot.test.mock.bean.MockBean":
        "org.springframework.boot.test.mock.mockito.MockBean",
    # @Transactional lives in ...transaction.annotation, NOT ...test.annotation
    # (which only holds @Rollback/@Commit/@DirtiesContext). Full-FQN match, so the
    # legitimate ...test.annotation.* imports are untouched.
    "org.springframework.test.annotation.Transactional":
        "org.springframework.transaction.annotation.Transactional",
}


def _fix_bad_test_imports(output_dir: str, generated_files: list) -> list:
    """Rewrite known-bad Spring Boot test import packages to their real FQN.
    Idempotent: once rewritten the bad string is gone, so re-runs are no-ops."""
    changes = []
    for rel in generated_files:
        if not rel.endswith(".java"):
            continue
        path = os.path.join(output_dir, rel)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        new = src
        for bad, good in _BAD_TEST_IMPORTS.items():
            new = new.replace(bad, good)
        if new != src:
            open(path, "w").write(new)
            changes.append(f"rewrote test import in {rel}")
    return changes


def _ensure_starter_test(output_dir: str) -> str:
    """Guarantee spring-boot-starter-test is on the pom. Every generated project
    writes JUnit/Mockito/@WebMvcTest test sources, but the architect sometimes
    omits the starter — leaving the ENTIRE test tree uncompilable (`package
    org.junit.jupiter.api does not exist`, etc.). Unconditional + idempotent."""
    pom_path = os.path.join(output_dir, "pom.xml")
    if not os.path.exists(pom_path):
        return ""
    pom = open(pom_path).read()
    if "spring-boot-starter-test" in pom or "</dependencies>" not in pom:
        return ""
    dep = (
        "\n        <dependency>"
        "\n            <groupId>org.springframework.boot</groupId>"
        "\n            <artifactId>spring-boot-starter-test</artifactId>"
        "\n            <scope>test</scope>"
        "\n        </dependency>"
    )
    pom = pom.replace("</dependencies>", dep + "\n    </dependencies>", 1)
    open(pom_path, "w").write(pom)
    return "pom.xml: added spring-boot-starter-test"


_IMPORT_ALIAS_RE = re.compile(r"^[ \t]*import\s+([\w.]+)\s+as\s+(\w+)\s*;[ \t]*\n?", re.MULTILINE)


def _fix_import_aliases(output_dir: str, generated_files: list) -> list:
    """Strip non-Java import aliases (`import a.b.C as D;`) — a foreign-language
    idiom (Python/Kotlin/Scala) the model occasionally leaks, which is a hard
    parse error (`';' expected`) that blocks the whole build and that the LLM
    re-emits across refinement rounds. Each alias import is removed and its uses
    are rewritten to the fully-qualified name — always collision-safe (the alias
    usually exists precisely to dodge a simple-name clash with another import).
    Idempotent; runs before the symbol table is built so the file parses."""
    changes = []
    for rel in generated_files:
        if not rel.endswith(".java"):
            continue
        path = os.path.join(output_dir, rel)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        aliases = _IMPORT_ALIAS_RE.findall(src)
        if not aliases:
            continue
        new_src = _IMPORT_ALIAS_RE.sub("", src)
        for fqn, alias in aliases:
            new_src = re.sub(rf"\b{re.escape(alias)}\b", fqn, new_src)
        if new_src != src:
            open(path, "w").write(new_src)
            changes.append(
                f"stripped import-alias(es) in {rel} "
                f"({', '.join(a for _, a in aliases)} → FQN)")
    return changes


_PKG_DECL_RE = re.compile(r"(?m)^[ \t]*package\s+[\w.]+\s*;")


def _split_concatenated_files(blackboard, generated_files: list) -> list:
    """A single `.java` file the model emitted as MULTIPLE concatenated compilation
    units (signalled by a SECOND top-level `package ...;` mid-file) is a hard parse
    error (`class, interface, enum, or record expected`). Keep the first unit in the
    original file; for each trailing unit, drop it if a sibling file for that type
    already exists (a duplicate concatenation — the common case), else write it to
    its own correctly-named file via the blackboard's package-aware path resolver.
    Idempotent: a split file is left with exactly one package declaration."""
    output_dir = str(blackboard.output_dir)
    changes = []
    for rel in list(generated_files):
        if not rel.endswith(".java"):
            continue
        path = os.path.join(output_dir, rel)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        bounds = [m.start() for m in _PKG_DECL_RE.finditer(src)]
        if len(bounds) < 2:
            continue
        head = src[:bounds[0]]  # preserve any license/comment preamble
        units = [src[bounds[i]:(bounds[i + 1] if i + 1 < len(bounds) else len(src))]
                 for i in range(len(bounds))]
        open(path, "w").write(head + units[0])  # keep the first unit in this file
        notes = []
        for unit in units[1:]:
            tm = _TYPE_RE.search(unit)
            if not tm:
                continue
            type_name = tm.group(2)
            target_rel = blackboard.resolve_path(f"{type_name}.java", unit)
            if os.path.exists(os.path.join(output_dir, target_rel)):
                notes.append(f"dropped duplicate {type_name}")
                continue
            blackboard.write_file(f"{type_name}.java", unit.strip() + "\n")
            blackboard.record_file(target_rel)
            notes.append(f"split out {type_name} → {target_rel}")
        changes.append(f"de-concatenated {rel}: {'; '.join(notes)}")
    return changes


def _toplevel_types(src: str) -> list:
    """Brace-depth-0 type declarations in a Java source: [(kw_start, end, name)]
    where kw_start is the index of the class/interface/enum/record keyword and
    end is just past its matching closing brace. Skips //, /* */, "..." and
    '...' so braces/keywords inside comments and literals don't fool the scan.
    Types with no body (e.g. `record R(int x);`) are not recorded — the splitter
    then no-ops on them, which is safe."""
    types = []
    i, n = 0, len(src)
    depth = 0
    pending = None  # (kw_start, name) awaiting its body's closing brace
    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            i = n if j == -1 else j + 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == '"' or c == "'":
            q = c
            i += 1
            while i < n and src[i] != q:
                if src[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        if c == "{":
            depth += 1
            i += 1
            continue
        if c == "}":
            depth -= 1
            i += 1
            if depth == 0 and pending is not None:
                types.append((pending[0], i, pending[1]))
                pending = None
            continue
        if depth == 0 and pending is None and (c.isalpha() or c == "_"):
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            if src[i:j] in ("class", "interface", "enum", "record"):
                k = j
                while k < n and src[k] in " \t\r\n":
                    k += 1
                m = k
                while m < n and (src[m].isalnum() or src[m] == "_"):
                    m += 1
                name = src[k:m]
                if name:
                    pending = (i, name)
            i = j
            continue
        i += 1
    return types


def _decl_start_index(src: str, kw_start: int) -> int:
    """Start of a type declaration including its modifiers and any immediately
    preceding annotation lines, so a split-out/dropped unit carries its full
    header (not a bare `class X`)."""
    start = src.rfind("\n", 0, kw_start) + 1  # start of the keyword's own line
    while start > 0:
        prev_nl = src.rfind("\n", 0, start - 1)
        line = src[prev_nl + 1:start - 1].strip()
        if line.startswith("@"):
            start = prev_nl + 1
        else:
            break
    return start


def _sibling_exists(output_dir: str, this_rel: str, name: str) -> bool:
    """True if some OTHER generated file is already the dedicated <name>.java."""
    target = f"{name}.java"
    this_norm = this_rel.replace("\\", "/")
    for root, _, files in os.walk(output_dir):
        if target in files:
            rel = os.path.relpath(os.path.join(root, target), output_dir).replace("\\", "/")
            if rel != this_norm:
                return True
    return False


def _split_multiclass_files(blackboard, generated_files: list) -> list:
    """A single `.java` file containing MULTIPLE top-level classes under ONE
    package declaration (so `_split_concatenated_files`, which keys on a second
    `package ...;`, never fires). javac rejects this as `duplicate class` when a
    trailing class also has its own dedicated file (the observed failure), or as
    a public-type/filename mismatch otherwise. Keep the type matching the file
    name (else the first); for each extra top-level type drop it if a sibling
    file already defines it, else split it into its own correctly-named file.
    Idempotent: leaves exactly one top-level type per file."""
    output_dir = str(blackboard.output_dir)
    changes = []
    for rel in list(generated_files):
        if not rel.endswith(".java"):
            continue
        path = os.path.join(output_dir, rel)
        if not os.path.exists(path):
            continue
        src = open(path).read()
        kinds = _toplevel_types(src)
        if len(kinds) < 2:
            continue
        spans = sorted((_decl_start_index(src, ks), end, name) for ks, end, name in kinds)
        preamble = src[:spans[0][0]]  # package + imports + leading comments
        stem = os.path.splitext(os.path.basename(rel))[0]
        keep_idx = next((i for i, (_s, _e, nm) in enumerate(spans) if nm == stem), 0)
        notes = []
        kept = []
        for idx, (s, e, name) in enumerate(spans):
            unit = src[s:e]
            if idx == keep_idx:
                kept.append(unit)
                continue
            if _sibling_exists(output_dir, rel, name):
                notes.append(f"dropped duplicate {name}")
                continue
            new_src = preamble + unit.strip() + "\n"
            blackboard.write_file(f"{name}.java", new_src)
            blackboard.record_file(blackboard.resolve_path(f"{name}.java", new_src))
            notes.append(f"split out {name}")
        if notes:
            open(path, "w").write(preamble + "\n".join(u.strip() for u in kept) + "\n")
            changes.append(f"multiclass {rel}: {'; '.join(notes)}")
    return changes


def _find_rel(output_dir: str, basename: str) -> str:
    """Resolve a bare filename (as reported by mvn) to its path under output_dir."""
    if os.path.exists(os.path.join(output_dir, basename)):
        return basename
    for root, _, files in os.walk(output_dir):
        if basename in files:
            return os.path.relpath(os.path.join(root, basename), output_dir)
    return basename


# The original Spring PetClinic pom binds spring-javaformat:validate + checkstyle +
# nohttp to an early phase, so `mvn test-compile` fails on WHITESPACE before javac
# even runs — masking real compile errors and starving refinement on formatting the
# LLM can't satisfy. Skip those style goals on every gate/repair compile; the build
# is auto-formatted instead (see _apply_format), so the final artifact still passes
# the project's own `mvn verify`.
STYLE_SKIP_FLAGS = [
    "-Dspring-javaformat.validate.skip=true",
    "-Dcheckstyle.skip=true",
    "-Dnohttp.checkstyle.validate.skip=true",
]


def _mvn_compile_argv(goal_phases: list) -> list:
    """Build the maven argv for a compile gate. When AGILE_BUILD_MODULE is set, scope
    the build to that reactor module (+ its upstream deps via -am) so a multi-module
    reactor compiles only the feature's subtree instead of all modules. Default (env
    unset) = whole-project, byte-identical to prior behavior."""
    argv = ["mvn", *goal_phases, *STYLE_SKIP_FLAGS]
    module = os.getenv("AGILE_BUILD_MODULE", "").strip()
    if module:
        argv += ["-pl", module, "-am"]
    return argv


def _apply_format(output_dir: str) -> None:
    """Auto-format generated sources with spring-javaformat:apply so the on-disk tree
    satisfies the project's own format gate. Best-effort: failures are non-fatal."""
    if os.getenv("AGILE_BUILD_MODULE", "").strip():
        return  # non-Spring reactor: no spring-javaformat plugin to run
    if not os.path.exists(os.path.join(output_dir, "pom.xml")):
        return
    try:
        subprocess.run(["mvn", "-q", "spring-javaformat:apply"], cwd=output_dir,
                       capture_output=True, text=True,
                       timeout=int(os.getenv("AGILE_MVN_TIMEOUT", "300")))
    except Exception as e:
        print(f"⚠️ spring-javaformat:apply skipped: {e}")


def _restore_missing_poms(output_dir: str) -> list:
    """Restore any module pom.xml that cline deleted mid-session, from the snapshot taken
    at pipeline init (pom_snapshot.json). Runs before every compile/test invocation so a
    deleted child module pom can't abort the Maven reactor at POM-parse before javac even
    runs. Lives here (not sprint.py) because repair's _compile_raw/_compile_main are the
    compiles that actually gate the pipeline."""
    snap_path = os.path.join(output_dir, "pom_snapshot.json")
    if not os.path.exists(snap_path):
        return []
    try:
        with open(snap_path) as fh:
            snap = json.load(fh)
    except Exception:
        return []
    restored = []
    for rel, content in (snap or {}).items():
        p = os.path.join(output_dir, rel)
        if not os.path.exists(p):
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as fh:
                    fh.write(content)
                restored.append(rel)
            except Exception:
                pass
    if restored:
        print(f"🔧 pom.xml guard: restored {len(restored)} deleted module pom(s) — {restored}")
    return restored


def _restore_missing_baseline_files(output_dir: str) -> list:
    """Baseline-integrity guard, generalizing the pom guard to the whole staged baseline:
    any file recorded in the init-time manifest (baseline_src.json — src/** + poms) that
    has vanished from disk is restored from the pristine baseline tree named by
    AGILE_BASELINE_REF. Missing files only — files modified in place (hygiene, feature
    edits) are never touched, matching `rsync --ignore-existing` semantics. Defends the
    Jun-28-class burst (whole baseline source trees deleted), which the pom-only snapshot
    cannot. No-op when the env var or manifest is absent."""
    import shutil
    ref = os.getenv("AGILE_BASELINE_REF", "").strip()
    if not ref or not os.path.isdir(ref):
        return []
    manifest_path = os.path.join(output_dir, "baseline_src.json")
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path) as fh:
            manifest = json.load(fh)
    except Exception:
        return []
    restored = []
    for rel in (manifest or {}):
        dst = os.path.join(output_dir, rel)
        if os.path.exists(dst):
            continue
        src = os.path.join(ref, rel)
        if not os.path.isfile(src):
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            restored.append(rel)
        except Exception:
            pass
    if restored:
        print(f"🛡️ baseline guard: restored {len(restored)} deleted baseline file(s) "
              f"from {ref} — first few: {restored[:5]}")
    return restored


def _ensure_tree_integrity(output_dir: str) -> None:
    """Run both integrity guards (pom snapshot + baseline manifest) before any build/test
    invocation, so external file deletion can never abort the reactor or blind a gate."""
    _restore_missing_poms(output_dir)
    _restore_missing_baseline_files(output_dir)


def _compile_raw(output_dir: str):
    _ensure_tree_integrity(output_dir)
    if not os.path.exists(os.path.join(output_dir, "pom.xml")):
        return 0, ""
    try:
        res = subprocess.run(_mvn_compile_argv(["clean", "test-compile"]), cwd=output_dir,
                             capture_output=True, text=True,
                             timeout=int(os.getenv("AGILE_MVN_TIMEOUT", "300")))
        return res.returncode, res.stdout
    except Exception as e:
        print(f"⚠️ repair compile failed to run: {e}")
        return 1, ""


def _compile_main(output_dir: str):
    """Compile ONLY the production sources (`mvn compile`). Used to tell a genuinely
    broken application apart from a build that's red purely because a generated test
    file doesn't compile — main-compile is the hard gate, test-compile is secondary."""
    _ensure_tree_integrity(output_dir)
    if not os.path.exists(os.path.join(output_dir, "pom.xml")):
        return 0, ""
    try:
        res = subprocess.run(_mvn_compile_argv(["compile"]), cwd=output_dir,
                             capture_output=True, text=True,
                             timeout=int(os.getenv("AGILE_MVN_TIMEOUT", "300")))
        return res.returncode, res.stdout
    except Exception as e:
        print(f"⚠️ repair main-compile failed to run: {e}")
        return 1, ""


def repair_sources(blackboard, stack: dict, max_iters: int = 5) -> tuple:
    """Entry point: source hygiene + a bounded, monotone, compiler-error-driven
    repair loop. Returns (changes, compile_result) where compile_result is
    {"compiles": bool, "issues": [...]} reflecting the tree this pass leaves on
    disk, or None when there is nothing to compile (non-Java / no pom). The caller
    (_run_qa_sprint) reuses this instead of recompiling the identical tree."""
    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"
    if not is_java:
        return [], None

    from agile_team_cline.sprint import _run_source_hygiene, _parse_maven_issues, _is_spring
    changes = list(_run_source_hygiene(blackboard, stack))

    output_dir = str(blackboard.output_dir)
    gen0 = blackboard.read("blackboard").get("generated_files", [])
    changes.extend(_split_concatenated_files(blackboard, gen0))
    gen0 = blackboard.read("blackboard").get("generated_files", [])
    changes.extend(_split_multiclass_files(blackboard, gen0))
    gen0 = blackboard.read("blackboard").get("generated_files", [])
    changes.extend(_fix_import_aliases(output_dir, gen0))
    changes.extend(_fix_bad_test_imports(output_dir, gen0))
    # spring-boot-starter-test is meaningless (and corrupting) on a non-Spring pom — a
    # jakarta/JAX-RS module (TeaStore) already ships its own JUnit 4 test deps.
    ch = _ensure_starter_test(output_dir) if _is_spring(stack) else ""
    if ch:
        changes.append(ch)
    # Auto-format the (re)generated tree so it satisfies the project's spring-javaformat
    # gate. The compile loop below skips the style goals (STYLE_SKIP_FLAGS), so it sees
    # only real javac errors; this keeps the on-disk artifact properly formatted.
    _apply_format(output_dir)
    # Track the compile result that matches the tree currently on disk. `None`
    # means edits were applied after the last compile, so the state is stale and
    # must be re-verified before we trust it.
    last = None
    for _ in range(max_iters):
        code, stdout = _compile_raw(output_dir)
        last = (code, stdout)
        if code == 0:
            break
        records = parse_maven_errors(stdout)
        gen = blackboard.read("blackboard").get("generated_files", [])
        table = build_symbol_table(output_dir, gen)
        applied = []
        ch = _fix_missing_maven_dep(output_dir, records)
        if ch:
            applied.append(ch)
        done_imports = set()
        done_finders = set()
        done_setters = set()
        done_class_imports = set()
        for rec in records:
            if rec["kind"] == "missing_package_import":
                k = (rec["file"], rec["bad_package"])
                if k in done_imports:
                    continue
                done_imports.add(k)
                ch = _fix_import(output_dir, rec, table)
                if ch:
                    applied.append(ch)
            elif rec["kind"] == "missing_repo_method":
                k = (rec["owner"], rec["method"])
                if k in done_finders:
                    continue
                done_finders.add(k)
                ch = _add_repo_finder(output_dir, rec, table)
                if ch:
                    applied.append(ch)
            elif rec["kind"] == "missing_symbol" and rec.get("method") == "setId":
                k = (rec["file"], rec["owner"])
                if k in done_setters:
                    continue
                done_setters.add(k)
                ch = _fix_test_id_setter(output_dir, rec, table)
                if ch:
                    applied.append(ch)
            elif rec["kind"] == "missing_symbol" and rec.get("missing_class"):
                k = (rec["file"], rec["missing_class"])
                if k in done_class_imports:
                    continue
                done_class_imports.add(k)
                ch = _add_missing_class_import(output_dir, rec, table)
                if ch:
                    applied.append(ch)
        if not applied:
            break  # nothing deterministic left — leave residue to the LLM loop
        changes.extend(applied)
        last = None  # edits applied; the last compile no longer matches disk

    if changes:
        print(f"🔧 Semantic repair: {len(changes)} fix(es) — {changes}")

    # Ensure we end on a compile result that matches the tree on disk, so the
    # caller never has to recompile.
    if last is None:
        last = _compile_raw(output_dir)
    if not os.path.exists(os.path.join(output_dir, "pom.xml")):
        return changes, None
    code, stdout = last
    issues = _parse_maven_issues(stdout, output_dir, code)
    if code == 0:
        # test-compile passed → both main and tests compile.
        return changes, {"compiles": True, "test_compiles": True, "issues": issues}
    # test-compile failed. Split the signal: is the PRODUCTION code broken, or only
    # a generated test? main-compile is the hard gate (drives acceptance + the
    # last-green guard); a broken test must only flag its own file, not sink an
    # otherwise-shippable build. The residual `issues` here are the test-source
    # errors (test-compile builds main first, so if main were broken it'd have
    # failed there) → they surface as file-scoped issues for refinement.
    main_code, _ = _compile_main(output_dir)
    return changes, {"compiles": main_code == 0, "test_compiles": False, "issues": issues}
