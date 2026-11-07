"""VeriMAP-style per-task verification function (VF) library.

Each Python VF: (output_dir, args, blackboard) -> Optional[str]
  None = pass; str = the VeriMAP "assertion message" (feedback for self-refine).

Planner emits VF descriptors {vf: name, args: {...}} or {nl_vf: criterion} on each task.
run_task_vfs() dispatches them; attach_verifications() assigns them by task.type.

AGILE_VERIFY_AWARE modes:
  "0"      (default) — no-op; byte-identical to pre-VeriMAP behaviour.
  "shadow" — run VFs, log failures, return {} (detection only; no rejection).
  "1"      — run VFs, return failures → feed rejection + self-refine.

Safety fence: NL VFs are additive-only. They may reject a task; they never certify
one and never gate the last-green snapshot / rollback. Only deterministic Python VFs
hold acceptance + rollback authority.
"""
import os
import re
from typing import Dict, List, Optional

try:
    from agile_team_cline import java_probe as _jp
    _HAS_PROBE = True
except Exception:
    _HAS_PROBE = False

_JAVA_ID_RE = re.compile(r'^[A-Za-z_$][A-Za-z0-9_$]*$')


# ── Path helpers ─────────────────────────────────────────────────────────────

def _find_src_roots(output_dir: str) -> list:
    try:
        from agile_team_cline.workspace_discovery import _find_src_roots as _fsr
        return _fsr(output_dir) or []
    except Exception:
        return []


def _schema_path(output_dir: str, dialect: str) -> str:
    """Locate schema.sql for dialect via multi-module walk (fixes sprint.py:1715 path bug)."""
    for root in _find_src_roots(output_dir):
        # root = .../module/src/main/java → go up 3 levels to module root
        module_root = root
        for _ in range(3):
            module_root = os.path.dirname(module_root)
        candidate = os.path.join(module_root, "src", "main", "resources", "db", dialect, "schema.sql")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(output_dir, "src", "main", "resources", "db", dialect, "schema.sql")


def _data_path(output_dir: str, dialect: str) -> str:
    """Locate data.sql for dialect via multi-module walk."""
    for root in _find_src_roots(output_dir):
        module_root = root
        for _ in range(3):
            module_root = os.path.dirname(module_root)
        candidate = os.path.join(module_root, "src", "main", "resources", "db", dialect, "data.sql")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(output_dir, "src", "main", "resources", "db", dialect, "data.sql")


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _file_content(blackboard, target_file: str) -> Optional[str]:
    fname = os.path.basename(target_file)
    rel = blackboard.find_existing(fname)
    if not rel:
        return None
    try:
        return (blackboard.output_dir / rel).read_text()
    except Exception:
        return None


# ── Tier 1: deterministic, safe to enforce ───────────────────────────────────

def vf_file_exists(output_dir: str, args: dict, blackboard) -> Optional[str]:
    fname = args.get("file", "")
    if not fname:
        return None
    if not blackboard.find_existing(os.path.basename(fname)):
        return f"FILE MISSING: {fname} was not generated — task is incomplete."
    return None


def vf_parse_ok(output_dir: str, args: dict, blackboard) -> Optional[str]:
    if not _HAS_PROBE:
        return None
    fname = args.get("file", "")
    if not fname:
        return None
    rel = blackboard.find_existing(os.path.basename(fname))
    if not rel:
        return None
    full = os.path.join(str(blackboard.output_dir), rel)
    if _jp.parse_errors([full]):
        return f"PARSE ERROR: {fname} is truncated or has a syntax error. Rewrite the complete file."
    return None


def vf_valid_java_identifier(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Closes Bug 1: labels like 'Index (Featured)' must not flow into class/file names."""
    name = args.get("name", "")
    if not name:
        return None
    bare = name[:-5] if name.endswith(".java") else name
    bare = os.path.basename(bare)
    if not _JAVA_ID_RE.match(bare):
        sanitized = re.sub(r'\s+(.)', lambda m: m.group(1).upper(), bare)
        sanitized = re.sub(r'[^A-Za-z0-9_$]', '', sanitized)
        if sanitized and not _JAVA_ID_RE.match(sanitized[0]):
            sanitized = sanitized.lstrip('0123456789')
        sanitized = (sanitized[0].upper() + sanitized[1:]) if sanitized else "UnknownClass"
        return (f"INVALID IDENTIFIER: '{bare}' contains characters that are not valid in a Java "
                f"class/file name. Rename to '{sanitized}' (valid camelCase).")
    return None


def vf_table_in_all_dialects(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Closes Bug 3: h2-only gate + single-module path bug (sprint.py:1715).
    Uses the live @Table annotation name (same authority as the schema gate) rather than
    the contract-derived name, so both authorities agree and repair converges."""
    if not _HAS_PROBE:
        return None
    table = (args.get("table") or "").lower().strip()
    if not table:
        return None
    # Skip if ddl-auto=create/update (auto-DDL, no manual schema needed)
    props = _read(os.path.join(output_dir, "src", "main", "resources", "application.properties"))
    m = re.search(r"ddl-auto\s*=\s*(\S+)", props)
    if m and m.group(1).lower() not in ("none", "validate"):
        return None
    # Prefer live @Table annotation name — aligns VF with schema gate's authority so
    # both check the same name and auto-repair converges instead of cycling.
    target_file = args.get("target_file", "")
    if target_file:
        rel = blackboard.find_existing(os.path.basename(target_file))
        if rel:
            full = os.path.join(str(blackboard.output_dir), rel)
            live = _jp.entity_tables([full])
            if live:
                table = next(iter(live.keys())).lower()
    missing = []
    for dialect in ("h2", "mysql", "postgres"):
        sql = _read(_schema_path(output_dir, dialect))
        if table not in _jp.schema_tables(sql):
            missing.append(dialect)
    if missing:
        return (f"SCHEMA GAP: table '{table}' missing CREATE TABLE in: {', '.join(missing)}. "
                f"Add CREATE TABLE {table} (...) to db/{'/db/'.join(missing)}/schema.sql. "
                f"With ddl-auto=none every endpoint touching this entity 500s at runtime.")
    return None


def vf_seed_rows_present(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Closes Bug 3: data.sql never checked by any gate."""
    table = (args.get("table") or "").lower().strip()
    if not table:
        return None
    insert_re = re.compile(rf'INSERT\s+INTO\s+["`]?{re.escape(table)}["`]?\b', re.I)
    missing = []
    for dialect in ("h2", "mysql", "postgres"):
        path = _data_path(output_dir, dialect)
        text = _read(path)
        if not text:
            continue  # data.sql absent — not required; skip
        if not insert_re.search(text):
            missing.append(dialect)
    if missing:
        return (f"SEED GAP: no INSERT INTO {table} in data.sql for: {', '.join(missing)}. "
                f"Add seed rows to db/<dialect>/data.sql consistent with the schema and "
                f"existing seed-data style.")
    return None


def vf_endpoint_served(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Check each contract endpoint has a matching route handler.
    Uses Spring controller_mappings for Spring stacks; _jaxrs_mappings for JAX-RS."""
    if not _HAS_PROBE:
        return None
    endpoints = args.get("endpoints", [])
    if not endpoints:
        return None
    is_spring = args.get("is_spring", True)
    try:
        from agile_team_cline.sprint import _norm_endpoint_path
    except Exception:
        return None
    if is_spring:
        mapped = _jp.controller_mappings(output_dir)
    else:
        try:
            from agile_team_cline.sprint import _jaxrs_mappings
            mapped = _jaxrs_mappings(output_dir)
        except Exception:
            return None
    missing = []
    for ep in endpoints:
        parts = ep.split(" ", 1)
        if len(parts) != 2:
            continue
        method, path = parts[0].upper(), _norm_endpoint_path(parts[1])
        if (method, path) not in mapped:
            missing.append(ep)
    if missing:
        if is_spring:
            return (f"ENDPOINT GAP: no @*Mapping for: {', '.join(missing)}. "
                    f"Implement them in the controller (or extend the existing one).")
        else:
            return (f"ENDPOINT GAP: no @Path/@GET for: {', '.join(missing)}. "
                    f"Implement them in the JAX-RS resource (extend the existing one with "
                    f"@GET/@POST + @Path; do NOT create a Spring controller).")
    return None


# ── Tier 2: deterministic but parse-sensitive; shadow-validate before enforcing ──

def vf_template_or_servlet_for_view(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Closes Bug 2: Spring Thymeleaf views silently dropped (AGILE_PRESENTATION is jakarta-only)."""
    is_spring = args.get("is_spring", False)
    view_name = args.get("view_name", "")
    if is_spring:
        template = args.get("template", view_name.lower())
        tmpl_root = os.path.join(output_dir, "src", "main", "resources", "templates")
        found = False
        # Check direct path first
        for suffix in ("", ".html"):
            candidate = os.path.join(tmpl_root, *(template + suffix).split("/"))
            if os.path.exists(candidate):
                found = True
                break
        if not found:
            # Walk for any .html file containing view_name
            for dp, _, files in os.walk(tmpl_root):
                for fn in files:
                    if fn.endswith(".html") and view_name.lower() in fn.lower():
                        found = True
                        break
                if found:
                    break
        if not found:
            hint = f"templates/{template}.html"
            return (f"PRESENTATION GAP (Spring): no Thymeleaf template found for view '{view_name}'. "
                    f"Create {hint}. The controller must be @Controller returning a view name string, "
                    f"not @RestController returning JSON.")
    else:
        try:
            from agile_team_cline.sprint import _webservlet_routes, _webapp_pages, _norm_endpoint_path
        except Exception:
            return None
        issues = []
        route = args.get("route", "")
        if route and _norm_endpoint_path(route) not in _webservlet_routes(output_dir):
            issues.append(f"no @WebServlet serves route '{route}'")
        jsp = os.path.basename(args.get("jsp", "")).lower()
        if jsp and jsp not in _webapp_pages(output_dir):
            issues.append(f"JSP '{jsp}' not found under WEB-INF/pages/")
        if issues:
            return f"PRESENTATION GAP (Jakarta): {'; '.join(issues)}."
    return None


def vf_controller_kind_matches(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Closes run #24: @RestController used where Spring MVC @Controller + Thymeleaf expected."""
    expected = args.get("expected_kind", "")  # "MVC" or "REST"
    target_file = args.get("file", "")
    if not expected or not target_file:
        return None
    content = _file_content(blackboard, target_file)
    if content is None:
        return None
    if expected == "MVC":
        if "@RestController" in content:
            return (f"CONTROLLER KIND MISMATCH: {os.path.basename(target_file)} uses @RestController "
                    f"but this feature requires an MVC @Controller returning Thymeleaf view names. "
                    f"Change to @Controller and return the view name string instead of a response object.")
    elif expected == "REST":
        if "@Controller" in content and "@RestController" not in content:
            return (f"CONTROLLER KIND MISMATCH: {os.path.basename(target_file)} uses @Controller (view) "
                    f"but expected @RestController for a JSON API.")
    return None


def vf_field_relationship(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Closes run #24: private Integer petId instead of @ManyToOne Pet pet."""
    target_file = args.get("file", "")
    field = args.get("field", "")    # e.g. "pet" (without "Id")
    if not target_file or not field:
        return None
    content = _file_content(blackboard, target_file)
    if content is None:
        return None
    # Bare scalar FK pattern: private Integer/Long/int/long <field>Id ;
    scalar_re = re.compile(
        rf'private\s+(?:Integer|Long|int|long|String)\s+{re.escape(field)}Id\s*;', re.M)
    if scalar_re.search(content) and "@ManyToOne" not in content:
        field_cap = field[0].upper() + field[1:]
        return (f"RELATIONSHIP GAP: {os.path.basename(target_file)} uses a bare scalar FK "
                f"(private Integer/Long {field}Id) instead of a JPA relationship. "
                f"Declare '@ManyToOne private {field_cap} {field};' (with @JoinColumn as needed).")
    return None


def vf_handoff_wired(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Closes Bug 4: servlet not calling registryclient LoadBalanced*Operations."""
    target_file = args.get("file", "")
    if not target_file:
        return None
    content = _file_content(blackboard, target_file)
    if content is None:
        return None
    if not re.search(r'LoadBalanced\w*Operations', content):
        return (f"HANDOFF GAP: {os.path.basename(target_file)} does not reference any "
                f"LoadBalanced*Operations. The servlet must call the registryclient to reach "
                f"the persistence service — do NOT call it directly.")
    return None


# ── Registry ─────────────────────────────────────────────────────────────────

def _class_level_path(src: str) -> Optional[str]:
    """The class-level @Path("...") of a JAX-RS resource = the last @Path before the first
    `class` keyword (method-level @Paths come after). None if the file is not a @Path resource."""
    m = re.search(r'\bclass\s+\w+', src)
    if not m:
        return None
    head = src[:m.start()]
    paths = re.findall(r'@Path\s*\(\s*"([^"]*)"', head)
    return paths[-1].strip("/") if paths else None


def vf_no_duplicate_route(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Closes the dupes false-green (run #51): the feature was cloned into a PARALLEL JAX-RS
    resource — a second class-level @Path("products") (ProductController.java) alongside the real
    ProductEndpoint.java — a deploy-time root-path collision the compiler cannot catch. The PO
    overrode QA's collision warning and accepted → false-green. Deterministic ceiling: within any
    single directory, two DISTINCT resource files sharing the same non-empty class-level @Path
    root is a collision (the correct fix is to add the new @GET/@Path method to the existing
    resource, not a parallel controller). JAX-RS only (Spring uses @RequestMapping); returns None
    (pass) on Spring or when clean. NOTE: TeaStore keeps *Repository classes in the `domain`
    package by convention, so repository PLACEMENT is deliberately NOT checked here."""
    if args.get("is_spring"):
        return None
    try:
        from agile_team_cline.workspace_discovery import _find_src_roots
        roots = _find_src_roots(output_dir) or [os.path.join(output_dir, "src", "main", "java")]
    except Exception:
        roots = [os.path.join(output_dir, "src", "main", "java")]
    by_dir: dict = {}          # dir -> { root_path -> set(filenames) }
    for root in roots:
        for dp, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
            for fn in files:
                if not fn.endswith(".java"):
                    continue
                try:
                    src = open(os.path.join(dp, fn), encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                cpath = _class_level_path(src)
                if cpath:
                    by_dir.setdefault(dp, {}).setdefault(cpath, set()).add(fn)
    issues = []
    for dp, roots_map in by_dir.items():
        for cpath, fnames in roots_map.items():
            if len(fnames) > 1:
                issues.append(
                    f"DUPLICATE JAX-RS ROOT @Path(\"{cpath}\"): {', '.join(sorted(fnames))} "
                    f"in {os.path.basename(dp)}/ — two resource classes claim the same root path "
                    f"(deploy-time collision). Consolidate into ONE resource; add the new @GET/@Path "
                    f"method to the existing resource instead of a parallel controller.")
    return " | ".join(issues) if issues else None


# Built-in / framework types that never need a project source file. Anything else
# appearing in a contract type expression is treated as a project type.
_BUILTIN_TYPES = {
    "String", "Integer", "Long", "Double", "Float", "Boolean", "Short", "Byte",
    "Character", "Object", "Number", "BigDecimal", "BigInteger", "UUID",
    "LocalDate", "LocalDateTime", "Instant", "Date", "Timestamp",
    "List", "Set", "Map", "Collection", "Iterable", "Optional", "Response",
    "void", "Void", "int", "long", "double", "float", "boolean", "short", "byte", "char",
}

# Annotations/signatures that make a file an ENTRY POINT: nothing else has to
# reference it because the framework (or the JVM) invokes it.
_ENTRYPOINT_RE = re.compile(
    r'@Path\s*\(|@WebServlet\b|@RestController\b|@Controller\b|'
    r'@SpringBootApplication\b|static\s+void\s+main\s*\(')

_LINE_COMMENT_RE = re.compile(r'//[^\n]*')
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.S)


def _strip_comments(src: str) -> str:
    return _LINE_COMMENT_RE.sub("", _BLOCK_COMMENT_RE.sub("", src))


def _contract_type_names(contract: dict) -> set:
    """Every project type the contract declares: entity/dto names plus all custom
    identifiers inside endpoint request/response type expressions (generics stripped)."""
    names = set()
    for e in contract.get("entities") or []:
        if isinstance(e, dict) and e.get("name"):
            names.add(e["name"])
    for ep in contract.get("endpoints") or []:
        if not isinstance(ep, dict):
            continue
        for key in ("request", "response"):
            expr = ep.get(key)
            if isinstance(expr, str):
                for ident in re.findall(r'[A-Za-z_]\w*', expr):
                    if ident not in _BUILTIN_TYPES and ident[:1].isupper():
                        names.add(ident)
    return names


def _java_sources(output_dir: str) -> dict:
    """{relpath: content} for every .java under the workspace src roots (skips target/.git)."""
    out = {}
    roots = _find_src_roots(output_dir) or [os.path.join(output_dir, "src", "main", "java")]
    seen_roots = set()
    for root in roots:
        # roots are .../src/main/java — widen to the module's src/ so tests count too
        module_src = os.path.dirname(os.path.dirname(root))
        if module_src in seen_roots or not os.path.isdir(module_src):
            continue
        seen_roots.add(module_src)
        for dp, dirs, files in os.walk(module_src):
            dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
            for fn in files:
                if fn.endswith(".java"):
                    p = os.path.join(dp, fn)
                    try:
                        out[os.path.relpath(p, output_dir)] = open(
                            p, encoding="utf-8", errors="ignore").read()
                    except OSError:
                        pass
    return out


def vf_contract_types_realized(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Two universal relationships between plan and code, checked on the whole tree:

    1. CONTRACT DRIFT — every type the contract declares (entity/dto names, endpoint
       request/response types) must exist as a source file AND be referenced by at least
       one other source file. A declared-but-absent or declared-but-unreferenced type
       means the plan and the code diverged, whichever side is wrong. Types already
       present in the pre-run baseline are grandfathered (they carry their own history).

    2. ORPHAN ARTIFACT — every NEW production file this run created must be referenced
       by at least one other source file, unless it is an entry point the framework
       invokes (@Path/@WebServlet/@*Controller/@SpringBootApplication/main). An orphan
       is always a defect: either it is unnecessary or the code that should use it
       doesn't. Contract-agnostic; requires the baseline snapshot (skipped without one).

    References are word-boundary matches on comment-stripped source, so a mention in
    javadoc or in a lookalike name (FooTest containing Foo) never counts."""
    contract = blackboard.read("contract") or {}
    baseline = blackboard.read("baseline_src") or {}
    sources = _java_sources(output_dir)
    if not sources:
        return None
    stripped = {rel: _strip_comments(src) for rel, src in sources.items()}
    baseline_basenames = {os.path.basename(rel) for rel in baseline}

    def _referenced(stem: str, own_rel: str) -> bool:
        pat = re.compile(rf'\b{re.escape(stem)}\b')
        return any(pat.search(body) for rel, body in stripped.items() if rel != own_rel)

    issues = []

    # 1. Contract realization
    on_disk = {}  # basename -> relpath
    for rel in stripped:
        on_disk.setdefault(os.path.basename(rel), rel)
    for name in sorted(_contract_type_names(contract)):
        fname = f"{name}.java"
        if fname in baseline_basenames:
            continue  # pre-existing type — grandfathered
        rel = on_disk.get(fname)
        if rel is None:
            issues.append(
                f"CONTRACT DRIFT: type {name} is declared in the contract "
                f"(entity/dto or endpoint request/response) but no {fname} exists. "
                f"Create it with that exact name, or align the code that serves it.")
        elif not _referenced(name, rel):
            issues.append(
                f"CONTRACT DRIFT: {fname} exists but no other source file references "
                f"{name} — the contract's type is dead code. Wire it into the code that "
                f"produces/serves it (typed return instead of ad-hoc maps), or delete it "
                f"and fix the contract.")

    # 2. Orphan artifacts (needs a baseline to know what is new)
    if baseline:
        for rel, body in stripped.items():
            norm = rel.replace("\\", "/")
            if "/src/main/" not in f"/{norm}" or rel in baseline:
                continue
            base = os.path.basename(rel)
            if base == "package-info.java" or _ENTRYPOINT_RE.search(body):
                continue
            stem = base[:-len(".java")]
            if not _referenced(stem, rel):
                issues.append(
                    f"ORPHAN ARTIFACT: {base} was created this run but nothing references "
                    f"{stem}. Wire it into the code that should use it, or delete it.")

    return " | ".join(issues) if issues else None


# ── Runtime-execution gate: the query must RUN, not merely compile ───────────
# Timing-exclusion guardrail. This VF must NEVER fire on "the junior is slow", only on
# "the junior throws / cannot execute". Two independent fences enforce that:
#   1. It reads only surefire <error> elements (a thrown exception). <failure> elements —
#      how an assertion, including any timing/perf assertion, manifests — are ignored.
#   2. Any error whose type or message carries performance vocabulary is skipped, so even
#      a thrown "too slow"/timeout can never trip it.
# A worse-but-working junior (extra queries, re-tokenizing — code that still executes)
# therefore sails through; only code that cannot run is re-opened for refinement.
_PERF_EXCLUDE_RE = re.compile(
    r'(elapsed|took\s+\d+\s*m?s|slower|faster|performance|\bperf\b|regression|latency|'
    r'throughput|benchmark|timed?\s*out|exceeded\s+\d+\s*m?s|\bbudget\b|'
    r'\bms\b.*(assert|expect)|(assert|expect).*\bms\b)',
    re.IGNORECASE)


def _surefire_report_dirs(output_dir: str) -> list:
    """Every target/surefire-reports directory under the workspace (reactor-safe)."""
    found = []
    for root in _find_src_roots(output_dir):
        module_root = root
        for _ in range(3):  # src/main/java → module root
            module_root = os.path.dirname(module_root)
        d = os.path.join(module_root, "target", "surefire-reports")
        if os.path.isdir(d):
            found.append(d)
    if not found:  # fallback: bounded walk in case src roots weren't discovered
        base_depth = output_dir.rstrip(os.sep).count(os.sep)
        for dp, dns, _fns in os.walk(output_dir):
            if os.path.basename(dp) == "surefire-reports":
                found.append(dp)
            if dp.count(os.sep) - base_depth > 6:
                dns[:] = []
    return sorted(set(found))


# ── JPA registration gate: a managed class the persistence unit never lists ──────────
# `mvn test` bootstraps the persistence unit in Java SE mode, where the provider is NOT
# required to auto-discover annotated classes — and EclipseLink does not. An @Entity absent
# from persistence.xml's <class> list compiles cleanly, then throws "The abstract schema
# type 'X' is unknown" on the first query naming it.
#
# This needs its own gate because failure→source attribution structurally cannot reach it:
# a stack trace names *executing code*, so it points at the repository that ran the query,
# never at the descriptor that is actually wrong (volatility 2026-07-28: 11 errors across 4
# test classes, every one tracing to PersistenceProductPriceChangeRepository; the real fix
# was one <class> line in an XML file that appears in no frame). Configuration defects have
# no stack frame — the answer is a static invariant, not better trace parsing.
_MANAGED_ANNO_RE = re.compile(
    r"^\s*@(?:jakarta\.persistence\.|javax\.persistence\.)?"
    r"(Entity|MappedSuperclass|Embeddable|Converter)\b", re.M)
_JAVA_PKG_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
_PU_CLASS_RE = re.compile(r"<class>\s*([\w.$]+)\s*</class>")
# When the unit opts into scanning, listing is not required and this whole check is moot.
_PU_SCANS_RE = re.compile(
    r"<exclude-unlisted-classes>\s*false\s*</exclude-unlisted-classes>", re.I)

_PU_REL = os.sep.join(["src", "main", "resources", "META-INF"])


def _persistence_units(output_dir: str) -> list:
    """Every main-scope persistence.xml in the workspace (target/ copies excluded, and
    scoped to AGILE_BUILD_MODULE when set). Empty on Spring stacks, which configure JPA
    through application.properties and ship no descriptor — so every caller no-ops there."""
    module = os.getenv("AGILE_BUILD_MODULE", "").strip()
    mods = [m.strip() for m in module.split(",") if m.strip()] if module else []
    found = []
    for dp, dns, fns in os.walk(output_dir):
        dns[:] = [d for d in dns if d not in ("target", ".git", "node_modules")]
        if "persistence.xml" not in fns or not dp.endswith(_PU_REL):
            continue
        p = os.path.join(dp, "persistence.xml")
        if mods and not any(m in p for m in mods):
            continue
        found.append(p)
    return sorted(found)


def unregistered_managed_classes(output_dir: str) -> Dict[str, list]:
    """{persistence.xml path: [(FQCN, annotation), …]} for JPA managed classes that exist
    on disk but are not listed in the unit. Empty when all are registered, when the unit
    sets exclude-unlisted-classes=false, or when there is no descriptor at all.

    Shared detector: the VF below reports it, and any future deterministic repair would
    consume the same function, so the two can never drift apart (the rule this codebase
    already applies to repair.parse_maven_errors)."""
    out: Dict[str, list] = {}
    for px in _persistence_units(output_dir):
        try:
            with open(px, encoding="utf-8", errors="ignore") as fh:
                xml = fh.read()
        except OSError:
            continue
        if _PU_SCANS_RE.search(xml):
            continue  # provider scans the unit root; explicit listing not required
        listed = set(_PU_CLASS_RE.findall(xml))
        java_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(px))), "java")
        if not os.path.isdir(java_root):
            continue
        missing = []
        for dp, _dn, fns in os.walk(java_root):
            for fn in fns:
                if not fn.endswith(".java"):
                    continue
                try:
                    with open(os.path.join(dp, fn), encoding="utf-8", errors="ignore") as fh:
                        txt = fh.read()
                except OSError:
                    continue
                # Line-anchored: a mention inside a comment or string never counts.
                m = _MANAGED_ANNO_RE.search(txt)
                if not m:
                    continue
                pkg = _JAVA_PKG_RE.search(txt)
                fqcn = f"{pkg.group(1)}.{fn[:-5]}" if pkg else fn[:-5]
                if fqcn not in listed:
                    missing.append((fqcn, m.group(1)))
        if missing:
            out[px] = sorted(missing)
    return out


def vf_entity_registered(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Every JPA managed class on disk must appear in the persistence unit's <class> list.

    Whole-tree check (like contract_types_realized): it scans the module rather than this
    task's target file, because the class and the descriptor are always different files.
    No-op on Spring (no persistence.xml) and on units that set exclude-unlisted-classes=false.

    Deliberately says nothing about performance — it only requires that the entity be
    *reachable*, exactly like vf_query_executes' run-don't-optimize contract."""
    missing = unregistered_managed_classes(output_dir)
    if not missing:
        return None
    parts = []
    for px, items in sorted(missing.items()):
        try:
            rel = os.path.relpath(px, output_dir)
        except ValueError:
            rel = px
        names = ", ".join(f"{c} (@{a})" for c, a in items)
        parts.append(f"{rel} is missing: {names}")
    return (
        "UNREGISTERED JPA CLASS — " + "; ".join(parts) + ". The tests bootstrap the "
        "persistence unit in Java SE mode, where EclipseLink does NOT auto-discover "
        "annotated classes: the code compiles, then every query naming the type fails with "
        "\"The abstract schema type is unknown\". Add a <class> entry with each "
        "fully-qualified name to that persistence.xml, in place — do not create a second "
        "descriptor and do not remove the entries already there.")


# Stack frames whose owning class is not project code. JDK9+ frames carry a module
# prefix ("java.base/java.util.stream.…"), so the module segment is stripped first.
_FOREIGN_FRAME_PREFIXES = (
    "java.", "javax.", "jakarta.", "sun.", "jdk.", "com.sun.",
    "org.junit", "junit.", "org.hamcrest", "org.mockito",
    "org.eclipse.", "org.apache.", "org.hibernate.", "org.springframework.",
)
# The method-name group admits <> so constructor (`.<init>`) and static-initializer
# (`.<clinit>`) frames match — a defect thrown from a JPA converter/entity constructor
# otherwise drops out of findall() and gets attributed to the calling file instead.
_FRAME_RE = re.compile(r"\bat\s+(?:[\w.$]+/)?([\w.$<>]+)\(([\w$]+\.java):\d+\)")


def _is_test_source(filename: str) -> bool:
    stem = filename[:-5] if filename.endswith(".java") else filename
    return stem.endswith(("Test", "Tests", "IT"))


def _first_project_frame(trace: str) -> Optional[str]:
    """The topmost stack frame that is neither library code nor a test class — i.e. the
    production source that actually threw. None when the trace has no such frame (a plain
    assertion failure originates in the test itself)."""
    for fqcn_method, filename in _FRAME_RE.findall(trace or ""):
        fqcn = fqcn_method.rsplit(".", 1)[0]
        if fqcn.startswith(_FOREIGN_FRAME_PREFIXES) or _is_test_source(filename):
            continue
        return filename
    return None


def owning_source_files(output_dir: str) -> Dict[str, tuple]:
    """Map each failing test class to the PRODUCTION file that threw inside it, read from
    the surefire XML the module's own `mvn test` already produced.

    Why this exists: sprint._parse_surefire attributes a failure to `<FailingTestClass>.java`
    — the test, not the code under test — and the refinement brief is filtered to the
    current tasks' target files. When the defect lives in a pre-existing file that owns no
    backlog task (rfm 2026-08-02: UserRepository.getRfmSegments threw ClassCastException on
    6/6 cases, while the only task was the DTO), refinement can literally never open it. The
    stack trace names the file; nothing was reading it.

    TIMING-EXCLUSION GUARANTEE — inherited wholesale from vf_query_executes above, and the
    reason this is safe to feed back to the junior. It reads only <error> (thrown), never
    <failure> (assertion), so a perf/timing assertion can never route; and it additionally
    skips any error carrying performance vocabulary. The pipeline must never help the junior
    be FASTER, only make it RUN. Note the restriction costs nothing: a production frame only
    appears in the trace when production code threw, so the failures the guardrail permits
    and the failures this can serve are the same set.

    Returns {"<TestClass>.java": ("<OwningSource>.java", "ExcType: message")}; empty when
    nothing qualifies.
    """
    import xml.etree.ElementTree as ET

    mapping: Dict[str, tuple] = {}
    for d in _surefire_report_dirs(output_dir):
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for fn in names:
            if not (fn.startswith("TEST-") and fn.endswith(".xml")):
                continue
            try:
                root = ET.parse(os.path.join(d, fn)).getroot()
            except Exception:
                continue
            for tc in root.iter("testcase"):
                err = tc.find("error")  # <error> only; <failure> (assertions) ignored by design
                if err is None:
                    continue
                etype = (err.get("type") or "").strip()
                emsg = (err.get("message") or "").strip()
                trace = err.text or ""
                if _PERF_EXCLUDE_RE.search(f"{etype} {emsg} {trace}"):
                    continue  # timing-excluded: never route on slowness
                owner = _first_project_frame(trace)
                if not owner:
                    continue
                cls = (tc.get("classname") or "").rsplit(".", 1)[-1].split("$")[0]
                if not cls:
                    continue
                key = f"{cls}.java"
                if key in mapping:
                    continue  # first failure per test class is enough to aim refinement
                first = emsg.splitlines()[0].strip() if emsg else ""
                mapping[key] = (owner, f"{etype.split('.')[-1] or 'Exception'}: {first[:200]}")
    return mapping


def vf_query_executes(output_dir: str, args: dict, blackboard) -> Optional[str]:
    """Runtime-execution gate: the task's query/endpoint must EXECUTE, not just compile.

    Reads the surefire reports the module's `mvn test` already produced and surfaces any
    thrown exception (<error>) scoped to this task's classes — a QueryException,
    PersistenceException, NPE, etc. For a DB-query feature that is the difference between a
    *broken* sample and a *worse-but-working* one: the naive junior must still run correctly,
    just (possibly) slower. Feeding the real exception back re-opens the task so the junior
    fixes it during refinement, exactly as it would after running its own tests.

    TIMING-EXCLUSION GUARANTEE — this gate is blind to slowness. It reads only <error>
    (thrown) entries, never <failure> (assertion) entries, so a perf/timing assertion can
    never trip it; and it additionally skips any error whose text carries performance
    vocabulary. It emits no hint about *how* to write the query — only that it must run —
    so the naive-implementation signal the experiment measures is preserved."""
    import xml.etree.ElementTree as ET
    hints = [h for h in (args.get("class_hints") or []) if h]
    report_dirs = _surefire_report_dirs(output_dir)
    if not report_dirs:
        return None  # no test run to read — the compile gate is the signal here

    seen = set()
    defects = []
    for d in report_dirs:
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        for fn in names:
            if not (fn.startswith("TEST-") and fn.endswith(".xml")):
                continue
            try:
                root = ET.parse(os.path.join(d, fn)).getroot()
            except Exception:
                continue
            for tc in root.iter("testcase"):
                err = tc.find("error")  # <error> only; <failure> (assertions) ignored by design
                if err is None:
                    continue
                etype = (err.get("type") or "").strip()
                emsg = (err.get("message") or "").strip()
                haystack = f"{tc.get('classname', '')} {tc.get('name', '')} {etype} {emsg} {err.text or ''}"
                if _PERF_EXCLUDE_RE.search(haystack):
                    continue  # timing-excluded: never fire on slowness
                if hints and not any(h in haystack for h in hints):
                    continue  # outside this task's scope
                first = emsg.splitlines()[0].strip() if emsg else ""
                key = (etype, first)
                if key in seen:
                    continue
                seen.add(key)
                short = (first or etype or "exception")[:200]
                defects.append(f"{etype.split('.')[-1] or 'Exception'}: {short}")

    if not defects:
        return None
    more = f" (+{len(defects) - 1} more)" if len(defects) > 1 else ""
    return (
        f"RUNTIME EXECUTION DEFECT — the query/endpoint compiles but THROWS at run time "
        f"instead of executing: {defects[0]}{more}. Make the operation run to completion "
        f"against the test database (fix the query/mapping so it does not throw). This gate "
        f"ignores how SLOW the code is — it only requires that it executes without error.")


VF_REGISTRY = {
    "file_exists":                  vf_file_exists,
    "parse_ok":                     vf_parse_ok,
    "valid_java_identifier":        vf_valid_java_identifier,
    "table_in_all_dialects":        vf_table_in_all_dialects,
    "seed_rows_present":            vf_seed_rows_present,
    "endpoint_served":              vf_endpoint_served,
    "template_or_servlet_for_view": vf_template_or_servlet_for_view,
    "controller_kind_matches":      vf_controller_kind_matches,
    "field_relationship":           vf_field_relationship,
    "handoff_wired":                vf_handoff_wired,
    "no_duplicate_route":           vf_no_duplicate_route,
    "contract_types_realized":      vf_contract_types_realized,
    "query_executes":               vf_query_executes,
    "entity_registered":            vf_entity_registered,
}


# ── Runner ───────────────────────────────────────────────────────────────────

def run_task_vfs(tasks: list, blackboard, stack: dict,
                 mode: str = "1") -> Dict[str, List[str]]:
    """Run all VF descriptors on each task's generated output.

    Returns {task_id: [feedback_strings]} when mode="1";
    returns {} when mode="shadow" (logs only) or mode="0" (no-op).
    """
    if mode == "0":
        return {}
    output_dir = str(blackboard.output_dir)
    result: Dict[str, List[str]] = {}

    for task in tasks:
        tid = task.get("id", "")
        verifications = task.get("verifications", [])
        if not verifications:
            continue
        msgs: List[str] = []
        for vfd in verifications:
            if "vf" in vfd:
                fn = VF_REGISTRY.get(vfd["vf"])
                if fn is None:
                    continue
                try:
                    msg = fn(output_dir, vfd.get("args", {}), blackboard)
                except Exception as exc:
                    msg = f"VF ERROR ({vfd['vf']}): {exc}"
                if msg:
                    msgs.append(msg)
                    if mode == "shadow":
                        print(f"   🔍 [SHADOW VF] [{tid}] {msg}")
            elif "nl_vf" in vfd and mode == "1":
                msg = _run_nl_vf(output_dir, vfd["nl_vf"],
                                 task.get("target_file", ""), blackboard)
                if msg:
                    msgs.append(f"[SEMANTIC] {msg}")
        if msgs and mode == "1":
            result[tid] = msgs
            for m in msgs:
                print(f"   🔴 [VF] [{tid}] {m}")
    return result


def _run_nl_vf(output_dir: str, criterion: str, target_file: str,
               blackboard) -> Optional[str]:
    """Natural-language VF: LLM verifier scoped to one task's files.
    Additive-only — errors are silent so a flaky LLM never blocks a compiling tree."""
    import json
    try:
        from agile_team_cline.cline_runner import run_cline_with_retry
    except Exception:
        return None
    fname = os.path.basename(target_file) if target_file else ""
    rel = blackboard.find_existing(fname) if fname else None
    if not rel:
        return None
    try:
        content = (blackboard.output_dir / rel).read_text()[:4000]
    except Exception:
        return None
    brief = (
        f"You are a code verifier. Check ONLY the following file against the criterion.\n\n"
        f"FILE: {rel}\n```java\n{content}\n```\n\n"
        f"CRITERION: {criterion}\n\n"
        f'Respond ONLY with valid JSON on one line: {{"pass": true/false, "feedback": "..."}}. '
        f"If pass=false, feedback must explain specifically what is wrong and how to fix it."
    )
    system = (
        "Code verifier. Respond ONLY with a single JSON object on one line: "
        '{"pass": true, "feedback": ""} or {"pass": false, "feedback": "issue + fix"}. '
        "No markdown. Only JSON."
    )
    try:
        res = run_cline_with_retry(
            output_dir, brief, plan=True, system=system,
            timeout=int(os.getenv("AGILE_LLM_TIMEOUT", "300")),
            title=f"nl_vf_{fname[:20]}",
        )
        m = re.search(r'\{[^{}]*"pass"[^{}]*\}', res.text or "")
        if m:
            obj = json.loads(m.group())
            if not obj.get("pass", True):
                return obj.get("feedback", "Semantic check failed.")
    except Exception:
        pass  # NL VF errors never block
    return None


# ── Planner: attach VF descriptors to tasks ──────────────────────────────────

def attach_verifications(task: dict, contract: dict, stack: dict) -> dict:
    """Attach VeriMAP verification descriptors to a task (additive, in-place + return).
    Called by build_coverage_tasks when AGILE_VERIFY_AWARE != "0".
    Deterministic: maps task.type → VF set; planner emits only name+args, never code."""
    spring = ("spring" in str(stack.get("framework", "")).lower()
              or "spring" in str(stack.get("name", "")).lower())
    task_type = task.get("type", "")
    target_file = task.get("target_file", "")
    task_id = task.get("id", "")
    vfs: list = []

    # Derive entity name from task id (e.g. "ENT-Vaccination" → "Vaccination")
    entity_name = ""
    for prefix in ("ENT-", "REPO-", "SVC-", "CTL-", "DTO-", "WEB-", "JSP-"):
        if task_id.startswith(prefix):
            entity_name = task_id[len(prefix):]
            break

    # Find contract entity spec for this entity
    entity_spec = None
    if entity_name and isinstance(contract.get("entities"), list):
        entity_spec = next(
            (e for e in contract["entities"] if e.get("name") == entity_name), None)

    def _table(name: str) -> str:
        if entity_spec and entity_spec.get("table"):
            return entity_spec["table"].lower()
        s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s).lower()

    # ── file_exists + parse_ok — every Java task ─────────────────────────────
    if target_file:
        vfs.append({"vf": "file_exists", "args": {"file": target_file}})
        if target_file.endswith(".java"):
            vfs.append({"vf": "parse_ok", "args": {"file": target_file}})

    # ── contract realization + orphan artifacts — every code-producing task ──
    # Global invariant (whole-tree check, not scoped to this task's file): every type
    # the contract declares must exist AND be referenced; every new production file must
    # be referenced or be an entry point. Attached to code tasks so both the QA VF pass
    # and the PO floor consult it — a green-but-contract-drifted tree can no longer be
    # floor-certified.
    if task_type in ("entity", "dto", "controller", "endpoint", "repository", "service"):
        vfs.append({"vf": "contract_types_realized", "args": {}})

    # ── entity_registered — non-Spring persisted types only ──────────────────
    # Attached to BOTH the entity task (which creates the class) and the repository task
    # (whose query is what actually throws when the registration is missing), so whichever
    # one is under review re-opens. Whole-tree check, so no args. Spring is excluded: it
    # scans the classpath and ships no persistence.xml, so the VF would only ever no-op.
    if not spring and task_type in ("entity", "repository"):
        vfs.append({"vf": "entity_registered", "args": {}})

    # ── valid_java_identifier — servlet tasks only (.java files; JSP names need not be valid Java ids) ─
    if task_type in ("servlet", "view") and target_file and target_file.endswith(".java"):
        bare = os.path.basename(target_file)[:-5]  # strip .java
        if not _JAVA_ID_RE.match(bare):
            vfs.append({"vf": "valid_java_identifier", "args": {"name": os.path.basename(target_file)}})

    # ── entity: schema + seed + FK relationships (Spring only) ──────────────
    if task_type == "entity" and entity_name and spring:
        table = _table(entity_name)
        vfs.append({"vf": "table_in_all_dialects", "args": {"table": table, "target_file": target_file}})
        vfs.append({"vf": "seed_rows_present", "args": {"table": table}})
        if entity_spec:
            for field_name in (entity_spec.get("fields") or {}):
                if field_name.endswith("Id") and field_name != "id":
                    vfs.append({"vf": "field_relationship", "args": {
                        "file": target_file,
                        "owner": entity_name,
                        "field": field_name[:-2],
                    }})

    # ── controller: endpoint coverage + Spring MVC vs REST + Thymeleaf check ─
    if task_type in ("controller", "endpoint") and target_file:
        owned_eps = []
        if isinstance(contract.get("endpoints"), list):
            for ep in contract["endpoints"]:
                method = ep.get("method", "GET") if isinstance(ep, dict) else "GET"
                path = ep.get("path", "") if isinstance(ep, dict) else ""
                if method and path:
                    owned_eps.append(f"{method} {path}")
        if owned_eps:
            vfs.append({"vf": "endpoint_served", "args": {"endpoints": owned_eps, "is_spring": spring}})
        if task_type == "controller" and spring:
            views = contract.get("views") or []
            if views:
                # Controller is expected to be MVC (Thymeleaf)
                vfs.append({"vf": "controller_kind_matches",
                            "args": {"file": target_file, "expected_kind": "MVC"}})
                for v in views:
                    if not isinstance(v, dict) or not v.get("name"):
                        continue
                    tmpl = v.get("template") or v.get("name", "").lower()
                    vfs.append({"vf": "template_or_servlet_for_view", "args": {
                        "is_spring": True,
                        "view_name": v["name"],
                        "template": tmpl,
                    }})
                    vfs.append({"nl_vf": (
                        f"Confirm the controller is @Controller (not @RestController), returns the "
                        f"view name '{tmpl}' as a String, and populates the Model with the data "
                        f"the '{v['name']}' Thymeleaf template needs."
                    )})
            else:
                # No views — REST controller expected
                vfs.append({"vf": "controller_kind_matches",
                            "args": {"file": target_file, "expected_kind": "REST"}})

    # ── runtime-execution gate: repository/endpoint queries must actually RUN ─
    # Reads the surefire reports the module's test run already produced; re-opens the task
    # if its query/endpoint THROWS (broken sample) — never if it merely runs slowly
    # (worse-but-working sample stays valid). Scoped to this task's classes via hints so
    # unrelated failures elsewhere don't attach here.
    if task_type in ("repository", "endpoint", "controller"):
        hints = set()
        if target_file:
            base = os.path.basename(target_file)
            hints.add(base[:-5] if base.endswith(".java") else base)
        if entity_name:
            hints.add(entity_name)
        for e in (contract.get("entities") or []):
            if isinstance(e, dict) and e.get("name"):
                hints.add(e["name"])
        for ep in (contract.get("endpoints") or []):
            if not isinstance(ep, dict):
                continue
            if ep.get("response"):
                hints.add(str(ep["response"]))
            seg = str(ep.get("path", "")).rstrip("/").split("/")[-1]
            for tok in re.split(r'[-_]', seg):
                if len(tok) >= 4:
                    hints.add(tok[:1].upper() + tok[1:])
        vfs.append({"vf": "query_executes",
                    "args": {"class_hints": sorted(h for h in hints if h)}})

    # ── servlet (Jakarta): view + registryclient handoff ─────────────────────
    if task_type == "servlet" and target_file:
        view_name = task_id[len("WEB-"):] if task_id.startswith("WEB-") else entity_name
        view_spec = next(
            (v for v in (contract.get("views") or [])
             if isinstance(v, dict) and v.get("name") == view_name), {})
        vfs.append({"vf": "template_or_servlet_for_view", "args": {
            "is_spring": False,
            "view_name": view_name,
            "route": view_spec.get("route", ""),
            "jsp": view_spec.get("jsp", ""),
        }})
        vfs.append({"vf": "handoff_wired", "args": {"file": target_file}})

    task["verifications"] = vfs
    return task
