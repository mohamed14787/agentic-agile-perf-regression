"""tree-sitter Java probe for the coverage/QA gates.

Deterministic, no JVM, parses source text so it works on a NON-compiling tree (exactly
when the refinement loop needs endpoint info). Three pure functions used by sprint.py:

  controller_mappings(output_dir) -> set[(METHOD, norm_path)]   # JSON-serving routes only
  controllers(output_dir)         -> list[{class, kind, base_path, endpoints}]
  parse_errors(paths)             -> list[path]                 # truncated/unparseable

Importing this module fails cleanly if the wheels are absent; sprint.py guards with
_HAS_TS and falls back to its regex scanners. This module must NOT import sprint
(circular) — the path normalizer is replicated here to match sprint._norm_endpoint_path.
"""
import os
import re

from tree_sitter import Language, Parser
import tree_sitter_java as tsj

_LANG = Language(tsj.language())
_PARSER = Parser(_LANG)

_MAPPING_HTTP = {
    "GetMapping": "GET", "PostMapping": "POST", "PutMapping": "PUT",
    "DeleteMapping": "DELETE", "PatchMapping": "PATCH",
}


def _norm_endpoint_path(path: str) -> str:
    """Mirror of sprint._norm_endpoint_path: strip path-variable names, normalize slashes.
    '/pets/{petId}/visits' -> '/pets/{}/visits'."""
    p = re.sub(r"\{[^}]*\}", "{}", (path or "").strip())
    p = "/" + p.strip("/")
    return p.rstrip("/") or "/"


def _txt(node, src):
    return src[node.start_byte:node.end_byte].decode("utf-8", "replace")


def _annotations(node, src):
    """[(name, raw-args-or-None)] for annotations directly modifying `node`."""
    out = []
    for child in node.children:
        if child.type == "modifiers":
            for m in child.children:
                if m.type in ("annotation", "marker_annotation"):
                    name = args = None
                    for c in m.children:
                        if c.type in ("identifier", "scoped_identifier"):
                            name = _txt(c, src).split(".")[-1]
                        elif c.type == "annotation_argument_list":
                            args = _txt(c, src)
                    if name:
                        out.append((name, args))
    return out


def _first_path(args):
    m = re.findall(r'"([^"]*)"', args or "")
    return m[0] if m else ""


def _join(base, sub):
    base = (base or "").rstrip("/")
    sub = (sub or "").strip()
    if sub and not sub.startswith("/"):
        sub = "/" + sub
    return (base + sub) or "/"


def _iter_java(output_dir):
    src_root = os.path.join(output_dir, "src", "main", "java")
    walk_root = src_root if os.path.isdir(src_root) else output_dir
    for dirpath, dirs, files in os.walk(walk_root):
        dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
        for fn in files:
            if fn.endswith(".java"):
                yield os.path.join(dirpath, fn)


def _parse(path):
    with open(path, "rb") as f:
        raw = f.read()
    return _PARSER.parse(raw), raw


def _controllers_in(root, raw):
    """Yield controller dicts found in one parsed file."""
    results = []

    def walk(node):
        if node.type == "class_declaration":
            anns = _annotations(node, raw)
            names = {a[0] for a in anns}
            if names & {"Controller", "RestController"}:
                cls = next((_txt(c, raw) for c in node.children
                            if c.type == "identifier"), "?")
                is_rest = "RestController" in names
                base = ""
                for n, a in anns:
                    if n == "RequestMapping":
                        base = _first_path(a)
                eps = []
                body = next((c for c in node.children if c.type == "class_body"), None)
                if body:
                    for m in body.children:
                        if m.type != "method_declaration":
                            continue
                        manns = _annotations(m, raw)
                        mnames = {a[0] for a in manns}
                        serves_json = is_rest or "ResponseBody" in mnames
                        for n, a in manns:
                            if n in _MAPPING_HTTP:
                                eps.append({"http": _MAPPING_HTTP[n],
                                            "path": _join(base, _first_path(a)),
                                            "serves_json": serves_json})
                            elif n == "RequestMapping":
                                verbs = [v for v in ("GET", "POST", "PUT", "DELETE", "PATCH")
                                         if v in (a or "")] or ["GET"]
                                for v in verbs:
                                    eps.append({"http": v,
                                                "path": _join(base, _first_path(a)),
                                                "serves_json": serves_json})
                results.append({"class": cls,
                                "kind": "RestController" if is_rest else "Controller(view)",
                                "base_path": base, "endpoints": eps})
        for c in node.children:
            walk(c)

    walk(root)
    return results


def controllers(output_dir):
    """All controllers under the tree with type, base path, and endpoints."""
    out = []
    for path in _iter_java(output_dir):
        try:
            tree, raw = _parse(path)
        except Exception:
            continue
        for c in _controllers_in(tree.root_node, raw):
            c["file"] = path
            out.append(c)
    return out


def controller_mappings(output_dir):
    """Set of (METHOD, normalized_path) actually served as JSON across the whole tree.
    View-only endpoints are excluded, so a Thymeleaf @Controller never 'covers' a REST
    contract endpoint."""
    mapped = set()
    for c in controllers(output_dir):
        for e in c["endpoints"]:
            if e["serves_json"]:
                mapped.add((e["http"], _norm_endpoint_path(e["path"])))
    return mapped


def parse_errors(paths):
    """Subset of `paths` whose Java tree has a parse error (truncated/garbled write)."""
    bad = []
    for p in paths:
        if not p.endswith(".java"):
            continue
        try:
            tree, _ = _parse(p)
        except Exception:
            bad.append(p)
            continue
        if tree.root_node.has_error:
            bad.append(p)
    return bad


def _snake_case(name):
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower()


def _entities_in(root, raw):
    """[(class_name, table_name)] for every @Entity class declared in one parsed file."""
    results = []

    def walk(node):
        if node.type == "class_declaration":
            anns = _annotations(node, raw)
            names = {a[0] for a in anns}
            if "Entity" in names:
                cls = next((_txt(c, raw) for c in node.children
                            if c.type == "identifier"), "?")
                table = None
                for n, a in anns:
                    if n == "Table":
                        m = re.search(r'name\s*=\s*"([^"]*)"', a or "")
                        if m:
                            table = m.group(1)
                if not table:
                    table = _snake_case(cls)
                results.append((cls, table.lower()))
        for c in node.children:
            walk(c)

    walk(root)
    return results


def entity_tables(paths):
    """{table_name: path} for every @Entity class found across `paths` (.java files).
    @Table(name=...) wins; otherwise the table name is the snake_case of the class name."""
    out = {}
    for path in paths:
        if not path.endswith(".java"):
            continue
        try:
            tree, raw = _parse(path)
        except Exception:
            continue
        for _cls, table in _entities_in(tree.root_node, raw):
            out[table] = path
    return out


def schema_tables(sql_text):
    """Lowercased set of table names declared via CREATE TABLE in `sql_text`."""
    return {m.lower() for m in
            re.findall(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["`]?(\w+)', sql_text or "", re.I)}


def _public_methods_in(root, raw):
    """Public, non-getter/setter, non-@Override method names declared on top-level
    classes of one parsed file."""
    out = []

    def walk(node, in_class=False):
        if node.type == "class_declaration":
            body = next((c for c in node.children if c.type == "class_body"), None)
            if body:
                for m in body.children:
                    if m.type != "method_declaration":
                        continue
                    anns = _annotations(m, raw)
                    if "Override" in {a[0] for a in anns}:
                        continue
                    mods = next((c for c in m.children if c.type == "modifiers"), None)
                    if not mods or "public" not in _txt(mods, raw):
                        continue
                    name_node = next((c for c in m.children if c.type == "identifier"), None)
                    if not name_node:
                        continue
                    name = _txt(name_node, raw)
                    if re.match(r"^(get|set|is)[A-Z]", name):
                        continue
                    out.append(name)
        for c in node.children:
            walk(c, in_class)

    walk(root)
    return out


def unreferenced_public_methods(service_paths, output_dir):
    """[(file, method_name)] for public, non-getter/setter, non-@Override methods of
    `service_paths` classes that no `.method(`/`::method` call site exists for anywhere
    in the project's main sources (including the declaring file itself)."""
    declared = {}
    for path in service_paths:
        if not path.endswith(".java"):
            continue
        try:
            tree, raw = _parse(path)
        except Exception:
            continue
        methods = _public_methods_in(tree.root_node, raw)
        if methods:
            declared[path] = methods
    if not declared:
        return []

    all_paths = list(_iter_java(output_dir))
    sources = {}
    for p in all_paths:
        try:
            with open(p, "rb") as f:
                sources[p] = f.read().decode("utf-8", "replace")
        except Exception:
            continue

    out = []
    for path, methods in declared.items():
        for name in methods:
            pattern = re.compile(r"(?:\.|::)\s*" + re.escape(name) + r"\s*\(")
            if not any(pattern.search(text) for text in sources.values()):
                out.append((path, name))
    return out
