#!/usr/bin/env python3
"""
paper_propagate.py — STEP C of Liao et al., ICSE'25: propagating local performance
deviations to the architectural model (§III-C, Fig. 3).

This is the paper's actual contribution and the piece this repo did not have. `traces.json`
already carried the call graph it needs; nothing consumed it until now.

The paper's four sub-steps, and what each becomes here:

 1. LOCAL DEPENDENCY GRAPH (Fig. 3a). "a directed acyclic graph, with nodes representing the
    local components, and edges representing their interaction dependencies (e.g. function
    calls). Performance information is represented as node attributes." -> built from
    `profile/traces.json` (`edges`: caller/callee/calls) plus `profile/summary.csv`
    (mean_ns, n_calls as node attributes). Cycles are possible in practice (recursion), so
    traversal carries a path-visited set rather than trusting acyclicity.

 2. DEVIATED SUBGRAPH (Fig. 3a). The nodes flagged by step B, marked in the graph.

 3. MAP LOCAL -> SYSTEM LEVEL via maximum common subgraph (Fig. 3b). The paper uses the
    ISMAGS algorithm because its two graphs are genuinely different objects: a local graph
    recovered from unit tests, and a system graph recovered from load-driven runtime. Nodes
    here are fully-qualified Java operation signatures, which are globally unique labels, so
    the isomorphism SEARCH is degenerate: the maximum common subgraph of two graphs whose
    nodes carry unique comparable labels is just the subgraph induced on their shared label
    set. Implemented as signature-identity mapping and disclosed as a deviation.

    But the search being degenerate does NOT make the step a no-op, and treating it as one is
    a real bug. The mapping's other job is RESTRICTION: the system graph contains only
    production components, so anything in the local graph with no counterpart there — above
    all the test harness that drove the measurement — is dropped. Fig. 3b shows exactly this,
    mapping `Function_B3`/`Function_B4` into `Microservice_B` while `Class_B_Test` from
    Fig. 3a has no image at the system level.

    Measured consequence, on the paper's own L2 (Persistence) data: with test code retained,
    the only trace roots are `RepositoryTest.setup()` and `RepositoryTest.testRepos()`, so the
    "subsystem total" becomes the 1157 ms runtime of the whole JUnit method and the injected
    regression is diluted into invisibility — the propagated deviation came out at +5.4% for a
    10% injection and +5.1% for a 250% one, i.e. flat and slightly INVERTED with respect to
    intensity. `is_production_signature` below is the filter; callers can override it.

    WHAT THE RESTRICTION DOES AND DOES NOT FIX (corrected 2026-08-07; the earlier claim here
    that it "makes the same data yield a monotonic response" was measured on the wrong
    quantity). It restores monotonicity of the SIGNAL — the injected component's own
    contribution to the sum is +0.43% / +1.42% / +6.56% for the 10/50/250% injections. The
    REPORTED total stays non-monotonic at +7.10% / +8.87% / +8.68%, because ~10% run-to-run
    drift in the L2 candidate profiles flags 11 of 19 top-level components as deviated and
    contributes more to the sum than the injection does at Low and Medium.

    It also does not, and cannot, fix L2's magnitude. The injected `getAllEntities` is 2.65% of
    the top-level sum (against 98.62% at L1 and 96.58% at L3), so §III-C's denominator caps the
    propagated deviation at 250% x 2.65% ~ 6.6% no matter how clean the measurement. That is a
    property of the paper's aggregation rule, not of this filter: a change confined to a small
    component is structurally attenuated. See docs/PAPER-ALGORITHM-EVALUATION.md §2 Gate 1.

 4. BOTTOM-UP PROPAGATION to top-level components (Fig. 3c). "we further employ a bottom-up
    strategy to propagate performance deviations upwards through the hierarchical dependency
    layers... We begin this process from the identified components of concern and extend it
    until it reaches the top-level components of that subsystem, since these components
    encompass the performance of their invoked components."

    TOP-LEVEL COMPONENTS. `traces.json`'s `entry_points` is exactly the right notion: it is
    the set of signatures observed as a Kieker TRACE ROOT, i.e. invoked from outside the
    instrumented scope (`parse_kieker.build_call_graph` adds a signature there when its
    trace-stack depth is 0). It is NOT the same as "has no incoming edges" — a method can be
    a root in some traces and a callee in others (`getEntityClass()` is both), which is why
    root invocations are derived below rather than read off the node degree.

    PER-INVOCATION MULTIPLICITY. Edge counts in traces.json are totals over all traces, so
    the number of times v runs per invocation of u is calls(u->v) / invocations(u), with
    invocations(u) = n_calls(u) from summary.csv. Multiplicity from a root r to a deviated
    node d is the sum over all r->d paths of the product of the edge multiplicities along the
    path. A deviated node's MD contributes to r scaled by that multiplicity.

 5. SUBSYSTEM PERFORMANCE AND SERVICE DEMAND (Fig. 3d). "we calculate the overall performance
    of the subsystem by summing the top-level component performance... and calculate the
    deviation (i.e. relative difference) between the overall performance of the subsystem
    before and after the changes. With this deviation... we then update the corresponding
    performance parameters within the architectural model."
    So: D_updated = D_original * (1 + relative_deviation).

WEIGHTING (`--weight`). REVISED 2026-08-06 after re-reading §III-C; the previous note here
had this backwards and is retracted. The paper does not merely omit call-count weighting —
it raises that exact idea and rejects it, in the opening sentence of §III-C:

    "intuitively, one may utilize the local performance deviations AND THE NUMBER OF TIMES
     THE COMPONENTS ARE CALLED as a proxy to determine system performance regressions.
     However, such a naive approach may overlook the crucial impact of the system's
     complexity and dynamics, such as resource contention, queue wait times, and various
     workloads. Therefore, the architectural model is used to detect performance
     regressions..."

Call frequency is meant to be handled DOWNSTREAM, by the queueing model's workload intensity
and contention parameters — not by weighting this summation. The aggregation itself is the
plain unweighted sum of §III-C's closing sentence ("we calculate the overall performance of
the subsystem by summing up the top-level component performance").

So `weight="none"` is the paper's rule and the default. `weight="calls"` is NOT an alternative
reading of the paper: it reconstructs the naive proxy the paper argues against, and is kept
only as a labelled sensitivity variant. Do not describe it as the faithful setting.

Note that multiplicity DOES enter step C — but in sub-step 4 above, where a deviated node's MD
is scaled by how many times an ancestor calls it while propagating up the call graph. That is
independent of `--weight` and is always applied. The distinction is: multiplicity WITHIN the
propagation is the paper's; weighting the FINAL subsystem sum by root invocation count is the
proxy it disclaims.

§V-B's remark that the L2/L3 components "are often executed infrequently during system
runtime, e.g. only once, further diminishing their impact" is a post-hoc explanation of why
the MODEL predicted no regression — an emergent consequence of small service demand — not a
weighting term applied in step C.

TOP-LEVEL SCOPE (`--rest-scope`). §III-C propagates "until it reaches the top-level components
of that subsystem, since these components encompass the performance of their invoked
components", and Fig. 3b marks the subsystem boundary with «REST API» edges: the top-level
components are the operations OTHER SERVICES CALL. Our profiles come from unit tests, where
the caller is the test harness, so `build_graph`'s default rule (production node with
root_calls > 0) admits anything a test touched directly — constructors, `getEntityClass`, and
repositories with no endpoint at all. Passing `--rest-scope <workspace>` restricts the sum to
repository operations actually invoked from the persistence service's `rest/*Endpoint` classes,
which is the observable stand-in for that boundary (each endpoint method is a thin delegate to
one repository call, so their inclusive times differ only by JAX-RS overhead, which the
component profile cannot see anyway — a disclosed approximation, not an equivalence).

CROSS-CHECK. Kieker `OperationExecutionRecord` times are INCLUSIVE (a caller's measured time
already contains its callees'), so a top-level component's own step-B MD is an independent
measurement of the same quantity the propagation synthesizes. `propagated_md_ns` (the paper's
method, the thing under test) and `measured_md_ns` (inclusive, direct) are both reported per
top-level component, with their discrepancy, so step C is validated rather than assumed.

Programmatic use:
  from paper_propagate import propagate
  res = propagate(traces, stepb_report, base_summary, weight="none")
"""
import csv
import os
import re
import sys

# Class-name suffixes that mark test-harness code rather than a production component. These
# are the JUnit/Maven-Surefire conventions the subject systems follow.
_TEST_CLASS_SUFFIXES = ("Test", "Tests", "TestCase", "TestSuite", "IT", "ITCase")

_CLASS_RE = re.compile(r"([A-Za-z_$][\w$]*)\.[\w$<>]+\s*\(")


def class_of_signature(sig: str) -> str:
    """Simple class name out of a fully-qualified Kieker operation signature."""
    m = _CLASS_RE.search(sig)
    return m.group(1) if m else ""


def method_of_signature(sig: str) -> str:
    """Bare method name out of a fully-qualified Kieker operation signature.

    Kieker writes `<modifiers> <ret> <fqcn>.<method>(<args>)`, so the method name is the last
    dotted segment before the argument list. Constructors come through as `<init>`.
    """
    head = sig.split("(", 1)[0]
    return head.rsplit(".", 1)[-1] if "." in head else head


# `XRepository.REPOSITORY.method(` — how every endpoint in the persistence service reaches a
# repository (see e.g. CategoryEndpoint.getBasketLift). Matching on the (class, method) PAIR
# rather than the bare method name matters: `createEntity` exists on six repositories and only
# some of them have an endpoint.
_REST_CALL_RE = re.compile(r"\b(\w+Repository)\.REPOSITORY\.(\w+)\s*\(")

_REST_PKG = os.path.join(
    "services", "tools.descartes.teastore.persistence", "src", "main", "java",
    "tools", "descartes", "teastore", "persistence", "rest")


def rest_exposed_operations(workspace: str) -> set:
    """{(RepositoryClass, method)} reachable from the persistence REST layer.

    Static scan rather than dynamic observation: no JUnit test loads a `rest.*Endpoint`, so
    weaving that package produces zero records (confirmed live — see the K3 finding in
    docs/INSTRUMENTATION-AUDIT-2026-08-05.md). The endpoints are therefore invisible to the
    profile and the boundary has to come from the source.
    """
    rest_dir = os.path.join(workspace, _REST_PKG)
    if not os.path.isdir(rest_dir):
        raise FileNotFoundError(
            f"no persistence REST layer under {rest_dir!r} — --rest-scope needs a workspace "
            f"root (the directory containing services/), not a profile directory")
    pairs = set()
    for name in sorted(os.listdir(rest_dir)):
        if not name.endswith(".java"):
            continue
        with open(os.path.join(rest_dir, name), encoding="utf-8", errors="replace") as fh:
            pairs.update(_REST_CALL_RE.findall(fh.read()))
    return pairs


def workspace_for_profile(profile_dir: str):
    """Workspace root holding the REST layer for a profile directory, or None.

    Profiles live at `<workspace>/profile*`, so the workspace is simply the parent. Returns
    None when that parent has no persistence REST package, which is the case that must NOT be
    silently mis-scoped: the paper's own replication data (bare kieker.dat, no source tree) and
    any profile parked outside a workspace both land here and stay unscoped.
    """
    ws = os.path.dirname(os.path.abspath(profile_dir))
    return ws if os.path.isdir(os.path.join(ws, _REST_PKG)) else None


def make_rest_scope(pairs: set):
    """Predicate keeping only signatures whose (class, method) is REST-exposed."""
    def keep(sig: str) -> bool:
        return (class_of_signature(sig), method_of_signature(sig)) in pairs
    return keep


def is_production_signature(sig: str) -> bool:
    """False for test-harness components, which have no counterpart in the system graph.

    See sub-step 3 in the module docstring: dropping these is what the paper's local->system
    mapping does, and keeping them silently destroys the propagated signal.
    """
    cls = class_of_signature(sig)
    return not cls.endswith(_TEST_CLASS_SUFFIXES)


def load_summary(path: str) -> dict[str, dict]:
    """Load `profile/summary.csv` -> {sig: {n_calls, mean_ns, ...}} with numeric coercion."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rec = dict(row)
            for k in ("n_calls", "mean_ns", "stddev_ns", "min_ns", "p50_ns",
                      "p95_ns", "p99_ns", "max_ns"):
                if rec.get(k) not in (None, ""):
                    rec[k] = float(rec[k])
            rec["n_calls"] = int(rec.get("n_calls") or 0)
            out[rec["operation_signature"]] = rec
    return out


def build_graph(traces: dict, summary: dict[str, dict], production=None,
                top_level_scope=None) -> dict:
    """Build the local dependency graph, restricted to production components.

    `production` is a predicate on a signature; it defaults to `is_production_signature`.
    Pass `lambda _s: True` to keep everything (what the pre-restriction behaviour did).

    Returns {"out_edges", "in_edges", "nodes", "root_calls", "top_level", "excluded",
             "entry_points"}.

    `root_calls[sig]` is how many times the signature entered the subsystem from OUTSIDE:
    total invocations minus invocations attributable to a production caller. Clamped at 0
    because the edge counts and the summary counts are aggregated independently from the same
    traces, so rounding can make the subtraction slightly negative.

    `top_level` is every production node with `root_calls > 0` — the components actually
    invoked from outside the subsystem. This subsumes the older "use traces.json
    entry_points" rule (a Kieker trace root is by definition a call entering the instrumented
    scope) while also catching a component that is a root in some traces and a callee in
    others, and while excluding test-harness roots.
    """
    production = production or is_production_signature

    out_edges: dict[str, list[tuple[str, int]]] = {}
    in_edges: dict[str, list[tuple[str, int]]] = {}
    excluded = set()
    for e in traces.get("edges", []):
        caller, callee, calls = e["caller"], e["callee"], int(e["calls"])
        if not production(caller):
            excluded.add(caller)
            continue                      # a call FROM test code is an entry into the subsystem
        if not production(callee):
            excluded.add(callee)
            continue
        out_edges.setdefault(caller, []).append((callee, calls))
        in_edges.setdefault(callee, []).append((caller, calls))

    all_nodes = (set(summary) | set(traces.get("entry_points", []))
                 | {e["caller"] for e in traces.get("edges", [])}
                 | {e["callee"] for e in traces.get("edges", [])})
    excluded |= {n for n in all_nodes if not production(n)}
    nodes = {n for n in all_nodes if production(n)}

    root_calls = {}
    for sig in nodes:
        total = summary.get(sig, {}).get("n_calls", 0)
        incoming = sum(c for _caller, c in in_edges.get(sig, []))
        root_calls[sig] = max(0, total - incoming)

    top_level = sorted(s for s in nodes if root_calls.get(s, 0) > 0)
    # `root_calls > 0` means "entered from outside the production subsystem", which in a
    # unit-test profile means "the test harness called it" — not the paper's «REST API»
    # boundary. When a scope predicate is supplied, narrow to it and record what went, so the
    # restriction is auditable rather than a silent change of denominator.
    scope_excluded = []
    if top_level_scope is not None:
        scope_excluded = sorted(s for s in top_level if not top_level_scope(s))
        top_level = [s for s in top_level if top_level_scope(s)]
        if not top_level:
            raise ValueError(
                "top-level scope excluded every component — the subsystem sum would be empty. "
                "Check that --rest-scope points at the workspace whose profile this is.")

    return {
        "top_level_scope_excluded": scope_excluded,
        "out_edges": out_edges,
        "in_edges": in_edges,
        "entry_points": list(traces.get("entry_points", [])),
        "nodes": nodes,
        "root_calls": root_calls,
        "top_level": top_level,
        "excluded": sorted(excluded),
    }


def multiplicity(graph: dict, summary: dict[str, dict], src: str, dst: str,
                 max_depth: int = 32) -> float:
    """Expected invocations of `dst` per single invocation of `src`.

    Sum over all src->dst paths of the product of per-edge multiplicities
    calls(u->v)/invocations(u). `_path` guards against recursion cycles; `max_depth` bounds
    pathological graphs.
    """
    if src == dst:
        return 1.0

    def walk(u: str, depth: int, path: frozenset) -> float:
        if depth > max_depth:
            return 0.0
        total = 0.0
        inv_u = summary.get(u, {}).get("n_calls", 0)
        if inv_u <= 0:
            return 0.0
        for v, calls in graph["out_edges"].get(u, []):
            if v in path:
                continue                      # cycle — do not re-enter
            m_edge = calls / inv_u
            if v == dst:
                total += m_edge
            else:
                total += m_edge * walk(v, depth + 1, path | {v})
        return total

    return walk(src, 0, frozenset({src}))


def propagate(traces: dict, stepb: dict, summary: dict[str, dict],
              weight: str = "none", production=None, top_level_scope=None,
              aggregation: str = "construct") -> dict:
    """Propagate step-B deviations up to top-level components and derive the demand scaling.

    traces:      parsed profile/traces.json of the BASELINE (original) track
    stepb:       report from paper_local_deviations.identify_deviations
    summary:     parsed profile/summary.csv of the BASELINE track
    weight:      "none" (paper-literal unweighted sum) or "calls" (root-invocation weighted)
    aggregation: "construct" (sum all reachable deviated descendants scaled by multiplicity)
                 or "outermost" (prune callees when a top-level root is itself deviated,
                 preventing inclusive double-counting)

    Returns a report including `relative_deviation` and `demand_scale` (= 1 + rel_dev), which
    is what step D applies to the affected station's service demand.
    """
    if weight not in ("none", "calls"):
        raise ValueError(f"unknown weight mode: {weight!r}")
    if aggregation not in ("construct", "outermost"):
        raise ValueError(f"unknown aggregation mode: {aggregation!r}")

    graph = build_graph(traces, summary, production=production,
                        top_level_scope=top_level_scope)
    # Deviated components that are test code cannot be propagated to a system-level component
    # (they have no image in the system graph), so they are dropped here too — but reported
    # rather than dropped silently: a deviation vanishing from the result without a trace is
    # exactly the failure mode that made L2 look flat before the restriction was added.
    deviated = {c["operation_signature"]: c for c in stepb.get("deviated", [])
                if c["operation_signature"] in graph["nodes"]}
    dropped_deviated = sorted({c["operation_signature"] for c in stepb.get("deviated", [])}
                              - set(deviated))
    measured = {c["operation_signature"]: c for c in stepb.get("components", [])}

    # Top-level components of the subsystem, restricted to production code and to those we
    # have a baseline measurement for: a component with no summary row cannot contribute a
    # performance number to the subsystem sum.
    top_level = [s for s in graph["top_level"] if s in summary]

    rows = []
    for r in top_level:
        contributions = []
        prop_md = 0.0
        if aggregation == "outermost" and r in deviated:
            # Outermost dominance: r is already deviated and its inclusive timing
            # already encompasses all downstream callee slowdowns.
            dev_r = deviated[r]
            prop_md = dev_r["md_ns"]
            contributions.append({
                "operation_signature": r,
                "multiplicity": 1.0,
                "md_ns": dev_r["md_ns"],
                "contribution_ns": dev_r["md_ns"],
                "outermost_dominated": True,
            })
        else:
            for d, dev in deviated.items():
                m = multiplicity(graph, summary, r, d)
                if m <= 0.0:
                    continue
                contrib = dev["md_ns"] * m
                prop_md += contrib
                contributions.append({
                    "operation_signature": d,
                    "multiplicity": m,
                    "md_ns": dev["md_ns"],
                    "contribution_ns": contrib,
                })
        w = float(graph["root_calls"].get(r, 0)) if weight == "calls" else 1.0
        meas = measured.get(r, {})
        rows.append({
            "operation_signature": r,
            "baseline_mean_ns": summary[r].get("mean_ns", 0.0),
            "propagated_md_ns": prop_md,
            # Independent inclusive measurement of the same quantity — the cross-check.
            "measured_md_ns": meas.get("md_ns"),
            "measured_deviated": bool(meas.get("deviated")),
            "root_calls": graph["root_calls"].get(r, 0),
            "weight": w,
            "n_deviated_descendants": len(contributions),
            "contributions": sorted(contributions,
                                    key=lambda c: -abs(c["contribution_ns"])),
        })

    base_total = sum(r["baseline_mean_ns"] * r["weight"] for r in rows)
    upd_total = sum((r["baseline_mean_ns"] + r["propagated_md_ns"]) * r["weight"] for r in rows)
    rel_dev = (upd_total - base_total) / base_total if base_total > 0 else 0.0

    # Same aggregate built from the direct inclusive measurements, for the cross-check.
    meas_upd = sum((r["baseline_mean_ns"] + (r["measured_md_ns"] or 0.0)) * r["weight"]
                   for r in rows)
    meas_rel = (meas_upd - base_total) / base_total if base_total > 0 else 0.0

    return {
        "weight": weight,
        "aggregation": aggregation,
        "mapping": "signature-identity (ISMAGS degenerate: node labels are unique "
                   "fully-qualified signatures) — disclosed deviation",
        "n_top_level": len(rows),
        "n_deviated": len(deviated),
        "n_excluded_components": len(graph["excluded"]),
        "excluded_components": graph["excluded"][:20],
        # Deviated components with no counterpart in the production graph (test harness, or
        # absent from the baseline summary). They contribute nothing to the propagation.
        "dropped_deviated": dropped_deviated,
        "n_deviated_reached": len({c["operation_signature"]
                                   for r in rows for c in r["contributions"]}),
        "baseline_subsystem_ns": base_total,
        "updated_subsystem_ns": upd_total,
        "relative_deviation": rel_dev,
        "demand_scale": 1.0 + rel_dev,
        "crosscheck_measured_relative_deviation": meas_rel,
        "crosscheck_discrepancy": rel_dev - meas_rel,
        "top_level": sorted(rows, key=lambda r: -abs(r["propagated_md_ns"])),
        "unreached_deviated": sorted(
            set(deviated) - {c["operation_signature"] for r in rows for c in r["contributions"]}),
    }


def main() -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline_profile", help="senior (original) profile dir")
    ap.add_argument("stepb_json", help="report from paper_local_deviations.py --out")
    ap.add_argument("--weight", choices=["none", "calls"], default="none")
    ap.add_argument("--aggregation", choices=["construct", "outermost"], default="construct",
                    help="aggregation rule: 'construct' (sum all reachable) or 'outermost' (prune callees of deviated roots)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.stepb_json) as fh:
        stepb = json.load(fh)
    with open(os.path.join(args.baseline_profile, "traces.json")) as fh:
        traces = json.load(fh)
    summary = load_summary(os.path.join(args.baseline_profile, "summary.csv"))

    res = propagate(traces, stepb, summary, weight=args.weight, aggregation=args.aggregation)
    print(f"top-level components: {res['n_top_level']}   deviated: {res['n_deviated']} "
          f"(reached {res['n_deviated_reached']})   weight={res['weight']}")
    print(f"subsystem: {res['baseline_subsystem_ns'] / 1e6:.4f} -> "
          f"{res['updated_subsystem_ns'] / 1e6:.4f} ms   "
          f"relative deviation {res['relative_deviation'] * 100:+.3f}%   "
          f"demand_scale x{res['demand_scale']:.5f}")
    print(f"cross-check (direct inclusive measurement): "
          f"{res['crosscheck_measured_relative_deviation'] * 100:+.3f}%   "
          f"discrepancy {res['crosscheck_discrepancy'] * 100:+.3f} pp")
    if res["unreached_deviated"]:
        print(f"\ndeviated but NOT reachable from any top-level component "
              f"({len(res['unreached_deviated'])}):")
        for s in res["unreached_deviated"][:8]:
            print(f"  {s[:100]}")
    print(f"\n{'prop MD (ms)':>13} {'meas MD (ms)':>13} {'roots':>8} {'desc':>5}  signature")
    for r in res["top_level"][:12]:
        mm = r["measured_md_ns"]
        print(f"{r['propagated_md_ns'] / 1e6:>13.5f} "
              f"{(mm / 1e6 if mm is not None else float('nan')):>13.5f} "
              f"{r['root_calls']:>8} {r['n_deviated_descendants']:>5}  "
              f"{r['operation_signature'][:70]}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
