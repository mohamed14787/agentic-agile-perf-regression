"""No-LLM suite for the Liao et al. ICSE'25 algorithm implementation (2026-07-27).

Run:  .venv/bin/python -m agile_team_cline.tests.test_paper_algorithm

A1  qn_solver open network — M/M/1 reduces to R = D/(1-rho); M/M/m matches a hand-computed
    Erlang-C result; saturation is reported, not raised; Little's law holds.
A2  qn_solver closed network — exact MVA against hand-computed N=1 and N=2 values, and the
    N->inf throughput asymptote X -> 1/max(D).
A3  paper_propagate multiplicity — per-invocation call multiplicity along a chain, summed over
    multiple paths, and terminating on a cycle.
A4  paper_propagate root_calls — trace-root invocations derived as n_calls minus incoming
    edge calls (entry_points is a "was a trace root" set, NOT "has no incoming edges").
A5  paper_propagate propagate() — relative deviation and demand_scale on a hand-computed
    graph, unweighted (paper-literal) and call-weighted.
A6  paper_local_deviations — the paper's step-B rule fires on a clearly shifted distribution
    and does NOT fire on identical distributions; MD sign convention (positive = slower).
A7  paper_detect PropagationKernel — the fast bootstrap kernel agrees with the full
    propagate() on the same inputs (the kernel is an optimisation, not a second algorithm).
A10 paper_propagate REST scope — top-level components are the subsystem's «REST API» boundary
    (Fig. 3b), not every component a unit test happened to call directly.
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "profiling"))
import qn_solver
import paper_propagate
import paper_local_deviations
import paper_detect

_fails = []


def check(name, cond):
    """Record a named assertion, and FAIL the enclosing test when it does not hold.

    The raise is load-bearing — without it only `main()` ever reads `_fails`, so pytest
    (which calls test_* directly) reported green over real failures. See test_profiling."""
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        _fails.append(name)
        raise AssertionError(name)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _summary(entries):
    """{sig: n_calls or (n_calls, mean_ns)} -> load_summary-shaped dict."""
    out = {}
    for sig, val in entries.items():
        n, mean = val if isinstance(val, tuple) else (val, 1000.0)
        out[sig] = {"operation_signature": sig, "n_calls": n, "mean_ns": float(mean)}
    return out


# --------------------------------------------------------------------------- A1
def test_a1_open_network():
    # M/M/1: D=0.1s, lambda=5/s -> rho=0.5, R = 0.1/0.5 = 0.2s
    res = qn_solver.solve_open(5.0, {"s": 0.1})
    check("A1 M/M/1 utilization = lambda*D",
          close(res["utilization"]["s"], 0.5))
    check("A1 M/M/1 R = D/(1-rho)",
          close(res["residence_time"]["s"], 0.2))
    check("A1 open network X = lambda", close(res["throughput"], 5.0))
    check("A1 Little's law L = lambda*R", close(res["queue_length"]["s"], 5.0 * 0.2))

    # Two stations sum their residence times.
    two = qn_solver.solve_open(5.0, {"a": 0.1, "b": 0.05})
    check("A1 response time sums over stations",
          close(two["response_time_s"], 0.1 / 0.5 + 0.05 / 0.75))

    # Erlang-C at m=1 reduces algebraically to rho.
    check("A1 erlang_c(a, 1) == a", close(qn_solver.erlang_c(0.4, 1), 0.4))

    # M/M/2 hand-computed: D=0.1, lambda=10 -> a=1.0, m=2.
    #   B(1)=1/(1+1)=0.5 ; B(2)=0.5/(2+0.5)=0.2 ; rho=0.5
    #   C = 0.2/(1-0.5*0.8) = 0.2/0.6 = 1/3
    #   Wq = C/(m/D - lambda) = (1/3)/(20-10) = 1/30 ; R = 1/30 + 0.1 = 0.133333...
    mm2 = qn_solver.solve_open(10.0, {"s": 0.1}, servers={"s": 2})
    check("A1 M/M/2 Erlang-C waiting time",
          close(mm2["residence_time"]["s"], 1.0 / 30.0 + 0.1, 1e-9))
    check("A1 M/M/2 per-server utilization", close(mm2["utilization"]["s"], 0.5))

    # Saturation is reported, not raised.
    sat = qn_solver.solve_open(20.0, {"s": 0.1})
    check("A1 saturated station reported", sat["saturated"] == ["s"])
    check("A1 saturated R is inf", math.isinf(sat["response_time_s"]))

    # Erlang-B is stable at large m (factorial form would overflow).
    check("A1 erlang_b stable at m=500", 0.0 <= qn_solver.erlang_b(100.0, 500) < 1e-12)


# --------------------------------------------------------------------------- A2
def test_a2_closed_mva():
    D = {"a": 0.1, "b": 0.2}
    # N=1, Z=0: R = sum(D) = 0.3 ; X = 1/0.3
    r1 = qn_solver.solve_closed_mva(1, 0.0, D)
    check("A2 MVA N=1 response time", close(r1["response_time_s"], 0.3))
    check("A2 MVA N=1 throughput", close(r1["throughput"], 1.0 / 0.3))
    check("A2 MVA N=1 utilization U_k = X*D_k",
          close(r1["utilization"]["b"], (1.0 / 0.3) * 0.2))

    # N=2 by hand: Q from n=1 is (0.3333.., 0.6666..)
    #   R_a = 0.1*(1+1/3)=0.133333 ; R_b = 0.2*(1+2/3)=0.333333 ; R=0.466666
    r2 = qn_solver.solve_closed_mva(2, 0.0, D)
    check("A2 MVA N=2 response time", close(r2["response_time_s"], 0.1 * 4 / 3 + 0.2 * 5 / 3))
    check("A2 MVA N=2 throughput", close(r2["throughput"], 2.0 / (0.1 * 4 / 3 + 0.2 * 5 / 3)))

    # Throughput asymptote: X -> 1/max(D) = 5 as N grows.
    big = qn_solver.solve_closed_mva(400, 0.0, D)
    check("A2 MVA throughput approaches 1/max(D)", abs(big["throughput"] - 5.0) < 0.05)
    check("A2 MVA bottleneck utilization approaches 1",
          0.98 < big["utilization"]["b"] <= 1.0)

    # Think time reduces throughput.
    z = qn_solver.solve_closed_mva(2, 1.0, D)
    check("A2 think time lowers throughput", z["throughput"] < r2["throughput"])


# --------------------------------------------------------------------------- A3
def test_a3_multiplicity():
    # Chain r -> a -> b. r ran 10x and called a 10x  => m(r->a) = 1
    #                    a ran 10x and called b 30x  => m(a->b) = 3
    #                    m(r, b) = 1 * 3 = 3
    traces = {"edges": [{"caller": "r", "callee": "a", "calls": 10},
                        {"caller": "a", "callee": "b", "calls": 30}],
              "entry_points": ["r"]}
    summ = _summary({"r": 10, "a": 10, "b": 30})
    g = paper_propagate.build_graph(traces, summ)
    check("A3 multiplicity of a direct edge",
          close(paper_propagate.multiplicity(g, summ, "r", "a"), 1.0))
    check("A3 multiplicity composes along a chain",
          close(paper_propagate.multiplicity(g, summ, "r", "b"), 3.0))
    check("A3 multiplicity to self is 1",
          close(paper_propagate.multiplicity(g, summ, "r", "r"), 1.0))
    check("A3 multiplicity of an unreachable node is 0",
          close(paper_propagate.multiplicity(g, summ, "b", "r"), 0.0))

    # Two disjoint paths r->a->d and r->b->d sum: (10/10)*(20/10) + (10/10)*(30/10) = 5
    t2 = {"edges": [{"caller": "r", "callee": "a", "calls": 10},
                    {"caller": "r", "callee": "b", "calls": 10},
                    {"caller": "a", "callee": "d", "calls": 20},
                    {"caller": "b", "callee": "d", "calls": 30}],
          "entry_points": ["r"]}
    s2 = _summary({"r": 10, "a": 10, "b": 10, "d": 50})
    g2 = paper_propagate.build_graph(t2, s2)
    check("A3 multiplicity sums over multiple paths",
          close(paper_propagate.multiplicity(g2, s2, "r", "d"), 5.0))

    # Cycle must terminate (recursion is possible in real traces).
    t3 = {"edges": [{"caller": "r", "callee": "a", "calls": 10},
                    {"caller": "a", "callee": "r", "calls": 10}],
          "entry_points": ["r"]}
    s3 = _summary({"r": 10, "a": 10})
    g3 = paper_propagate.build_graph(t3, s3)
    try:
        m = paper_propagate.multiplicity(g3, s3, "r", "a")
        check("A3 cycle terminates without recursion error", close(m, 1.0))
    except RecursionError:
        check("A3 cycle terminates without recursion error", False)


# --------------------------------------------------------------------------- A4
def test_a4_root_calls():
    # `x` ran 100 times; 70 of those were calls from `r`; so 30 were trace roots.
    traces = {"edges": [{"caller": "r", "callee": "x", "calls": 70}],
              "entry_points": ["r", "x"]}
    summ = _summary({"r": 40, "x": 100})
    g = paper_propagate.build_graph(traces, summ)
    check("A4 root_calls = n_calls - incoming edge calls", g["root_calls"]["x"] == 30)
    check("A4 root_calls of a pure root is its n_calls", g["root_calls"]["r"] == 40)
    check("A4 a signature can be both root and callee",
          "x" in g["entry_points"] and "x" in g["in_edges"])

    # Clamped at 0 rather than going negative when the two aggregates disagree.
    g2 = paper_propagate.build_graph(
        {"edges": [{"caller": "r", "callee": "x", "calls": 900}], "entry_points": ["x"]},
        _summary({"r": 10, "x": 100}))
    check("A4 root_calls clamped at zero", g2["root_calls"]["x"] == 0)


# --------------------------------------------------------------------------- A5
def test_a5_propagate():
    # Top-level r (mean 1000ns) and s (mean 3000ns). Deviated node b sits under r with
    # multiplicity 3 and has MD = +100ns  =>  propagated MD at r = +300ns.
    traces = {"edges": [{"caller": "r", "callee": "a", "calls": 10},
                        {"caller": "a", "callee": "b", "calls": 30}],
              "entry_points": ["r", "s"]}
    summ = _summary({"r": (10, 1000.0), "a": (10, 500.0), "b": (30, 100.0),
                     "s": (5, 3000.0)})
    stepb = {"deviated": [{"operation_signature": "b", "md_ns": 100.0}],
             "components": [{"operation_signature": "b", "md_ns": 100.0, "deviated": True}]}

    res = paper_propagate.propagate(traces, stepb, summ, weight="none")
    # unweighted: base = 1000 + 3000 = 4000 ; updated = (1000+300) + 3000 = 4300
    check("A5 unweighted baseline subsystem total", close(res["baseline_subsystem_ns"], 4000.0))
    check("A5 unweighted updated subsystem total", close(res["updated_subsystem_ns"], 4300.0))
    check("A5 relative deviation", close(res["relative_deviation"], 300.0 / 4000.0))
    check("A5 demand_scale = 1 + rel_dev", close(res["demand_scale"], 1.075))
    r_row = next(x for x in res["top_level"] if x["operation_signature"] == "r")
    check("A5 propagated MD scaled by multiplicity", close(r_row["propagated_md_ns"], 300.0))
    check("A5 all deviated nodes reached", res["n_deviated_reached"] == 1)

    # call-weighted: weights are root_calls -> r:10, s:5
    # base = 1000*10 + 3000*5 = 25000 ; updated = 1300*10 + 15000 = 28000
    resw = paper_propagate.propagate(traces, stepb, summ, weight="calls")
    check("A5 call-weighted baseline total", close(resw["baseline_subsystem_ns"], 25000.0))
    check("A5 call-weighted relative deviation",
          close(resw["relative_deviation"], 3000.0 / 25000.0))
    check("A5 weighting changes the answer",
          not close(resw["relative_deviation"], res["relative_deviation"]))

    # A deviated node with no counterpart in the production graph is REPORTED, not silently
    # dropped — a deviation vanishing without a trace is the failure mode that made the
    # paper's L2 data look flat before the test-harness restriction existed.
    stepb2 = {"deviated": [{"operation_signature": "orphan", "md_ns": 5000.0}],
              "components": []}
    res2 = paper_propagate.propagate(traces, stepb2, summ, weight="none")
    check("A5 deviated node outside the production graph is reported",
          res2["dropped_deviated"] == ["orphan"] and close(res2["relative_deviation"], 0.0))


# --------------------------------------------------------------------------- A8
def test_a8_test_harness_restriction():
    """The local->system mapping must drop test code (paper Fig. 3b).

    Regression test for the bug the paper's own L2 data exposed: with the JUnit method left in
    the graph it becomes the only trace root, so the subsystem total is the whole test's
    runtime and an injected regression is diluted to nothing.
    """
    check("A8 test classes are not production",
          not paper_propagate.is_production_signature(
              "public void tools.descartes.teastore.persistence.RepositoryTest.testRepos()"))
    check("A8 production classes are production",
          paper_propagate.is_production_signature(
              "public long tools.descartes.teastore.persistence.domain.ProductRepository"
              ".createEntity(tools.descartes.teastore.entities.Product)"))
    for suffix in ("Tests", "TestCase", "IT"):
        check(f"A8 {suffix} suffix treated as test code",
              not paper_propagate.is_production_signature(f"public void com.x.Foo{suffix}.bar()"))
    check("A8 class name parsed out of a signature",
          paper_propagate.class_of_signature(
              "public java.util.List com.x.Repo.getAll(long,int)") == "Repo")

    # T (test) calls svc; svc calls helper. Only svc/helper are production, so svc — not T —
    # is the top-level component and the subsystem total excludes T's inflated runtime.
    traces = {"edges": [{"caller": "public void com.x.FooTest.t()",
                         "callee": "public void com.x.Svc.handle()", "calls": 10},
                        {"caller": "public void com.x.Svc.handle()",
                         "callee": "public void com.x.Helper.work()", "calls": 20}],
              "entry_points": ["public void com.x.FooTest.t()"]}
    summ = _summary({"public void com.x.FooTest.t()": (10, 1_000_000.0),
                     "public void com.x.Svc.handle()": (10, 1000.0),
                     "public void com.x.Helper.work()": (20, 400.0)})
    g = paper_propagate.build_graph(traces, summ)
    check("A8 test method excluded from the graph",
          "public void com.x.FooTest.t()" in g["excluded"])
    check("A8 top-level is the production entry, not the test",
          g["top_level"] == ["public void com.x.Svc.handle()"])

    # Helper slows by 100ns with multiplicity 2 -> +200ns on a 1000ns subsystem = +20%.
    # Had the test method stayed in, the base would be 1_001_000ns and the answer ~0.02%.
    stepb = {"deviated": [{"operation_signature": "public void com.x.Helper.work()",
                           "md_ns": 100.0}], "components": []}
    res = paper_propagate.propagate(traces, stepb, summ, weight="none")
    check("A8 subsystem total excludes the test harness",
          close(res["baseline_subsystem_ns"], 1000.0))
    check("A8 propagated deviation is not diluted by the test runtime",
          close(res["relative_deviation"], 0.2))


# --------------------------------------------------------------------------- A10
def test_a10_rest_scope():
    """Top-level components are the subsystem's REST boundary, not "whatever a test called".

    §III-C propagates up to "the top-level components of that subsystem" and Fig. 3b draws that
    boundary with «REST API» edges. A8 already removes the test METHOD from the graph, but the
    components it called directly still all count as top-level — including constructors and
    internal helpers that no endpoint exposes. This narrows the sum to the operations an
    endpoint actually invokes.
    """
    exposed = "public java.util.List com.x.Svc.handle()"
    internal = "public void com.x.Svc.<init>()"
    helper = "public void com.x.Helper.work()"
    traces = {"edges": [{"caller": "public void com.x.FooTest.t()", "callee": exposed,
                         "calls": 10},
                        {"caller": "public void com.x.FooTest.t()", "callee": internal,
                         "calls": 1},
                        {"caller": exposed, "callee": helper, "calls": 20}],
              "entry_points": ["public void com.x.FooTest.t()"]}
    summ = _summary({exposed: (10, 1000.0), internal: (1, 4000.0), helper: (20, 400.0)})

    check("A10 method name parsed out of a signature",
          paper_propagate.method_of_signature(exposed) == "handle")
    check("A10 constructor method name is <init>",
          paper_propagate.method_of_signature(internal) == "<init>")

    # Without a scope both Svc.handle and Svc.<init> are top-level, and the constructor's
    # 4000ns — pure test-fixture cost with no endpoint behind it — is 80% of the denominator.
    g_open = paper_propagate.build_graph(traces, summ)
    check("A10 unscoped top-level admits the constructor",
          set(g_open["top_level"]) == {exposed, internal})

    scope = paper_propagate.make_rest_scope({("Svc", "handle")})
    g_rest = paper_propagate.build_graph(traces, summ, top_level_scope=scope)
    check("A10 scoped top-level is the exposed operation only",
          g_rest["top_level"] == [exposed])
    check("A10 what the scope removed is recorded, not silently dropped",
          g_rest["top_level_scope_excluded"] == [internal])

    # helper +100ns at multiplicity 2 = +200ns. Denominator 1000 (scoped) vs 5000 (unscoped).
    stepb = {"deviated": [{"operation_signature": helper, "md_ns": 100.0}], "components": []}
    open_res = paper_propagate.propagate(traces, stepb, summ, weight="none")
    rest_res = paper_propagate.propagate(traces, stepb, summ, weight="none",
                                         top_level_scope=scope)
    check("A10 unscoped deviation is diluted by the unexposed constructor",
          close(open_res["relative_deviation"], 0.04))
    check("A10 scoped deviation is measured against the exposed subsystem",
          close(rest_res["relative_deviation"], 0.2))

    # A scope that matches nothing would make the subsystem total zero, and a zero denominator
    # would surface as a division error deep in step D rather than as a bad --rest-scope path.
    raised = False
    try:
        paper_propagate.build_graph(traces, summ,
                                    top_level_scope=paper_propagate.make_rest_scope(set()))
    except ValueError:
        raised = True
    check("A10 an empty scope is rejected, not silently zero", raised)

    # The static scan reads (Repository, method) PAIRS: createEntity exists on many
    # repositories and only some of them are behind an endpoint.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rest_dir = os.path.join(tmp, paper_propagate._REST_PKG)
        os.makedirs(rest_dir)
        with open(os.path.join(rest_dir, "ProductEndpoint.java"), "w") as fh:
            fh.write("@GET @Path(\"x\")\n"
                     "public List<X> x() {\n"
                     "  return ProductRepository.REPOSITORY.getAlsoBought(id);\n"
                     "}\n"
                     "public long add(P p) {\n"
                     "  return StockRepository.REPOSITORY.createEntity(p);\n"
                     "}\n")
        pairs = paper_propagate.rest_exposed_operations(tmp)
    check("A10 scan finds the endpoint's repository calls",
          pairs == {("ProductRepository", "getAlsoBought"), ("StockRepository", "createEntity")})
    check("A10 scan matches on the (class, method) pair",
          ("ProductRepository", "createEntity") not in pairs)


# --------------------------------------------------------------------------- A6
def test_a6_step_b_rule():
    # Identical distributions: no significance, no effect -> not deviated.
    same = list(range(100, 200))
    rep = paper_local_deviations.identify_deviations({"x": same}, {"x": list(same)},
                                                    normalize="none")
    c = rep["components"][0]
    check("A6 identical distributions are not deviated", c["deviated"] is False)
    check("A6 identical distributions have MD 0", close(c["md_ns"], 0.0))

    # Cleanly separated distributions: significant AND large effect -> deviated, MD > 0.
    lo = [100 + i for i in range(200)]
    hi = [10000 + i for i in range(200)]
    rep2 = paper_local_deviations.identify_deviations({"x": lo}, {"x": hi}, normalize="none")
    c2 = rep2["components"][0]
    check("A6 shifted distribution is deviated", c2["deviated"] is True)
    check("A6 p-value below 0.05", c2["p_value"] <= 0.05)
    check("A6 effect size not negligible", c2["magnitude"] != "negligible")
    check("A6 MD positive when candidate is slower", c2["md_ns"] > 0)

    # Direction: candidate faster -> negative MD, still a "deviation" (paper's test is
    # two-sided and its Table I carries negative MPD rows).
    rep3 = paper_local_deviations.identify_deviations({"x": hi}, {"x": lo}, normalize="none")
    c3 = rep3["components"][0]
    check("A6 MD negative when candidate is faster", c3["md_ns"] < 0)
    check("A6 faster component still flagged as deviated", c3["deviated"] is True)

    # A large effect on too few samples cannot reach significance -> not deviated.
    rep4 = paper_local_deviations.identify_deviations({"x": [100, 101, 102]},
                                                     {"x": [900, 901, 902]},
                                                     normalize="none")
    check("A6 large effect on tiny n is not significant",
          rep4["components"][0]["deviated"] is False)

    # NEW / REMOVED components are tracked separately from the compared set.
    rep5 = paper_local_deviations.identify_deviations({"a": same}, {"b": same},
                                                     normalize="none")
    check("A6 new and removed components tracked",
          rep5["new_components"] == ["b"] and rep5["removed_components"] == ["a"]
          and rep5["n_compared"] == 0)


# --------------------------------------------------------------------------- A7
def test_a7_kernel_matches_propagate():
    """The bootstrap kernel must be an optimisation of propagate(), not a reimplementation."""
    traces = {"edges": [{"caller": "r", "callee": "a", "calls": 10},
                        {"caller": "a", "callee": "b", "calls": 30}],
              "entry_points": ["r", "s"]}
    summ = _summary({"r": (10, 1000.0), "a": (10, 500.0), "b": (30, 100.0),
                     "s": (5, 3000.0)})
    stepb = {"deviated": [{"operation_signature": "b", "md_ns": 100.0}], "components": []}

    for weight in ("none", "calls"):
        full = paper_propagate.propagate(traces, stepb, summ, weight=weight)
        kern = paper_detect.PropagationKernel(traces, ["b"], summ, weight=weight)
        base_means = {s: r["mean_ns"] for s, r in summ.items()}
        cand_means = dict(base_means)
        cand_means["b"] = base_means["b"] + 100.0     # same MD as stepb
        bt, ut = kern.totals(base_means, cand_means)
        check(f"A7 kernel baseline total matches propagate (weight={weight})",
              close(bt, full["baseline_subsystem_ns"]))
        check(f"A7 kernel updated total matches propagate (weight={weight})",
              close(ut, full["updated_subsystem_ns"]))
        rel = (ut - bt) / bt
        check(f"A7 kernel relative deviation matches propagate (weight={weight})",
              close(rel, full["relative_deviation"]))


# --------------------------------------------------------------------------- A9
def test_a9_response_time_sampling():
    """Sampled response times must match the analytic solve and carry real spread.

    Regression test for the effect-size saturation the paper's data exposed: distributions
    that are too tight drive Cliff's Delta to 1.0 for arbitrarily small shifts, classifying
    every version as a regression.
    """
    import random
    import statistics
    model = {"form": "open", "arrival_rate": 50.0, "servers": {"a": 2, "b": 2},
             "scaled_station": "a"}
    demands = {"a": 0.004, "b": 0.009}
    rng = random.Random(7)
    xs = qn_solver.sample_response_times(model, demands, 40000, rng)
    analytic = qn_solver.solve(model, demands)["response_time_s"]
    check("A9 sampled mean matches the analytic response time",
          abs(statistics.mean(xs) - analytic) / analytic < 0.05)
    # An M/M/m sojourn time is roughly exponential, so the spread is the same order as the
    # mean. A near-zero stdev would mean the distribution cannot discriminate effect sizes.
    check("A9 sampled distribution has realistic spread",
          statistics.pstdev(xs) > 0.4 * statistics.mean(xs))

    # A tiny demand shift must NOT register as a large effect.
    from nonparametric_stats import cliffs_delta
    small = qn_solver.sample_response_times(
        model, {"a": 0.004 * 1.002, "b": 0.009}, 40000, random.Random(8))
    _d, mag = cliffs_delta(xs, small)
    check("A9 a 0.2% demand shift is a negligible effect", mag == "negligible")

    # A large one must.
    big = qn_solver.sample_response_times(
        model, {"a": 0.004 * 3.0, "b": 0.009}, 40000, random.Random(9))
    d2, mag2 = cliffs_delta(xs, big)
    check("A9 a 3x demand shift is a non-negligible effect", mag2 != "negligible" and d2 > 0)

    # Saturation is refused rather than silently returning nonsense.
    try:
        qn_solver.sample_response_times(model, {"a": 0.5, "b": 0.009}, 10, random.Random(1))
        check("A9 saturated model raises", False)
    except qn_solver.SaturatedError:
        check("A9 saturated model raises", True)


# --------------------------------------------------------------------------- A10
def test_a10_endpoint_scoped():
    """Endpoint/class-scoped response time must match BCMP analytic response time and isolate classes."""
    import random
    import statistics
    model = {
        "form": "open",
        "arrival_rate": 50.0,
        "servers": {"WebUI": 2, "Auth": 2},
        "demands": {"WebUI": 0.008, "Auth": 0.007},
        "request_mix": {"login": 0.4, "browse": 0.6},
        "per_class_service_s": {
            "WebUI": {"login": 0.005, "browse": 0.010},
            "Auth": {"login": 0.0175, "browse": 0.0},
        }
    }
    # Demands consistency:
    # D_WebUI = 0.4*0.005 + 0.6*0.010 = 0.008
    # D_Auth = 0.4*0.0175 + 0.6*0.0 = 0.007
    demands = dict(model["demands"])

    res_login = qn_solver.class_response_time(model, demands, "login")
    res_browse = qn_solver.class_response_time(model, demands, "browse")

    # Browse does not visit Auth, so its response time is just WebUI residence time
    check("A10 browse response time excludes Auth service",
          "Auth" not in res_browse["service_times"])
    check("A10 login response time includes Auth service",
          "Auth" in res_login["service_times"])

    # Sample specifically for login
    rng = random.Random(42)
    xs_login = qn_solver.sample_response_times(model, demands, 20000, rng, target_class="login")
    mean_samp = statistics.mean(xs_login)
    check("A10 target_class sampled mean matches class_response_time",
          abs(mean_samp - res_login["response_time_s"]) / res_login["response_time_s"] < 0.05)

    # Shifting Auth demand affects login significantly, while browse is unaffected
    upd_demands = dict(demands)
    upd_demands["Auth"] = 0.007 * 1.5  # +50% on Auth
    xs_login_upd = qn_solver.sample_response_times(model, upd_demands, 20000, random.Random(43), target_class="login")
    xs_browse_upd = qn_solver.sample_response_times(model, upd_demands, 20000, random.Random(44), target_class="browse")
    xs_browse = qn_solver.sample_response_times(model, demands, 20000, random.Random(45), target_class="browse")

    from nonparametric_stats import cliffs_delta
    _d_b, mag_browse = cliffs_delta(xs_browse, xs_browse_upd)
    d_l, mag_login = cliffs_delta(xs_login, xs_login_upd)
    check("A10 endpoint-scoped shifts login significantly", mag_login != "negligible" and d_l > 0)
    check("A10 browse unaffected by Auth shift", mag_browse == "negligible")


def main():
    test_a1_open_network()
    test_a2_closed_mva()
    test_a3_multiplicity()
    test_a4_root_calls()
    test_a5_propagate()
    test_a6_step_b_rule()
    test_a7_kernel_matches_propagate()
    test_a8_test_harness_restriction()
    test_a9_response_time_sampling()
    test_a10_endpoint_scoped()
    print()
    print("RESULT:", "ALL PASS" if not _fails else f"{len(_fails)} FAIL: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
