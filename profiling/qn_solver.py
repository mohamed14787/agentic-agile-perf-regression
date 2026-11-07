#!/usr/bin/env python3
"""
qn_solver.py — the architectural-model solver for step D of Liao et al., ICSE'25.

The paper builds Queueing Petri Nets and solves them with SimQPN. We solve queueing
NETWORKS analytically instead. That substitution is licensed by the paper itself:

    "our study does not rely on specific types of architectural models, and the construction
     of architectural models is also not the primary focus of our study"   (§IV-E)

and queueing networks are the first modelling formalism it lists as valid (§II, refs 36/38/
49/63). The practical reason is that QPME/SimQPN is an abandoned Java tool (last release
~2012); depending on it would add a large integration risk for no gain in the quantity we
actually need, which is a predicted end-to-end response time per model configuration.

TWO SOLVERS, because the model form has to match the workload form:

  * `solve_open` — OPEN network, fixed arrival rate. This is the primary path: the paper's
    own Fig. 1b parameterises the workload place with a THROUGHPUT (320 req/s), i.e. an open
    workload, and its TeaStore evaluation drives a fixed 50 req/s (§IV-C). Stations are
    M/M/m queues.

  * `solve_closed_mva` — CLOSED network, fixed population N with think time Z, solved by
    exact Mean Value Analysis (Reiser & Lavenberg). Used when the fetched model turns out to
    be closed (a finite user population rather than an arrival rate).

Both take service demands D_k in SECONDS per request and return response time in seconds.
Service demand is the total service requirement of one request at station k (Denning &
Buzen's operational definition), so D_k already folds in visit counts — which is what makes
the propagated per-subsystem scaling in paper_propagate.py directly applicable to it.

M/M/m waiting time uses Erlang-C, computed from the numerically stable Erlang-B recursion
rather than the factorial form (which overflows for m beyond ~170 and loses precision well
before that). For m=1 this reduces algebraically to the textbook M/M/1 result R = D/(1-rho),
which the unit tests assert.

Pure stdlib. No scipy.
"""

SATURATION_EPS = 1e-9


class SaturatedError(ValueError):
    """Raised when offered load meets or exceeds capacity, so no steady state exists."""


def erlang_b(a: float, m: int) -> float:
    """Erlang-B blocking probability for offered load `a` erlangs and `m` servers.

    Recursion B(0)=1, B(k) = a*B(k-1) / (k + a*B(k-1)) — stable for large m, unlike the
    factorial closed form.
    """
    b = 1.0
    for k in range(1, m + 1):
        b = (a * b) / (k + a * b)
    return b


def erlang_c(a: float, m: int) -> float:
    """Erlang-C probability that an arriving request has to queue (M/M/m)."""
    rho = a / m
    if rho >= 1.0 - SATURATION_EPS:
        raise SaturatedError(f"per-server utilization {rho:.6f} >= 1")
    b = erlang_b(a, m)
    return b / (1.0 - rho * (1.0 - b))


def solve_open(arrival_rate: float, demands: dict[str, float],
               servers: dict[str, int] = None) -> dict:
    """Solve an open queueing network of M/M/m stations.

    arrival_rate: system arrival rate lambda (requests/second)
    demands:      {station: service demand D_k in seconds/request}
    servers:      {station: m_k}, default 1 server per station

    Returns {response_time_s, throughput, utilization: {k: U_k}, residence_time: {k: R_k},
             queue_length: {k: L_k}, saturated: [k, ...]}.
    A station at or beyond capacity is reported in `saturated` with R_k = inf rather than
    raising, so a caller sweeping load levels can see WHERE the model breaks down instead of
    losing the whole solve.
    """
    servers = servers or {}
    util, resid, qlen, saturated = {}, {}, {}, []
    total_r = 0.0
    for k, d in demands.items():
        m = max(1, int(servers.get(k, 1)))
        a = arrival_rate * d                      # offered load in erlangs
        u = a / m                                 # per-server utilization
        util[k] = u
        if u >= 1.0 - SATURATION_EPS:
            saturated.append(k)
            resid[k] = float("inf")
            qlen[k] = float("inf")
            total_r = float("inf")
            continue
        if m == 1:
            r_k = d / (1.0 - u)                   # M/M/1
        else:
            c = erlang_c(a, m)
            wq = c / (m / d - arrival_rate)       # m*mu - lambda, with mu = 1/d
            r_k = wq + d
        resid[k] = r_k
        qlen[k] = arrival_rate * r_k              # Little's law
        if total_r != float("inf"):
            total_r += r_k
    return {
        "model": "open",
        "arrival_rate": arrival_rate,
        "response_time_s": total_r,
        "throughput": arrival_rate,               # open network: X = lambda in steady state
        "utilization": util,
        "residence_time": resid,
        "queue_length": qlen,
        "saturated": saturated,
    }


def solve_closed_mva(n_users: int, think_time: float, demands: dict[str, float]) -> dict:
    """Exact Mean Value Analysis for a single-class closed network (Reiser & Lavenberg).

    n_users:    population N
    think_time: Z in seconds (delay station)
    demands:    {station: D_k in seconds}

    Iterates n = 1..N with R_k = D_k*(1 + Q_k), X = n/(Z + sum R_k), Q_k = X*R_k.
    Returns {response_time_s, throughput, utilization, residence_time, queue_length}.
    """
    if n_users < 1:
        raise ValueError("n_users must be >= 1")
    stations = list(demands)
    qlen = {k: 0.0 for k in stations}
    x = 0.0
    resid = {k: 0.0 for k in stations}
    for n in range(1, n_users + 1):
        resid = {k: demands[k] * (1.0 + qlen[k]) for k in stations}
        r_total = sum(resid.values())
        x = n / (think_time + r_total)
        qlen = {k: x * resid[k] for k in stations}
    return {
        "model": "closed",
        "n_users": n_users,
        "think_time": think_time,
        "response_time_s": sum(resid.values()),
        "throughput": x,
        "utilization": {k: x * demands[k] for k in stations},
        "residence_time": resid,
        "queue_length": qlen,
        "saturated": [k for k in stations if x * demands[k] >= 1.0 - SATURATION_EPS],
    }


def sample_response_times(model: dict, demands: dict[str, float], n: int,
                          rng, class_aware: bool = False,
                          visits: dict[str, float] = None,
                          target_class: str = None,
                          per_class_service: dict = None) -> list[float]:
    """Draw `n` per-request end-to-end response times from an open M/M/m network.

    WHY THIS EXISTS. Step D compares the original and updated models with a rank test and an
    effect size. That needs two SAMPLES of predicted response time. The paper gets them from
    SimQPN, whose output is per-request response times carrying the queueing system's own
    variability. Solving analytically yields one number per model, and substituting a
    bootstrap-over-measurements distribution for it is NOT equivalent: those distributions are
    orders of magnitude tighter than real response times, so Cliff's Delta saturates at 1.0
    for any non-zero shift and every version is classified as a regression. Validating against
    the paper's own data caught exactly that — a predicted +0.0142 ms shift was reported as a
    "large" effect.

    The M/M/m sojourn time is exact and needs no simulator: waiting time is 0 with probability
    1 - C (Erlang-C) and Exponential(m*mu - lambda) otherwise, and service time is
    Exponential(mu), independent. Summing over the stations a request visits gives its
    end-to-end response time.

    VISIT-COUNT MULTIPLICITY (V_k > 1, opt-in via `visits` or model["visits"]).
    When a request visits station k multiple times (V_k visits on average), service demand
    D_k = V_k * S_k is the sum of V_k visits of mean duration S_k = D_k / V_k.
    Drawing a single Exponential(1/D_k) preserves the mean but exaggerates variance by V_k
    (variance D_k^2 instead of D_k^2 / V_k). For V_k > 1, service time is sampled from an
    Erlang-V_k distribution (Gamma with shape V_k, scale D_k / V_k), which preserves mean D_k
    while scaling variance to D_k^2 / V_k. Waiting time is drawn across V_k independent visits,
    each with queueing probability C and queued delay Exponential(V_k * (m*mu - lambda)),
    preserving expected wait C / (m*mu - lambda) with proper multi-visit dispersion.

    CLASS-AWARE ROUTING (`class_aware=True`, opt-in). The default sends every sampled request
    through EVERY station using mix-averaged demands. The real workload is a 5-class mixture in
    which a request visits only its own class's stations: `loginaction` (33.2% of the measured
    mix) is the only class paying Auth's 19.5 ms, `product` (16.6%) the only one visiting
    Recommender, and WebUI demand spans 4.4-18.5 ms across classes. Mix-averaging therefore
    deletes the BETWEEN-class variance, which dominates: measured over 20,000 draws on the
    baseline model, the default gives sd 13.2 ms against class-aware's ~20-24 ms while the means
    agree (25.1 ms). Under-dispersion inflates Cliff's Delta, so the default detects at a lower
    demand scaling than a faithful sampler would.

    SimQPN simulates the coloured QPN and routes by class, so class-aware is the faithful analog
    and the default collapse is an undisclosed deviation. Note the direction — it makes step D
    detect LESS, not more.

    ENDPOINT-SCOPED SAMPLING (`target_class="loginaction"`, opt-in).
    SimQPN's Table I(a) in Liao et al. (ICSE 2025) reports the response time of the specific
    URL probe under test (e.g. `/loginAction` for L1, `/category` for L2, `/product` for L3),
    rather than a system-wide average pooled across unrelated transactions. When `target_class`
    is provided, every sample is drawn exclusively for requests belonging to that transaction
    class, reproducing the exact probe-level sojourn time distributions evaluated in SimQPN.
    """
    if model.get("form", "open") != "open":
        raise ValueError("sample_response_times supports open networks only; "
                         "closed models have no closed-form sojourn distribution here")
    lam = model["arrival_rate"]
    servers = model.get("servers") or {}
    visits_map = visits or model.get("visits") or {}

    # Station-level queueing is driven by the AGGREGATE load at that station regardless of which
    # class a given request belongs to, so waiting time is computed from `demands` in both modes.
    # Only the set of stations visited and the service time drawn there differ.
    params = {}
    for k, d in demands.items():
        if d <= 0:
            continue
        m = max(1, int(servers.get(k, 1)))
        a = lam * d
        if a / m >= 1.0 - SATURATION_EPS:
            raise SaturatedError(f"station {k} saturated (rho={a / m:.6f})")
        mu = 1.0 / d
        V_k = float(visits_map.get(k, 1.0))
        params[k] = (erlang_c(a, m), m * mu - lam, mu, d, V_k)

    if not class_aware and target_class is None:
        out = []
        for _ in range(n):
            total = 0.0
            for c, wait_rate, mu, d, V_k in params.values():
                if V_k > 1.0:
                    n_v = max(1, int(round(V_k)))
                    for _ in range(n_v):
                        if rng.random() < c:
                            total += rng.expovariate(V_k * wait_rate)
                    total += rng.gammavariate(V_k, d / V_k)
                else:
                    if rng.random() < c:
                        total += rng.expovariate(wait_rate)
                    total += rng.expovariate(mu)
            out.append(total)
        return out

    mix = model.get("request_mix") or {}
    per_class = per_class_service or model.get("per_class_service_s") or {}
    if not mix or not per_class:
        raise ValueError("class_aware / target_class sampling needs `request_mix` and `per_class_service_s` "
                         "in the model descriptor")

    # `demands` may be a SCALED copy of the model's own demands (step D scales one station).
    # Carry that scaling into the per-class service times, or the class-aware path would
    # silently ignore the very perturbation it is being asked to evaluate.
    base = model.get("demands") or {}
    scale = {k: (demands[k] / base[k] if base.get(k) else 1.0) for k in demands}

    classes = sorted(mix)
    weights = [mix[c] for c in classes]
    out = []
    for _ in range(n):
        cls = target_class if target_class is not None else rng.choices(classes, weights)[0]
        total = 0.0
        for k, (c, wait_rate, _mu, _d, V_k) in params.items():
            d_cls = per_class.get(k, {}).get(cls)
            if not d_cls:
                continue                      # this class does not visit this station
            d_actual = d_cls if per_class_service else (d_cls * scale.get(k, 1.0))
            if V_k > 1.0:
                n_v = max(1, int(round(V_k)))
                for _ in range(n_v):
                    if rng.random() < c:
                        total += rng.expovariate(V_k * wait_rate)
                total += rng.gammavariate(V_k, d_actual / V_k)
            else:
                if rng.random() < c:
                    total += rng.expovariate(wait_rate)
                total += rng.expovariate(1.0 / d_actual)
        out.append(total)
    return out


def class_response_time(model: dict, demands: dict[str, float], target_class: str,
                        per_class_service: dict = None) -> dict:
    """Exact Jackson/BCMP expected response time for a specific request class (open network).

    E[R_c] = sum_{k: S_{k, c} > 0} (S_{k, c} + W_k)
    where W_k = C_k / (m_k / D_k - lambda) is the queue waiting time at station k.
    """
    lam = model["arrival_rate"]
    servers = model.get("servers") or {}
    per_class = per_class_service or model.get("per_class_service_s") or {}
    base = model.get("demands") or {}
    scale = {k: (demands[k] / base[k] if base.get(k) else 1.0) for k in demands}

    total_r = 0.0
    waits = {}
    service_times = {}
    saturated = []
    for k, d in demands.items():
        if d <= 0:
            continue
        m = max(1, int(servers.get(k, 1)))
        a = lam * d
        if a / m >= 1.0 - SATURATION_EPS:
            saturated.append(k)
            continue
        mu = 1.0 / d
        c = erlang_c(a, m)
        wq = c / (m * mu - lam)
        waits[k] = wq
        d_cls = per_class.get(k, {}).get(target_class, 0.0)
        if d_cls > 0:
            d_actual = d_cls if per_class_service else (d_cls * scale.get(k, 1.0))
            service_times[k] = d_actual
            total_r += d_actual + wq

    if saturated:
        total_r = float("inf")

    return {
        "class": target_class,
        "response_time_s": total_r,
        "service_times": service_times,
        "waiting_times": waits,
        "saturated": saturated,
    }


def solve(model: dict, demands: dict[str, float]) -> dict:
    """Solve whichever form `model` describes, with `demands` overriding its own.

    `model` is the architectural model descriptor produced by paper_model.py:
      {"form": "open",   "arrival_rate": float, "servers": {...}}
      {"form": "closed", "n_users": int, "think_time": float}
    """
    form = model.get("form", "open")
    if form == "open":
        return solve_open(model["arrival_rate"], demands, model.get("servers"))
    if form == "closed":
        return solve_closed_mva(model["n_users"], model.get("think_time", 0.0), demands)
    raise ValueError(f"unknown model form: {form!r}")
