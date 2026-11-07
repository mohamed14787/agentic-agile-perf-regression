#!/usr/bin/env python3
"""
paper_model.py — turn Liao et al.'s own TeaStore QPN into a solvable model descriptor.

Reads `Model/teastore_model.qpe` from the paper's replication package (Zenodo
10.5281/zenodo.13135986, fetched by scripts/fetch_paper_artifacts.py) and emits the descriptor
`profiling/qn_solver.py` consumes. The topology, the server counts and the service demands are
all THEIRS; only the solver is ours.

WHAT THE .qpe ACTUALLY CONTAINS (QPME 2.x XML):

  * 5 service queues, FCFS, `number-of-servers="2"` each:
        WebUI, Auth, Persistence, ImageProvider, Recommender
    plus a `Workload` queue (1 server) that acts as the arrival generator.
  * 26 queueing places named `<ServiceInitial>/<endpoint>_<page>`, e.g. `P/getProduct_product`
    is the Persistence service's work for a `product` page request. Each place carries one
    `<color-ref xsi:type="queueing-color-reference" distribution-function="Exponential"
    lambda="...">`, so its mean service time is 1/lambda seconds.
  * 6 request-type colors — index, login, loginAction, logout, category, product — and the
    Workload place's own color, whose `lambda="50.0"` is the paper's stated 50 req/s TeaStore
    workload intensity (§IV-C).

So the model is a MULTI-CLASS open network: a request of type `c` visits exactly the places
whose color is `c`, once each. We collapse it to the single aggregate class our solver takes:

    D_k = sum over request types c of  mix[c] * (service time of station k for type c)

which is the standard operational definition of service demand (total service requirement at
station k per request, averaged over the request mix). Utilization and residence time then
follow from the M/M/m formulas in qn_solver.

REQUEST MIX. The mix is not in the .qpe — it lives in the transition firing structure, and all
firing weights in the file are 1.0. Rather than guess, `--mix measured` (the default) derives it
empirically from the paper's OWN system-level measurements
(`System_perf_res/Original_workload/Original/rep*/...-timestamps.csv`), which is the most
defensible source available: those runs are the ground truth the paper's Table I is scored
against. Measured on rep1 it is 15 minutes long at 49.98 req/s — matching the paper's stated
15-minute steady state and 50 req/s exactly — with the mix

    loginAction 33.2%, index 16.8%, login 16.8%, category 16.6%, product 16.6%, logout 0%

`logout` never appears in the measured traffic even though the QPN models it, so it carries
weight 0 under `--mix measured`. `--mix uniform` weights all 6 colors equally instead; the
difference is reported so the choice is auditable rather than buried.

SCALED STATION. `Persistence`, because the Kieker instrumentation in this project covers the
persistence service's `*Repository` domain methods. Step C's relative subsystem deviation is a
ratio, so it applies directly to that station's demand.

Usage:
  python3 profiling/paper_model.py --out paper_artifacts/teastore_model.json
  python3 profiling/paper_model.py --mix uniform --print-only
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QPE = os.path.join(REPO, "paper_artifacts", "teastore", "TeaStore",
                           "Model", "teastore_model.qpe")
DEFAULT_SYSRES = os.path.join(REPO, "paper_artifacts", "teastore", "TeaStore",
                              "System_perf_res", "Original_workload", "Original")
WEBUI_PREFIX = "http://localhost:8080/tools.descartes.teastore.webui"
WORKLOAD_QUEUE = "Workload"


def parse_qpe(path: str) -> dict:
    """Parse a QPME .qpe into {queues: {name: {servers, strategy}}, places: [...],
    arrival_rate: float}."""
    root = ET.parse(path).getroot()

    queues_by_id = {}
    for q in root.iter("queue"):
        queues_by_id[q.get("id")] = {
            "name": q.get("name"),
            "servers": int(q.get("number-of-servers") or 1),
            "strategy": q.get("strategy"),
        }
    colors = {c.get("id"): c.get("name") for c in root.iter("color")}

    places, arrival_rate = [], None
    for pl in root.iter("place"):
        qref = pl.get("queue-ref")
        if qref not in queues_by_id:
            continue
        queue = queues_by_id[qref]
        for cr in pl.iter("color-ref"):
            lam = cr.get("lambda")
            if lam is None:
                continue
            lam = float(lam)
            entry = {
                "place": pl.get("name"),
                "queue": queue["name"],
                "servers": queue["servers"],
                "color": colors.get(cr.get("color-id")),
                "distribution": cr.get("distribution-function"),
                "lambda": lam,
                "mean_service_s": 1.0 / lam if lam else 0.0,
            }
            if queue["name"] == WORKLOAD_QUEUE:
                arrival_rate = lam          # the Workload place's rate IS the arrival rate
            else:
                places.append(entry)

    return {"queues": {q["name"]: q for q in queues_by_id.values()},
            "places": places, "arrival_rate": arrival_rate}


def _page_of_url(url: str) -> str:
    """Normalise a measured URL to a QPN request-type name (lower-cased, marker stripped).

    The measured URLs carry an HTTP-method marker whose case is not guaranteed (`[POST]`), so
    lower-casing happens BEFORE the marker is stripped. Getting this backwards silently sends
    every loginAction request — a third of the workload, and the one carrying Auth's dominant
    19.5 ms service time — into an unmatched bucket with weight 0.
    """
    p = url.replace(WEBUI_PREFIX, "").split("?")[0].lower()
    p = re.sub(r"\[[a-z]+\]", "", p).strip("/")
    return p or "index"


def measured_mix(sysres_dir: str) -> dict:
    """Request-type mix measured from the paper's own system-level runs (all reps pooled)."""
    counts = defaultdict(int)
    files = sorted(glob.glob(os.path.join(sysres_dir, "rep*", "*timestamps.csv")))
    if not files:
        raise FileNotFoundError(f"no *timestamps.csv under {sysres_dir}/rep*")
    for f in files:
        with open(f, newline="", errors="replace") as fh:
            for row in csv.DictReader(fh):
                url = row.get("Url")
                if url:
                    counts[_page_of_url(url)] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}, total, len(files)


def build_model(qpe: dict, mix: dict, scaled_station: str = "Persistence",
                arrival_rate: float = None, provenance: str = "") -> dict:
    """Collapse the multi-class QPN into the single-class descriptor qn_solver.solve takes."""
    demands = defaultdict(float)
    servers = {}
    per_class = defaultdict(lambda: defaultdict(float))
    unmatched = []

    for p in qpe["places"]:
        q, color = p["queue"], (p["color"] or "").lower()
        servers[q] = p["servers"]
        w = mix.get(color, 0.0)
        if color not in mix:
            unmatched.append(f"{p['place']} (color={p['color']})")
        demands[q] += w * p["mean_service_s"]
        per_class[q][color] += p["mean_service_s"]

    # Stations present in the topology but never exercised by the mix still belong in the model
    # with demand 0 rather than vanishing from it.
    for q in qpe["queues"]:
        if q != WORKLOAD_QUEUE:
            demands.setdefault(q, 0.0)
            servers.setdefault(q, qpe["queues"][q]["servers"])

    if scaled_station not in demands:
        raise ValueError(f"scaled_station {scaled_station!r} not in {sorted(demands)}")

    return {
        "form": "open",
        "arrival_rate": arrival_rate if arrival_rate is not None else qpe["arrival_rate"],
        "demands": dict(demands),
        "servers": servers,
        "scaled_station": scaled_station,
        "request_mix": mix,
        "per_class_service_s": {q: dict(v) for q, v in per_class.items()},
        # QPN colors the mix gives no weight to (modelled but never exercised).
        "unmatched_places": unmatched,
        # Mix entries with no corresponding QPN color — these WOULD be a name-normalisation
        # bug, unlike the above, so they are reported separately.
        "mix_keys_without_color": sorted(
            set(mix) - {(p["color"] or "").lower() for p in qpe["places"]}),
        "provenance": provenance,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qpe", default=DEFAULT_QPE)
    ap.add_argument("--sysres", default=DEFAULT_SYSRES,
                    help="dir holding rep*/ *timestamps.csv for --mix measured")
    ap.add_argument("--mix", choices=["measured", "uniform"], default="measured")
    ap.add_argument("--scaled-station", default="Persistence")
    ap.add_argument("--arrival-rate", type=float, default=None,
                    help="override the Workload place's rate")
    ap.add_argument("--out", default=None)
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.qpe):
        print(f"!! {args.qpe} not found — run scripts/fetch_paper_artifacts.py first",
              file=sys.stderr)
        return 1

    qpe = parse_qpe(args.qpe)
    print(f"parsed {args.qpe}")
    print(f"  queues: " + ", ".join(
        f"{n}(m={q['servers']})" for n, q in qpe["queues"].items()))
    print(f"  queueing places: {len(qpe['places'])}   "
          f"arrival_rate from Workload place: {qpe['arrival_rate']} req/s")

    if args.mix == "measured":
        mix, n, nfiles = measured_mix(args.sysres)
        prov = (f"topology+demands: {os.path.relpath(args.qpe, REPO)} (Liao et al. ICSE'25 "
                f"replication package, Zenodo 10.5281/zenodo.13135986); "
                f"request mix measured from {nfiles} rep file(s), {n} requests, under "
                f"{os.path.relpath(args.sysres, REPO)}")
        print(f"  mix source: measured, {n} requests over {nfiles} rep file(s)")
    else:
        colors = sorted({(p["color"] or "").lower() for p in qpe["places"]})
        mix = {c: 1.0 / len(colors) for c in colors}
        prov = (f"topology+demands: {os.path.relpath(args.qpe, REPO)} (Liao et al. ICSE'25); "
                f"request mix: uniform over {len(colors)} QPN colors")
        print(f"  mix source: uniform over {len(colors)} colors")

    model = build_model(qpe, mix, args.scaled_station, args.arrival_rate, prov)

    print(f"\n  {'request type':<16}{'mix':>9}")
    for c, w in sorted(model["request_mix"].items(), key=lambda kv: -kv[1]):
        print(f"  {c:<16}{w * 100:>8.2f}%")
    if model["unmatched_places"]:
        print(f"\n  QPN places whose request type never appears in the measured traffic "
              f"(demand contribution 0 — modelled but not exercised):")
        for u in model["unmatched_places"]:
            print(f"    {u}")
    if model["mix_keys_without_color"]:
        print(f"\n  !! measured request types with NO matching QPN color — this is a "
              f"name-normalisation bug, not a modelling gap:")
        for u in model["mix_keys_without_color"]:
            print(f"    {u}")

    from qn_solver import solve
    res = solve(model, model["demands"])
    print(f"\n  {'station':<16}{'m':>3}{'D (ms)':>10}{'U':>9}{'R_k (ms)':>11}")
    for k in sorted(model["demands"], key=lambda x: -model["demands"][x]):
        print(f"  {k:<16}{model['servers'][k]:>3}{model['demands'][k] * 1000:>10.4f}"
              f"{res['utilization'][k]:>9.4f}{res['residence_time'][k] * 1000:>11.4f}")
    print(f"\n  predicted system response time: {res['response_time_s'] * 1000:.4f} ms "
          f"at {model['arrival_rate']} req/s")
    if res["saturated"]:
        print(f"  !! SATURATED stations: {res['saturated']}")
    print(f"  scaled station for step C/D: {model['scaled_station']} "
          f"(D = {model['demands'][model['scaled_station']] * 1000:.4f} ms)")

    if args.out and not args.print_only:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(model, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
