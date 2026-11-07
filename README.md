# Dual-Track Performance Regression Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-195%20passed%2C%207%20skipped-brightgreen.svg)](#running-the-test-suite)

A research framework for building datasets of **performance regressions in AI-generated code** without synthetic fault injection.

A multi-agent Agile pipeline drives the [Cline](https://github.com/cline/cline) CLI to implement one feature twice, once with a larger "senior" model and once with a smaller "junior" model. Both implementations must pass the same frozen test suite. The pair is then profiled with Kieker and compared with non-parametric statistics. The repository also contains a re-implementation of the architectural regression detector from Liao et al., *Early Detection of Performance Regressions by Bridging Local Performance Data and Architectural Models* (ICSE 2025), with an analytical queueing-network solver in place of SimQPN.

> **Scope of this repository.** This is the framework code only. The generated datasets (`perf_dataset*.jsonl`, `teastore-dataset/`), the TeaStore baseline checkout, the paper's replication artifacts (`paper_artifacts/`), and several TeaStore-specific batch drivers are not tracked. The profiling and detection layers were built and validated against [TeaStore](https://github.com/DescartesResearch/TeaStore) and still assume it in places. [Target-specific assumptions](#target-specific-assumptions) lists where.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Pick a stack preset](#1-pick-a-stack-preset)
  - [2. Run one dual-track feature](#2-run-one-dual-track-feature)
  - [3. Profile a workspace on its own](#3-profile-a-workspace-on-its-own)
  - [4. Gate the measurement](#4-gate-the-measurement)
  - [5. Chain features on a trunk](#5-chain-features-on-a-trunk)
  - [6. System-level oracle](#6-system-level-oracle)
  - [7. Regression detection (Liao et al.)](#7-regression-detection-liao-et-al)
- [Profiling & Statistical Method](#profiling--statistical-method)
- [Target-Specific Assumptions](#target-specific-assumptions)
- [Configuration Reference](#configuration-reference)
- [Running the Test Suite](#running-the-test-suite)
- [License, Citation & AI Assistance](#license-citation--ai-assistance)

---

## How It Works

Fault-injection benchmarks (an inserted `Thread.sleep`, a mutated query) are easy to produce but do not look like the inefficiencies real developers or code assistants introduce. This framework produces **natural counterfactual pairs** instead:

- **Senior track:** a larger model plans the feature, writes the contract (`contract.json`) and the tests (`src/test/**`), and implements it.
- **Junior track:** a smaller model, ideally from the same family, implements the same feature against the senior's frozen contract and tests. It cannot author tests or revise the contract.

A pair is admitted to the dataset only if both tracks compile, pass with zero failures, and run the same number of tests. Any performance difference therefore comes from how the feature was implemented, not from what it does.

`agile_team_cline/dual_track.py` runs five phases per feature:

| Phase | What happens |
| :--- | :--- |
| **A. Senior** | Stage `<prefix>-senior` by rsyncing the baseline (excluding `target/` and `.git`), then run the pipeline with the senior model and `AGILE_AUTHOR_TESTS=1`. `--reuse-senior <dir>` skips this phase and reuses an existing senior workspace. |
| **B. Freeze & fork** | Stage `<prefix>-junior` from the baseline and copy in the senior's `contract.json` and every `src/test/**` file. |
| **C. Junior** | Run the pipeline with the junior model, `AGILE_AUTHOR_TESTS=0` and `AGILE_CONTRACT_ESCALATION=0`. `--junior-brief` appends extra implementation constraints, for example `scripts/junior_inefficiency_menu.txt` to induce a regression on purpose. |
| **D. Equivalence & profiling** | Run the full frozen suite once per track with no profiling. If both are green, profile each track once more with Kieker, scoped to a single exerciser test class. |
| **E. Dataset row** | Append a JSON row to the dataset (default `perf_dataset.jsonl`). It holds the equivalence verdict, per-track surefire counts and perf summaries, the local regression report, `feature_exercised`, `deployability`, chain metadata, and `admitted`. |

```
            Feature request
                  │
     ┌────────────┴────────────┐
     ▼                         ▼
 Senior track               Junior track
 plan + tests + impl  ──►   frozen contract + tests, impl only
     │                         │
     └────────────┬────────────┘
                  ▼
   Equivalence gate (full frozen suite, unprofiled)
                  ▼
   Kieker profiling (one JVM per track, 30 in-session repeats)
                  ▼
   Dataset row  ──►  measurement gate · system oracle · Liao et al. steps B→D
```

### The agent pipeline

`agile_team_cline/pipeline.py` (`python -m agile_team_cline.pipeline "<feature>" --stack <preset> --output-dir <dir>`) runs a sprint loop:

- **Architect** (CAMEL agent, or Cline when `AGILE_ARCHITECT_BACKEND=cline`) plans the contract and the backlog (`ceremonies/planning.py`).
- **Cline** writes the code. `cline_runner.py` is the only module that talks to the Cline CLI.
- **QA** (CAMEL) produces a test report from compile and test results, with per-task verification functions from `verify_lib.py`.
- **Product Owner** (CAMEL) accepts or rejects tasks from the QA report (`ceremonies/review.py`). A deterministic floor in `pipeline.py` constrains the verdict.
- **Tech-debt analyst** (CAMEL, architect model) proposes fix tasks (`ceremonies/tech_debt.py`).
- **Contract escalation** (`ceremonies/contract_escalation.py`) sends contract defects back to the architect. It is disabled on the junior track.

`scrum_master.py` is plain orchestration code that manages sprint state on the blackboard, not an LLM agent. `repair.py` applies deterministic, compiler-error-driven fixes to Java projects. `smoke.py` boots a Spring Boot app and probes GET routes for 5xx errors.

---

## Repository Structure

```text
.
├── agile_team_cline/              # Multi-agent pipeline + dual-track engine
│   ├── agents/                    # CAMEL agent factory (base.py) and role prompts (roles.py)
│   ├── ceremonies/                # planning, review (QA + PO), tech debt, contract escalation
│   ├── shared/                    # Blackboard state store, JSON extraction, LLM retry helpers
│   ├── stacks/                    # Stack presets: java-spring, java-jakarta, python-fastapi
│   ├── tests/                     # 19 test modules (no LLM or network needed)
│   ├── cline_runner.py            # Cline CLI auth + headless act-mode sessions (JSONL output)
│   ├── config.py                  # Token budgets and CAMEL model construction
│   ├── dual_track.py              # Senior/junior engine (phases A–E)
│   ├── java_probe.py              # tree-sitter Java probe: controllers, routes, truncated files
│   ├── pipeline.py                # Sprint loop entry point
│   ├── repair.py                  # Deterministic compile-error repair for Java projects
│   ├── scrum_master.py            # Sprint state orchestration (not an LLM agent)
│   ├── smoke.py                   # Runtime smoke gate for Spring Boot apps
│   ├── sprint.py                  # Single-sprint execution, QA gates, pom/source guards
│   ├── verify_lib.py              # Per-task verification functions (VeriMAP-style)
│   └── workspace_discovery.py     # Brownfield discovery of existing packages and files
├── profiling/                     # Measurement, statistics, and the Liao et al. pipeline
│   ├── oracle/                    # TeaStore JMeter plans + adapt_jmx.py (stock-JMeter rewrite)
│   ├── aop.xml                    # AspectJ load-time weaving config for Kieker
│   ├── kieker.local.properties    # Kieker FileWriter config
│   ├── kieker_args.py             # Builds the Surefire -DargLine for Kieker
│   ├── parse_kieker.py            # Parses Kieker ASCII .dat records → summary.csv / traces
│   ├── compare_kieker.py          # Per-method senior vs junior comparison
│   ├── nonparametric_stats.py     # Wilcoxon rank-sum + Cliff's delta (stdlib only)
│   ├── system_oracle_stats.py     # Same verdict over JMeter .jtl results
│   ├── paper_local_deviations.py  # Liao et al. step B: local deviations
│   ├── paper_propagate.py         # Liao et al. step C: propagation over the call graph
│   ├── paper_model.py             # Liao et al.'s TeaStore .qpe → solver model descriptor
│   ├── qn_solver.py               # Open M/M/m (Erlang-C) and closed exact-MVA solver
│   └── paper_detect.py            # Liao et al. step D: model-based detection with bootstrap
└── scripts/                       # Orchestration and analysis tools
    ├── run_dual_track.sh          # Launcher for dual_track.py (sources .venv and .env)
    ├── junior_inefficiency_menu.txt # Anti-pattern catalog for --junior-brief
    ├── check_single_injection.py  # Checks a junior's claimed anti-pattern against its diff
    ├── kieker_profile.sh          # Standalone in-session Kieker profiling of a workspace
    ├── measurement_gate.py        # Checks C1–C7 that a profile measured what it claims
    ├── apply_setup_split.py       # Removes the fixture-loading phase from a profile
    ├── remeasure_chain.py         # Re-profiles chain workspaces without LLM calls
    ├── advance_trunk.sh           # Mirrors an accepted senior into the trunk, with deletion guard
    ├── lib_trunk_chain.sh         # Chain decision functions (resume, landable, manifest)
    ├── restore_workspace_sources.sh # Restores missing files from src_snapshot.tar.gz
    ├── kill_cline_daemons.sh      # Kills stale cline-hub-daemon processes before a run
    ├── dataset_view.py            # "Last row wins" view and counts over an append-only dataset
    ├── oracle_compose.py          # docker-compose override that builds from workspace source
    ├── run_oracle.sh              # System-level JMeter oracle for a senior/junior pair
    ├── merge_oracle_into_rows.py  # Folds oracle_stats.json into dataset rows
    ├── run_paper_algorithm.py     # Runs steps B→C→D over every profiled pair
    └── score_paper_algorithm.py   # Confusion matrix of predictions vs the oracle
```

---

## Prerequisites

- **OS:** macOS or Linux. Development was on macOS (Apple silicon), and several script defaults assume Homebrew paths such as `JAVA_HOME=/opt/homebrew/opt/openjdk`.
- **Python 3.10+.** The code uses `X | None` annotations at runtime. Development used 3.14.
- **Java 17+ and Maven 3.8+.** The equivalence gate and profiling call `mvn test`. For a Java 8 test JVM, set `KIEKER_NO_ADD_OPENS=1`.
- **Cline CLI.** `cline_runner.py` was checked against 3.0.15:
  ```bash
  npm install -g cline@3.0.15
  ```
- **An OpenAI-compatible LLM endpoint** for both the CAMEL roles and Cline.
- **Kieker 2.0.0 AspectJ agent** (`kieker-2.0.0-aspectj.jar`) for method-level profiling. The default location is `~/.m2/repository/net/kieker-monitoring/kieker/2.0.0/kieker-2.0.0-aspectj.jar`. Set `AJ_WEAVER` to override it.
- **Docker and JMeter**, only for the system-level oracle (`scripts/run_oracle.sh`).
- **rsync**, used to stage workspaces.

---

## Installation

```bash
git clone https://github.com/mohamed14787/agentic-agile-perf-regression.git
cd agentic-agile-perf-regression

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r agile_team_cline/requirements.txt
```

`requirements.txt` pins the versions the test suite was verified with. `camel-ai` 0.2.90 fails to import with `mcp` 2.x, so keep `mcp` pinned. The tree-sitter packages are optional, and the pipeline falls back to regex scanners without them. NumPy and SciPy are not needed. The statistics and the solver use only the standard library.

Create a `.env` at the repository root. `scripts/run_dual_track.sh` sources it, and `.gitignore` excludes it.

```bash
# Endpoint used by the CAMEL roles, and by Cline unless AGILE_CLINE_* is set
OPENAI_API_KEY="..."
OPENAI_API_BASE_URL="https://your-openai-compatible-endpoint/v1"

# Models for the CAMEL roles (QA, Product Owner, Architect)
TECHTEAM_MODEL_TYPE="<model-id>"
TECHTEAM_ARCHITECT_MODEL_TYPE="<model-id>"

# Optional: send Cline to a different endpoint than the CAMEL roles
# AGILE_CLINE_API_KEY="..."
# AGILE_CLINE_API_BASE_URL="..."
```

The senior and junior models are not read from `.env` by the Python code. Pass them as `--senior-model` and `--junior-model`, or as `SENIOR_MODEL` and `JUNIOR_MODEL` to the shell launcher. Both become `AGILE_CLINE_MODEL` for their track, so they select the implementation model only. Planning, QA, and review use the same models on both tracks.

Verify the install:

```bash
pytest agile_team_cline/tests/ -q   # expected: 195 passed, 7 skipped
```

---

## Usage

Run commands from the repository root with the virtual environment active and `PYTHONPATH=.`.

### 1. Pick a stack preset

`load_stack()` in `pipeline.py` loads a preset by file name from `agile_team_cline/stacks/`:

| Preset | Framework | Dual-track support |
| :--- | :--- | :--- |
| `java-spring` | Spring Boot / Maven / JUnit 5 | Yes |
| `java-jakarta` | Jakarta EE / JAX-RS / JPA (TeaStore) | Yes. This preset produced the dataset. |
| `python-fastapi` | FastAPI / pytest / SQLAlchemy | Pipeline only. The equivalence gate and profiling call `mvn`. |

To add a stack, add `agile_team_cline/stacks/<name>.json` with at least `name`, `language`, `framework`, `build_tool`, `test_framework` and `file_extension`. The optional keys are `task_types`, `annotations` and `imports`, which `java-spring.json` shows. The pipeline treats a stack as Java when `language` is `Java` or `file_extension` is `.java`.

### 2. Run one dual-track feature

```bash
PYTHONPATH=. python -m agile_team_cline.dual_track \
  --feature "Add GET /products/cheapest/{limit} returning the cheapest products by price" \
  --stack java-jakarta \
  --baseline /path/to/clean/target-repo \
  --out-prefix myfeature \
  --senior-model <large-model-id> \
  --junior-model <small-model-id> \
  --maven-test-args "-pl <module> -am" \
  --dataset perf_dataset.jsonl
```

This creates `myfeature-senior/` and `myfeature-junior/` in the current directory and appends one row to the dataset. The dataset is an append-only log, so re-running a feature adds a new row. `scripts/dataset_view.py` gives the "last row wins" view.

These options are also available:

- `--feature-id`: the dataset key. It defaults to `--out-prefix`.
- `--reuse-senior <dir>`: reuse a finished senior workspace and run only the junior track.
- `--junior-brief "$(cat scripts/junior_inefficiency_menu.txt)"`: induce a regression in the junior's implementation while keeping it functionally equivalent. `scripts/check_single_injection.py` checks the junior's claimed anti-patterns against its diff.

The staging step fails if the baseline has no `.java` files, unless `DUAL_ALLOW_EMPTY_BASELINE=1` is set. It also fails if rsync copied fewer `.java` files than the baseline contains.

`scripts/run_dual_track.sh` wraps the same command. It sources `.venv` and `.env`, sets the pipeline flags used for the dataset (verification-aware QA, Cline as architect backend, thinking modes, timeouts), and logs to `dual_track_<prefix>_<timestamp>.log`. Its defaults are TeaStore-specific (`STACK=java-jakarta`, `BASELINE=TeaStore`, and the persistence module as build scope):

```bash
JUNIOR_MODEL=<small-model-id> SENIOR_MODEL=<large-model-id> \
BASELINE=/path/to/TeaStore DUAL_FEATURE="<feature text>" OUT_PREFIX=myfeature \
zsh scripts/run_dual_track.sh
```

`JUNIOR_MODEL` is required. `SENIOR_MODEL` falls back to `TECHTEAM_MODEL_TYPE`.

> **Profiling on a non-TeaStore target.** By default the profiled run is scoped to the test class `FeaturePerfExerciseTest` and switched on with `-Dteastore.perf.*` properties. The framework does not generate that class, so the target repository must provide it. Without it, the equivalence verdict is still valid but the profile is empty. In that case, set `DUAL_KIEKER=0` for equivalence-only rows, or set `DUAL_KIEKER_TEST_SCOPE=""` to profile the whole suite. Profiling the whole suite can produce several Kieker sessions, one per forked JVM, which weakens the in-session protocol.

### 3. Profile a workspace on its own

```bash
# One Kieker session with 30 in-session repeats (PERF_repeats overrides the count)
zsh scripts/kieker_profile.sh <workspace> [iterations=1] [module=services/tools.descartes.teastore.persistence] [out-root]

python profiling/parse_kieker.py <run_root> [--raw]            # → summary.csv
python profiling/compare_kieker.py <baseline_run> <candidate_run> \
  [--threshold-pct 20] [--min-delta-ms 0.5] [--min-calls 5] [--apply-offset]
```

### 4. Gate the measurement

`scripts/measurement_gate.py` checks that a pair's profile measured what it claims to measure:

```bash
python scripts/measurement_gate.py <feature> --dataset <dir containing <feature>-senior/-junior> \
  --method <feature method name> [--profile-dir profile_only] [--repeats 30]
```

| Check | Condition |
| :--- | :--- |
| C1 | The feature method appears in `summary.csv`. |
| C2 | The feature method's `n_calls` equals `repeats`. |
| C3 | `meta.json` records the expected repeat count. |
| C4 | `meta.json.only` is the feature method, so the isolated workload was used rather than the 25-query sweep. |
| C5 | The harness fingerprint matches the current `aop.xml`, `kieker_profile.sh`, and exerciser source. |
| C6 | Both tracks agree on the feature method's call count. |
| C7 | The profile contains exactly one Kieker session. |

Pass `--method`, because the default lookup file `scripts/feature_methods.json` is not in the repository. Bare feature ids get the `teastore-hist-` prefix unless `<feature>-senior` exists. The default `--dataset` is `teastore-dataset/`.

### 5. Chain features on a trunk

In a longitudinal chain, feature *N+1* is staged from a trunk that already contains the accepted senior code of feature *N*. Junior workspaces never land on the trunk.

```bash
rsync -a /path/to/clean/target-repo/ my-trunk/

# after <prefix>-senior has passed its tests:
zsh scripts/advance_trunk.sh my-trunk <prefix>-senior

# then use my-trunk as the next feature's --baseline
```

`advance_trunk.sh` does an `rsync --delete` mirror. It excludes build output, profiles, logs, and pipeline bookkeeping such as `blackboard.json` and `sprint_backlog.json`. It runs a dry run first and aborts if more than `ADVANCE_MAX_DELETES` (default 10) files would be deleted. `ADVANCE_FORCE=1` overrides the abort.

`scripts/lib_trunk_chain.sh` holds the decision functions the batch driver used: resume, landable, prune unexercised rows, and update `chain_manifest.json`. The batch driver itself (`run_teastore_chained_history.sh`) is not in this repository. `scripts/remeasure_chain.py` re-profiles an existing chain without LLM calls. It expects `chain_manifest.json` and `perf_dataset_teastore_history.jsonl`, and it hardcodes the TeaStore persistence module as the Maven scope.

### 6. System-level oracle

```bash
zsh scripts/run_oracle.sh <senior_ws> <junior_ws> <ticket_id> \
  [threads=5] [duration_s=30] [warmup_s=5] [rps=5] [repeats=1]

# Closer to the paper's protocol:
zsh scripts/run_oracle.sh <sen> <jun> PERF-x 5 600 180 50 3
```

The oracle builds both workspaces into Docker images (`scripts/oracle_compose.py` writes a compose override that builds from workspace source rather than pulling upstream images). It load-tests both with the same JMeter scenario, alternating senior and junior when `repeats > 1`, and writes `oracle_stats.json` into the junior workspace. `scripts/merge_oracle_into_rows.py` copies those results into dataset rows. The oracle is TeaStore-specific: it uses TeaStore's services, compose layout, and `profiling/oracle/teastore*.jmx`. Its documented deviations from the paper are Docker instead of Podman, stock JMeter thread groups instead of the BlazeMeter plugin, a single machine, and shorter runs.

### 7. Regression detection (Liao et al.)

1. **Build the architectural model.** `profiling/paper_model.py` converts Liao et al.'s TeaStore QPN (`teastore_model.qpe`, five FCFS service stations with two servers each, plus a workload queue) into the descriptor `qn_solver.py` reads. It expects the paper's replication package (Zenodo [10.5281/zenodo.13135986](https://doi.org/10.5281/zenodo.13135986)) under `paper_artifacts/teastore/`. You must download it yourself, because the fetch script is not in this repository.
   ```bash
   python profiling/paper_model.py --out my_model.json [--mix measured|uniform] [--arrival-rate R]
   ```
2. **Run steps B→C→D** over every profiled pair in `teastore-dataset/`:
   ```bash
   python scripts/run_paper_algorithm.py --model my_model.json \
     [--profile-dir profile_only] [--replicates 1000] [--seed 20260727] [--all-variants]
   ```
   Results go to `paper_algorithm_results.jsonl`.
3. **Score against the oracle:**
   ```bash
   python scripts/score_paper_algorithm.py --results paper_algorithm_results.jsonl \
     --dataset <dataset.jsonl>
   ```
   Pass both flags. The defaults point to files that are not in the repository.

---

## Profiling & Statistical Method

Managed-runtime timings are noisy. JIT warm-up, GC pauses, and host load all shift them. On this project's hardware, two back-to-back `mvn test` runs differed by up to about 30%. That offset once flipped the sign of a real +24% regression. The protocol below addresses this.

1. **In-session repetition.** Following Liao et al.'s replication package, each track is profiled in one JVM and one Kieker session, with the feature query repeated 30 times inside it (`DUAL_PERF_repeats`). The profiled run is scoped with `-Dtest=FeaturePerfExerciseTest`, so Surefire's per-class forking cannot split it into many cold sessions. Repeats are warm: only the query is repeated, not the test setup. This is a disclosed deviation from the paper.
2. **Separate functional verdict.** Equivalence comes from an unprofiled run of the full frozen suite, never from the scoped profiling run. If either track fails, timing is skipped.
3. **Order control.** With the default `DUAL_KIEKER_ITERS=1`, each track is profiled once. A hash of the pair's directory name decides which track runs first (the cold slot), so cold slots split roughly 50/50 across pairs instead of always falling to the senior. The choice is recorded as `lead_track` in `meta.json`. With more iterations, the order alternates.
4. **Control-method offset (diagnostic by default).** `compare_kieker.py` estimates a run offset as the median candidate/baseline mean ratio over methods whose source is identical in both tracks. Since 2026-07-30 it only reports this offset (`run_offset_pct`, `n_control_methods`) as a contamination signal and does not divide it out, which matches the paper. `--apply-offset` (or `apply_offset=True`) applies the correction for sensitivity analysis.
5. **Local regression flag** (`compare_kieker.py`). A method is flagged when its mean rises at least 20%, the increase is at least 0.5 ms, both sides have at least 5 calls, and a noise guard passes. With raw samples, the guard requires Wilcoxon p ≤ 0.05 and a non-negligible Cliff's delta. With summaries only, it falls back to a 2σ test on the means. A method that clears the thresholds but fails the guard is reported as `suspect`, not as a regression.
6. **Statistics** (`nonparametric_stats.py`, standard library only). The test is a two-sided Wilcoxon rank-sum (Mann–Whitney U) with normal approximation and tie correction. Effect size is Cliff's delta,

   $$d = \frac{\#(\text{candidate} > \text{baseline}) - \#(\text{candidate} < \text{baseline})}{n_1 \cdot n_2}$$

   with magnitude labels |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, otherwise large. The same helper serves the Kieker signal, the JMeter oracle, and the paper's steps B and D, which use p < 0.05 and a non-negligible effect.

### Liao et al. re-implementation

| Step | Module | Implementation |
| :--- | :--- | :--- |
| B: local deviations | `paper_local_deviations.py` | Wilcoxon p < 0.05 and non-negligible Cliff's delta, then the mean difference for deviating components. |
| C: propagation | `paper_propagate.py` | Builds a local dependency graph from `traces.json` and propagates deviations to per-station demand scaling. |
| D: detection | `paper_detect.py` + `qn_solver.py` | Solves the original and updated models, samples response times, and runs a block bootstrap with Wilcoxon and Cliff's delta. |

The paper solves Queueing Petri Nets with SimQPN, a simulator. This repository solves queueing networks analytically instead. The default is an open network of M/M/m stations, with Erlang-C computed through the Erlang-B recursion. A closed exact-MVA solver is used when the model has a fixed population. The paper states that its approach does not depend on a specific model type. This substitution is still a deviation, and predicted values can differ from SimQPN's. `run_paper_algorithm.py` sweeps the other dataset-driven choices as variants: normalization, weighting, and bootstrap.

---

## Target-Specific Assumptions

The core engine (staging, freeze and fork, equivalence gate) works on any Maven project. These parts assume TeaStore:

- **Profiling workload:** the `FeaturePerfExerciseTest` class and the `-Dteastore.perf.*` properties.
- **Default Maven scope:** `services/tools.descartes.teastore.persistence` in `run_dual_track.sh`, `kieker_profile.sh`, and `remeasure_chain.py`.
- **System oracle:** TeaStore Docker services, compose layout, and JMeter plans.
- **Architectural model:** Liao et al.'s TeaStore `.qpe` file, read from `paper_artifacts/`.
- **Default paths:** `teastore-dataset/`, `perf_dataset_teastore_history.jsonl`, `chain_manifest.json`, and the `teastore-hist-` feature-id prefix in the analysis scripts.

These files are referenced in the code but are not in this repository: `scripts/run_teastore_chained_history.sh`, `scripts/feature_methods.json`, `scripts/feature_endpoint.py`, `scripts/fetch_paper_artifacts.py`, and `scripts/run_oracle_session.sh`.

---

## Configuration Reference

These are the most-used environment variables and their defaults in code. `run_dual_track.sh` overrides several of them.

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | *(none)* | API key for the CAMEL roles, and for Cline unless `AGILE_CLINE_API_KEY` is set |
| `OPENAI_API_BASE_URL` | *(none; `OPENAI_BASE_URL` is also read)* | OpenAI-compatible endpoint |
| `AGILE_CLINE_API_KEY` / `AGILE_CLINE_API_BASE_URL` | *(fall back to `OPENAI_*`)* | Separate endpoint for Cline only |
| `TECHTEAM_MODEL_TYPE` | `qwen-max` | Model for the CAMEL roles (QA, Product Owner). Set it explicitly. |
| `TECHTEAM_ARCHITECT_MODEL_TYPE` | `google/gemma-4-31B-it` | Model for the architect, planner, and tech-debt agents |
| `TECHTEAM_TEMPERATURE` | `0.2` | CAMEL sampling temperature. Not sent to `api.openai.com`. |
| `AGILE_LLM_TIMEOUT` | `600` | Per-request timeout for CAMEL calls, in seconds |
| `AGILE_CLINE_TIMEOUT` | `1800` | Wall-clock limit per Cline session, in seconds |
| `AGILE_CLINE_COMPACTION` | `agentic` | Cline context compaction strategy |
| `AGILE_ARCHITECT_BACKEND` | `camel` | `cline` makes Cline the architect when a workspace exists |
| `AGILE_VERIFY_AWARE` | `0` | Verification functions: `0` off, `shadow` log only, `1` reject and self-refine |
| `AGILE_MAX_TASKS_PER_SPRINT` / `AGILE_TOTAL_SPRINTS` | `6` / `5` | Sprint sizing |
| `DUAL_KIEKER` | `1` | Kieker profiling in the dual-track engine |
| `DUAL_INTERLEAVE` | `1` | Use the interleaved profiler. `0` profiles the tracks sequentially. |
| `DUAL_KIEKER_ITERS` | `1` | Profiled JVM sessions per track |
| `DUAL_PERF_repeats` | `30` | In-session repeats of the feature query |
| `DUAL_PERF_only` | *(unset)* | Profile one method in isolation. The profile is written to `profile_only/`. |
| `DUAL_KIEKER_TEST_SCOPE` | `FeaturePerfExerciseTest` | Test class for profiled runs. Empty profiles the whole suite. |
| `DUAL_KIEKER_THRESHOLD_PCT` / `_MIN_DELTA_MS` / `_MIN_CALLS` | `20` / `0.5` / `5` | Local regression thresholds |
| `DUAL_REQUIRE_FEATURE_EXERCISED` | `1` | Reject a row when profiling shows the feature code never ran |
| `DUAL_SENIOR_AUTHOR_TESTS` | `1` | `0` judges equivalence on the baseline's existing tests only |
| `DUAL_SRC_SNAPSHOT` | `1` | Write `src_snapshot.tar.gz` when staging a workspace |
| `DUAL_ALLOW_EMPTY_BASELINE` | `0` | Allow a baseline with no `.java` files (greenfield) |
| `AJ_WEAVER` / `AOP_XML` / `KIEKER_PROPS` | see `profiling/kieker_args.py` | Kieker agent and config paths |
| `ADVANCE_MAX_DELETES` / `ADVANCE_FORCE` | `10` / `0` | Deletion guard in `advance_trunk.sh` |

---

## Running the Test Suite

The 19 test modules in `agile_team_cline/tests/` need no LLM, network, or Maven. They cover the pipeline gates and brief construction, workspace staging and pom guards, the profiling harness, the trunk-chain logic, the system oracle, and the Liao et al. pipeline and solver.

```bash
pytest agile_team_cline/tests/ -q
# 195 passed, 7 skipped

pytest agile_team_cline/tests/test_paper_algorithm.py -v   # steps B–D and the QN solver
pytest agile_team_cline/tests/test_profiling.py -v         # Kieker args, meta.json, comparison
pytest agile_team_cline/tests/test_trunk_chain.py -v       # advance_trunk + lib_trunk_chain
pytest agile_team_cline/tests/test_baseline_staging.py -v  # staging integrity guards
```

The 7 skipped tests need artifacts that are not in the repository. Six in `test_system_oracle.py` need `scripts/feature_endpoint.py`. One in `test_vf_ceiling.py` needs a local `teastore-probe-stockledger` workspace.

---

## License, Citation & AI Assistance

Released under the [MIT License](LICENSE).

```bibtex
@software{elsharawy2026agentic,
  author = {Mohamed Elsharawy},
  title  = {Agentic Simulation of Agile Software Evolution for Performance Regression Benchmarks},
  year   = {2026},
  url    = {https://github.com/mohamed14787/agentic-agile-perf-regression}
}
```

The detection pipeline re-implements Liao et al., *Early Detection of Performance Regressions by Bridging Local Performance Data and Architectural Models*, ICSE 2025. Its replication package is [10.5281/zenodo.13135986](https://doi.org/10.5281/zenodo.13135986).

**AI assistance:** Parts of the code and documentation were written with AI coding assistants (Claude, Kiro).
