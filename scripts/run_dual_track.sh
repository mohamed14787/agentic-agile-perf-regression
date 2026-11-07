#!/bin/zsh
# Dual-Track Implementation Engine launcher (thesis Component 2).
# Runs ONE feature through a Senior (large) and Junior (small) model of the SAME
# family, with a frozen shared contract + test suite, then an equivalence gate.
#
# ADDITIVE: invokes the existing pipeline via agile_team_cline.dual_track — touches
# no pipeline internals. Set SENIOR_MODEL / JUNIOR_MODEL to two scales of one family.

cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
source .venv/bin/activate
set -a; source .env; set +a
export PYTHONPATH=.

# ── Senior vs Junior: SAME family, DIFFERENT scale (the single variable) ──────
# Set these to concrete ids available on your OpenAI-compatible endpoint.
SENIOR_MODEL="${SENIOR_MODEL:-${TECHTEAM_MODEL_TYPE}}"     # large / senior
JUNIOR_MODEL="${JUNIOR_MODEL:?set JUNIOR_MODEL to a SMALLER model of the same family}"

# ── Planning held CONSTANT across tracks (architect/QA model unchanged) ───────
export AGILE_VERIFY_AWARE=1            # keep dataset samples valid (junior worse-but-working)
export AGILE_ARCHITECT_BACKEND=cline
export AGILE_INCREMENTAL=1

export AGILE_ARCHITECT_THINKING=1
export AGILE_QA_THINKING=1
export AGILE_CLINE_THINKING=high

export AGILE_CLINE_PROVIDER=openai-compatible
export AGILE_CLINE_TIMEOUT=5400
export AGILE_LLM_TIMEOUT=1800
export AGILE_ARCHITECT_CLINE_TIMEOUT=5400
export AGILE_TEST_CLINE_TIMEOUT=900
export AGILE_CLINE_RETRIES=1
export AGILE_CLINE_GRANULARITY=sprint
export AGILE_CLINE_INJECT_SYMBOLS=1
export AGILE_CLINE_NO_LOMBOK=1
unset AGILE_WENYAN; unset AGILE_CHINESE

# ── Feature (default: single-module cheapest-products — fastest to validate) ──
STACK="${STACK:-java-jakarta}"
BASELINE="${BASELINE:-TeaStore}"
OUT_PREFIX="${OUT_PREFIX:-teastore-cheapest-dual}"
export AGILE_BUILD_MODULE="${TEASTORE_MODULES:-services/tools.descartes.teastore.persistence}"
export AGILE_BUILD_MODULE_DEPS="${TEASTORE_DEPS:-interfaces/tools.descartes.teastore.entities,utilities/tools.descartes.teastore.registryclient}"
# equivalence-gate maven scope (mirror the build scope; -am pulls deps)
MAVEN_TEST_ARGS="${MAVEN_TEST_ARGS:--pl ${AGILE_BUILD_MODULE} -am}"

FEATURE="${DUAL_FEATURE:-Add a cheapest-products query to the persistence service: a JAX-RS endpoint GET /products/cheapest/{limit} on the existing product resource that returns the {limit} cheapest products ordered by listPriceInCents ascending, backed by a new ProductRepository query method. Follow the existing ProductEndpoint and ProductRepository patterns. Work only within the persistence service.}"

TS=$(date +%Y%m%d_%H%M%S)
echo "=== START dual-track ${OUT_PREFIX} (senior=${SENIOR_MODEL} junior=${JUNIOR_MODEL}) $(date +%H:%M:%S) ==="
python -u -m agile_team_cline.dual_track \
  --feature "$FEATURE" --stack "$STACK" \
  --baseline "$BASELINE" --out-prefix "$OUT_PREFIX" \
  --senior-model "$SENIOR_MODEL" --junior-model "$JUNIOR_MODEL" \
  --maven-test-args "$MAVEN_TEST_ARGS" \
  --dataset "${DUAL_DATASET:-perf_dataset.jsonl}" \
  > "dual_track_${OUT_PREFIX}_${TS}.log" 2>&1
echo "=== DONE dual-track rc=$? $(date +%H:%M:%S) -> log: dual_track_${OUT_PREFIX}_${TS}.log ==="
