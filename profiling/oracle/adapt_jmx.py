#!/usr/bin/env python3
"""adapt_jmx.py — disclosed deviation: replace the paper's BlazeMeter-plugin load-generation
elements (com.blazemeter.jmeter.threads.concurrency.ConcurrencyThreadGroup +
kg.apc.jmeter.timers.VariableThroughputTimer) with stock JMeter equivalents (ThreadGroup +
ConstantThroughputTimer), so the script runs on a plain `jmeter` install with no third-party
plugin. Confirmed necessary by direct dry-run: `jmeter -n -t teastore.jmx ...` fails at XML
load with CannotResolveClassException for ConcurrencyThreadGroup — the plugin genuinely isn't
present and homebrew's jmeter formula ships no plugin-manager CLI wrapper to install it.

Everything else in the script (HTTP flow: Home/Login/LoginAction/Category/Product/Logout,
regex extractors, cookie manager, user-defined ${hostname}/${port}/${rps}/${thread}/
${duration}/${warmup} properties) is preserved byte-for-byte — only the load-generation
ENGINE changes, not the test SCENARIO. Same disclosure spirit as the existing
Podman->Docker-Compose substitution in the system oracle protocol.

Second disclosed deviation: the source .jmx ships every HTTPSamplerProxy with empty
connect_timeout/response_timeout (unbounded waits). Confirmed live: a single stalled
backend call hung an entire JMeter run for 1h45m with zero log output before the process
had to be killed — the same unbounded-read hang class already guarded against for the
warmup curl probe in scripts/run_oracle.sh (-m 5). This script now fills those two
stringProps with concrete millisecond bounds (10s connect / 120s response — generous
headroom above the ~27s max latency observed live under QEMU emulation, so a slow-but-
working backend call isn't misread as a sample error, while still failing well short of
an indefinite hang) on every sampler.

Usage: python3 adapt_jmx.py <input.jmx> <output.jmx>
"""
import re
import sys

_BLAZEMETER_BLOCK_RE = re.compile(
    r'<com\.blazemeter\.jmeter\.threads\.concurrency\.ConcurrencyThreadGroup.*?'
    r'</kg\.apc\.jmeter\.timers\.VariableThroughputTimer>\n\s*<hashTree/>\n',
    re.DOTALL,
)

_CONNECT_TIMEOUT_MS = "10000"
_RESPONSE_TIMEOUT_MS = "120000"

_CONNECT_TIMEOUT_RE = re.compile(
    r'<stringProp name="HTTPSampler\.connect_timeout"></stringProp>'
)
_RESPONSE_TIMEOUT_RE = re.compile(
    r'<stringProp name="HTTPSampler\.response_timeout"></stringProp>'
)

_STANDARD_BLOCK = """      <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="Standard Thread Group (adapted from bzm Concurrency Thread Group)" enabled="true">
        <stringProp name="ThreadGroup.on_sample_error">continue</stringProp>
        <elementProp name="ThreadGroup.main_controller" elementType="LoopController" guiclass="LoopControlPanel" testclass="LoopController" testname="Loop Controller" enabled="true">
          <boolProp name="LoopController.continue_forever">true</boolProp>
          <stringProp name="LoopController.loops">-1</stringProp>
        </elementProp>
        <stringProp name="ThreadGroup.num_threads">${thread}</stringProp>
        <stringProp name="ThreadGroup.ramp_time">${warmup}</stringProp>
        <boolProp name="ThreadGroup.scheduler">true</boolProp>
        <stringProp name="ThreadGroup.duration">${duration}</stringProp>
        <stringProp name="ThreadGroup.delay"></stringProp>
      </ThreadGroup>
      <hashTree>
        <ConstantThroughputTimer guiclass="TestBeanGUI" testclass="ConstantThroughputTimer" testname="Constant Throughput Timer (adapted, target ${rps} req/s across all threads)" enabled="true">
          <stringProp name="calcMode">2</stringProp>
          <stringProp name="throughput">${__jexl3(${rps} * 60)}</stringProp>
        </ConstantThroughputTimer>
        <hashTree/>
"""


def adapt(src: str) -> str:
    out, n = _BLAZEMETER_BLOCK_RE.subn(_STANDARD_BLOCK, src, count=1)
    if n != 1:
        sys.exit("adapt_jmx: expected exactly one BlazeMeter thread-group block to replace, "
                 f"found {n} — the source .jmx shape changed, update the regex.")

    out, n_connect = _CONNECT_TIMEOUT_RE.subn(
        f'<stringProp name="HTTPSampler.connect_timeout">{_CONNECT_TIMEOUT_MS}</stringProp>', out
    )
    out, n_response = _RESPONSE_TIMEOUT_RE.subn(
        f'<stringProp name="HTTPSampler.response_timeout">{_RESPONSE_TIMEOUT_MS}</stringProp>', out
    )
    if n_connect == 0 or n_response == 0:
        sys.exit("adapt_jmx: expected empty HTTPSampler connect/response timeout props to fill, "
                 f"found connect={n_connect} response={n_response} — the source .jmx shape changed.")
    return out


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: adapt_jmx.py <input.jmx> <output.jmx>")
    with open(sys.argv[1], encoding="utf-8") as fh:
        src = fh.read()
    adapted = adapt(src)
    with open(sys.argv[2], "w", encoding="utf-8") as fh:
        fh.write(adapted)
    print(f"wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
