"""Runtime smoke gate (plan section C): boot the Spring Boot app on a free port and
probe GET routes for 5xx responses. Catches the bug classes that `mvn test` misses —
missing DB tables (ddl-auto=none + no CREATE TABLE) and dead computations (a value
that's "computed" but never recalculated) — without running the model.

Hard wall-clock deadlines everywhere: a deep SSL/socket read can block forever
without one (see the FastAPI TLS-hang lesson), so readiness polling and the overall
probe loop both have a `time.monotonic()`-based budget, and the process group is
always torn down in `finally`.
"""
import os
import re
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from agile_team_cline.repair import STYLE_SKIP_FLAGS

_STARTUP_TIMEOUT = int(os.getenv("AGILE_SMOKE_STARTUP_TIMEOUT", "180"))
_OVERALL_TIMEOUT = int(os.getenv("AGILE_SMOKE_TIMEOUT", "300"))


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _kill(proc) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass


def run_smoke(output_dir: str, paths: list) -> dict:
    """Boot `mvn spring-boot:run` on a free port, wait for readiness with a hard
    wall-clock deadline, GET-probe each entry of `paths` ('{...}' path-vars -> '1'),
    and report any 5xx. Always tears down the process group. Never raises."""
    result = {"ran": False, "probed": 0, "failed": [], "boot_error": None}
    if os.getenv("AGILE_BUILD_MODULE", "").strip():
        return result  # non-Spring/reactor: compile gate is the signal, no boot probe
    if not os.path.exists(os.path.join(output_dir, "pom.xml")):
        return result

    port = _free_port()
    log = deque(maxlen=400)
    proc = None
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            ["mvn", "spring-boot:run", *STYLE_SKIP_FLAGS,
             f"-Dspring-boot.run.jvmArguments=-Dserver.port={port} -Dserver.address=127.0.0.1"],
            cwd=output_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )

        def _drain():
            try:
                for line in proc.stdout:
                    log.append(line)
            except Exception:
                pass

        threading.Thread(target=_drain, daemon=True).start()

        base = f"http://127.0.0.1:{port}"
        ready = False
        while time.monotonic() - start < _STARTUP_TIMEOUT:
            if proc.poll() is not None:
                result["boot_error"] = "process exited during startup:\n" + "".join(log)[-2000:]
                return result
            try:
                urlopen(base + "/", timeout=5)
                ready = True
            except HTTPError:
                ready = True  # any HTTP status (incl. 404) means the server is up
            except (URLError, OSError):
                pass
            if ready:
                break
            time.sleep(2)
        if not ready:
            result["boot_error"] = (f"app did not become ready within {_STARTUP_TIMEOUT}s:\n"
                                     + "".join(log)[-2000:])
            return result

        result["ran"] = True
        seen = set()
        for path in paths:
            if time.monotonic() - start > _OVERALL_TIMEOUT:
                break
            if not path or path in seen:
                continue
            seen.add(path)
            url = base + re.sub(r"\{[^}]*\}", "1", path)
            result["probed"] += 1
            try:
                urlopen(url, timeout=10)
            except HTTPError as e:
                if e.code >= 500:
                    try:
                        detail = e.read(200).decode("utf-8", "replace")
                    except Exception:
                        detail = ""
                    result["failed"].append({"method": "GET", "path": path,
                                              "status": e.code, "detail": detail})
            except URLError:
                continue
    except Exception as e:
        result["boot_error"] = f"smoke gate error: {e}"
    finally:
        if proc is not None:
            _kill(proc)
    return result
