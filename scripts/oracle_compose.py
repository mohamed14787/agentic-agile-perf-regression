#!/usr/bin/env python3
"""oracle_compose.py — generate a per-workspace docker-compose override for the system oracle.

CRITICAL FIX vs. the shipped compose file: `targets/TeaStore/examples/docker/
docker-compose_default.yaml` uses `image: descartesresearch/teastore-*` — PREBUILT images
pulled from Docker Hub, not built from the workspace's own source. Load-testing that file
as-is would exercise the stock upstream TeaStore, not our senior/junior generated trees,
silently invalidating every oracle row. Each workspace module already ships its own
`Dockerfile` (confirmed: `<workspace>/services/*/Dockerfile`, `.../utilities/*/Dockerfile` —
each just COPYs its own `target/*.war` onto a shared `descartesresearch/teastore-base` tomcat
image), so this generator replaces every app service's `image:` with `build: {context:
<workspace>/<module-dir>}`, keeping `db` (no application source, a seeded MySQL image) and
every other key (expose/ports/environment) byte-identical to the shipped file.

Usage (library):
    from oracle_compose import generate
    generate(workspace_dir, out_path, cpu_limit="1.0")

CLI:
    python3 scripts/oracle_compose.py <workspace_dir> <out_path> [--cpu-limit 1.0]
"""
import os
import sys

import yaml

# service name -> module dir (relative to workspace root). Fixed for TeaStore.
# `db` has no entry: no source, image stays as shipped.
_SERVICE_MODULE = {
    "registry": "services/tools.descartes.teastore.registry",
    "persistence": "services/tools.descartes.teastore.persistence",
    "auth": "services/tools.descartes.teastore.auth",
    "image": "services/tools.descartes.teastore.image",
    "recommender": "services/tools.descartes.teastore.recommender",
    "webui": "services/tools.descartes.teastore.webui",
}

_BASE_COMPOSE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "targets", "TeaStore", "examples", "docker", "docker-compose_default.yaml")

# Container command that starts Tomcat WITHOUT Kieker/AspectJ instrumentation.
#
# WHY THIS IS NEEDED. `descartesresearch/teastore-base` runs
#   sh -c "java -jar dockermemoryconfigurator.jar $TOMCAT_HEAP_MEM_PERCENTAGE; \
#          start.sh && catalina.sh run"
# and `start.sh` ends by WRITING /usr/local/tomcat/bin/setenv.sh with
#   JAVA_OPTS="-javaagent:/kieker/agent/agent.jar ... \
#              -Dorg.aspectj.weaver.loadtime.configuration=aop.xml ..."
# unconditionally — there is no environment toggle. `start.sh` does separately set
# `kieker.monitoring.enabled=false` when no writer is configured, which stops records being
# WRITTEN, but the agent still loads and AspectJ still weaves every class that Tomcat, Jersey
# and the WAR load. All of the cost, none of the data.
#
# Two independent reasons that is wrong for the oracle:
#   1. CORRECTNESS. The oracle measures end-to-end response time. Load-time-weaving every
#      method of the application under measurement inflates exactly the quantity being
#      measured, and does so unevenly across services.
#   2. FEASIBILITY. Measured live on this machine (amd64 images under aarch64 emulation, each
#      service capped at 1.0 CPU per the paper's restriction): auth, image and recommender sat
#      pinned at 100% CPU for 20+ minutes without finishing WAR deployment, so they never
#      registered and webui served HTTP 500 ("No Server registered for Service") forever.
#
# Since setenv.sh is written by start.sh, it can only be neutralised BETWEEN start.sh and
# catalina.sh — hence overriding the command rather than passing JAVA_OPTS through the
# environment, which setenv.sh would simply overwrite.
#
# WHY THE AGENT IS KEPT AND ONLY WEAVING IS DISABLED. Dropping `-javaagent` outright was tried
# first and breaks the application: the agent jar is what puts slf4j-api on the classpath, so
# without it every service dies with
#     NoClassDefFoundError: org/slf4j/spi/LoggerFactoryBinder
#     -> Context [/tools.descartes.teastore.webui] startup failed due to previous errors
# (`/usr/local/slf4j-simple-1.7.21.jar` on the classpath is only the binding, not the API).
# So the agent stays, and instead AspectJ is told to skip the Tomcat webapp class loaders —
# which is where essentially all of the application's classes live, and therefore where all of
# the weaving cost is. The JVM environment is otherwise byte-identical to a normal TeaStore
# run, so the measured system is still the same system.
#
# `$$` escapes compose's own variable interpolation so the shell, not compose, expands it.
_SKIP_LOADERS = ("java.net.URLClassLoader"
                 ",org.apache.catalina.loader.ParallelWebappClassLoader"
                 ",org.apache.catalina.loader.WebappClassLoaderBase")

_NO_INSTRUMENTATION_CMD = [
    "/bin/sh", "-c",
    "java -jar /usr/local/tomcat/bin/dockermemoryconfigurator.jar "
    "$${TOMCAT_HEAP_MEM_PERCENTAGE}; "
    "/usr/local/tomcat/bin/start.sh && "
    "sed -i 's#-Daj.weaving.loadersToSkip=java.net.URLClassLoader#"
    f"-Daj.weaving.loadersToSkip={_SKIP_LOADERS}#' "
    "/usr/local/tomcat/bin/setenv.sh && "
    "exec /usr/local/tomcat/bin/catalina.sh run",
]


# Host port the persistence service is published on so the load generator can drive the
# feature endpoints directly. 8080 is taken by webui; 8081 is free on this machine and is
# what run_oracle.sh / the JMeter plan default to.
PERSISTENCE_HOST_PORT = int(os.getenv("ORACLE_PERSISTENCE_PORT", "8081"))


def _module_buildable(workspace_dir: str, module_rel: str) -> bool:
    return os.path.isfile(os.path.join(workspace_dir, module_rel, "Dockerfile"))


def generate(workspace_dir: str, out_path: str, cpu_limit: str = "1.0",
             base_compose: str = _BASE_COMPOSE,
             disable_instrumentation: bool = True) -> dict:
    """Read the shipped compose file as a template and emit a build-from-source override for
    `workspace_dir` at `out_path`. Returns the generated dict (also written to disk). Raises
    if a service the template declares has no corresponding on-disk Dockerfile (fail loud,
    not a silently-broken build)."""
    with open(base_compose, encoding="utf-8") as fh:
        compose = yaml.safe_load(fh)

    # The shipped compose file declares NO dependency ordering on the database. TeaStore's
    # persistence service resolves its datasource once at startup and silently falls back to an
    # embedded H2 URL when MySQL is unreachable — but the container ships only the MySQL
    # connector, so the fallback dies with
    #     Configuration error. Class [org.h2.Driver] not found
    # in an unrecoverable retry loop. Whether that happens is pure timing: `descartesresearch/
    # teastore-db` is an amd64 image, so under emulation a first-run MariaDB initialisation
    # ("Initializing database files") can outlast the application's startup. Observed live —
    # copurchase's senior track came up fine and its junior track failed this way on the very
    # next run of the same script.
    #
    # A healthcheck plus `depends_on: condition: service_healthy` makes the ordering explicit
    # instead of leaving it to chance.
    db = compose.get("services", {}).get("db")
    if db is not None:
        # The probe must test THE APPLICATION'S OWN CONNECTION, not merely that mysqld answers.
        # Two weaker probes were tried and both failed, in opposite directions:
        #   * `mysqladmin ping` with no credentials — permanently rejected ("Access denied for
        #     user 'root'@'127.0.0.1' (using password: NO)"), so db never went healthy and
        #     persistence never started at all.
        #   * ...accepting "Access denied" as proof of life — goes healthy as soon as mysqld
        #     answers, which is BEFORE the `teadb` schema and `teauser` grant are seeded, so
        #     persistence still connected too early and still fell back to H2.
        # The credentials below are the ones persistence.xml actually uses
        # (jdbc:mysql://.../teadb, teauser/teapassword), so a successful SELECT here means the
        # exact connection the application is about to make will succeed.
        db.setdefault("healthcheck", {
            "test": ["CMD-SHELL",
                     "mysql -h 127.0.0.1 -u teauser -pteapassword "
                     "-e 'SELECT 1' teadb >/dev/null 2>&1"],
            "interval": "5s",
            "timeout": "5s",
            "retries": 90,
            "start_period": "30s",
        })

    for name, svc in compose.get("services", {}).items():
        module_rel = _SERVICE_MODULE.get(name)
        if module_rel is None:
            continue  # e.g. db — no source to build, keep the shipped image: as-is
        if name == "persistence":
            if db is not None:
                svc["depends_on"] = {"db": {"condition": "service_healthy"}}
            # Publish persistence on the host so the load generator can call the feature
            # endpoints. Every chained-history ticket says "work only within the persistence
            # service", so the feature IS a persistence REST endpoint and the webui has no
            # page that reaches it. The shipped compose file publishes only webui (8080) and
            # db (3306), which is why the oracle's workload could reach nothing but the stock
            # browse path — measuring an unmodified code path and reporting "no regression"
            # for four consecutive rows. Publishing a port adds no CPU or memory pressure and
            # leaves the service-to-service topology untouched.
            svc.setdefault("ports", [])
            mapping = f"{PERSISTENCE_HOST_PORT}:8080"
            if mapping not in svc["ports"]:
                svc["ports"].append(mapping)
        if not _module_buildable(workspace_dir, module_rel):
            raise FileNotFoundError(
                f"oracle_compose: service '{name}' expects a Dockerfile at "
                f"{os.path.join(workspace_dir, module_rel)} but none exists — is this a "
                f"TeaStore workspace with all modules staged?")
        svc.pop("image", None)
        # Absolute path: Docker Compose resolves a RELATIVE build.context against the
        # compose FILE's own directory, not the caller's cwd — since the generated file is
        # written inside `workspace_dir` itself (see run_oracle.sh), a relative
        # "<workspace_dir>/<module>" path double-prefixes to
        # "<workspace_dir>/<workspace_dir>/<module>" (confirmed via a live run: "unable to
        # prepare context: path '.../ws/ws/services/...' not found"). Absolute sidesteps
        # the ambiguity regardless of where the compose file ends up living.
        svc["build"] = {"context": os.path.abspath(os.path.join(workspace_dir, module_rel))}
        svc.setdefault("deploy", {}).setdefault("resources", {}).setdefault(
            "limits", {})["cpus"] = cpu_limit
        if disable_instrumentation:
            svc["command"] = list(_NO_INSTRUMENTATION_CMD)

    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(compose, fh, sort_keys=False)
    return compose


def main() -> None:
    args = sys.argv[1:]
    cpu_limit = "1.0"
    pos = []
    i = 0
    while i < len(args):
        if args[i] == "--cpu-limit":
            cpu_limit = args[i + 1]; i += 2
        else:
            pos.append(args[i]); i += 1
    if len(pos) != 2:
        sys.exit("usage: oracle_compose.py <workspace_dir> <out_path> [--cpu-limit 1.0]")
    generate(pos[0], pos[1], cpu_limit=cpu_limit)
    print(f"wrote {pos[1]}")


if __name__ == "__main__":
    main()
