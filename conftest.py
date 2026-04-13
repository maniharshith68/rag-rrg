"""
conftest.py

Pytest plugin — place at project root: rag-rrg/conftest.py

Automatically loaded by pytest before every test run, regardless of which
command is used:
    python3 -m pytest tests/unit/ -v
    python3 -m pytest tests/integration/ -v
    python3 -m pytest tests/ -v
    python3 -m pytest tests/integration/ -v -s

What it does:
    1. Creates a timestamped log file in logs/ for every test run
    2. Writes the full test session output (header, each result, warnings, summary)
    3. Prints the log file path at the end of every run so you always know where it is

Log file naming:
    logs/test_{scope}_{YYYYMMDD_HHMMSS}.log

    scope is inferred from the paths pytest collected:
        unit          → only tests/unit/ was run
        integration   → only tests/integration/ was run
        all           → both or full tests/ was run
        custom        → any other combination

No changes to test files needed. No aliases needed. No extra flags needed.
Just run pytest normally and logs appear automatically.

Compatible with pytest 9.0.x — pytest_warning_recorded uses (warning_message, when, nodeid)
only, as the filename parameter was removed in pytest 8+.
"""

from pathlib import Path
from datetime import datetime
import pytest


# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR      = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# SCOPE DETECTION
# ─────────────────────────────────────────────

def _infer_scope(config) -> str:
    """
    Infer a human-readable scope label from the paths pytest was invoked with.
    Used to name the log file meaningfully.
    """
    args = [str(a) for a in config.args]

    has_unit        = any("unit"        in a for a in args)
    has_integration = any("integration" in a for a in args)
    has_tests       = any(
        a in ("tests", "tests/", str(PROJECT_ROOT / "tests"))
        for a in args
    )

    if has_unit and not has_integration:
        return "unit"
    if has_integration and not has_unit:
        return "integration"
    if has_unit and has_integration:
        return "all"
    if has_tests or not args:
        return "all"
    return "custom"


# ─────────────────────────────────────────────
# LOG FILE HANDLER
# ─────────────────────────────────────────────

class _SessionLogWriter:
    """
    Writes structured test session output to a log file.
    Holds the file handle open for the duration of the pytest session.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._fh      = open(log_path, "w", buffering=1, encoding="utf-8")
        self._passed  = 0
        self._failed  = 0
        self._skipped = 0
        self._errors  = 0

    def write(self, text: str):
        self._fh.write(text)
        self._fh.flush()

    def close(self):
        self._fh.close()

    def increment(self, outcome: str):
        if outcome == "passed":
            self._passed  += 1
        elif outcome == "failed":
            self._failed  += 1
        elif outcome == "skipped":
            self._skipped += 1
        else:
            self._errors  += 1

    @property
    def totals(self):
        return self._passed, self._failed, self._skipped, self._errors


# ─────────────────────────────────────────────
# PYTEST HOOKS
# ─────────────────────────────────────────────

def pytest_configure(config):
    """Called after command-line options have been parsed and all plugins loaded."""
    scope    = _infer_scope(config)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"test_{scope}_{ts}.log"
    log_path = LOG_DIR / log_name

    writer = _SessionLogWriter(log_path)
    config._session_log_writer = writer

    writer.write(f"{'=' * 70}\n")
    writer.write(f"pytest session — {scope.upper()} tests\n")
    writer.write(f"Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    writer.write(f"Log file : {log_path}\n")
    writer.write(
        f"Command  : python3 -m pytest "
        f"{' '.join(str(a) for a in config.args)}\n"
    )
    writer.write(f"{'=' * 70}\n\n")


def pytest_collection_finish(session):
    """Called after collection — log how many tests were collected."""
    writer = getattr(session.config, "_session_log_writer", None)
    if writer is None:
        return
    writer.write(f"Collected {len(session.items)} item(s)\n\n")


def pytest_sessionstart(session):
    """Store writer reference on the pytest module so logreport hook can reach it."""
    writer = getattr(session.config, "_session_log_writer", None)
    if writer:
        pytest._current_session_log_writer = writer


def pytest_runtest_logreport(report):
    """Called after each test phase. Log the result of the call phase."""
    # Log the call phase (actual test body).
    # Also log setup/teardown if they failed, so errors are not silent.
    if report.when == "call":
        pass
    elif report.when in ("setup", "teardown") and report.failed:
        pass
    else:
        return

    writer = getattr(pytest, "_current_session_log_writer", None)
    if writer is None:
        return

    outcome = report.outcome   # "passed" / "failed" / "skipped"
    writer.increment(outcome)

    symbol = {
        "passed" : "PASSED ",
        "failed" : "FAILED ",
        "skipped": "SKIPPED",
    }.get(outcome, outcome.upper())

    writer.write(f"{symbol}  {report.nodeid}\n")

    # Write full traceback for failures
    if report.failed and report.longrepr:
        writer.write("\n")
        writer.write("  " + str(report.longrepr).replace("\n", "\n  "))
        writer.write("\n\n")

    # Write skip reason
    if report.skipped:
        if hasattr(report, "wasxfail"):
            writer.write(f"  Reason: {report.wasxfail}\n")
        elif (
            report.longrepr
            and isinstance(report.longrepr, tuple)
            and len(report.longrepr) == 3
        ):
            writer.write(f"  Reason: {report.longrepr[2]}\n")


def pytest_sessionfinish(session, exitstatus):
    """Write session summary and close the log file."""
    writer = getattr(session.config, "_session_log_writer", None)
    if writer is None:
        return

    passed, failed, skipped, errors = writer.totals
    total = passed + failed + skipped + errors

    writer.write(f"\n{'─' * 70}\n")
    writer.write(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    writer.write(f"Results  : {total} tests\n")
    writer.write(f"  PASSED : {passed}\n")
    if failed:
        writer.write(f"  FAILED : {failed}\n")
    if skipped:
        writer.write(f"  SKIPPED: {skipped}\n")
    if errors:
        writer.write(f"  ERRORS : {errors}\n")

    status_map = {
        0: "OK — all tests passed",
        1: "TESTS FAILED",
        2: "INTERRUPTED",
        3: "INTERNAL ERROR",
        4: "USAGE ERROR",
        5: "NO TESTS COLLECTED",
    }
    status_label = status_map.get(int(exitstatus), f"EXIT CODE {exitstatus}")
    writer.write(f"Status   : {status_label}\n")
    writer.write(f"{'=' * 70}\n")

    log_path = writer.log_path
    writer.close()

    print(f"\n[pytest] Test log saved → {log_path}")


def pytest_warning_recorded(warning_message, when, nodeid):
    """
    Log warnings into the session log file.

    Note: 'filename' parameter was removed from this hook in pytest 8+.
    This signature is compatible with pytest 9.0.x.
    """
    writer = getattr(pytest, "_current_session_log_writer", None)
    if writer is None:
        return

    msg = str(warning_message.message)

    # Log each unique warning only once to avoid FAISS SWIG spam
    if not hasattr(pytest, "_logged_warnings"):
        pytest._logged_warnings = set()
    if msg not in pytest._logged_warnings:
        pytest._logged_warnings.add(msg)
        location = nodeid or "session"
        writer.write(f"WARNING  {location}: {msg}\n")


def pytest_unconfigure(config):
    """Clean up global state attached to the pytest module."""
    if hasattr(pytest, "_current_session_log_writer"):
        del pytest._current_session_log_writer
    if hasattr(pytest, "_logged_warnings"):
        del pytest._logged_warnings
