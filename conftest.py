"""
pytest bridge for the hand-rolled suites (Session G).

Every suite keeps working as a script (`python test_core.py` prints each check).
Under pytest, each `test_*` function is a test item, and it FAILS if any `check()`
it ran appended to its module's FAIL list -- the failing check names become the
failure message. Nothing in the suites had to be rewritten, and the two ways of
running them cannot disagree.

    pytest                      # everything (~15 minutes)
    pytest -m "not slow"        # the quick tier (~3 minutes)
    pytest test_rough.py -k positivity
"""

import pytest

# Suites (or single tests) that take minutes; CI runs them in the full tier only.
SLOW = {
    "test_rough_calibration.py": None,          # whole module
    "test_filters.py": None,
    "test_nongaussian.py": None,                # Session H: particle / cf filters, ~4 minutes
    "test_protocol.py": {"test_rv_filter_and_protocol"},   # Session I: runs the particle filter
    "test_rough.py": {"test_monte_carlo", "test_positivity_scheme", "test_solver_robustness"},
    "test_hawkes.py": {"test_second_spike_monte_carlo", "test_kurtosis"},
    "test_calibrate.py": {"test_multistart", "test_identifiability", "test_identifiability_mechanism"},
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        fname = item.fspath.basename
        if fname in SLOW and (SLOW[fname] is None or item.name in SLOW[fname]):
            item.add_marker(pytest.mark.slow)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """The individual checks behind the test items, per suite."""
    import sys
    rows = []
    for name, mod in sorted(sys.modules.items()):
        if name.startswith("test_") and hasattr(mod, "PASS") and hasattr(mod, "FAIL"):
            rows.append((name, len(mod.PASS), len(mod.FAIL)))
    if rows:
        terminalreporter.section("checks behind the tests")
        for name, p, f in rows:
            terminalreporter.write_line(f"{name + '.py':30s} {p:4d} passed  {f:3d} failed")
        terminalreporter.write_line(f"{'total':30s} {sum(r[1] for r in rows):4d} passed  {sum(r[2] for r in rows):3d} failed")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Fail the test (not its teardown) when any of its checks failed."""
    fail = getattr(getattr(item, "module", None), "FAIL", None)
    before = len(fail) if fail is not None else 0
    result = yield
    if fail is not None and len(fail) > before:
        new = fail[before:]
        raise AssertionError(f"{len(new)} check(s) failed: " + " | ".join(new))
    return result
