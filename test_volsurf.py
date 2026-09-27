"""
Session N -- the entry point (volsurf.py) and the licensed-data guard.

  guard     what counts as a licensed recording (everything the recorder writes under
            captures/real/, not the gitignored report and synthetic output), and that the
            current tree tracks none of it
  cli       every subcommand is registered and `study` lists every study script that exists

    python test_volsurf.py
"""

import pathlib
import sys

import volsurf

PASS, FAIL = [], []
ROOT = pathlib.Path(__file__).parent


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def test_guard():
    print("\nthe licensed-data guard")
    cases = {
        "captures/real/SPY/2026-10-01/SPY_2026-10-01_10-00-00ET.json": True,
        "captures/real/SPY/2026-10-01/session.jsonl": True,
        "captures/real/history_SPY_2026-10-01.json": True,
        "captures\\real\\SPY\\x.json": True,
        "captures/real/report/report.json": False,
        "captures/real/synthetic/SPY_x.json": False,
        "captures/rl.json": False,
        "models/har.py": False,
    }
    got = {p: bool(volsurf.licensed_paths([p])) for p in cases}
    wrong = [p for p in cases if got[p] != cases[p]]
    check("recordings are recognised, and report / synthetic output is not", not wrong,
          f"misclassified: {wrong}" if wrong else f"{len(cases)} paths")
    rc = volsurf.main(["guard", "--ci", "public"])
    check("the tree as committed tracks no recording, so the guard passes even as public", rc == 0)


def test_cli():
    print("\nthe entry point")
    missing = [f for f, _ in volsurf.STUDIES.values() if not (ROOT / f).exists()]
    check("every study the CLI lists exists", not missing, f"missing: {missing}" if missing else
          f"{len(volsurf.STUDIES)} studies")
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        volsurf.main([])
    text = buf.getvalue()
    subs = ("doctor", "health", "test", "demo", "pipeline", "study", "rl", "record", "guard")
    check("every subcommand is registered and listed", all(s_ in text for s_ in subs),
          ", ".join(s_ for s_ in subs if s_ not in text) or "all present")


if __name__ == "__main__":
    print("=" * 74)
    print("Session N -- volsurf.py and the licensed-data guard")
    print("=" * 74)
    test_guard()
    test_cli()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
