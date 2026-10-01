"""
Session N -- the entry point (volsurf.py) and the licensed-data guard.

  guard     what counts as a licensed recording (everything the recorder writes under
            captures/real/, not the gitignored report and synthetic output), and that the
            current tree tracks none of it
  hook      (Session O) the guard as a pre-commit hook: a real commit in a throwaway repository
            is refused while "public", allowed while "private", and a foreign hook is left alone
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


def test_hook():
    """Session O: the guard as a pre-commit hook, exercised on a real commit in a temp repo."""
    print("\nthe pre-commit hook, in a throwaway repository")
    import os
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        repo = pathlib.Path(d)

        def git(*args, vis=None):
            env = dict(os.environ)
            if vis:
                env["VOLSURF_VISIBILITY"] = vis
            return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
                                   "-c", "commit.gpgsign=false", *args], cwd=d, capture_output=True,
                                  text=True, env=env)

        git("init", "-q")
        rc = volsurf.main(["guard", "--install-hook", "--repo", d])
        hook = repo / ".git" / "hooks" / "pre-commit"
        check("the hook installs, and says whose it is", rc == 0 and hook.exists()
              and volsurf.HOOK_MARK in hook.read_text(encoding="utf-8"))
        rec = repo / "captures" / "real" / "SPY" / "2026-10-01"
        rec.mkdir(parents=True)
        (rec / "SPY_2026-10-01_10-00-00ET.json").write_text("{}", encoding="utf-8")
        (repo / "notes.txt").write_text("x", encoding="utf-8")
        git("add", "notes.txt")
        r = git("commit", "-q", "-m", "notes", vis="public")
        check("an ordinary commit passes, public or not", r.returncode == 0, (r.stdout + r.stderr).strip()[:80])
        git("add", "captures")
        r = git("commit", "-q", "-m", "a recording", vis="public")
        n = git("rev-list", "--count", "HEAD").stdout.strip()
        check("a commit carrying a recording is REFUSED while the repository is public",
              r.returncode != 0 and n == "1" and "REFUSED" in r.stdout + r.stderr, f"exit {r.returncode}, {n} commit(s)")
        r = git("commit", "-q", "-m", "a recording", vis="private")
        n = git("rev-list", "--count", "HEAD").stdout.strip()
        check("...and allowed while it is private", r.returncode == 0 and n == "2", f"exit {r.returncode}, {n} commit(s)")
        volsurf.main(["guard", "--uninstall-hook", "--repo", d])
        check("uninstalling removes it", not hook.exists())
        hook.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        rc = volsurf.main(["guard", "--install-hook", "--repo", d])
        check("a hook that is not the guard's is left alone", rc == 1 and hook.read_text(encoding="utf-8").endswith("exit 0\n"))


if __name__ == "__main__":
    print("=" * 74)
    print("Session N -- volsurf.py and the licensed-data guard")
    print("=" * 74)
    test_guard()
    test_hook()
    test_cli()
    print("\n" + "=" * 74)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
