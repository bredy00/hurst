"""
One entry point for everything in this project.

    python volsurf.py                    what is here and what to run
    python volsurf.py doctor             is this checkout able to run? (no network, no TWS)
    python volsurf.py health             the standing analytical checks
    python volsurf.py test [quick|full]  the test suites
    python volsurf.py demo               the live dashboard on a fake market, no TWS
    python volsurf.py pipeline           the real-data pipeline on a synthetic recording
    python volsurf.py study [name ...]   a study script, or list them
    python volsurf.py rl [section ...]   the reinforcement-learning study (off by default)
    python volsurf.py record --check     the IBKR smoke test (needs IB Gateway)

There was no single entry point before Session M: the README listed twenty commands and you
had to know which one answered your question. `doctor` in particular exists because the two
things newcomers hit -- the wrong interpreter and a missing ibapi -- both produce tracebacks
that point somewhere else entirely.
"""

import argparse
import importlib
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent
VENV = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

STUDIES = {
    "hedging": ("study_hedging.py", "discrete hedging: BS delta, hedged Monte Carlo, variance swap"),
    "lift-44": ("study_lift_44.py", "40 / 44 / 48 nodes: kernel, prices, cost, the lift against exact fGn"),
    "egarch": ("study_egarch.py", "the T-EGARCH scan against rough Heston's own integrated variance"),
    "trial": ("study_trial_bl_hurst.py", "Black-Litterman, FF4 and the dual Kalman filter on H"),
    "rl": ("study_rl.py", "the reinforcement-learning framework, graded"),
    "h-error-bars": ("study_h_error_bars.py", "the profile's robust SE on H against its true spread, 21 seeds"),
    "finer-lift": ("study_finer_lift.py", "the Session H lift decision (24 / 32 / 40)"),
    "zero-boundary": ("study_zero_boundary.py", "Kalman vs cf vs particle filter at the zero boundary"),
    "fbm": ("study_fbm_methods.py", "fBm generators compared; Shevchenko's steps checked"),
    "jump-modes": ("study_jump_modes.py", "driver vs direct rough jumps"),
    "stability": ("study_stability_constants.py", "the Riccati stability edges on a given lift"),
    "identifiability": ("study_h_identifiability.py", "what pins H, and what does not"),
    "weighting": ("study_h_weighting.py", "which quote weights pin H"),
    "cir-vs-rough": ("study_cir_vs_rough.py", "CIR against the lifted rough filter"),
    "isometry": ("study_lewis_isometry.py", "the lift's Ito isometry and the Lewis tail"),
}


def _run(args, **kw):
    """Run a subprocess with THIS interpreter, from the project root."""
    return subprocess.call([sys.executable, *args], cwd=str(ROOT), **kw)


def cmd_doctor(_):
    """Check the things that break a fresh checkout, and say what to do about each."""
    ok = True
    print(f"interpreter      {sys.executable}")
    print(f"python           {sys.version.split()[0]}")
    if sys.version_info[:2] != (3, 12):
        print("  ! the suites are verified on 3.12; other versions are untested")
    if VENV.exists() and pathlib.Path(sys.executable).resolve() != VENV.resolve():
        print(f"  ! this is not the project venv. The pinned versions are there:\n"
              f"      {VENV} volsurf.py ...")
        ok = False
    print(f"working directory {ROOT}")

    pins = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==")
            pins[name.strip()] = ver.strip()
    for name, want in pins.items():
        mod = {"pillow": "PIL", "ibapi": "ibapi", "pytest": "pytest"}.get(name, name)
        try:
            m = importlib.import_module(mod)
            got = getattr(m, "__version__", "?")
            flag = "" if str(got).startswith(want.split(".post")[0]) else f"  (pinned {want})"
            print(f"  {name:16s} {got}{flag}")
        except ImportError:
            hint = ("  -- only the IBKR recorder and the live dashboard need it; everything else runs"
                    if name == "ibapi" else "")
            print(f"  {name:16s} MISSING{hint}")
            if name != "ibapi":
                ok = False

    try:
        from threadpoolctl import threadpool_info
        threads = {i["internal_api"]: i["num_threads"] for i in threadpool_info()}
        print(f"BLAS threads     {threads}")
        if any(v > 1 for v in threads.values()):
            print("  . the Riccati step is fastest single-threaded; the calibration pins it itself,")
            print("    and OPENBLAS_NUM_THREADS=1 helps any timing you take by hand")
    except ImportError:
        print("BLAS threads     threadpoolctl missing (optional)")

    import models.rough_heston as rh
    print(f"default lift     N = {rh.N_DEFAULT} to {rh.ETA_N_DEFAULT:.0e}/y")
    err = rh.kernel_error(0.02)[0]
    print(f"  kernel error at H = 0.02 on [1 d, 2 y]: {100 * err:.3f}%  (must be under 1%)")
    ok &= err < 0.01

    import rl
    print(f"RL framework     {'ON' if rl.is_enabled() else 'off'} "
          f"({'set VOLSURF_RL=1 or call rl.enable() to turn it on' if not rl.is_enabled() else ''})")

    real = list((ROOT / "captures" / "real").glob("SPY*")) if (ROOT / "captures" / "real").exists() else []
    print(f"recorded data    {len(real)} SPY recording(s) under captures/real")
    if not real:
        print("  . none yet: `volsurf.py pipeline` runs the whole thing on a synthetic recording")
    print("\n" + ("all good" if ok else "something above needs fixing"))
    return 0 if ok else 1


def cmd_health(a):
    return _run(["healthcheck.py"] + (["--md"] if a.md else []) + (["--trend"] if a.trend else []))


def cmd_test(a):
    tier = a.tier or "quick"
    if tier == "quick":
        return _run(["-m", "pytest", "-q", "-m", "not slow"])
    if tier == "full":
        return _run(["-m", "pytest", "-q"])
    return _run(["-m", "pytest", "-q", tier])


def cmd_demo(_):
    return _run(["volatility_surface_3.py", "--demo"])


def cmd_pipeline(a):
    return _run(["run_real_data.py", "--synthetic"] + (["--quick"] if a.quick else []))


def cmd_study(a):
    if not a.name:
        width = max(len(k) for k in STUDIES)
        print("studies (python volsurf.py study <name> [args]):\n")
        for k, (f, d) in STUDIES.items():
            print(f"  {k:{width}s}  {d}")
        return 0
    name = a.name[0]
    if name not in STUDIES:
        print(f"no study called {name!r}. Known: {', '.join(sorted(STUDIES))}")
        return 2
    return _run([STUDIES[name][0], *a.name[1:]])


def cmd_rl(a):
    env = dict(os.environ, VOLSURF_RL="1")
    return _run(["study_rl.py", *a.section], env=env)


def cmd_record(a):
    return _run(["record_chains.py", *(["--check"] if a.check else a.rest)])


LICENSED_ROOT = "captures/real/"
LICENSED_EXEMPT = ("captures/real/report/", "captures/real/synthetic/")


def licensed_paths(paths):
    """The recordings among `paths`: everything the recorder writes under captures/real/
    (chains by symbol and day, session.jsonl, history_*.json), less the gitignored report
    and synthetic output, which contain no market data."""
    out = []
    for p in paths:
        q = p.replace("\\", "/")
        if q.startswith(LICENSED_ROOT) and not q.startswith(LICENSED_EXEMPT):
            out.append(q)
    return out


def repo_visibility():
    """'public', 'private', or None when it cannot be determined (no gh, no network)."""
    try:
        out = subprocess.run(["gh", "repo", "view", "--json", "visibility", "-q", ".visibility"],
                             cwd=str(ROOT), capture_output=True, text=True, timeout=20)
        v = out.stdout.strip().lower()
        return v if v in ("public", "private", "internal") else None
    except (OSError, subprocess.SubprocessError):
        return None


def cmd_guard(a):
    """
    Refuse to let licensed market data reach a public repository.

    The IBKR recordings are licensed; the recorder writes them under captures/real/, and
    .gitignore deliberately leaves them trackable so a PRIVATE repository backs them up. On a
    public repository that same rule would publish them on the first commit after the
    account goes live -- an irreversible mistake, since a pushed file is copied and cached.
    Locally this checks tracked files AND the untracked files `git add -A` would pick up;
    in CI (`--ci public|private`) it checks what the checkout holds.
    """
    vis = a.ci or repo_visibility()
    tracked = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True,
                             text=True).stdout.split()
    stageable = [] if a.ci else subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=str(ROOT),
        capture_output=True, text=True).stdout.split()
    hit_t, hit_s = licensed_paths(tracked), licensed_paths(stageable)
    print(f"repository visibility: {vis or 'unknown (gh unavailable)'}")
    print(f"recordings tracked by git: {len(hit_t)}; untracked but stageable: {len(hit_s)}")
    for p in (hit_t + hit_s)[:10]:
        print(f"  {p}")
    if vis == "public" and (hit_t or hit_s):
        print("\nREFUSED: licensed recordings would be published by this public repository. "
              "Make the repository private, or add captures/real/ to .gitignore, before committing.")
        return 1
    if vis is None and (hit_t or hit_s):
        print("\nWARNING: recordings present and the visibility could not be checked; confirm the "
              "repository is private before pushing.")
        return 0 if not a.strict else 1
    print("ok: nothing licensed can be published from here")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="volsurf.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("doctor", help="is this checkout able to run?").set_defaults(fn=cmd_doctor)
    p = sub.add_parser("health", help="the standing analytical checks")
    p.add_argument("--md", action="store_true")
    p.add_argument("--trend", action="store_true")
    p.set_defaults(fn=cmd_health)
    p = sub.add_parser("test", help="the test suites")
    p.add_argument("tier", nargs="?", help="quick (default), full, or a file")
    p.set_defaults(fn=cmd_test)
    sub.add_parser("demo", help="the live dashboard on a fake market").set_defaults(fn=cmd_demo)
    p = sub.add_parser("pipeline", help="the pipeline on a synthetic recording")
    p.add_argument("--quick", action="store_true")
    p.set_defaults(fn=cmd_pipeline)
    p = sub.add_parser("study", help="a study script, or list them")
    p.add_argument("name", nargs="*")
    p.set_defaults(fn=cmd_study)
    p = sub.add_parser("rl", help="the RL study (turns the framework on for the run)")
    p.add_argument("section", nargs="*")
    p.set_defaults(fn=cmd_rl)
    p = sub.add_parser("record", help="the IBKR recorder")
    p.add_argument("--check", action="store_true")
    p.add_argument("rest", nargs="*")
    p.set_defaults(fn=cmd_record)
    p = sub.add_parser("guard", help="refuse to publish licensed recordings from a public repo")
    p.add_argument("--ci", choices=("public", "private", "internal"), help="visibility, as CI knows it")
    p.add_argument("--strict", action="store_true", help="fail when the visibility is unknown")
    p.set_defaults(fn=cmd_guard)

    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        print("\nstart with:  python volsurf.py doctor")
        return 0
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
