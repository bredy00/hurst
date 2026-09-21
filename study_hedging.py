"""
Discrete hedging under rough Heston: Black-Scholes delta vs Hedged Monte Carlo (Session J).

A one-month at-the-money call, written at the model's cf price and hedged with the
underlying alone, rebalanced every 1.4 hours up to once a month, on 20 000 simulated
paths (the rules are fitted on another 20 000):

  bs_fixed   Black-Scholes delta at the ATM implied volatility fixed at inception
  bs_model   Black-Scholes delta at sigma_hat, the model's expected average volatility
             to maturity, re-marked at every rebalancing date
  hmc        Hedged Monte Carlo: the discrete-time risk-minimising hedge (Potters,
             Bouchaud & Sestovic 2001), fitted by backward regression

across H = 0.05, 0.12, 0.25, 0.45 at the same vol-of-vol, plus a control with nearly
deterministic variance (xi = 0.01), where Black-Scholes is the right model and the
discrete-hedging error must follow the leading-order law

    Var[error] = (dt / 2) E[ int_0^T Gamma^2 S^4 sigma^4 dt ]      (Bertsimas, Kogan & Lo 2000)

With stochastic volatility the error does not vanish as dt -> 0 (volatility risk is not
hedged by the underlying); the question is how large that floor is, how it depends on H,
and how much the risk-minimising hedge -- which uses the price-volatility correlation --
takes off it.

    python study_hedging.py          -> captures/hedging.json, captures/hedging.png (~6 minutes)
"""

import json
import math
import pathlib
import time

import numpy as np

import models.hedging as hd
import models.rough_heston as rh
import pricing.fourier as fo

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "captures" / "hedging.json"
T, N_STEPS, K = 1.0 / 12, 512, 1.0
N_TRAIN = N_TEST = 20_000
EVERY = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
RULES = ("bs_fixed", "bs_model", "hmc")
BASE = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
CONFIGS = [("H = 0.05", dict(BASE, H=0.05)), ("H = 0.12", dict(BASE, H=0.12)),
           ("H = 0.25", dict(BASE, H=0.25)), ("H = 0.45", dict(BASE, H=0.45)),
           ("control: xi = 0.01", dict(BASE, H=0.12, xi=0.01))]


def say(msg):
    print(msg, flush=True)


def bkl_std(paths, K, every):
    """Leading-order discrete-hedging std, sqrt((dt/2) E[sum Gamma^2 S^4 sigma^4 dt]), along the paths."""
    n = len(paths["t"]) - 1
    h = paths["T"] / n
    dt = every * h
    acc = np.zeros(paths["S"].shape[1])
    for k in range(n):
        tau = paths["tau"][k]
        S = paths["S"][k]
        sig = hd.model_sigma(paths, k)
        st = sig * math.sqrt(tau)
        d1 = np.log(S / K) / st + 0.5 * st
        gamma = np.exp(-0.5 * d1 * d1) / (math.sqrt(2 * math.pi) * S * st)
        acc += gamma ** 2 * S ** 4 * sig ** 4 * h
    return math.sqrt(0.5 * dt * float(np.mean(acc)))


def fit_floor(dts, sds):
    """sd^2 = floor^2 + a dt^gamma, by least squares on the log of the excess over a floor profile."""
    from scipy.optimize import curve_fit
    dts, sds = np.asarray(dts, float), np.asarray(sds, float)

    def f(dt, fl, a, g):
        return np.sqrt(fl * fl + a * dt ** g)
    try:
        (fl, a, g), cov = curve_fit(f, dts, sds, p0=(sds.min() * 0.9, sds.max() ** 2 / dts.max(), 1.0),
                                    bounds=([0.0, 0.0, 0.2], [np.inf, np.inf, 2.0]), maxfev=20000)
        se = np.sqrt(np.diag(cov))
        return {"floor": float(fl), "gamma": float(g), "gamma_se": float(se[2]), "a": float(a)}
    except (RuntimeError, ValueError):
        return {"floor": float("nan"), "gamma": float("nan"), "gamma_se": float("nan"), "a": float("nan")}


def run_config(label, pd, seed):
    t0 = time.perf_counter()
    P = rh.RoughHestonParams(**pd)
    price = float(rh.call_prices(np.array([0.0]), T, P)[0][0])
    iv = float(fo.implied_vol_from_call(price, 0.0, T))
    train = hd.simulate_paths(P, T, N_STEPS, N_TRAIN, seed=seed)
    test = hd.simulate_paths(P, T, N_STEPS, N_TEST, seed=seed + 1)
    row = {"params": pd, "price": price, "atm_iv": iv, "sigma_hat0": math.sqrt(test["FV"][0, 0] / T),
           "E_S_T": float(test["S"][-1].mean()), "intervals": []}
    h = T / N_STEPS
    for every in EVERY:
        fit = hd.hmc_fit(train, K, every)
        r = {"every": every, "dt_years": every * h, "dt_hours": every * h * 365 * 24,
             "hmc_C0": fit["C0"], "hmc_C0_se": fit["C0_se"]}
        for rule in RULES:
            e = hd.hedge_error(test, K, every, price, rule, fit=fit, sigma_fixed=iv)
            loss = -e
            r[rule] = {"mean": float(e.mean() / price), "sd": float(e.std() / price),
                       "q99_loss": float(np.quantile(loss, 0.99) / price),
                       "mean_se": float(e.std() / math.sqrt(len(e)) / price)}
        if "control" in label:
            r["bkl_sd"] = bkl_std(test, K, every) / price
        row["intervals"].append(r)
    for rule in RULES:
        row[f"fit_{rule}"] = fit_floor([r["dt_years"] for r in row["intervals"][:7]],
                                       [r[rule]["sd"] for r in row["intervals"][:7]])
    fin = row["intervals"][0]
    daily = next(r for r in row["intervals"] if r["every"] == 16)
    say(f"{label}: cf price {price:.5f} (ATM IV {iv:.4f}); HMC price {fin['hmc_C0']:.5f} +/- {fin['hmc_C0_se']:.5f}; "
        f"residual sd / price at 1.4 h: BS fixed {fin['bs_fixed']['sd']:.3f}, BS model {fin['bs_model']['sd']:.3f}, "
        f"HMC {fin['hmc']['sd']:.3f}; daily: {daily['bs_fixed']['sd']:.3f} / {daily['bs_model']['sd']:.3f} / "
        f"{daily['hmc']['sd']:.3f}; floor fit (HMC) {row['fit_hmc']['floor']:.3f}, exponent "
        f"{row['fit_hmc']['gamma']:.2f} +/- {row['fit_hmc']['gamma_se']:.2f}  ({time.perf_counter() - t0:.0f}s)")
    if "control" in label:
        say("   control, BS delta vs the Bertsimas-Kogan-Lo law: " + "  ".join(
            f"{r['dt_hours']:.1f} h: {r['bs_model']['sd']:.4f} vs {r['bkl_sd']:.4f}" for r in row["intervals"][:6]))
    return row


def main():
    res = {"T": T, "n_steps": N_STEPS, "K": K, "n_train": N_TRAIN, "n_test": N_TEST, "configs": {}}
    for i, (label, pd) in enumerate(CONFIGS):
        res["configs"][label] = run_config(label, pd, seed=100 + 10 * i)
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    plot(res)
    return res


def plot(res=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
    from ui import theme as th
    res = res or json.loads(OUT.read_text(encoding="utf-8"))
    t = th.apply("light")
    # fixed palette order; slot 3 (aqua) is below 3:1 on the light surface, so its lines are labelled directly
    col = {"hmc": th.series(0), "bs_fixed": th.series(1), "bs_model": th.series(2)}
    lab = {"bs_fixed": "BS delta, implied vol fixed at inception", "bs_model": "BS delta, model vol re-marked",
           "hmc": "Hedged Monte Carlo (risk-minimising)"}
    short = {"bs_fixed": "BS fixed", "bs_model": "BS re-marked", "hmc": "HMC"}
    fig = plt.figure(figsize=(17, 5.8))
    gs = fig.add_gridspec(1, 4, wspace=0.34)
    fig.text(0.010, 0.975, "Discrete hedging under rough Heston: a one-month at-the-money call", fontsize=14,
             fontweight="semibold", color=t["ink"], va="top")
    fig.text(0.010, 0.925, f"{res['n_test']:,} test paths; rules fitted on another {res['n_train']:,}; "
             "residual = sd of the writer's terminal hedging error / option price; xi = 0.3, rho = -0.7",
             fontsize=9.5, color=t["ink2"], va="top")
    labels = [k for k in res["configs"] if not k.startswith("control")]
    Hs = np.array([res["configs"][k]["params"]["H"] for k in labels])

    a = fig.add_subplot(gs[0, 0])
    main_ = res["configs"]["H = 0.12"]
    ctrl = res["configs"]["control: xi = 0.01"]
    hrs = np.array([r["dt_hours"] for r in main_["intervals"]])
    for rule in ("hmc", "bs_fixed", "bs_model"):
        ys = [r[rule]["sd"] for r in main_["intervals"]]
        a.loglog(hrs, ys, "o-", color=col[rule], lw=2, ms=5, mec=t["surface"], label=lab[rule])
    a.loglog(hrs, [r["bs_model"]["sd"] for r in ctrl["intervals"]], "s--", color=t["ink2"], lw=1.4, ms=4,
             label="control (xi = 0.01): BS delta")
    a.loglog(hrs, [r["bkl_sd"] for r in ctrl["intervals"]], ":", color=t["ink"], lw=1.4,
             label="Bertsimas-Kogan-Lo law, sqrt(dt)")
    # labels in the lines' own vertical order: re-marked runs above fixed, HMC below both
    for rule, dy in (("hmc", 0.93), ("bs_fixed", 0.965), ("bs_model", 1.075)):
        a.text(hrs[0] * 0.9, main_["intervals"][0][rule]["sd"] * dy, short[rule], color=t["ink2"], fontsize=7.5,
               ha="right", va="center")
    from matplotlib.ticker import FuncFormatter
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(NullFormatter())
    a.set_xlim(hrs[0] * 0.2, hrs[-1] * 1.3)
    a.set_xlabel("rebalancing interval (hours)")
    a.set_ylabel("residual sd / price")
    a.set_title("(a) H = 0.12: rebalancing cannot remove the vol risk")
    a.legend(loc="lower right", fontsize=7)

    b = fig.add_subplot(gs[0, 1])
    for rule in ("hmc", "bs_fixed", "bs_model"):
        ys = [res["configs"][k]["intervals"][0][rule]["sd"] for k in labels]
        b.plot(Hs, ys, "o-", color=col[rule], lw=2, ms=6, mec=t["surface"], label=lab[rule])
    b.text(Hs[0] + 0.01, res["configs"][labels[0]]["intervals"][0]["bs_model"]["sd"], " BS re-marked", color=t["ink2"],
           fontsize=7.5, va="bottom")
    b.set_xlabel("H")
    b.set_ylabel("residual sd / price, 1.4-hour rebalancing")
    b.set_title("(b) the floor rises as H falls")
    b.legend(loc="upper right", fontsize=7)

    c = fig.add_subplot(gs[0, 2])
    for rule in ("hmc", "bs_fixed", "bs_model"):
        g = [res["configs"][k][f"fit_{rule}"]["gamma"] for k in labels]
        e = [res["configs"][k][f"fit_{rule}"]["gamma_se"] for k in labels]
        c.errorbar(Hs, g, yerr=e, fmt="o-", color=col[rule], lw=2, ms=6, mec=t["surface"], capsize=3,
                   ecolor=t["muted"], label=lab[rule])
    c.axhline(1.0, color=t["ink2"], ls="--", lw=1.2)
    c.text(0.46, 1.005, "smooth vol: 1", color=t["ink2"], fontsize=7.5, ha="right", va="bottom")
    hh = np.linspace(0.03, 0.5, 50)
    c.plot(hh, 2 * hh, ":", color=t["ink2"], lw=1.2)
    c.text(0.36, 0.60, "2H: pure rough\nvanna term", color=t["ink2"], fontsize=7.5, ha="left", va="top")
    c.set_ylim(0.0, 1.12)
    c.set_xlabel("H")
    c.set_ylabel("exponent gamma in sd^2 = floor^2 + a dt^gamma")
    c.set_title("(c) roughness slows the approach to the limit")
    c.legend(loc="lower right", fontsize=7)

    d = fig.add_subplot(gs[0, 3])
    for rule in ("hmc", "bs_fixed"):
        vals = [res["configs"][k]["intervals"][0][rule]["q99_loss"] for k in labels]
        d.plot(Hs, vals, "o-", color=col[rule], lw=2, ms=6, mec=t["surface"], label=lab[rule])
    d.set_xlabel("H")
    d.set_ylabel("99% loss quantile / price, 1.4 h")
    d.set_title("(d) the writer's tail")
    d.legend(loc="upper right", fontsize=7)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.83, bottom=0.12)
    fig.savefig(ROOT / "captures" / "hedging.png", dpi=120)


if __name__ == "__main__":
    main()
