"""The Session H pipeline diagram (captures/pipeline_h.png) for the A-H overview: python capture_pipeline_h.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 6.6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.6)
ax.axis("off")
C = {"you": "#fde2c8", "rec": "#dbe9f6", "store": "#eeeeee", "proc": "#e3f1df", "new": "#fff2b3", "out": "#f6d6dc"}


def box(x, y, w, h, text, kind, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=C[kind], ec="#555555", lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5,
            fontweight="bold" if bold else "normal", wrap=True)
    return (x, y, w, h)


def arrow(a, b, dy_a=0.0, dy_b=0.0):
    xa, ya, wa, ha = a
    xb, yb, wb, hb = b
    ax.add_patch(FancyArrowPatch((xa + wa, ya + ha / 2 + dy_a), (xb, yb + hb / 2 + dy_b),
                                 arrowstyle="-|>", mutation_scale=11, lw=1.0, color="#444444",
                                 connectionstyle="arc3,rad=0.0"))


ib = box(0.2, 2.9, 1.6, 0.9, "IB Gateway\n(your login)", "you", bold=True)
rc = box(2.3, 4.3, 1.9, 0.9, "record_chains.py\nOTM grid, parity\nforwards, UTC", "rec")
rhs = box(2.3, 1.4, 1.9, 0.9, "record_history.py\n5 y daily, 12 m\n5-minute bars", "rec")
snap = box(4.7, 4.3, 1.6, 0.9, "snapshots\ncaptures/real/SPY", "store")
hist = box(4.7, 1.4, 1.6, 0.9, "history\nJSON", "store")
chk = box(6.8, 5.35, 1.8, 0.8, "checks: vega units,\nparity, tau", "proc")
surf = box(6.8, 4.25, 1.8, 0.8, "surface: IV from mid,\nhybrid weights", "proc")
clk = box(6.8, 3.15, 1.8, 0.8, "trading clock:\nomega, event days", "new", bold=True)
rv = box(6.8, 1.4, 1.8, 0.8, "realised variance", "proc")
fh = box(9.1, 5.35, 1.9, 0.8, "Heston fit", "proc")
fr = box(9.1, 4.25, 1.9, 0.8, "rough Heston fit\n(either lift: --lift)", "new")
sk = box(9.1, 3.15, 1.9, 0.8, "ATM skew slope,\ncalendar + trading clock", "new")
kf = box(9.1, 1.95, 1.9, 0.8, "Kalman: CIR, lifted rough\n(RV observation)", "proc")
nz = box(9.1, 0.85, 1.9, 0.8, "zero-boundary filters\n(spot variance; RV\nobservation pending)", "new")
sf = box(9.1, -0.05 + 0.1, 1.9, 0.6, "structure function", "proc")
H = box(11.6, 2.6, 2.2, 1.3, "four readings of H\n+ Heston vs rough\nreport.md / figure", "out", bold=True)
ax.add_patch(FancyArrowPatch((rv[0] + rv[2], rv[1] + rv[3] / 2), (nz[0], nz[1] + nz[3] / 2), arrowstyle="-|>",
                             mutation_scale=11, lw=1.0, color="#888888", linestyle="--"))
for a, b in ((ib, rc), (ib, rhs), (rc, snap), (rhs, hist), (snap, chk), (snap, surf), (snap, clk), (hist, rv),
             (chk, fh), (surf, fh), (surf, fr), (surf, sk), (clk, sk), (rv, kf), (rv, sf),
             (fh, H), (fr, H), (sk, H), (kf, H), (sf, H)):
    arrow(a, b)
ax.text(0.2, 6.35, "The pipeline after Session H", fontsize=12, fontweight="bold")
for i, (k, lab) in enumerate((("you", "needs you"), ("rec", "recorder"), ("store", "recorded files"),
                              ("proc", "built A-G"), ("new", "new or extended in H"), ("out", "output"))):
    ax.add_patch(FancyBboxPatch((4.3 + i * 1.62, 6.28), 0.25, 0.2, boxstyle="round,pad=0.01", fc=C[k], ec="#555555"))
    ax.text(4.62 + i * 1.62, 6.38, lab, fontsize=7.5, va="center")
fig.tight_layout()
fig.savefig(r"C:\Projects\volatility-surface-tuning\captures\pipeline_h.png", dpi=130)
print("saved")
