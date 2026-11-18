import matplotlib.pyplot as plt
from batch_data import load_epoch_table

df = load_epoch_table(sleep_only=False)
for (s, n), g in list(df.groupby(["subject", "night"]))[:1]:
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.step(g.epoch_idx * 0.5 / 60, g.stage, where="post")
    sp = g[g.in_sleep_period]
    ax.axvspan(sp.epoch_idx.min() * 0.5 / 60, sp.epoch_idx.max() * 0.5 / 60, alpha=0.15)
    ev = g[g.awakening]
    ax.plot(ev.epoch_idx * 0.5 / 60, ev.stage, "rv")
    ax.set_title(f"subject {s}, night {n}")
    ax.set_ylabel("sleep stage")
    ax.set_yticks(range(5))
    ax.set_yticklabels(["W", "N1", "N2", "N3", "REM"])
    ax.set_xlabel("time (hours)")
    ax.invert_yaxis()
plt.show()