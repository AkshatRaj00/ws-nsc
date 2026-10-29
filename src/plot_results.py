import json, os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 11, "lines.linewidth": 2.0})

with open("data/results.json", "r") as f:
    data = json.load(f)

records, summary = data["records"], data["summary"]
tasks, anchor = data["args"]["tasks"], data["args"]["anchor"]
stages_labels = ["Init"] + [f"Post-{t}" for t in tasks]
os.makedirs("paper_figures", exist_ok=True)

# Figure 1: Loss curves
fig1, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for idx, key in enumerate([anchor] + tasks):
    ax = axes[idx]
    for r in records:
        traj = [r["init"][key]] + [s["post"][key] for s in r["stages"]]
        ax.plot(range(len(stages_labels)), traj, marker="o", label=r["label"])
    ax.set_title(f"Eval: {key.upper()}", fontweight="bold")
    ax.set_xticks(range(len(stages_labels)))
    ax.set_xticklabels(stages_labels, rotation=25)
    ax.grid(True, linestyle="--", alpha=0.4)
    if idx == 0:
        ax.set_ylabel("Cross-Entropy Loss (nats)")
        ax.legend()

fig1.tight_layout(pad=1.5)
fig1.savefig("paper_figures/fig1_loss_trajectories.png", dpi=300)
fig1.savefig("paper_figures/fig1_loss_trajectories.pdf")
plt.close(fig1)

# Figure 2: Trade-off
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for row in summary:
    ax1.scatter(row["avg_forgetting_rel_%"], row["plasticity_retention"] * 100, s=130, label=row["label"])
    ax1.annotate(row["label"], (row["avg_forgetting_rel_%"], row["plasticity_retention"] * 100), xytext=(5, 5), textcoords="offset points")

ax1.axvline(0, color="gray", linestyle=":")
ax1.set_xlabel(r"Relative Forgetting on Old Tasks (%) $\leftarrow$ Better")
ax1.set_ylabel(r"Plasticity Retention (%) $\rightarrow$ Better")
ax1.set_title("Pareto Trade-off", fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.4)

methods = [r["label"] for r in summary]
x = np.arange(len(methods))
ax2.bar(x - 0.18, [r["rank_frac"] * 100 for r in summary], 0.36, label="Protected Rank %", color="#4A90E2", edgecolor="black")
ax2.bar(x + 0.18, [r["dW_kept"] * 100 for r in summary], 0.36, label="Norm Kept %", color="#50E3C2", edgecolor="black")
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.set_ylabel("Percentage (%)")
ax2.set_title("Subspace Capacity vs Norm", fontweight="bold")
ax2.legend()
ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

fig2.tight_layout(pad=1.5)
fig2.savefig("paper_figures/fig2_stability_plasticity_tradeoff.png", dpi=300)
fig2.savefig("paper_figures/fig2_stability_plasticity_tradeoff.pdf")
plt.close(fig2)
print("SUCCESS: Figure 1 & Figure 2 generated inside 'paper_figures/'")
