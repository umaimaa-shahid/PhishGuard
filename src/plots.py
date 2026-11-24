import os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, roc_curve)

os.makedirs("results/figures", exist_ok=True)
preds = pickle.load(open("results/preds.pkl", "rb"))
models = ["LogReg", "RandomForest", "XGBoost"]
splits = ["random", "domain_disjoint"]

# Confusion matrices: 2 rows (splits) x 3 cols (models)
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for i, s in enumerate(splits):
    for j, m in enumerate(models):
        y, p = preds[(m, s)]
        cm = confusion_matrix(y, (p >= 0.5).astype(int))
        ConfusionMatrixDisplay(cm, display_labels=["legit", "phish"]).plot(
            ax=axes[i, j], colorbar=False, values_format="d")
        axes[i, j].set_title(f"{m} | {s}")
plt.tight_layout()
plt.savefig("results/figures/confusion_matrices.png", dpi=150)
plt.close()

# PR and ROC curves: one figure per split
for s in splits:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for m in models:
        y, p = preds[(m, s)]
        pr, rc, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(rc, pr, label=m)
        a2.plot(fpr, tpr, label=m)
    a1.set(xlabel="Recall", ylabel="Precision", title=f"PR curve ({s})")
    a2.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC curve ({s})")
    a2.plot([0, 1], [0, 1], "k--", lw=0.5)
    # Curves hug the corner, so zoom in to make them readable
    a1.set_xlim(0.95, 1.001); a1.set_ylim(0.95, 1.001)
    a2.set_xlim(0, 0.05); a2.set_ylim(0.95, 1.001)
    a1.legend(); a2.legend()
    plt.tight_layout()
    plt.savefig(f"results/figures/pr_roc_{s}.png", dpi=150)
    plt.close()

# F1 gap bar chart at higher precision
import pandas as pd
g = pd.read_csv("results/f1_gap.csv", index_col=0)
print(g.round(6))
g["gap"].plot.bar(title="F1(random) - F1(domain-disjoint)")
plt.axhline(0, color="k", lw=0.5)
plt.tight_layout()
plt.savefig("results/figures/f1_gap.png", dpi=150)

# Error counts for the analysis
import pandas as pd
r = pd.read_csv("results/metrics.csv")
print(r[["model", "split", "fp", "fn", "tp", "tn"]].to_string(index=False))