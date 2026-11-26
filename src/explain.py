# Part B.3: SHAP explanations for final model_...
# Makes one global plot (which features matter most overall) and three
# single-URL plots (one caught phishing, one missed phishing, one legitimate).
# Needs models/final_model.joblib from final_model.py.
# Saves: results/figures/shap_global.png, shap_caught.png, shap_missed.png, shap_legitimate.png
import joblib, shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.common import *

saved_ = joblib.load("models/final_model.joblib")
model_, features_ = saved_["model"], saved_["features"]
prep_, clf_ = model_.named_steps["prep"], model_.named_steps["clf"]
# strip "num__" / "cat__" prefix , so plot labels are just feature names_
names_ = [n.split("__")[1] for n in prep_.get_feature_names_out()]
explainer_ = shap.TreeExplainer(clf_)

dataframe = load_data()
_, Xte_, _, yte_ = domain_disjoint_split(dataframe)
Xte_ = Xte_[features_]


def shap_values(X):
    # SHAP explains XGBoost step only, so we run preprocessing ourselves 1stly...
    return explainer_(pd.DataFrame(prep_.transform(X), columns=names_))


# Global imp (5,000 test URLs, sample to keep it fast)
sv_ = shap_values(Xte_.sample(5000, random_state=SEED))
shap.plots.beeswarm(sv_, max_display=12, show=False)
plt.savefig("results/figures/shap_global.png", dpi=150, bbox_inches="tight")
plt.close()

# One caught, one missed, and  one legitimate URL (first match in each of the group)...
p_ = model_.predict_proba(Xte_)[:, 1]
y_ = yte_.values
cases = {
    "caught": np.where((y_ == 1) & (p_ >= 0.5))[0][0],
    "missed": np.where((y_ == 1) & (p_ < 0.5))[0][0],
    "legitimate": np.where((y_ == 0) & (p_ < 0.5))[0][0],
}
for name, i in cases.items():
    url = dataframe.loc[Xte_.index[i], "URL"]
    print(f"{name:10s} p={p_[i]:.4f}  {url}")
    shap.plots.waterfall(shap_values(Xte_.iloc[[i]])[0], max_display=10, show=False)
    plt.title(f"{name}: {url[:60]}  (P(phishing) = {p_[i]:.3f})", fontsize=9)
    plt.savefig(f"results/figures/shap_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
