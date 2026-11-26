#Part B.1:feature-group ablation....
# We train  XGBoost on the domain-disjoint split, eliminate one group of features at a time,
#and observe how much the F1 score drop_s when compared to using all features...

# Saves : results/figures/ablation_f1_drop.png && results/ablation.csv 
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.common import *


def xgb_model_(X_):
    # dense output ,so XGBoost treats 0 as 0 (not "missing");  keeps  SHAP correct...
    prep_ = make_preprocessor(X_).set_params(sparse_threshold=0)
    clf_ = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=SEED)
    return Pipeline([("prep", prep_), ("clf", clf_)])


if __name__ == "__main__":
    # feature groups_ are defined once in this file,  so every script uses same ones
    g = json.load(open("data/feature_groups.json"))
    groups_ = {
        "baseline": [],
        "- leak_suspect": g["leak_suspect"],
        "- domain/TLD": g["domain_tld"],
        "- char_ratio": g["char_ratio"],
        "- all together": g["leak_suspect"] + g["domain_tld"] + g["char_ratio"],
    }

    Xtr_, Xte_, ytr_, yte_ = domain_disjoint_split(load_data())
    rows_ = []
    for name_, drop_ in groups_.items():
        cols = [c for c in Xtr_.columns if c not in drop_]
        #fresh model per group, trained on train only && scored on the test split
        model_ = xgb_model_(Xtr_[cols]).fit(Xtr_[cols], ytr_)
        rows_.append({"removed": name_, **evaluate(model_, Xte_[cols], yte_)})

    result_ = pd.DataFrame(rows_)
    #drop_ =baseline F1 - this row's F1(bigger means group matter more)
    result_["f1_drop"] = result_["f1"][0] - result_["f1"]
    result_.to_csv("results/ablation.csv", index=False)
    print(result_[["removed", "f1", "fnr", "f1_drop"]].round(4))

    # skips baseline row, so it has no drop_ to show
    result_[1:].plot.bar(x="removed", y="f1_drop", legend=False, rot=0,
                     title="F1 drop_ when feature group is removed")
    plt.tight_layout()
    plt.savefig("results/figures/ablation_f1_drop.png", dpi=150)
