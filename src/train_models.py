import os, pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.common import *

os.makedirs("results", exist_ok=True)

def build_models(X):
    return {
        "LogReg": Pipeline([("prep", make_preprocessor(X, scale=True)),
                            ("clf", LogisticRegression(max_iter=2000, random_state=SEED))]),
        "RandomForest": Pipeline([("prep", make_preprocessor(X)),
                                  ("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED))]),
        "XGBoost": Pipeline([("prep", make_preprocessor(X)),
                             ("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                                   eval_metric="logloss", n_jobs=-1, random_state=SEED))]),
    }

if __name__ == "__main__":
    df = load_data()
    splits = {"random": random_split(df), "domain_disjoint": domain_disjoint_split(df)}
    rows, preds = [], {}

    for split_name, (Xtr, Xte, ytr, yte) in splits.items():
        for model_name, model in build_models(Xtr).items():
            print(f"Training {model_name} on {split_name} split...")
            model.fit(Xtr, ytr)
            m = evaluate(model, Xte, yte)
            rows.append({"model": model_name, "split": split_name, **m})
            preds[(model_name, split_name)] = (yte.values, model.predict_proba(Xte)[:, 1])

    res = pd.DataFrame(rows)
    res.to_csv("results/metrics.csv", index=False)
    with open("results/preds.pkl", "wb") as f:
        pickle.dump(preds, f)

    cols = ["model", "split", "precision", "recall", "f1", "macro_f1", "fnr", "pr_auc", "roc_auc"]
    print(res[cols].round(4).to_string(index=False))

    # F1 gap: random minus domain-disjoint
    piv = res.pivot(index="model", columns="split", values="f1")
    piv["gap"] = piv["random"] - piv["domain_disjoint"]
    print("\nF1 gap (random - domain_disjoint):\n", piv.round(4))
    piv.to_csv("results/f1_gap.csv")