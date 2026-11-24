import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.common import *

df = load_data()
Xtr, Xte, ytr, yte = domain_disjoint_split(df)
model = Pipeline([("prep", make_preprocessor(Xtr)),
                  ("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                        eval_metric="logloss", n_jobs=-1, random_state=SEED))])
model.fit(Xtr, ytr)
p = model.predict_proba(Xte)[:, 1]
pred = (p >= 0.5).astype(int)

fn = Xte[(pred == 0) & (yte.values == 1)]
print("Missed phishing (false negatives):", len(fn))
out = df.loc[fn.index, ["URL", "URLLength", "NoOfOtherSpecialCharsInURL", "URLSimilarityIndex",
                        "TLDLegitimateProb", "URLCharProb", "IsHTTPS"]].copy()
out["phish_prob"] = p[(pred == 0) & (yte.values == 1)]
print(out.to_string(index=False))

print("\nMedian feature values, test set by class:")
cols = ["URLLength", "NoOfOtherSpecialCharsInURL", "URLSimilarityIndex", "IsHTTPS"]
print(df.loc[Xte.index].groupby(TARGET)[cols].median())