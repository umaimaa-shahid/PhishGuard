# Part B.2: train final model and save it for Streamlit app.
# We train XGBoost on the domain-disjoint split, remove three features
#that appear to be dataset leaks, and output the test results.
# Saves: models/final_model.joblib (the model and its feature list)
import os, joblib
from src.common import *
from src.ablation import xgb_model_

# these 3 were  computed over whole dataset, so we dont rust them
DROP_ = ["URLSimilarityIndex", "URLCharProb", "TLDLegitimateProb"]

Xtr_, Xte_, ytr_, yte_ = domain_disjoint_split(load_data())
features = [c for c in Xtr_.columns if c not in DROP_]

model_ = xgb_model_(Xtr_[features]).fit(Xtr_[features], ytr_)
print(evaluate(model_, Xte_[features], yte_))

os.makedirs("models", exist_ok=True)
# save feature list too,  so app builds its input in same col order...
joblib.dump({"model": model_, "features": features}, "models/final_model.joblib")
