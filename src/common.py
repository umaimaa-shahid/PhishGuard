import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, average_precision_score,
                             roc_auc_score)

DATA_PATH = "data/cleaned_with_domains.csv"
TARGET = "is_phishing"
NON_FEATURES = ["URL", "Domain", "registered_domain"]  # never used as features
SEED = 42


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def _xy(df):
    y = df[TARGET].astype(int)
    X = df.drop(columns=NON_FEATURES + [TARGET])
    groups = df["registered_domain"]
    return X, y, groups


def random_split(df, test_size=0.2, seed=SEED):
    X, y, _ = _xy(df)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def domain_disjoint_split(df, test_size=0.2, seed=SEED):
    X, y, groups = _xy(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    # sanity check: no domain appears in both sets
    assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[te])), "domain leak!"
    return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]


def make_preprocessor(X, scale=False):
    """Fit this on TRAIN only (use it inside a Pipeline)."""
    cat = ["TLD"] if "TLD" in X.columns else []
    num = [c for c in X.columns if c not in cat]
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num_tf, num),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat),
    ])


def evaluate(model, X_test, y_test):
    """Phishing = 1. Returns the metrics required by the brief (no accuracy)."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "fnr": fn / (fn + tp),
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }