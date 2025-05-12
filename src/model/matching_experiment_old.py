# ----------------------------------------------------------
# 0. Imports
# ----------------------------------------------------------
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
import shap  # Explainability

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
from sklearn.cluster import KMeans  # ← NEW

from gurobipy import Model, GRB, quicksum

# ------------------ Models ------------------
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


# ----------------------------------------------------------
# 1. Data utilities
# ----------------------------------------------------------
def load_and_split_data(path: str):
    """
    Collapse target to binary (>=3 vs <3) and do a temporal split.
    """
    print("Loading and splitting data …")
    df = pd.read_parquet(path).drop(columns=["magnitudes_list"])

    # df = df.query("station_code.isin(['PAS', 'BSC', 'MAN', 'MAG'])")

    df["date"] = pd.to_datetime(df["date"])
    df.drop(columns=["daily_max_30d_mean"], inplace=True)

    X = df.drop(columns=["max_mag_next_30d", "target_class"])
    y = ((df["target_class"] - 1) >= 3).astype(int)  # binary

    X_train = X.loc[X.date < "2024-01-01"].copy()
    X_test = X.loc[X.date >= "2024-01-30"].copy()
    y_train = y.loc[X.date < "2024-01-01"].copy()
    y_test = y.loc[X.date >= "2024-01-30"].copy()

    X_train.drop(columns=["date"], inplace=True)
    X_test.drop(columns=["date"], inplace=True)
    print("Data ready.")
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------
# 2a. Random undersampler  ⇢ NEW
# ----------------------------------------------------------
def undersample_random(X_train, y_train, random_state=42):
    """
    Keep all positives; randomly sample an equal number of negatives.
    """
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        print("Random undersampling skipped – one class missing.")
        return X_train, y_train

    np.random.seed(random_state)
    samp_neg = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
    keep = np.concatenate([pos_idx, samp_neg])
    print(
        f"Random undersample → retained {len(keep)} samples "
        f"({len(pos_idx)} pos + {len(samp_neg)} neg)"
    )
    return X_train.iloc[keep].reset_index(drop=True), y_train.iloc[keep].reset_index(
        drop=True
    )


# ----------------------------------------------------------
# 2b. Gurobi matching undersampler
# ----------------------------------------------------------
def gurobi_match_positives(Xp, Xn):
    """
    Bipartite 1-to-≤1 matching:
        min Σ z_ij ||xp_i – xn_j||²
        Σ_j z_ij = 1  (each positive matched)
        Σ_i z_ij ≤ 1  (neg used ≤1)
    Skips pairs whose distance is not finite.
    """
    n_pos, n_neg = Xp.shape[0], Xn.shape[0]
    D = np.sum((Xp[:, None, :] - Xn[None, :, :]) ** 2, axis=2)

    if np.isnan(D).any() or np.isinf(D).any():
        print("   Distance matrix had NaN/Inf – filtered offending pairs.")

    m = Model("match")
    m.setParam("OutputFlag", 0)
    z = {}
    for i in range(n_pos):
        for j in range(n_neg):
            if np.isfinite(D[i, j]):
                z[i, j] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{j}")

    m.setObjective(quicksum(D[i, j] * z[i, j] for (i, j) in z), GRB.MINIMIZE)

    for i in range(n_pos):
        m.addConstr(quicksum(z[i, j] for j in range(n_neg) if (i, j) in z) == 1)
    for j in range(n_neg):
        m.addConstr(quicksum(z[i, j] for i in range(n_pos) if (i, j) in z) <= 1)

    m.optimize()
    matched = [j for (i, j) in z if z[i, j].X > 0.5]
    return np.array(matched, dtype=int)


def undersample_by_matching(X_train, y_train, feats, n_clusters=10, random_state=42):
    """
    Keep all positives; pick a Gurobi-matched set of negatives.
    NEW: run matching separately inside K-means clusters to keep
    sub-problems small.
    """
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        print("Matching skipped – one class missing.")
        return X_train, y_train

    # --------------------------------------------------
    # Impute → scale so no NaNs in distance calc
    # --------------------------------------------------
    pipe = Pipeline(
        [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    )
    X_std = pipe.fit_transform(X_train[feats])

    # --------------------------------------------------
    # Cluster to limit match size
    # --------------------------------------------------
    k = min(n_clusters, len(X_train))
    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X_std)

    rng = np.random.default_rng(random_state)
    chosen_negs = []

    # --------------------------------------------------
    # Cluster-wise matching
    # --------------------------------------------------
    for c in range(k):
        clust_idx = np.where(labels == c)[0]
        pos_c = [i for i in clust_idx if i in pos_idx]
        neg_c = [i for i in clust_idx if i in neg_idx]

        if not pos_c:
            continue  # no positives in cluster
        if not neg_c:
            # fallback: random negatives from global pool
            sampled = rng.choice(neg_idx, size=len(pos_c), replace=False)
            chosen_negs.extend(sampled)
            continue

        Xp = X_std[pos_c, :]
        Xn = X_std[neg_c, :]
        sel = gurobi_match_positives(Xp, Xn)
        chosen_negs.extend(np.array(neg_c)[sel])

    keep = np.unique(np.concatenate([pos_idx, chosen_negs]))
    print(
        f"Clustered matching → retained {len(keep)} samples "
        f"({len(pos_idx)} pos + {len(keep) - len(pos_idx)} neg)"
    )
    return X_train.iloc[keep].reset_index(drop=True), y_train.iloc[keep].reset_index(
        drop=True
    )


# ----------------------------------------------------------
# 3. Model helper
# ----------------------------------------------------------
def get_clf():
    return LGBMClassifier(
        num_leaves=10,
        max_depth=2,
        min_child_samples=50,
        colsample_bytree=0.7,
        subsample=0.7,
        learning_rate=0.01,
        n_estimators=50,
        boosting_type="rf",
        reg_alpha=0.8,
        reg_lambda=0.8,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
    )


# ----------------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------------
def predict_thresh(pipe, X, τ=0.5):
    return (pipe.predict_proba(X)[:, 1] >= τ).astype(int)


def evaluate(pipe, X, y, title, τ=0.5):
    preds = predict_thresh(pipe, X, τ)
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds)
    print(f"{title}  acc={acc:.4f}\n{cm}")

    probs = pipe.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    print(f"{title}  AUC={roc_auc:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.title(f"{title} ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(classification_report(y, preds))


# ----------------------------------------------------------
# 5. Main run
# ----------------------------------------------------------
undersampling_random = False
undersampling_matching = True

X_train, X_test, y_train, y_test = load_and_split_data(
    "../../data/features/earthquake_features.parquet"
)

# drop sparse/list columns
drop_cols = list(X_train.columns[X_train.isna().mean() >= 0.5]) + [
    c for c in X_train.columns if "list" in c
]
X_train.drop(columns=drop_cols, inplace=True)
X_test.drop(columns=drop_cols, inplace=True)

# feature subset for both model & matching
feats = [
    "time_since_class_4",
    "time_since_class_3",
    "rolling_T_value",
    "daily_count_30d_sum",
    "daily_b_value",
    "rolling_dE_half",
    "daily_etas_intensity",
    "time_since_class_2",
    "daily_count_7d_sum",
]
X_train, X_test = X_train[feats], X_test[feats]

# -------- choose undersampling method -------------
if undersampling_random:
    X_train, y_train = undersample_random(X_train, y_train, random_state=42)
elif undersampling_matching:
    X_train, y_train = undersample_by_matching(
        X_train, y_train, feats, n_clusters=1, random_state=42
    )

prep = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
clf = get_clf()
pipe = Pipeline([("prep", prep), ("clf", clf)]).fit(X_train, y_train)

evaluate(pipe, X_train, y_train, "Train")
evaluate(pipe, X_test, y_test, "Test")

# SHAP
explainer = shap.TreeExplainer(clf)
shap_vals = explainer.shap_values(prep.transform(X_train))
shap.summary_plot(shap_vals, X_train, plot_type="bar", max_display=10)
