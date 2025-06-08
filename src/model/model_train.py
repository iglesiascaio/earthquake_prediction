import warnings, numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from .utils import load_and_split_data, evaluate

warnings.filterwarnings("ignore")


# ========== 1. Preprocessing ========== #
def build_preprocessor():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


# ========== 2. Model factory ========== #
def get_model_and_param_grid(model_name: str):
    if model_name == "LGBMClassifier":
        model = LGBMClassifier(
            learning_rate=0.01,
            n_estimators=100,
            is_unbalance=True,
            random_state=42,
            verbose=-1,
        )
        param_grid = {
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 3, 5],
            "model__learning_rate": [0.005, 0.01, 0.05],
        }
    else:
        model = LogisticRegression(max_iter=1000)
        param_grid = {"model__C": [0.1, 1, 10]}
    return model, param_grid


# ========== 3. Evaluation utils ========== #
def predict_with_thresholds(pipe, X, thresholds=None, default_rule="argmax"):
    if not thresholds:
        return pipe.predict(X)
    probas = pipe.predict_proba(X)
    classes = pipe.named_steps["model"].classes_
    idx_map = {c: i for i, c in enumerate(classes)}
    preds = (
        np.argmax(probas, axis=1) if default_rule == "argmax" else np.full(len(X), -1)
    )
    for cls, τ in thresholds.items():
        if cls in idx_map:
            preds[probas[:, idx_map[cls]] >= τ] = cls
    return preds


# ========== 4. Run Config ========== #
MODEL_NAME = "LGBMClassifier"
DO_TUNING = False
CLASS_THRESHOLDS = None  # Optional: e.g., {2: 0.3, 3: 0.25}
DATA_PATHS = {
    "daily": "./data/features/earthquake_features.parquet",
    "seismic": "./data/features/df_seismic.parquet",
}
TOP_FEATURES = [
    "time_since_class_3",
    "rolling_T_value",
    "daily_count_30d_sum",
    "daily_b_value",
    "rolling_dE_half",
    "daily_etas_intensity",
    "time_since_class_2",
    "daily_count_7d_sum",
]
RUN_DIR = Path("results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ========== 5. Load + Train ========== #

X_train, X_test, y_train, y_test, _ = load_and_split_data(
    daily_path="./data/features/earthquake_features.parquet",
    seismic_path="./data/embeddings/embeddings_190102.pkl",
    drop_sparse=True,
    sparse_thresh=0.5,
    drop_list_cols=True,
    select_top_feats=True,
    top_features=TOP_FEATURES,
    merge_embeddings=True,
    keep_embeddings=True,
    cutoff_date="2024-01-01",
)

preprocessor = build_preprocessor()
model, param_grid = get_model_and_param_grid(MODEL_NAME)
pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

if DO_TUNING:
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="accuracy",
        n_iter=5,
        random_state=42,
        n_jobs=-1,
        verbose=2,
    )
    pipe = search.fit(X_train, y_train).best_estimator_
else:
    pipe.fit(X_train, y_train)

# ========== 6. Evaluate ========== #
y_pred_train = predict_with_thresholds(pipe, X_train, thresholds=CLASS_THRESHOLDS)
y_proba_train = pipe.predict_proba(X_train)

y_pred_test = predict_with_thresholds(pipe, X_test, thresholds=CLASS_THRESHOLDS)
y_proba_test = pipe.predict_proba(X_test)


evaluate(
    y_pred_train,
    y_proba_train,
    X_train,
    y_train,
    "Train",
    RUN_DIR,
)
evaluate(y_pred_test, y_proba_test, X_test, y_test, "Test", RUN_DIR)
print(f"\n✅ Results saved to: {RUN_DIR}")
