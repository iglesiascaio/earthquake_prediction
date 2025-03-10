#!/usr/bin/env python
import os
import datetime
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import shap  # SHAP for model explainability
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

# ------------------ Model Imports ------------------
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# =============================================================================
# FUNCTIONS
# =============================================================================


def load_and_split_data(path: str, task: str):
    """
    Load the dataset, clean it, and perform a temporal train-test split.
    """
    print("Loading and splitting data...")
    df = pd.read_parquet(path)
    df = df.drop(columns=["magnitudes_list"])

    # Drop rows with missing target values
    if task == "regression":
        df = df.dropna(subset=["max_mag_next_30d"])
    else:
        df = df.dropna(subset=["target_class"])

    df["date"] = pd.to_datetime(df["date"])
    df.drop(columns=["daily_max_30d_mean"], inplace=True)

    # Select target column based on task
    target_col = "max_mag_next_30d" if task == "regression" else "target_class"
    X = df.drop(columns=["max_mag_next_30d", "target_class"])
    y = df[target_col]

    # For classification, shift labels to start at 0
    if task == "classification":
        y = y - 1

    # Temporal train-test split:
    #   Train: dates before 2024-01-01
    #   Test: dates from 2024-01-30 onward
    X_train = X.loc[X.date < "2024-01-01"].copy()
    X_test = X.loc[X.date >= "2024-01-30"].copy()
    y_train = y.loc[X.date < "2024-01-01"].copy()
    y_test = y.loc[X.date >= "2024-01-30"].copy()

    # Drop the date column since it's no longer needed
    X_train = X_train.drop(columns=["date"])
    X_test = X_test.drop(columns=["date"])

    print("Data loading complete.")
    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """
    Build a preprocessing pipeline with median imputation and standard scaling.
    """
    print("Building preprocessor pipeline...")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def get_model_and_param_grid(model_name: str, task: str, y_train=None):
    """
    Given a model name and task, return the model instance and its hyperparameter grid.
    Here we use a simplified LGBMClassifier with bias-introducing hyperparameters.
    """
    if task == "classification":
        if model_name == "LGBMClassifier":
            model = LGBMClassifier(
                # Tree structure & complexity control:
                num_leaves=10,  # Lower value reduces complexity
                max_depth=2,  # Limits tree depth
                min_child_samples=50,  # More samples per leaf
                # Regularization:
                reg_alpha=0.8,  # L1 regularization
                reg_lambda=0.8,  # L2 regularization
                # Sampling (introduces stochasticity):
                colsample_bytree=0.7,
                subsample=0.7,
                subsample_freq=1,
                # Boosting control:
                learning_rate=0.01,
                n_estimators=100,
                boosting_type="gbdt",
                # Early stopping & convergence:
                min_gain_to_split=0.01,
                max_bin=255,
                # Handling imbalanced classes:
                is_unbalance=True,  # Alternatively, use class_weight='balanced'
                # Randomness:
                random_state=42,
                verbose=-1,
            )
            # Expanded hyperparameter search space for tuning:
            param_grid = {
                "model__num_leaves": [15, 31, 50, 70, 100],
                "model__max_depth": [-1, 5, 10, 15, 20],
                "model__learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
                "model__min_child_samples": [10, 20, 30, 50],
                "model__subsample": [0.5, 0.7, 0.9, 1.0],
                "model__colsample_bytree": [0.5, 0.7, 0.9, 1.0],
                "model__reg_alpha": [0, 0.01, 0.1, 1, 10],
                "model__reg_lambda": [0, 0.01, 0.1, 1, 10],
            }
        elif model_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(max_iter=1000, random_state=42)
            param_grid = {"model__C": [0.1, 1, 10]}
        elif model_name == "KNNClassifier":
            from sklearn.neighbors import KNeighborsClassifier

            model = KNeighborsClassifier()
            param_grid = {"model__n_neighbors": [3, 5, 7]}
        elif model_name == "SVC":
            from sklearn.svm import SVC

            model = SVC(probability=True, random_state=42, kernel="sigmoid")
            param_grid = {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}
        elif model_name == "RandomForestClassifier":
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(random_state=42)
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
            }
        else:
            raise ValueError(f"Invalid classification model name: {model_name}")
    else:
        raise ValueError("This script is configured for classification tasks only.")
    return model, param_grid


def evaluate_model(
    model_pipeline: Pipeline, X, y, dataset: str, task: str, save_dir: str
):
    """
    Evaluate the given model pipeline on a dataset, print performance metrics,
    and save the confusion matrix plot and metrics to files.
    """
    print(f"Evaluating model on {dataset} data...")
    y_pred = model_pipeline.predict(X)
    report_str = ""

    if task == "regression":
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        report_str += f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}\n"
    else:
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report_str += f"Accuracy: {acc:.4f}\n\nConfusion Matrix:\n{cm}\n\n"

        # Plot and save the confusion matrix heatmap
        plt.figure(figsize=(6, 4))
        classes = np.unique(np.concatenate((y, y_pred)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"{dataset} Confusion Matrix")
        cm_filename = os.path.join(save_dir, f"{dataset}_confusion_matrix.png")
        plt.savefig(cm_filename)
        plt.close()
        print(f"Saved confusion matrix plot to {cm_filename}")

        # Generate classification report and append to report string
        report = classification_report(y, y_pred)
        report_str += f"Classification Report:\n{report}\n"

    # Save evaluation metrics to a text file
    eval_filename = os.path.join(save_dir, f"{dataset}_evaluation.txt")
    with open(eval_filename, "w") as f:
        f.write(report_str)
    print(f"Saved evaluation metrics to {eval_filename}")


def plot_shap_values(
    model_pipeline: Pipeline, X: pd.DataFrame, model_name: str, save_dir: str
):
    """
    Compute and plot SHAP values for tree-based models, then save the SHAP summary plot.
    """
    print("Computing SHAP values...")
    tree_model = model_pipeline.named_steps["model"]
    X_trans = model_pipeline.named_steps["preprocessor"].transform(X)
    X_trans_df = pd.DataFrame(X_trans, columns=X.columns)

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_trans_df)

    # Generate SHAP summary plot and save it
    shap.summary_plot(
        shap_values, X_trans_df, plot_type="bar", max_display=10, show=False
    )
    shap_filename = os.path.join(save_dir, f"{model_name}_shap_summary.png")
    plt.savefig(shap_filename, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {shap_filename}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    # Create a timestamped folder for results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("results", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Set task and tuning flag
    task = "classification"  # This script is set up for classification
    do_tuning = True  # Set to True if you want hyperparameter tuning

    # Select the model(s) to run (using LGBMClassifier here)
    selected_models = ["LGBMClassifier"]

    # Load and split the data
    X_train, X_test, y_train, y_test = load_and_split_data(
        "../../data/features/earthquake_features.parquet", task
    )

    # Drop columns with high missing rates and any column with "list" in its name
    drop_cols = list(X_train.columns[X_train.isna().mean() >= 0.5]) + [
        col for col in X_train.columns if "list" in col
    ]
    X_train.drop(columns=drop_cols, inplace=True)
    X_test.drop(columns=drop_cols, inplace=True)

    # Select top features (adjust as needed)
    top_10_features = [
        "time_since_class_4",
        "time_since_class_3",
        "rolling_T_value",
        "daily_count_30d_sum",
        "daily_b_value",
        "rolling_dE_half",
        "daily_etas_intensity",
        "time_since_class_2",
        "daily_min",
        "daily_count_7d_sum",
    ]
    X_train = X_train[top_10_features]
    X_test = X_test[top_10_features]

    # Build the preprocessor pipeline
    preprocessor = build_preprocessor()

    # Set up TimeSeriesSplit cross-validation
    cv = TimeSeriesSplit(n_splits=3)

    # Process each selected model
    for model_name in selected_models:
        print("\n" + "=" * 60)
        print(f"Processing model: {model_name}")

        # Get model instance and hyperparameter grid
        model, param_grid = get_model_and_param_grid(model_name, task, y_train=y_train)

        # Build the full pipeline: preprocessor followed by the model
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # Optionally perform hyperparameter tuning
        if do_tuning and param_grid:
            print(
                "Tuning hyperparameters using RandomizedSearchCV with TimeSeriesSplit CV..."
            )
            gs = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                cv=cv,
                scoring="accuracy",
                verbose=3,
                n_iter=50,
                random_state=42,
            )
            gs.fit(X_train, y_train)
            best_pipe = gs.best_estimator_
            print("Best Parameters:", gs.best_params_)
            results_df = pd.DataFrame(gs.cv_results_).sort_values(
                "mean_test_score", ascending=False
            )
            tuning_filename = os.path.join(save_dir, "cv_tuning_summary.txt")
            results_df[
                ["params", "mean_test_score", "std_test_score", "rank_test_score"]
            ].head(10).to_csv(tuning_filename, sep="\t")
            print(f"Saved CV tuning summary to {tuning_filename}")
        else:
            print("Fitting model without tuning...")
            best_pipe = pipe.fit(X_train, y_train)

        print("X_train shape:", X_train.shape)
        # Evaluate the model on training and test data, saving outputs
        evaluate_model(
            best_pipe,
            X_train,
            y_train,
            dataset="Training",
            task=task,
            save_dir=save_dir,
        )
        evaluate_model(
            best_pipe, X_test, y_test, dataset="Test", task=task, save_dir=save_dir
        )

        # Compute and save SHAP summary plot
        plot_shap_values(best_pipe, X_train, model_name, save_dir=save_dir)

        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
