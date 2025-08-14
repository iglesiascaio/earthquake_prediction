import warnings
import numpy as np
import pandas as pd
import json
import time
import os
from datetime import datetime
from pathlib import Path

# Set environment variables to prevent OMP errors
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Force printing and disable buffering for supercloud
import builtins
original_print = builtins.print
def force_print(*args, **kwargs):
    original_print(*args, **kwargs)
    import sys
    sys.stdout.flush()
builtins.print = force_print

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
# Note: XGBoost and CatBoost will be imported dynamically if needed

from .utils import load_and_split_data, evaluate

warnings.filterwarnings("ignore")

# ========== 1. Preprocessing ========== #
def build_preprocessor():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),  # Handle non-numeric data
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
        # Exhaustive LGBM parameter grid
        param_grid = {
            "model__num_leaves": [15, 31, 63, 127, 255],
            "model__max_depth": [-1, 3, 5, 7, 9, 11],
            "model__learning_rate": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            "model__n_estimators": [50, 100, 200, 300, 500],
            "model__min_child_samples": [10, 20, 50, 100],
            "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
            "model__reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
        }
    elif model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            "model__C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
            "model__penalty": ["l1", "l2", "elasticnet", None],
            "model__solver": ["liblinear", "saga"],
            "model__class_weight": ["balanced", None],
        }
    elif model_name == "RandomForestClassifier":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            "model__n_estimators": [50, 100, 200, 300, 500],
            "model__max_depth": [3, 5, 7, 9, 11, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", None],
            "model__bootstrap": [True, False],
            "model__class_weight": ["balanced", "balanced_subsample", None],
        }
    elif model_name == "XGBClassifier":
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
        param_grid = {
            "model__n_estimators": [50, 100, 200, 300, 500],
            "model__max_depth": [3, 5, 7, 9, 11],
            "model__learning_rate": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "model__reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
            "model__reg_lambda": [0.0, 0.1, 0.5, 1.0, 2.0],
            "model__min_child_weight": [1, 3, 5, 7],
            "model__gamma": [0.0, 0.1, 0.2, 0.5],
        }
    elif model_name == "CatBoostClassifier":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(random_state=42, verbose=False)
        param_grid = {
            "model__iterations": [50, 100, 200, 300, 500],
            "model__depth": [3, 5, 7, 9, 11],
            "model__learning_rate": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
            "model__l2_leaf_reg": [1, 3, 5, 7, 9],
            "model__border_count": [32, 64, 128, 254],
            "model__bagging_temperature": [0.0, 0.5, 1.0, 2.0],
            "model__random_strength": [0.0, 0.1, 0.5, 1.0],
        }
    else:
        # Default fallback
        model = LogisticRegression(max_iter=1000, random_state=42)
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

# ========== 4. Scenario Testing ========== #
def test_scenario(scenario_config, feature_strategy, model_name, do_tuning, results_dir):
    """Test a specific scenario and return results."""
    
    scenario_name = f"{scenario_config['name']}_{feature_strategy['name']}"
    print(f"\n--- Testing scenario: {scenario_name} ---")
    print(f"  Feature strategy: {feature_strategy['name']}")
    print(f"  Embeddings: merge={scenario_config['merge_embeddings']}, keep={scenario_config['keep_embeddings']}")
    
    try:
        # Load data for this scenario and feature strategy
        print("Loading data...")
        X_train, X_test, y_train, y_test, df_full = load_and_split_data(
            daily_path="./data/features/earthquake_features.parquet",
            seismic_path="./data/embeddings/Embeddings_192142.pkl",
            select_top_feats=feature_strategy["select_top_feats"],
            top_features=feature_strategy["top_features"],
            merge_embeddings=scenario_config["merge_embeddings"],
            keep_embeddings=scenario_config["keep_embeddings"],
            cutoff_date="2024-01-01",
        )
        
        # Data validation and cleaning
        print("Validating and cleaning data...")
        
        # Convert to numeric, coercing errors to NaN
        X_train = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce')
        X_test = pd.DataFrame(X_test).apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN values with 0 (since we're using constant imputer)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Convert back to numpy arrays
        X_train = X_train.values
        X_test = X_test.values
        
        # The load_and_split_data function has a bug: it only filters embeddings for select_top_feats=True
        # We need to manually filter embeddings for the all_features case
        print(f"  Data loaded with parameters:")
        print(f"    select_top_feats: {feature_strategy['select_top_feats']}")
        print(f"    merge_embeddings: {scenario_config['merge_embeddings']}")
        print(f"    keep_embeddings: {scenario_config['keep_embeddings']}")
        print(f"    X_train shape: {X_train.shape}")
        print(f"    X_test shape: {X_test.shape}")
        
        # Manual embedding filtering for all_features case
        if not feature_strategy["select_top_feats"] and not scenario_config["keep_embeddings"]:
            print(f"  🔧 Manual filtering: Removing embedding columns for all_features scenario")
            
            # Get column names from df_full to identify embedding columns
            embedding_cols = [c for c in df_full.columns if c.startswith("emb_")]
            print(f"    Found {len(embedding_cols)} embedding columns: {embedding_cols[:3]}{'...' if len(embedding_cols) > 3 else ''}")
            
            # Find indices of non-embedding columns in X_train/X_test
            non_embedding_cols = [c for c in df_full.columns if not c.startswith("emb_") and c not in ["max_mag_next_30d", "target_class", "date", "station_code"]]
            
            # Get indices of these columns in the original data
            col_to_idx = {col: i for i, col in enumerate(df_full.columns) if col not in ["max_mag_next_30d", "target_class", "date", "station_code"]}
            feature_indices = [col_to_idx[col] for col in non_embedding_cols if col in col_to_idx]
            
            # Filter X_train and X_test
            X_train = X_train[:, feature_indices]
            X_test = X_test[:, feature_indices]
            
            print(f"    Filtered to {X_train.shape[1]} non-embedding features")
            print(f"    New X_train shape: {X_train.shape}")
            print(f"    New X_test shape: {X_test.shape}")
        
        print(f"Final feature dimension: {X_train.shape[1]}")
        print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        print(f"Data types: {X_train.dtype}, NaN values: {np.isnan(X_train).sum()}")
        
        # Create scenario-specific directory for evaluation files
        scenario_dir = results_dir / f"{model_name}_{scenario_name}"
        scenario_dir.mkdir(exist_ok=True)
        print(f"  📁 Evaluation files will be saved to: {scenario_dir}")
        
        # Build and train pipeline
        print("Building and training pipeline...")
        start_time = time.time()
        
        preprocessor = build_preprocessor()
        model, param_grid = get_model_and_param_grid(model_name)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        
        if do_tuning:
            print("Running hyperparameter tuning...")
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                cv=TimeSeriesSplit(n_splits=3),
                scoring="accuracy",
                n_iter=50,  # Reduced from 100 to prevent resource exhaustion
                random_state=42,
                n_jobs=1,  # Single process to avoid OMP errors
                verbose=1,  # Show progress
            )
            pipe = search.fit(X_train, y_train).best_estimator_
            print(f"Best parameters: {search.best_params_}")
        else:
            pipe.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # ========== 5. Evaluate ========== #
        print("Evaluating model...")
        
        # Train predictions
        y_pred_train = predict_with_thresholds(pipe, X_train, thresholds=None)
        y_proba_train = pipe.predict_proba(X_train)
        
        # Test predictions
        y_pred_test = predict_with_thresholds(pipe, X_test, thresholds=None)
        y_proba_test = pipe.predict_proba(X_test)
        
        # Evaluate and save to scenario-specific directory
        evaluate(y_pred_train, y_proba_train, X_train, y_train, "Train", scenario_dir)
        evaluate(y_pred_test, y_proba_test, X_test, y_test, "Test", scenario_dir)
        
        # Extract metrics from evaluation output (similar to GNN script)
        import io
        import sys
        import re
        
        # Capture stdout to get metrics
        old_stdout = sys.stdout
        
        # Get training metrics
        sys.stdout = io.StringIO()
        evaluate(y_pred_train, y_proba_train, X_train, y_train, "Train", scenario_dir)
        train_output = sys.stdout.getvalue()
        
        # Get test metrics
        sys.stdout = io.StringIO()
        evaluate(y_pred_test, y_proba_test, X_test, y_test, "Test", scenario_dir)
        test_output = sys.stdout.getvalue()
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Extract metrics using regex
        train_auc_match = re.search(r'Train Average ROC AUC: ([\d.]+)', train_output)
        test_auc_match = re.search(r'Test Average ROC AUC: ([\d.]+)', test_output)
        
        train_auc = float(train_auc_match.group(1)) if train_auc_match else np.nan
        test_auc = float(test_auc_match.group(1)) if test_auc_match else np.nan
        
        train_acc_match = re.search(r'Train Accuracy: ([\d.]+)', train_output)
        test_acc_match = re.search(r'Test Accuracy: ([\d.]+)', test_output)
        
        train_acc = float(train_acc_match.group(1)) if train_acc_match else np.nan
        test_acc = float(test_acc_match.group(1)) if test_acc_match else np.nan
        
        print(f"  ✅ Completed: {scenario_name}")
        print(f"     Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}")
        print(f"     Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
        print(f"     Training Time: {training_time:.1f}s")
        
        return {
            "scenario": scenario_name,
            "model_name": model_name,
            "feature_strategy": feature_strategy["name"],
            "embeddings_merge": scenario_config["merge_embeddings"],
            "embeddings_keep": scenario_config["keep_embeddings"],
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0],
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "test_accuracy": test_acc,
            "test_auc": test_auc,
            "training_time": training_time,
            "do_tuning": do_tuning,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error in scenario {scenario_name}: {str(e)}")
        return {
            "scenario": scenario_name,
            "model_name": model_name,
            "feature_strategy": feature_strategy["name"],
            "embeddings_merge": scenario_config["merge_embeddings"],
            "embeddings_keep": scenario_config["keep_embeddings"],
            "n_features": np.nan,
            "n_train_samples": np.nan,
            "n_test_samples": np.nan,
            "train_accuracy": np.nan,
            "train_auc": np.nan,
            "test_accuracy": np.nan,
            "test_auc": np.nan,
            "training_time": np.nan,
            "do_tuning": do_tuning,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# ========== 6. Main Execution ========== #
def main():
    # Configuration
    MODEL_NAME = "LGBMClassifier"  # Options: "LGBMClassifier", "LogisticRegression", "RandomForestClassifier", "XGBClassifier", "CatBoostClassifier"
    DO_TUNING = True  # Enable exhaustive hyperparameter tuning
    CLASS_THRESHOLDS = None
    
    DATA_PATHS = {
        "daily": "./data/features/earthquake_features.parquet",
        "seismic": "./data/embeddings/Embeddings_192142.pkl",
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
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = Path(f"results/ml_hyperparameter_tuning_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define scenarios (same as GNN script)
    embedding_scenarios = [
        {"merge_embeddings": True, "keep_embeddings": False, "name": "no_embeddings_limited_data"},
        {"merge_embeddings": True, "keep_embeddings": True, "name": "embeddings_limited_data"},
        {"merge_embeddings": False, "keep_embeddings": False, "name": "no_embeddings_full_data"}
    ]
    
    # Define feature strategies
    feature_strategies = [
        {"name": "top_features", "select_top_feats": True, "top_features": TOP_FEATURES},
        {"name": "all_features", "select_top_feats": False, "top_features": None}
    ]
    
    # Calculate total combinations
    total_combinations = len(embedding_scenarios) * len(feature_strategies)
    
    # Save configuration
    config = {
        "timestamp": timestamp,
        "model": "LGBMClassifier",
        "do_tuning": DO_TUNING,
        "tuning_iterations": 50,
        "data_paths": DATA_PATHS,
        "top_features": TOP_FEATURES,
        "cutoff_date": "2024-01-01",
        "total_combinations": total_combinations,
    }
    
    with open(results_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Starting EXHAUSTIVE LightGBM hyperparameter tuning experiment")
    print(f"Testing {len(embedding_scenarios)} scenarios × {len(feature_strategies)} feature strategies = {total_combinations} total combinations")
    print(f"Each combination will test {50} LightGBM hyperparameter configurations (reduced for stability)")
    print(f"Results will be saved to: {results_dir}")
    print(f"Configuration saved to: {results_dir}/experiment_config.json")
    print("=" * 80)
    
    # Initialize results storage
    results = []
    partial_results_file = results_dir / "partial_results.csv"
    
    # Test each scenario and feature strategy combination with LightGBM
    current_combination = 0
    
    for scenario_config in embedding_scenarios:
        for feature_strategy in feature_strategies:
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"Testing Combination {current_combination}/{total_combinations}")
            print(f"Model: LGBMClassifier")
            print(f"Scenario: {scenario_config['name']}")
            print(f"Features: {feature_strategy['name']}")
            print(f"{'='*60}")
            
            # Test this combination
            result = test_scenario(
                scenario_config, 
                feature_strategy, 
                "LGBMClassifier", 
                DO_TUNING, 
                results_dir
            )
            
            results.append(result)
            
            # Save partial results after each combination
            df_partial = pd.DataFrame(results)
            df_partial.to_csv(partial_results_file, index=False)
            print(f"  💾 Partial results saved to: {partial_results_file}")
    
    # Save final results
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETED!")
    print(f"Results saved to: {results_dir}")
    
    # Create final CSV with all results
    df_final = pd.DataFrame(results)
    final_csv_path = results_dir / "final_results.csv"
    df_final.to_csv(final_csv_path, index=False)
    print(f"Final results saved to: {final_csv_path}")
    
    # Create summary table (filter out error rows)
    summary_data = df_final[df_final.get("error", "").isna()].copy()
    if not summary_data.empty:
        summary_csv_path = results_dir / "results_summary.csv"
        summary_data.to_csv(summary_csv_path, index=False)
        print(f"Summary table saved to: {summary_csv_path}")
        
        # Print top 3 scenarios by test accuracy
        print("\nTop 3 scenarios by test accuracy:")
        top_scenarios = summary_data.nlargest(3, "test_accuracy")[["scenario", "feature_strategy", "test_accuracy", "test_auc"]]
        print(top_scenarios.to_string(index=False))
    
    # Show directory structure
    print(f"\n📁 Directory structure created:")
    print(f"  {results_dir}/")
    print(f"  ├── experiment_config.json")
    print(f"  ├── partial_results.csv")
    print(f"  ├── final_results.csv")
    print(f"  └── [model_name]_[scenario_name]/")
    print(f"      ├── metrics_Train.txt")
    print(f"      ├── metrics_Test.txt")
    print(f"      ├── roc_Train.png")
    print(f"      ├── roc_Test.png")
    print(f"      └── [other evaluation files]")
    
    print(f"\nConfiguration saved to: {results_dir}/experiment_config.json")
    print(f"Partial results saved to: {results_dir}/partial_results.csv")
    print(f"Final results saved to: {results_dir}/final_results.csv")

if __name__ == "__main__":
    main()
