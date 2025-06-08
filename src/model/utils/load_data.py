"""
Data loading & temporal train/test split.
No ML-specific imports here so the function can be reused anywhere.
"""

from typing import Tuple
import pandas as pd


def load_and_split_data(
    daily_path: str,
    seismic_path: str,
    drop_sparse: bool = True,
    sparse_thresh: float = 0.5,
    drop_list_cols: bool = True,
    select_top_feats: bool = True,
    top_features: list = None,
    keep_embeddings: bool = True,
    merge_embeddings: bool = True,
    date_col: str = "date",
    target: str = "target_class",
    cutoff_date: str = "2024-01-01",
):
    """
    Load, merge, clean, and split the earthquake dataset.

    Args:
        daily_path: path to daily features parquet file
        seismic_path: path to pickled seismic embeddings file
        drop_sparse: whether to drop columns with too many NaNs
        sparse_thresh: threshold for NaN percentage to drop a column
        drop_list_cols: whether to drop columns containing 'list'
        select_top_feats: whether to select a fixed set of top features
        top_features: list of top features to keep (required if select_top_feats is True)
        keep_embeddings: whether to keep emb_* columns
        merge_embeddings: whether to merge the seismic embedding dataset   # <── NEW ARG
        date_col: column used for temporal split
        target: name of target column
        cutoff_date: split date for train/test

    Returns:
        X_train, X_test, y_train, y_test
    """
    import pandas as pd

    # -------------------- Read --------------------
    df_daily = pd.read_parquet(daily_path)
    df_daily[date_col] = pd.to_datetime(df_daily[date_col])  # ensure datetime

    if merge_embeddings:
        df_seismic = pd.read_pickle(seismic_path)

        emb_dim = len(df_seismic.iloc[0]["embedding"])
        emb_cols = [f"emb_{i}" for i in range(emb_dim)]
        df_emb = pd.DataFrame(df_seismic["embedding"].tolist(), columns=emb_cols)
        df_seismic_exp = pd.concat(
            [df_seismic.drop(columns=["embedding"]), df_emb], axis=1
        )

        # -------------------- Align Time --------------------
        df_seismic_exp["period_end"] = pd.to_datetime(
            df_seismic_exp["period_end"]
        ).dt.tz_convert(None)
        df_seismic_exp["period_end"] = pd.to_datetime(
            df_seismic_exp["period_end"].dt.date
        )

        # -------------------- Merge --------------------
        df = pd.merge(
            df_daily,
            df_seismic_exp.drop(columns=["period_start", "label"]),
            left_on=[date_col, "station_code"],
            right_on=["period_end", "station"],
            how="inner",
        ).drop(columns=["period_end", "station"])
    else:
        # Skip merging — work with daily data only
        df = df_daily.copy()

    # -------------------- Drop NaNs, set target --------------------
    df = df.dropna(subset=[target]).copy()
    df[date_col] = pd.to_datetime(df[date_col])

    y = df[target] - 1  # labels from 0
    X = df.drop(columns=["max_mag_next_30d", target])

    # -------------------- Optional Dropping --------------------
    drop_cols = []
    if drop_sparse:
        drop_cols += list(X.columns[X.isna().mean() >= sparse_thresh])
    if drop_list_cols:
        drop_cols += [c for c in X.columns if "list" in c]
    X.drop(columns=drop_cols, inplace=True)

    # -------------------- Select features --------------------
    emb_cols = [c for c in X.columns if c.startswith("emb_")]
    if select_top_feats:
        if not top_features:
            raise ValueError(
                "`top_features` must be provided if `select_top_feats=True`."
            )
        selected = top_features + (emb_cols if keep_embeddings else [])
        X = X[selected + [date_col]].copy()

    # -------------------- Temporal Split --------------------
    cutoff = pd.to_datetime(cutoff_date)
    X_train = X[X[date_col] < cutoff].copy()
    X_test = X[X[date_col] >= cutoff].copy()
    y_train = y.loc[X_train.index]
    y_test = y.loc[X_test.index]

    # -------------------- Drop Date Col --------------------
    X_train = X_train.drop(columns=[date_col], errors="ignore")
    X_test = X_test.drop(columns=[date_col], errors="ignore")

    # -------------------- Common Labels --------------------
    common = set(y_train.unique()) & set(y_test.unique())
    X_train, y_train = X_train[y_train.isin(common)], y_train[y_train.isin(common)]
    X_test, y_test = X_test[y_test.isin(common)], y_test[y_test.isin(common)]

    # Drop date column before returning
    X_train.drop(columns=[date_col], inplace=True, errors="ignore")
    X_test.drop(columns=[date_col], inplace=True, errors="ignore")

    # df full for reference (e.g. useful in GNN code)
    base_cols = [date_col, "station_code", target] + list(X.columns)
    unique_cols = list(dict.fromkeys(base_cols))
    df_full = df[unique_cols].copy()

    print(f"✅ Data ready. Train rows: {len(X_train)}, Test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test, df_full
