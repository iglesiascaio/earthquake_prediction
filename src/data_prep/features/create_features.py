#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_features.py

Reads earthquake data (Parquet) and config parameters from `00-features-config.yaml`,
computes a variety of features (b-value, ETAS placeholder, etc.), and
creates a target variable: "Max earthquake magnitude in the next N days" 
for each event. Optionally convert that target to a multi-class label.
"""

import os
import yaml
import numpy as np
import pandas as pd
from obspy import UTCDateTime

# ---------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------


def compute_b_value(magnitudes):
    """
    Compute Gutenberg-Richter b-value in a simplistic manner.
    Real-world usage would prefer robust max-likelihood methods or curve-fitting.
    b = log10(e) / (mean(M - Mmin))
    """
    if len(magnitudes) < 2:
        return np.nan
    m_min = np.min(magnitudes)
    m_mean = np.mean(magnitudes)
    return np.log10(np.e) / (m_mean - m_min + 1e-6)


def placeholder_etas_value(row, etas_params):
    """
    Placeholder for ETAS feature (Epidemic-Type Aftershock Sequence).
    A real ETAS model is more complex.
    This function might, for instance, incorporate:
      - (1) distance between events
      - (2) times between events
      - (3) magnitude-based triggering function
    For now, we return a simple function of magnitude/time for demonstration.
    """
    alpha = etas_params.get("alpha", 1.0)
    c_param = etas_params.get("c", 0.01)
    p_param = etas_params.get("p", 1.1)

    # Example: ETAS "product" ~ (10^(alpha*M)) / (t + c)^p
    # We assume row["T_since_last"] is in days for this demonstration (or hours).
    # You might need to unify the unit carefully.
    t_since_last = row.get("T_since_last_days", 1.0)  # days
    mag = row.get("magnitude", 1.0)

    val = (10 ** (alpha * mag)) / ((t_since_last + c_param) ** p_param)
    return val


def magnitude_to_class(mag, bin_edges):
    """
    Convert magnitude to discrete classes based on bin_edges.
    E.g., bin_edges = [2, 4, 6, 8, 10] -> classes:
      - <2.0 => class 0
      - [2.0,4.0) => class 1
      - [4.0,6.0) => class 2
      - [6.0,8.0) => class 3
      - [8.0,10.0) => class 4
      - >=10 -> class 5
    """
    if pd.isna(mag):
        return np.nan
    # np.digitize returns indices of the bins to which each value belongs.
    # But we shift by 1 to have < first bin as 0, etc.
    return np.digitize([mag], bin_edges)[0]


# ---------------------------------------------
# MAIN CREATE FEATURES FUNCTION
# ---------------------------------------------


def main():
    # 1. Load config
    config_file = "../../../config/10-features-config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["features_config"]

    import ipdb

    ipdb.set_trace()
    input_data_path = config["input_data_path"]
    output_data_path = config["output_data_path"]

    fp = config["feature_params"]
    etas_params = config["etas_params"]
    tp = config["target_params"]

    # 2. Read the raw data
    df = pd.read_parquet(input_data_path, engine="pyarrow")

    # Ensure datetime
    df["time_utc"] = pd.to_datetime(df["time_utc"])
    df.sort_values("time_utc", inplace=True)

    # 3. Basic features from original table/paper
    #    (latitude, longitude, rolling depth, etc.)

    # Rolling mean of Depth (window in # of events)
    rolling_depth_window = fp["rolling_depth_window"]
    df["rolling_mean_depth"] = (
        df["depth_km"].rolling(window=rolling_depth_window, min_periods=1).mean()
    )

    # Time since last event (in days, for consistent usage with ETAS)
    df["time_utc_shift"] = df["time_utc"].shift(1)
    df["T_since_last_days"] = (
        df["time_utc"] - df["time_utc_shift"]
    ).dt.total_seconds() / (3600.0 * 24.0)
    df["T_since_last_days"].fillna(0.0, inplace=True)

    # b-value (rolling approach over the last N events)
    b_value_window = fp["b_value_window"]
    b_vals = []
    for i in range(len(df)):
        start_idx = max(0, i - b_value_window + 1)
        subset = df.iloc[start_idx : i + 1]
        b_vals.append(compute_b_value(subset["magnitude"].values))
    df["b_value"] = b_vals

    # Delta b-value (i vs i-2)
    df["b_value_shift2"] = df["b_value"].shift(2)
    df["Delta_b_i_i_2"] = df["b_value"] - df["b_value_shift2"]

    # Rolling max magnitude in last X days (e.g., 7d)
    rolling_mag_stats_window = fp["rolling_mag_stats_window"]
    df.set_index("time_utc", inplace=True)
    df["M_last_week_max"] = df["magnitude"].rolling(rolling_mag_stats_window).max()

    # Rolling count of events in last X days (e.g., 30d)
    rolling_count_window = fp["rolling_count_window"]
    df["N_eq_30"] = df["magnitude"].rolling(rolling_count_window).count()

    # Reset index
    df.reset_index(inplace=True)

    # Drop helper columns no longer needed
    df.drop(columns=["time_utc_shift"], inplace=True)

    # 4. Add ETAS placeholder feature
    #    We'll compute it row-by-row for demonstration
    df["etas_value"] = df.apply(
        lambda row: placeholder_etas_value(row, etas_params), axis=1
    )

    # 5. Create the TARGET variable:
    #    "Max earthquake magnitude in the next N days in this station"
    #    We'll do it per event, so for each row, find events that occur
    #    within [time_utc, time_utc + next_days_for_target) and get the max magnitude.
    next_days_for_target = tp["next_days_for_target"]

    # Sort again by time
    df.sort_values("time_utc", inplace=True)
    # We'll do a merge_asof approach or manual loop. For large data, merge_asof is more efficient.
    df["max_mag_next_30d"] = np.nan

    # Convert times to numeric for merge_asof
    df["timestamp"] = df["time_utc"].astype(np.int64) // 10**9  # seconds since epoch
    df = df.reset_index(drop=True)

    # We'll create a shifted DataFrame with the same columns
    # but we also store a "time_min" that is the original time + 30 days
    df_shifted = df.copy()
    # time_min in seconds
    day_in_seconds = 86400
    df_shifted["timestamp_min"] = (
        df_shifted["timestamp"] + next_days_for_target * day_in_seconds
    )

    # We can do an approach:
    # For each row i, select the slice from i to as far forward as time < time_min
    # and take the max magnitude. This can be done with a custom search or a loop.
    # For demonstration, a naive loop (will be slow for huge data):
    # You may want a more efficient approach if data is large.

    for i in range(len(df_shifted)):
        current_ts = df_shifted.loc[i, "timestamp"]
        max_ts = df_shifted.loc[i, "timestamp_min"]
        # Filter future events within next_days_for_target
        # (We assume the DataFrame is sorted by timestamp)
        future_subset = df_shifted[
            (df_shifted["timestamp"] >= current_ts) & (df_shifted["timestamp"] < max_ts)
        ]
        if not future_subset.empty:
            df_shifted.loc[i, "max_mag_next_30d"] = future_subset["magnitude"].max()
        else:
            df_shifted.loc[i, "max_mag_next_30d"] = np.nan

    # Merge back or simply rename
    df["max_mag_next_30d"] = df_shifted["max_mag_next_30d"].values

    # Now optionally convert that target magnitude to a multi-class label
    bin_edges = tp.get("magnitude_bin_edges", [])
    if bin_edges:
        # E.g. create a new column: "target_class"
        df["target_class"] = df["max_mag_next_30d"].apply(
            lambda m: magnitude_to_class(m, bin_edges)
        )
    else:
        df["target_class"] = np.nan  # or skip

    # 6. Clean up (drop helper columns if desired)
    df.drop(columns=["timestamp"], inplace=True)
    df_shifted = None  # free memory

    # 7. Save final feature DataFrame
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df.to_parquet(output_data_path, index=False, engine="pyarrow")

    print(f"Feature table created and saved to: {output_data_path}")
    print(f"Total rows in feature table: {len(df)}")


if __name__ == "__main__":
    main()
