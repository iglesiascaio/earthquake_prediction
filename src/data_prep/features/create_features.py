#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_features.py

Reads earthquake data (Parquet) and config parameters from `10-features-config.yaml`,
computes a variety of features:
  - Magnitude conversion (to ML) using SCEDC (2024) formulas
  - Distance-based filtering (within 50 km of station)
  - Rolling features (b-value, rolling mean depth, etc.)
  - Vectorized ETAS intensity feature (Ogata, 1988; Zhuang et al., 2004)
  - Creates a target variable "Max earthquake magnitude in the next N days"
    (optionally converted to multi-class)
Organized into modular functions to follow best practices.
"""

import os
import yaml
import numpy as np
import pandas as pd
from haversine import haversine, Unit
from typing import Dict


# -------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -------------------------------------------------------------------


def convert_magnitude(row: pd.Series) -> float:
    """
    Convert different magnitude types to Local Magnitude (ML) using known formulas from SCEDC:
      - Mw  -> ML = (Mw  - 0.40125) / 0.853
      - Mlr -> ML = (Mlr - 0.40125) / 0.853
    (SCEDC, 2024)

    If magnitude_type is Ml, return original magnitude.
    Otherwise, return NaN if unknown type.
    """
    if row["magnitude_type"] == "Ml":
        return row["magnitude"]
    elif row["magnitude_type"] == "Mw":
        return (row["magnitude"] - 0.40125) / 0.853
    elif row["magnitude_type"] == "Mlr":
        return (row["magnitude"] - 0.40125) / 0.853
    else:
        return np.nan


def compute_b_value(magnitudes: np.ndarray) -> float:
    """
    Compute Gutenberg-Richter b-value using maximum likelihood estimation.
    Source: Aki (1965), Gutenberg & Richter (1954)

    b = log10(e) / (mean(M) - Mmin + 1e-6)
    """
    if len(magnitudes) < 2:
        return np.nan
    m_min = np.min(magnitudes)
    m_mean = np.mean(magnitudes)
    return np.log10(np.e) / (m_mean - m_min + 1e-6)


def magnitude_to_class(mag: float, bin_edges: list) -> float:
    """
    Convert a single magnitude to a discrete class index based on bin_edges.

    Example:
        bin_edges = [2, 4, 6, 8]
        Classes:
          mag < 2 => class 0
          2 <= mag < 4 => class 1
          4 <= mag < 6 => class 2
          6 <= mag < 8 => class 3
          mag >= 8 => class 4
    """
    if pd.isna(mag):
        return np.nan
    return np.digitize([mag], bin_edges)[0]


# -------------------------------------------------------------------
# 2. FEATURE-BUILDING FUNCTIONS
# -------------------------------------------------------------------


def filter_distance_to_station(
    df: pd.DataFrame, max_distance_km: float
) -> pd.DataFrame:
    """
    Filter earthquakes by distance (km) to a station.

    Expects columns:
      'latitude', 'longitude'
      'station_latitude', 'station_longitude'
    Returns only those rows within `max_distance_km`.
    """
    df["distance_to_station_km"] = df.apply(
        lambda row: haversine(
            (row["latitude"], row["longitude"]),
            (row["station_latitude"], row["station_longitude"]),
            unit=Unit.KILOMETERS,
        ),
        axis=1,
    )
    return df[df["distance_to_station_km"] <= max_distance_km].copy()


def compute_rolling_features(df: pd.DataFrame, feature_params: Dict) -> pd.DataFrame:
    """
    Compute various rolling or window-based features:
      - rolling mean of depth
      - time since last event
      - rolling b-value
      - rolling max magnitude & event count

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least:
          'time_utc', 'depth_km', 'magnitude'.
    feature_params : Dict
        Dictionary specifying window sizes, e.g.:
            {
                "rolling_depth_window": 5,    # int (event-based window)
                "b_value_window": 50,        # int (event-based window)
                "rolling_mag_stats_window": "7d",  # time-based rolling
                "rolling_count_window": "30d"      # time-based rolling
            }

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with new rolling features.
    """
    # Sort by time
    df = df.sort_values("time_utc").copy()

    # 1. Rolling mean of depth (window in # of events)
    rolling_depth_window = feature_params["rolling_depth_window"]
    df["rolling_mean_depth"] = (
        df["depth_km"].rolling(window=rolling_depth_window, min_periods=1).mean()
    )

    # 2. Time since last event (in days)
    df["time_utc_shift"] = df["time_utc"].shift(1)
    df["T_since_last_days"] = (
        df["time_utc"] - df["time_utc_shift"]
    ).dt.total_seconds() / (3600.0 * 24.0)
    df["T_since_last_days"].fillna(0.0, inplace=True)
    df.drop(columns=["time_utc_shift"], inplace=True)

    # 3. Rolling b-value (over the last N events)
    b_value_window = feature_params["b_value_window"]
    b_vals = []
    for i in range(len(df)):
        start_idx = max(0, i - b_value_window + 1)
        subset = df.iloc[start_idx : i + 1]
        b_vals.append(compute_b_value(subset["magnitude"].values))
    df["b_value"] = b_vals

    # Delta b-value (current vs 2 steps back)
    df["b_value_shift2"] = df["b_value"].shift(2)
    df["Delta_b_i_i_2"] = df["b_value"] - df["b_value_shift2"]

    # 4. Rolling max magnitude in last X days (time-based window)
    df.set_index("time_utc", inplace=True)
    rolling_mag_stats_window = feature_params["rolling_mag_stats_window"]
    df["M_last_week_max"] = df["magnitude"].rolling(rolling_mag_stats_window).max()

    # 5. Rolling count of events in last X days (time-based window)
    rolling_count_window = feature_params["rolling_count_window"]
    df["N_eq_30"] = df["magnitude"].rolling(rolling_count_window).count()

    # Reset index so 'time_utc' remains a column
    df.reset_index(inplace=True)

    return df


# -------------------------------------------------------------------
# 3. ETAS FUNCTION
# -------------------------------------------------------------------


def compute_etas_intensity(df: pd.DataFrame, etas_params: Dict) -> pd.DataFrame:
    """
    Efficiently compute ETAS intensity for each event using the standard formula:
      λ(t, x, y) = μ + ∑[t_i < t] [
          K * 10^(α*(m_i - M0)) * (t - t_i + c)^(-p) * exp(-d^2 / (2σ^2))
      ]
    (Ogata, 1988; Zhuang et al., 2004)

    Expects columns:
      - 'time_utc' (datetime)
      - 'latitude', 'longitude' (float, degrees)
      - 'magnitude_ml' (float) for the event's magnitude
    Also 'sigma' in km, 'c' in days, etc.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new column 'etas_intensity'.
    """
    # Sort events by time ascending
    df = df.sort_values("time_utc").reset_index(drop=True)

    # Extract arrays for speed
    times = df["time_utc"].values.astype("datetime64[s]")  # seconds resolution
    lats = df["latitude"].values
    lons = df["longitude"].values
    mags = df["magnitude_ml"].values
    n = len(df)

    # Unpack ETAS parameters
    mu = etas_params["mu"]
    K = etas_params["K"]
    alpha = etas_params["alpha"]
    M0 = etas_params["M0"]
    c_ = etas_params["c"]  # must be in days
    p_ = etas_params["p"]
    sigma = etas_params["sigma"]  # must be in km

    # Convert times to float in days
    times_days = times.astype(float) / 86400.0

    intensities = np.zeros(n, dtype=float)

    for i in range(n):
        # Indices of all previous events
        j_idx = np.arange(i)
        dt = times_days[i] - times_days[j_idx]  # time difference in days

        # Keep only dt > 0
        mask = dt > 0
        if not np.any(mask):
            # no earlier event => intensity = mu
            intensities[i] = mu
            continue

        j_idx = j_idx[mask]
        dt = dt[mask]

        # Spatial distances
        dist = np.array(
            [
                haversine((lats[i], lons[i]), (lats[j], lons[j]), unit=Unit.KILOMETERS)
                for j in j_idx
            ]
        )

        # Magnitude factor: 10^(alpha * (m_j - M0))
        mag_factor = 10.0 ** (alpha * (mags[j_idx] - M0))
        # Time decay: (dt + c)^(-p)
        time_decay = (dt + c_) ** (-p_)
        # Spatial decay: exp(-dist^2 / (2*sigma^2))
        spatial_decay = np.exp(-(dist**2) / (2.0 * sigma**2))

        intensities[i] = mu + (K * mag_factor * time_decay * spatial_decay).sum()

    df["etas_intensity"] = intensities
    return df


# -------------------------------------------------------------------
# 4. TARGET CREATION FUNCTION
# -------------------------------------------------------------------


def create_target_variable(df: pd.DataFrame, target_params: Dict) -> pd.DataFrame:
    """
    Create the target variable:
      - "Max earthquake magnitude in the next N days" => 'max_mag_next_30d'
      - Optionally convert to multi-class bins => 'target_class'

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'time_utc' and 'magnitude'.
    target_params : Dict
        Dictionary with:
            "next_days_for_target": int,
            "magnitude_bin_edges": list (optional)

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with new target columns: 'max_mag_next_30d', 'target_class' (if bins provided).
    """
    next_days_for_target = target_params["next_days_for_target"]
    bin_edges = target_params.get("magnitude_bin_edges", [])

    # 1. Sort by time & convert times to numeric for merging
    df = df.sort_values("time_utc").reset_index(drop=True)
    df["timestamp"] = df["time_utc"].astype(np.int64) // 10**9  # seconds since epoch

    # 2. Prepare a copy with an upper bound in seconds
    day_in_seconds = 86400
    df_shifted = df.copy()
    df_shifted["timestamp_min"] = (
        df_shifted["timestamp"] + next_days_for_target * day_in_seconds
    )

    df["max_mag_next_30d"] = np.nan

    # 3. Naive loop: for each row i, search future events in [t, t + next_days_for_target)
    for i in range(len(df_shifted)):
        current_ts = df_shifted.loc[i, "timestamp"]
        max_ts = df_shifted.loc[i, "timestamp_min"]
        future_subset = df_shifted[
            (df_shifted["timestamp"] >= current_ts) & (df_shifted["timestamp"] < max_ts)
        ]
        if not future_subset.empty:
            df_shifted.loc[i, "max_mag_next_30d"] = future_subset["magnitude"].max()
        else:
            df_shifted.loc[i, "max_mag_next_30d"] = np.nan

    df["max_mag_next_30d"] = df_shifted["max_mag_next_30d"].values

    # 4. Convert to classes if bin_edges given
    if bin_edges:
        df["target_class"] = df["max_mag_next_30d"].apply(
            lambda m: magnitude_to_class(m, bin_edges)
        )
    else:
        df["target_class"] = np.nan

    # 5. Clean up
    df.drop(columns=["timestamp"], inplace=True)

    return df


# -------------------------------------------------------------------
# 5. MAIN FUNCTION
# -------------------------------------------------------------------


def main():
    # 1. Load configuration
    config_file = "../../../config/10-features-config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["features_config"]

    input_data_path = config["input_data_path"]
    output_data_path = config["output_data_path"]
    feature_params = config["feature_params"]
    etas_params = config["etas_params"]
    target_params = config["target_params"]

    # 2. Read the raw data
    df = pd.read_parquet(input_data_path, engine="pyarrow")

    import ipdb

    ipdb.set_trace()

    # 3. Convert magnitudes to ML
    df["magnitude_ml"] = df.apply(convert_magnitude, axis=1)
    df.dropna(subset=["magnitude_ml"], inplace=True)

    # 4. Filter earthquakes near the station (within 50 km)
    #    (Assumes the dataset has 'station_latitude' and 'station_longitude' columns.)
    df = filter_distance_to_station(df, max_distance_km=50)

    # 5. Build rolling features (uses the original 'magnitude' column for b-value, etc.)
    #    NOTE: If you prefer to use 'magnitude_ml' for b_value, you can swap it inside compute_rolling_features.
    df = compute_rolling_features(df, feature_params)

    # 6. Compute ETAS intensity (using 'magnitude_ml')
    df = compute_etas_intensity(df, etas_params)

    # 7. Create target variable
    df = create_target_variable(df, target_params)

    # 8. Save final DataFrame
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df.to_parquet(output_data_path, index=False, engine="pyarrow")

    print(f"Feature table created and saved to: {output_data_path}")
    print(f"Total rows in feature table: {len(df)}")


# -------------------------------------------------------------------
# 6. ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
