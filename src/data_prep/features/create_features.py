#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_features.py

1) Read earthquake (event-based) data from Parquet.
2) Convert to a daily DataFrame:
   - Each row = one calendar day (whether or not a quake happened).
   - Aggregates (lists of magnitudes, depths, sqrt(energy), daily max, daily count, etc.).
3) Compute all features on the daily DataFrame:
   - Rolling b-value in a time-based window (e.g., last 30 days).
   - "Time since last earthquake" (days).
   - "Time since last earthquake in each magnitude class" (one column per class).
   - Typical time series features (rolling sums, means, diffs).
   - NEW: Depth-based features (daily avg, min, max, and range of depth).
   - NEW: Energy features (total and average sqrt(energy) per day).
   - NEW: Rolling seismological parameters from the last 100 events:
         • T value (elapsed days between first and 100th event),
         • dE^(1/2) (sum of sqrt(energy) divided by T),
         • Magnitude deficit (observed daily max minus expected max from a GR-based estimation).
4) Create target variable: "max magnitude over the next N days."
5) Save final daily DataFrame to Parquet for ML usage.
"""

import os
import yaml
import numpy as np
import pandas as pd
import math
from typing import Dict, List, Optional
from haversine import haversine, Unit
import re

# -------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -------------------------------------------------------------------


def convert_magnitude(row: pd.Series) -> float:
    """
    Convert various magnitude types to Local Magnitude (ML) using known formulas.

    - Mlr -> ML = (Mlr - 0.40125) / 0.853
    - For others, return the magnitude as is.
    """
    mtype = row["magnitude_type"]
    mag = row["magnitude"]

    if mtype == "Ml":
        return mag
    elif mtype == "Mlr":
        return (mag - 0.40125) / 0.853
    else:
        return mag


def magnitude_to_class(mag: float, bin_edges: list) -> Optional[int]:
    """
    Convert a single magnitude to a discrete class index based on bin_edges.
    """
    if pd.isna(mag):
        return np.nan
    return np.digitize([mag], bin_edges)[0]


def compute_b_value(magnitudes: List[float]) -> float:
    """
    Compute Gutenberg-Richter b-value using maximum likelihood:
      b = ln(10) / [mean(M) - min(M)]  (with a tiny offset to avoid division by zero)
    """
    if len(magnitudes) < 2:
        return np.nan
    mags = np.array(magnitudes, dtype=float)
    m_min = np.min(mags)
    m_mean = np.mean(mags)
    return np.log(10) / (m_mean - m_min + 1e-6)


def compute_event_sqrt_energy(row: pd.Series) -> float:
    """
    Compute the square root of the earthquake energy for an event.
    Energy E = 10^(11.8 + 1.5 * magnitude_ml) (in ergs)
    Return sqrt(E) = 10^((11.8 + 1.5*magnitude_ml)/2)
    """
    E = 10 ** (11.8 + 1.5 * row["magnitude_ml"])
    return math.sqrt(E)


# -------------------------------------------------------------------
# 2. STEP: CONVERT EVENT-BASED DATA -> DAILY DATA
# -------------------------------------------------------------------


def filter_by_distance(df_events: pd.DataFrame, max_distance_km: float) -> pd.DataFrame:
    """
    Keep only events within `max_distance_km` of the station.
    """
    df = df_events.copy()
    df["distance_to_station_km"] = df.apply(
        lambda row: haversine(
            (row["latitude"], row["longitude"]),
            (row["station_latitude"], row["station_longitude"]),
            unit=Unit.KILOMETERS,
        ),
        axis=1,
    )
    return df[df["distance_to_station_km"] <= max_distance_km].copy()


def convert_events_to_daily(
    df_events: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    bin_edges: List[float],
) -> pd.DataFrame:
    """
    Create a daily DataFrame covering [start_date, end_date], inclusive.
    Each row = 1 day. For each day, store:
      - magnitudes_list: list of magnitudes (for b-value, etc.)
      - depth_list: list of event depths (km)
      - sqrt_energy_list: list of sqrt(energy) values per event
      - daily_count, daily_max, daily_min, daily_avg (of magnitudes)
      - Daily class counts for each magnitude class.
    """
    df = df_events.copy()
    # Round quake times to day
    df["date"] = df["time_utc"].dt.floor("D")
    # Compute sqrt_energy for each event
    df["sqrt_energy"] = df.apply(compute_event_sqrt_energy, axis=1)

    # Create mag_class for each event if bin_edges provided
    if bin_edges:
        df["mag_class"] = df["magnitude_ml"].apply(
            lambda m: magnitude_to_class(m, bin_edges)
        )
    else:
        df["mag_class"] = np.nan

    # Group by day
    group = df.groupby("date")

    # Aggregate: lists of magnitudes, depths, sqrt_energy values
    daily_agg = group.agg(
        {
            "magnitude_ml": list,
            "depth_km": list,
            "sqrt_energy": list,
        }
    )
    daily_agg.rename(
        columns={
            "magnitude_ml": "magnitudes_list",
            "depth_km": "depth_list",
            "sqrt_energy": "sqrt_energy_list",
        },
        inplace=True,
    )

    # Basic daily stats for magnitudes
    daily_agg["daily_count"] = group["magnitude_ml"].count()
    daily_agg["daily_max"] = group["magnitude_ml"].max()
    daily_agg["daily_min"] = group["magnitude_ml"].min()
    daily_agg["daily_avg"] = group["magnitude_ml"].mean()

    # For class counts
    if bin_edges:
        class_counts = group["mag_class"].value_counts().unstack(fill_value=0)
        class_counts = class_counts.add_prefix("daily_class_count_")
        daily_agg = daily_agg.join(class_counts, how="left")

    # Reindex to ensure every day in [start_date, end_date] is present
    full_range = pd.date_range(start_date, end_date, freq="D")
    daily_agg = daily_agg.reindex(full_range)
    daily_agg.index.name = "date"

    # Fill missing values for days with no events
    daily_agg["daily_count"] = daily_agg["daily_count"].fillna(0)
    daily_agg["magnitudes_list"] = daily_agg["magnitudes_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    daily_agg["depth_list"] = daily_agg["depth_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    daily_agg["sqrt_energy_list"] = daily_agg["sqrt_energy_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    for col in daily_agg.columns:
        if col.startswith("daily_class_count_"):
            daily_agg[col] = daily_agg[col].fillna(0)

    daily_agg.reset_index(inplace=True)
    return daily_agg


# -------------------------------------------------------------------
# 2a. NEW: Additional daily features from depth and energy lists
# -------------------------------------------------------------------


def add_depth_energy_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional daily features:
      - Depth features: daily average, min, max depth and depth range.
      - Energy features: total and average sqrt(energy) from events.
    """
    df = df_daily.copy()

    def safe_mean(lst):
        return np.mean(lst) if lst else np.nan

    def safe_min(lst):
        return np.min(lst) if lst else np.nan

    def safe_max(lst):
        return np.max(lst) if lst else np.nan

    # Depth-based features
    df["daily_avg_depth"] = df["depth_list"].apply(safe_mean)
    df["daily_min_depth"] = df["depth_list"].apply(safe_min)
    df["daily_max_depth"] = df["depth_list"].apply(safe_max)
    df["daily_depth_range"] = df["daily_max_depth"] - df["daily_min_depth"]

    # Energy features
    df["total_sqrt_energy"] = df["sqrt_energy_list"].apply(
        lambda lst: np.sum(lst) if lst else 0
    )
    df["avg_sqrt_energy"] = df.apply(
        lambda row: (
            row["total_sqrt_energy"] / row["daily_count"]
            if row["daily_count"] > 0
            else np.nan
        ),
        axis=1,
    )
    return df


# -------------------------------------------------------------------
# 2b. NEW: Rolling seismological features based on the last 100 events
# -------------------------------------------------------------------


def compute_rolling_seismological_features(
    df_events: pd.DataFrame, df_daily: pd.DataFrame, window_size: int = 100
) -> pd.DataFrame:
    """
    For each day in the daily DataFrame, use the last `window_size` events (from df_events)
    occurring before the current day to compute:
      - rolling_T_value: time elapsed (in days) between the first and the last event.
      - rolling_dE_half: sum of sqrt(energy) over the window divided by T_value.
      - rolling_magnitude_deficit: daily_max minus an estimated expected maximum magnitude,
        computed using a simplified Gutenberg–Richter formulation:
            a_est = log10(window_size) + b_value * (min magnitude in window)
            Mmax_expected = a_est / b_value
    If there are fewer than window_size events before a day, features are set to NaN.
    """
    df_daily = df_daily.copy()
    # Ensure df_events is sorted by time
    df_events_sorted = df_events.sort_values("time_utc").copy()
    # Prepare lists to hold new feature values per day
    T_values = []
    dE_half_values = []
    mag_deficit_values = []

    # Convert time_utc to pandas Timestamp for easier manipulation
    df_events_sorted["time_utc"] = pd.to_datetime(df_events_sorted["time_utc"])

    for current_day in df_daily["date"]:
        # Get events before the current day
        past_events = df_events_sorted[df_events_sorted["time_utc"] < current_day]
        if len(past_events) < window_size:
            T_values.append(np.nan)
            dE_half_values.append(np.nan)
            mag_deficit_values.append(np.nan)
            continue

        window_events = past_events.iloc[-window_size:]
        t_first = window_events["time_utc"].iloc[0]
        t_last = window_events["time_utc"].iloc[-1]
        # Compute T_value in days (could be fractional)
        T_value = (t_last - t_first).total_seconds() / 86400.0
        T_values.append(T_value if T_value > 0 else np.nan)

        # Compute dE_half = sum(sqrt_energy) / T_value
        sum_sqrt_energy = window_events["sqrt_energy"].sum()
        dE_half = sum_sqrt_energy / T_value if T_value > 0 else np.nan
        dE_half_values.append(dE_half)

        # Compute b-value from window magnitudes
        window_mags = window_events["magnitude_ml"].tolist()
        b_val = compute_b_value(window_mags)
        if np.isnan(b_val) or len(window_mags) == 0:
            mag_deficit_values.append(np.nan)
        else:
            # Estimate 'a' using a simplified approach: a_est = log10(window_size) + b_val * (min magnitude)
            m_min = np.min(window_mags)
            a_est = math.log10(window_size) + b_val * m_min
            Mmax_expected = a_est / b_val if b_val != 0 else np.nan
            # Magnitude deficit: daily_max (from df_daily) - Mmax_expected
            daily_max = df_daily[df_daily["date"] == current_day]["daily_max"].values[0]
            mag_deficit = (
                daily_max - Mmax_expected
                if not np.isnan(daily_max) and not np.isnan(Mmax_expected)
                else np.nan
            )
            mag_deficit_values.append(mag_deficit)

    df_daily["rolling_T_value"] = T_values
    df_daily["rolling_dE_half"] = dE_half_values
    df_daily["rolling_magnitude_deficit"] = mag_deficit_values
    return df_daily


# -------------------------------------------------------------------
# 3. FEATURE COMPUTATIONS ON DAILY DATA
# -------------------------------------------------------------------


def compute_time_since_last_eq(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 'time_since_last_eq' in days.
    If daily_count > 0, reset to 0; otherwise, increment from the previous day.
    """
    df = df_daily.sort_values("date").copy()
    event_occurred = df["daily_count"] > 0
    last_event_date = df["date"].where(event_occurred).ffill()
    df["time_since_last_eq"] = (df["date"] - last_event_date).dt.days
    return df


def compute_time_since_last_eq_per_class(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    For each magnitude class, compute a column `time_since_class_{c}` representing the days since
    the last earthquake in that class.
    """
    df = df_daily.sort_values("date").copy()
    class_cols = [col for col in df.columns if re.match(r"daily_class_count_\d+", col)]
    class_numbers = [int(col.split("_")[-1]) for col in class_cols]
    if not class_numbers:
        return df
    max_class = max(class_numbers)
    for c in range(max_class + 1):
        col_name = f"time_since_class_{c}"
        daily_col = f"daily_class_count_{c}"
        if daily_col not in df.columns:
            df[daily_col] = 0
        event_occurred = df[daily_col] > 0
        last_event_date = df["date"].where(event_occurred).ffill()
        df[col_name] = (df["date"] - last_event_date).dt.days.fillna(0).astype(int)
    return df


def compute_rolling_b_value_daily(
    df_daily: pd.DataFrame, days_window: int = 30
) -> pd.DataFrame:
    """
    For each day, compute the b-value using all magnitudes from the past `days_window` days.
    """
    df = df_daily.sort_values("date").copy()
    b_values = []
    for i in range(len(df)):
        current_day = df.loc[i, "date"]
        start_day = current_day - pd.Timedelta(days=days_window - 1)
        subset = df[(df["date"] >= start_day) & (df["date"] <= current_day)]
        all_mags = []
        for mags_list in subset["magnitudes_list"]:
            all_mags.extend(mags_list)
        bval = compute_b_value(all_mags)
        b_values.append(bval)
    df["daily_b_value"] = b_values
    return df


def compute_additional_time_series_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute typical time series features such as differences and rolling aggregates.
    """
    df = df_daily.sort_values("date").copy()
    df["daily_count_diff"] = df["daily_count"].diff()
    df["daily_max_diff"] = df["daily_max"].diff()
    df["daily_count_7d_sum"] = df["daily_count"].rolling(7).sum()
    df["daily_max_7d_mean"] = df["daily_max"].rolling(7).mean()
    df["daily_count_30d_sum"] = df["daily_count"].rolling(30).sum()
    df["daily_max_30d_mean"] = df["daily_max"].rolling(30).mean()
    return df


# -------------------------------------------------------------------
# 4. ETAS INTENSITY (DAILY)
# -------------------------------------------------------------------


def compute_daily_etas(
    df_daily: pd.DataFrame, df_events: pd.DataFrame, etas_params: Dict
) -> pd.DataFrame:
    """
    For each day, compute ETAS intensity at that day's reference time using the formula:
      λ_i = μ + Σ_j [ K * 10^(α * (m_j - M0)) * (t_i - t_j + c)^(-p) * exp(-dist^2 / (2*sigma^2)) ]
    where the sum is over all prior events.
    """
    df_out = df_daily.sort_values("date").copy().reset_index(drop=True)
    df_events_sorted = df_events.sort_values("time_utc").reset_index(drop=True)
    if len(df_events_sorted) == 0:
        mu = etas_params["mu"]
        df_out["daily_etas_intensity"] = mu
        return df_out

    station_lat = df_events_sorted["station_latitude"].iloc[0]
    station_lon = df_events_sorted["station_longitude"].iloc[0]
    event_times = df_events_sorted["time_utc"].values.astype("datetime64[s]")
    event_days_float = event_times.astype(float) / 86400.0
    event_lats = df_events_sorted["latitude"].values
    event_lons = df_events_sorted["longitude"].values
    event_mags = df_events_sorted["magnitude_ml"].values

    mu = etas_params["mu"]
    K_ = etas_params["K"]
    alpha = etas_params["alpha"]
    M0 = etas_params["M0"]
    c_ = etas_params["c"]
    p_ = etas_params["p"]
    sigma = etas_params["sigma"]

    daily_etas = []
    for i in range(len(df_out)):
        day_i = df_out.loc[i, "date"]
        t_i_float = day_i.value / 1e9 / 86400.0
        mask = event_days_float < t_i_float
        if not np.any(mask):
            daily_etas.append(mu)
            continue
        e_times = event_days_float[mask]
        e_lats = event_lats[mask]
        e_lons = event_lons[mask]
        e_mags = event_mags[mask]
        dt = t_i_float - e_times
        dist_array = np.array(
            [
                haversine(
                    (station_lat, station_lon), (lat_, lon_), unit=Unit.KILOMETERS
                )
                for lat_, lon_ in zip(e_lats, e_lons)
            ]
        )
        mag_factor = 10.0 ** (alpha * (e_mags - M0))
        time_decay = (dt + c_) ** (-p_)
        spatial_decay = np.exp(-(dist_array**2) / (2.0 * sigma**2))
        lam_i = mu + (K_ * mag_factor * time_decay * spatial_decay).sum()
        daily_etas.append(lam_i)
    df_out["daily_etas_intensity"] = daily_etas
    return df_out


# -------------------------------------------------------------------
# 5. TARGET CREATION (DAILY)
# -------------------------------------------------------------------


def create_daily_target(df_daily: pd.DataFrame, target_params: Dict) -> pd.DataFrame:
    """
    Create a target column for the maximum magnitude over the next N days.
    Optionally, assign a target class based on magnitude bin edges.
    """
    df = df_daily.sort_values("date").reset_index(drop=True)
    next_days_for_target = target_params["next_days_for_target"]
    bin_edges = target_params.get("magnitude_bin_edges", [])
    col_name = f"max_mag_next_{next_days_for_target}d"
    df[col_name] = np.nan
    for i in range(len(df)):
        current_date = df.loc[i, "date"]
        future_date = current_date + pd.Timedelta(days=next_days_for_target)
        subset = df[(df["date"] > current_date) & (df["date"] < future_date)]
        if not subset.empty:
            df.loc[i, col_name] = subset["daily_max"].max()
        else:
            df.loc[i, col_name] = np.nan
    if bin_edges:
        df["target_class"] = df[col_name].apply(
            lambda m: magnitude_to_class(m, bin_edges)
        )
    else:
        df["target_class"] = np.nan
    return df


# -------------------------------------------------------------------
# 6. MAIN PIPELINE
# -------------------------------------------------------------------


def main():
    # 1. Read config
    config_file = "../../../config/10-features-config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["features_config"]

    input_data_path = config["input_data_path"]
    output_data_path = config["output_data_path"]

    target_params = config["target_params"]
    bin_edges = target_params.get("magnitude_bin_edges", [])
    feature_params = config["feature_params"]
    max_distance_km = feature_params.get("max_distance_km", 50)
    b_value_rolling_window_days = feature_params.get("b_value_rolling_window_days", 30)
    etas_params = config["etas_params"]

    # 2. Read event-based data
    df_events = pd.read_parquet(input_data_path, engine="pyarrow")

    # 3. Convert magnitudes => ML, filter by distance
    df_events["magnitude_ml"] = df_events.apply(convert_magnitude, axis=1)
    df_events.dropna(subset=["magnitude_ml"], inplace=True)
    df_events = filter_by_distance(df_events, max_distance_km)
    # FIX: Compute sqrt_energy column in df_events (needed for rolling seismological features)
    df_events["sqrt_energy"] = df_events.apply(compute_event_sqrt_energy, axis=1)

    # 4. Determine daily date range
    if len(df_events) == 0:
        print("No events found after filtering. Creating an empty daily table.")
        min_date = pd.Timestamp("2020-01-01")
        max_date = pd.Timestamp("2020-01-01")
    else:
        min_date = df_events["time_utc"].min().floor("D")
        max_date = df_events["time_utc"].max().floor("D")

    # 5. Convert events to daily
    df_daily = convert_events_to_daily(
        df_events, start_date=min_date, end_date=max_date, bin_edges=bin_edges
    )

    # 6. Compute daily ETAS intensity
    df_daily = compute_daily_etas(df_daily, df_events, etas_params)

    # 7. Compute daily-based features:
    #  7a) Time since last event
    df_daily = compute_time_since_last_eq(df_daily)
    #  7b) Time since last event per magnitude class (if bin_edges provided)
    if bin_edges:
        df_daily = compute_time_since_last_eq_per_class(df_daily)
    #  7c) Rolling b-value over a specified window
    df_daily = compute_rolling_b_value_daily(
        df_daily, days_window=b_value_rolling_window_days
    )
    #  7d) Additional typical time series features
    df_daily = compute_additional_time_series_features(df_daily)
    #  7e) NEW: Add depth and energy features
    df_daily = add_depth_energy_features(df_daily)
    #  7f) NEW: Compute rolling seismological features (T value, dE^(1/2), magnitude deficit)
    df_daily = compute_rolling_seismological_features(
        df_events, df_daily, window_size=100
    )

    # 8. Create daily-level target (max magnitude in next N days)
    df_daily = create_daily_target(df_daily, target_params)

    # 9. Save final daily DataFrame
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df_daily.to_parquet(output_data_path, index=False, engine="pyarrow")

    print(f"Daily feature table saved to: {output_data_path}")
    print(f"Total rows in daily feature table: {len(df_daily)}")


if __name__ == "__main__":
    main()
