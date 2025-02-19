#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
create_features.py

1) Read earthquake (event-based) data from Parquet.
2) Convert to a daily DataFrame:
   - Each row = one calendar day (whether or not a quake happened).
   - Aggregates (list of magnitudes, daily max, daily count, etc.).
3) Compute all features on the daily DataFrame:
   - Rolling b-value in a time-based window (e.g., last 30 days).
   - "Time since last earthquake" (days).
   - "Time since last earthquake in each magnitude class" (one column per class).
   - Other typical time series features (rolling sums, means, diffs).
4) Create target variable: "max magnitude over the next N days."
5) Save final daily DataFrame to Parquet for ML usage.
"""

import os
import yaml
import numpy as np
import pandas as pd
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
    - Mw  -> Retain as is (conversion requires region-specific formula)
    - Mh, Mb, Mun -> Conversion not available; return value of magnitude.
    - Obs: main earthquakes are typically reported in Mu, but we can find easily that it seems to be equivalent to Mw.

    Parameters:
    row (pd.Series): A pandas Series containing 'magnitude_type' and 'magnitude'.

    Returns:
    float: Converted magnitude in ML scale or NaN if conversion isn't possible.
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
    Returns an integer class, or NaN if mag is NaN.
    """
    if pd.isna(mag):
        return np.nan
    return np.digitize([mag], bin_edges)[0]


def compute_b_value(magnitudes: List[float]) -> float:
    """
    Compute Gutenberg-Richter b-value using maximum likelihood:
      b = ln(10) / [mean(M) - min(M)]  (plus a tiny offset to avoid /0)
    Expects 'magnitudes' to be a list of floats.
    """
    if len(magnitudes) < 2:
        return np.nan
    mags = np.array(magnitudes, dtype=float)
    m_min = np.min(mags)
    m_mean = np.mean(mags)
    return np.log(10) / (m_mean - m_min + 1e-6)


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
      - All quake magnitudes in a list (col = 'magnitudes_list')
      - daily_count (number of quakes that day)
      - daily_max (max magnitude that day)
      - daily_min (min magnitude that day)
      - daily_avg (mean magnitude)
      - daily_class_counts_{c} for each magnitude class c

    If a day has no quakes, it will have an empty list, daily_count=0, etc.
    """

    df = df_events.copy()
    # Round quake times to day
    df["date"] = df["time_utc"].dt.floor("D")

    # We also create a class for each event, so we can do daily class counts
    if bin_edges:
        df["mag_class"] = df["magnitude_ml"].apply(
            lambda m: magnitude_to_class(m, bin_edges)
        )
    else:
        df["mag_class"] = np.nan

    # Group by day, then aggregate
    # We'll store a list of magnitudes in that day for b-value, etc.
    group = df.groupby("date")

    # Build the aggregator
    daily_agg = group.agg(
        {
            "magnitude_ml": list,  # store all magnitudes in a list
        }
    )
    daily_agg.rename(columns={"magnitude_ml": "magnitudes_list"}, inplace=True)

    # Add daily_count, daily_max, daily_min, daily_avg
    daily_agg["daily_count"] = group["magnitude_ml"].count()
    daily_agg["daily_max"] = group["magnitude_ml"].max()
    daily_agg["daily_min"] = group["magnitude_ml"].min()
    daily_agg["daily_avg"] = group["magnitude_ml"].mean()

    # For class counts, pivot
    if bin_edges:
        class_counts = group["mag_class"].value_counts().unstack(fill_value=0)
        # This yields columns for each class index. e.g. 1,2,3...
        # We'll rename them to daily_class_count_{c}
        class_counts = class_counts.add_prefix("daily_class_count_")
        daily_agg = daily_agg.join(class_counts, how="left")

    # Reindex to ensure every day in [start_date .. end_date] is present
    full_range = pd.date_range(start_date, end_date, freq="D")
    daily_agg = daily_agg.reindex(full_range)
    daily_agg.index.name = "date"

    # Fill days with no quakes => daily_count=0, daily_max=NaN, etc.
    # 'magnitudes_list' => empty list for no quakes
    # We'll fill missing daily_count with 0, but leave daily_max, daily_min, daily_avg as NaN.
    daily_agg["daily_count"] = daily_agg["daily_count"].fillna(0)
    # For any missing 'magnitudes_list', we put an empty list
    daily_agg["magnitudes_list"] = daily_agg["magnitudes_list"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    # For missing daily_class_count_*, fill with 0
    for col in daily_agg.columns:
        if col.startswith("daily_class_count_"):
            daily_agg[col] = daily_agg[col].fillna(0)

    daily_agg.reset_index(inplace=True)
    return daily_agg


# -------------------------------------------------------------------
# 3. FEATURE COMPUTATIONS ON DAILY DATA
# -------------------------------------------------------------------


def compute_time_since_last_eq(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 'time_since_last_eq' in days.
    If daily_count > 0, reset to 0; otherwise, increment from the previous day.
    """

    df = df_daily.sort_values("date").copy()

    # Create a mask where an earthquake occurred (daily_count > 0)
    event_occurred = df["daily_count"] > 0

    # Assign unique event numbers by cumulatively summing occurrences
    event_number = event_occurred.cumsum()

    # Get the last event date for each row using forward fill
    last_event_date = df["date"].where(event_occurred).ffill()

    # Compute the days since the last event
    df["time_since_last_eq"] = (df["date"] - last_event_date).dt.days

    return df


def compute_time_since_last_eq_per_class(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    For each possible class (determined from daily_class_count_* columns),
    create a column `time_since_class_{c}` that increments daily unless
    the day has >=1 quake in that class.
    """

    df = df_daily.sort_values("date").copy()

    # Identify the highest class number dynamically
    class_cols = [col for col in df.columns if re.match(r"daily_class_count_\d+", col)]
    class_numbers = [int(col.split("_")[-1]) for col in class_cols]

    if not class_numbers:
        return df  # No class count columns, return unchanged

    max_class = max(class_numbers)

    for c in range(max_class + 1):
        col_name = f"time_since_class_{c}"
        daily_col = f"daily_class_count_{c}"

        if daily_col not in df.columns:
            df[daily_col] = 0  # Ensure missing class columns default to zero

        # Identify where events occurred in this class
        event_occurred = df[daily_col] > 0

        # Get the last event date for each row using forward fill
        last_event_date = df["date"].where(event_occurred).ffill()

        # Compute days since last event
        df[col_name] = (df["date"] - last_event_date).dt.days.fillna(0).astype(int)

    return df


def compute_rolling_b_value_daily(
    df_daily: pd.DataFrame, days_window: int = 30
) -> pd.DataFrame:
    """
    For each day, gather all magnitudes from the past `days_window` days (including current day),
    flatten them, and compute a b-value. Store in a column 'daily_b_value'.
    """
    df = df_daily.sort_values("date").copy()
    b_values = []
    for i in range(len(df)):
        # current day
        current_day = df.loc[i, "date"]
        start_day = current_day - pd.Timedelta(days=days_window - 1)
        # subset the df between [start_day, current_day]
        subset = df[(df["date"] >= start_day) & (df["date"] <= current_day)]
        # flatten magnitudes
        all_mags = []
        for mags_list in subset["magnitudes_list"]:
            all_mags.extend(mags_list)
        bval = compute_b_value(all_mags)
        b_values.append(bval)
    df["daily_b_value"] = b_values
    return df


def compute_additional_time_series_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Typical ML-friendly time series features on daily data, such as:
      - daily_count_diff, daily_max_diff
      - rolling sums/means for daily_count, daily_max
    Example windows: 7-day, 30-day
    """
    df = df_daily.sort_values("date").copy()
    df["daily_count_diff"] = df["daily_count"].diff()
    df["daily_max_diff"] = df["daily_max"].diff()

    # 7-day rolling
    df["daily_count_7d_sum"] = df["daily_count"].rolling(7).sum()
    df["daily_max_7d_mean"] = df["daily_max"].rolling(7).mean()

    # 30-day rolling
    df["daily_count_30d_sum"] = df["daily_count"].rolling(30).sum()
    df["daily_max_30d_mean"] = df["daily_max"].rolling(30).mean()

    return df


# -------------------------------------------------------------------
# 4. TARGET CREATION (DAILY)
# -------------------------------------------------------------------


def create_daily_target(df_daily: pd.DataFrame, target_params: Dict) -> pd.DataFrame:
    """
    max_mag_next_{N}d: the maximum daily_max in the next N days (strictly after the current day).
    If bin_edges in target_params, also produce 'target_class'.

    We'll use 'daily_max' to represent that day's largest quake magnitude.
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
# 5. MAIN PIPELINE
# -------------------------------------------------------------------


def main():
    # 1. Read config
    config_file = "../../../config/10-features-config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["features_config"]

    input_data_path = config["input_data_path"]
    output_data_path = config["output_data_path"]

    # We assume bin_edges might come from target_params, for example
    target_params = config["target_params"]
    bin_edges = target_params.get("magnitude_bin_edges", [])

    # Some generic feature params
    feature_params = config["feature_params"]
    max_distance_km = feature_params.get("max_distance_km", 50)
    b_value_rolling_window_days = feature_params.get("b_value_rolling_window_days", 30)

    # 2. Read event-based data
    df_events = pd.read_parquet(input_data_path, engine="pyarrow")

    # 3. Convert magnitudes => ML, filter by distance
    df_events["magnitude_ml"] = df_events.apply(convert_magnitude, axis=1)
    df_events.dropna(subset=["magnitude_ml"], inplace=True)
    df_events = filter_by_distance(df_events, max_distance_km)

    # 4. Determine daily date range
    min_date = df_events["time_utc"].min().floor("D")
    max_date = df_events["time_utc"].max().floor("D")

    # 5. Convert events to daily
    df_daily = convert_events_to_daily(
        df_events, start_date=min_date, end_date=max_date, bin_edges=bin_edges
    )

    # 6. Compute daily-based features
    #    6a) "time since last eq"
    df_daily = compute_time_since_last_eq(df_daily)

    #    6b) time since last eq for each magnitude class
    #        (One column per class: time_since_class_0, time_since_class_1, etc.)
    if bin_edges:
        df_daily = compute_time_since_last_eq_per_class(df_daily)

    #    6c) daily b-value (rolling window in days)
    #        We'll gather the past X days of magnitudes_list and compute b-value
    df_daily = compute_rolling_b_value_daily(
        df_daily, days_window=b_value_rolling_window_days
    )

    #    6d) Additional typical time series features
    df_daily = compute_additional_time_series_features(df_daily)

    # 7. Create daily-level target (max magnitude in next N days)
    df_daily = create_daily_target(df_daily, target_params)

    # 8. Save final daily DataFrame
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df_daily.to_parquet(output_data_path, index=False, engine="pyarrow")

    print(f"Daily feature table saved to: {output_data_path}")
    print(f"Total rows in daily feature table: {len(df_daily)}")


if __name__ == "__main__":
    main()
