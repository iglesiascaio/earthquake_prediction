#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import pandas as pd
from obspy import UTCDateTime, Catalog
from obspy.clients.fdsn import Client


def get_station_metadata(fdsn_client, network_code, station_code):
    """Retrieve metadata for one station."""
    try:
        inv = fdsn_client.get_stations(
            network=network_code, station=station_code, level="station"
        )
        stn = inv[0][0]  # first network, first station
        return {
            "station_code": station_code,
            "station_latitude": stn.latitude,
            "station_longitude": stn.longitude,
            "station_elevation_m": stn.elevation,
        }
    except Exception:
        return {
            "station_code": station_code,
            "station_latitude": None,
            "station_longitude": None,
            "station_elevation_m": None,
        }


def fetch_events_in_range(fdsn_client, start_time, end_time, min_mag=1.0, max_mag=9.0):
    """Fetch a catalog of events for a given time range."""
    try:
        return fdsn_client.get_events(
            starttime=start_time,
            endtime=end_time,
            minmagnitude=min_mag,
            maxmagnitude=max_mag,
        )
    except Exception as e:
        print(f"[WARN] Failed to fetch events from {start_time} to {end_time}: {e}")
        return None


def get_earthquake_data_chunked(
    start_time,
    end_time,
    min_magnitude=1.0,
    max_magnitude=9.0,
    network_code="CI",
    station_code="PAS",
    chunk_days=30,
):
    """
    Fetch earthquake events in chunks for a given station (if station filtering is enabled).
    Enriches the events with the station metadata.
    """
    try:
        fdsn_client = Client("SCEDC")
    except Exception:
        print("[WARN] SCEDC unavailable, falling back to IRIS.")
        fdsn_client = Client("IRIS")

    # Fetch station metadata for this station
    station_meta = get_station_metadata(fdsn_client, network_code, station_code)

    # Query events in chunked intervals
    chunk_catalogs = []
    cur_start = start_time
    seconds_per_day = 86400
    while cur_start < end_time:
        cur_end = cur_start + chunk_days * seconds_per_day
        if cur_end > end_time:
            cur_end = end_time

        cat_chunk = fetch_events_in_range(
            fdsn_client,
            start_time=cur_start,
            end_time=cur_end,
            min_mag=min_magnitude,
            max_mag=max_magnitude,
        )
        if cat_chunk is not None:
            chunk_catalogs.append(cat_chunk)
        cur_start = cur_end

    # Combine all chunks into a single catalog
    combined_catalog = Catalog()
    for cat in chunk_catalogs:
        combined_catalog.extend(cat)

    # Build event records and append station metadata
    records = []
    for event in combined_catalog:
        if not event.origins or not event.magnitudes:
            continue

        origin = event.origins[0]
        magnitude = event.magnitudes[0]

        record = {
            "event_id": event.resource_id.id,
            "time_utc": origin.time.datetime,
            "latitude": origin.latitude,
            "longitude": origin.longitude,
            "depth_km": origin.depth / 1000.0 if origin.depth else None,
            "magnitude": magnitude.mag,
            "magnitude_type": magnitude.magnitude_type,
            "event_type": str(event.event_type) if event.event_type else None,
            # Station metadata from current iteration
            "station_code": station_code,
            "station_latitude": station_meta["station_latitude"],
            "station_longitude": station_meta["station_longitude"],
            "station_elevation_m": station_meta["station_elevation_m"],
        }
        records.append(record)

    return pd.DataFrame(records)


def main():
    # 1. Load configuration
    config_file = "../../../config/00-download-config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        all_config = yaml.safe_load(f)
    config = all_config["download_config"]

    start_time = UTCDateTime(config["start_time"])
    end_time = UTCDateTime(config["end_time"])
    min_magnitude = config["min_magnitude"]
    max_magnitude = config["max_magnitude"]
    network_code = config["network"]

    # Here the config is assumed to have a "stations" list:
    station_codes = config["stations"]  # e.g., ["PAS", "ABC", "XYZ"]

    raw_data_dir = config["data_paths"]["raw_data"]
    os.makedirs(raw_data_dir, exist_ok=True)

    # 2. Loop over all stations to fetch and combine event data
    all_dfs = []
    for station_code in station_codes:
        print(f"[INFO] Processing station: {station_code}")
        df_station = get_earthquake_data_chunked(
            start_time=start_time,
            end_time=end_time,
            min_magnitude=min_magnitude,
            max_magnitude=max_magnitude,
            network_code=network_code,
            station_code=station_code,
            chunk_days=30,  # can also be set in the config
        )
        all_dfs.append(df_station)

    # Optionally, deduplicate events if the same event appears across stations.
    df_combined = pd.concat(all_dfs, ignore_index=True)
    print(f"[INFO] Total events downloaded (combined): {len(df_combined)}")

    output_file_path = os.path.join(raw_data_dir, config["output_file_name"])
    df_combined.to_parquet(output_file_path, index=False, engine="pyarrow")
    print(f"[INFO] Data successfully saved to: {output_file_path}")


if __name__ == "__main__":
    main()
