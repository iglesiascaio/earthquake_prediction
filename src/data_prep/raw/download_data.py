#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
download_data.py

Reads config from `../../config/00-download-config.yaml` (under `download_config`),
fetches earthquake catalogs from SCEDC or IRIS as fallback,
optionally pulls station metadata for a single station,
and saves a combined DataFrame to Parquet.

New: We chunk the query into monthly intervals to avoid timeouts.
"""

import os
import yaml
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from datetime import timedelta


def get_station_metadata(fdsn_client, network_code, station_code):
    """
    Retrieve station metadata (latitude, longitude, elevation) from the FDSN client.
    Returns (station_lat, station_lon, station_elev_m). If not found, returns (None, None, None).
    """
    try:
        inv = fdsn_client.get_stations(
            network=network_code, station=station_code, level="station"
        )
        stn = inv[0][0]  # first network, first station
        return (stn.latitude, stn.longitude, stn.elevation)
    except Exception:
        # If station query fails, return None
        return (None, None, None)


def fetch_events_in_range(fdsn_client, start_time, end_time, min_mag=1.0, max_mag=9.0):
    """
    Fetch an event catalog for a given time range from the given FDSN client.
    We do a basic try/except to handle any timeouts or server errors.
    Returns an ObsPy Catalog or None if an error.
    """
    try:
        cat = fdsn_client.get_events(
            starttime=start_time,
            endtime=end_time,
            minmagnitude=min_mag,
            maxmagnitude=max_mag,
        )
        return cat
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
    Fetch earthquake catalogs from SCEDC or IRIS as fallback, but in smaller chunks.
    - Splits the time range into intervals of `chunk_days`.
    - Optionally queries station metadata for the given network_code & station_code.
    Returns a pandas DataFrame with enriched event information from all chunks combined.
    """

    # 1) Try SCEDC first, fallback to IRIS
    try:
        fdsn_client = Client("SCEDC")  # SoCal Earthquake Data Center
    except:
        print("[WARN] SCEDC unavailable, falling back to IRIS.")
        fdsn_client = Client("IRIS")

    # 2) Fetch station metadata
    station_lat, station_lon, station_elev = get_station_metadata(
        fdsn_client, network_code, station_code
    )

    # 3) Generate chunked time intervals
    #    We'll break [start_time, end_time] into intervals of `chunk_days` length
    chunk_catalogs = []
    cur_start = start_time
    while cur_start < end_time:
        cur_end = cur_start + chunk_days * 86400  # chunk_days in seconds
        if cur_end > end_time:
            cur_end = end_time

        # Fetch the events for this chunk
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

    # 4) Combine all chunked catalogs into a single ObsPy Catalog
    from obspy import Catalog

    combined_catalog = Catalog()
    for c in chunk_catalogs:
        combined_catalog.extend(c)

    # 5) Transform combined catalog into a dict of records
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
            # Station metadata
            "station_latitude": station_lat,
            "station_longitude": station_lon,
            "station_elevation_m": station_elev,
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df


def main():
    """
    Main function:
     - Reads config from 00-download-config.yaml (../../config/00-download-config.yaml)
     - Uses config parameters to fetch earthquakes in chunks (to avoid timeouts)
     - Saves DataFrame to Parquet
    """

    # 1. Load config
    config_file = "../../../config/00-download-config.yaml"
    with open(config_file, "r", encoding="utf-8") as f:
        all_config = yaml.safe_load(f)
    config = all_config["download_config"]

    # 2. Parse parameters
    start_time = UTCDateTime(config["start_time"])
    end_time = UTCDateTime(config["end_time"])
    min_magnitude = config["min_magnitude"]
    max_magnitude = config["max_magnitude"]
    network_code = config["network"]
    station_code = config["station"]

    raw_data_dir = config["data_paths"]["raw_data"]
    output_file_name = config["output_file_name"]
    output_file_path = os.path.join(raw_data_dir, output_file_name)

    # 3. Ensure the raw data directory exists
    os.makedirs(raw_data_dir, exist_ok=True)

    # 4. Fetch the event data in smaller increments (e.g., 30 days)
    #    You can adjust chunk_days in your config or hardcode it
    chunk_days = 30  # or 15, or 7, etc., to reduce each query size

    df_quakes = get_earthquake_data_chunked(
        start_time=start_time,
        end_time=end_time,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        network_code=network_code,
        station_code=station_code,
        chunk_days=chunk_days,
    )

    print(f"[INFO] Total events downloaded (combined): {len(df_quakes)}")

    # 5. Save DataFrame to Parquet
    df_quakes.to_parquet(output_file_path, index=False, engine="pyarrow")

    print(f"[INFO] Data successfully saved to: {output_file_path}")


if __name__ == "__main__":
    main()
