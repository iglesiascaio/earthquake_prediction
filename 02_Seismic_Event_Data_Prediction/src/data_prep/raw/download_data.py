#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import pandas as pd
from obspy import UTCDateTime, Catalog
from obspy.clients.fdsn import Client

# ---------------------------------------------------------------------------
# 1. Station metadata helper (one tiny improvement)
# ---------------------------------------------------------------------------


def get_station_metadata(fdsn_client, network_code, station_code):
    """Retrieve metadata for one station."""
    for level in ("station", "channel"):  # ← NEW: second try at 'channel'
        try:
            inv = fdsn_client.get_stations(
                network=network_code, station=station_code, level=level
            )
            stn = inv[0][0]  # first network, first station
            return {
                "station_code": station_code,
                "station_latitude": stn.latitude,
                "station_longitude": stn.longitude,
                "station_elevation_m": stn.elevation,
            }
        except Exception:
            continue

    # still nothing → return None
    return {
        "station_code": station_code,
        "station_latitude": None,
        "station_longitude": None,
        "station_elevation_m": None,
    }


# ---------------------------------------------------------------------------
# 2. Event download once for all stations
# ---------------------------------------------------------------------------


def download_catalog_chunked(
    fdsn_client, start_time, end_time, min_mag, max_mag, chunk_days
):
    """Fetch one regional catalog in time chunks and return a single Catalog."""
    chunks = Catalog()
    cur = start_time
    daysec = 86400
    while cur < end_time:
        nxt = min(cur + chunk_days * daysec, end_time)
        try:
            cat = fdsn_client.get_events(
                starttime=cur,
                endtime=nxt,
                minmagnitude=min_mag,
                maxmagnitude=max_mag,
            )
            chunks.extend(cat)
        except Exception as e:
            print(f"[WARN] event fetch {cur}–{nxt}: {e}")
        cur = nxt
    return chunks


# ---------------------------------------------------------------------------
# 3. Build one DataFrame and replicate per station
# ---------------------------------------------------------------------------


def catalog_to_dataframe(catalog: Catalog) -> pd.DataFrame:
    """Convert an ObsPy Catalog to a bare-bones DataFrame (no station cols)."""
    recs = []
    for ev in catalog:
        if not ev.origins or not ev.magnitudes:
            continue
        o = ev.origins[0]
        m = ev.magnitudes[0]
        recs.append(
            dict(
                event_id=ev.resource_id.id,
                time_utc=o.time.datetime,
                latitude=o.latitude,
                longitude=o.longitude,
                depth_km=o.depth / 1000 if o.depth else None,
                magnitude=m.mag,
                magnitude_type=m.magnitude_type,
                event_type=str(ev.event_type) if ev.event_type else None,
            )
        )
    return pd.DataFrame.from_records(recs)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------


def main():
    # ------------------------------------------------------------------
    # read config
    # ------------------------------------------------------------------
    cfg_file = "../../../config/00-download-config.yaml"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["download_config"]

    start = UTCDateTime(cfg["start_time"])
    end = UTCDateTime(cfg["end_time"])
    min_mag = cfg["min_magnitude"]
    max_mag = cfg["max_magnitude"]
    net = cfg["network"]
    station_codes = cfg["stations"]  # ← stays in YAML
    chunk_days = cfg.get("chunk_days", 30)

    raw_dir = cfg["data_paths"]["raw_data"]
    os.makedirs(raw_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # connect once
    # ------------------------------------------------------------------
    try:
        cli = Client("SCEDC")
    except Exception:
        print("[WARN] SCEDC unavailable, falling back to IRIS.")
        cli = Client("IRIS")

    # ------------------------------------------------------------------
    # download ONE catalog for all stations
    # ------------------------------------------------------------------
    print("[INFO] downloading regional catalog …")
    catalog = download_catalog_chunked(cli, start, end, min_mag, max_mag, chunk_days)
    df_events = catalog_to_dataframe(catalog)
    print(f"[INFO] events in regional catalog: {len(df_events)}")

    # ------------------------------------------------------------------
    # replicate rows per station; attach metadata
    # ------------------------------------------------------------------
    frames = []
    for st in station_codes:
        meta = get_station_metadata(cli, net, st)
        if meta["station_latitude"] is None:
            print(f"[WARN] {st}: metadata not found; kept as None.")
        df_st = df_events.copy()
        df_st["station_code"] = st
        df_st["station_latitude"] = meta["station_latitude"]
        df_st["station_longitude"] = meta["station_longitude"]
        df_st["station_elevation_m"] = meta["station_elevation_m"]
        frames.append(df_st)

    df_all = pd.concat(frames, ignore_index=True)
    print(f"[INFO] total rows (events × stations): {len(df_all)}")

    out_path = os.path.join(raw_dir, cfg["output_file_name"])
    df_all.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[INFO] saved to {out_path}")


if __name__ == "__main__":
    main()
