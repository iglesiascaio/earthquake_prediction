import os
import logging
from obspy import read, Stream, Trace
from typing import Optional, Dict
import numpy as np


def combine_streams(window_len: int, shift: Optional[int] = None) -> None:
    """
    Combine daily processed *.mseed files into multi-day streams.
    Creates a single trace spanning the entire window by handling location code
    conflicts - ensures all traces have the same location code before merging.

    Parameters
    ----------
    window_len : int
        Number of consecutive days to merge per output file.
    shift : int, optional
        Number of days to advance the window before the next merge.
        • shift == window_len  →  non-overlapping windows (default)
        • shift  < window_len  →  overlapping / sliding windows
        • shift  > window_len  →  gapped windows
    """
    if shift is None:
        shift = window_len
    if shift <= 0 or window_len <= 0:
        raise ValueError("`window_len` and `shift` must be positive integers")

    base_dir  = "/home/gridsan/mknuth/01_Seismic_Wave_Data_Prediction/01_Data/01_Seismic_Wave_Data"
    channels  = ("BHE", "BHN", "BHZ")
    out_dir   = os.path.join(base_dir, f"Combined_Processed_Streams_{window_len}_new_2025_07_28")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    for year in sorted(os.listdir(base_dir)):
        print(year)
        year_path = os.path.join(base_dir, year)
        if not os.path.isdir(year_path):
            continue

        for station in sorted(os.listdir(year_path)):
            station_path = os.path.join(year_path, station)
            if not os.path.isdir(station_path):
                continue

            for channel in channels:
                chan_path = os.path.join(station_path, channel)
                if not os.path.isdir(chan_path):
                    continue      # channel missing for this station-year

                # ------------------------------------------------------
                #  Gather daily files for this station-year-channel
                # ------------------------------------------------------
                daily_files = [
                    os.path.join(chan_path, f)
                    for f in os.listdir(chan_path)
                    if f.endswith("_processed.mseed")
                ]
                daily_files.sort()            # YYYY-MM-DD in name → lexical = chrono
                n_files = len(daily_files)
                if n_files == 0:
                    continue

                # ------------------------------------------------------
                #  Plan windows (sliding / gapped / non-overlapping)
                # ------------------------------------------------------
                start_idxs   = range(0, n_files - window_len + 1, shift)
                n_windows    = len(list(start_idxs))
                last_used    = (list(start_idxs)[-1] + window_len) if n_windows else 0
                leftovers    = n_files - last_used

                print(
                    f"Processing {year}  {station}  {channel}  "
                    f"→ {n_files} files   "
                    f"({n_windows} window(s) of {window_len} d, shift {shift} d, "
                    f"{leftovers} leftover)"
                )

                # ------------------------------------------------------
                #  Merge each window
                # ------------------------------------------------------
                for w, start in enumerate(start_idxs, 1):
                    group_files = daily_files[start : start + window_len]
                    try:
                        # First, determine the dominant location code
                        loc_count: Dict[str, int] = {}
                        all_traces = []
                        
                        for fp in group_files:
                            st = read(fp)
                            for tr in st:
                                all_traces.append(tr)
                                # Handle empty location field (None, empty string, etc.)
                                loc = tr.stats.location if tr.stats.location else ""
                                loc_count[loc] = loc_count.get(loc, 0) + 1
                        
                        # Find most common location
                        if not loc_count:
                            raise ValueError("No traces found in window files")
                            
                        dominant_loc = max(loc_count, key=loc_count.get)
                        # Format location for logging - show empty as "(empty)"
                        display_loc = dominant_loc if dominant_loc else "(empty)"
                        logging.info(f"Location counts: {loc_count}, using dominant location: {display_loc}")
                        
                        # Set all traces to the dominant location
                        st_uniform = Stream()
                        for tr in all_traces:
                            # Create a new trace with consistent location
                            new_tr = Trace(data=tr.data)
                            # Copy all stats
                            new_tr.stats = tr.stats.copy()
                            # Override location with dominant one
                            # Empty strings are normalized to empty string
                            new_tr.stats.location = dominant_loc if dominant_loc else ""
                            st_uniform.append(new_tr)
                        
                        # Now we can merge as usual
                        st_merged = st_uniform.merge(method=1, fill_value="interpolate")

                        first_date = os.path.basename(group_files[0]).split("_")[2]
                        last_date  = os.path.basename(group_files[-1]).split("_")[2]

                        out_name = (
                            f"{station}_{channel}_{first_date}_to_{last_date}.mseed"
                        )
                        
                        st_merged.write(os.path.join(out_dir, out_name), format="MSEED")
                        logging.info(
                            "Saved %s (unified location: %s, from %d traces)", 
                            out_name, display_loc, len(all_traces)
                        )

                    except Exception as exc:
                        logging.error(
                            "Error combining %s – %s: %s",
                            group_files[0],
                            group_files[-1],
                            exc,
                        )

                # ------------------------------------------------------
                #  Summary print for this triplet
                # ------------------------------------------------------
                if n_windows:
                    print(
                        f"  ✔  Wrote {n_windows} combined streams; "
                        f"skipped {leftovers} leftover day(s)\n"
                    )
                else:
                    print("  ⚠  Not enough data for a single window; nothing saved\n")
                    
                    
                    
                    
                    
if __name__ == "__main__":
    # Examples
    # combine_streams(window_len=10)            # original behaviour (10-day, no overlap)
    # combine_streams(window_len=10, shift=5)   # 10-day windows every 5 days
    combine_streams(window_len=50, shift=7)
