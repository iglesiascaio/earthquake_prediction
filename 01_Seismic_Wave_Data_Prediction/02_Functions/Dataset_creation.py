from torch.utils.data import Dataset, DataLoader
import torch
from obspy import read
import numpy as np
import os
import pandas as pd
from datetime import datetime
from obspy import UTCDateTime
import torch.nn.functional as F
from collections import defaultdict

class SeisLMWindowedDataset(Dataset):
    def __init__(self, train_data, earthquake_csv, window_size_days=1, use_tabular_features=False, downsampling_rate = 8):
        """
        Dataset that splits seismograms into windows of specified size in days.
        
        Args:
            train_data: List of dictionaries with file paths and labels
            earthquake_csv: Path to the earthquake dataframe with additional features
            window_size_days: Size of each window in days
            use_tabular_features: Whether to include tabular features from dataframe
        """
        print("Initializing SeisLMWindowedDataset")
        self.train_data = train_data
        self.window_size_days = window_size_days
        print(f"Window size (in days): {self.window_size_days}")
        self.use_tabular_features = use_tabular_features
        self.downsampling_rate = downsampling_rate
        print(f"Using downsampling rate: {self.downsampling_rate}")
        
        # Load the earthquake dataframe with features if needed
        if self.use_tabular_features:
            print("Loading tabular features from earthquake_csv")
            
            self.earthquake_df = pd.read_parquet(earthquake_csv)
            
            # Ensure the date column is datetime type
            self.earthquake_df["date"] = pd.to_datetime(self.earthquake_df["date"])
            
            # Get the list of feature columns (excluding date and target columns)
            self.feature_columns = [col for col in self.earthquake_df.columns 
                                   if col not in ["date", "max_mag_next_30d", "Converted Magnitude"]]
            
            print(f"Tabular feature columns: {self.feature_columns}")

    def __len__(self):
        return len(self.train_data)
    
    def pad_to_multiple_of_four(self, data):
        data_t = torch.tensor(data, dtype=torch.float32)
        length = data_t.shape[0]
        if length % 4 != 0:
            padding_length = 4 - (length % 4)
            data_t = F.pad(data_t, (0, padding_length), mode='constant', value=0)
        return data_t

    def get_tabular_features(self, end_date):
        """Extract tabular features for a given date from the dataframe"""
        # Convert end_date to pandas datetime if it's not already
        end_date = pd.to_datetime(end_date)
        
        # Find the matching row in the dataframe
        row = self.earthquake_df[self.earthquake_df["date"] == end_date]
        
        if row.empty:
            print(f"Warning: No matching row found for date {end_date}. Using zeros for tabular features.")
            # Return zeros if no match is found
            return torch.zeros(len(self.feature_columns), dtype=torch.float32)
        
        # Extract the features from the row
        features = row[self.feature_columns].values[0]
        return torch.tensor(features, dtype=torch.float32)
    
    def __getitem__(self, idx):
        item = self.train_data[idx]
        channel_windows = []
        #print(item['file_paths'])
        period_start, period_end = None, None

        for file_path in item['file_paths']:
            try:
                # Load stream
                stream = read(file_path)
                
                if period_start is None:
                    period_start = stream[0].stats.starttime
                    period_end = stream[0].stats.endtime
                    station_name = stream[0].stats.station
                

                # Trim the stream to end at the beginning of the last day
                last_day_start = UTCDateTime(stream[0].stats.endtime.date)
                #print(f"Stream end date: {last_day_start}")
                stream[0].trim(endtime=last_day_start - 1e-6)

                # Decimate the stream (reduce sampling rate by factor of 8 if desired)
                stream.decimate(self.downsampling_rate)

                # Convert to tensor
                data_tensor = torch.tensor(stream[0].data, dtype=torch.float32)

                # Standardize
                #print("Standardizing data")
                mean = data_tensor.mean()
                std = data_tensor.std()
                std_value = std.item() if std.item() != 0 else 1
                standardized_data = (data_tensor - mean) / std_value
                #print("Standardized data succesful")

                # Determine the sampling rate after decimation
                # e.g., original 40Hz => after decimate(8) => effectively 5Hz
                current_sr = stream[0].stats.sampling_rate

                # Calculate how many samples in one day at the current sampling rate
                samples_per_day = int(current_sr * 86400)
                #print(f"Samples per day {samples_per_day}")
                # Now multiply by the number of days to get our window size in samples
                window_size_samples = int(samples_per_day * self.window_size_days)

                # Calculate number of windows
                data_length = len(standardized_data)
                num_windows = max(1, data_length // window_size_samples)

                # Create windows for this channel
                windows = []
                for i in range(num_windows):
                    start_idx = i * window_size_samples
                    end_idx = min(start_idx + window_size_samples, data_length)

                    window_data = standardized_data[start_idx:end_idx]

                    # If last window is smaller, pad it
                    if len(window_data) < window_size_samples:
                        padding = window_size_samples - len(window_data)
                        window_data = F.pad(window_data, (0, padding), 'constant', 0)

                    windows.append(window_data)

                # Stack windows for this channel into a tensor: [num_windows, window_size]
                channel_windows.append(torch.stack(windows))

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                # Create dummy windows for this channel
                dummy_windows = torch.zeros((1, 1), dtype=torch.float32)
                channel_windows.append(dummy_windows)

        # Ensure all channels have the same number of windows
        min_windows = min(windows.shape[0] for windows in channel_windows)
        #print(f"Channel window counts: {[w.shape[0] for w in channel_windows]}")
        #print(f"Using minimum number of windows: {min_windows}")

        # Truncate to the minimum number of windows
        channel_windows = [windows[:min_windows] for windows in channel_windows]
        #print(f"After truncation: {[w.shape for w in channel_windows]}")

        # Stack channels to get [num_channels, num_windows, window_size]
        windows_tensor = torch.stack(channel_windows)
        #print(f"After stacking: Shape = {windows_tensor.shape} "
              #f"[num_channels, num_windows, window_size]")

        # Reshape to [num_windows, num_channels, window_size]
        windows_tensor = windows_tensor.permute(1, 0, 2)
        #print(f"After permute: Shape = {windows_tensor.shape} "
              #f"[num_windows, num_channels, window_size]")

        # Original continuous label
        continuous_label = torch.tensor(item['label'], dtype=torch.float32)

        # Convert to class index
        class_label = magnitude_to_class(continuous_label.item())
        class_label = torch.tensor(class_label, dtype=torch.long) 
        
        sample_meta = {
            "station":      station_name,              # ← new
            "period_start": str(period_start),
            "period_end":   str(period_end),
            "label":        class_label.item()       # ← new (or class_label.item())
        }

        # Get tabular features if enabled
        if self.use_tabular_features:
            tabular_features = self.get_tabular_features(end_date)
            return windows_tensor, tabular_features, sample_meta, class_label
        else:
            return windows_tensor, sample_meta, class_label


def get_data_loader(train_data, earthquake_csv, window_size_days=1, batch_size=1, shuffle=False, use_tabular_features=False,downsampling_rate = 8):
    """
    Create a DataLoader for the windowed dataset.
    
    Args:
        train_data: List of dictionaries with file paths and labels
        earthquake_csv: Path to the earthquake dataframe with additional features
        window_size_days: Size of each window in days
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        use_tabular_features: Whether to include tabular features from dataframe
    """
    dataset = SeisLMWindowedDataset(
        train_data, 
        earthquake_csv,
        window_size_days=window_size_days,
        use_tabular_features=use_tabular_features,
        downsampling_rate = downsampling_rate
    )
    
    # Define a custom collate function to handle variable batch contents
    def custom_collate_fn(batch):
        if use_tabular_features:
            # 4-tuple from __getitem__: (waves, tab_feats, meta, label)
            waves, tab_feats, metas, labels = zip(*batch)

            waves  = torch.stack(waves)                       # [B, …]
            tabs   = torch.stack(tab_feats)                   # [B, F]
            labels = torch.tensor(labels, dtype=torch.long)   # [B]

            return waves, tabs, list(metas), labels           # ← metas kept
        else:
            # 3-tuple: (waves, meta, label)
            waves, metas, labels = zip(*batch)

            waves  = torch.stack(waves)                       # [B, …]
            labels = torch.tensor(labels, dtype=torch.long)

            return waves, list(metas), labels                 # ← metas kept

    # Always pass the collate_fn so behaviour is consistent
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn          # <- always
    )
    return loader

def magnitude_to_class(magnitude):
    """Convert continuous magnitude to discrete class index."""
    if magnitude < 1.5:
        return 0  # Low magnitude
    elif magnitude < 2.5:
        return 1  # Moderate magnitude
    elif magnitude < 3.5:
        return 2  # High magnitude
    else:
        return 3  # Very high magnitude
    

def get_max_magnitude_in_next_30_days(end_date, earthquake_csv):
    """
    Get the maximum earthquake magnitude in the next 30 days.
    """
    future_end_date = end_date + pd.Timedelta(days=30)
    df = pd.read_csv(earthquake_csv, parse_dates=["Time"])
    mask = (df["Time"] >= end_date) & (df["Time"] <= future_end_date)
    future_earthquakes = df[mask]

    if future_earthquakes.empty:
        return 0.0  # No event found

    max_magnitude = future_earthquakes["Converted Magnitude"].max()
    return max_magnitude

def get_max_magnitude_in_next_30_days_Caio_dataset(
    end_date: datetime,
    station: str,
    earthquake_csv: str,
):
    """
    Fetch max_mag_next_30d for the given end_date/station.
    If the exact station code is absent, try its 3-letter core (e.g. PASC→PAS);
    if that still fails, fall back to the PAS entry and warn the user.
    """
    df = pd.read_parquet(earthquake_csv)
    df["date"] = pd.to_datetime(df["date"])
    df['station_code'] = df['station_code'].replace({'PAS':'PASC'})
    end_date = pd.to_datetime(end_date)

    # --- Helper to query one station code -------------------------------------
    def _query(st_code: str):
        return df.loc[
            (df["date"] == end_date) & (df["station_code"] == st_code),
            "max_mag_next_30d",
        ]
    
    
    #  try the full code coming from the wavestream filename
    row = _query(station)
    #print(row)


    # 3️⃣ final fallback → PAS (always present in Caio dataset)
    if row.empty or row.isna().all():
        row = _query("PASC")
        if row.empty:
            raise ValueError(
                f"No magnitude entry for '{station}' (or fallback 'PAS') "
                f"on {end_date.date()}"
            )
        print(
            f"[INFO] No magnitude found for station '{station}' on "
            f"{end_date.date()} – using PAS value instead."
        )

    return row.iloc[0]  # scalar (float) value



"""def create_train_dataset(earthquake_csv, combined_stream_dir):
    CHANNELS = ("BHE", "BHN", "BHZ")
   
    print("Creating train dataset...")
    # Base directory for combined streams
    print('directory')
    print(os.listdir(combined_stream_dir))
    
    channels = ["BHE", "BHN", "BHZ"]

    train_data = []

    # Separate lists for each channel
    bhe_files = []
    bhn_files = []
    bhz_files = []

    # Access the folder and categorize files
    for file_name in os.listdir(combined_stream_dir):
        if file_name.startswith("BHE"):
            bhe_files.append(file_name)
        elif file_name.startswith("BHN"):
            bhn_files.append(file_name)
        elif file_name.startswith("BHZ"):
            bhz_files.append(file_name)

    # Sort each list based on the start date extracted from the filename
    bhe_files.sort()
    bhn_files.sort()
    bhz_files.sort()

    # Combine the sorted lists to create triplets
    for bhe, bhn, bhz in zip(bhe_files, bhn_files, bhz_files):
        try:
            # Collect file paths for the current triplet
            file_paths = [
                os.path.join(combined_stream_dir, bhe),
                os.path.join(combined_stream_dir, bhn),
                os.path.join(combined_stream_dir, bhz)
            ]

            # Extract the end date from the filename
            end_date_str = bhe[-16:-6]  # Example: '2023-11-26'
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            # Get the label (maximum magnitude in the next 30 days)
            max_magnitude = get_max_magnitude_in_next_30_days_Caio_dataset(end_date, earthquake_csv)

            # Append the file paths and label to the dataset
            train_data.append({
                "file_paths": file_paths,
                "label": max_magnitude
            })

        except Exception as e:
            print(f"Error processing triplet: {bhe}, {bhn}, {bhz}, Error: {e}")

    print(f"Number of training samples: {len(train_data)}")
    #print(train_data)
    return train_data"""


# channels we care about
CHANNELS = ("BHE", "BHN", "BHZ")

def create_train_dataset_new(earthquake_csv: str, combined_stream_dir: str):
    """
    Build a list of {file_paths: [BHE, BHN, BHZ], label: float} dictionaries.
    One entry per (station, year, window) triplet that has data for all channels.
    """
    print("Creating train dataset …")
    print("Folder contents:", len(os.listdir(combined_stream_dir)), "files")

    # ------------------------------------------------------------------
    # 1. Collect files → groups[(station, year)][channel] = [filename, …]
    # ------------------------------------------------------------------
    groups = defaultdict(lambda: defaultdict(list))

    for fname in os.listdir(combined_stream_dir):
        if not fname.endswith(".mseed"):
            continue

        try:
            station, channel, start_date, *_ = fname.split("_")
            if channel not in CHANNELS:
                continue
            year = start_date[:4]                # take year from start date
            groups[(station, year)][channel].append(fname)
        except ValueError:
            print("⚠  Skipping unrecognised filename:", fname)

    # ------------------------------------------------------------------
    # 2. Sort each channel list chronologically (start_date already in name)
    # ------------------------------------------------------------------
    for (station, year), ch_dict in groups.items():
        for ch in CHANNELS:
            ch_dict[ch].sort()

    # ------------------------------------------------------------------
    # 3. Build triplets only when every channel has a file at that index
    # ------------------------------------------------------------------
    train_data = []
    for (station, year), ch_dict in groups.items():
        n_triplets = min(len(ch_dict["BHE"]),
                         len(ch_dict["BHN"]),
                         len(ch_dict["BHZ"]))

        if n_triplets == 0:
            print(f"⚠  {station} {year}: incomplete channel data, skipped")
            continue

        print(f"{station} {year}: creating {n_triplets} samples")

        for i in range(n_triplets):
            bhe = ch_dict["BHE"][i]
            bhn = ch_dict["BHN"][i]
            bhz = ch_dict["BHZ"][i]

            # ------------------------------------------------------------------
            # File paths
            # ------------------------------------------------------------------
            paths = [os.path.join(combined_stream_dir, fn) for fn in (bhe, bhn, bhz)]
            #print(paths)

            # ------------------------------------------------------------------
            # STATION = up to first '_'  → 'BRE'
            station = bhe.split("_", 1)[0]
            
            # End-date comes after "_to_" in the filename
            # <station>_<channel>_<start>_to_<end>.mseed
            end_date_str = bhe.split("_to_")[-1].split(".")[0]   # e.g. "2024-12-30"
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            # ------------------------------------------------------------------
            # Label = max magnitude in the next 30 days
            # ------------------------------------------------------------------
            label = get_max_magnitude_in_next_30_days_Caio_dataset(end_date,station,
                                                                   earthquake_csv)

            train_data.append({"file_paths": paths,
                               "label": label})

    print("Finished – total samples:", len(train_data))
    return train_data