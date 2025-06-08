#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helpers for earthquake GNN experiments.

This file contains ONLY graph-specific utilities; it depends on
PyTorch Geometric but *not* on scikit-learn or LightGBM.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


# ==========================================================
# 1.  Spatial graph utilities
# ==========================================================
def build_radius_graph(
    lat: np.ndarray,
    lon: np.ndarray,
    radius_km: float = 300.0,
    k_fallback: int = 8,
) -> torch.Tensor:
    """
    Undirected Haversine-radius graph.

    Returns:
        edge_index (2, 2|E|) LongTensor.
    """
    R = 6_371.0
    lat_r, lon_r = np.radians(lat[:, None]), np.radians(lon[:, None])
    dlat, dlon = lat_r.T - lat_r, lon_r.T - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    edges = []
    for i in range(len(lat)):
        nbrs = np.where((dist[i] <= radius_km) & (dist[i] > 0))[0]
        if nbrs.size == 0:  # isolated → k-NN fallback
            nbrs = np.argsort(dist[i])[1 : k_fallback + 1]
        edges.extend([(i, j) for j in nbrs])
    edges = list(set(edges) | {(j, i) for (i, j) in edges})
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def load_station_metadata(meta_path: str) -> pd.DataFrame:
    """
    Read metadata parquet → add consecutive `node_index`.
    """
    meta = (
        pd.read_parquet(meta_path)[
            ["station_code", "station_latitude", "station_longitude"]
        ]
        .drop_duplicates()
        .sort_values("station_code")
        .reset_index(drop=True)
    )
    meta["node_index"] = np.arange(len(meta))
    return meta


# ==========================================================
# 2.  Per-day graph dataset
# ==========================================================
class EarthquakeGraphDataset(Dataset):
    """
    One Data object per calendar day; node order follows `meta.node_index`.
    """

    def __init__(
        self,
        df_feat: pd.DataFrame,
        df_meta: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        med: np.ndarray,
        mu: np.ndarray,
        sig: np.ndarray,
        edge_index: torch.Tensor,
        dates: List[pd.Timestamp],
    ):
        super().__init__()
        self.df, self.meta = df_feat, df_meta
        self.feats, self.tgt = feature_cols, target_col
        self.med, self.mu, self.sig = med, mu, sig
        self.edge_index, self.dates = edge_index, dates
        self.code2idx = dict(zip(self.meta.station_code, self.meta.node_index))
        self.N = len(self.meta)

    # ------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.dates)

    def _standardise(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mu) / self.sig

    def __getitem__(self, idx: int) -> Data:
        day_rows = self.df[self.df.date == self.dates[idx]]
        x = np.tile(self.med, (self.N, 1)).astype(np.float32)
        y = np.full(self.N, -1, dtype=np.int64)

        for _, r in day_rows.iterrows():
            j = self.code2idx.get(r.station_code, None)
            if j is None:
                continue
            x[j] = r[self.feats].astype(np.float32).values
            y[j] = int(r[self.tgt])

        x = np.where(np.isnan(x), self.med, x)
        x = self._standardise(x)

        if (y == -1).any():
            mode = int(np.bincount(y[y != -1]).argmax())
            y[y == -1] = mode

        return Data(
            x=torch.from_numpy(x),
            edge_index=self.edge_index,
            y=torch.from_numpy(y),
            date=str(self.dates[idx].date()),
        )


# ==========================================================
# 3.  Convenience wrappers
# ==========================================================
def split_dates(df: pd.DataFrame, cutoff: str = "2024-01-01"):
    """Return sorted train/test date lists."""
    cut = pd.to_datetime(cutoff)
    dates = pd.to_datetime(df["date"])
    tr = sorted(dates[dates < cut].unique())
    te = sorted(dates[dates >= cut].unique())
    return tr, te


def normalisation_stats(
    df_train: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = df_train[feature_cols].median().astype(np.float32).values
    mu = df_train[feature_cols].mean().astype(np.float32).values
    sig = df_train[feature_cols].std(ddof=0).replace(0, 1).astype(np.float32).values
    return med, mu, sig


# ------------------------------------------------------------------
def make_dataloaders(
    df_full: pd.DataFrame,
    meta: pd.DataFrame,
    feature_cols: List[str],
    edge_index: torch.Tensor,
    batch_size: int,
    cutoff: str = "2024-01-01",
):
    """
    Build train / test DataLoaders for the GNN and return
    (dl_train, dl_test, n_classes, med, mu, sig).

    * Ensures labels are mapped to contiguous integers 0 … n_classes-1.
    """
    # -------- 0. contiguous label mapping ------------------------- #
    classes = sorted(df_full["target_class"].unique())
    class2idx = {c: i for i, c in enumerate(classes)}
    df_work = df_full.copy()
    df_work["target_idx"] = df_work["target_class"].map(class2idx)

    # -------- 1. train / test date splits ------------------------- #
    tr_dates, te_dates = split_dates(df_work, cutoff)

    # -------- 2. normalisation stats ------------------------------ #
    med, mu, sig = normalisation_stats(
        df_work[df_work.date.isin(tr_dates)], feature_cols
    )

    # -------- 3. datasets & loaders ------------------------------- #
    ds_tr = EarthquakeGraphDataset(
        df_work,
        meta,
        feature_cols,
        "target_idx",
        med,
        mu,
        sig,
        edge_index,
        tr_dates,
    )
    ds_te = EarthquakeGraphDataset(
        df_work,
        meta,
        feature_cols,
        "target_idx",
        med,
        mu,
        sig,
        edge_index,
        te_dates,
    )
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    n_classes = len(classes)
    return dl_tr, dl_te, n_classes, med, mu, sig
