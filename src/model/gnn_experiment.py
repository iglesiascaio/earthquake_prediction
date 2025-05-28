# ==========================================================
# 0. Imports
# ==========================================================
import warnings, numpy as np, pandas as pd
from typing import List, Tuple

warnings.filterwarnings("ignore")

import torch
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)


# ----------------------------------------------------------
# 1.  Spatial-graph construction (undirected)
# ----------------------------------------------------------
def build_radius_graph(
    lat: np.ndarray,
    lon: np.ndarray,
    radius_km: float = 300.0,
    k_fallback: int = 8,
) -> torch.Tensor:
    """Return undirected edge_index for stations ordered exactly as ``lat, lon``."""
    R = 6371.0
    lat_r, lon_r = np.radians(lat[:, None]), np.radians(lon[:, None])
    dlat, dlon = lat_r.T - lat_r, lon_r.T - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # N×N

    edges: List[Tuple[int, int]] = []
    for i in range(len(lat)):
        nbrs = np.where((dist[i] <= radius_km) & (dist[i] > 0))[0]
        if nbrs.size == 0:
            nbrs = np.argsort(dist[i])[1 : k_fallback + 1]
        edges.extend([(i, j) for j in nbrs])

    edges = list(set(edges + [(j, i) for (i, j) in edges]))  # symmetrise + dedup
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ----------------------------------------------------------
# 2.  PyG Dataset  (uses **meta.node_index**)
# ----------------------------------------------------------
class EarthquakeGraphDataset(Dataset):
    """Graph snapshot per day; standardisation uses train-split stats only."""

    def __init__(
        self,
        df_feat: pd.DataFrame,
        df_meta: pd.DataFrame,  # MUST contain 'node_index'
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

        # deterministic mapping station_code → node_index
        self.code2idx = dict(zip(self.meta.station_code, self.meta.node_index))
        self.N = len(self.meta)

    def __len__(self):
        return len(self.dates)

    def _standardise(self, a):
        return (a - self.mu) / self.sig

    def __getitem__(self, idx):
        # filter the date
        day = self.dates[idx]
        rows = self.df[self.df.date == day]

        # initialise x and y
        x = np.tile(self.med, (self.N, 1)).astype(np.float32)
        y = np.full(self.N, -1, dtype=np.int64)

        # update x and y for each station
        for _, r in rows.iterrows():
            j = self.code2idx[r.station_code]
            x[j] = r[self.feats].astype(np.float32).values
            y[j] = int(r[self.tgt])

        x = np.where(np.isnan(x), self.med, x)
        x = self._standardise(x)

        # if missing target, fill with mode
        if (y == -1).any():
            mode = np.bincount(y[y != -1]).argmax()
            y[y == -1] = mode

        return Data(
            x=torch.from_numpy(x),
            edge_index=self.edge_index,
            y=torch.from_numpy(y),
            date=str(day.date()),
        )


# ----------------------------------------------------------
# 3.  GraphSAGE model
# ----------------------------------------------------------
class StationGNN(nn.Module):
    def __init__(self, in_dim, hidden=128, n_layers=3, n_classes=4):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden)])
        self.convs.extend(SAGEConv(hidden, hidden) for _ in range(n_layers - 2))
        self.convs.append(SAGEConv(hidden, hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, n_classes)
        )

    def forward(self, data):
        x, edge = data.x, data.edge_index
        for conv in self.convs:
            x = torch.relu(conv(x, edge))
        return self.head(x)


# ----------------------------------------------------------
# 4.  Data loading / split
# ----------------------------------------------------------
def load_tables(feat_path, meta_path, feat_cols):
    df = pd.read_parquet(feat_path).drop(columns=["magnitudes_list"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["target_class"])
    df["target_class"] = df["target_class"].astype(int) - 1

    n_classes = df["target_class"].nunique()
    assert set(df["target_class"].unique()) == set(range(n_classes))

    meta = (
        pd.read_parquet(meta_path)[
            ["station_code", "station_latitude", "station_longitude"]
        ]
        .drop_duplicates()
        .sort_values("station_code")
        .reset_index(drop=True)
    )
    meta["node_index"] = np.arange(len(meta))  # explicit index
    return df, meta, n_classes


def split_dates(df, cutoff="2024-01-01"):
    train = sorted(df.loc[df.date < cutoff, "date"].unique())
    test = sorted(df.loc[df.date >= cutoff, "date"].unique())
    return train, test


# ----------------------------------------------------------
# 5.  Train / eval helpers
# ----------------------------------------------------------
def train_epoch(model, loader, optim, loss_fn):
    model.train()
    tot = 0
    for data in loader:
        data = data.to(DEVICE)
        optim.zero_grad()
        loss = loss_fn(model(data), data.y)
        loss.backward()
        optim.step()
        tot += loss.item() * data.num_nodes
    return tot / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    p_logits, t = [], []
    for data in loader:
        data = data.to(DEVICE)
        logits = model(data)
        p_logits.append(logits.cpu().numpy())
        t.append(data.y.cpu().numpy())

    y_score = np.concatenate(p_logits)  # shape [N, C]
    y_true = np.concatenate(t)  # shape [N]
    y_pred = y_score.argmax(axis=1)

    # Accuracy and confusion matrix
    print(f" accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(" Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    # Binarize labels
    n_classes = y_score.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    try:
        # Average AUC
        auc_macro = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")
        print(f" average ROC AUC (macro): {auc_macro:.4f}")

        # Plot per-class ROC curves
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("One-vs-Rest ROC Curves")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except ValueError as e:
        print(f" AUC could not be computed: {e}")


# ----------------------------------------------------------
# 6.  Main
# ----------------------------------------------------------
if __name__ == "__main__":
    FEAT_PATH = "../../data/features/earthquake_features.parquet"
    META_PATH = "../../data/raw/earthquake_data.parquet"
    FEATS = [
        "time_since_class_4",
        "time_since_class_3",
        "rolling_T_value",
        "daily_count_30d_sum",
        "daily_b_value",
        "rolling_dE_half",
        "daily_etas_intensity",
        "time_since_class_2",
        "daily_count_7d_sum",
    ]
    EPOCHS, BATCH, LR, HID, RADIUS = 40, 16, 1e-3, 32, 100.0

    # full load
    df, meta, C = load_tables(FEAT_PATH, META_PATH, FEATS)

    # train/test split
    tr_dates, te_dates = split_dates(df)

    # train-only stats
    train_df = df[df.date.isin(tr_dates)]
    med = train_df[FEATS].median().astype(np.float32).values
    mu = train_df[FEATS].mean().astype(np.float32).values
    sig = train_df[FEATS].std(ddof=0).replace(0, 1).astype(np.float32).values

    # graph (order matches meta.node_index)
    edge_index = build_radius_graph(
        meta.station_latitude.values, meta.station_longitude.values, radius_km=RADIUS
    )

    ds_tr = EarthquakeGraphDataset(
        df, meta, FEATS, "target_class", med, mu, sig, edge_index, tr_dates
    )
    ds_te = EarthquakeGraphDataset(
        df, meta, FEATS, "target_class", med, mu, sig, edge_index, te_dates
    )

    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False)

    model = StationGNN(len(FEATS), HID, 3, C).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss = nn.CrossEntropyLoss()

    # # Compute class weights inversely proportional to class frequencies
    # class_counts = train_df["target_class"].value_counts().sort_index().values
    # class_weights = 1.0 / class_counts
    # class_weights = class_weights / class_weights.sum()  # normalize
    # loss = nn.CrossEntropyLoss(
    #     weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    # )

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, dl_tr, optim, loss)
        print(f"Epoch {ep:3} | loss {tr_loss:.4f}", end="", flush=True)
        if ep % 5 == 0 or ep == EPOCHS:
            print(" | test:", end="", flush=True)
            evaluate(model, dl_te)
        else:
            print(flush=True)

    print("\\nFinal metrics:")
    evaluate(model, dl_te)
