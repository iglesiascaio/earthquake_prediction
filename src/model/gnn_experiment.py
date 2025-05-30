# ==========================================================
# CONFIG FLAGS
# ==========================================================
USE_EMBEDDINGS = True  # include emb_* columns
FILTER_TO_COMMON = False  # only used when USE_EMBEDDINGS=False
# True  → keep only stations present in embeddings
# False → keep every station in tabular data

# ==========================================================
# 0. Imports
# ==========================================================
import warnings, numpy as np, pandas as pd, torch
from typing import List, Tuple

warnings.filterwarnings("ignore")
from torch import nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", DEVICE)


# ==========================================================
# 1.  Spatial graph
# ==========================================================
def build_radius_graph(lat, lon, radius_km=300.0, k_fallback=8):
    R = 6371.0
    lat_r, lon_r = np.radians(lat[:, None]), np.radians(lon[:, None])
    dlat, dlon = lat_r.T - lat_r, lon_r.T - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlon / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    edges = []
    for i in range(len(lat)):
        nbrs = np.where((dist[i] <= radius_km) & (dist[i] > 0))[0]
        if nbrs.size == 0:
            nbrs = np.argsort(dist[i])[1 : k_fallback + 1]
        edges.extend([(i, j) for j in nbrs])
    edges = list(set(edges + [(j, i) for (i, j) in edges]))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


# ==========================================================
# 2.  Dataset
# ==========================================================
class EarthquakeGraphDataset(Dataset):
    def __init__(
        self,
        df_feat,
        df_meta,
        feature_cols,
        target_col,
        med,
        mu,
        sig,
        edge_index,
        dates,
    ):
        super().__init__()
        self.df, self.meta = df_feat, df_meta
        self.feats, self.tgt = feature_cols, target_col
        self.med, self.mu, self.sig = med, mu, sig
        self.edge_index, self.dates = edge_index, dates
        self.code2idx = dict(zip(self.meta.station_code, self.meta.node_index))
        self.N = len(self.meta)

    def __len__(self):
        return len(self.dates)

    def _standardise(self, a):
        return (a - self.mu) / self.sig

    def __getitem__(self, idx):
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
            mode = np.bincount(y[y != -1]).argmax()
            y[y == -1] = mode
        return Data(
            x=torch.from_numpy(x),
            edge_index=self.edge_index,
            y=torch.from_numpy(y),
            date=str(self.dates[idx].date()),
        )


# ==========================================================
# 3.  GraphSAGE model
# ==========================================================
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
# 4.  Loading utilities (tabular ± embeddings)
# ----------------------------------------------------------
def load_tables(feat_path, emb_path, meta_path, tabular_cols):
    df_daily = pd.read_parquet(feat_path).drop(columns=["magnitudes_list"])
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    emb_cols = []
    if USE_EMBEDDINGS:
        df_emb_raw = pd.read_pickle(emb_path)
        emb_dim = len(df_emb_raw.iloc[0]["embedding"])
        emb_cols = [f"emb_{i}" for i in range(emb_dim)]
        df_emb = pd.DataFrame(df_emb_raw["embedding"].tolist(), columns=emb_cols)
        df_seis = pd.concat([df_emb_raw.drop(columns=["embedding"]), df_emb], axis=1)
        df_seis["period_end"] = pd.to_datetime(df_seis["period_end"]).dt.tz_convert(
            None
        )
        df_seis["period_end"] = pd.to_datetime(df_seis["period_end"].dt.date)
        df = pd.merge(
            df_daily,
            df_seis.drop(columns=["period_start", "label"]),
            left_on=["date", "station_code"],
            right_on=["period_end", "station"],
            how="inner",
        ).drop(columns=["period_end", "station"])
    else:
        df = df_daily.copy()

    df = df.dropna(subset=["target_class"])
    df["target_class"] = df["target_class"].astype(int) - 1
    n_classes = df["target_class"].nunique()

    meta_full = pd.read_parquet(meta_path)[
        ["station_code", "station_latitude", "station_longitude"]
    ].drop_duplicates()

    if USE_EMBEDDINGS or FILTER_TO_COMMON:
        meta = meta_full[meta_full.station_code.isin(df.station_code.unique())].copy()
    else:
        meta = meta_full[
            meta_full.station_code.isin(df_daily.station_code.unique())
        ].copy()

    meta = meta.sort_values("station_code").reset_index(drop=True)
    meta["node_index"] = np.arange(len(meta))

    full_feats = tabular_cols + emb_cols
    return df, meta, full_feats, n_classes


def split_dates(df, cutoff="2024-01-01"):
    train = sorted(df.loc[df.date < cutoff, "date"].unique())
    test = sorted(df.loc[df.date >= cutoff, "date"].unique())
    return train, test


def train_epoch(model, loader, optim, loss_fn):
    """One training epoch; returns average per-node cross-entropy on the *train* loader."""
    model.train()
    tot_loss, tot_nodes = 0.0, 0
    for data in loader:
        data = data.to(DEVICE)
        optim.zero_grad()
        loss = loss_fn(model(data), data.y)  # mean over nodes in this graph
        loss.backward()
        optim.step()
        tot_loss += loss.item() * data.num_nodes
        tot_nodes += data.num_nodes
    return tot_loss / tot_nodes


# ==========================================================
# 5.  Loss helpers + full evaluation
# ==========================================================
def avg_cross_entropy(model, loader, loss_fn):
    model.eval()
    tot_loss = 0
    tot_nodes = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            l = loss_fn(model(data), data.y)
            tot_loss += l.item() * data.num_nodes
            tot_nodes += data.num_nodes
    return tot_loss / tot_nodes


@torch.no_grad()
def evaluate_full(model, loader):
    model.eval()
    logits, y_true = [], []
    for data in loader:
        data = data.to(DEVICE)
        logits.append(model(data).cpu().numpy())
        y_true.append(data.y.cpu().numpy())
    y_score = np.concatenate(logits)
    y_true = np.concatenate(y_true)
    y_pred = y_score.argmax(1)

    print(f" accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(" Confusion matrix:\n", confusion_matrix(y_true, y_pred))

    y_bin = label_binarize(y_true, classes=list(range(y_score.shape[1])))
    try:
        auc_macro = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")
        print(f" average ROC AUC: {auc_macro:.4f}")
    except ValueError:
        print("AUC could not be computed (missing class in this split).")


# ==========================================================
# 6.  Main
# ==========================================================
if __name__ == "__main__":
    FEAT_PATH = "../../data/features/earthquake_features.parquet"
    EMB_PATH = "../../data/features/embeddings_190102.pkl"
    META_PATH = "../../data/raw/earthquake_data.parquet"

    TABULAR_FEATS = [
        "time_since_class_3",
        "rolling_T_value",
        "daily_count_30d_sum",
        "daily_b_value",
        "rolling_dE_half",
        "daily_etas_intensity",
        "time_since_class_2",
        "daily_count_7d_sum",
    ]

    EPOCHS, BATCH, LR, HID, RADIUS = 60, 16, 1e-3, 8, 100.0

    # ------------- data ----------------
    df, meta, FEATS, C = load_tables(FEAT_PATH, EMB_PATH, META_PATH, TABULAR_FEATS)
    tr_dates, te_dates = split_dates(df)

    stats_df = df[df.date.isin(tr_dates)]
    med = stats_df[FEATS].median().astype(np.float32).values
    mu = stats_df[FEATS].mean().astype(np.float32).values
    sig = stats_df[FEATS].std(ddof=0).replace(0, 1).astype(np.float32).values

    edge_index = build_radius_graph(
        meta.station_latitude.values, meta.station_longitude.values, RADIUS
    )

    ds_tr = EarthquakeGraphDataset(
        df, meta, FEATS, "target_class", med, mu, sig, edge_index, tr_dates
    )
    ds_te = EarthquakeGraphDataset(
        df, meta, FEATS, "target_class", med, mu, sig, edge_index, te_dates
    )

    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False)

    # ------------- model --------------
    model = StationGNN(len(FEATS), HID, 3, C).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # ------------- training loop --------------
    for ep in range(1, EPOCHS + 1):
        tr_ce = train_epoch(model, dl_tr, optim, loss_fn)
        te_ce = avg_cross_entropy(model, dl_te, loss_fn)
        print(f"Epoch {ep:3} | train CE {tr_ce:.4f} | test CE {te_ce:.4f}", end="")
        if ep % 5 == 0 or ep == EPOCHS:
            print("  → detailed metrics:")
            evaluate_full(model, dl_te)
        else:
            print()

    print("\nFinal detailed test metrics:")
    evaluate_full(model, dl_te)
