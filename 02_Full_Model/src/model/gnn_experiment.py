#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Earthquake GNN experiment — **minimal main** that relies on:

* utils.load_and_split_data  → shared preprocessing
* gnn_helpers                → graph-specific utilities
* utils.evaluate             → common evaluation
"""
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch_geometric.nn import SAGEConv

from .utils import load_and_split_data, evaluate
import src.model.utils.gnn_helper as gh

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility (same as hyperparameter tuning script)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import random
random.seed(RANDOM_SEED)

# Force CUDA to be deterministic for reproducible results
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class StationGNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden=128, n_layers=3, n_classes=4):
        super().__init__()
        self.convs = torch.nn.ModuleList([SAGEConv(in_dim, hidden)])
        self.convs.extend(SAGEConv(hidden, hidden) for _ in range(n_layers - 2))
        self.convs.append(SAGEConv(hidden, hidden))
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, data):
        x, edge = data.x, data.edge_index
        for conv in self.convs:
            x = torch.relu(conv(x, edge))
        return self.head(x)


from torch_geometric.nn import GATConv  # ← new import


class StationGAT(torch.nn.Module):
    """
    3-layer Graph Attention Network for node-wise classification
    (earthquake-risk classes per station–day).

    Args
    ----
    in_dim   : input feature size per node
    hidden   : hidden size per attention head
    n_layers : total graph layers (≥ 2)
    n_heads  : # attention heads per layer
    n_classes: # target classes
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        n_classes: int = 4,
    ):
        super().__init__()

        assert n_layers >= 2, "Need at least 2 GAT layers"

        # First layer: in_dim → hidden
        self.convs = torch.nn.ModuleList(
            [
                GATConv(
                    in_dim,
                    hidden,
                    heads=n_heads,
                    concat=True,  # output dim = hidden * n_heads
                    dropout=0.1,
                )
            ]
        )

        # Middle layers keep same hidden size
        for _ in range(n_layers - 2):
            self.convs.append(
                GATConv(
                    hidden * n_heads, hidden, heads=n_heads, concat=True, dropout=0.1
                )
            )

        # Last GAT layer keeps dimension but no head concatenation
        self.convs.append(
            GATConv(
                hidden * n_heads,
                hidden,
                heads=1,  # heads=1 → output dim = hidden
                concat=False,
                dropout=0.1,
            )
        )

        # ---------- MLP head ----------
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, data):
        x, edge = data.x, data.edge_index
        for conv in self.convs:
            x = torch.relu(conv(x, edge))
        return self.head(x)


# ------------------ CONFIG ------------------ #
DATA = {
    "daily": "./data/features/earthquake_features.parquet",
    "seismic": "./data/embeddings/Embeddings_192142.pkl",
    "meta": "./data/raw/earthquake_data.parquet",
}
TOP_FEATS = [
    "time_since_class_3",
    "rolling_T_value",
    "daily_count_30d_sum",
    "daily_b_value",
    "rolling_dE_half",
    "daily_etas_intensity",
    "time_since_class_2",
    "daily_count_7d_sum",
]
CUT = "2024-01-01"
BATCH, EPOCHS, LR, HID, RADIUS = 16, 60, 1e-3, 16, 100.0
RUN_DIR = Path("results") / f"gnn_{datetime.now():%Y-%m-%d_%H-%M-%S}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 1. shared preprocessing -------------------- #
_, _, _, _, df_full = load_and_split_data(
    DATA["daily"],
    DATA["seismic"],
    select_top_feats=True,
    top_features=TOP_FEATS,
    merge_embeddings=True,
    keep_embeddings=False,
    cutoff_date=CUT,
)

print(df_full.head())
print(df_full.station_code.unique())
FEATS = TOP_FEATS + [c for c in df_full.columns if c.startswith("emb_")]

# ---------- 2. graph construction ---------------------- #
meta = gh.load_station_metadata(DATA["meta"])
edge_index = gh.build_radius_graph(
    meta.station_latitude.values, meta.station_longitude.values, RADIUS
)
dl_tr, dl_te, C, *_ = gh.make_dataloaders(
    df_full, meta, FEATS, edge_index, batch_size=BATCH, cutoff=CUT
)

# ---------- 3. model & optimiser ----------------------- #
model = StationGNN(len(FEATS) * 2, hidden=HID, n_layers=3, n_classes=C).to(DEVICE)
# model = StationGAT(
#     len(FEATS) * 2, hidden=HID, n_layers=3, n_heads=4, n_classes=C  # tweakable
# ).to(DEVICE)

optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()


# ---------- 4. training loop --------------------------- #
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode="min", patience=5, factor=0.5, verbose=True
)

best_loss = float("inf")  # lowest validation CE seen so far
wait, patience = 0, 12  # stop if no improvement for `patience` epochs
best_state = None  # keep the weights that achieved best_loss


def run_epoch(loader, train: bool):
    model.train() if train else model.eval()
    tot, nodes = 0.0, 0
    for data in loader:
        data = data.to(DEVICE)
        if train:
            optim.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        if train:
            loss.backward()
            optim.step()
        tot += loss.item() * data.num_nodes
        nodes += data.num_nodes
    return tot / nodes  # average cross-entropy per node


for ep in range(1, EPOCHS + 1):
    tr_ce = run_epoch(dl_tr, train=True)
    val_ce = run_epoch(dl_te, train=False)

    scheduler.step(val_ce)  # ↓ LR if plateau
    improved = val_ce < best_loss - 1e-6  # tiny tolerance
    if improved:
        best_loss, wait = val_ce, 0
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    else:
        wait += 1

    print(
        f"Epoch {ep:3} │ Train CE {tr_ce:.4f} │ Val CE {val_ce:.4f}"
        f" │ LR {optim.param_groups[0]['lr']:.2e}" + ("  *best*" if improved else "")
    )

    if wait >= patience:
        print(f"⏹️  Early stopping triggered after {ep} epochs.")
        break

# ---------- 5. reload best weights ---------------------------------- #
model.load_state_dict(best_state)
print(f"🗄️  Restored weights with best Val CE = {best_loss:.4f}")


# ---------- 5. evaluation ------------------------------ #
def logits_and_labels(loader):
    model.eval()
    outs, ys = [], []
    with torch.no_grad():
        for d in loader:
            d = d.to(DEVICE)
            outs.append(model(d).cpu().numpy())
            ys.append(d.y.cpu().numpy())
    return np.concatenate(outs), np.concatenate(ys)


log_tr, y_tr = logits_and_labels(dl_tr)
log_te, y_te = logits_and_labels(dl_te)

evaluate(log_tr.argmax(1), log_tr, pd.DataFrame(), y_tr, "Train", RUN_DIR)
evaluate(log_te.argmax(1), log_te, pd.DataFrame(), y_te, "Test", RUN_DIR)
print(f"\n✅ Artifacts saved to: {RUN_DIR}")
