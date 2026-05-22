"""
Bipartite GAT teleconnection model: Niño 3.4, Atl3, IOD → farm yield bias (t/ha).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv

from models.features.pape import REGION_KEYS

INDEX_NODE_NAMES = ("nino34", "atl3", "iod")
N_INDEX_NODES = len(INDEX_NODE_NAMES)
N_NODES = N_INDEX_NODES + 1
FARM_NODE_IDX = 3

INDEX_FEATURE_DIM = 5
FARM_FEATURE_DIM = 10


def _index_node_features(monthly: np.ndarray) -> np.ndarray:
    """
    Build 5-d features from a monthly index series (length >= 13).

    Rolling 12-month mean/std and lags 3/6/12 at the series end.
    """
    s = np.asarray(monthly, dtype=np.float64).reshape(-1)
    if s.size < 13:
        pad = np.full(13 - s.size, float(s[0]) if s.size else 0.0)
        s = np.concatenate([pad, s])
    window = s[-12:]
    mean_12 = float(window.mean())
    std_12 = float(window.std()) if window.size > 1 else 0.0
    end = s[-1]
    lag3 = float(s[-4]) if s.size >= 4 else end
    lag6 = float(s[-7]) if s.size >= 7 else end
    lag12 = float(s[-13]) if s.size >= 13 else end
    return np.array([mean_12, std_12, lag3, lag6, lag12], dtype=np.float32)


def _farm_node_features(region_id: int, lat: float, lon: float) -> np.ndarray:
    one_hot = np.zeros(len(REGION_KEYS), dtype=np.float32)
    rid = int(region_id) % len(REGION_KEYS)
    one_hot[rid] = 1.0
    lat_norm = float(np.clip((lat + 15.0) / 30.0, 0.0, 1.0) * 2.0 - 1.0)
    lon_norm = float(np.clip((lon + 90.0) / 180.0, 0.0, 1.0) * 2.0 - 1.0)
    return np.concatenate([one_hot, [lat_norm, lon_norm]]).astype(np.float32)


def _star_graph_edge_index() -> Tensor:
    """Undirected edges between each index node and farm node 3."""
    src: list[int] = []
    dst: list[int] = []
    for i in range(N_INDEX_NODES):
        src.extend([i, FARM_NODE_IDX])
        dst.extend([FARM_NODE_IDX, i])
    return torch.tensor([src, dst], dtype=torch.long)


@dataclass(frozen=True)
class TeleconnectionFeatures:
    """Per-farm teleconnection inputs for the GNN."""

    nino34: np.ndarray
    atl3: np.ndarray
    iod: np.ndarray
    region_id: int
    lat: float
    lon: float

    @classmethod
    def from_dict(
        cls,
        indices: dict[str, np.ndarray],
        *,
        region_id: int,
        lat: float,
        lon: float,
    ) -> TeleconnectionFeatures:
        return cls(
            nino34=np.asarray(indices["nino34"], dtype=np.float32),
            atl3=np.asarray(indices["atl3"], dtype=np.float32),
            iod=np.asarray(indices["iod"], dtype=np.float32),
            region_id=int(region_id),
            lat=float(lat),
            lon=float(lon),
        )


def build_teleconnection_graph(
    features: TeleconnectionFeatures | dict[str, np.ndarray],
    *,
    region_id: int | None = None,
    lat: float | None = None,
    lon: float | None = None,
) -> Data:
    """Build a single ``torch_geometric.data.Data`` graph (4 nodes)."""
    if isinstance(features, TeleconnectionFeatures):
        tf = features
    else:
        tf = TeleconnectionFeatures.from_dict(
            features,
            region_id=int(region_id if region_id is not None else 0),
            lat=float(lat if lat is not None else 6.0),
            lon=float(lon if lon is not None else -2.0),
        )

    def _pad_index(feat: np.ndarray) -> np.ndarray:
        out = np.zeros(FARM_FEATURE_DIM, dtype=np.float32)
        out[:INDEX_FEATURE_DIM] = feat
        return out

    index_feats = [
        _pad_index(_index_node_features(tf.nino34)),
        _pad_index(_index_node_features(tf.atl3)),
        _pad_index(_index_node_features(tf.iod)),
    ]
    farm_feat = _farm_node_features(tf.region_id, tf.lat, tf.lon)
    x = torch.tensor(np.stack(index_feats + [farm_feat], axis=0), dtype=torch.float32)
    return Data(x=x, edge_index=_star_graph_edge_index())


class TeleconnectionGNN(nn.Module):
    """
    4-node bipartite GAT: climate indices → farm → scalar ``delta_y`` (t/ha).

    Final readout layer is zero-initialized for identity at load time.
    """

    def __init__(self, hidden: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.index_stem = nn.Linear(INDEX_FEATURE_DIM, hidden)
        self.farm_stem = nn.Linear(FARM_FEATURE_DIM, hidden)
        self.gat1 = GATv2Conv(hidden, hidden // heads, heads=heads, concat=True)
        self.gat2 = GATv2Conv(hidden, hidden, heads=1, concat=False)
        self.readout = nn.Linear(hidden, 1)
        self._zero_init_readout()

    def _zero_init_readout(self) -> None:
        nn.init.zeros_(self.readout.weight)
        if self.readout.bias is not None:
            nn.init.zeros_(self.readout.bias)

    def forward_graph(self, data: Data | Batch) -> Tensor:
        """Return ``delta_y`` shape ``[batch]`` from PyG batch."""
        x = data.x
        device = x.device
        node_type = torch.arange(x.size(0), device=device) % N_NODES
        index_mask = node_type < N_INDEX_NODES
        farm_mask = node_type == FARM_NODE_IDX

        h = torch.empty(x.size(0), self.hidden, device=device, dtype=x.dtype)
        h[index_mask] = self.index_stem(x[index_mask, :INDEX_FEATURE_DIM])
        h[farm_mask] = self.farm_stem(x[farm_mask])

        h = self.gat1(h, data.edge_index).relu()
        h = self.gat2(h, data.edge_index).relu()
        return self.readout(h[farm_mask]).view(-1)

    def forward(
        self,
        features: TeleconnectionFeatures | dict[str, np.ndarray] | list[TeleconnectionFeatures],
        *,
        region_id: int | None = None,
        lat: float | None = None,
        lon: float | None = None,
    ) -> Tensor:
        """
        Compute teleconnection bias for one or more farms.

        Returns shape ``[B]`` (t/ha).
        """
        if isinstance(features, list):
            graphs = [build_teleconnection_graph(f) for f in features]
        elif isinstance(features, TeleconnectionFeatures):
            graphs = [build_teleconnection_graph(features)]
        else:
            graphs = [
                build_teleconnection_graph(
                    features,
                    region_id=region_id,
                    lat=lat,
                    lon=lon,
                )
            ]
        batch = Batch.from_data_list(graphs).to(next(self.parameters()).device)
        return self.forward_graph(batch)


__all__ = [
    "INDEX_NODE_NAMES",
    "TeleconnectionFeatures",
    "TeleconnectionGNN",
    "build_teleconnection_graph",
]
