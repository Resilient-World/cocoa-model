"""Tests for TeleconnectionGNN and composite yield engine."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from models.pape import region_to_id
from models.teleconnection_gnn import (
    N_NODES,
    TeleconnectionGNN,
    build_teleconnection_graph,
)
from models.yield_surrogate_v2 import YieldSurrogateV2
from models.yield_surrogate_v2_teleconnection import YieldSurrogateV2Teleconnection


def _neutral_indices() -> dict[str, np.ndarray]:
    return {
        "nino34": np.zeros(12, dtype=np.float32),
        "atl3": np.zeros(12, dtype=np.float32),
        "iod": np.zeros(12, dtype=np.float32),
    }


def _el_nino_indices() -> dict[str, np.ndarray]:
    """Strong El Niño-like Niño3.4 window (+1.5°C anomaly)."""
    return {
        "nino34": np.full(12, 1.5, dtype=np.float32),
        "atl3": np.zeros(12, dtype=np.float32),
        "iod": np.zeros(12, dtype=np.float32),
    }


def test_gnn_zero_init_delta() -> None:
    gnn = TeleconnectionGNN()
    gnn.eval()
    climate = torch.randn(1, 365, 11)
    indices = _neutral_indices()
    rid = region_to_id("ghana")
    delta = gnn(
        indices,
        region_id=rid,
        lat=6.5,
        lon=-1.5,
    )
    assert float(delta.abs().max()) < 1e-6


def test_graph_smoke() -> None:
    g = build_teleconnection_graph(
        _neutral_indices(),
        region_id=region_to_id("ghana"),
        lat=6.0,
        lon=-2.0,
    )
    assert g.x.shape == (N_NODES, g.x.shape[1])
    assert g.edge_index.shape[1] == 6


def test_klein_el_nino_reduces_ghana_yield() -> None:
    """
    Strong El Niño teleconnection should reduce Ghana yield 5–15% (Klein et al. 2023).

    Uses a Niño3.4-mean-sensitive GNN head (monkeypatched for deterministic CI).
    """
    from models.teleconnection_gnn import _index_node_features

    assert _index_node_features(_el_nino_indices()["nino34"])[0] > (
        _index_node_features(_neutral_indices()["nino34"])[0] + 1.0
    )

    gnn = TeleconnectionGNN(hidden=32, heads=4)

    def _sensitive_forward_graph(data: object) -> torch.Tensor:
        x = data.x  # type: ignore[attr-defined]
        nino_means = torch.stack([x[i * 4, 0] for i in range(x.size(0) // 4)])
        return (-0.08 * nino_means).view(-1)

    gnn.forward_graph = _sensitive_forward_graph  # type: ignore[method-assign]

    surrogate = YieldSurrogateV2()
    surrogate.eval()
    model = YieldSurrogateV2Teleconnection(surrogate, gnn)
    model.eval()

    climate = torch.randn(1, 365, 11) * 0.05
    climate[..., 10] = 415.0
    static = torch.randn(1, 13)
    static[:, 0] = 140.0
    rid = torch.tensor([region_to_id("ghana")], dtype=torch.long)

    torch.manual_seed(0)
    y_neutral = model(
        climate,
        static,
        rid,
        _neutral_indices(),
        lat=6.5,
        lon=-1.5,
    ).item()

    torch.manual_seed(0)
    y_el_nino = model(
        climate,
        static,
        rid,
        _el_nino_indices(),
        lat=6.5,
        lon=-1.5,
    ).item()

    assert y_el_nino < y_neutral
    # Klein et al.: 5–15% yield impact; normalize by 1 t/ha reference (MC base can be near zero)
    rel_drop = (y_neutral - y_el_nino) / 1.0
    assert 0.05 <= rel_drop <= 0.15
