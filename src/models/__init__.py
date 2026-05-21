"""Model architectures and inference wrappers."""

from models.conformal import (
    ConformalInterval,
    MondrianConformalYield,
    SplitConformalYield,
    load_conformal,
    load_conformal_if_exists,
    save_conformal,
)
from models.casej_surrogate import CASEJPhysicsLoss, CASEJSurrogate, load_casej_surrogate
from models.joint_exposure_yield import (
    JointHead,
    JointMultiTaskLoss,
    JointOutputs,
    load_joint_head,
)
from models.pape import PhenologyAwarePositionalEncoding
from models.yield_surrogate import (
    MCDropout,
    PhysicsInformedYieldLoss,
    YieldPrediction,
    YieldSurrogateModel,
    predict_with_uncertainty,
)
from models.teleconnection_gnn import TeleconnectionGNN
from models.yield_surrogate_v2 import YieldSurrogateV2
from models.yield_surrogate_v2_teleconnection import YieldSurrogateV2Teleconnection

__all__ = [
    "CASEJPhysicsLoss",
    "CASEJSurrogate",
    "load_casej_surrogate",
    "JointHead",
    "JointMultiTaskLoss",
    "JointOutputs",
    "load_joint_head",
    "ConformalInterval",
    "MondrianConformalYield",
    "SplitConformalYield",
    "load_conformal",
    "load_conformal_if_exists",
    "save_conformal",
    "MCDropout",
    "PhysicsInformedYieldLoss",
    "YieldPrediction",
    "PhenologyAwarePositionalEncoding",
    "TeleconnectionGNN",
    "YieldSurrogateModel",
    "YieldSurrogateV2",
    "YieldSurrogateV2Teleconnection",
    "predict_with_uncertainty",
]
