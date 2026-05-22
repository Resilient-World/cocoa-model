"""Model architectures and inference wrappers."""

from models.conformal.conformal import (
    ConformalInterval,
    MondrianConformalYield,
    SplitConformalYield,
    load_conformal,
    load_conformal_if_exists,
    save_conformal,
)
from models.features.pape import PhenologyAwarePositionalEncoding
from models.features.teleconnection_gnn import TeleconnectionGNN
from models.process.casej_surrogate import CASEJPhysicsLoss, CASEJSurrogate, load_casej_surrogate
from models.surrogate.joint_exposure_yield import (
    JointHead,
    JointMultiTaskLoss,
    JointOutputs,
    load_joint_head,
)
from models.surrogate.yield_surrogate import (
    MCDropout,
    PhysicsInformedYieldLoss,
    YieldPrediction,
    YieldSurrogateModel,
    predict_with_uncertainty,
)
from models.surrogate.yield_surrogate_v2 import YieldSurrogateV2
from models.surrogate.yield_surrogate_v2_teleconnection import YieldSurrogateV2Teleconnection

__all__ = [
    "CASEJPhysicsLoss",
    "CASEJSurrogate",
    "ConformalInterval",
    "JointHead",
    "JointMultiTaskLoss",
    "JointOutputs",
    "MCDropout",
    "MondrianConformalYield",
    "PhenologyAwarePositionalEncoding",
    "PhysicsInformedYieldLoss",
    "SplitConformalYield",
    "TeleconnectionGNN",
    "YieldPrediction",
    "YieldSurrogateModel",
    "YieldSurrogateV2",
    "YieldSurrogateV2Teleconnection",
    "load_casej_surrogate",
    "load_conformal",
    "load_conformal_if_exists",
    "load_joint_head",
    "predict_with_uncertainty",
    "save_conformal",
]
