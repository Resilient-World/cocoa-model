"""MLflow PyFunc wrappers for yield and segmentation checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import torch


class YieldSurrogatePyfunc(mlflow.pyfunc.PythonModel):
    """Load :class:`~models.yield_surrogate_v2.YieldSurrogateV2` from artifact path."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from models.yield_surrogate_v2 import YieldSurrogateV2

        ckpt = context.artifacts["checkpoint"]
        self._model = YieldSurrogateV2.from_checkpoint(Path(ckpt))
        self._model.eval()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: dict[str, Any] | None = None,
    ) -> list[float]:
        import numpy as np

        arr = np.asarray(model_input, dtype=np.float32)
        if arr.ndim == 2:
            from models.yield_surrogate import N_STATIC_SITE

            climate = torch.from_numpy(arr).unsqueeze(0)
            static = torch.zeros(1, N_STATIC_SITE)
            region_id = torch.zeros(1, dtype=torch.long)
        else:
            raise ValueError("Expected 2-D climate array [T, C]")
        with torch.no_grad():
            out = self._model(climate, static, region_id)
        return out.cpu().numpy().ravel().tolist()


def log_yield_checkpoint(
    checkpoint_path: Path,
    *,
    artifact_path: str = "checkpoint",
) -> None:
    """Log yield surrogate as PyFunc model artifact on the active MLflow run."""
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=YieldSurrogatePyfunc(),
        artifacts={artifact_path: str(checkpoint_path)},
    )
