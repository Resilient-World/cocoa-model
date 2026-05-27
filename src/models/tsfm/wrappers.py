"""
Hugging Face wrappers for time-series foundation models (TSFMs).

Each wrapper exposes a uniform ``.forecast(history, horizon, num_samples)``
method returning :class:`TsfmForecast` quantile bands (p10, p50, p90).

Models:
- ``amazon/chronos-2`` (Apache-2.0)
- ``google/timesfm-2.5-200m-pytorch`` (Apache-2.0)
- ``Maple728/TimeMoE-50M`` (Apache-2.0) — production default
- ``Salesforce/moirai-2.0-R-small`` (Apache-2.0)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
import torch

log = structlog.get_logger(__name__)


@dataclass
class TsfmForecast:
    """Quantile-band forecast from a TSFM wrapper."""

    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    samples: np.ndarray | None = None

    def __post_init__(self) -> None:
        for arr in (self.p10, self.p50, self.p90):
            if arr.ndim == 0:
                raise ValueError("Quantile arrays must be at least 1-D")


class TsfmWrapper(ABC):
    """Abstract base for TSFM wrappers with lazy model loading."""

    model_id: str
    _model: Any = None
    _device: str = "cpu"

    def __init__(self, device: str | None = None) -> None:
        if device is not None:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @abstractmethod
    def _load_model(self) -> Any:
        """Download and instantiate the HF model (called once, lazily)."""

    @abstractmethod
    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        num_samples: int = 100,
    ) -> TsfmForecast:
        """
        Produce quantile-band forecast.

        Parameters
        ----------
        history:
            ``[time_steps, features]`` array. Column 0 is the target (yield);
            remaining columns are climate covariates.
        horizon:
            Number of future time steps to predict.
        num_samples:
            Number of stochastic draws for quantile estimation.
        """

    def _extract_target(self, history: np.ndarray) -> np.ndarray:
        """Return 1-D target series (column 0)."""
        if history.ndim == 1:
            return history.astype(np.float32)
        return history[:, 0].astype(np.float32)

    def _compute_quantiles(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute p10, p50, p90 from sample draws ``[num_samples, horizon]``."""
        p10 = np.percentile(samples, 10, axis=0)
        p50 = np.percentile(samples, 50, axis=0)
        p90 = np.percentile(samples, 90, axis=0)
        return p10, p50, p90


class Chronos2Wrapper(TsfmWrapper):
    """Wrapper for ``amazon/chronos-2`` (Apache-2.0)."""

    model_id = "amazon/chronos-2"

    def _load_model(self) -> Any:
        try:
            import chronos
        except ImportError:
            raise ImportError(
                "chronos package required for Chronos2Wrapper. "
                "Install with: pip install chronos"
            )
        log.info("Loading Chronos-2 from HF", model_id=self.model_id, device=self._device)
        return chronos.Chronos2Pipeline.from_pretrained(
            self.model_id,
            device_map=self._device,
            torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
        )

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        num_samples: int = 100,
    ) -> TsfmForecast:
        target = self._extract_target(history)
        context = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        samples_tensor = self.model.predict(
            context,
            prediction_length=horizon,
            num_samples=num_samples,
            limit_prediction_length=False,
        )
        samples = samples_tensor.detach().cpu().numpy().squeeze(0)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        p10, p50, p90 = self._compute_quantiles(samples)
        return TsfmForecast(p10=p10, p50=p50, p90=p90, samples=samples)


class TimesFM2Wrapper(TsfmWrapper):
    """Wrapper for ``google/timesfm-2.5-200m-pytorch`` (Apache-2.0)."""

    model_id = "google/timesfm-2.5-200m-pytorch"

    def _load_model(self) -> Any:
        try:
            import timesfm
        except ImportError:
            raise ImportError(
                "timesfm package required for TimesFM2Wrapper. "
                "Install with: pip install timesfm"
            )
        log.info("Loading TimesFM-2.5 from HF", model_id=self.model_id, device=self._device)
        return timesfm.TimesFm(
            context_len=512,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend=self._device,
            per_core_batch_size=32,
        )

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        num_samples: int = 100,
    ) -> TsfmForecast:
        target = self._extract_target(history)
        if history.ndim == 2 and history.shape[1] > 1:
            covariates = history[:, 1:].astype(np.float32)
        else:
            covariates = None

        samples_list = []
        for _ in range(num_samples):
            forecast_output = self.model.forecast(
                [target],
                covariates=[covariates] if covariates is not None else None,
            )
            samples_list.append(forecast_output[0][:horizon])
        samples = np.stack(samples_list, axis=0)
        p10, p50, p90 = self._compute_quantiles(samples)
        return TsfmForecast(p10=p10, p50=p50, p90=p90, samples=samples)


class TimeMoEWrapper(TsfmWrapper):
    """Wrapper for ``Maple728/TimeMoE-50M`` (Apache-2.0) — production default."""

    model_id = "Maple728/TimeMoE-50M"

    def _load_model(self) -> Any:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers required for TimeMoEWrapper. Install with: pip install transformers"
            )
        log.info("Loading TimeMoE-50M from HF", model_id=self.model_id, device=self._device)
        return AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
        ).to(self._device)

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        num_samples: int = 100,
    ) -> TsfmForecast:
        target = self._extract_target(history)
        context = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self._device)

        samples_list = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_samples):
                output = self.model.generate(context, max_new_tokens=horizon)
                pred = output[:, -horizon:].detach().cpu().numpy().squeeze(0)
                samples_list.append(pred)
        samples = np.stack(samples_list, axis=0)
        p10, p50, p90 = self._compute_quantiles(samples)
        return TsfmForecast(p10=p10, p50=p50, p90=p90, samples=samples)


class Moirai2Wrapper(TsfmWrapper):
    """Wrapper for ``Salesforce/moirai-2.0-R-small`` (Apache-2.0)."""

    model_id = "Salesforce/moirai-2.0-R-small"

    def _load_model(self) -> Any:
        try:
            from uni2ts.model.moirai import MoiraiForecast
        except ImportError:
            raise ImportError(
                "uni2ts package required for Moirai2Wrapper. "
                "Install with: pip install uni2ts"
            )
        log.info("Loading Moirai-2 from HF", model_id=self.model_id, device=self._device)
        return MoiraiForecast.from_pretrained(self.model_id).to(self._device)

    def forecast(
        self,
        history: np.ndarray,
        horizon: int,
        num_samples: int = 100,
    ) -> TsfmForecast:
        target = self._extract_target(history)
        context = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self._device)

        samples_list = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.model(
                    context,
                    prediction_length=horizon,
                    num_samples=1,
                )
                samples_list.append(pred.detach().cpu().numpy().squeeze(0))
        samples = np.stack(samples_list, axis=0)
        p10, p50, p90 = self._compute_quantiles(samples)
        return TsfmForecast(p10=p10, p50=p50, p90=p90, samples=samples)


def build_wrapper(model_name: str, device: str | None = None) -> TsfmWrapper:
    """Factory for TSFM wrappers by short name.

    Parameters
    ----------
    model_name:
        One of ``chronos-2``, ``timesfm``, ``timemoe``, ``moirai``.
    device:
        Torch device string (``cpu``, ``cuda``, ``mps``). Auto-detected if None.
    """
    registry: dict[str, type[TsfmWrapper]] = {
        "chronos-2": Chronos2Wrapper,
        "timesfm": TimesFM2Wrapper,
        "timemoe": TimeMoEWrapper,
        "moirai": Moirai2Wrapper,
    }
    cls = registry.get(model_name)
    if cls is None:
        raise ValueError(
            f"Unknown TSFM model '{model_name}'. Choose from: {sorted(registry.keys())}"
        )
    return cls(device=device)
