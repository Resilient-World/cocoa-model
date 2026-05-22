"""
Aurora encoder latent adapter for YieldSurrogateV2 / CSSVD landscape side-inputs.

LoRA fine-tune targets attention projections; per-region adapters saved as
``models/aurora_lora_<region>.safetensors``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
import torch
from torch import Tensor, nn

log = structlog.get_logger(__name__)

DEFAULT_LORA_TARGETS = ("self_attention.qkv", "self_attention.proj")
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32


def apply_lora(
    model: nn.Module,
    *,
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS,
    r: int = DEFAULT_LORA_R,
    alpha: int = DEFAULT_LORA_ALPHA,
) -> Any:
    """Attach PEFT LoRA adapters to Aurora attention modules."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(model, config)


class AuroraBackboneAdapter(nn.Module):
    """
    Extract 3D Perceiver encoder latent ``[B, H/P, W/P, L]`` from Aurora for downstream heads.

    The wrapped Aurora model is held by reference; ``forward`` accepts an ``aurora.Batch``.
    """

    def __init__(self, aurora_model: nn.Module) -> None:
        super().__init__()
        self.aurora = aurora_model
        self._peft_wrapped = False

    def _encoder_module(self) -> nn.Module:
        enc = getattr(self.aurora, "encoder", None)
        if enc is None and hasattr(self.aurora, "base_model"):
            enc = getattr(self.aurora.base_model, "encoder", None)
        if enc is None:
            raise AttributeError("Aurora model has no .encoder; cannot extract latent")
        return enc  # type: ignore[no-any-return]

    def forward(self, batch: Any) -> Tensor:
        """Return encoder latent; shape ``[B, H/P, W/P, L]`` when Perceiver is used."""
        enc = self._encoder_module()
        if hasattr(enc, "forward"):
            out = enc(batch)
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(out, Tensor):
                return out
        raise RuntimeError("Could not extract Aurora encoder latent from forward pass")

    def enable_lora(
        self,
        *,
        target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS,
        r: int = DEFAULT_LORA_R,
        alpha: int = DEFAULT_LORA_ALPHA,
    ) -> None:
        """Wrap Aurora with PEFT LoRA on attention layers."""
        if self._peft_wrapped:
            return
        try:
            self.aurora = apply_lora(
                self.aurora,
                target_modules=target_modules,
                r=r,
                alpha=alpha,
            )
            self._peft_wrapped = True
        except Exception as exc:
            log.warning("aurora_lora_attach_failed", error=str(exc))
            if hasattr(self.aurora, "use_lora"):
                object.__setattr__(self.aurora, "use_lora", True)

    def save_region_adapter(self, region: str, path: Path | str) -> None:
        """Save LoRA weights for a cocoa region."""
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {}
        if hasattr(self.aurora, "save_pretrained"):
            tmp = path.with_suffix(".peft_tmp")
            self.aurora.save_pretrained(tmp)
            for f in tmp.glob("adapter_model*.safetensors"):
                from safetensors.torch import load_file

                state.update(load_file(f))
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
        else:
            for name, param in self.aurora.named_parameters():
                if "lora" in name.lower():
                    state[name] = param.detach().cpu()
        if not state:
            raise ValueError(f"No LoRA weights to save for region {region}")
        save_file(state, path)

    def load_region_adapter(self, region: str, path: Path | str) -> None:
        """Load per-region LoRA adapter from ``models/aurora_lora_<region>.safetensors``."""
        from safetensors.torch import load_file

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        state = load_file(path)
        if hasattr(self.aurora, "load_adapter"):
            self.aurora.load_adapter(str(path.parent), adapter_name=region)
            return
        missing = []
        for name, tensor in state.items():
            try:
                param = self.aurora.get_parameter(name.replace("base_model.", ""))
                param.data.copy_(tensor.to(param.device, dtype=param.dtype))
            except Exception:
                missing.append(name)
        if len(missing) == len(state):
            log.warning("aurora_lora_partial_load", region=region, n_missing=len(missing))
