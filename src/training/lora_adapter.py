"""HuggingFace PEFT LoRA adapters for per-region cocoa backbone specialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, cast

import structlog
import torch
from torch import nn

log = structlog.get_logger(__name__)

BackboneName: TypeAlias = Literal["galileo", "aef", "agrifm", "terramind", "olmoearth"]


class _SavePretrained(Protocol):
    def save_pretrained(self, path: Path) -> None: ...


class _LoadAdapter(Protocol):
    def load_adapter(self, path: str, *, adapter_name: str) -> Any: ...


AUTO_TARGET_MODULES: dict[BackboneName, tuple[str, ...]] = {
    "galileo": ("qkv", "proj", "query", "key", "value"),
    "aef": ("0", "2"),
    "agrifm": ("qkv", "proj", "fc1", "fc2"),
    "terramind": ("qkv", "proj", "fc1", "fc2"),
    "olmoearth": ("q_proj", "k_proj", "v_proj", "out_proj", "qkv", "proj"),
}


class LoRALinear(nn.Module):
    """Minimal Linear LoRA fallback used when PEFT is unavailable."""

    def __init__(self, base: nn.Linear, *, r: int, alpha: int) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scaling = float(alpha) / float(r)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def _freeze_non_lora(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def _target_modules(
    backbone_name: BackboneName, target_modules: tuple[str, ...] | None
) -> tuple[str, ...]:
    return target_modules if target_modules is not None else AUTO_TARGET_MODULES[backbone_name]


def _replace_child(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if isinstance(parent, nn.Sequential) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _apply_manual_lora(
    model: nn.Module, targets: tuple[str, ...], *, r: int, alpha: int
) -> nn.Module:
    replaced = 0
    for full_name, module in list(model.named_modules()):
        if not full_name or not isinstance(module, nn.Linear):
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf not in targets and not any(target in full_name for target in targets):
            continue
        parent = model
        parts = full_name.split(".")
        for part in parts[:-1]:
            parent = (
                parent[int(part)]
                if isinstance(parent, nn.Sequential) and part.isdigit()
                else getattr(parent, part)
            )
        _replace_child(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha))
        replaced += 1
    if replaced == 0:
        for param in model.parameters():
            param.requires_grad = True
    return model


def apply_lora_to_backbone(
    backbone: nn.Module,
    backbone_name: BackboneName,
    r: int = 16,
    alpha: int = 32,
    target_modules: tuple[str, ...] | None = None,
) -> nn.Module:
    """Freeze ``backbone`` and attach PEFT LoRA adapters to matching modules."""
    targets = _target_modules(backbone_name, target_modules)
    for param in backbone.parameters():
        param.requires_grad = False
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-not-found]
        from peft.peft_model import PeftModel  # type: ignore[import-not-found]

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=list(targets),
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        wrapped = get_peft_model(backbone, config)
        _freeze_non_lora(wrapped)
        return cast(PeftModel, wrapped)
    except Exception as exc:
        log.warning(
            "peft_lora_attach_failed",
            backbone_name=backbone_name,
            target_modules=targets,
            error=str(exc),
        )
        wrapped = _apply_manual_lora(backbone, targets, r=r, alpha=alpha)
        return wrapped


def _adapter_path(path: Path | str, backbone_name: str, region: str) -> Path:
    root = Path(path)
    if root.suffix:
        return root
    return root / f"{backbone_name}_lora_{region}.safetensors"


def _lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_" in name or "lora" in name.lower()
    }


def save_lora_for_region(
    model: nn.Module,
    region: str,
    path: Path | str,
    *,
    backbone_name: str = "backbone",
) -> Path:
    """Save only adapter weights for ``region`` as safetensors."""
    out = _adapter_path(path, backbone_name, region)
    out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        tmp_dir = out.parent / f".{out.stem}_peft"
        saver = cast(_SavePretrained, model)
        saver.save_pretrained(tmp_dir)
        candidates = sorted(tmp_dir.glob("adapter_model*.safetensors"))
        if candidates:
            import shutil

            shutil.copy2(candidates[0], out)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return out
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)
    state = _lora_state_dict(model)
    if not state:
        state = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
    if not state:
        raise ValueError(f"No trainable LoRA parameters found for region {region!r}")
    from safetensors.torch import save_file

    save_file(state, out)
    return out


def load_lora_for_region(
    backbone: nn.Module,
    region: str,
    path: Path | str,
    *,
    backbone_name: BackboneName = "galileo",
) -> nn.Module:
    """Load a per-region LoRA adapter into ``backbone``."""
    adapter_file = _adapter_path(path, backbone_name, region)
    if not adapter_file.is_file():
        raise FileNotFoundError(adapter_file)
    if hasattr(backbone, "load_adapter"):
        loader = cast(_LoadAdapter, backbone)
        loader.load_adapter(str(adapter_file.parent), adapter_name=region)
        return backbone
    wrapped = apply_lora_to_backbone(backbone, backbone_name)
    from safetensors.torch import load_file

    state = load_file(adapter_file)
    missing, unexpected = wrapped.load_state_dict(state, strict=False)
    log.info(
        "loaded_lora_adapter",
        region=region,
        path=str(adapter_file),
        missing=len(missing),
        unexpected=len(unexpected),
    )
    return wrapped


def trainable_parameter_fraction(model: nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return float(trainable / total) if total else 0.0


__all__ = [
    "AUTO_TARGET_MODULES",
    "BackboneName",
    "apply_lora_to_backbone",
    "load_lora_for_region",
    "save_lora_for_region",
    "trainable_parameter_fraction",
]
