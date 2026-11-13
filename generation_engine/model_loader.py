from dataclasses import dataclass
from typing import Any


@dataclass
class ShapEModels:
    torch: Any
    device: Any
    transmitter: Any
    text_model: Any
    diffusion: Any


def load_shap_e_models() -> ShapEModels:
    """Load Shap-E models only when generation is requested."""
    try:
        import torch
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
        from shap_e.models.download import load_config, load_model
    except ImportError as exc:
        raise RuntimeError(
            "Shap-E dependencies are not installed. "
            "Install the model dependencies before generating assets."
        ) from exc

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    transmitter = load_model("transmitter", device=device)
    text_model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    return ShapEModels(
        torch=torch,
        device=device,
        transmitter=transmitter,
        text_model=text_model,
        diffusion=diffusion,
    )
