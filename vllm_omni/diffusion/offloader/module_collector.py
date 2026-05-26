# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PipelineModules:
    dits: list[nn.Module]
    dit_names: list[str]
    encoders: list[nn.Module]
    encoder_names: list[str]
    vae: nn.Module | None = None
    resident_modules: list[nn.Module] | None = None
    resident_names: list[str] | None = None


class ModuleDiscovery:
    """Discovers pipeline components for offloading"""

    DIT_ATTRS = ["transformer", "transformer_2", "dit", "language_model", "transformer_blocks"]
    ENCODER_ATTRS = ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]
    VAE_ATTRS = ["vae"]

    @staticmethod
    def _get_named_modules(pipeline: nn.Module, attrs: list[str]) -> tuple[list[nn.Module], list[str]]:
        modules: list[nn.Module] = []
        names: list[str] = []

        for attr in attrs:
            module_obj = getattr(pipeline, attr, None)
            if module_obj is None:
                continue

            if not isinstance(module_obj, nn.Module):
                logger.warning("Expected %s to be nn.Module, got %r", attr, type(module_obj))
                continue

            if module_obj in modules:
                continue

            modules.append(module_obj)
            names.append(attr)

        return modules, names

    @staticmethod
    def discover(pipeline: nn.Module) -> PipelineModules:
        """Discover DiT, encoder, and VAE modules from pipeline.

        Args:
            pipeline: Diffusion pipeline model

        Returns:
            PipelineModules with lists of discovered modules and names
        """
        # Collect DiT/transformer modules
        dit_attrs = getattr(pipeline, "_dit_modules", ModuleDiscovery.DIT_ATTRS)
        dit_modules, dit_names = ModuleDiscovery._get_named_modules(pipeline, dit_attrs)

        # Collect all encoders
        encoder_attrs = getattr(pipeline, "_encoder_modules", ModuleDiscovery.ENCODER_ATTRS)
        encoders, encoder_names = ModuleDiscovery._get_named_modules(pipeline, encoder_attrs)

        # Collect VAE
        vae = None
        vae_attrs = getattr(pipeline, "_vae_modules", ModuleDiscovery.VAE_ATTRS)
        for attr in vae_attrs:
            module = getattr(pipeline, attr, None)
            if module is not None:
                if not isinstance(module, nn.Module):
                    logger.warning("Expected %s to be nn.Module, got %r", attr, type(module))
                    continue
                vae = module
                break

        resident_attrs = getattr(pipeline, "_resident_modules", [])
        resident_modules, resident_names = ModuleDiscovery._get_named_modules(pipeline, resident_attrs)

        return PipelineModules(
            dits=dit_modules,
            dit_names=dit_names,
            encoders=encoders,
            encoder_names=encoder_names,
            vae=vae,
            resident_modules=resident_modules,
            resident_names=resident_names,
        )
