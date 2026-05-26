# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from vllm_omni.entrypoints.openai.utils import is_single_stage_diffusion, resolve_diffusion_od_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _StageCfg:
    stage_type: str


class _EngineClient:
    def __init__(self, od_config=None, stage_configs=None):
        self._od_config = od_config
        self.stage_configs = stage_configs or []

    def get_diffusion_od_config(self):
        return self._od_config


class _DiffusionEngine:
    def __init__(self, od_config=None):
        self.od_config = od_config


def test_resolve_diffusion_od_config_prefers_engine_client_value():
    cfg = {"model_class_name": "SenseNovaU1Pipeline"}
    engine_client = _EngineClient(od_config=cfg)
    assert resolve_diffusion_od_config(engine_client, _DiffusionEngine(od_config=None)) == cfg


def test_resolve_diffusion_od_config_falls_back_to_diffusion_engine_attr():
    engine_client = _EngineClient(od_config=None)
    fallback_cfg = {"model_class_name": "SenseNovaU1Pipeline"}
    assert resolve_diffusion_od_config(engine_client, _DiffusionEngine(od_config=fallback_cfg)) == fallback_cfg


def test_is_single_stage_diffusion_detects_single_diffusion_stage():
    engine_client = _EngineClient(stage_configs=[_StageCfg(stage_type="diffusion")])
    assert is_single_stage_diffusion(engine_client)


def test_is_single_stage_diffusion_rejects_non_diffusion_or_multi_stage():
    assert not is_single_stage_diffusion(_EngineClient(stage_configs=[_StageCfg(stage_type="llm")]))
    assert not is_single_stage_diffusion(
        _EngineClient(stage_configs=[_StageCfg(stage_type="diffusion"), _StageCfg(stage_type="llm")])
    )
