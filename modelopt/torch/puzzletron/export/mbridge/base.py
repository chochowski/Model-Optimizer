#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mixin class for bridges that support heterogeneous layer architectures.

This module provides a mixin class for converting models with block_configs
(heterogeneous layer configurations) to Megatron-Core format via Megatron-Bridge.
"""

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field, fields

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.transformer_config import HeterogeneousTransformerConfig
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.transformer.spec_utils import ModuleSpec


def heterogeneous_layer_spec(config) -> ModuleSpec:
    """Get GPT heterogeneous layer spec using Transformer Engine."""
    return get_gpt_heterogeneous_layer_spec(config, use_te=True)


@dataclass
class GenericHeterogeneousProvider(GPTModelProvider, HeterogeneousTransformerConfig):
    """Generic provider for AnyModel checkpoints with block_configs.

    Carries both base + heterogeneous fields and any model-specific fields from the
    wrapped provider (e.g. GPT-OSS YARN, Mistral scale_factor) in extra_provider_params,
    so getattr(provider, 'yarn_rotary_scaling_factor') etc. work.
    """

    # Heterogeneous configuration fields
    heterogeneous_layers_config_path: str | None = None
    heterogeneous_layers_config_encoded_json: str = ""
    transformer_layer_spec: ModuleSpec | Callable = heterogeneous_layer_spec

    # Model-specific fields not in GPTModelProvider / HeterogeneousTransformerConfig
    # (e.g. yarn_rotary_scaling_factor, scale_factor, moe_*). Preserved from the
    # wrapped provider and exposed via __getattr__.
    extra_provider_params: dict = field(default_factory=dict)

    def __getattr__(self, name: str):
        """Expose extra_provider_params as attributes; handle per_block_parameters."""
        if name == "extra_provider_params":
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        extra = object.__getattribute__(self, "extra_provider_params")
        if name in extra:
            return extra[name]
        if name == "per_block_parameters":
            try:
                return object.__getattribute__(self, name)
            except AttributeError:
                return []
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class HeterogeneousBridgeMixin:
    """Mixin for bridges supporting heterogeneous layer architectures (block_configs).

    Must be used with multiple inheritance alongside a model-specific bridge.
    Use (ModelBridge, HeterogeneousBridgeMixin) and override provider_bridge to call
    the model bridge first, then wrap_provider_with_heterogeneous.
    Example: class PuzzletronLlamaAnyModelBridge(LlamaBridge, HeterogeneousBridgeMixin)
    """

    def wrap_provider_with_heterogeneous(
        self, provider: GPTModelProvider, hf_pretrained: PreTrainedCausalLM
    ) -> GPTModelProvider:
        """Wrap a GPTModelProvider with heterogeneous layer config (block_configs).

        Use this after calling the model-specific bridge's provider_bridge() so that
        provider_bridge is invoked first from the model bridge, then heterogeneous
        wrapping is applied via this method.
        """
        provider_kwargs = dataclasses.asdict(provider)
        valid_fields = {f.name for f in fields(GenericHeterogeneousProvider)}

        # Split into: fields we set on GenericHeterogeneousProvider vs model-specific extra
        known_kwargs = {k: v for k, v in provider_kwargs.items() if k in valid_fields}
        extra_params = {k: v for k, v in provider_kwargs.items() if k not in valid_fields}

        known_kwargs["heterogeneous_layers_config_encoded_json"] = (
            self._build_heterogeneous_config_json(hf_pretrained.config)
        )
        known_kwargs["transformer_layer_spec"] = heterogeneous_layer_spec

        known_kwargs["extra_provider_params"] = extra_params
        return GenericHeterogeneousProvider(**known_kwargs)

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GPTModelProvider:
        """Convert HF AnyModel config to Megatron GPTModelProvider.

        Calls the parent bridge's provider_bridge() first, then wraps with heterogeneous
        config via wrap_provider_with_heterogeneous(). Subclasses that list the model
        bridge first in the base order should override this to call super().provider_bridge()
        then wrap_provider_with_heterogeneous() so the model bridge is invoked first.
        """
        parent_provider = super().provider_bridge(hf_pretrained)  # type: ignore[misc]
        return self.wrap_provider_with_heterogeneous(parent_provider, hf_pretrained)

    @classmethod
    def megatron_to_hf_config(cls, provider: GPTModelProvider) -> dict:
        raise NotImplementedError(
            "megatron_to_hf_config() not yet implemented for AnyModel bridges. "
            "AnyModel bridges require special handling for heterogeneous layer configurations."
        )

    def _build_heterogeneous_config_json(self, hf_config) -> str:
        """Build heterogeneous layers config JSON from HF config."""

        hf_config_dict = json.loads(hf_config.to_json_string())

        mcore_block_configs = [
            self._convert_block_config(block) for block in hf_config_dict["block_configs"]
        ]
        return json.dumps({"block_configs": mcore_block_configs}, ensure_ascii=False)

    def _convert_block_config(self, block: dict) -> dict:
        """Convert a single block config from HF format to MCore format."""
        return {
            "attention": self._convert_attention_config(block["attention"]),
            "ffn": self._convert_ffn_config(block["ffn"]),
        }

    def _convert_attention_config(self, attention_config: dict) -> dict:
        """Convert attention config from HF format to MCore format."""
        attention_config = attention_config.copy()
        attention_config["num_query_groups"] = attention_config.pop("num_key_value_heads")
        return attention_config

    def _convert_ffn_config(self, ffn_config: dict) -> dict:
        """Convert FFN/MLP config from HF format to MCore format."""
        ffn_config = ffn_config.copy()
        ffn_config["ffn_hidden_size"] = ffn_config.pop("intermediate_size")
        return ffn_config
