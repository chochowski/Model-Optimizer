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

"""Megatron Bridge for Puzzletron GPT-OSS-based AnyModel heterogeneous checkpoints."""

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import GptOssForCausalLM

from modelopt.torch.puzzletron.export.mbridge.base import HeterogeneousBridgeMixin


@MegatronModelBridge.register_bridge(source=GptOssForCausalLM, target=GPTModel)
class PuzzletronGptOssAnyModelBridge(GPTOSSBridge, HeterogeneousBridgeMixin):
    """
    Megatron Bridge for Puzzletron GPT-OSS-based AnyModel checkpoints.

    Extends GPTOSSBridge with support for heterogeneous layer architectures (block_configs).
    All GPT-OSS-specific settings are inherited from GPTOSSBridge.

    Note on "No mapping found for megatron_param" warnings:
        This bridge uses GPTOSSBridge's mapping_registry(), which defines parameter
        name mappings for homogeneous GPT-OSS (same layer layout every layer). The
        Megatron model built from GenericHeterogeneousProvider is heterogeneous: layer
        layout and param paths can vary per block (e.g. decoder.layers.*.mixer.* for
        Mamba, different layernorm paths). Any Megatron parameter whose name does not
        match the homogeneous patterns will trigger the warning. To fix, override
        mapping_registry() to add MegatronParamMapping entries for the heterogeneous
        param names (e.g. mixer.*, per-block attention/ffn paths) so they map to the
        corresponding HF AnyModel state dict keys.
    """

    def provider_bridge(self, hf_pretrained):
        """Call GPTOSSBridge first, then wrap with heterogeneous config."""
        provider = super().provider_bridge(hf_pretrained)
        return self.wrap_provider_with_heterogeneous(provider, hf_pretrained)
