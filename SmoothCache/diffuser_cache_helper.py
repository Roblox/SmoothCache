# Copyright 2022 Roblox Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper Class for Diffusion Transformer Implemented at 
   https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/dit"""

from .smooth_cache_helper import SmoothCacheHelper

try:
    from diffusers.models.attention import BasicTransformerBlock
except ImportError:
    print("Warning: Diffusers library is not installed. DiffuserCacheHelper cannot be used.")
    BasicTransformerBlock = None

class DiffuserCacheHelper(SmoothCacheHelper):
    def __init__(self, model, schedule):
        if BasicTransformerBlock is None:
            raise ImportError("Diffusers library is not installed. DiffuserCacheHelper cannot be used.")
        block_classes = BasicTransformerBlock
        components_to_wrap = ['attn1']
        super().__init__(
            model=model,
            block_classes=block_classes,
            components_to_wrap=components_to_wrap,
            schedule=schedule
        )
