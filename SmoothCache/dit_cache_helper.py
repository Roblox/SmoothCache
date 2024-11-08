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
   https://github.com/facebookresearch/DiT"""

from .smooth_cache_helper import SmoothCacheHelper

try:
    # Assuming DiTBlock is defined in 'models/dit.py' in the DiT repository
    from models import DiTBlock
except ImportError:
    print("Warning: DiT library is not accessible. DiTCacheHelper cannot be used.")
    DiTBlock = None

class DiTCacheHelper(SmoothCacheHelper):
    def __init__(self, model, schedule):
        if DiTBlock is None:
            raise ImportError("DiT library is not accessible. DiTCacheHelper cannot be used.")
        block_classes = DiTBlock
        components_to_wrap = ['attn', 'mlp']
        super().__init__(
            model=model,
            block_classes=block_classes,
            components_to_wrap=components_to_wrap,
            schedule=schedule
        )
