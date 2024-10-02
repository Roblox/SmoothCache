# SmoothCache/diffuser_cache_helper.py

from .smooth_cache_helper import SmoothCacheHelper
from diffusers.models.attention import BasicTransformerBlock

class DiffuserCacheHelper(SmoothCacheHelper):
    def __init__(self, model, cache_interval=1, skip_mode='uniform'):
        block_classes = BasicTransformerBlock
        components_to_wrap = ['attn1',  'ff']
        super().__init__(
            model=model,
            block_classes=block_classes,
            components_to_wrap=components_to_wrap,
        )
