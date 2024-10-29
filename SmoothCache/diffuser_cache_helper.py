# SmoothCache/diffuser_cache_helper.py

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
