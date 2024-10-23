# SmoothCache/dit_cache_helper.py

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
