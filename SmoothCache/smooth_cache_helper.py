# SmoothCache/smooth_cache_helper.py

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

# Import BasicTransformerBlock from the Diffusers library
from diffusers.models.attention import BasicTransformerBlock

class SmoothCacheHelper:
    def __init__(self, model):
        self.model = model  # DiTTransformer2DModel instance
        self.original_forwards = {}
        self.cache = {}
        self.params = {
            'cache_interval': 1,  # Default caching interval
            'skip_mode': 'uniform',  # Caching policy
        }
        # Use per-module step counters
        self.current_steps = {}
        self.start_steps = {}

    def enable(self):
        self.reset_state()
        self.wrap_attn1_modules()

    def disable(self):
        self.unwrap_attn1_modules()
        self.reset_state()

    def set_params(self, cache_interval=1, skip_mode='uniform'):
        self.params['cache_interval'] = cache_interval
        self.params['skip_mode'] = skip_mode

    def reset_state(self):
        self.current_steps = {}
        self.start_steps = {}
        self.cache.clear()

    def is_skip_step(self, full_name):
        if self.start_steps[full_name] is None:
            self.start_steps[full_name] = self.current_steps[full_name]
        # return (self.current_steps[full_name] - self.start_steps[full_name]) % self.params['cache_interval'] != 0
        return False  # Extend with other skip modes if needed

    def wrap_attn1_modules(self):
        # Find and wrap all attn1 modules in BasicTransformerBlocks
        for name, module in self.model.named_modules():
            if isinstance(module, BasicTransformerBlock):
                if hasattr(module, 'attn1'):
                    attn1 = module.attn1
                    full_name = f"{name}.attn1"
                    # Store original forward method
                    self.original_forwards[full_name] = attn1.forward
                    # Create wrapped forward method
                    wrapped_forward = self.create_wrapped_forward(full_name, attn1.forward)
                    # Replace the attn1's forward method
                    attn1.forward = wrapped_forward

    def unwrap_attn1_modules(self):
        # Restore original forward methods
        for full_name, original_forward in self.original_forwards.items():
            module = self.get_module_by_name(self.model, full_name)
            if module is not None:
                module.forward = original_forward

    def create_wrapped_forward(self, full_name, original_forward):
        def wrapped_forward(*args, **kwargs):
            # Initialize step counters for this module if not already done
            if full_name not in self.current_steps:
                self.current_steps[full_name] = 0
                self.start_steps[full_name] = None

            # Increment current_step for this module
            self.current_steps[full_name] += 1

            if self.is_skip_step(full_name) and full_name in self.cache:
                # Use cached output during skipped steps
                print("returning cache result for ",  full_name, " at step ", self.current_steps)
                return self.cache[full_name]
            else:
                # Compute output and cache it
                output = original_forward(*args, **kwargs)
                self.cache[full_name] = output
                print("returning normal result for ",  full_name, " at step ", self.current_steps)
                return output
        return wrapped_forward

    def get_module_by_name(self, model, full_name):
        # Utility function to retrieve a module by its full name
        names = full_name.split('.')
        module = model
        for name in names:
            if hasattr(module, name):
                module = getattr(module, name)
            else:
                return None
        return module
