# SmoothCache/smooth_cache_helper.py

from typing import Dict, Any, Optional, List, Union, Type
import torch
import torch.nn as nn

class SmoothCacheHelper:
    def __init__(
        self,
        model: nn.Module,
        block_classes: Union[Type[nn.Module], List[Type[nn.Module]]],
        components_to_wrap: List[str],
    ):
        """
        Generalized SmoothCacheHelper to wrap specified components in specified block classes.

        Args:
            model (nn.Module): The model to wrap.
            block_classes (Type[nn.Module] or List[Type[nn.Module]]): The block class(es) to search for.
            components_to_wrap (List[str]): The names of the components within the blocks to wrap.
        """
        self.model = model
        self.block_classes = block_classes if isinstance(block_classes, list) else [block_classes]
        self.components_to_wrap = components_to_wrap

        self.original_forwards = {}
        self.cache = {}
        # Use per-module step counters
        self.current_steps = {}
        self.start_steps = {}

    def enable(self):
        self.reset_state()
        self.wrap_components()

    def disable(self):
        self.unwrap_components()
        self.reset_state()

    def reset_state(self):
        self.current_steps = {}
        self.start_steps = {}
        self.cache.clear()

    def is_skip_step(self, full_name):
        if full_name not in self.start_steps or self.start_steps[full_name] is None:
            self.start_steps[full_name] = self.current_steps[full_name]
        # return False if self.current_steps[full_name] % 2 else True  # Extend with other skip modes if needed
        return False  # Extend with other skip modes if needed

    def wrap_components(self):
        # Wrap specified components within each block class
        for block_name, block in self.model.named_modules():
            if any(isinstance(block, cls) for cls in self.block_classes):
                self.wrap_block_components(block, block_name)

    def wrap_block_components(self, block, block_name):
        #TODO: verify block exists
        for comp_name in self.components_to_wrap:
            if hasattr(block, comp_name):
                component = getattr(block, comp_name)
                full_name = f"{block_name}.{comp_name}"
                # Store original forward method
                self.original_forwards[full_name] = component.forward
                # Create wrapped forward method
                wrapped_forward = self.create_wrapped_forward(full_name, component.forward)
                # Replace the component's forward method
                component.forward = wrapped_forward

    def unwrap_components(self):
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
                print("returning cache result for ",  full_name, " at step ", self.current_steps[full_name])
                return self.cache[full_name]
            else:
                # Compute output and cache it
                output = original_forward(*args, **kwargs)
                self.cache[full_name] = output
                print("returning normal result for ",  full_name, " at step ", self.current_steps[full_name])
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
