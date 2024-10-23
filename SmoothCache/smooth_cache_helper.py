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
        schedule: Dict[str, List[int]],
    ):
        """
        Generalized SmoothCacheHelper to wrap specified components in specified block classes.

        Args:
            model (nn.Module): The model to wrap.
            block_classes (Type[nn.Module] or List[Type[nn.Module]]): The block class(es) to search for.
            components_to_wrap (List[str]): The names of the components within the blocks to wrap.
            schedule (Dict[str, List[int]]): A dictionary mapping component names to lists of 0/1 values for each timestep.
                                              1 means run normally, 0 means use cached result.
        """
        self.model = model
        self.block_classes = block_classes if isinstance(block_classes, list) else [block_classes]
        self.components_to_wrap = components_to_wrap
        self.schedule = schedule

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
        # Extract component name and block index from full_name
        names = full_name.split('.')
        component_name = names[-1]  # e.g., 'attn' or 'mlp'
        block_index = names[-2]     # e.g., '0', '1', '2', etc.
        schedule_key_with_index = f"{component_name}-{block_index}"
        schedule_key_without_index = component_name

        # Determine which schedule key to use
        if schedule_key_with_index in self.schedule:
            # Use the schedule specific to the block
            schedule_key = schedule_key_with_index
        elif schedule_key_without_index in self.schedule:
            # Use the general schedule for the component
            schedule_key = schedule_key_without_index
        else:
            # If neither key is in the schedule, do not skip
            return False

        # Get the current timestep for this module
        current_step = self.current_steps.get(full_name, 0) - 1  # Adjust index to start from 0

        # Retrieve the schedule list for the selected key
        schedule_list = self.schedule[schedule_key]

        if current_step < 0 or current_step >= len(schedule_list):
            # If current_step is out of bounds, do not skip
            return False

        # 1 means run normally, 0 means use cached result (skip computation)
        skip = schedule_list[current_step] == 0

        return skip



    def wrap_components(self):
        # Wrap specified components within each block class
        for block_name, block in self.model.named_modules():
            if any(isinstance(block, cls) for cls in self.block_classes):
                self.wrap_block_components(block, block_name)

    def wrap_block_components(self, block, block_name):
        #TODO: verify block exists
        if len(self.components_to_wrap) > 0:
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
        # Clear original_forwards to avoid accumulating stale states
        self.original_forwards.clear()

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
                print("Returning cached result for", full_name, "at step", self.current_steps[full_name])
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
