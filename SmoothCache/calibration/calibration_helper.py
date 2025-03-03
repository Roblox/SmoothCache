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

import json
import re
import statistics
from typing import Dict, List, Optional, Union, Type
import torch
import torch.nn as nn
from pathlib import Path


def rel_l1_loss(prev_output, cur_output):
    """
    Compute the relative L1 loss between prev_output and cur_output as a single float.
    
    Args:
        prev_output (torch.Tensor): Previous layer output. Shape: [batch_size, channels, ...]
        cur_output (torch.Tensor): Current layer output. Shape: [batch_size, channels, ...]
    
    Returns:
        float: Relative L1 loss across the entire batch, on flattened inputs, 
        Since DiTPipeline will duplicate the batch anyway. 
    """
    output_diff = prev_output.float() - cur_output.float()  
    numerator = torch.norm(output_diff, p=1)               
    denominator = torch.norm(cur_output.float(), p=1)      
    relative_l1 = numerator / denominator         
    return relative_l1.cpu().item()   

class CalibrationHelper:
    def __init__(
        self,
        model: nn.Module,
        block_classes: Union[Type[nn.Module], List[Type[nn.Module]]],
        components_to_wrap: List[str],
        calibration_lookahead: int = 3,
        calibration_threshold: float = 0.0,
        schedule_length: int = 50,
        log_file: str = "calibration_schedule.json"
    ):
        """
        Base CalibrationHelper that dynamically wraps specified components for calibration.
        
        Args:
            model (nn.Module): The model whose components we want to calibrate.
            block_classes (Union[Type[nn.Module], List[Type[nn.Module]]]): The block class(es) identifying which blocks to wrap.
            components_to_wrap (List[str]): Component names within each block to wrap (e.g. ['attn1', 'mlp']).
            calibration_lookahead (int): Number of steps to look back when computing errors.
            log_file (str): Path to save the generated schedule.
        """
        self.model = model
        self.block_classes = block_classes if isinstance(block_classes, list) else [block_classes]
        self.components_to_wrap = components_to_wrap

        self.calibration_lookahead = calibration_lookahead
        # Validate calibration_lookahead
        if self.calibration_lookahead <= 0:
            raise ValueError("calibration_lookahead must be greater than 0.")

        self.calibration_threshold = calibration_threshold
        self.schedule_length = schedule_length
        self.log_file = log_file

        # Tracking original forward methods
        self.original_forwards = {}

        # Tracking steps and outputs
        self.current_steps = {}
        self.previous_layer_outputs = {}
        self.calibration_results = {}

        # State
        self.enabled = False

    def enable(self):
        """
        Enable calibration mode by wrapping components at runtime.
        After enabling, simply run your pipeline once to collect calibration data.
        """
        if self.enabled:
            return
        self.enabled = True
        self.reset_state()
        self.wrap_components()

    def disable(self):
        """
        Disable calibration mode, unwrap the components, generate the schedule, and save it.
        Ensures that the destination directory exists before writing the schedule JSON.
        """
        if not self.enabled:
            return
        self.enabled = False
        self.unwrap_components()
        generated_schedule = self.generate_schedule()

        log_path = Path(self.log_file)
        if log_path.parent:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("w") as f:
            f.write("{\n")
            for i, (key, value) in enumerate(generated_schedule.items()):
                # Serialize the list as a compact JSON list
                value_str = json.dumps(value, separators=(',', ':'))
                # Write the key-value pair
                f.write(f'    "{key}": {value_str}')
                if i < len(generated_schedule) - 1:
                    f.write(",\n")
                else:
                    f.write("\n")
            f.write("}\n")
        
        self.reset_state()

    def reset_state(self):
        """
        Reset internal state.
        """
        self.current_steps.clear()
        self.previous_layer_outputs.clear()
        self.calibration_results.clear()

    def wrap_components(self):
        """
        Wrap the specified components in the given block classes.
        """
        for block_name, block in self.model.named_modules():
            if any(isinstance(block, cls) for cls in self.block_classes):
                self.wrap_block_components(block, block_name)

    def wrap_block_components(self, block, block_name: str):
        """
        Wrap the target components (e.g., 'attn1') in each block.
        """
        for comp_name in self.components_to_wrap:
            if hasattr(block, comp_name):
                component = getattr(block, comp_name)
                full_name = f"{block_name}.{comp_name}"
                self.original_forwards[full_name] = component.forward
                wrapped_forward = self.create_wrapped_forward(full_name, component.forward)
                component.forward = wrapped_forward

    def unwrap_components(self):
        """
        Restore original forward methods for all wrapped components.
        """
        for full_name, original_forward in self.original_forwards.items():
            module = self.get_module_by_name(self.model, full_name)
            if module is not None:
                module.forward = original_forward
        self.original_forwards.clear()

    def create_wrapped_forward(self, full_name: str, original_forward):
        """
        Create a wrapped forward method that intercepts outputs, computes errors, and stores them.
        """
        def wrapped_forward(*args, **kwargs):
            # Increment step counter
            step = self.current_steps.get(full_name, 0) + 1
            self.current_steps[full_name] = step

            # Call original forward
            output = original_forward(*args, **kwargs)

            # 'output' is the layer output for this component. We treat it as a torch.Tensor
            # Store and compute error vs previous steps
            # Initialize storage if not present
            if full_name not in self.previous_layer_outputs:
                self.previous_layer_outputs[full_name] = [None] * self.calibration_lookahead
            if full_name not in self.calibration_results:
                self.calibration_results[full_name] = [[] for _ in range(self.calibration_lookahead)]

            current_output = output
            # Compare with previous outputs
            for j in range(self.calibration_lookahead):
                prev_output = self.previous_layer_outputs[full_name][j]
                if prev_output is not None and current_output is not None:
                    # Compute error
                    error = rel_l1_loss(prev_output, current_output)
                    self.calibration_results[full_name][j].append(error)

            # Update previous outputs
            self.previous_layer_outputs[full_name].insert(0, current_output.detach().clone())
            if len(self.previous_layer_outputs[full_name]) > self.calibration_lookahead:
                self.previous_layer_outputs[full_name].pop()

            return output
        return wrapped_forward

    def generate_schedule(self):
        """
        Generate schedules for each exact component name (e.g., 'attn1', 'mlp1', etc.) 
        using n-row scanning logic, where n is arbitrary based on calibration_lookahead.
        
        For example, if self.calibration_results has keys:
        'transformer_blocks.0.attn1', 'transformer_blocks.1.attn1', 'transformer_blocks.0.mlp1', ...
        we parse out the last part (e.g., 'attn1', 'mlp1') as `component_full`,
        and group all blocks that share that same component_full.

        Each group yields n arrays: row0, row1, ..., row(n-1)_list, averaged across all blocks,
        then scanned to produce the schedule.

        Returns:
            A dictionary like:
            {
                'attn1': [schedule_length schedule],
                'mlp1':  [schedule_length schedule],
                ...
            }
        """
        import numpy as np
        from collections import defaultdict

        # Dictionary: component_name -> list of lists for each row
        component_to_rows = defaultdict(list)

        # Step A: Collect row arrays by exact component name
        for full_name, sublists in self.calibration_results.items():
            if len(sublists) < self.calibration_lookahead:
                # skip if incomplete
                continue

            # e.g., 'transformer_blocks.0.attn1' => component_full='attn1'
            component_full = full_name.split('.')[-1]  # e.g., 'attn1'

            # sublists is a list of row arrays for this component
            component_to_rows[component_full].append(sublists)

        final_schedules = {}

        # Step B: For each component_full, average rows and produce schedule
        for component_full, sublist_groups in component_to_rows.items():
            # Assuming each sublist_group has the same number of rows (calibration_lookahead)
            num_rows = len(sublist_groups[0]) if sublist_groups else 0

            # Average each row across all blocks
            averaged_rows = []
            for row_idx in range(num_rows):
                row_arrays = [sublist[row_idx] for sublist in sublist_groups]
                avg_row_list = self._average_arrays(row_arrays)
                averaged_rows.append(avg_row_list)

            schedule = self._scan_nrows_sublists(averaged_rows, self.calibration_threshold)
            final_schedules[component_full] = schedule

        print(final_schedules)
        return final_schedules

    def _average_arrays(self, array_list):
        """
        Given a list of 1D numpy arrays of potentially different lengths, 
        compute the average across them at each index. 
        Returns a Python list of floats for the average.
        e.g. if array_list = [arrA(len=49), arrB(len=49), arrC(len=48), ...] 
        we find max_len, sum, count -> average.
        """
        import numpy as np
        if not array_list:
            return []

        max_len = max(len(arr) for arr in array_list)
        sum_vals = np.zeros(max_len, dtype=float)
        count_vals = np.zeros(max_len, dtype=int)

        for arr in array_list:
            for i, val in enumerate(arr):
                sum_vals[i] += val
                count_vals[i] += 1

        avg_arr = np.zeros(max_len, dtype=float)
        for i in range(max_len):
            if count_vals[i] > 0:
                avg_arr[i] = sum_vals[i] / count_vals[i]
        return avg_arr.tolist()

    def _scan_nrows_sublists(self, row_lists, threshold):
        """
        Scan through multiple rows (arbitrary number) in reverse order to produce a schedule.
        
        Parameters:
            row_lists (list of lists): A list where each element is a row's list of values
                                    ordered from highest priority to lowest.
            threshold (float): The threshold value to check against.

        Returns:
            schedule (list): The generated schedule based on the scanning logic.
        """
        schedule = [None] * self.schedule_length
        i = 0

        while i < self.schedule_length:
            idx = i
            used = False

            # Iterate through each row in reverse order (highest priority first)
            for row_idx in range(len(row_lists)-1, -1, -1):
                current_row_list = row_lists[row_idx]
                if idx >= len(current_row_list):
                    continue  # Skip if index is out of bounds for this row

                if current_row_list[idx] <= threshold:
                    # Activate the current step
                    schedule[i] = 1
                    
                    # Determine how many steps to skip based on the row priority
                    num_skips = row_idx + 1  # More skips for higher priority rows
                    skip_steps = []
                    for s in range(1, num_skips + 1):
                        skip_step = i + s
                        if skip_step < self.schedule_length:
                            schedule[skip_step] = 0
                            skip_steps.append(skip_step)
                    
                    # Move the index past the skipped steps
                    i += (num_skips + 1)  # Move to the step after the last skip
                    used = True
                    break

            if not used:
                # Fallback: Activate current step without skipping
                schedule[i] = 1
                i += 1

        # Override the first and last steps to be active
        if self.schedule_length > 0:
            schedule[0] = 1
            schedule[-1] = 1

        # Fill any remaining None values with 1
        for x in range(self.schedule_length):
            if schedule[x] is None:
                schedule[x] = 1

        return schedule

    def get_module_by_name(self, model, full_name):
        """
        Utility to retrieve a module by full name.
        """
        names = full_name.split('.')
        module = model
        for name in names:
            if hasattr(module, name):
                module = getattr(module, name)
            else:
                return None
        return module
