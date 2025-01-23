# calibration_helper.py
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
        self.calibration_threshold = calibration_threshold
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
            print(len(self.calibration_results[full_name][j]))
\
            # Update previous outputs
            self.previous_layer_outputs[full_name].insert(0, current_output.detach().clone())
            if len(self.previous_layer_outputs[full_name]) > self.calibration_lookahead:
                self.previous_layer_outputs[full_name].pop()

            return output
        return wrapped_forward

    def generate_schedule(self):
        """
        Generate schedules for each exact component name (e.g. 'attn1', 'mlp1', etc.)
        using the 3-row scanning logic.

        For example, if self.calibration_results has keys:
        'transformer_blocks.0.attn1', 'transformer_blocks.1.attn1', 'transformer_blocks.0.mlp1', ...
        we parse out the last part (e.g. 'attn1', 'mlp1') as `component_full`,
        and group all blocks that share that same component_full.

        Each group yields 3 arrays: row0, row1, row2, averaged across all blocks,
        then scanned to produce a 50-length schedule.

        Returns:
            A dictionary like:
            {
            'attn1': [50-length schedule],
            'mlp1':  [50-length schedule],
            ...
            }
        """
        import numpy as np
        from collections import defaultdict

        # Dictionary: component_name -> [ list_of_arrays_row0, list_of_arrays_row1, list_of_arrays_row2 ]
        component_to_rows = defaultdict(lambda: [[], [], []])

        # Step A: Collect row0, row1, row2 arrays by exact component name
        for full_name, sublists in self.calibration_results.items():
            if len(sublists) < self.calibration_lookahead:
                # skip if incomplete
                continue

            # e.g. 'transformer_blocks.0.attn1' => component_full='attn1'
            component_full = full_name.split('.')[-1]  # e.g. 'attn1'

            # sublists is e.g. [row0_list, row1_list, row2_list]
            # convert each to numpy array
            for row_idx in range(len(sublists)):
                arr = np.array(sublists[row_idx], dtype=float)
                component_to_rows[component_full][row_idx].append(arr)

        final_schedules = {}

        # Step B: For each component_full, average row0, row1, row2, then produce 50-length schedule
        for component_full, row_lists in component_to_rows.items():
            row0_arrays = row_lists[0]  # list of np arrays for row0
            row1_arrays = row_lists[1]  # row1
            row2_arrays = row_lists[2]  # row2

            avg0_list = self._average_arrays(row0_arrays)  # length ~ 49
            avg1_list = self._average_arrays(row1_arrays)  # length ~ 48
            avg2_list = self._average_arrays(row2_arrays)  # length ~ 47

            schedule = self._scan_3row_sublists(avg0_list, avg1_list, avg2_list, self.calibration_threshold)
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

    def _scan_3row_sublists(self, row0_list, row1_list, row2_list, threshold):
        """
        Based on your scanning logic:
        - We produce a schedule of length 50.
        - schedule[0] = 1
        - For each i in [1..49], we check row2[i-1], row1[i-1], row0[i-1] (if in range)
        * if row2[i-1] <= threshold => schedule i=1, i+1..i+3=0, skip i+4
        * else if row1[i-1] <= threshold => schedule i=1, i+1..i+2=0, skip i+3
        * else if row0[i-1] <= threshold => schedule i=1, i+1=0, skip i+2
        * else => schedule i=1, skip i+1
        - Finally override schedule[49] = 1
        """

        schedule = [None]*50
        i = 0

        while i < 50:
            # breakpoint()
            idx = i  # to read from row2, row1, row0
            used = False

            # check row2 if idx < len(row2_list)
            if idx < len(row2_list):
                if row2_list[idx] <= threshold:
                    schedule[i] = 1
                    for skip_step in (i+1, i+2, i+3):
                        if skip_step < 50:
                            schedule[skip_step] = 0
                    i += 4
                    used = True
            if not used and idx < len(row1_list):
                if row1_list[idx] <= threshold:
                    schedule[i] = 1
                    for skip_step in (i+1, i+2):
                        if skip_step < 50:
                            schedule[skip_step] = 0
                    i += 3
                    used = True
            if not used and idx < len(row0_list):
                if row0_list[idx] <= threshold:
                    schedule[i] = 1
                    if i+1 < 50:
                        schedule[i+1] = 0
                    i += 2
                    used = True
            if not used:
                # fallback => schedule[i]=1
                schedule[i] = 1
                i += 1
            # print(schedule)
            # breakpoint()

        # override schedule[49] = 1
        schedule[0] = 1
        schedule[-1] = 1

        # fill any None with 1
        for x in range(50):
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
