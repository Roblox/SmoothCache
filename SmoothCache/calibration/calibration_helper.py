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
        Generate one averaged schedule for each component type (e.g. "attn", "mlp") using
        only the last sublist of errors (j = self.calibration_lookahead - 1).
        - We'll gather all blocks with the same component type,
        e.g. "transformer_blocks.2.attn1" => "attn"
        - We'll average their error curves (the last sublist in calibration_results) by timestep.
        - We'll compare that averaged error to self.calibration_threshold to produce 0/1 decisions.
        - We forcibly set the first self.calibration_lookahead timesteps to 1 (always compute).
        """
        from collections import defaultdict
        import re

        # Group the last-sublist-of-errors by component type
        component_type_to_block = defaultdict(list)

        # We only calculate error curve with max lookahead
        j_index = self.calibration_lookahead - 1

        for full_name, error_sublists in self.calibration_results.items():
            # error_sublists is a list of length = calibration_lookahead
            # We specifically want error_sublists[j_index]
            if j_index >= len(error_sublists):
                continue  # or skip if it doesn't exist for some reason

            sublist_errors = error_sublists[j_index]  # This is a list of floats, length = #timesteps for that block

            # Extract the component type
            # e.g. "transformer_blocks.2.attn1" -> "attn"
            names = full_name.split('.')
            component_full = names[-1]  # e.g. 'attn1'
            match = re.match(r"([a-zA-Z]+)", component_full)
            component_type = match.group(1) if match else component_full  # e.g. "attn", "mlp"

            component_type_to_block[component_type].append(sublist_errors)

        # Produce a schedule for each component type
        final_schedules = {}

        for comp_type, list_of_error_lists in component_type_to_block.items():
            # list_of_error_lists is like [ [err_blockA], [err_blockB], ... ]

            # find the max length among them
            max_len = 0
            for errs in list_of_error_lists:
                if len(errs) > max_len:
                    max_len = len(errs)

            # We'll sum across blocks at each timestep
            sum_at_timestep = [0.0] * max_len
            count_at_timestep = [0]   * max_len

            # Accumulate
            for err_list in list_of_error_lists:
                for i, val in enumerate(err_list):
                    sum_at_timestep[i] += val
                    count_at_timestep[i] += 1

            # Now compute average
            avg_errors = []
            for i in range(max_len):
                if count_at_timestep[i] > 0:
                    avg_val = sum_at_timestep[i] / count_at_timestep[i]
                else:
                    avg_val = 0.0  # or skip
                avg_errors.append(avg_val)
            print(avg_errors)
            # Build schedule
            schedule_list = []
            for i in range(max_len):
                # Compare to threshold
                val = avg_errors[i]
                decision = 1 if val > self.calibration_threshold else 0
                schedule_list.append(decision)

            final_schedules[comp_type] = [1] * self.calibration_lookahead + schedule_list
            # print(final_schedules)

        return final_schedules

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
