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

from typing import List, Optional
import torch.nn as nn
from .calibration_helper import CalibrationHelper

try:
    from diffusers.models.attention import BasicTransformerBlock
except ImportError:
    BasicTransformerBlock = None

class DiffuserCalibrationHelper(CalibrationHelper):
    def __init__(
        self,
        model: nn.Module,
        calibration_lookahead: int = 3,
        calibration_threshold: float = 0.0,
        schedule_length: int = 50,
        log_file: str = "calibration_schedule.json",
        components_to_wrap: Optional[List[str]] = None
    ):
        """
        Diffuser-specific CalibrationHelper derived from CalibrationHelper.

        Args:
            model (nn.Module): The model to wrap (e.g., pipe.transformer).
            calibration_lookahead (int): Steps to look back for error calculation.
            calibration_threshold (float): Cutoff L1 error value to enable caching.
            schedule_length (int): Length of the generated schedule, 1:1 mapped to pipeline timesteps.
            log_file (str): Path to save the generated schedule JSON.
            components_to_wrap (List[str], optional): List of component names to wrap.
                Defaults to ['attn1'].
        
        Raises:
            ImportError: If diffusers' BasicTransformerBlock is unavailable.
        """
        if BasicTransformerBlock is None:
            raise ImportError("Diffusers library not installed or BasicTransformerBlock not found.")

        block_classes = [BasicTransformerBlock]
        if components_to_wrap is None:
            components_to_wrap = ['attn1']
        super().__init__(
            model=model,
            block_classes=block_classes,
            components_to_wrap=components_to_wrap,
            calibration_lookahead=calibration_lookahead,
            calibration_threshold=calibration_threshold,
            schedule_length=schedule_length,
            log_file=log_file
        )
