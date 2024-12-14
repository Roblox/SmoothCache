# SmoothCache/calibration/diffuser_calibration_helper.py

from typing import List, Union, Type
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
        log_file: str = "calibration_schedule.json"
    ):
        """
        Diffuser-specific CalibrationHelper derived from CalibrationHelper.

        Args:
            model (nn.Module): The model to wrap (e.g., pipe.transformer).
            calibration_lookahead (int): Steps to look back for error calculation.
            calibration_threshold (float): Cutoff L1 error value to enable caching.
            log_file (str): Path to save the generated schedule JSON.

        Raises:
            ImportError: If diffusers' BasicTransformerBlock is unavailable.
        """
        if BasicTransformerBlock is None:
            raise ImportError("Diffusers library not installed or BasicTransformerBlock not found.")

        block_classes = [BasicTransformerBlock]
        components_to_wrap = ['attn1']  # Wrap 'attn1' component

        super().__init__(
            model=model,
            block_classes=block_classes,
            components_to_wrap=components_to_wrap,
            calibration_lookahead=calibration_lookahead,
            calibration_threshold=calibration_threshold,
            log_file=log_file
        )
