"""
Package for aproximating functions from SEGY velocity models.
"""

from .sizing_function_utils import (
    create_sizing_function,
    read_segy_velocity_model,
    calculate_wavelength_sizing,
    interpolate_size
)

__all__ = [
    'create_sizing_function',
    'read_segy_velocity_model', 
    'calculate_wavelength_sizing',
    'interpolate_size'
]