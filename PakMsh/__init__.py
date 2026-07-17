"""
PakMsh: A comprehensive mesh generation toolkit for velocity models
based on the Bubble Packing algorithm.

This package provides tools for:
- Reading SEGY velocity models and creating sizing functions
- Generating adaptive mesh points using density-based sampling
- Smoothing and optimizing mesh point distributions
- Visualizing meshes and analyzing mesh quality

Modules:
--------
- sizingUtils: SEGY reading and mesh sizing function creation
- generationUtils: Mesh generation algorithms based on Circle Packing
- smoothingUtils: Mesh optimization and smoothing algorithms
- plotUtils: Visualization and analysis tools
- statUtils: Mesh quality metrics
"""

__version__ = "1.0.0"
__author__ = "R. A. Soares Jr."

# Import package submodules.
from . import sizingUtils
from . import generationUtils 
from . import smoothingUtils
from . import plotUtils
from . import statUtils

# Expose commonly used functions at package level.
from .sizingUtils import (
    create_sizing_function,
    read_segy_velocity_model,
    calculate_wavelength_sizing,
    interpolate_size
)

from .generationUtils import (
    generate_mesh
)

from .smoothingUtils import (
    physical_smooth,
    cvt_smooth_cpu,
    cvt_smooth_gpu,
    smart_laplacian_smooth_numba
)

from .plotUtils import (
    plot_mesh,
    plot_mesh_circles,
    plot_fast_voronoi
)

from .statUtils import (
    mesh_sizing_check,
    get_mesh_stats
)

__all__ = [
    # Package submodules.
    'sizingUtils',
    'generationUtils', 
    'smoothingUtils',
    'plotUtils',
    'statUtils',
	
    # Public function exports.
    'create_sizing_function',
    'read_segy_velocity_model',
    'calculate_wavelength_sizing', 
    'interpolate_size',
    'generate_mesh',
    'physical_smooth',
    'cvt_smooth_cpu',
    'cvt_smooth_gpu',
    'smart_laplacian_smooth_numba',
    'plot_mesh',
    'plot_mesh_circles',
	'plot_fast_voronoi',
    'mesh_sizing_check',
    'get_mesh_stats'
]