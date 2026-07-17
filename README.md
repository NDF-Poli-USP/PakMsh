# PakMsh

**PakMsh** is a two-dimensional adaptive mesh-generation package designed for geophysical and seismic velocity models.

The software converts a SEG-Y velocity model into a spatial element-sizing field and generates an initial point distribution using adaptive bubble packing. The initial mesh can then be improved using physical relaxation, density-weighted centroidal Voronoi tessellation on the CPU, discrete centroidal Voronoi tessellation on the GPU, and quality-preserving Laplacian smoothing.

PakMsh supports:

- wavelength-based mesh sizing from SEG-Y velocity models;
- variable element sizes across heterogeneous media;
- adaptive bubble-packing point generation;
- rectangular and superelliptical padding;
- preservation of internal physical-model boundaries;
- physical spring-based mesh smoothing;
- density-weighted CPU CVT smoothing;
- optional GPU CVT smoothing with CuPy;
- quality-preserving Numba Laplacian smoothing;
- Delaunay triangulation;
- mesh-quality and sizing-conformity statistics;
- mesh, bubble, sizing-field, and Voronoi visualization;
- Gmsh 2.2 mesh export through Meshio.

---

## Tested environment

PakMsh was tested with the following environment:

```text
Python: 3.12.3
Platform: Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
numpy: 2.2.5
scipy: 1.15.3
matplotlib: 3.10.3
numba: 0.61.2
segyio: 1.9.13
cupy-cuda12x: 13.6.0

GPU INFORMATION
---------------
CuPy: 13.6.0
CUDA runtime: 12090
CUDA driver: 13000
GPU devices: 1
GPU name: NVIDIA GeForce RTX 3070 Ti Laptop GPU
```

---

## Installation on WSL

Open the WSL terminal in the directory where PakMsh will be used.

### Update the package list

```bash
sudo apt update
```

### Install Python 3.12 and virtual-environment support

```bash
sudo apt install python3.12 python3.12-venv python3-pip
```

### Create a virtual environment

```bash
python3.12 -m venv pakmsh_env
```

### Activate the environment

```bash
source pakmsh_env/bin/activate
```

### Update pip

```bash
pip install --upgrade pip
```

### Install NumPy

```bash
pip install numpy==2.2.5
```

### Install SciPy

```bash
pip install scipy==1.15.3
```

### Install Matplotlib

```bash
pip install matplotlib==3.10.3
```

### Install Numba

```bash
pip install numba==0.61.2
```

### Install Segyio

```bash
pip install segyio==1.9.13
```

### Install Meshio for mesh export

```bash
pip install meshio
```

### Install Jupyter Notebook to run the supplied example

```bash
pip install notebook
```

---

## BP2004 quick-start example: CPU CVT mesh

The following minimal example downloads the **BP 2004 velocity model**, creates a wavelength-based sizing function, generates the initial PakMsh point distribution, applies CPU centroidal Voronoi tessellation smoothing, plots the final mesh, and exports it in Gmsh 2.2 `.msh` format.

### Download the BP 2004 velocity model

The velocity model is publicly available from the Open Source Geoscience Amazon S3 repository:

[Download `vel_z6.25m_x12.5m_exact.segy.gz`](http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/vel_z6.25m_x12.5m_exact.segy.gz)

Download and decompress it from the terminal:

```bash
mkdir -p data

wget -O data/vel_z6.25m_x12.5m_exact.segy.gz \
  http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/vel_z6.25m_x12.5m_exact.segy.gz

gunzip -f data/vel_z6.25m_x12.5m_exact.segy.gz
```

The resulting file should be:

```text
data/vel_z6.25m_x12.5m_exact.segy
```

### Minimal CPU CVT example

Save the following script as `bp2004_cvt_cpu.py` in the directory containing the `PakMsh` package:

```python
import numpy as np
import meshio

from PakMsh.generationUtils import generate_mesh
from PakMsh.sizingUtils import create_sizing_function
from PakMsh.smoothingUtils import cvt_smooth_cpu
from PakMsh.plotUtils import plot_mesh


# BP 2004 physical domain.
depth_z = -12_000.0
length_x = 67_000.0

# Downloaded SEG-Y velocity model.
segy_file = "data/vel_z6.25m_x12.5m_exact.segy"
segy_bbox = (depth_z, 0.0, 0.0, length_x)

# Wave-propagation and sizing parameters.
elements_per_wavelength = 2
maximum_frequency = 9.0
grading = 0.85

# Bubble-packing parameters used by the BP2004 example.
sampling_resolution = 8_379
maximum_points = 5_000_000
overlap = 1.25

# Create the wavelength-based element-sizing function.
sizing_function, minimum_size, maximum_size = create_sizing_function(
    fname=segy_file,
    hmin=0.0,
    bbox=segy_bbox,
    wl=elements_per_wavelength,
    freq=maximum_frequency,
    pad_type=None,
    grade=grading,
)

# Generate the initial adaptive bubble-packing mesh.
points, initial_triangles, boundary_points = generate_mesh(
    x_range=(0.0, length_x),
    z_range=(0.0, depth_z),
    npoints=maximum_points,
    density_function=sizing_function,
    N=sampling_resolution,
    pad_type=None,
    ellipse_n=3.0,
    padding_x=0.0,
    padding_z=0.0,
    subdomain=True,
    overlap=overlap,
    fineness="custom",
    f_min=minimum_size,
    f_max=maximum_size,
)

# Apply density-weighted centroidal Voronoi smoothing on the CPU.
cvt_points, cvt_triangles = cvt_smooth_cpu(
    points.copy(),
    sizing_function,
    0.0,          # x minimum
    length_x,     # x maximum
    depth_z,      # z minimum
    0.0,          # z maximum
    iterations=150,
    influence=1.0,
    hold_boundary=True,
    boundary_points=boundary_points,
)

# Plot the final CPU-CVT mesh.
plot_mesh(
    cvt_points,
    x_range=(0.0, length_x),
    z_range=(0.0, depth_z),
    density_function=sizing_function,
    show_points=False,
    show_density=False,
    show_element_size=False,
    filename="BP2004_CVT_CPU",
)

# Export the two-dimensional mesh as Gmsh 2.2 ASCII.
mesh_points_3d = np.column_stack(
    (
        cvt_points[:, 0],
        cvt_points[:, 1],
        np.zeros(len(cvt_points)),
    )
)

mesh = meshio.Mesh(
    points=mesh_points_3d,
    cells=[
        ("triangle", np.asarray(cvt_triangles, dtype=np.int32)),
    ],
)

meshio.write(
    "BP2004_CVT_CPU.msh",
    mesh,
    file_format="gmsh22",
    binary=False,
)

print("Created BP2004_CVT_CPU.msh")
```

Run the example with:

```bash
python bp2004_cvt_cpu.py
```

The full-resolution settings above can be computationally demanding. For a faster installation test, temporarily use:

```python
sampling_resolution = 2_000
```

and:

```python
iterations=10
```

The lower-resolution test verifies the workflow but does not reproduce the mesh density shown below.

### BP2004 mesh after CPU CVT smoothing

<p align="center">
  <img src="images/BP2004_CVT_CPU.jpg"
       alt="BP2004 adaptive triangular mesh after CPU centroidal Voronoi smoothing"
       width="100%">
</p>

<p align="center">
  <em>Adaptive PakMsh triangular mesh for the BP 2004 velocity model after density-weighted CPU CVT smoothing.</em>
</p>

---

## Optional GPU installation

CuPy is optional. PakMsh can be imported and its CPU functions can be used without CuPy.

When CuPy is not installed, PakMsh displays a warning informing the user that GPU smoothing is disabled. The bubble-packing generator, sizing utilities, plotting utilities, mesh statistics, physical smoothing, CPU CVT, and Laplacian smoothing remain available.

### Check whether WSL detects the NVIDIA GPU

```bash
nvidia-smi
```

The NVIDIA driver must be installed on Windows. WSL uses the Windows NVIDIA driver to access the GPU.

### Remove conflicting CuPy packages

```bash
pip uninstall -y cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x
```

### Install the tested CUDA 12.x CuPy package

```bash
pip install cupy-cuda12x==13.6.0
```

### Verify CuPy and CUDA

```bash
python -c "import cupy as cp; print('CuPy:', cp.__version__); print('CUDA runtime:', cp.cuda.runtime.runtimeGetVersion()); print('CUDA driver:', cp.cuda.runtime.driverGetVersion()); print('GPU devices:', cp.cuda.runtime.getDeviceCount()); print('GPU test:', int(cp.arange(10).sum()))"
```

A successful test returns:

```text
GPU test: 45
```

---

## Package structure

The package is organized into five utility modules:

```text
PakMsh/
├── generationUtils/
│   ├── __init__.py
│   └── generation_utils.py
├── sizingUtils/
│   ├── __init__.py
│   └── sizing_function_utils.py
├── smoothingUtils/
│   ├── __init__.py
│   └── smoothing_utils.py
├── plotUtils/
│   ├── __init__.py
│   └── plot_utils.py
├── statUtils/
│   ├── __init__.py
│   └── stat_utils.py
└── __init__.py
```

Keep the `PakMsh` directory in the same project directory as the scripts or notebooks that use it.

Example project structure:

```text
project/
├── PakMsh/
├── BP2004Example.ipynb
├── velocity_model.segy
└── output_meshes/
```

### Verify the package import

Run this command from the directory containing the `PakMsh` folder:

```bash
python -c "import PakMsh; print('PakMsh imported successfully')"
```

---

## `sizingUtils`

The `sizingUtils` package creates spatial mesh-sizing functions from seismic velocity data.

### `create_sizing_function`

Reads a SEG-Y velocity model and constructs an interpolated element-size function.

The function:

- reads the velocity field;
- replaces zero-valued cells with a water velocity;
- calculates element size from velocity, frequency, and elements per wavelength;
- applies an optional minimum element size;
- extends the sizing grid into rectangular or elliptical padding;
- optionally smooths the sizing field using a two-dimensional Savitzky-Golay filter;
- returns the interpolated sizing function and its minimum and maximum values.

### `read_segy_velocity_model`

Reads all traces and samples from a SEG-Y velocity model and returns the velocity array and model dimensions.

### `calculate_wavelength_sizing`

Calculates the target element size from the local wave velocity.

The sizing criterion is based on:

```text
element size = velocity / (frequency × elements per wavelength)
```

### `interpolate_size`

Evaluates the discrete sizing grid at arbitrary physical coordinates using regular-grid interpolation.

### `apply_savitzky_golay_filter_2d`

Smooths abrupt horizontal and vertical variations in the element-sizing grid to improve mesh grading.

---

## `generationUtils`

The `generationUtils` package creates the initial adaptive mesh using bubble packing.

### `generate_mesh`

Generates an adaptive two-dimensional point mesh and returns:

- the mesh-point coordinates;
- the Delaunay triangulation;
- the fixed boundary-point coordinates.

The generator supports:

- custom or preset sampling resolution;
- nonuniform element sizes;
- controlled bubble overlap;
- rectangular physical domains;
- rectangular padding;
- superelliptical padding;
- internal rectangular subdomain boundaries;
- fixed external and internal boundaries;
- vectorized sizing-function evaluation;
- Numba-accelerated collision detection;
- active-neighbor filtering;
- column-by-column interior-point generation.

### Boundary placement utilities

The module contains separate routines for:

- horizontal boundaries;
- vertical boundaries;
- lower superellipse boundaries;
- corner management;
- tangent or near-tangent boundary-circle placement.

### `process_column_fast`

Uses Numba to generate collision-free points along one vertical sampling column.

The routine uses:

- active neighboring points;
- row-indexed collision searches;
- local circle radii;
- overlap control;
- adaptive vertical skipping after each accepted point.

### `filter_points_inside_superellipse`

Removes points outside a specified superelliptical domain.

---

## `smoothingUtils`

The `smoothingUtils` package improves point distribution and triangle quality after the initial bubble-packing stage.

### `physical_smooth`

Applies a physical spring-relaxation procedure to the Delaunay edges.

The method:

- builds a Delaunay triangulation at every iteration;
- compares actual edge lengths with the target sizing function;
- applies spring-like forces;
- accumulates point displacements;
- keeps specified boundary points fixed;
- returns the smoothed coordinates and triangle connectivity.

### `cvt_smooth_cpu`

Applies density-weighted centroidal Voronoi tessellation on the CPU.

The method:

- builds a Delaunay triangulation;
- reconstructs Voronoi contributions from triangle circumcenters and edge midpoints;
- divides each triangle into six subtriangles;
- evaluates the sizing-dependent density at subtriangle centroids;
- accumulates weighted masses and moments;
- moves each free generator toward its weighted centroid;
- preserves specified boundaries;
- returns the smoothed coordinates and triangle connectivity.

The default two-dimensional density law is based on the inverse fourth power of the target element size.

### `cvt_smooth_gpu`

Applies discrete density-weighted CVT smoothing using CuPy and custom CUDA kernels.

The GPU method uses:

- a discrete integration grid;
- spatial bins for local point searches;
- a GPU Voronoi identifier map;
- local triangle reconstruction from Voronoi junctions;
- density-weighted subtriangle integration;
- double-precision atomic accumulation of masses and moments;
- optional fixed boundaries;
- configurable grid resolution, bin size, density power, and iteration count.

The function returns the GPU-smoothed coordinates and final Delaunay triangle connectivity.

If CuPy is unavailable, GPU smoothing is disabled while all CPU utilities remain usable.

### `smart_laplacian_smooth_numba`

Applies quality-preserving Laplacian smoothing using Numba.

For each movable point, the algorithm:

- calculates the average position of neighboring vertices;
- evaluates the minimum quality of its incident triangles;
- tests the candidate Laplacian position;
- accepts the movement only when local quality is maintained or improved;
- stops early if no point can be moved.

### Internal quality and connectivity utilities

The module also contains utilities for:

- normalized triangle-quality calculation;
- local one-ring quality calculation;
- point-neighbor connectivity;
- point-to-triangle incidence maps;
- Numba-accelerated smoothing loops.

---

## `plotUtils`

The `plotUtils` package provides visualization tools for meshes and sizing fields.

### `plot_mesh`

Creates and saves a Delaunay mesh plot.

The plot can display:

- mesh edges;
- mesh generators;
- the background sizing or velocity field;
- actual mean element sizes;
- normalized triangle quality;
- the lowest-quality element;
- configurable coordinate limits.

### `plot_mesh_circles`

Plots the local bubble associated with every generated point.

This visualization is useful for checking:

- local target sizes;
- packing density;
- overlap;
- boundary placement;
- graded transitions.

### `plot_fast_voronoi`

Creates a Voronoi diagram from the mesh generators for inspection of the point distribution and CVT structure.

---

## `statUtils`

The `statUtils` package evaluates triangle quality and conformity with the target sizing field.

### `get_mesh_stats`

Constructs a Delaunay triangulation and returns:

- the number of triangles;
- the mean normalized triangle quality;
- the minimum normalized triangle quality.

The quality measure ranges from zero for a degenerate triangle to one for an equilateral triangle.

### `mesh_sizing_check`

Compares actual triangle sizes with the target sizing function.

The function:

- calculates triangle centroids;
- evaluates the target element size at each centroid;
- calculates the actual size from the mean triangle-edge length;
- measures elements that are larger than the target;
- returns a global conformity score;
- can return detailed triangle-level diagnostic arrays.

PakMsh generation and smoothing utilities store coordinates in `(x, z)` order. The sizing-statistics utility supports both `(x, z)` and `(z, x)` input through its coordinate-order option.

---

## BP2004 example notebook

`BP2004Example.ipynb` demonstrates the complete PakMsh workflow using a SEG-Y velocity model.

The notebook includes:

- creation of the wavelength-based sizing function;
- adaptive bubble-packing mesh generation;
- optional rectangular or elliptical padding;
- physical smoothing;
- CPU CVT smoothing;
- optional GPU CVT smoothing;
- quality-preserving Laplacian smoothing;
- execution-time measurements using `time.perf_counter`;
- mesh-quality statistics;
- sizing-conformity statistics;
- bubble, mesh, sizing-field, and Voronoi plots;
- Gmsh 2.2 ASCII mesh export with Meshio.

### Start Jupyter Notebook

```bash
source pakmsh_env/bin/activate
jupyter notebook
```

Open `BP2004Example.ipynb` from the Jupyter interface.

---

## Mesh export

The supplied example exports triangular meshes using Meshio in Gmsh 2.2 ASCII format.

The exported coordinates are placed in the Gmsh `(x, y, z)` system as:

```text
x = PakMsh horizontal coordinate
y = PakMsh depth coordinate
z = 0
```

This keeps the two-dimensional geophysical mesh correctly oriented in Gmsh and ParaView.

The example exports meshes after:

- physical smoothing;
- CPU CVT smoothing;
- GPU CVT smoothing;
- GPU CVT followed by Laplacian smoothing.

---

## Running PakMsh

Activate the environment:

```bash
source pakmsh_env/bin/activate
```

Run a Python script:

```bash
python run_mesh.py
```

Start the supplied notebook:

```bash
jupyter notebook BP2004Example.ipynb
```

Deactivate the environment when finished:

```bash
deactivate
```

---

## Performance notes

- The first Numba execution includes compilation overhead.
- The first CUDA execution may include CUDA kernel compilation overhead.
- Use later executions when measuring steady-state performance.
- GPU operations may be asynchronous; synchronize the GPU when measuring isolated CUDA execution time.
- Higher bubble-generation resolution improves placement accuracy but increases preprocessing cost.
- Higher GPU CVT integration resolution improves the discrete Voronoi approximation but increases memory use and execution time.
- Large meshes may require substantial system memory and GPU memory.
- Only one CuPy package variant should be installed at a time.

---

## Current dependencies

### Required

- Python 3.12
- NumPy
- SciPy
- Matplotlib
- Numba
- Segyio

### Optional

- CuPy with CUDA 12.x for GPU CVT
- Meshio for Gmsh mesh export
- Jupyter Notebook for the supplied BP2004 example
