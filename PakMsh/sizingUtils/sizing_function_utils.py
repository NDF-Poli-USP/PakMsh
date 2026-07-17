import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator

def create_sizing_function(fname, hmin=None, bbox=None, wl=10, freq=2, pad_type=None, pad_size_x=-1.0,pad_size_z=-1.0,grade=None,vp_water=None):
    """Creates an interpolated mesh-sizing function from a SEG-Y velocity model.
    
    Parameters
    ----------
    fname : str or path-like
        SEG-Y velocity-model file.
    hmin : float, optional
        Minimum permitted element size.
    bbox : tuple of float, optional
        Bounding box ordered as ``(z_min, z_max, x_min, x_max)``.
    wl : float, optional
        Number of elements per wavelength.
    freq : float, optional
        Maximum modeled frequency in hertz.
    pad_type : {None, "rectangular", "elliptical"}, optional
        Padding geometry applied to the sizing grid.
    pad_size_x : float, optional
        Horizontal padding distance.
    pad_size_z : float, optional
        Vertical padding distance.
    grade : float, optional
        Smoothing grade used to derive Savitzky-Golay window lengths.
    vp_water : float, optional
        Velocity assigned to zero-valued water cells.
    
    Returns
    -------
    sizing_function : callable
        Function evaluating element size at ``(x, y)``.
    minimum_size : float
        Minimum value in the processed sizing grid.
    maximum_size : float
        Maximum value in the processed sizing grid.
    """
    
    # Read velocity model with provided bbox.
    vp, n_samples, n_traces = read_segy_velocity_model(fname)
    # Set water velocity if value = 0.
    if vp_water is not None:
        vp = np.where(vp == 0, vp_water, vp)
    else:
        vp = np.where(vp == 0, 1500.0, vp)
    # Calculate wavelength-based sizing.
    cell_size = calculate_wavelength_sizing(vp, wl, freq)
    # Enforce minimum element size.
    if hmin is not None:
        cell_size = np.maximum(cell_size, hmin) # Values already above hmin remain unchanged.

    # Apply padding.
    if (pad_type == "rectangular" or pad_type == "elliptical" ):
        dz=(bbox[1]-bbox[0])/n_traces
        dx=(bbox[3]-bbox[2])/n_samples
        nnz = int(pad_size_z / dz)
        nnx = int(pad_size_x / dx)
        print(nnx,nnz,n_samples,n_traces)
        print(pad_size_z,bbox[0],bbox[2])
        padding = ((0, nnz), (nnx, nnx))
        cell_size = np.pad(cell_size, padding, "edge")
        bbox = (
                bbox[0] - pad_size_z,
                bbox[1],
                bbox[2] - pad_size_x,
                bbox[3] + pad_size_x,
            )

    print(cell_size.shape[0])
    print(cell_size.shape[1])
    if grade is not None:
        window_length_z = int((1.0 - grade) *0.1* cell_size.shape[0])
        window_length_x = int((1.0 - grade) *0.1* cell_size.shape[1])
        cell_size = apply_savitzky_golay_filter_2d(cell_size,window_length_x,window_length_z )
        
    print("Function Minimum and Maximum values:")
    print(cell_size.min(),cell_size.max())
    # Create interpolation function.
    def sizing_function(x, y):
        """Evaluates the processed sizing grid at supplied coordinates.
        
        Parameters
        ----------
        x : float or array-like
            Horizontal coordinate or coordinates.
        y : float or array-like
            Vertical coordinate or coordinates.
        
        Returns
        -------
        size : float or numpy.ndarray
            Interpolated element size at each coordinate.
        """
        
        return interpolate_size(x, y, cell_size, bbox)
    
    return sizing_function,cell_size.min(),cell_size.max()

def read_segy_velocity_model(fname):
    """Reads a velocity model from a SEG-Y file.
    
    Parameters
    ----------
    fname : str or path-like
        SEG-Y file to read.
    
    Returns
    -------
    vp : numpy.ndarray
        Velocity array with shape ``(n_samples, n_traces)``.
    n_traces : int
        Number of traces in the file.
    n_samples : int
        Number of samples per trace.
    
    Raises
    ------
    ImportError
        If the optional ``segyio`` dependency is unavailable.
    """
    import segyio
    import numpy as np
    
    print(f"Reading SEGY file: {fname}")
    
    # Open SEGY file.
    with segyio.open(fname, 'r', ignore_geometry=True) as segy:
         
        n_traces = len(segy.trace)
        n_samples = len(segy.samples)
        
        # Read traces directly into array.
        vp = np.zeros((n_samples, n_traces))
        for i in range(n_traces):
            vp[:, i] = segy.trace[i]
    print(f"Final velocity range: {vp.min():.1f} - {vp.max():.1f}")
    return vp,n_traces,n_samples

def calculate_wavelength_sizing(vp, wl, freq):
    """Calculates target element sizes from a wavelength criterion.
    
    Parameters
    ----------
    vp : array-like
        Wave-propagation velocity values.
    wl : float
        Number of elements per wavelength.
    freq : float
        Maximum modeled frequency in hertz.
    
    Returns
    -------
    cell_size : numpy.ndarray
        Target element sizes computed as ``vp / (freq * wl)``.
    """
    wavelength = vp / freq
    
    # Element size = wavelength / number of elements per wavelength.
    cell_size = wavelength / wl
    
    return cell_size

def interpolate_size(x, y, cell_size, bbox):
    """Interpolates a two-dimensional sizing grid at arbitrary coordinates.
    
    Parameters
    ----------
    x : float or array-like
        Horizontal coordinate or coordinates.
    y : float or array-like
        Vertical coordinate or coordinates.
    cell_size : numpy.ndarray
        Two-dimensional element-size grid.
    bbox : tuple of float
        Grid bounds ordered as ``(z_min, z_max, x_min, x_max)``.
    
    Returns
    -------
    size : float or numpy.ndarray
        Interpolated element size with the broadcast input shape.
    """
    
    # Create coordinate arrays.
    z_coords = np.linspace(bbox[0], bbox[1], cell_size.shape[0])
    x_coords = np.linspace(bbox[2], bbox[3], cell_size.shape[1])
    cell_size_flipped = np.flipud(cell_size) 
    
    # Create interpolator.
    interpolator = RegularGridInterpolator(
        (z_coords, x_coords), 
        cell_size_flipped, 
        method='linear', 
        bounds_error=False, 
        fill_value=None
    )
    
    # Handle scalar or array inputs.
    if np.isscalar(x) and np.isscalar(y):
        points = np.array([[y, x]])  # Y corresponds to z (depth), x to x.
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        points = np.column_stack([y.ravel(), x.ravel()])
    
    result = interpolator(points)
    
    # Return scalar if input was scalar.
    if np.isscalar(x) and np.isscalar(y):
        return float(result[0])
    else:
        return result.reshape(x.shape)


def apply_savitzky_golay_filter_2d(grid_values, window_length_x=501, window_length_z=501, polyorder=3):
    """Applies sequential Savitzky-Golay smoothing along both grid axes.
    
    Parameters
    ----------
    grid_values : array-like of shape (n_rows, n_columns)
        Grid values to smooth.
    window_length_x : int, optional
        Window length used along axis 0.
    window_length_z : int, optional
        Window length used along axis 1.
    polyorder : int, optional
        Polynomial order used by each filter pass.
    
    Returns
    -------
    filtered_values : numpy.ndarray
        Smoothed grid with the same shape as the input.
    
    Raises
    ------
    ValueError
        If a window length or polynomial order is invalid for ``scipy.signal.savgol_filter``.
    """
    
    # Convert input to numpy array.
    grid_values = np.asarray(grid_values)
    
    rows, cols = grid_values.shape
    print(window_length_z,window_length_x)
    # First filter along axis 0 (columns), then along axis 1 (rows).
    filtered_values = savgol_filter(grid_values, window_length_x, polyorder, axis=0)
    filtered_values = savgol_filter(filtered_values, window_length_z, polyorder, axis=1)
    
    return filtered_values
