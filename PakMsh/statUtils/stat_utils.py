import numpy as np
from scipy.spatial import Delaunay

def get_mesh_stats(points):
    """Calculates triangle-count and quality statistics for a point mesh.
    
    Parameters
    ----------
    points : array-like of shape (n_points, 2)
        Coordinates used to construct a Delaunay triangulation.
    
    Returns
    -------
    n_triangles : int
        Number of triangles.
    mean_quality : float
        Mean normalized triangle quality.
    minimum_quality : float
        Minimum normalized triangle quality.
    """
    def triangle_quality(p1, p2, p3):
        """Computes normalized triangle quality from inradius and circumradius.
        
        Parameters
        ----------
        p1 : numpy.ndarray
            First vertex.
        p2 : numpy.ndarray
            Second vertex.
        p3 : numpy.ndarray
            Third vertex.
        
        Returns
        -------
        quality : float
            Quality in the interval ``[0, 1]``.
        """
        def edge_length(a, b):
            """Computes the Euclidean distance between two coordinates.
            
            Parameters
            ----------
            a : numpy.ndarray
                First coordinate.
            b : numpy.ndarray
                Second coordinate.
            
            Returns
            -------
            length : float
                Euclidean distance between the coordinates.
            """
            return np.linalg.norm(a - b)
        
        a = edge_length(p2, p3)
        b = edge_length(p1, p3) 
        c = edge_length(p1, p2)
        
        if a == 0 or b == 0 or c == 0:
            return 0.0
        
        s = (a + b + c) / 2.0
        discriminant = s * (s - a) * (s - b) * (s - c)
        
        if discriminant <= 0:
            return 0.0
        
        area = np.sqrt(discriminant)
        if area < 1e-12:
            return 0.0
        
        inradius = area / s
        circumradius = (a * b * c) / (4.0 * area)
        return np.clip(2.0 * inradius / circumradius, 0.0, 1.0)

    # Compute Delaunay Triangulation.
    tri = Delaunay(points)
    triangles = tri.simplices
    
    # Calculate quality for all triangles.
    qualities = []
    for simplex in triangles:
        p1, p2, p3 = points[simplex]
        qualities.append(triangle_quality(p1, p2, p3))
    
    qualities = np.array(qualities)
    
    # Compute stats.
    ntriangles = len(qualities)
    
    if ntriangles == 0:
        return 0, 0.0, 0.0
        
    qmean = np.mean(qualities)
    qmin = np.min(qualities)
    
    return ntriangles, qmean, qmin


def _as_zx(points, coord_order="zx"):
    """Converts point coordinates to canonical ``(z, x)`` column order.
    
    Parameters
    ----------
    points : array-like of shape (n_points, 2)
        Input coordinates.
    coord_order : {"zx", "xz"}, optional
        Column order used by the input.
    
    Returns
    -------
    points_zx : numpy.ndarray
        Coordinates ordered as ``(z, x)``.
    
    Raises
    ------
    ValueError
        If the point array is not two-dimensional with two columns or the order is unsupported.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (N,2)")
    if coord_order == "zx":
        return pts
    if coord_order == "xz":
        return pts[:, [1, 0]]
    raise ValueError("coord_order must be 'zx' or 'xz'")


def _wrap_sizing_fn(sizing_fn):
    """Wraps a sizing function so its result is returned as a NumPy array.
    
    Parameters
    ----------
    sizing_fn : callable
        Function evaluated as ``sizing_fn(z, x)``.
    
    Returns
    -------
    wrapped_function : callable
        Array-compatible sizing-function wrapper.
    """
    def s(z, x):
        """Evaluates the wrapped sizing function on NumPy coordinates.
        
        Parameters
        ----------
        z : array-like
            Vertical coordinate or coordinates.
        x : array-like
            Horizontal coordinate or coordinates.
        
        Returns
        -------
        sizes : numpy.ndarray
            Sizing values converted to floating-point NumPy form.
        """
        z = np.asarray(z)
        x = np.asarray(x)
        out = sizing_fn(z, x)
        return np.asarray(out, dtype=float)
    return s


def mesh_sizing_check(
    points,
    sizing_fn,
    coord_order="zx",
    return_all=True,
):
    """Evaluates mesh conformance to a target sizing function.
    
    Parameters
    ----------
    points : array-like of shape (n_points, 2)
        Mesh coordinates.
    sizing_fn : callable
        Target-size function evaluated as ``sizing_fn(z, x)``.
    coord_order : {"zx", "xz"}, optional
        Column order used by ``points``.
    return_all : bool, optional
        Whether to return triangle-level diagnostic arrays.
    
    Returns
    -------
    sizing_score : float
        One minus the mean oversizing deviation.
    results : dict, optional
        Triangle-level diagnostics returned when ``return_all`` is ``True``.
    
    Raises
    ------
    ValueError
        If the point array shape or coordinate order is invalid.
    """
    pts_zx = _as_zx(points, coord_order=coord_order)
    sfn = _wrap_sizing_fn(sizing_fn)
    
    # Construct the Delaunay triangulation.
    if len(pts_zx) < 3:
        if return_all:
             return 0.0, {}
        return 0.0

    tri = Delaunay(pts_zx)
    simplices = tri.simplices
    P = pts_zx[simplices]      # (M,3,2) in (z,x).
    
    # Compute triangle centroids.
    centroids = np.mean(P, axis=1)
    cg_z = centroids[:, 0]
    cg_x = centroids[:, 1]
    
    # Evaluate target sizes at triangle centroids.
    target_sizes = sfn(cg_z, cg_x)
    
    # Compute actual sizes from the mean edge length.
    v0 = P[:, 0, :]
    v1 = P[:, 1, :]
    v2 = P[:, 2, :]
    
    edge01 = np.linalg.norm(v1 - v0, axis=1)
    edge12 = np.linalg.norm(v2 - v1, axis=1)
    edge20 = np.linalg.norm(v0 - v2, axis=1)
    
    actual_sizes = (edge01 + edge12 + edge20) / 3.0
    
    # Avoid division by zero for targets.
    safe_target = np.where(target_sizes > 1e-12, target_sizes, 1e-12)
    
    # Calculate Ratio (Actual / Target).
    ratios = actual_sizes / safe_target
    
    # Only triangles that are larger than the target.
    mask = actual_sizes > safe_target      
    
    # Deviation only for those (since ratios > 1, abs(1-ratio) == ratio-1).
    deviations = ratios[mask] - 1.0
    
    # Mean over that subset.
    mean_deviation = deviations.mean() if deviations.size > 0 else 0.0
    global_score = 1.0 - mean_deviation
    
    if not return_all:
        return global_score

    results = {
        "tri": tri,
        "simplices": simplices,
        "points_zx": pts_zx,
        "centroids_zx": centroids,
        "target_sizes": target_sizes,
        "actual_sizes": actual_sizes,
        "element_ratios": ratios,
        "element_deviations": deviations,
        "mean_deviation": mean_deviation,
        "global_score": global_score
    }
    
    return global_score, results


