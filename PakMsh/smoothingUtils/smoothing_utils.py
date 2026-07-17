import numpy as np
import warnings
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from numba import njit, prange
try:
    import cupy
    CUPY_AVAILABLE = True

except ImportError:
    cupy = None
    CUPY_AVAILABLE = False

    warnings.warn(
        "CuPy is not installed. GPU smoothing is disabled, "
        "but all CPU PacMesh functions remain available.",
        RuntimeWarning,
        stacklevel=2,
    )



def cvt_smooth_cpu(
    points: np.ndarray,
    sizing_fn,          
    x1: float, x2: float,
    z1: float, z2: float,
    iterations: int = 5,
    influence: float = 1.0,
    hold_boundary: bool = True,
    boundary_points: np.ndarray = None,
    density_power: float = 4.0
) -> np.ndarray:

    """Applies density-weighted centroidal Voronoi smoothing to mesh points.
    
    Parameters
    ----------
    points : array-like of shape (n_points, 2)
        Input mesh coordinates.
    sizing_fn : callable
        Vectorized function returning target spacing at ``(x, z)``.
    x1 : float
        Minimum horizontal domain coordinate.
    x2 : float
        Maximum horizontal domain coordinate.
    z1 : float
        Minimum vertical domain coordinate.
    z2 : float
        Maximum vertical domain coordinate.
    iterations : int, optional
        Number of CVT iterations.
    influence : float, optional
        Fraction of each centroid displacement to apply.
    hold_boundary : bool, optional
        Whether points on the rectangular domain boundary remain fixed.
    boundary_points : array-like, optional
        Additional coordinates that remain fixed.
    density_power : float, optional
        Exponent in the density law ``1 / h**density_power``.
    
    Returns
    -------
    smoothed_points : numpy.ndarray
        Coordinates after density-weighted CVT relaxation.
    """
    pts = np.array(points, dtype=float, copy=True)
    pts = np.ascontiguousarray(points, dtype=float)
    n_points = pts.shape[0]

    # Identify fixed points.
    is_fixed = np.zeros(n_points, dtype=bool)

    if boundary_points is not None and len(boundary_points) > 0:
        tree = cKDTree(boundary_points)
        dists, _ = tree.query(pts)
        is_fixed |= dists < 1e-8

    if hold_boundary:
        on_wall = (
            (np.abs(pts[:, 0] - x1) < 1e-5) |
            (np.abs(pts[:, 0] - x2) < 1e-5) |
            (np.abs(pts[:, 1] - z1) < 1e-5) |
            (np.abs(pts[:, 1] - z2) < 1e-5)
        )
        is_fixed |= on_wall

    movable_mask = ~is_fixed

    # Pre-allocate accumulators.
    global_mass = np.zeros(n_points)
    global_moments = np.zeros((n_points, 2))

    for iteration in range(iterations):
        print(
            f"\rIteration {iteration + 1}/{iterations}",
            end="",
			flush=True,
        )
        # Construct the Delaunay triangulation.
        tri = Delaunay(pts)
        simplices = tri.simplices
        a = pts[simplices[:, 0]]
        b = pts[simplices[:, 1]]
        c = pts[simplices[:, 2]]

        # Compute triangle centroids.
        standard_centers = (a + b + c) / 3.0

        # Compute triangle circumcenters.
        # Cross-term denominator (twice area with sign).
        D = 2.0 * (
            a[:, 0] * (b[:, 1] - c[:, 1]) +
            b[:, 0] * (c[:, 1] - a[:, 1]) +
            c[:, 0] * (a[:, 1] - b[:, 1])
        )

        bad_tri_mask = np.abs(D) < 1e-12
        # Preserve the sign while avoiding a zero denominator.
        D_safe = D.copy()
        D_safe[bad_tri_mask] = np.sign(D_safe[bad_tri_mask]) * 1e-12
        D_safe[D_safe == 0.0] = 1e-12  # Replace an exactly zero denominator.

        a2 = a[:, 0]**2 + a[:, 1]**2
        b2 = b[:, 0]**2 + b[:, 1]**2
        c2 = c[:, 0]**2 + c[:, 1]**2

        Ux = (
            a2 * (b[:, 1] - c[:, 1]) +
            b2 * (c[:, 1] - a[:, 1]) +
            c2 * (a[:, 1] - b[:, 1])
        ) / D_safe

        Uz = (
            a2 * (c[:, 0] - b[:, 0]) +
            b2 * (a[:, 0] - c[:, 0]) +
            c2 * (b[:, 0] - a[:, 0])
        ) / D_safe

        circumcenters = np.column_stack((Ux, Uz))

        # Replace unstable circumcenters with triangle centroids.
        dist_cc_centroid = np.linalg.norm(circumcenters - standard_centers, axis=1)
        raw_areas = 0.5 * np.abs(D)
        approx_h = np.sqrt(np.maximum(raw_areas, 1e-16))
        unstable_mask = dist_cc_centroid > (1.5 * approx_h)

        cc = circumcenters.copy()
        replace_mask = bad_tri_mask | unstable_mask
        cc[replace_mask] = standard_centers[replace_mask]

        # Compute edge midpoints.
        m_ab = 0.5 * (a + b)
        m_bc = 0.5 * (b + c)
        m_ca = 0.5 * (c + a)

        # Reset the mass and moment accumulators.
        global_mass.fill(0.0)
        global_moments.fill(0.0)

        # Build the six subtriangles associated with each triangle.
        # For each main triangle, we have 6 sub-triangles:
        # A: (A, M_ab, CC) and (A, M_ca, CC).
        # B: (B, M_bc, CC) and (B, M_ab, CC).
        # C: (C, M_ca, CC) and (C, M_bc, CC).

        n_tri = simplices.shape[0]

        # Node indices to accumulate into.
        idx_all = np.concatenate([
            simplices[:, 0], simplices[:, 0],
            simplices[:, 1], simplices[:, 1],
            simplices[:, 2], simplices[:, 2],
        ])

        # Corner points P of each sub-triangle.
        p_all = np.vstack([
            a, a,
            b, b,
            c, c,
        ])

        # Edge midpoints M for each sub-triangle.
        m_all = np.vstack([
            m_ab, m_ca,
            m_bc, m_ab,
            m_ca, m_bc,
        ])

        # Circumcenters CC for each sub-triangle.
        cc_all = np.vstack([
            cc, cc, cc, cc, cc, cc
        ])

        # Compute subtriangle centroids and areas.
        sub_c = (p_all + m_all + cc_all) / 3.0

        u = m_all - p_all
        v = cc_all - p_all
        sub_area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])

        # Evaluate density at subtriangle centroids.
        h_vals = sizing_fn(sub_c[:, 0], sub_c[:, 1])   # Evaluate the sizing function in vectorized form.
        h_vals = np.maximum(h_vals, 1e-12)
        densities = 1.0 / (h_vals ** density_power)

        masses = sub_area * densities
        moments = masses[:, None] * sub_c

        # Accumulate all mass and moment contributions.
        np.add.at(global_mass, idx_all, masses)
        np.add.at(global_moments, idx_all, moments)

        # Update movable points.
        valid_mask = (global_mass > 1e-12) & movable_mask
        new_locs = global_moments[valid_mask] / global_mass[valid_mask, None]
        pts[valid_mask] += influence * (new_locs - pts[valid_mask])

        # Clamp to domain.
        pts[valid_mask, 0] = np.clip(pts[valid_mask, 0], x1, x2)
        pts[valid_mask, 1] = np.clip(pts[valid_mask, 1], z1, z2)
        triangulation = Delaunay(pts)

    return pts, triangulation.simplices




@njit(fastmath=True, cache=True)
def get_triangle_quality(p1, p2, p3):
    """Computes normalized triangle quality using inradius and circumradius.
    
    Parameters
    ----------
    p1 : array-like
        First vertex.
    p2 : array-like
        Second vertex.
    p3 : array-like
        Third vertex.
    
    Returns
    -------
    quality : float
        Quality in the interval ``[0, 1]``.
    """
    # Euclidean distances squared.
    d12_sq = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
    d13_sq = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2
    d23_sq = (p2[0]-p3[0])**2 + (p2[1]-p3[1])**2
    
    a = np.sqrt(d23_sq)
    b = np.sqrt(d13_sq)
    c = np.sqrt(d12_sq)

    if a == 0 or b == 0 or c == 0:
        return 0.0

    s = (a + b + c) * 0.5
    discriminant = s * (s - a) * (s - b) * (s - c)

    if discriminant <= 0.0:
        return 0.0
    
    area = np.sqrt(discriminant)
    
    if area < 1e-12:
        return 0.0
    
    val = (8.0 * discriminant) / (s * a * b * c)
    
    # Clip the scalar explicitly for Numba compatibility.
    if val <= 0.0:
        return 0.0
    elif val >= 1.0:
        return 1.0
    else:
        return val

@njit(fastmath=True)
def get_ring_quality(point_idx, p_coords, tri_offsets, tri_indices, simplices, all_points):
    """Computes the minimum quality of triangles incident to one point.
    
    Parameters
    ----------
    point_idx : int
        Index of the point being evaluated.
    p_coords : array-like
        Candidate coordinates for the selected point.
    tri_offsets : numpy.ndarray
        CSR offsets for the point-to-triangle map.
    tri_indices : numpy.ndarray
        CSR triangle indices.
    simplices : numpy.ndarray
        Triangle vertex indices.
    all_points : numpy.ndarray
        Coordinates of all mesh points.
    
    Returns
    -------
    minimum_quality : float
        Lowest incident-triangle quality.
    """
    min_q = 1.0
    
    start = tri_offsets[point_idx]
    end = tri_offsets[point_idx + 1]
    
    for k in range(start, end):
        tri_idx = tri_indices[k]
        
        # Get the 3 vertex indices of this triangle.
        idx0 = simplices[tri_idx, 0]
        idx1 = simplices[tri_idx, 1]
        idx2 = simplices[tri_idx, 2]
        
        # Determine coordinates (one is the candidate point, two are neighbors).
        # Evaluate the three vertex cases without temporary arrays.
        if idx0 == point_idx:
            q = get_triangle_quality(p_coords, all_points[idx1], all_points[idx2])
        elif idx1 == point_idx:
            q = get_triangle_quality(all_points[idx0], p_coords, all_points[idx2])
        else:
            q = get_triangle_quality(all_points[idx0], all_points[idx1], p_coords)

        if q < min_q:
            min_q = q
            if min_q == 0.0:
                return 0.0
                
    return min_q

@njit(fastmath=True)
def run_smart_smooth(points, simplices, 
                     neigh_offsets, neigh_indices, 
                     tri_offsets, tri_indices, 
                     boundary_mask, iterations, alpha):
    
    """Runs quality-preserving Laplacian smoothing on a fixed triangulation.
    
    Parameters
    ----------
    points : numpy.ndarray
        Input mesh coordinates.
    simplices : numpy.ndarray
        Triangle vertex indices.
    neigh_offsets : numpy.ndarray
        CSR offsets for point neighbors.
    neigh_indices : numpy.ndarray
        CSR point-neighbor indices.
    tri_offsets : numpy.ndarray
        CSR offsets for incident triangles.
    tri_indices : numpy.ndarray
        CSR incident-triangle indices.
    boundary_mask : numpy.ndarray of bool
        Mask selecting points that remain fixed.
    iterations : int
        Maximum number of smoothing iterations.
    alpha : float
        Laplacian displacement fraction.
    
    Returns
    -------
    smoothed_points : numpy.ndarray
        Coordinates after accepted quality-preserving moves.
    """
    n_points = len(points)
    current_points = points.copy()
    
    # Pre-allocate temporary array for checking.
    
    for _ in range(iterations):
        move_count = 0
        
        for i in range(n_points):
            if boundary_mask[i]:
                continue
            
            # Get Neighbors Range (CSR lookups).
            n_start = neigh_offsets[i]
            n_end = neigh_offsets[i+1]
            
            # Isolated point check.
            if n_start == n_end:
                continue
                
            # Calculate Laplacian Average.
            sum_x = 0.0
            sum_y = 0.0
            count = 0
            
            for k in range(n_start, n_end):
                n_idx = neigh_indices[k]
                sum_x += current_points[n_idx, 0]
                sum_y += current_points[n_idx, 1]
                count += 1
            
            avg_x = sum_x / count
            avg_y = sum_y / count
            
            # Candidate position.
            old_x = current_points[i, 0]
            old_y = current_points[i, 1]
            
            cand_x = (1 - alpha) * old_x + alpha * avg_x
            cand_y = (1 - alpha) * old_y + alpha * avg_y
            
            # Construct a temporary array for the candidate to reuse the geometry function.
            cand_pt = np.array([cand_x, cand_y], dtype=np.float64)
            curr_pt = current_points[i] # View.
            
            # Evaluate the candidate move.
            
            # Current Quality.
            curr_q = get_ring_quality(i, curr_pt, tri_offsets, tri_indices, simplices, current_points)
            
            # Candidate Quality.
            cand_q = get_ring_quality(i, cand_pt, tri_offsets, tri_indices, simplices, current_points)
            
            # Decision Rule.
            if cand_q >= curr_q - 1e-8:
                current_points[i, 0] = cand_x
                current_points[i, 1] = cand_y
                move_count += 1
        
        if move_count == 0:
            break
            
    return current_points

@njit
def build_point_to_simplex_map(n_points, simplices):
    """Builds a CSR point-to-simplex incidence map.
    
    Parameters
    ----------
    n_points : int
        Number of mesh points.
    simplices : numpy.ndarray
        Triangle vertex indices.
    
    Returns
    -------
    offsets : numpy.ndarray
        CSR offsets for each point.
    indices : numpy.ndarray
        Flattened incident-simplex indices.
    """
    # Pass 1: Count degrees.
    counts = np.zeros(n_points, dtype=np.int32)
    for i in range(len(simplices)):
        for j in range(3):
            pt = simplices[i, j]
            counts[pt] += 1
            
    # Pass 2: Build Offsets.
    offsets = np.zeros(n_points + 1, dtype=np.int32)
    current_offset = 0
    for i in range(n_points):
        offsets[i] = current_offset
        current_offset += counts[i]
    offsets[n_points] = current_offset
    
    # Pass 3: Fill Indices.
    indices = np.zeros(current_offset, dtype=np.int32)
    current_pos = offsets.copy()
    
    for i in range(len(simplices)):
        for j in range(3):
            pt = simplices[i, j]
            pos = current_pos[pt]
            indices[pos] = i
            current_pos[pt] += 1
            
    return offsets, indices

def smart_laplacian_smooth_numba(points, iterations=10, alpha=0.5, boundary_points=None):
    # Triangulate.
    """Applies Numba-accelerated quality-preserving Laplacian smoothing.
    
    Parameters
    ----------
    points : numpy.ndarray
        Input mesh coordinates.
    iterations : int, optional
        Maximum number of smoothing iterations.
    alpha : float, optional
        Laplacian displacement fraction.
    boundary_points : array-like, optional
        Fixed point indices or leading boundary coordinates.
    
    Returns
    -------
    smoothed_points : numpy.ndarray
        Smoothed mesh coordinates.
    simplices : numpy.ndarray
        Triangle vertex indices from the initial triangulation.
    """
    tri = Delaunay(points)
    
    # Build Neighbor Graph.
    neigh_offsets, neigh_indices = tri.vertex_neighbor_vertices
    
    # Build Point-to-Simplex Graph.
    tri_offsets, tri_indices = build_point_to_simplex_map(len(points), tri.simplices)
    
    # Handle Boundary Mask.
    n_points = len(points)
    boundary_mask = np.zeros(n_points, dtype=bool)
    
    if boundary_points is not None and len(boundary_points) > 0:
        bp = np.asarray(boundary_points)
        
        # Handle boundary_points are Indices (1D array of integers).
        if bp.ndim == 1 and np.issubdtype(bp.dtype, np.integer):
            boundary_mask[bp] = True
            
        # Handle boundary_points are Coordinates (2D array of floats).
        elif bp.ndim == 2:
            num_boundary = min(len(bp), n_points)
            boundary_mask[:num_boundary] = True

        # Handle Fallback for lists/mixed types.
        else:
            num_boundary = min(len(bp), n_points)
            boundary_mask[:num_boundary] = True

    # Run Numba Loop.
    points_f64 = points.astype(np.float64)
    
    new_points = run_smart_smooth(
        points_f64, 
        tri.simplices,
        neigh_offsets, neigh_indices,
        tri_offsets, tri_indices,
        boundary_mask,
        iterations,
        alpha
    )
    
    return new_points, tri.simplices

def physical_smooth(points, density_fn, h0=100.0, iterations=30, dt=0.2, subdomain=None, ellipse_points=None, boundary_points=None):
    """Applies iterative Physical smoothing.
    
    Parameters
    ----------
    points : numpy.ndarray
        Input mesh coordinates.
    density_fn : callable
        Function returning target edge length at ``(x, z)``.
    h0 : float, optional
        Reference edge-length argument retained by the interface.
    iterations : int, optional
        Number of spring-relaxation iterations.
    dt : float, optional
        Explicit displacement step.
    subdomain : object, optional
        Subdomain argument reserved by the interface.
    ellipse_points : array-like, optional
        Ellipse-boundary argument reserved by the interface.
    boundary_points : array-like, optional
        Leading point coordinates that remain fixed.
    
    Returns
    -------
    smoothed_points : numpy.ndarray
        Coordinates after spring relaxation.
    simplices : numpy.ndarray
        Triangle vertex indices after smoothing.
    """
    points = points.copy()
    n_points = len(points)
    
    # Initialize boundary mask - all points are movable by default.
    fixed_mask = np.zeros(n_points, dtype=bool)
    
    # Only add user-defined fixed boundary points.
    if boundary_points is not None and len(boundary_points) > 0:
        # The boundary points are just the first len(boundary_points) indices.
        num_boundary = min(len(boundary_points), n_points)
        boundary_indices = np.arange(num_boundary)
        fixed_mask[boundary_indices] = True
        
    for it in range(iterations):
        print(
            f"\rIteration {it + 1}/{iterations}",
            end="",
			flush=True,
        )
        tri = Delaunay(points)
        
        # Vectorized edge extraction.
        simplices = tri.simplices
        edges_raw = np.vstack([
            simplices[:, [0, 1]],
            simplices[:, [1, 2]], 
            simplices[:, [2, 0]]
        ])
        
        # Sort each edge to ensure consistent ordering.
        edges_sorted = np.sort(edges_raw, axis=1)
        
        # Get unique edges.
        edges = np.unique(edges_sorted, axis=0)
        
        # Vectorized calculations.
        p1_indices = edges[:, 0]
        p2_indices = edges[:, 1]
        p1 = points[p1_indices]
        p2 = points[p2_indices]
        
        # Calculate midpoints and evaluate density function in batch.
        midpoints = (p1 + p2) * 0.5
        h_values = density_fn(midpoints[:, 0], midpoints[:, 1]) 
    
        # Vectorized force calculation.
        edge_vectors = p1 - p2
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        
        # Avoid division by zero.
        nonzero_mask = edge_lengths > 1e-12
        edge_vectors = edge_vectors[nonzero_mask]
        edge_lengths = edge_lengths[nonzero_mask]
        h_values = h_values[nonzero_mask]
        p1_indices = p1_indices[nonzero_mask]
        p2_indices = p2_indices[nonzero_mask]
        
        # Calculate forces.
        force_magnitudes = (edge_lengths - h_values) / edge_lengths
        forces = edge_vectors * force_magnitudes[:, np.newaxis]
        
        # Accumulate forces.
        moved = np.zeros_like(points)
        
        # Add forces to p1 nodes (negative direction).
        valid_p1 = ~fixed_mask[p1_indices]
        if np.any(valid_p1):
            np.add.at(moved, p1_indices[valid_p1], -forces[valid_p1] * 0.5)
        
        # Add forces to p2 nodes (positive direction).
        valid_p2 = ~fixed_mask[p2_indices]
        if np.any(valid_p2):
            np.add.at(moved, p2_indices[valid_p2], forces[valid_p2] * 0.5)
        
        # Apply movement only to non-fixed points.
        points[~fixed_mask] += dt * moved[~fixed_mask]
    
    tri = Delaunay(points)
    return points, tri.simplices

if CUPY_AVAILABLE:
	def _compute_weights_flat_balanced(sizing_fn, x1, x2, z1, z2, Nx, Nz, density_power=4.0, chunk_z=256, dtype=cupy.float32):
	    """Precomputes a flattened GPU density grid in vertical chunks.
	    
	    Parameters
	    ----------
	    sizing_fn : callable
	        Function returning target spacing at ``(x, z)``.
	    x1 : float
	        Minimum horizontal domain coordinate.
	    x2 : float
	        Maximum horizontal domain coordinate.
	    z1 : float
	        Minimum vertical domain coordinate.
	    z2 : float
	        Maximum vertical domain coordinate.
	    Nx : int
	        Number of horizontal grid cells.
	    Nz : int
	        Number of vertical grid cells.
	    density_power : float, optional
	        Exponent in the density law ``1 / h**density_power``.
	    chunk_z : int, optional
	        Number of vertical rows processed per GPU chunk.
	    dtype : data-type, optional
	        CuPy data type used for grid storage.
	    
	    Returns
	    -------
	    weights : cupy.ndarray
	        Flattened density weights.
	    dx : float
	        Horizontal grid spacing.
	    dz : float
	        Vertical grid spacing.
	    """
	    dx = (x2 - x1) / Nx
	    dz = (z2 - z1) / Nz
	    
	    x_lin = cupy.linspace(x1 + dx/2, x2 - dx/2, Nx, dtype=dtype)
	    weights_parts = []
	    
	    # Process in chunks to save GPU memory.
	    for z_start in range(0, Nz, chunk_z):
	        z_end = min(z_start + chunk_z, Nz)
	        nz_chunk = z_end - z_start
	        
	        z_lin_chunk = cupy.linspace(z1 + dz/2 + z_start*dz, 
	                                  z1 + dz/2 + (z_end-1)*dz, 
	                                  nz_chunk, dtype=dtype)
	        
	        X_chunk, Z_chunk = cupy.meshgrid(x_lin, z_lin_chunk)
	        
	        # Evaluate user sizing function.
	        h_val = _eval_sizing(sizing_fn, X_chunk.ravel(), Z_chunk.ravel(), 
	                             out_shape=X_chunk.shape, dtype=dtype)
	        
	        w_chunk = cupy.power(h_val, -density_power)
	        weights_parts.append(w_chunk)
	    
	    weights = cupy.vstack(weights_parts)
	    return weights.ravel(), dx, dz
	
	def _eval_sizing(sizing_fn, x, z, out_shape, dtype=cupy.float32):
	    """Evaluates a CPU sizing function and returns a CuPy array.
	    
	    Parameters
	    ----------
	    sizing_fn : callable
	        Sizing function accepting NumPy coordinates.
	    x : cupy.ndarray
	        Horizontal coordinates.
	    z : cupy.ndarray
	        Vertical coordinates.
	    out_shape : tuple of int
	        Required output shape.
	    dtype : data-type, optional
	        CuPy output data type.
	    
	    Returns
	    -------
	    sizes : cupy.ndarray
	        Sizing values reshaped to ``out_shape``.
	    """
	    x_np = cupy.asnumpy(x)
	    z_np = cupy.asnumpy(z)
	    res = sizing_fn(x_np, z_np)
	    if np.isscalar(res):
	        return cupy.full(out_shape, res, dtype=dtype)
	    return cupy.asarray(res, dtype=dtype).reshape(out_shape)
	
	# Define the CUDA kernels.
	
	# Count point references in each spatial bin.
	bin_counts_kernel_balanced = cupy.RawKernel(r'''
	extern "C" __global__
	void bin_counts_kernel_balanced(
	    const float* __restrict__ pts,
	    const float* __restrict__ radii,
	    int* __restrict__ counts,
	    int n_points,
	    int Nx, int Nz,
	    int BIN, int nbx, int nbz,
	    float x1, float z1, float inv_dx, float inv_dz
	) {
	    int i = blockIdx.x * blockDim.x + threadIdx.x;
	    if (i >= n_points) return;
	
	    float px = pts[i * 2 + 0];
	    float pz = pts[i * 2 + 1];
	    float r  = radii[i];
	
	    int min_ix = max(0, (int)((px - r - x1) * inv_dx));
	    int max_ix = min(Nx - 1, (int)((px + r - x1) * inv_dx));
	    int min_iz = max(0, (int)((pz - r - z1) * inv_dz));
	    int max_iz = min(Nz - 1, (int)((pz + r - z1) * inv_dz));
	
	    int min_bx = min_ix / BIN;
	    int max_bx = max_ix / BIN;
	    int min_bz = min_iz / BIN;
	    int max_bz = max_iz / BIN;
	
	    for (int bz = min_bz; bz <= max_bz; bz++) {
	        for (int bx = min_bx; bx <= max_bx; bx++) {
	            atomicAdd(&counts[bz * nbx + bx], 1);
	        }
	    }
	}
	''', 'bin_counts_kernel_balanced')
	
	# Fill the spatial-bin index array.
	bin_fill_kernel_balanced = cupy.RawKernel(r'''
	extern "C" __global__
	void bin_fill_kernel_balanced(
	    const float* __restrict__ pts,
	    const float* __restrict__ radii,
	    int* __restrict__ write_ptrs,
	    int* __restrict__ indices,
	    int n_points,
	    int Nx, int Nz,
	    int BIN, int nbx, int nbz,
	    float x1, float z1, float inv_dx, float inv_dz
	) {
	    int i = blockIdx.x * blockDim.x + threadIdx.x;
	    if (i >= n_points) return;
	
	    float px = pts[i * 2 + 0];
	    float pz = pts[i * 2 + 1];
	    float r  = radii[i];
	
	    int min_ix = max(0, (int)((px - r - x1) * inv_dx));
	    int max_ix = min(Nx - 1, (int)((px + r - x1) * inv_dx));
	    int min_iz = max(0, (int)((pz - r - z1) * inv_dz));
	    int max_iz = min(Nz - 1, (int)((pz + r - z1) * inv_dz));
	
	    int min_bx = min_ix / BIN;
	    int max_bx = max_ix / BIN;
	    int min_bz = min_iz / BIN;
	    int max_bz = max_iz / BIN;
	
	    for (int bz = min_bz; bz <= max_bz; bz++) {
	        for (int bx = min_bx; bx <= max_bx; bx++) {
	            int pos = atomicAdd(&write_ptrs[bz * nbx + bx], 1);
	            indices[pos] = i;
	        }
	    }
	}
	''', 'bin_fill_kernel_balanced')
	
	# Generate the Voronoi identifier map.
	generate_id_map_kernel_balanced = cupy.RawKernel(r'''
	extern "C" __global__
	void generate_id_map_balanced_f(
	    const float* __restrict__ pts,
	    const int* __restrict__ offsets,
	    const int* __restrict__ indices,
	    int* __restrict__ out_ids,
	    int Nx, int Nz, int BIN, int nbx,
	    float x1, float z1, float dx, float dz
	) {
	    int ix = blockIdx.x * blockDim.x + threadIdx.x;
	    int iz = blockIdx.y * blockDim.y + threadIdx.y;
	
	    if (ix >= Nx || iz >= Nz) return;
	
	    float px = x1 + (ix + 0.5f) * dx;
	    float pz = z1 + (iz + 0.5f) * dz;
	
	    int bx = ix / BIN;
	    int bz = iz / BIN;
	    int bin_idx = bz * nbx + bx;
	
	    int start = offsets[bin_idx];
	    int end   = offsets[bin_idx + 1];
	
	    int best_id = -1;
	    float best_d2 = 1e30f;
	
	    for (int k = start; k < end; k++) {
	        int pid = indices[k];
	        float dx_p = px - pts[pid * 2 + 0];
	        float dz_p = pz - pts[pid * 2 + 1];
	        float d2 = dx_p*dx_p + dz_p*dz_p;
	
	        // Resolve equal distances.
	        // If distances are identical, strictly prefer the smaller PID.
	        if (d2 < best_d2 || (d2 == best_d2 && pid < best_id)) {
	            best_d2 = d2;
	            best_id = pid;
	        }
	    }
	    // Store a one-based identifier.
	    out_ids[iz * Nx + ix] = best_id + 1;
	}
	''', 'generate_id_map_balanced_f', options=('--use_fast_math',))
	
	# Integrate subtriangles with float64 atomics.
	integrate_triangles_kernel = cupy.RawKernel(r'''
	extern "C" {
	
	// Sample the precomputed density grid.
	__device__ float sample_density(
	    float x, float z, 
	    const float* density_grid, 
	    int Nx, int Nz, 
	    float x1, float z1, float inv_dx, float inv_dz
	) {
	    int ix = (int)((x - x1) * inv_dx);
	    int iz = (int)((z - z1) * inv_dz);
	    ix = max(0, min(Nx - 1, ix));
	    iz = max(0, min(Nz - 1, iz));
	    return density_grid[iz * Nx + ix];
	}
	
	// Accumulate the contribution of one subtriangle.
	// Use double-precision mass and moment buffers to reduce atomic noise.
	__device__ void accum_sub(
	    int pid, 
	    float px, float pz, float mx, float mz, float ccx, float ccz,
	    double* global_mass,      
	    double* global_moments_x,
	    double* global_moments_z,
	    const float* density_grid,
	    int Nx, int Nz, float x1, float z1, float inv_dx, float inv_dz
	) {
	    // Centroid of the sub-triangle.
	    float cx = (px + mx + ccx) * 0.3333333f;
	    float cz = (pz + mz + ccz) * 0.3333333f;
	
	    // Area of sub-triangle.
	    float ux = mx - px; float uz = mz - pz;
	    float vx = ccx - px; float vz = ccz - pz;
	    float area = 0.5f * fabsf(ux * vz - uz * vx);
	
	    // Density at centroid.
	    float rho = sample_density(cx, cz, density_grid, Nx, Nz, x1, z1, inv_dx, inv_dz);
	
	    // Weighted Mass and Moments.
	    double m = (double)(area * rho);
	    
	    // Accumulate moments with double-precision atomics.
	    atomicAdd(&global_mass[pid], m);
	    atomicAdd(&global_moments_x[pid], m * (double)cx);
	    atomicAdd(&global_moments_z[pid], m * (double)cz);
	}
	
	__global__
	void integrate_triangles(
	    const int* __restrict__ id_map,
	    const float* __restrict__ pts,
	    const float* __restrict__ density_grid,
	    double* __restrict__ global_mass,       // Store double-precision sums.
	    double* __restrict__ global_moments_x,  // Store double-precision sums.
	    double* __restrict__ global_moments_z,  // Store double-precision sums.
	    int Nx, int Nz,
	    float x1, float z1, float inv_dx, float inv_dz
	) {
	    int ix = blockIdx.x * blockDim.x + threadIdx.x;
	    int iz = blockIdx.y * blockDim.y + threadIdx.y;
	
	    if (ix >= Nx - 1 || iz >= Nz - 1) return;
	
	    // Get 4 neighbors.
	    int ids[4];
	    ids[0] = id_map[iz * Nx + ix];
	    ids[1] = id_map[iz * Nx + (ix + 1)];
	    ids[2] = id_map[(iz + 1) * Nx + ix];
	    ids[3] = id_map[(iz + 1) * Nx + (ix + 1)];
	
	    // Find unique IDs.
	    int unique[4];
	    int count = 0;
	    for (int i = 0; i < 4; i++) {
	        int val = ids[i];
	        if (val <= 0) continue; 
	        bool seen = false;
	        for (int j = 0; j < count; j++) { if (unique[j] == val) seen = true; }
	        if (!seen) unique[count++] = val;
	    }
	
	    if (count < 3) return;
	
	    // Lambda macro to process one triangle (p0, p1, p2).
	    #define PROCESS_TRIANGLE(id0, id1, id2) { \
	        float ax = pts[(id0-1)*2+0]; float az = pts[(id0-1)*2+1]; \
	        float bx = pts[(id1-1)*2+0]; float bz = pts[(id1-1)*2+1]; \
	        float cx = pts[(id2-1)*2+0]; float cz = pts[(id2-1)*2+1]; \
	        \
	        float sc_x = (ax + bx + cx) * 0.333333f; \
	        float sc_z = (az + bz + cz) * 0.333333f; \
	        \
	        float D = 2.0f * (ax * (bz - cz) + bx * (cz - az) + cx * (az - bz)); \
	        float D_safe = D; \
	        if (fabsf(D) < 1e-12f) D_safe = copysignf(1e-12f, D); \
	        \
	        float a2 = ax*ax + az*az; \
	        float b2 = bx*bx + bz*bz; \
	        float c2 = cx*cx + cz*cz; \
	        \
	        float Ux = (a2 * (bz - cz) + b2 * (cz - az) + c2 * (az - bz)) / D_safe; \
	        float Uz = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D_safe; \
	        \
	        float area_raw = 0.5f * fabsf(D); \
	        float approx_h = sqrtf(fmaxf(area_raw, 1e-16f)); \
	        float dist_sq = (Ux - sc_x)*(Ux - sc_x) + (Uz - sc_z)*(Uz - sc_z); \
	        \
	        float cc_x = Ux; float cc_z = Uz; \
	        if (fabsf(D) < 1e-12f || dist_sq > (2.25f * approx_h * approx_h)) { \
	            cc_x = sc_x; cc_z = sc_z; \
	        } \
	        \
	        float mab_x = (ax + bx)*0.5f; float mab_z = (az + bz)*0.5f; \
	        float mbc_x = (bx + cx)*0.5f; float mbc_z = (bz + cz)*0.5f; \
	        float mca_x = (cx + ax)*0.5f; float mca_z = (cz + az)*0.5f; \
	        \
	        accum_sub(id0-1, ax, az, mab_x, mab_z, cc_x, cc_z, global_mass, global_moments_x, global_moments_z, density_grid, Nx, Nz, x1, z1, inv_dx, inv_dz); \
	        accum_sub(id0-1, ax, az, mca_x, mca_z, cc_x, cc_z, global_mass, global_moments_x, global_moments_z, density_grid, Nx, Nz, x1, z1, inv_dx, inv_dz); \
	        \
	        accum_sub(id1-1, bx, bz, mbc_x, mbc_z, cc_x, cc_z, global_mass, global_moments_x, global_moments_z, density_grid, Nx, Nz, x1, z1, inv_dx, inv_dz); \
	        accum_sub(id1-1, bx, bz, mab_x, mab_z, cc_x, cc_z, global_mass, global_moments_x, global_moments_z, density_grid, Nx, Nz, x1, z1, inv_dx, inv_dz); \
	        \
	        accum_sub(id2-1, cx, cz, mca_x, mca_z, cc_x, cc_z, global_mass, global_moments_x, global_moments_z, density_grid, Nx, Nz, x1, z1, inv_dx, inv_dz); \
	        accum_sub(id2-1, cx, cz, mbc_x, mbc_z, cc_x, cc_z, global_mass, global_moments_x, global_moments_z, density_grid, Nx, Nz, x1, z1, inv_dx, inv_dz); \
	    }
	
	    if (count == 3) {
	        PROCESS_TRIANGLE(unique[0], unique[1], unique[2]);
	    } else if (count == 4) {
	        PROCESS_TRIANGLE(unique[0], unique[1], unique[2]);
	        PROCESS_TRIANGLE(unique[0], unique[2], unique[3]);
	    }
	}
	}
	''', 'integrate_triangles', options=('--use_fast_math',))


def cvt_smooth_gpu(
    points,
    sizing_fn,
    x1, x2, z1, z2,
    N,
    iterations=5,
    influence=1.0,
    hold_boundary=True,
    boundary_points=None,
    density_power=4.0,
    b_size_mult=1.5,
    BIN=16,
    weight_chunk_z=256,
    dtype=np.float32,
):
    # Initialize the GPU CVT data.
    """Applies GPU-accelerated discrete weighted CVT smoothing.
    
    Parameters
    ----------
    points : array-like of shape (n_points, 2)
        Input mesh coordinates.
    sizing_fn : callable
        Function returning target spacing at ``(x, z)``.
    x1 : float
        Minimum horizontal domain coordinate.
    x2 : float
        Maximum horizontal domain coordinate.
    z1 : float
        Minimum vertical domain coordinate.
    z2 : float
        Maximum vertical domain coordinate.
    N : int
        Number of horizontal integration cells.
    iterations : int, optional
        Number of CVT iterations.
    influence : float, optional
        Base displacement fraction.
    hold_boundary : bool, optional
        Whether points on the rectangular domain boundary remain fixed.
    boundary_points : array-like, optional
        Additional coordinates that remain fixed.
    density_power : float, optional
        Exponent in the density law.
    b_size_mult : float, optional
        Multiplier used to size spatial-hash influence radii.
    BIN : int, optional
        GPU spatial-bin width in grid cells.
    weight_chunk_z : int, optional
        Vertical chunk size for density-grid evaluation.
    dtype : data-type, optional
        NumPy data type of the returned coordinates.
    
    Returns
    -------
    smoothed_points : numpy.ndarray
        GPU-smoothed coordinates copied back to host memory.
    """
    pts = cupy.asarray(points, dtype=cupy.float32)
    n_points = int(pts.shape[0])
    x1f, x2f, z1f, z2f = map(float, (x1, x2, z1, z2))
    
    Nx = int(N)
    Nz = int(abs(Nx * (z2f - z1f) / (x2f - x1f)))
    Nx, Nz = max(1, Nx), max(1, Nz)

    dx = (x2f - x1f) / Nx
    dz = (z2f - z1f) / Nz
    inv_dx = 1.0 / dx
    inv_dz = 1.0 / dz

    # Precompute the density grid.
    weights, _, _ = _compute_weights_flat_balanced(
        sizing_fn, x1f, x2f, z1f, z2f,
        Nx=Nx, Nz=Nz, density_power=density_power,
        chunk_z=weight_chunk_z, dtype=cupy.float32,
    )
    
    # Build the fixed-point mask.
    is_fixed = cupy.zeros((n_points,), dtype=cupy.bool_)

    if boundary_points is not None and len(boundary_points) > 0:
        b = cupy.asarray(boundary_points, dtype=cupy.float32)
        min_d2 = cupy.full((n_points,), cupy.float32(1e30), dtype=cupy.float32)
        chunk = 4096
        for j0 in range(0, int(b.shape[0]), chunk):
            bj = b[j0:j0 + chunk]
            diff = pts[:, None, :] - bj[None, :, :]
            d2 = cupy.sum(diff * diff, axis=2)
            min_d2 = cupy.minimum(min_d2, cupy.min(d2, axis=1))
        is_fixed |= (min_d2 < cupy.float32(1e-16))

    if hold_boundary:
        eps = cupy.float32(1e-5)
        on_wall = (
            (cupy.abs(pts[:, 0] - cupy.float32(x1f)) < eps) | (cupy.abs(pts[:, 0] - cupy.float32(x2f)) < eps) |
            (cupy.abs(pts[:, 1] - cupy.float32(z1f)) < eps) | (cupy.abs(pts[:, 1] - cupy.float32(z2f)) < eps)
        )
        is_fixed |= on_wall

    movable = ~is_fixed

    # Configure CUDA launch dimensions.
    BIN = int(BIN)
    nbx = (Nx + BIN - 1) // BIN
    nbz = (Nz + BIN - 1) // BIN
    n_bins = nbx * nbz

    threads = 256
    blocks_pts = (n_points + threads - 1) // threads
    block_pix = (32, 8)
    grid_pix = ((Nx + block_pix[0] - 1) // block_pix[0], 
                (Nz + block_pix[1] - 1) // block_pix[1])
    
    # Buffers.
    out_ids = cupy.empty((Nx * Nz,), dtype=cupy.int32)
    counts = cupy.zeros((n_bins,), dtype=cupy.int32)
    offsets = cupy.zeros((n_bins + 1,), dtype=cupy.int32)
    indices = cupy.empty((n_points * 9,), dtype=cupy.int32)

    # Use float64 accumulators for summation.
    global_mass = cupy.zeros((n_points,), dtype=cupy.float64)
    global_moments_x = cupy.zeros((n_points,), dtype=cupy.float64)
    global_moments_z = cupy.zeros((n_points,), dtype=cupy.float64)

    influence_f = cupy.float32(influence)
    bmult_f = cupy.float32(b_size_mult)

    # Iterate the GPU CVT update.
    for it in range(int(iterations)):
        h_g = _eval_sizing(sizing_fn, pts[:, 0], pts[:, 1], out_shape=(n_points,), dtype=cupy.float32)
        h_g = cupy.maximum(h_g, cupy.float32(1e-12))
        radii = h_g * bmult_f

        # Build the spatial hash.
        counts.fill(0)
        bin_counts_kernel_balanced((blocks_pts,), (threads,),
                           (pts, radii, counts, n_points, Nx, Nz, BIN, nbx, nbz,
                            cupy.float32(x1f), cupy.float32(z1f), cupy.float32(inv_dx), cupy.float32(inv_dz)))

        cupy.cumsum(counts, out=offsets[1:])
        total_refs = int(offsets[-1].get())
        if total_refs == 0: continue
        if total_refs > indices.size:
            indices = cupy.empty((int(total_refs * 1.5),), dtype=cupy.int32)

        write_ptrs = offsets[:-1].copy()
        bin_fill_kernel_balanced((blocks_pts,), (threads,),
                        (pts, radii, write_ptrs, indices, n_points, Nx, Nz, BIN, nbx, nbz,
                         cupy.float32(x1f), cupy.float32(z1f), cupy.float32(inv_dx), cupy.float32(inv_dz)))

        # Generate the Voronoi grid.
        generate_id_map_kernel_balanced(grid_pix, block_pix,
                               (pts, offsets, indices, out_ids,
                                Nx, Nz, BIN, nbx,
                                cupy.float32(x1f), cupy.float32(z1f), cupy.float32(dx), cupy.float32(dz)))

        # Integrate masses and moments in float64.
        # Reset the mass and moment accumulators.
        global_mass.fill(0.0)
        global_moments_x.fill(0.0)
        global_moments_z.fill(0.0)

        # Run Integration Kernel.
        integrate_triangles_kernel(grid_pix, block_pix,
            (out_ids, pts, weights, 
             global_mass, global_moments_x, global_moments_z, 
             Nx, Nz, cupy.float32(x1f), cupy.float32(z1f), cupy.float32(inv_dx), cupy.float32(inv_dz)))

        # Update movable point positions.
        # Threshold higher due to double precision.
        valid = (global_mass > 1e-18)
        upd = movable & valid
        
        if cupy.any(upd):
            curr_x = pts[upd, 0]
            curr_z = pts[upd, 1]
            
            # Cast back to float32 for final position update.
            targ_x = (global_moments_x[upd] / global_mass[upd]).astype(cupy.float32)
            targ_z = (global_moments_z[upd] / global_mass[upd]).astype(cupy.float32)
            
            damp = influence_f * (1.0 + (iterations - it)/iterations)
            
            pts[upd, 0] = curr_x + damp * (targ_x - curr_x)
            pts[upd, 1] = curr_z + damp * (targ_z - curr_z)

            pts[upd, 0] = cupy.clip(pts[upd, 0], cupy.float32(x1f), cupy.float32(x2f))
            pts[upd, 1] = cupy.clip(pts[upd, 1], cupy.float32(z1f), cupy.float32(z2f))

    out = cupy.asnumpy(pts)
    triangulation = Delaunay(out)
    return out.astype(dtype, copy=False), triangulation.simplices