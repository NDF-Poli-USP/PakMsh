import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from typing import Callable, Tuple, List


def place_circles_horizontal_boundary(points, x_min, x_max, z_fixed, density_function, N):
	"""Places tangent circles along a horizontal boundary.
	
	Parameters
	----------
	points : list of array-like
	    Existing point coordinates to retain.
	x_min : float
	    Left boundary coordinate.
	x_max : float
	    Right boundary coordinate.
	z_fixed : float
	    Constant vertical coordinate of the boundary.
	density_function : callable
	    Function returning the local target circle diameter at ``(x, z)``.
	N : int
	    Boundary sampling argument retained by the public interface.
	
	Returns
	-------
	boundary_points : list
	    Existing points followed by the generated horizontal-boundary points.
	"""
	# Remove old corner points if present (with tolerance for floating point precision).
	tolerance = 1e-6
	points = [pt for pt in points if not ((abs(pt[0] - x_min) < tolerance and abs(pt[1] - z_fixed) < tolerance) or
										  (abs(pt[0] - x_max) < tolerance and abs(pt[1] - z_fixed) < tolerance))]
	
	# Start with corner circles.
	centers = [(x_min, z_fixed), (x_max, z_fixed)]
	
	# Place circles greedily from left to right.
	current_x = x_min
	
	while True:
		# Get radius of current circle.
		r_current = density_function(current_x, z_fixed)/2
		
		# Calculate next position (circles just touching).
		# Move right by 2 * current radius to find center of next circle.
		next_x = current_x + 2 * r_current
		
		# Get radius of the next circle at that position.
		r_next = density_function(next_x, z_fixed)/2
		
		# Adjust position so circles actually touch (distance between centers = sum of radii).
		actual_next_x = current_x + (r_current + r_next)
		
		# Check if this circle would collide with right corner circle.
		r_right = density_function(x_max, z_fixed)/2
		distance_to_right = x_max - actual_next_x
		
		# Place the circle.
		centers.append((actual_next_x, z_fixed))
		current_x = actual_next_x
		
		if distance_to_right < r_next + r_right + tolerance:
			# Would collide with right circle, stop.
			break
	
	# Calculate how far the last circle should move left to avoid collision.
	last_circle_x = centers[-1][0]  # X-coordinate of the last placed circle.
	r_last = density_function(last_circle_x, z_fixed)/2
	r_right = density_function(x_max, z_fixed)/2
	
	# Position where the last circle should be to just touch the right circle.
	corrected_x = x_max - r_last - r_right
	
	# Distance to move left.
	move_left_distance = last_circle_x - corrected_x
	
	# Calculate percentage reduction needed.
	# Exclude: left circle (index 0), right circle (index 1), and first circle after left (index 2).
	# So we modify circles from index 3 onwards.
	total_diameter_to_reduce = 0
	
	for i in range(3, len(centers)):  # Start from index 3 (second circle after left).
		circle_x = centers[i][0]
		r_circle = density_function(circle_x, z_fixed)/2
		total_diameter_to_reduce += 2 * r_circle  # Each radius reduction saves 2*r in diameter.
	
	if total_diameter_to_reduce > 0:
		radius_reduction_percentage = move_left_distance / total_diameter_to_reduce
	else:
		radius_reduction_percentage = 0
	
	# Clear centers and redo the placement with reduced radii.
	centers = [(x_min, z_fixed), (x_max, z_fixed)]
	current_x = x_min
	circle_index = 0  # Track which circle we're placing.
	
	while True:
		# Get radius of current circle.
		r_current = density_function(current_x, z_fixed)/2
		# Apply reduction if this is not left, right, or first after left.
		if circle_index >= 2:  # Circles from index 2 onwards (first after left gets reduction).
			r_current = r_current * (1 - radius_reduction_percentage)
		
		# Calculate next position.
		next_x = current_x + 2 * r_current
		
		# Get radius of the next circle at that position.
		r_next = density_function(next_x, z_fixed)/2
		# Apply reduction if the next circle is not left, right, or first after left.
		if circle_index + 1 >= 2:  # Next circle will be at index circle_index + 1.
			r_next = r_next * (1 - radius_reduction_percentage)
		
		# Adjust position so circles actually touch.
		actual_next_x = current_x + (r_current + r_next)
		
		# Check if this circle would collide with right corner circle.
		r_right = density_function(x_max, z_fixed)/2  # Right circle radius not reduced.
		distance_to_right = x_max - actual_next_x
		
		if distance_to_right < r_next + r_right + tolerance:
			# Would collide with right circle, stop.
			break

		# Place the circle.
		centers.append((actual_next_x, z_fixed))
		current_x = actual_next_x
		circle_index += 1
	
	# Test for additional circle placement between last positioned circle and right circle.
	if len(centers) >= 3:  # Make sure we have at least left, right, and one other circle.
		last_positioned_x = centers[-1][0]  # X-coordinate of last positioned circle (not right corner).
		right_circle_x = centers[1][0]  # X-coordinate of right corner circle (x_max).
		
		# Get radii of both circles.
		r_last_positioned = density_function(last_positioned_x, z_fixed)/2
		if len(centers) > 3:  # If last positioned circle had reduction applied.
			r_last_positioned = r_last_positioned * (1 - radius_reduction_percentage)
		
		r_right_circle = density_function(right_circle_x, z_fixed)/2  # Right circle never gets reduction.
		
		# Calculate actual distance between circle surfaces (not centers).
		center_distance = right_circle_x - last_positioned_x
		surface_distance = center_distance - r_last_positioned - r_right_circle
		
		# Calculate mean radius.
		mean_radius = (r_last_positioned + r_right_circle)
		
		# Test: if surface distance > half the mean radius, place a circle in the middle.
		if surface_distance > mean_radius / 2:
			middle_x = (last_positioned_x + right_circle_x) / 2
			centers.append((middle_x, z_fixed))
		else:
			# Gap is too small for additional circle, move last positioned circle by half the surface distance.
			new_x = last_positioned_x + surface_distance / 2
			centers[-1] = (new_x, z_fixed)  # Update the last positioned circle's position.
	
	return points + centers
def place_circles_vertical_boundary(points, z_min, z_max, x_fixed, density_function, N):
	"""Places tangent circles along a vertical boundary.
	
	Parameters
	----------
	points : list of array-like
	    Existing point coordinates to retain.
	z_min : float
	    Lower boundary coordinate.
	z_max : float
	    Upper boundary coordinate.
	x_fixed : float
	    Constant horizontal coordinate of the boundary.
	density_function : callable
	    Function returning the local target circle diameter at ``(x, z)``.
	N : int
	    Boundary sampling argument retained by the public interface.
	
	Returns
	-------
	boundary_points : list
	    Existing points followed by the generated vertical-boundary points.
	"""
	# Remove old corner points if present (with tolerance for floating point precision).
	tolerance = 1e-6
	points = [pt for pt in points if not ((abs(pt[0] - x_fixed) < tolerance and abs(pt[1] - z_min) < tolerance) or
										  (abs(pt[0] - x_fixed) < tolerance and abs(pt[1] - z_max) < tolerance))]
	
	# Start with corner circles.
	centers = [(x_fixed, z_min), (x_fixed, z_max)]
	
	# Place circles greedily from top to bottom.
	current_z = z_max
	
	while True:
		# Get radius of current circle.
		r_current = density_function(x_fixed, current_z)/2
		
		# Calculate next position (circles just touching).
		# Move down by 2 * current radius to find center of next circle.
		next_z = current_z - 2 * r_current
		
		# Get radius of the next circle at that position.
		r_next = density_function(x_fixed, next_z)/2
		
		# Adjust position so circles actually touch (distance between centers = sum of radii).
		actual_next_z = current_z - (r_current + r_next)
		
		# Check if this circle would collide with bottom corner circle.
		r_bottom = density_function(x_fixed, z_min)/2
		distance_to_bottom = actual_next_z - z_min
		
		# Place the circle.
		centers.append((x_fixed, actual_next_z))
		current_z = actual_next_z
		
		if distance_to_bottom < r_next + r_bottom + tolerance:
			# Would collide with bottom circle, stop.
			break
	
	# Calculate how far the last circle should move up to avoid collision.
	last_circle_z = centers[-1][1]  # Z-coordinate of the last placed circle.
	r_last = density_function(x_fixed, last_circle_z)/2
	r_bottom = density_function(x_fixed, z_min)/2
	
	# Position where the last circle should be to just touch the bottom circle.
	corrected_z = z_min + r_last + r_bottom
	
	# Distance to move up.
	move_up_distance = corrected_z - last_circle_z
	
	# Calculate percentage reduction needed.
	# Exclude: top circle (index 1), bottom circle (index 0), and first circle after top (index 2).
	# So we modify circles from index 3 onwards.
	total_diameter_to_reduce = 0
	
	for i in range(3, len(centers)):  # Start from index 3 (second circle after top).
		circle_z = centers[i][1]
		r_circle = density_function(x_fixed, circle_z)/2
		total_diameter_to_reduce += 2 * r_circle  # Each radius reduction saves 2*r in diameter.
	
	if total_diameter_to_reduce > 0:
		radius_reduction_percentage = move_up_distance / total_diameter_to_reduce
	else:
		radius_reduction_percentage = 0

	
	# Clear centers and redo the placement with reduced radii.
	centers = [(x_fixed, z_min), (x_fixed, z_max)]
	current_z = z_max
	circle_index = 0  # Track which circle we're placing.
	
	while True:
		# Get radius of current circle.
		r_current = density_function(x_fixed, current_z)/2
		# Apply reduction if this is not top, bottom, or first after top.
		if circle_index >= 2:  # Circles from index 2 onwards (first after top gets reduction).
			r_current = r_current * (1 - radius_reduction_percentage)
		
		# Calculate next position.
		next_z = current_z - 2 * r_current
		
		# Get radius of the next circle at that position.
		r_next = density_function(x_fixed, next_z)/2
		# Apply reduction if the next circle is not top, bottom, or first after top.
		if circle_index + 1 >= 2:  # Next circle will be at index circle_index + 1.
			r_next = r_next * (1 - radius_reduction_percentage)
		
		# Adjust position so circles actually touch.
		actual_next_z = current_z - (r_current + r_next)
		
		# Check if this circle would collide with bottom corner circle.
		r_bottom = density_function(x_fixed, z_min)/2  # Bottom circle radius not reduced.
		distance_to_bottom = actual_next_z - z_min
		
		if distance_to_bottom < r_next + r_bottom + tolerance:
			# Would collide with bottom circle, stop.
			break

		# Place the circle.
		centers.append((x_fixed, actual_next_z))
		current_z = actual_next_z
		circle_index += 1
	
	# Test for additional circle placement between last positioned circle and bottom circle.
	if len(centers) >= 3:  # Make sure we have at least top, bottom, and one other circle.
		last_positioned_z = centers[-1][1]  # Z-coordinate of last positioned circle (not bottom corner).
		bottom_circle_z = centers[0][1]  # Z-coordinate of bottom corner circle (z_min).
		
		# Get radii of both circles.
		r_last_positioned = density_function(x_fixed, last_positioned_z)/2
		if len(centers) > 3:  # If last positioned circle had reduction applied.
			r_last_positioned = r_last_positioned * (1 - radius_reduction_percentage)
		
		r_bottom_circle = density_function(x_fixed, bottom_circle_z)/2  # Bottom circle never gets reduction.
		
		# Calculate actual distance between circle surfaces (not centers).
		center_distance = last_positioned_z - bottom_circle_z
		surface_distance = center_distance - r_last_positioned - r_bottom_circle
		
		# Calculate mean radius.
		mean_radius = (r_last_positioned + r_bottom_circle)
		
		# Test: if surface distance > half the mean radius, place a circle in the middle.
		if surface_distance > mean_radius / 2:
			middle_z = (last_positioned_z + bottom_circle_z) / 2
			centers.append((x_fixed, middle_z))
		else:
			# Gap is too small for additional circle, move last positioned circle by half the surface distance.
			new_z = last_positioned_z - surface_distance / 2
			centers[-1] = (x_fixed, new_z)  # Update the last positioned circle's position.
	
	return points + centers

def place_circles_ellipse_boundary(points, a, b, exponent, xc, zc, 
								   radius_function, N):
	"""Places nonoverlapping circles along the lower half of a superellipse.
	
	Parameters
	----------
	points : list of array-like
	    Existing point coordinates that receive the accepted boundary points.
	a : float
	    Horizontal semi-axis of the superellipse.
	b : float
	    Vertical semi-axis of the superellipse.
	exponent : float
	    Superellipse shape exponent.
	xc : float
	    Horizontal coordinate of the superellipse center.
	zc : float
	    Vertical coordinate of the superellipse center.
	radius_function : callable
	    Function returning the local target circle diameter at ``(x, z)``.
	N : int
	    Boundary sampling argument retained by the public interface.
	
	Returns
	-------
	points : list
	    Updated point list containing accepted superellipse-boundary points.
	edge_left : list or None
	    Left intersection of the superellipse with ``z = 0``.
	edge_right : list or None
	    Right intersection of the superellipse with ``z = 0``.
	"""
	def superellipse(t, a, b, n):
		"""Evaluates the translated superellipse parameterization.
		
		Parameters
		----------
		t : float
		    Parametric angle in radians.
		a : float
		    Horizontal semi-axis.
		b : float
		    Vertical semi-axis.
		n : float
		    Superellipse exponent.
		
		Returns
		-------
		coordinate : tuple of float
		    The translated ``(x, z)`` coordinate.
		"""
		cos_t = np.cos(t)
		sin_t = np.sin(t)
		x = np.sign(cos_t) * (np.abs(cos_t) ** (2 / n)) * a
		z = np.sign(sin_t) * (np.abs(sin_t) ** (2 / n)) * b
		return x + xc, z + zc

	# Sample full ellipse finely for arc-length parameterization.
	t_dense = np.linspace(0, 2 * np.pi, 12000)
	xy_dense = np.array([superellipse(t, a, b, exponent) for t in t_dense])
	dx = np.diff(xy_dense[:, 0])
	dz = np.diff(xy_dense[:, 1])
	segment_lengths = np.sqrt(dx**2 + dz**2)
	arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

	# Get all interpolated points.
	total_length = arc_lengths[-1]
	target_lengths = np.linspace(0, total_length, 12000)
	x_interp = np.interp(target_lengths, arc_lengths, xy_dense[:, 0])
	z_interp = np.interp(target_lengths, arc_lengths, xy_dense[:, 1])

	# Calculate edge points at z = 0 analytically.
	z_target = 0.0
	z_rel = z_target - zc
	z_term = abs(z_rel / b) ** exponent
	
	edge_left = None
	edge_right = None
	
	if z_term < 1.0:
		x_term = 1.0 - z_term
		x_displacement = a * (x_term ** (1.0 / exponent))
		
		x_left = xc - x_displacement
		x_right = xc + x_displacement
		
		edge_left = [x_left, z_target]
		edge_right = [x_right, z_target]

	# Separate points into left and right halves, bottom half only (z < 0).
	bottom_points = [(x, z) for x, z in zip(x_interp, z_interp) 
					 if z < -1e-6]  # Only bottom half, avoid z=0 duplicates.
	
	# Split into left and right halves based on x coordinate relative to center.
	left_points = [(x, z) for x, z in bottom_points if x <= xc]
	right_points = [(x, z) for x, z in bottom_points if x > xc]
	
	# Sort left points: from edge (z=0) toward bottom center (most negative z).
	# Process the left side.
	left_points.sort(key=lambda pt: -pt[1])  # Sort by -z (decreasing z).
	
	# Sort right points: from edge (z=0) toward bottom center (most negative z).
	# Process the right side.
	right_points.sort(key=lambda pt: -pt[1])  # Sort by -z (decreasing z).

	# Track placed points and their radii for collision detection.
	placed_points = []
	placed_radii = []

	def try_place_point(x, z):
		"""Adds a superellipse point when its circle does not overlap accepted circles.
		
		Parameters
		----------
		x : float
		    Horizontal coordinate of the candidate.
		z : float
		    Vertical coordinate of the candidate.
		
		Returns
		-------
		accepted : bool
		    ``True`` when the point is added; otherwise ``False``.
		"""
		r = radius_function(x, z)/2
		
		# Check collision with all previously placed points.
		collision = any((x - px)**2 + (z - pz)**2 < (r + pr)**2
					   for (px, pz), pr in zip(placed_points, placed_radii))
		
		if not collision:
			points.append([x, z])
			placed_points.append((x, z))
			placed_radii.append(r)
			return True
		return False

	# Place edge points first if they exist.
	if edge_left is not None:
		try_place_point(edge_left[0], edge_left[1])
	if edge_right is not None:
		try_place_point(edge_right[0], edge_right[1])

	# Place points from left edge toward bottom center.
	for x, z in left_points:
		try_place_point(x, z)

	# Place points from right edge toward bottom center.
	for x, z in right_points:
		try_place_point(x, z)

	return points, edge_left, edge_right

def filter_points_inside_superellipse(points, a, b, n, center_x=0, center_y=0, eps=1e-12):
	"""Filters coordinates using a superellipse inclusion test.
	
	Parameters
	----------
	points : array-like of shape (n_points, 2)
	    Coordinates to test.
	a : float
	    Horizontal semi-axis.
	b : float
	    Vertical semi-axis.
	n : float
	    Superellipse shape exponent.
	center_x : float, optional
	    Horizontal coordinate of the superellipse center.
	center_y : float, optional
	    Vertical coordinate of the superellipse center.
	eps : float, optional
	    Lower clipping value used in normalized coordinates.
	
	Returns
	-------
	filtered_points : numpy.ndarray
	    Coordinates located inside or on the superellipse.
	"""
	points = np.array(points)
	
	# Translate points to superellipse-centered coordinate system.
	x_centered = points[:, 0] - center_x
	y_centered = points[:, 1] - center_y
	
	# Apply superellipse inequality: |x/a|^n + |y/b|^n <= 1.
	# Use eps to avoid potential issues with zero values.
	abs_x_norm = np.clip(np.abs(x_centered / a), eps, None)
	abs_y_norm = np.clip(np.abs(y_centered / b), eps, None)
	
	# Calculate superellipse values.
	superellipse_values = abs_x_norm**n + abs_y_norm**n
	
	# Filter points inside (value <= 1).
	inside_mask = superellipse_values <= 1.0
	
	return points[inside_mask] 

from numba import njit

@njit(fastmath=True)
def process_column_fast(x_curr, z_min, dz, Ny, f_col, neighbors, overlap, max_per_row):
    """Generates collision-free candidates for one vertical sampling column.
    
    Parameters
    ----------
    x_curr : float
        Horizontal coordinate of the current column.
    z_min : float
        Vertical coordinate of the first row.
    dz : float
        Vertical grid spacing.
    Ny : int
        Number of vertical grid rows.
    f_col : numpy.ndarray
        Circle radii for the current column.
    neighbors : numpy.ndarray
        Active neighbor rows stored as ``(x, z, radius)``.
    overlap : float
        Collision-distance scaling factor.
    max_per_row : int
        Maximum neighbor references stored for each row.
    
    Returns
    -------
    new_points : list of tuple
        Accepted ``(x, z, radius)`` candidates for the column.
    """
    new_points = []
    
    n_neighbors = len(neighbors)
    
    # Build the row-indexed neighbor map.
    # Row_map stores INDICES of neighbors located at that Z-row.
    # Use the dynamic max_per_row passed from the main loop.
    row_map = np.full((Ny, max_per_row), -1, dtype=np.int32)
    row_counts = np.zeros(Ny, dtype=np.int32)
    
    # Find max neighbor radius (for safe vertical search range).
    f_max_neigh = 0.0
    
    if n_neighbors > 0:
        for k in range(n_neighbors):
            pz = neighbors[k, 1]
            pf = neighbors[k, 2]
            
            if pf > f_max_neigh: f_max_neigh = pf
            
            # Map neighbor to exact grid index.
            z_idx = int(round((pz - z_min) / dz))
            
            if 0 <= z_idx < Ny:
                c = row_counts[z_idx]
                if c < max_per_row:
                    row_map[z_idx, c] = k # Store neighbor index.
                    row_counts[z_idx] += 1

    # Process rows in the current column.
    j = 0
    while j < Ny:
        f_c = f_col[j] 
        z_c = z_min + j * dz 
        
        is_collision = False
        
        # Search neighboring rows within the collision range.
        if n_neighbors > 0:
            # Check only rows whose vertical separation is smaller than the summed radii.
            dist_limit = f_c + f_max_neigh
            
            # Convert physical distance to index range.
            idx_range = int(np.ceil(dist_limit / abs(dz)))
            
            k_min = max(0, j - idx_range)
            k_max = min(Ny, j + idx_range + 1)
            
            # Iterate only the specific rows that might collide.
            for k in range(k_min, k_max):
                count = row_counts[k]
                if count == 0: continue
                
                # Check neighbors on this specific row.
                for slot in range(count):
                    # Retrieve neighbor data.
                    neigh_idx = row_map[k, slot]
                    
                    px = neighbors[neigh_idx, 0]
                    pz = neighbors[neigh_idx, 1]
                    pf = neighbors[neigh_idx, 2]
                    
                    sum_radii = f_c + pf
                    
                    # Fast Bounding Box.
                    if abs(x_curr - px) > sum_radii: continue
                    
                    # Exact Distance Check.
                    dist_sq = (x_curr - px)**2 + (z_c - pz)**2
                    min_dist_sq = (sum_radii**2) / overlap
                    
                    if dist_sq < min_dist_sq:
                        is_collision = True
                        break
                
                if is_collision: break
        
        # Store the column result.
        if not is_collision:
            # Point is valid.
            new_points.append((x_curr, z_c, f_c))
            
            # Advance by the local spacing interval.
            req_dist = 2.0 * f_c
            skip_distance = int(np.ceil(req_dist / abs(dz))) - 1
            if skip_distance < 0: skip_distance = 0
            
            j += skip_distance + 1
        else:
            # Collision detected, move 1 step.
            j += 1
            
    return new_points
def generate_mesh(x_range: Tuple[float, float], 
                           z_range: Tuple[float, float], 
                           npoints: int, 
                           density_function: Callable[[float, float], float],
                           N: int,
                           pad_type: str =None,
                           padding_x: float = None,
                           padding_z: float = None,
                           ellipse_n: float = None,
                           subdomain: bool = True,
                           overlap: float = 1.0,
				           fineness: str = None,
				           f_min: float = None,
				           f_max: float = None,) -> np.ndarray:

    """Generates an adaptive two-dimensional point mesh by bubble packing.
    
    Parameters
    ----------
    x_range : tuple of float
        Horizontal limits of the physical model.
    z_range : tuple of float
        Vertical limits of the physical model.
    npoints : int
        Maximum number of mesh points to return.
    density_function : callable
        Function returning the target local spacing at ``(x, z)``.
    N : int
        Number of horizontal sampling columns for custom fineness.
    pad_type : {None, "rectangular", "elliptical"}, optional
        Geometry used to pad the physical model.
    padding_x : float, optional
        Horizontal padding distance.
    padding_z : float, optional
        Vertical padding distance.
    ellipse_n : float, optional
        Superellipse exponent for elliptical padding.
    subdomain : bool, optional
        Whether to preserve and smooth the original model boundary inside padding.
    overlap : float, optional
        Collision-distance scaling factor used during point acceptance.
    fineness : str, optional
        Preset grid resolution: ``very coarse``, ``coarse``, ``fine``, ``very fine``, or ``custom``.
    f_min : float, optional
        Minimum spacing estimate used by fineness presets.
    f_max : float, optional
        Maximum spacing estimate used to size the active-neighbor window.
    
    Returns
    -------
    points : numpy.ndarray
        Generated coordinates with at most ``npoints`` rows.
    triangulation : scipy.spatial.Delaunay
        Delaunay triangulation of the returned coordinates.
    boundary_points : list
        Coordinates classified as fixed boundary points.
    """
    box_xmin,box_xmax = x_range
    box_zmax,box_zmin = z_range         
    # Rectangle center.
    xc = box_xmax / 2.0 
    zc = box_zmin / 2.0     

    if(pad_type==None): 
        domain_xmin, domain_xmax = x_range
        domain_zmax, domain_zmin = z_range
    
    if(pad_type=="elliptical"):
        
        # Ellipse semi-axes.
        ellipse_a = (box_zmin / 2.0) - padding_z  
        ellipse_b = (box_xmax / 2.0) + padding_x  
        # Domain bounding box.
        domain_zmin = zc + ellipse_a
        domain_zmax =  0.0
        domain_xmin = xc - ellipse_b 
        domain_xmax = xc + ellipse_b 
        bbox = (domain_zmin, domain_zmax, domain_xmin, domain_xmax)
        rect_x = (box_xmin, box_xmax)
        rect_z = (box_zmax, box_zmin)
    
    if(pad_type=="rectangular"):  
        # Domain bounding box.
        domain_zmin = box_zmin - padding_z 
        domain_zmax =  0.0
        domain_xmin = 0.0 - padding_x
        domain_xmax = box_xmax + padding_x
        bbox = (domain_zmin, domain_zmax, domain_xmin, domain_xmax)
        rect_x = (box_xmin, box_xmax)
        rect_z = (box_zmax, box_zmin)
    
    x_min, x_max = domain_xmin, domain_xmax
    z_min, z_max = domain_zmax, domain_zmin
	
    if(fineness=="very coarse"):
	    N = int(x_max/(f_min * 2))
	    Ny = abs(int(N*(z_max - z_min)/(x_max - x_min)))
    if(fineness=="coarse"):
	    N = int(x_max/(f_min))
	    Ny = abs(int(N*(z_max - z_min)/(x_max - x_min)))
    if(fineness=="fine"):
	    N = int(x_max/(f_min /2))
	    Ny = abs(int(N*(z_max - z_min)/(x_max - x_min)))
    if(fineness=="very fine"):
	    N = int(x_max/(f_min /4))
	    Ny = abs(int(N*(z_max - z_min)/(x_max - x_min)))
    if(fineness=="custom" or fineness==None):
	    N = N
	    Ny = abs(int(N*(z_max - z_min)/(x_max - x_min)))
	
    if(f_min==None or f_max==None):
        print("Evaluating density function...")
        Ny = abs(int(N*(z_max - z_min)/(x_max - x_min)))
        x_sample = np.linspace(x_min, x_max, N)
        z_sample = np.linspace(z_min, z_max, Ny)
        X_sample, Z_sample = np.meshgrid(x_sample, z_sample)        
        x_flat = X_sample.flatten()
        z_flat = Z_sample.flatten()        
        try:
            f_vals = density_function(x_flat, z_flat)/2
        except:
            print("Density function doesn't support vectorization for sampling, using element-wise evaluation...")
            f_vals = np.array([density_function(x, z)/2 for x, z in zip(x_flat, z_flat)])        
        f_min, f_max = np.min(f_vals), np.max(f_vals)
        print(f"Density function range: [{f_min:.3f}, {f_max:.3f}]")

    dx = ((x_max - x_min) / (N - 1))
    dz = ((z_max - z_min) / (Ny - 1))

    print("Pre-computing density values for grid centers...")
    x_centers = np.array([x_min + i * dx for i in range(N)])
    z_centers = np.array([z_min + j * dz for j in range(Ny)])
    X_centers, Z_centers = np.meshgrid(x_centers, z_centers)

    x_flat = X_centers.flatten()
    z_flat = Z_centers.flatten()

    try:
        f_centers_flat = density_function(x_flat, z_flat)/2
        f_centers = f_centers_flat.reshape(Ny, N)
    except:
        print("Density function doesn't support vectorization, using element-wise evaluation...")
        f_centers = np.array([density_function(x, z)/2 for x, z in zip(x_flat, z_flat)])
        f_centers = f_centers.reshape(N, Ny)
          
    print("Placing boundary circles...")
    # Initialize boundary_points array.
    points = []
    boundary_points = []
    if(pad_type=="elliptical"):
        if(subdomain==True):
            rx_min, rx_max = rect_x
            rz_min, rz_max = rect_z
            
            points = place_circles_vertical_boundary(points,rz_max, rz_min, rx_min, density_function, Ny)
    
            points = place_circles_horizontal_boundary(points,rx_min, rx_max, rz_max, density_function, N)
            
            points = place_circles_vertical_boundary(points,rz_max, rz_min, rx_max, density_function, Ny)
    
            points = place_circles_horizontal_boundary(points,rx_min, rx_max, rz_min, density_function, N)
    
            points,edge_left,edge_right = place_circles_ellipse_boundary(points,  ellipse_b,ellipse_a,ellipse_n, xc,zc, density_function, N)
    
            points = place_circles_horizontal_boundary(points,edge_left[0], rx_min, z_min, density_function, N)
    
            points = place_circles_horizontal_boundary(points,rx_max, edge_right[0], z_min, density_function, N)
        else:
            rx_min, rx_max = rect_x
            rz_min, rz_max = rect_z
            points = place_circles_horizontal_boundary(points,rx_min, rx_max, rz_min, density_function, N)
    
            points,edge_left,edge_right = place_circles_ellipse_boundary(points,  ellipse_b,ellipse_a,ellipse_n, xc,zc, density_function, N)
    
            points = place_circles_horizontal_boundary(points,edge_left[0], rx_min, z_min, density_function, N)

            points = place_circles_horizontal_boundary(points,rx_max, edge_right[0], z_min, density_function, N)
    
            
        boundary_points = points.copy()

    if(pad_type=="rectangular"):
        if(subdomain==True):
            rx_min, rx_max = rect_x
            rz_min, rz_max = rect_z
    
            points = place_circles_vertical_boundary(points,rz_max, rz_min, rx_min, density_function, N)
    
            points = place_circles_horizontal_boundary(points,rx_min, rx_max, rz_max, density_function, N)
            
            points = place_circles_vertical_boundary(points,rz_max, rz_min, rx_max, density_function, N)
    
            points = place_circles_horizontal_boundary(points,rx_min, rx_max, rz_min, density_function, N)
    
            points = place_circles_vertical_boundary(points,z_max, z_min, x_min, density_function, N)
    
            points = place_circles_horizontal_boundary(points,x_min, x_max, z_max, density_function, N)
    
            points = place_circles_vertical_boundary(points,z_max, z_min, x_max, density_function, N)
    
            points = place_circles_horizontal_boundary(points,x_min, rx_min, z_min, density_function, N)
    
            points = place_circles_horizontal_boundary(points,rx_max, x_max, z_min, density_function, N)
        else:
            points = place_circles_vertical_boundary(points,z_max, z_min, x_min, density_function, Ny)

            points = place_circles_horizontal_boundary(points,x_min, x_max, z_max, density_function, N)
            
            points = place_circles_vertical_boundary(points,z_max, z_min, x_max, density_function, Ny)
    
            points = place_circles_horizontal_boundary(points,x_min, x_max, z_min, density_function, N)
            
        boundary_points = points.copy()

    if(pad_type==None):
        
        points = place_circles_vertical_boundary(points,z_max, z_min, x_min, density_function, Ny)

        points = place_circles_horizontal_boundary(points,x_min, x_max, z_max, density_function, N)
        
        points = place_circles_vertical_boundary(points,z_max, z_min, x_max, density_function, Ny)

        points = place_circles_horizontal_boundary(points,x_min, x_max, z_min, density_function, N)

        boundary_points = points.copy()

    print("Boundary circles placing complete.")
    
    # Generate interior points column by column.
    
    # Initialize Active Neighbors.
    if len(points) > 0:
        pts_arr = np.array(points)
        
        # Recover 'f' values.
        existing_f = []
        for p in points:
            ix = int(round((p[0] - x_min) / dx))
            iz = int(round((p[1] - z_min) / dz))
            ix = max(0, min(N-1, ix))
            iz = max(0, min(Ny-1, iz))
            existing_f.append(f_centers[iz, ix])
        
        # Active neighbors: [x, z, f].
        active_neighbors = np.column_stack((pts_arr[:, 0], pts_arr[:, 1], existing_f))
    else:
        active_neighbors = np.empty((0, 3))
    
    # Calculate collision reach.
    max_collision_distance = 2 * f_max
    
    # Calculate the row capacity for the collision window.
    # Add +2 as a small safety buffer for edge cases in rounding.
    max_per_row = int(np.ceil(max_collision_distance / dx)) + 2
    
    print(f"Processing mesh...")
    #print(f"Collision Window: {max_collision_distance:.2f}, Max neighbors per row: {max_per_row}")
    
    # Process the grid column by column.
    for i in range(N):
        if len(points) >= npoints:
            print(f"Reached {npoints} points")
            break
            
        current_x = x_min + i * dx
        
        # Discard neighbors outside the active collision window.
        if len(active_neighbors) > 0:
            cutoff_x = current_x - max_collision_distance
            if active_neighbors[0, 0] < cutoff_x:
                active_neighbors = active_neighbors[active_neighbors[:, 0] >= cutoff_x]

        # Generate candidates for the current column.
        f_col = f_centers[:, i]
        # Keep only neighbors that can collide with the current column.
        if active_neighbors.shape[0] > 0:
            local_mask = (
                np.abs(active_neighbors[:, 0] - current_x)
                <= max_collision_distance
            )
            local_neighbors = active_neighbors[local_mask]
        
            # Determine the exact row capacity required locally.
            if local_neighbors.shape[0] > 0:
                row_indices = np.rint(
                    (local_neighbors[:, 1] - z_min) / dz
                ).astype(np.int64)
        
                valid = (
                    (row_indices >= 0)
                    & (row_indices < Ny)
                )
        
                if np.any(valid):
                    max_per_row = max(
                        1,
                        int(
                            np.bincount(
                                row_indices[valid],
                                minlength=Ny,
                            ).max()
                        ),
                    )
                else:
                    max_per_row = 1
            else:
                max_per_row = 1
        else:
            local_neighbors = active_neighbors
            max_per_row = 1
        new_pts_list = process_column_fast(
            current_x, z_min, dz, Ny, f_col, 
            local_neighbors, overlap, max_per_row
        )
        
        # Store accepted candidates.
        if len(new_pts_list) > 0:
            new_arr = np.array(new_pts_list)
            points.extend(new_arr[:, :2].tolist())
            active_neighbors = np.vstack((active_neighbors, new_arr))
            
        # Progress.
        if i % max(1, int(N/20)) == 0:
            print(f"[Pass 1] Progress: {i/N*100:.1f}% - Points: {len(points)}")

    print(f"Final Point Count: {len(points)}")

    

    if(pad_type=="elliptical"):
        points = filter_points_inside_superellipse(points,  ellipse_b,ellipse_a,ellipse_n, xc,zc)

    all_points = []
    all_points.extend(points)
    points_array = np.array(all_points[:npoints])
    tri = Delaunay(points_array)
    
    return points_array, tri, boundary_points
