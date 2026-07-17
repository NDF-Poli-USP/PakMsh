import numpy as np
from scipy.spatial import Delaunay
from typing import Tuple, Callable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_mesh(points: np.ndarray, 
              x_range: tuple[float, float], 
              z_range: tuple[float, float],
              density_function=None,
              show_points: bool = True,
              show_density: bool = True,
              show_element_size: bool = True,
              element_size_scale: float = 1000.0,
              element_quality: bool = False,
              WorstElement: bool = False,
              filename: str = "mesh"):
    """Plots and saves a Delaunay mesh visualization.
    
    Parameters
    ----------
    points : numpy.ndarray of shape (n_points, 2)
        Mesh coordinates.
    x_range : tuple of float
        Horizontal plot limits.
    z_range : tuple of float
        Vertical plot limits.
    density_function : callable, optional
        Background sizing or velocity field evaluated at ``(x, z)``.
    show_points : bool, optional
        Whether to draw mesh generators.
    show_density : bool, optional
        Whether to draw the background field when available.
    show_element_size : bool, optional
        Whether to shade triangles by mean edge length.
    element_size_scale : float, optional
        Element-size scaling argument reserved by the plotting interface.
    element_quality : bool, optional
        Whether to shade triangles by quality.
    WorstElement : bool, optional
        Whether to highlight only the lowest-quality triangle.
    filename : str, optional
        Base name for the generated PNG file.
    
    Returns
    -------
    result : None
        The function saves a PNG file and displays the figure.
    """

    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from scipy.spatial import Delaunay
    import numpy as np
    import uuid

    # Configure plotting constants.
    FONT_SIZE = 40  
    TICK_SIZE = 30

    # Compute triangle quality for plotting.
    def triangle_quality(p1, p2, p3):
        """Computes the normalized inradius-to-circumradius triangle quality.
        
        Parameters
        ----------
        p1 : numpy.ndarray
            First triangle vertex.
        p2 : numpy.ndarray
            Second triangle vertex.
        p3 : numpy.ndarray
            Third triangle vertex.
        
        Returns
        -------
        quality : float
            Triangle quality in the interval ``[0, 1]``.
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

    # Construct the Delaunay triangulation.
    triangulation = Delaunay(points)

    fig, ax = plt.subplots(1, 1, figsize=(40, 40))

    # Highlight the lowest-quality element.
    if WorstElement:
        qualities = []
        for simplex in triangulation.simplices:
            p1, p2, p3 = points[simplex]
            qualities.append(triangle_quality(p1, p2, p3))

        worst_idx = int(np.argmin(qualities))
        worst_triangle = triangulation.simplices[worst_idx]
        verts = points[triangulation.simplices]

        pc_all = PolyCollection(verts, 
                                facecolors='lightgray', 
                                edgecolors='lightgray', 
                                linewidths=0.8, 
                                alpha=0.7)
        ax.add_collection(pc_all)

        worst_vert = points[worst_triangle]
        pc_worst = PolyCollection([worst_vert], 
                                  facecolors='red', 
                                  edgecolors='red', 
                                  linewidths=2.0, 
                                  alpha=1.0)
        ax.add_collection(pc_worst)

        ax.set_title(f'Delaunay Triangulation - Worst Element Highlighted\n'
                     f'Worst Quality: {np.min(qualities):.3f}', fontsize=FONT_SIZE)

    # Shade elements by triangle quality.
    elif element_quality:
        qualities = []
        for simplex in triangulation.simplices:
            p1, p2, p3 = points[simplex]
            qualities.append(triangle_quality(p1, p2, p3))

        tripcolor_plot = ax.tripcolor(
            points[:, 0], points[:, 1], triangulation.simplices,
            facecolors=qualities, cmap='RdYlGn',
            edgecolors='k', alpha=0.9,
            vmin=0, vmax=1
        )
        
        cbar = plt.colorbar(tripcolor_plot, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8)
        cbar.set_label('Element Quality (0=bad, 1=perfect)', fontsize=FONT_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

        ax.set_title(f'Delaunay Triangulation - Colored by Element Quality\nMean: {np.mean(qualities):.3f}, Min: {np.min(qualities):.3f}', fontsize=FONT_SIZE)

    # Shade elements by mean edge length.
    elif show_element_size: # Render element-size shading whenever requested.
        triangle_element_sizes = []
        for simplex in triangulation.simplices:
            p1, p2, p3 = points[simplex]
            
            # Calculate the 3 edge lengths.
            edge1 = np.linalg.norm(p1 - p2)
            edge2 = np.linalg.norm(p2 - p3)
            edge3 = np.linalg.norm(p3 - p1)
            
            # Calculate actual size (Average Edge Length).
            # Use the mean edge length as the element-size metric.
            actual_size = (edge1 + edge2 + edge3) / 3.0
            
            triangle_element_sizes.append(actual_size)
        
        tripcolor_plot = ax.tripcolor(points[:, 0], points[:, 1], triangulation.simplices, 
                                      facecolors=triangle_element_sizes, 
                                      cmap='viridis', alpha=0.7, edgecolors='none')
        
        ax.triplot(points[:, 0], points[:, 1], triangulation.simplices, color='black', alpha=1.0, linewidth=0.5)
        
        cbar_elem = plt.colorbar(tripcolor_plot, ax=ax, orientation='horizontal', pad=0.02, shrink=0.8)
        cbar_elem.set_label('Actual Element Size (Avg Edge Length)', fontsize=FONT_SIZE)
        cbar_elem.ax.tick_params(labelsize=TICK_SIZE)

        ax.set_title(f'Delaunay Triangulation - Colored by Actual Element Size', fontsize=FONT_SIZE)

    # Plot the sizing function as a background contour.
    elif show_density and density_function is not None:
        grid_x = np.linspace(x_range[0], x_range[1], 3000)
        grid_z = np.linspace(z_range[0], z_range[1], 3000)
        X, Z = np.meshgrid(grid_x, grid_z)

        try:
            vals = density_function(X, Z)
        except:
            vec_density = np.vectorize(density_function)
            vals = vec_density(X, Z)

        contour_plot = ax.contourf(X, Z, vals, levels=100, cmap='viridis', alpha=0.9)
        
        
        # Place the color bar below the plot.
        cbar_dens = plt.colorbar(contour_plot, ax=ax, orientation='horizontal', pad=0.04, shrink=0.8)
        cbar_dens.set_label('Velocity (m/s)', fontsize=FONT_SIZE)
        cbar_dens.ax.tick_params(labelsize=TICK_SIZE)
        
        ax.set_title(f'Avenir model', fontsize=FONT_SIZE)

    # Plot the unshaded triangulation.
    else:
        ax.triplot(points[:, 0], points[:, 1], triangulation.simplices, color='black', alpha=0.7, linewidth=1)

    # Plot the mesh generators.
    if show_points:
        ax.scatter(points[:, 0], points[:, 1], c='red', s=6, alpha=0.9, edgecolors='black', linewidth=0.3, zorder=5)

    ax.set_xlim(x_range)
    ax.set_ylim(z_range)
    ax.invert_yaxis()
    
    ax.set_xlabel('Length (m)', fontsize=FONT_SIZE)
    ax.set_ylabel('Depth (m)', fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    random_id = str(uuid.uuid4())[:4]
    output_filename = f"{filename}_{random_id}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()



def plot_mesh_circles(points, f_centers_fn, x_min, x_max, z_min, z_max, ax=None, circle_resolution=16, filename="bubbles"):
    """Plots and saves the local sizing circles associated with mesh points.
    
    Parameters
    ----------
    points : array-like of shape (n_points, 2)
        Circle-center coordinates.
    f_centers_fn : callable
        Function returning the local circle radius at ``(x, z)``.
    x_min : float
        Minimum horizontal plot coordinate.
    x_max : float
        Maximum horizontal plot coordinate.
    z_min : float
        Minimum vertical plot coordinate.
    z_max : float
        Maximum vertical plot coordinate.
    ax : matplotlib.axes.Axes, optional
        Axes that receive the circle collection.
    circle_resolution : int, optional
        Number of line segments used to approximate each circle.
    filename : str, optional
        Base name for the generated PNG file.
    
    Returns
    -------
    result : None
        The function saves a PNG file and displays the figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection
    import uuid

    # Configure plotting constants.
    FONT_SIZE = 40  
    TICK_SIZE = 30

    if ax is None:
        fig, ax = plt.subplots(figsize=(40, 40))
    
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    
    
    x_vals = points[:, 0]
    z_vals = points[:, 1]
    radii = f_centers_fn(x_vals, z_vals)/2
    
    # Pre-compute unit circle.
    theta = np.linspace(0, 2*np.pi, circle_resolution + 1)
    unit_circle = np.column_stack([np.cos(theta), np.sin(theta)])
    
    # Vectorized circle generation.
    n_points = len(points)
    circles = np.zeros((n_points, circle_resolution + 1, 2))
    
    # Broadcast operations.
    circles[:, :, 0] = x_vals[:, np.newaxis] + radii[:, np.newaxis] * unit_circle[np.newaxis, :, 0]
    circles[:, :, 1] = z_vals[:, np.newaxis] + radii[:, np.newaxis] * unit_circle[np.newaxis, :, 1]
    
    # Create LineCollection.
    lc = LineCollection(circles, colors='blue', linewidths=0.5)
    ax.add_collection(lc)
    
    # Add center points.
    
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    # Save and display the figure.
    plt.tight_layout()
    random_id = str(uuid.uuid4())[:4]
    output_filename = f"{filename}_{random_id}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_fast_voronoi(points):
    """
    Fast plotting of the True Voronoi diagram.
    Uses Scipy's internal plotting routines.
    """
    # 1. Compute Voronoi 
    vor = Voronoi(points)
    
    # 2. Plot
    fig, ax = plt.subplots(figsize=(40, 40))
    
    # voronoi_plot_2d 
    voronoi_plot_2d(vor, ax=ax, 
                    show_vertices=False, 
                    line_colors='black', 
                    line_width=1, 
                    line_alpha=0.6, 
                    point_size=0)
    
    # 3. Crop the view
    margin = 0.05
    p_min = points.min(axis=0)
    p_max = points.max(axis=0)
    
    ax.set_xlim(p_min[0] - margin, p_max[0] + margin)
    ax.set_ylim(p_min[1] - margin, p_max[1] + margin)
    ax.set_aspect('equal')
    ax.set_title("True Voronoi Diagram")
    
    plt.show()
