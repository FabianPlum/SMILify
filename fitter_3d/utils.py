import os
import torch
from multiprocessing import Pool, cpu_count, set_start_method

import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use Agg backend for non-GUI/non-interactive plotting

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from matplotlib.tri import Triangulation

import config

# Set the multiprocessing start method to "spawn"
set_start_method('spawn', force=True)


def try_mkdir(loc):
    if not os.path.isdir(loc):
        os.mkdir(loc)


def try_mkdirs(locs):
    for loc in locs: try_mkdir(loc)


def equal_3d_axes(ax, X, Y, Z, zoom=1.0):
    """Sets all axes to same lengthscale through trick found here:
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to"""

    xmax, xmin, ymax, ymin, zmax, zmin = X.max(), X.min(), Y.max(), Y.min(), Z.max(), Z.min()

    max_range = np.array([xmax - xmin, ymax - ymin, zmax - zmin]).max() / (2.0 * zoom)

    mid_x = (xmax + xmin) * 0.5
    mid_y = (ymax + ymin) * 0.5
    mid_z = (zmax + zmin) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_trisurfs(ax, verts, faces, change_lims=False, color="darkcyan",
                  zoom=1.5, n_meshes=1, alpha=1.0):
    """
    """

    trisurfs = []

    for n in range(n_meshes):
        points = verts[n].cpu().detach().numpy()
        faces = faces[n].cpu().detach().numpy()

        X, Y, Z = np.moveaxis(points, -1, 0)

        tri = Triangulation(X, Y, triangles=faces).triangles

        trisurf_shade = ax.plot_trisurf(X, Y, Z, triangles=tri, alpha=alpha, color=color,
                                        shade=True)  # shade entire mesh
        trisurfs += [trisurf_shade]

    if change_lims: equal_3d_axes(ax, X, Y, Z, zoom=zoom)

    return trisurfs


def plot_mesh(ax, mesh: Meshes, label="", colour="blue", equalize=True, zoom=1.5, alpha=1.0):
    """Given a PyTorch Meshes object, plot the mesh on a 3D axis

    :param equalize: Flag to match axes to be the same scale
    :param zoom: Zoom factor on equalised axes
    """

    verts = mesh.verts_padded()
    faces = mesh.faces_padded()

    trisurfs = plot_trisurfs(ax, verts, faces, color=colour, change_lims=equalize, zoom=zoom,
                             alpha=alpha)

    if label == "SMAL":
        points = verts[0].cpu().detach().numpy()
        X, Y, Z = np.rollaxis(points, -1)

        ax.scatter(X, Y, Z, s=0.02, color="red")

    ax.plot([], [], color=colour, label=label)
    ax.legend(loc='upper right')

    return trisurfs


def plot_set_of_meshes(args):
    target_mesh, src_mesh, name, title, figtitle, out_dir = args

    fig = plt.figure(figsize=(15, 5))
    axes = [fig.add_subplot(int(f"13{n}"), projection="3d") for n in range(1, 4)]

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    colours = ["green", "blue"]
    labels = ["target", "SMAL"]
    for i, mesh in enumerate([target_mesh, src_mesh]):
        for j, ax in enumerate([axes[1 + i == 1], axes[2]]):
            plot_mesh(ax, mesh, colour=colours[i], label=labels[i], alpha=[1, 0.5][j])

    fig.suptitle(figtitle)
    for ax in axes:
        ax.legend(loc='upper right')

    try_mkdir(out_dir)

    plt.savefig(f"{out_dir}/{name} - {title}.png")
    plt.close(fig)


def plot_meshes(target_meshes, src_meshes, mesh_names=[], title="", figtitle="",
                out_dir="static_fits_output/pointclouds", max_workers=None, 
                plot_normals=False, normals_samples=1000):
    """Plot and save figures of point clouds using multiprocessing
    
    Args:
        target_meshes: List of target PyTorch3D Meshes objects
        src_meshes: List of source PyTorch3D Meshes objects
        mesh_names: Optional list of names for each mesh
        title: Title for the plots
        figtitle: Super title for the plots
        out_dir: Directory to save the output images
        max_workers: Maximum number of worker processes for multiprocessing
        plot_normals: Whether to also generate normal plots
        normals_samples: Number of normal vectors to sample for normal plots
    """
    if max_workers is None:
        max_workers = cpu_count()

    # Detach the tensors before passing them to the multiprocessing pool
    target_meshes_detached = [mesh.clone().detach() for mesh in target_meshes]
    src_meshes_detached = [mesh.clone().detach() for mesh in src_meshes]

    args_list = []
    for n in range(len(target_meshes_detached)):
        name = n if not mesh_names else mesh_names[n]
        args_list.append((target_meshes_detached[n], src_meshes_detached[n], name, title, figtitle, out_dir))

    with Pool(processes=max_workers) as pool:
        pool.map(plot_set_of_meshes, args_list)
    
    # Generate normal plots if requested
    if plot_normals:
        # Create normals output directory
        normals_dir = os.path.join(out_dir, "normals")
        try_mkdir(normals_dir)
        
        # Generate normal plots for each mesh pair
        for n in range(len(target_meshes)):
            name = n if not mesh_names else mesh_names[n]
            
            # Generate comparison plot
            comparison_path = os.path.join(normals_dir, f"{name} - {title} - comparison.png")
            plot_mesh_normals_comparison(
                target_meshes[n], 
                src_meshes[n],
                output_path=comparison_path,
                title=f"{figtitle} - {name} - {title}",
                num_samples=normals_samples
            )
            
            # Generate individual normal plots
            target_path = os.path.join(normals_dir, f"{name} - {title} - target_normals.png")
            src_path = os.path.join(normals_dir, f"{name} - {title} - src_normals.png")
            
            plot_mesh_normals_high_res(
                target_meshes[n],
                output_path=target_path,
                title=f"Target Normals - {name}",
                num_samples=normals_samples,
                mesh_color="green"
            )
            
            plot_mesh_normals_high_res(
                src_meshes[n],
                output_path=src_path,
                title=f"Source Normals - {name}",
                num_samples=normals_samples,
                mesh_color="blue"
            )


def plot_pointcloud(ax, mesh, label="", colour="blue",
                    equalize=True, zoom=1.5):
    """Given a Meshes object, plots the mesh on ax (ax must have projection=3d).

    equalize = adjust axis limits such that all axes are equal"""

    verts = mesh.verts_packed()
    x, y, z = verts.clone().detach().cpu().unbind(1)
    s = ax.scatter3D(x, y, z, c=colour, label=label, alpha=0.3)

    if equalize:
        equal_3d_axes(ax, x, y, z, zoom=zoom)

    return s, (x, y, z)


def plot_pointclouds(target_meshes, src_meshes, mesh_names=[], title="", figtitle="",
                     out_dir="static_fits_output/pointclouds"):
    """Plot and save fig of point clouds, with 3 figs side by side:
    [target mesh, src_mesh, both]"""

    for n in range(len(target_meshes)):
        fig = plt.figure(figsize=(15, 5))
        axes = [fig.add_subplot(int(f"13{n}"), projection="3d") for n in range(1, 4)]

        for ax in axes:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        colours = ["green", "blue"]
        labels = ["target", "SMAL"]
        for i, mesh in enumerate([target_meshes[n], src_meshes[n]]):
            for ax in [axes[1 + i == 1], axes[2]]:
                plot_pointcloud(ax, mesh, colour=colours[i], label=labels[i])

        fig.suptitle(figtitle)
        for ax in axes:
            ax.legend()

        if not mesh_names:
            name = n
        else:
            name = mesh_names[n]

        try_mkdir(out_dir)

        plt.legend(loc='upper right')
        plt.savefig(
            f"{out_dir}/{name} - {title}.png")
        plt.close(fig)


def cartesian_rotation(dim="x", rot=0):
    """Given a cartesian direction of rotation, and a rotation in radians, returns Tensor rotation matrix"""

    i = "xyz".find(dim)
    R = torch.eye(3)
    if rot != 0:
        j, k = (i + 1) % 3, (i + 2) % 3  # other two of cyclic triplet
        R[j, j] = R[k, k] = np.cos(rot)
        R[j, k] = - np.sin(rot)
        R[k, j] = np.sin(rot)

    return R


def stack_as_batch(tensor: torch.Tensor, n_repeats=1, dim=0) -> torch.Tensor:
    """Inserts new dim dimension, and stacks tensor n times along that dimension"""
    res = tensor.unsqueeze(dim)
    repeats = [1] * res.ndim
    repeats[dim] = n_repeats  # repeat across target dimension
    res = res.repeat(*repeats)
    return res


def animator(ax):
    """Wrapper used for animating meshes.
    - Clears all current trisurfs
    - Runs func, which returns new meshes
    - Plot these meshes.

    func must contain at least verts, faces"""

    def wrapper(func):
        # aux is wrapper function sent to wrap around existing anim
        def aux(frame):
            [child.remove() for child in ax.get_children() if isinstance(child, Poly3DCollection)]
            kwargs = func(frame)
            assert "mesh" in kwargs, "anim function must return 'mesh' object"
            plot_mesh(ax, **kwargs)

        return aux

    return wrapper


def save_animation(fig, func, n_frames, fmt="gif", fps=15, title="output", callback=True, **kwargs):
    """Save matplotlib animation."""

    arap_utils.save_animation(fig, func, n_frames, fmt=fmt, fps=fps, title=title, callback=callback, **kwargs)


def load_meshes(mesh_dir=None, mesh_files=None, sorting=lambda arr: arr, n_meshes=None, frame_step=1, device="cuda:0"):
    """Given a dir of .obj files or a list of .obj files, loads all and returns mesh names, and meshes as Mesh object.

    :param mesh_dir: Location of directory of .obj files (mutually exclusive with mesh_files)
    :param mesh_files: List of .obj file paths (mutually exclusive with mesh_dir)
    :param sorting: Optional lambda function to sort listdir (only used with mesh_dir)
    :param n_meshes: Optional slice of first n_meshes in sorted dir
    :param frame_step: For animations, optional step through sorted dir (only used with mesh_dir)
    :param device: torch device

    :returns mesh_names: list of all names of mesh files loaded
    :returns target_meshes: PyTorch3D Meshes object of all loaded meshes
    """

    # load all meshes
    mesh_names = []
    all_verts, all_faces_idx = [], []

    if mesh_dir is not None and mesh_files is not None:
        raise ValueError("Cannot specify both mesh_dir and mesh_files")
    elif mesh_dir is not None:
        file_list = [f for f in os.listdir(mesh_dir) if ".obj" in f]
        obj_list = sorting(file_list)[::frame_step]  # get sorted list of obj files, applying frame step
        if n_meshes is not None:
            obj_list = obj_list[:n_meshes]
        obj_list = [os.path.join(mesh_dir, obj_file) for obj_file in obj_list]
    elif mesh_files is not None:
        obj_list = mesh_files
    else:
        raise ValueError("Must specify either mesh_dir or mesh_files")

    for obj_file in obj_list:
        mesh_names.append(os.path.basename(obj_file)[:-4])  # Get name of mesh without .obj extension
        verts, faces, aux = load_obj(obj_file, load_textures=False)  # Load mesh with no textures
        faces_idx = faces.verts_idx.to(device)  # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
        verts = verts.to(device)  # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh

        # Center and scale for normalisation purposes
        centre = verts.mean(0)
        verts = verts - centre
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale

        # ROTATE TARGET MESH TO GET IN DESIRED DIRECTION - UNCOMMENT IF USING
        # R1 = cartesian_rotation("z", np.pi/2).to(device)
        # R2 = cartesian_rotation("y", np.pi/2).to(device)
        # verts = torch.mm(verts, R1.T)
        # verts = torch.mm(verts, R2.T)

        all_verts.append(verts), all_faces_idx.append(faces_idx)

    print(f"{len(all_verts)} target meshes loaded.")

    target_meshes = Meshes(verts=all_verts, faces=all_faces_idx)  # All loaded target meshes together

    return mesh_names, target_meshes


def compute_thinness_scores(mesh: Meshes, n_neighbors=50, max_faces_per_batch=5000):
    """
    Compute a 'thinness' score for each face in the mesh.
    
    The thinness score is based on the variation of normal directions in the neighborhood
    of each face. High variation indicates thin or high-curvature regions.
    
    Args:
        mesh: PyTorch3D Meshes object
        n_neighbors: Number of neighbors to consider for each face
        max_faces_per_batch: Maximum number of faces to process at once to avoid OOM errors
        
    Returns:
        scores: Tensor of shape [batch_size, num_faces] with thinness scores
    """
    # Get face normals
    face_normals = mesh.faces_normals_padded()  # [batch_size, num_faces, 3]
    
    batch_size = face_normals.shape[0]
    scores = []
    
    for b in range(batch_size):
        # Get face centers for distance calculation
        verts = mesh.verts_padded()[b]
        faces = mesh.faces_padded()[b]
        face_verts_idx = faces
        face_verts = torch.index_select(verts, 0, face_verts_idx.reshape(-1)).reshape(-1, 3, 3)
        face_centers = face_verts.mean(dim=1)  # [num_faces, 3]
        
        # Get normals of the current batch
        normals = face_normals[b]  # [num_faces, 3]
        
        # Normalize normals to ensure dot products are in [-1, 1]
        normals = torch.nn.functional.normalize(normals, dim=1)
        
        # Process in batches to avoid OOM
        num_faces = face_centers.shape[0]
        face_variations = []
        
        # If we have a small number of faces, process all at once
        if num_faces <= max_faces_per_batch:
            # Compute pairwise distances between face centers
            face_dists = torch.cdist(face_centers, face_centers)
            
            # Set diagonal to inf to exclude self
            face_dists.fill_diagonal_(float('inf'))
            
            # Get top-k nearest neighbors for all faces at once
            k = min(n_neighbors, num_faces-1)
            _, nn_idx = torch.topk(face_dists, k, dim=1, largest=False)
            
            # Gather neighbor normals for all faces - [num_faces, n_neighbors, 3]
            neighbor_normals = normals[nn_idx]
            
            # Reshape current face normals for broadcasting - [num_faces, 1, 3]
            normals_expanded = normals.unsqueeze(1)
            
            # Compute dot products between each face normal and its neighbors - [num_faces, n_neighbors]
            dot_products = torch.sum(neighbor_normals * normals_expanded, dim=2)
            
            # Clamp to avoid numerical issues with acos
            dot_products = torch.clamp(dot_products, -0.999, 0.999)
            
            # Convert to angles - [num_faces, n_neighbors]
            angles = torch.acos(dot_products)
            
            # Compute variation (standard deviation) for each face - [num_faces]
            variation = torch.std(angles, dim=1)
            face_variations.append(variation)
        else:
            # Process in batches
            for i in range(0, num_faces, max_faces_per_batch):
                end_idx = min(i + max_faces_per_batch, num_faces)
                batch_centers = face_centers[i:end_idx]
                batch_normals = normals[i:end_idx]
                
                # For each face in the batch, find its nearest neighbors
                batch_variations = []
                
                for j in range(batch_centers.shape[0]):
                    # Compute distances from this face to all other faces
                    dists = torch.norm(face_centers - batch_centers[j:j+1], dim=1)
                    
                    # Set distance to self to infinity
                    dists[i+j] = float('inf')
                    
                    # Get indices of k nearest neighbors
                    k = min(n_neighbors, num_faces-1)
                    _, nn_indices = torch.topk(dists, k, largest=False)
                    
                    # Get normals of neighbors
                    nn_normals = normals[nn_indices]
                    
                    # Compute dot products with neighbors
                    dot_products = torch.sum(nn_normals * batch_normals[j:j+1], dim=1)
                    
                    # Clamp to avoid numerical issues
                    dot_products = torch.clamp(dot_products, -0.999, 0.999)
                    
                    # Convert to angles
                    angles = torch.acos(dot_products)
                    
                    # Compute variation
                    variation = torch.std(angles)
                    batch_variations.append(variation)
                
                # Combine batch variations
                face_variations.append(torch.stack(batch_variations))
        
        # Combine all variations
        variation = torch.cat(face_variations)
        
        # Normalize to [0, 1] range for easier visualization
        if variation.max() > variation.min():
            normalized_variation = (variation - variation.min()) / (variation.max() - variation.min())
        else:
            normalized_variation = torch.zeros_like(variation)
            
        scores.append(normalized_variation)
    
    return torch.stack(scores)


def plot_mesh_normals_high_res(mesh: Meshes, output_path, title="Surface Normals", 
                              num_samples=1000, normal_length=0.05, normal_color="red",
                              mesh_color="blue", mesh_alpha=0.3, figsize=(10, 10), dpi=300,
                              color_by_thinness=True, n_neighbors_thinness=50):
    """
    Create a high-resolution plot of mesh surface normals.
    
    This function visualizes a mesh with its surface normals as arrows originating from
    the face centers. It's useful for analyzing the orientation and quality of mesh surfaces.
    
    Args:
        mesh (Meshes): PyTorch3D Meshes object to visualize
        output_path (str): Path where the output image will be saved
        title (str): Title for the plot
        num_samples (int): Number of face normals to randomly sample and plot.
                           If -1 or greater than the number of faces, all normals will be plotted
        normal_length (float): Length scaling factor for the normal vectors
        normal_color (str): Color of the normal vectors
        mesh_color (str): Color of the mesh
        mesh_alpha (float): Transparency of the mesh (0.0 to 1.0)
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): DPI for the output image
        color_by_thinness (bool): Whether to color normals by thinness (red=thin, green=thick)
        n_neighbors_thinness (int): Number of neighbors to consider for thinness calculation
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    # Set axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # Plot the mesh with transparency
    plot_mesh(ax, mesh, colour=mesh_color, alpha=mesh_alpha, equalize=True, label="Mesh")
    
    # Get mesh data
    verts = mesh.verts_padded()
    faces = mesh.faces_padded()
    
    # Calculate face normals
    mesh_normals = mesh.faces_normals_padded()  # [batch_size, num_faces, 3]
    
    # Calculate face centers manually
    face_verts_idx = faces[0]  # Use first batch
    face_verts = torch.index_select(verts[0], 0, face_verts_idx.reshape(-1)).reshape(-1, 3, 3)
    face_centers = face_verts.mean(dim=1)  # [num_faces, 3]
    
    # Convert to numpy for matplotlib
    centers_np = face_centers.detach().cpu().numpy()
    normals_np = mesh_normals[0].detach().cpu().numpy()  # Use first batch
    
    # Sample normals or use all
    num_faces = centers_np.shape[0]
    if num_samples == -1 or num_samples >= num_faces:
        # Use all normals
        indices = np.arange(num_faces)
    else:
        # Sample a subset of normals
        indices = np.random.choice(num_faces, num_samples, replace=False)
    
    # Compute thinness scores if needed
    if color_by_thinness:
        thinness_scores = compute_thinness_scores(mesh, n_neighbors=n_neighbors_thinness)
        thinness_np = thinness_scores[0].detach().cpu().numpy()  # Use first batch
        
        # Create a colormap from green (0) to red (1)
        # For the sampled indices
        sampled_thinness = thinness_np[indices]
        
        # Create colors array: [r, g, b] where r increases and g decreases with thinness
        colors = np.zeros((len(indices), 3))
        colors[:, 0] = sampled_thinness  # Red channel increases with thinness
        colors[:, 1] = 1 - sampled_thinness  # Green channel decreases with thinness
        
        # Create a custom colormap for the legend
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "thinness_cmap", [(0, 'green'), (1, 'red')])
        
        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
    else:
        # Use a single color for all normals
        colors = normal_color
    
    # Plot the normals
    X, Y, Z = centers_np[indices, 0], centers_np[indices, 1], centers_np[indices, 2]
    U, V, W = normals_np[indices, 0], normals_np[indices, 1], normals_np[indices, 2]
    
    # Scale the normals
    U = U * normal_length
    V = V * normal_length
    W = W * normal_length
    
    # Plot the quivers with colors
    quiver = ax.quiver(X, Y, Z, U, V, W, colors=colors, length=normal_length, normalize=True)
    
    # Add a colorbar if coloring by thinness
    if color_by_thinness:
        cbar = fig.colorbar(sm, ax=ax, label='Thinness Score')
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Thick', 'Medium', 'Thin'])
        
        # Add a labeled artist for the normals to include in the legend
        ax.plot([], [], color='red', label=f"Thin Regions")
        ax.plot([], [], color='green', label=f"Thick Regions")
    else:
        # Add a labeled artist for the normals to include in the legend
        ax.plot([], [], color=normal_color, label=f"Normals ({len(indices)} of {num_faces})")
    
    # Set title and adjust view
    ax.set_title(title)
    
    # Add legend with the labeled artists
    ax.legend(loc='upper right')
    
    # Ensure directory exists
    out_dir = os.path.dirname(output_path)
    try_mkdir(out_dir)
    
    # Save high-resolution figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_mesh_normals_comparison(target_mesh, src_mesh, output_path, title="Surface Normals Comparison", 
                               num_samples=500, normal_length=0.05, figsize=(15, 5), dpi=300,
                               color_by_thinness=True, n_neighbors_thinness=30):
    """
    Create a high-resolution comparison plot of surface normals for target and source meshes.
    
    This function creates a three-panel visualization:
    1. Left panel: Target mesh with its normals (in red)
    2. Middle panel: Source mesh with its normals (in red)
    3. Right panel: Both meshes overlaid with their respective normals colored to match the mesh
    
    This visualization is particularly useful for comparing the quality of mesh fitting
    by examining how well the normals of the source mesh align with those of the target.
    
    Args:
        target_mesh (Meshes): Target PyTorch3D Meshes object
        src_mesh (Meshes): Source PyTorch3D Meshes object
        output_path (str): Path where the output image will be saved
        title (str): Title for the overall plot
        num_samples (int): Number of face normals to randomly sample and plot.
                           If -1 or greater than the number of faces, all normals will be plotted
        normal_length (float): Length scaling factor for the normal vectors
        figsize (tuple): Figure size (width, height) in inches
        dpi (int): DPI for the output image
        color_by_thinness (bool): Whether to color normals by thinness
        n_neighbors_thinness (int): Number of neighbors to consider for thinness calculation
    """
    # Reduce the number of neighbors for thinness calculation to save memory
    n_neighbors_thinness = min(n_neighbors_thinness, 30)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    
    # Set axis labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    # Set subplot titles
    ax1.set_title("Target Mesh Normals")
    ax2.set_title("Source Mesh Normals")
    ax3.set_title("Overlaid Mesh Normals")
    
    # Define mesh colors
    target_color = "green"
    src_color = "blue"
    
    # Compute thinness scores if needed (with reduced sample size)
    if color_by_thinness:
        # Use a smaller sample for visualization
        target_thinness = compute_thinness_scores(target_mesh, n_neighbors=n_neighbors_thinness, max_faces_per_batch=200)
        src_thinness = compute_thinness_scores(src_mesh, n_neighbors=n_neighbors_thinness, max_faces_per_batch=200)
        
        # Create a colormap from green (0) to red (1)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "thinness_cmap", [(0, 'green'), (1, 'red')])
        
        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
    
    # Process each mesh for individual plots
    for ax, mesh, color, label, thinness_scores in [
        (ax1, target_mesh, target_color, "Target", target_thinness if color_by_thinness else None), 
        (ax2, src_mesh, src_color, "SMAL", src_thinness if color_by_thinness else None)
    ]:
        # Plot the mesh with transparency
        plot_mesh(ax, mesh, colour=color, alpha=0.3, equalize=True, label=label)
        
        # Get mesh data
        verts = mesh.verts_padded()
        faces = mesh.faces_padded()
        
        # Calculate face normals
        mesh_normals = mesh.faces_normals_padded()
        
        # Calculate face centers manually
        face_verts_idx = faces[0]  # Use first batch
        face_verts = torch.index_select(verts[0], 0, face_verts_idx.reshape(-1)).reshape(-1, 3, 3)
        face_centers = face_verts.mean(dim=1)
        
        # Convert to numpy for matplotlib
        centers_np = face_centers.detach().cpu().numpy()
        normals_np = mesh_normals[0].detach().cpu().numpy()
        
        # Sample normals or use all
        num_faces = centers_np.shape[0]
        if num_samples == -1 or num_samples >= num_faces:
            # Use all normals
            indices = np.arange(num_faces)
        else:
            # Sample a subset of normals
            indices = np.random.choice(num_faces, num_samples, replace=False)
        
        # Prepare colors based on thinness if needed
        if color_by_thinness:
            thinness_np = thinness_scores[0].detach().cpu().numpy()
            sampled_thinness = thinness_np[indices]
            
            # Create colors array: [r, g, b] where r increases and g decreases with thinness
            colors = np.zeros((len(indices), 3))
            colors[:, 0] = sampled_thinness  # Red channel increases with thinness
            colors[:, 1] = 1 - sampled_thinness  # Green channel decreases with thinness
        else:
            colors = "red"  # Default color if not coloring by thinness
        
        # Plot the normals
        X, Y, Z = centers_np[indices, 0], centers_np[indices, 1], centers_np[indices, 2]
        U, V, W = normals_np[indices, 0], normals_np[indices, 1], normals_np[indices, 2]
        
        # Scale the normals
        U = U * normal_length
        V = V * normal_length
        W = W * normal_length
        
        ax.quiver(X, Y, Z, U, V, W, colors=colors, length=normal_length, normalize=True)
        
        # Add legend
        if color_by_thinness:
            ax.plot([], [], color='red', label=f"Thin Regions")
            ax.plot([], [], color='green', label=f"Thick Regions")
        else:
            ax.plot([], [], color="red", label=f"Normals ({len(indices)} of {num_faces})")
        
        ax.legend(loc='upper right')
    
    # Add a colorbar if coloring by thinness (only need one for the figure)
    if color_by_thinness:
        cbar = fig.colorbar(sm, ax=[ax1, ax2, ax3], label='Thinness Score', shrink=0.6)
        cbar.set_ticks([0, 0.5, 1])
        cbar.set_ticklabels(['Thick', 'Medium', 'Thin'])
    
    # Plot both meshes in the third subplot with colored normals
    # First, plot both meshes with transparency
    plot_mesh(ax3, target_mesh, colour=target_color, alpha=0.2, equalize=True, label="Target")
    plot_mesh(ax3, src_mesh, colour=src_color, alpha=0.2, equalize=True, label="SMAL")
    
    # Get face counts for both meshes
    target_faces = target_mesh.faces_padded()[0].shape[0]
    src_faces = src_mesh.faces_padded()[0].shape[0]
    
    # Process each mesh for the overlaid plot
    for mesh, color, label, num_faces, thinness_scores in [
        (target_mesh, target_color, "Target", target_faces, target_thinness if color_by_thinness else None), 
        (src_mesh, src_color, "SMAL", src_faces, src_thinness if color_by_thinness else None)
    ]:
        # Get mesh data
        verts = mesh.verts_padded()
        faces = mesh.faces_padded()
        
        # Calculate face normals
        mesh_normals = mesh.faces_normals_padded()
        
        # Calculate face centers manually
        face_verts_idx = faces[0]  # Use first batch
        face_verts = torch.index_select(verts[0], 0, face_verts_idx.reshape(-1)).reshape(-1, 3, 3)
        face_centers = face_verts.mean(dim=1)
        
        # Convert to numpy for matplotlib
        centers_np = face_centers.detach().cpu().numpy()
        normals_np = mesh_normals[0].detach().cpu().numpy()
        
        # Sample normals or use all
        if num_samples == -1:
            # Use all normals
            indices = np.arange(num_faces)
        else:
            # Sample a subset of normals, half for each mesh in the overlay
            sample_size = min(num_faces, num_samples // 2)
            indices = np.random.choice(num_faces, sample_size, replace=False)
        
        # Prepare colors based on thinness if needed
        if color_by_thinness:
            thinness_np = thinness_scores[0].detach().cpu().numpy()
            sampled_thinness = thinness_np[indices]
            
            # Create colors array: [r, g, b] where r increases and g decreases with thinness
            colors = np.zeros((len(indices), 3))
            colors[:, 0] = sampled_thinness  # Red channel increases with thinness
            colors[:, 1] = 1 - sampled_thinness  # Green channel decreases with thinness
            
            # Add a slight tint based on the mesh color to differentiate target from source
            if color == target_color:  # Target mesh (green)
                colors[:, 2] = 0.3  # Add some blue to make it distinguishable
            else:  # Source mesh (blue)
                colors[:, 2] = 0.7  # Add more blue to make it distinguishable
        else:
            colors = color
        
        # Plot the normals
        X, Y, Z = centers_np[indices, 0], centers_np[indices, 1], centers_np[indices, 2]
        U, V, W = normals_np[indices, 0], normals_np[indices, 1], normals_np[indices, 2]
        
        # Scale the normals
        U = U * normal_length
        V = V * normal_length
        W = W * normal_length
        
        ax3.quiver(X, Y, Z, U, V, W, colors=colors, length=normal_length, normalize=True)
        
        # Add legend entries
        if color_by_thinness:
            if color == target_color:
                ax3.plot([], [], color=[1, 0, 0.3], label=f"Target Thin")
                ax3.plot([], [], color=[0, 1, 0.3], label=f"Target Thick")
            else:
                ax3.plot([], [], color=[1, 0, 0.7], label=f"SMAL Thin")
                ax3.plot([], [], color=[0, 1, 0.7], label=f"SMAL Thick")
        else:
            ax3.plot([], [], color=color, label=f"{label} Normals ({len(indices)} of {num_faces})")
    
    # Add legend to the third subplot
    ax3.legend(loc='upper right')
    
    # Set main title
    fig.suptitle(title, fontsize=16)
    
    # Ensure directory exists
    out_dir = os.path.dirname(output_path)
    try_mkdir(out_dir)
    
    # Save high-resolution figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
