import os
import torch
from multiprocessing import Pool, cpu_count, set_start_method
from typing import Union, Tuple
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals

from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments

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
        os.makedirs(loc, exist_ok=True)


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


def plot_vertex_heatmap(points, values, title="SDF Loss Contribution", output_path="sdf_loss_heatmap.png", 
                     figsize=(10, 10), dpi=300, target_points=None):
    """
    Creates a 3D visualization of points with colors based on their associated values.
    
    Args:
        points: Tensor or array of shape (P, 3) containing 3D points
        values: Tensor or array of shape (P,) containing values for each point
        title: Title for the plot
        output_path: Path to save the output image
        figsize: Size of the figurei
        dpi: DPI for the output image
        target_points: Optional tensor or array of shape (Q, 3) containing target points to plot with low opacity
    """
    # Convert to numpy if necessary
    if torch.is_tensor(points):
        points = points.numpy()
    if torch.is_tensor(values):
        values = values.numpy()
    if target_points is not None and torch.is_tensor(target_points):
        target_points = target_points.numpy()
    
    # Create figure with 3D projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colormap: grey for low values, red for high values
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "sdf_diff_cmap", [(0.8, 0.8, 0.8), (1, 0, 0)])
    
    # Ensure we have valid points and values
    if len(points) == 0:
        print(f"Warning: No points to visualize for {title}")
        plt.close(fig)
        return
    
    # Plot target points first (if provided) with very low opacity
    if target_points is not None and len(target_points) > 0:
        ax.scatter(
            target_points[:, 0], target_points[:, 1], target_points[:, 2],
            color='lightblue',
            s=10,  # Smaller size
            alpha=0.1,  # Very low opacity
            label='Target points'
        )
    
    # Plot points colored by their values
    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=values,
        cmap=cmap,
        s=20,  # Point size
        alpha=0.8,
        vmin=0.0,
        vmax=1.0,
        label='Source points'
    )
    
    # Add colorbar
    cbar = fig.colorbar(sc, ax=ax, label='Normalized SDF Difference')
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0 (Low)', '0.25', '0.5', '0.75', '1.0 (High)'])
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add legend
    ax.legend()
    
    # Ensure equal scale for all axes using equal_3d_axes function
    # If we have target points, include them in scale calculation
    if target_points is not None and len(target_points) > 0:
        all_x = np.concatenate([points[:, 0], target_points[:, 0]])
        all_y = np.concatenate([points[:, 1], target_points[:, 1]])
        all_z = np.concatenate([points[:, 2], target_points[:, 2]])
        equal_3d_axes(ax, all_x, all_y, all_z, zoom=1.0)
    else:
        equal_3d_axes(ax, points[:, 0], points[:, 1], points[:, 2], zoom=1.0)
    
    # Get rid of colored axes planes and gray panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # Remove grid
    ax.grid(False)
    
    # Save figure
    out_dir = os.path.dirname(output_path)
    try_mkdir(out_dir)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _SDF_distance_single_direction(
    x: torch.Tensor,
    y: torch.Tensor,
    x_sdf: torch.Tensor,
    y_sdf: torch.Tensor,
    x_lengths: torch.Tensor,
    y_lengths: torch.Tensor,
    k: int,
    batch_reduction: Union[str, None],
    point_reduction: Union[str, None],
    norm: int,
    visualize: bool = False,
    output_dir: str = "sdf_visualization",
    title: str = "sdf_loss_contribution",
    mesh_names: list = None,
    normalize_sdf: bool = True
):
    """
    Compute the SDF distance in a single direction (x to y).

    Args:
        x: FloatTensor of shape (N, P1, 3)
        y: FloatTensor of shape (N, P2, 3)
        x_sdf: FloatTensor of shape (N, P1)
        y_sdf: FloatTensor of shape (N, P2)
        x_lengths: LongTensor of shape (N,) giving the number of points in each cloud in x
        y_lengths: LongTensor of shape (N,) giving the number of points in each cloud in y
        k: Number of nearest neighbors to consider
        batch_reduction: Reduction operation to apply for the loss across the batch
        point_reduction: Reduction operation to apply for the loss across the points
        norm: int indicates the norm used for the distance
        visualize: Whether to generate visualization of per-vertex SDF loss contributions
        output_dir: Directory to save visualization plots
        title: Base title for the visualization plots
        mesh_names: List of names for each mesh in the batch
        normalize_sdf: Whether to apply Z-score normalization to SDF values

    Returns:
        Tensor giving the reduced SDF distance between the pointclouds in x and y
    """
    N, P1, D = x.shape
    
    # Find k nearest neighbors
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=k)
    
    # Get the indices of k nearest neighbors for each point in x
    nn_idx = x_nn.idx  # Shape: (N, P1, k)
    
    # Get the distances to k nearest neighbors
    nn_dists = x_nn.dists  # Shape: (N, P1, k)
    
    # Apply Z-score normalization to SDF values if requested
    # This makes SDF values comparable across different scales by standardizing distributions
    if normalize_sdf:
        # Create empty tensors to hold normalized SDF values
        x_sdf_normalized = torch.zeros_like(x_sdf)
        y_sdf_normalized = torch.zeros_like(y_sdf)
        
        # Normalize each batch separately to handle varying distributions
        for batch_idx in range(N):
            # Get valid points for this batch
            x_valid = x_sdf[batch_idx, :x_lengths[batch_idx]]
            y_valid = y_sdf[batch_idx, :y_lengths[batch_idx]]
            
            # Compute statistics for normalization
            x_mean = x_valid.mean()
            x_std = x_valid.std()
            y_mean = y_valid.mean()
            y_std = y_valid.std()
            
            # Prevent division by zero with small epsilon
            eps = 1e-8
            x_std = torch.clamp(x_std, min=eps)
            y_std = torch.clamp(y_std, min=eps)
            
            # Apply Z-score normalization: (value - mean) / std
            # This centers the distribution at 0 with standard deviation of 1
            x_sdf_normalized[batch_idx, :x_lengths[batch_idx]] = (x_valid - x_mean) / x_std
            y_sdf_normalized[batch_idx, :y_lengths[batch_idx]] = (y_valid - y_mean) / y_std
        
        # Use normalized SDF values for subsequent computations
        x_sdf = x_sdf_normalized
        y_sdf = y_sdf_normalized
    
    # Expand x_sdf for broadcasting
    x_sdf_expanded = x_sdf.unsqueeze(-1)  # Shape: (N, P1, 1)
    
    # Gather y_sdf values for the k nearest neighbors
    y_sdf_nn = torch.gather(y_sdf, 1, nn_idx.reshape(N, -1)).reshape(N, P1, k)  # Shape: (N, P1, k)
    
    # Compute absolute differences between SDF values
    sdf_diffs = torch.abs(x_sdf_expanded - y_sdf_nn)  # Shape: (N, P1, k)
    
    # Instead of using argmin (non-differentiable), use softmax with negative values (differentiable)
    # This creates a soft selection that will focus primarily on the minimum difference
    # but maintains gradient flow through all neighbors
    temperature = 0.1  # Controls how "sharp" the selection is (lower = closer to argmin)
    weights = torch.softmax(-sdf_diffs / temperature, dim=-1)  # Shape: (N, P1, k)
    
    # Weighted sum of distances
    sdf_dist = (weights * nn_dists).sum(dim=-1)  # Shape: (N, P1)
    
    # Generate visualizations if requested
    if visualize:
        try_mkdir(output_dir)
        
        # For visualization, let's compute the mean SDF difference for each point
        # We'll use the weights to get a weighted average of differences
        mean_sdf_diff = (weights * sdf_diffs).sum(dim=-1)  # Shape: (N, P1)
        
        # Generate visualization for each batch
        for batch_idx in range(N):
            # Get mesh name or use batch index
            if mesh_names is not None and batch_idx < len(mesh_names):
                mesh_name = mesh_names[batch_idx]
            else:
                mesh_name = f"batch{batch_idx}"
                
            # Normalize to [0, 1] range for visualization
            batch_diffs = mean_sdf_diff[batch_idx]
            max_diff = batch_diffs.max().item()
            if max_diff > 0:
                normalized_diffs = batch_diffs / max_diff
            else:
                normalized_diffs = torch.zeros_like(batch_diffs)
            
            # Get target points for this batch (up to y_lengths)
            target_points = y[batch_idx, :y_lengths[batch_idx]].detach().cpu()
            
            # Create visualization plot with mesh name at beginning of filename
            plot_vertex_heatmap(
                points=x[batch_idx, :x_lengths[batch_idx]].detach().cpu(),
                values=normalized_diffs[:x_lengths[batch_idx]].detach().cpu(),
                title=f"{mesh_name} - {title}",
                output_path=os.path.join(output_dir, f"{mesh_name}_{title}.png"),
                target_points=target_points
            )
    
    # Apply point reduction if specified
    if point_reduction is not None:
        sdf_dist = sdf_dist.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            sdf_dist /= x_lengths_clamped
            
        if batch_reduction is not None:
            # batch_reduction == "sum"
            sdf_dist = sdf_dist.sum()
            if batch_reduction == "mean":
                sdf_dist /= max(N, 1)
    
    return sdf_dist


def SDF_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    x_sdf: torch.Tensor,
    y_sdf: torch.Tensor,
    k: int,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    visualize: bool = False,
    output_dir: str = "sdf_visualization",
    title: str = "sdf_loss_contribution",
    mesh_names: list = None
):
    """
    Spatial Diameter Function (SDF) distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, 3) representing a batch of point clouds
           from the source mesh vertices with at most P1 points in each batch element,
           batch size N and 3 feature dimensions.
        y: FloatTensor of shape (N, P2, 3) representing a batch of point clouds
           from the target mesh vertices with at most P2 points in each batch element,
           batch size N and 3 feature dimensions.
        x_sdf: FloatTensor of shape (N, P1) containing the normalized spatial diameter
               values for each vertex in x.
        y_sdf: FloatTensor of shape (N, P2) containing the normalized spatial diameter
               values for each vertex in y.
        k: Number of nearest neighbors to consider when sampling the local SDF field.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2 (euclidean).
        single_directional: If False (default), loss comes from both directions.
            If True, loss is only computed from x to y.
        visualize: Whether to generate visualization of per-vertex SDF loss contributions
        output_dir: Directory to save visualization plots
        title: Base title for the visualization plots
        mesh_names: List of names for each mesh in the batch

    Returns:
        Tensor giving the reduced SDF distance between the pointclouds in x and y.
        If point_reduction is None, returns a 2-element tuple of Tensors containing
        forward and backward loss terms shaped (N, P1) and (N, P2) (if single_directional
        is False) or a Tensor containing loss terms shaped (N, P1) (if single_directional
        is True).
    """
    # Validate inputs
    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected x and y to be of shape (N, P, 3), got shapes {x.shape} and {y.shape}")
    
    if x_sdf.ndim != 2 or y_sdf.ndim != 2:
        raise ValueError(f"Expected x_sdf and y_sdf to be of shape (N, P), got shapes {x_sdf.shape} and {y_sdf.shape}")
    
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Batch sizes must be equal, got {x.shape[0]} and {y.shape[0]}")
    
    if x.shape[0] != x_sdf.shape[0] or y.shape[0] != y_sdf.shape[0]:
        raise ValueError(f"Batch sizes must match between points and SDF values, got {x.shape[0]}, {x_sdf.shape[0]} for x and {y.shape[0]}, {y_sdf.shape[0]} for y")
    
    if x.shape[1] != x_sdf.shape[1] or y.shape[1] != y_sdf.shape[1]:
        raise ValueError(f"Number of points must match number of SDF values, got {x.shape[1]}, {x_sdf.shape[1]} for x and {y.shape[1]}, {y_sdf.shape[1]} for y")
    
    if k < 1:
        raise ValueError("k must be at least 1")
    
    # Get batch size and number of points
    N, P1, D = x.shape
    P2 = y.shape[1]
    
    # Create lengths tensors (assuming all points are valid)
    x_lengths = torch.full((N,), P1, dtype=torch.int64, device=x.device)
    y_lengths = torch.full((N,), P2, dtype=torch.int64, device=y.device)
    
    # Prepare visualization directory with forward and backward subdirectories if needed
    if visualize:
        try_mkdir(output_dir)
        forward_dir = os.path.join(output_dir, "forward")
        try_mkdir(forward_dir)
        if not single_directional:
            backward_dir = os.path.join(output_dir, "backward")
            try_mkdir(backward_dir)
    
    # Compute forward direction
    sdf_dist = _SDF_distance_single_direction(
        x, y, x_sdf, y_sdf, x_lengths, y_lengths, k,
        batch_reduction, point_reduction, norm,
        visualize=visualize,
        output_dir=os.path.join(output_dir, "forward") if visualize else output_dir,
        title=title,
        mesh_names=mesh_names
    )
    
    if single_directional:
        return sdf_dist
    
    # Compute reverse direction
    sdf_dist_rev = _SDF_distance_single_direction(
        y, x, y_sdf, x_sdf, y_lengths, x_lengths, k,
        batch_reduction, point_reduction, norm,
        visualize=visualize,
        output_dir=os.path.join(output_dir, "backward") if visualize else output_dir,
        title=f"{title}_backward",
        mesh_names=mesh_names
    )
    
    if point_reduction is not None:
        return sdf_dist + sdf_dist_rev
    
    return sdf_dist, sdf_dist_rev


def sample_points_from_meshes_and_SDF(
    meshes,
    sdf_values: Union[torch.Tensor, list],
    num_samples: int = 10000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of meshes to a batch of pointclouds by uniformly sampling
    vertices from the mesh and return their corresponding SDF values.
    This version preserves gradients to allow proper optimization.

    Args:
        meshes: A Meshes object with a batch of N meshes.
        sdf_values: FloatTensor of shape (N, V) or (V,) containing SDF values for each vertex
                   in the meshes, where V is the number of vertices per mesh.
                   If shape is (V,), it is assumed that all meshes share the same vertex count
                   and the SDF values are valid for all meshes.
                   Can also be a list of N tensors, each with shape (V_i,) where V_i is the
                   number of vertices in the i-th mesh.
        num_samples: Integer giving the number of point samples per mesh.

    Returns:
        2-element tuple containing:
        - samples: FloatTensor of shape (N, num_samples, 3) giving the
          coordinates of sampled points for each mesh in the batch.
        - sdf_samples: FloatTensor of shape (N, num_samples) giving the
          SDF values corresponding to the sampled points.
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")
    
    # Process and validate SDF values
    device = meshes.device
    num_meshes = len(meshes)

    # Convert SDF values to appropriate tensor format
    if isinstance(sdf_values, list):
        if len(sdf_values) != num_meshes:
            raise ValueError(f"Number of SDF value tensors ({len(sdf_values)}) must match number of meshes ({num_meshes})")
        
        # Ensure all SDF values are tensors on the correct device
        for i, sdf in enumerate(sdf_values):
            if not torch.is_tensor(sdf):
                sdf_values[i] = torch.tensor(sdf, device=device)
            elif sdf.device != device:
                sdf_values[i] = sdf.to(device)
                
        # We'll handle this case with per-mesh processing
        single_tensor_sdf = False
    else:
        # Single tensor of SDF values
        if not torch.is_tensor(sdf_values):
            raise TypeError("sdf_values must be either a list of tensors or a single tensor")
        
        # Move to correct device if needed
        if sdf_values.device != device:
            sdf_values = sdf_values.to(device)

        # Expand to batch dimension if needed
        if sdf_values.ndim == 1:
            # Single set of SDF values for all meshes
            verts_per_mesh = meshes.num_verts_per_mesh()
            if verts_per_mesh.numel() == 0:
                raise ValueError("Meshes object appears to be empty")
            
            # Check if all meshes have the same number of vertices
            first_vert_count = verts_per_mesh[0].item()
            same_verts = all(count == first_vert_count for count in verts_per_mesh.tolist())
            if not same_verts:
                raise ValueError("When providing a single 1D tensor of SDF values, all meshes must have the same number of vertices")
            
            # Check if the SDF values shape matches the vertex count
            if sdf_values.shape[0] != first_vert_count:
                raise ValueError(f"Number of SDF values ({sdf_values.shape[0]}) must match number of vertices per mesh ({first_vert_count})")
            
            # Expand to batch dimension
            sdf_values = sdf_values.unsqueeze(0).expand(num_meshes, -1)
        
        single_tensor_sdf = True
    
    # Initialize output tensors
    samples = torch.zeros((num_meshes, num_samples, 3), device=device)
    sdf_samples = torch.zeros((num_meshes, num_samples), device=device)
    
    # Get mesh vertex data
    mesh_to_vert_idx = meshes.mesh_to_verts_packed_first_idx()
    verts_per_mesh = meshes.num_verts_per_mesh()
    
    # Sample for each mesh in the batch
    for i in range(num_meshes):
        num_verts = verts_per_mesh[i].item()
        if num_verts <= 0:
            continue  # Skip empty meshes
        
        # Get start index for this mesh's vertices
        vert_offset = mesh_to_vert_idx[i].item()
        
        # Sample indices with replacement
        indices = torch.randint(0, num_verts, (num_samples,), device=device)
        
        # Get global indices into verts_packed
        global_indices = vert_offset + indices
        
        # Sample vertices
        samples[i] = verts[global_indices]
        
        # Sample SDF values
        if single_tensor_sdf:
            # Use the expanded tensor
            sdf_samples[i] = sdf_values[i][indices]
        else:
            # Use the list of per-mesh tensors
            if i < len(sdf_values):
                sdf = sdf_values[i]
                if indices.max() < len(sdf):
                    sdf_samples[i] = sdf[indices]
                else:
                    # Handle out-of-bounds indices safely
                    safe_indices = torch.clamp(indices, 0, len(sdf) - 1)
                    sdf_samples[i] = sdf[safe_indices]
    
    return samples, sdf_samples
