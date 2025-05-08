import torch
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
import time
import psutil
import GPUtil
from collections import defaultdict
import pickle as pkl

# Add the parent directory to the Python path to find config module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import try_mkdir, plot_mesh, equal_3d_axes

class PerformanceMonitor:
    def __init__(self):
        self.timings = defaultdict(float)
        self.counts = defaultdict(int)
        self.start_times = {}
        
    def start(self, section):
        self.start_times[section] = time.time()
        
    def end(self, section):
        if section in self.start_times:
            elapsed = time.time() - self.start_times[section]
            self.timings[section] += elapsed
            self.counts[section] += 1
            del self.start_times[section]
    
    def get_gpu_stats(self):
        try:
            gpu = GPUtil.getGPUs()[0]  # Get first GPU
            return {
                'gpu_util': gpu.load * 100,
                'gpu_memory': gpu.memoryUtil * 100
            }
        except:
            return {'gpu_util': 0, 'gpu_memory': 0}
    
    def get_cpu_stats(self):
        return {
            'cpu_util': psutil.cpu_percent(interval=None),
            'memory_util': psutil.virtual_memory().percent
        }
    
    def report(self):
        total_time = sum(self.timings.values())
        print("\nPerformance Report:")
        print("-" * 80)
        print(f"{'Section':<30} {'Time (s)':<10} {'Calls':<10} {'Avg (ms)':<10} {'%Total':<10}")
        print("-" * 80)
        
        for section in sorted(self.timings.keys()):
            time_sec = self.timings[section]
            calls = self.counts[section]
            avg_ms = (time_sec / calls) * 1000 if calls > 0 else 0
            percent = (time_sec / total_time) * 100 if total_time > 0 else 0
            
            print(f"{section:<30} {time_sec:>10.3f} {calls:>10} {avg_ms:>10.2f} {percent:>10.1f}")
        
        print("-" * 80)
        print(f"Total time: {total_time:.2f} seconds")

def compute_ray_triangle_intersection(origin, direction, v0, v1, v2, epsilon=1e-6):
    """
    Compute ray-triangle intersection using the Möller–Trumbore algorithm.
    
    Args:
        origin (torch.Tensor): [3] ray origin
        direction (torch.Tensor): [3] ray direction (normalized)
        v0, v1, v2 (torch.Tensor): [3] triangle vertices
        epsilon (float): Small value to prevent division by zero
        
    Returns:
        t (float or None): Distance to intersection point, or None if no intersection
        intersection (torch.Tensor or None): [3] Intersection point, or None if no intersection
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = torch.cross(direction, edge2)
    a = torch.dot(edge1, h)
    
    # If ray is parallel to triangle plane
    if abs(a) < epsilon:
        return None, None
        
    f = 1.0 / a
    s = origin - v0
    u = f * torch.dot(s, h)
    
    # If intersection is outside first edge
    if u < 0.0 or u > 1.0:
        return None, None
        
    q = torch.cross(s, edge1)
    v = f * torch.dot(direction, q)
    
    # If intersection is outside second edge
    if v < 0.0 or u + v > 1.0:
        return None, None
        
    # Compute t to find intersection point
    t = f * torch.dot(edge2, q)
    
    # If intersection is behind ray origin
    if t < epsilon:
        return None, None
        
    intersection = origin + t * direction
    return t, intersection

def compute_ray_mesh_intersections_vectorized(origin, direction, verts, faces, exclude_face_idx=None, exclude_radius=0.01):
    """
    Vectorized computation of ray-mesh intersections using the Möller-Trumbore algorithm.
    
    Args:
        origin (torch.Tensor): [3] ray origin
        direction (torch.Tensor): [3] ray direction (normalized)
        verts (torch.Tensor): [V, 3] mesh vertices
        faces (torch.Tensor): [F, 3] mesh faces
        exclude_face_idx (int, optional): Index of face to exclude from intersection tests
        exclude_radius (float): Radius around origin to exclude from intersection tests
        
    Returns:
        intersections (torch.Tensor): [N, 3] intersection points
        distances (torch.Tensor): [N] distances to intersections
    """
    device = origin.device
    
    # Get triangle vertices for all faces at once
    v0 = verts[faces[:, 0]]  # [F, 3]
    v1 = verts[faces[:, 1]]  # [F, 3]
    v2 = verts[faces[:, 2]]  # [F, 3]
    
    # If excluding a face, create mask
    if exclude_face_idx is not None:
        face_mask = torch.ones(len(faces), dtype=torch.bool, device=device)
        face_mask[exclude_face_idx] = False
        v0 = v0[face_mask]
        v1 = v1[face_mask]
        v2 = v2[face_mask]
    
    # Compute edges and normal for all triangles
    edge1 = v1 - v0  # [F, 3]
    edge2 = v2 - v0  # [F, 3]
    
    # Compute h = direction × edge2 for all triangles at once
    h = torch.cross(direction.unsqueeze(0).expand_as(edge2), edge2)  # [F, 3]
    
    # Compute a = edge1 · h for all triangles
    a = torch.sum(edge1 * h, dim=1)  # [F]
    
    # Filter out triangles parallel to ray
    valid_mask = torch.abs(a) > 1e-6
    
    if not valid_mask.any():
        return torch.zeros((0, 3), device=device), torch.zeros(0, device=device)
    
    # Apply mask to all tensors
    v0 = v0[valid_mask]
    edge1 = edge1[valid_mask]
    edge2 = edge2[valid_mask]
    h = h[valid_mask]
    a = a[valid_mask]
    
    # Compute f = 1/a
    f = 1.0 / a
    
    # Compute s = origin - v0 for all triangles
    s = origin.unsqueeze(0) - v0  # [F, 3]
    
    # Compute u = f * (s · h)
    u = f * torch.sum(s * h, dim=1)  # [F]
    
    # Filter out intersections outside first edge
    valid_mask = (u >= 0.0) & (u <= 1.0)
    
    if not valid_mask.any():
        return torch.zeros((0, 3), device=device), torch.zeros(0, device=device)
    
    # Apply mask to all tensors
    v0 = v0[valid_mask]
    edge1 = edge1[valid_mask]
    edge2 = edge2[valid_mask]
    s = s[valid_mask]
    f = f[valid_mask]
    u = u[valid_mask]
    
    # Compute q = s × edge1
    q = torch.cross(s, edge1)  # [F, 3]
    
    # Compute v = f * (direction · q)
    v = f * torch.sum(direction.unsqueeze(0) * q, dim=1)  # [F]
    
    # Filter out intersections outside second edge
    valid_mask = (v >= 0.0) & (u + v <= 1.0)
    
    if not valid_mask.any():
        return torch.zeros((0, 3), device=device), torch.zeros(0, device=device)
    
    # Apply mask to all tensors
    v0 = v0[valid_mask]
    edge2 = edge2[valid_mask]
    q = q[valid_mask]
    f = f[valid_mask]
    
    # Compute t = f * (edge2 · q)
    t = f * torch.sum(edge2 * q, dim=1)  # [F]
    
    # Filter out intersections behind ray origin
    valid_mask = t > exclude_radius
    
    if not valid_mask.any():
        return torch.zeros((0, 3), device=device), torch.zeros(0, device=device)
    
    # Compute final intersection points
    t = t[valid_mask]
    intersections = origin.unsqueeze(0) + direction.unsqueeze(0) * t.unsqueeze(1)  # [N, 3]
    
    return intersections, t

def generate_random_directions_batch(normals: torch.Tensor, num_rays: int, device: torch.device) -> torch.Tensor:
    """
    Generate random directions in batches for multiple normals.
    
    Args:
        normals (torch.Tensor): [B, 3] batch of normal vectors
        num_rays (int): Number of rays per normal
        device: torch device
        
    Returns:
        directions (torch.Tensor): [B, num_rays, 3] random directions in hemisphere
    """
    batch_size = len(normals)
    
    # Generate random directions for all rays at once
    directions = torch.randn(batch_size, num_rays, 3, device=device)
    directions = directions / torch.norm(directions, dim=2, keepdim=True)
    
    # Compute dot products with normals for all directions at once
    dots = torch.bmm(directions, -normals.unsqueeze(2)).squeeze(2)  # [B, num_rays]
    
    # For directions in wrong hemisphere, reflect them
    mask = dots < 0
    directions[mask] = -directions[mask]
    
    return directions

def compute_sdf(mesh: Meshes, num_samples: int = 1000, num_rays: int = 30):
    """
    Compute the Spatial Diameter Function for sampled points on the mesh.
    
    Args:
        mesh (Meshes): PyTorch3D mesh object
        num_samples (int): Number of surface points to sample. If -1, samples all faces.
        num_rays (int): Number of rays to cast per point
        
    Returns:
        sample_points (torch.Tensor): [N, 3] sampled surface points
        diameters (torch.Tensor): [N] computed diameter values
    """
    device = mesh.device
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    
    # Compute bounding box diagonal and thresholds
    bbox_min = torch.min(verts, dim=0)[0]
    bbox_max = torch.max(verts, dim=0)[0]
    bbox_diagonal = torch.norm(bbox_max - bbox_min)
    
    print(f"Bounding box min: {bbox_min}")
    print(f"Bounding box max: {bbox_max}")
    print(f"Bounding box diagonal: {bbox_diagonal}")
    
    min_distance_threshold = bbox_diagonal * 0.001
    max_distance_threshold = bbox_diagonal * 0.2
    ray_origin_offset = bbox_diagonal * 0.0001
    
    # Sample faces
    if num_samples == -1:
        # Use all faces
        num_samples = len(faces)
        sampled_face_idxs = torch.arange(num_samples, device=device)
        print(f"\nSampling all {num_samples} faces...")
    else:
        # Sample random faces with probability proportional to face area
        face_areas = mesh.faces_areas_packed()
        face_probs = face_areas / face_areas.sum()
        sampled_face_idxs = torch.multinomial(face_probs, num_samples, replacement=True)
    
    # Get face vertices for sampled faces
    face_verts = verts[faces[sampled_face_idxs]]  # [num_samples, 3, 3]
    
    # For all faces mode, use face centroids as sample points
    if num_samples == len(faces):
        sample_points = torch.mean(face_verts, dim=1)  # [num_samples, 3]
    else:
        # Sample random points within each triangle using barycentric coordinates
        w1 = torch.sqrt(torch.rand(num_samples, device=device))
        w2 = torch.rand(num_samples, device=device) * (1 - w1)
        w0 = 1 - w1 - w2
        
        # Compute sample points
        sample_points = (w0.unsqueeze(-1) * face_verts[:, 0] +
                        w1.unsqueeze(-1) * face_verts[:, 1] +
                        w2.unsqueeze(-1) * face_verts[:, 2])  # [num_samples, 3]
    
    # Compute face normals for sampled faces
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / torch.norm(face_normals, dim=1, keepdim=True)
    
    # Initialize diameter values
    diameters = torch.zeros(num_samples, device=device)
    
    # Process in batches to avoid OOM for large meshes
    batch_size = 1000  # Adjust based on available GPU memory
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"\nProcessing {num_samples} faces in {num_batches} batches of size {batch_size}...")
    
    for batch in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_samples)
        batch_size_actual = end_idx - start_idx
        
        # Get batch data
        batch_normals = face_normals[start_idx:end_idx]  # [B, 3]
        batch_points = sample_points[start_idx:end_idx]   # [B, 3]
        batch_face_idxs = sampled_face_idxs[start_idx:end_idx]  # [B]
        
        # Generate all rays for the batch at once
        batch_directions = generate_random_directions_batch(batch_normals, num_rays, device)  # [B, num_rays, 3]
        
        # Offset origins for all points in batch
        batch_origins = batch_points.unsqueeze(1) + batch_normals.unsqueeze(1) * ray_origin_offset  # [B, 1, 3]
        
        # Process each point in batch
        for i in range(batch_size_actual):
            point_directions = batch_directions[i]  # [num_rays, 3]
            point_origin = batch_origins[i, 0]     # [3]
            
            # Initialize distances for this point
            valid_distances = []
            found_valid = False
            
            # Process rays until we find enough valid intersections or try all rays
            for direction in point_directions:
                if found_valid and len(valid_distances) >= max(int(num_samples/2), 1):  # Early termination if we have enough valid samples
                    break
                    
                intersections, distances = compute_ray_mesh_intersections_vectorized(
                    point_origin, direction, verts, faces,
                    exclude_face_idx=batch_face_idxs[i],
                    exclude_radius=ray_origin_offset)
                
                if len(distances) > 0:
                    # Get the furthest intersection for this ray
                    max_dist = torch.max(distances)
                    
                    # Check if distance is within valid range
                    if min_distance_threshold < max_dist < max_distance_threshold:
                        valid_distances.append(max_dist.item())
                        found_valid = True
            
            # Compute diameter for this point
            if valid_distances:
                diameters[start_idx + i] = torch.tensor(valid_distances, device=device).mean()
            else:
                diameters[start_idx + i] = min_distance_threshold
    
    return sample_points, diameters

def smooth_distances(points: torch.Tensor, distances: torch.Tensor, k: int = 100):
    """
    Smooth distances by averaging over k nearest neighbors.
    
    Args:
        points (torch.Tensor): [N, 3] sampled points
        distances (torch.Tensor): [N] distance values
        k (int): Number of nearest neighbors to consider
        
    Returns:
        smoothed_distances (torch.Tensor): [N] smoothed distance values
    """
    # Convert to numpy for KDTree
    points_np = points.cpu().numpy()
    distances_np = distances.cpu().numpy()
    
    # Build KD-tree
    tree = cKDTree(points_np)
    
    # Find k nearest neighbors for each point
    _, indices = tree.query(points_np, k=k)
    
    # Compute average distance for each point using its neighbors
    smoothed_distances = np.zeros_like(distances_np)
    for i in range(len(points_np)):
        neighbor_distances = distances_np[indices[i]]
        smoothed_distances[i] = np.mean(neighbor_distances)
    
    return torch.from_numpy(smoothed_distances).to(distances.device)

def visualize_sdf(mesh: Meshes, sample_points: torch.Tensor, diameters: torch.Tensor, 
                  output_path: str, title: str = "Spatial Diameter Function"):
    """
    Visualize the mesh with points colored by their SDF values.
    
    Args:
        mesh (Meshes): PyTorch3D mesh object
        sample_points (torch.Tensor): [N, 3] sampled points
        diameters (torch.Tensor): [N] diameter values
        output_path (str): Path to save the visualization
        title (str): Plot title
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh with transparency
    plot_mesh(ax, mesh, colour="gray", alpha=0.3, equalize=True, label="Mesh")
    
    # Apply log scaling to diameters
    log_diameters = torch.log1p(diameters)  # Using log1p to handle small values better
    
    # Normalize log-scaled diameters to [0, 1]
    normalized_diameters = (log_diameters - log_diameters.min()) / (log_diameters.max() - log_diameters.min())
    
    # Convert to numpy for plotting
    points_np = sample_points.cpu().numpy()
    diameters_np = normalized_diameters.cpu().numpy()
    
    # Create scatter plot with custom colormap
    scatter = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2],
                        c=diameters_np, cmap='RdYlGn_r', s=50)
    
    # Add colorbar with log-scale ticks
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Log-scaled Normalized Diameter')
    
    # Calculate and add actual diameter values to colorbar ticks
    log_ticks = np.linspace(log_diameters.min().item(), log_diameters.max().item(), 5)
    actual_values = torch.exp(torch.tensor(log_ticks)) - 1  # Reverse log1p
    cbar.set_ticks(np.linspace(0, 1, 5))  # Normalized positions
    cbar.set_ticklabels([f'{v:.2e}' for v in actual_values])  # Scientific notation
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Add legend
    ax.legend(['Mesh', 'Sample Points'])
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_rays(ax, origin, directions, distances=None, intersections=None, color='blue', alpha=0.5):
    """
    Visualize rays cast from a point.
    
    Args:
        ax: Matplotlib 3D axis
        origin (torch.Tensor): [3] origin point
        directions (torch.Tensor): [N, 3] ray directions
        distances (torch.Tensor, optional): [N] distances to intersections
        intersections (torch.Tensor, optional): [N, 3] intersection points
        color (str): Color for the rays
        alpha (float): Transparency for the rays
    """
    origin_np = origin.cpu().numpy()
    directions_np = directions.cpu().numpy()
    
    # Plot origin point
    ax.scatter(*origin_np, color='red', s=100, label='Sample Point')
    
    # If we have intersections, plot rays up to them
    if intersections is not None and distances is not None:
        intersections_np = intersections.cpu().numpy()
        for direction, intersection in zip(directions_np, intersections_np):
            ax.plot([origin_np[0], intersection[0]],
                   [origin_np[1], intersection[1]],
                   [origin_np[2], intersection[2]],
                   color=color, alpha=alpha)
    else:
        # Otherwise, plot rays with a fixed length
        ray_length = 0.1  # Arbitrary length for visualization
        for direction in directions_np:
            end_point = origin_np + direction * ray_length
            ax.plot([origin_np[0], end_point[0]],
                   [origin_np[1], end_point[1]],
                   [origin_np[2], end_point[2]],
                   color=color, alpha=alpha)

def generate_random_direction_in_cone(normal: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Generate a random direction within a 90-degree cone in the opposite direction of the normal.
    
    Args:
        normal (torch.Tensor): [3] surface normal vector (normalized)
        device: torch device
        
    Returns:
        direction (torch.Tensor): [3] random direction vector (normalized)
    """
    # Ensure we're working with a normalized vector
    normal = normal / torch.norm(normal)
    
    while True:
        # Generate random direction
        direction = torch.randn(3, device=device)
        direction = direction / torch.norm(direction)
        
        # Compute angle with -normal using dot product
        cos_theta = torch.dot(direction, -normal)
        
        # Check if direction is within 90-degree cone (cos_theta > 0)
        # and not too close to normal direction (avoid grazing angles)
        if cos_theta > 0:
            return direction

def debug_single_vertex(mesh: Meshes, num_rays: int = 30, output_dir: str = "sdf_output", seed: int = 0):
    """
    Debug mode that samples a single random vertex and visualizes all its ray casts.
    
    Args:
        mesh (Meshes): PyTorch3D mesh object
        num_rays (int): Number of rays to cast
        output_dir (str): Directory to save debug visualizations
        seed (int): Random seed for reproducible testing
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = mesh.device
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    
    # Compute bounding box diagonal and thresholds
    bbox_min = torch.min(verts, dim=0)[0]
    bbox_max = torch.max(verts, dim=0)[0]
    bbox_diagonal = torch.norm(bbox_max - bbox_min)
    
    print(f"Bounding box min: {bbox_min}")
    print(f"Bounding box max: {bbox_max}")
    print(f"Bounding box diagonal: {bbox_diagonal}")
    
    min_distance_threshold = bbox_diagonal * 0.0001
    max_distance_threshold = bbox_diagonal * 0.2
    ray_origin_offset = bbox_diagonal * 0.0001
    
    # Sample a random face
    face_areas = mesh.faces_areas_packed()
    face_probs = face_areas / face_areas.sum()
    sampled_face_idx = torch.multinomial(face_probs, 1)[0]
    
    # Get face vertices
    face_verts = verts[faces[sampled_face_idx]]  # [3, 3]
    
    # Sample a random point on the face
    w1 = torch.sqrt(torch.rand(1, device=device))
    w2 = torch.rand(1, device=device) * (1 - w1)
    w0 = 1 - w1 - w2
    
    # Compute sample point
    sample_point = (w0 * face_verts[0] +
                   w1 * face_verts[1] +
                   w2 * face_verts[2])  # [3]
    
    # Compute face normal
    normal = torch.cross(face_verts[1] - face_verts[0],
                        face_verts[2] - face_verts[0])
    normal = normal / torch.norm(normal)
    
    # Create figure for visualization
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh with transparency
    plot_mesh(ax, mesh, colour="gray", alpha=0.3, equalize=True, label="Mesh")
    
    # Store all ray information for visualization
    all_directions = []
    all_intersections = []
    all_distances = []
    valid_intersections = []
    valid_distances = []
    
    # Generate and cast rays
    print("\nCasting rays for debug visualization...")
    for _ in tqdm(range(num_rays)):
        # Generate random direction in the 90-degree cone
        direction = generate_random_direction_in_cone(normal, device)
        
        # Offset the ray origin
        offset_origin = sample_point + normal * ray_origin_offset
        
        # Cast ray with exclusion zone around source face
        intersections, distances = compute_ray_mesh_intersections_vectorized(
            offset_origin, direction, verts, faces,
            exclude_face_idx=sampled_face_idx,
            exclude_radius=ray_origin_offset * 2)
        
        # Store ray information
        all_directions.append(direction)
        
        if len(distances) > 0:
            min_dist_idx = torch.argmin(distances)
            intersection = intersections[min_dist_idx]
            distance = distances[min_dist_idx]
            total_distance = distance + ray_origin_offset
            
            all_intersections.append(intersection)
            all_distances.append(total_distance)
            
            if total_distance > min_distance_threshold and total_distance < max_distance_threshold:
                valid_intersections.append(intersection)
                valid_distances.append(total_distance)
    
    # Convert lists to tensors
    all_directions = torch.stack(all_directions)
    
    # Visualize rays
    if all_intersections:
        all_intersections = torch.stack(all_intersections)
        all_distances = torch.tensor(all_distances, device=device)
        visualize_rays(ax, offset_origin, all_directions,
                      all_distances, all_intersections,
                      color='blue', alpha=0.2)
    else:
        visualize_rays(ax, offset_origin, all_directions)
    
    # Visualize valid intersections
    if valid_intersections:
        valid_intersections = torch.stack(valid_intersections)
        ax.scatter(valid_intersections[:, 0].cpu(),
                  valid_intersections[:, 1].cpu(),
                  valid_intersections[:, 2].cpu(),
                  color='green', s=50, label='Valid Intersections')
    
    # Plot normal vector
    normal_length = float(bbox_diagonal * 0.1)  # Convert to Python float
    normal_end = sample_point + normal * normal_length
    ax.plot([sample_point[0].cpu(), normal_end[0].cpu()],
            [sample_point[1].cpu(), normal_end[1].cpu()],
            [sample_point[2].cpu(), normal_end[2].cpu()],
            color='red', linewidth=2, label='Surface Normal')
    
    # Visualize 90-degree cone
    theta = np.linspace(0, 2*np.pi, 100)
    radius = float(normal_length * np.sin(np.pi/4))  # Convert to Python float
    height = float(normal_length * np.cos(np.pi/4))  # Convert to Python float
    
    # Create points for cone visualization
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    circle_z = np.full_like(theta, height)
    
    # Transform cone to align with -normal direction
    # First, create rotation matrix to align [0,0,1] with -normal
    normal_np = -normal.cpu().numpy()
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, normal_np)
    
    # Move sample point to CPU for visualization
    sample_point_np = sample_point.cpu().numpy()
    
    if np.any(rotation_axis):  # if not parallel
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.dot(z_axis, normal_np)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Rodriguez rotation formula
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                     [rotation_axis[2], 0, -rotation_axis[0]],
                     [-rotation_axis[1], rotation_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * np.dot(K, K)
        
        # Transform circle points
        points = np.vstack((circle_x, circle_y, circle_z))
        points = np.dot(R, points)
        circle_x, circle_y, circle_z = points
    
    # Translate cone to sample point
    circle_x += sample_point_np[0]
    circle_y += sample_point_np[1]
    circle_z += sample_point_np[2]
    
    # Plot cone outline
    ax.plot(circle_x, circle_y, circle_z, 'r--', alpha=0.5, label='90° Cone')
    for i in range(0, len(theta), 10):
        ax.plot([sample_point_np[0], circle_x[i]],
                [sample_point_np[1], circle_y[i]],
                [sample_point_np[2], circle_z[i]],
                'r--', alpha=0.2)
    
    # Set title and labels
    ax.set_title("Ray Casting Debug Visualization")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    # Save debug visualization
    debug_path = os.path.join(output_dir, "ray_casting_debug.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(debug_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nDebug visualization saved to {debug_path}")
    print(f"Total rays cast: {num_rays}")
    print(f"Rays with intersections: {len(all_intersections)}")
    print(f"Rays with valid distances: {len(valid_intersections)}")
    if valid_distances:
        print(f"Valid distance range: {min(valid_distances):.6f} to {max(valid_distances):.6f}")
        print(f"Average valid distance: {sum(valid_distances)/len(valid_distances):.6f}")
        
        # Print angle statistics
        angles = torch.acos(torch.clamp(torch.matmul(all_directions, -normal), -1, 1))
        angles_degrees = angles * 180 / np.pi
        print(f"\nRay angle statistics (degrees):")
        print(f"Min angle: {angles_degrees.min().item():.2f}°")
        print(f"Max angle: {angles_degrees.max().item():.2f}°")
        print(f"Mean angle: {angles_degrees.mean().item():.2f}°")

def assign_vertex_sdf(verts: torch.Tensor, sample_points: torch.Tensor, smoothed_diameters: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Assign SDF values to each vertex based on k nearest sampled points.
    
    Args:
        verts (torch.Tensor): [V, 3] mesh vertices
        sample_points (torch.Tensor): [N, 3] sampled points
        smoothed_diameters (torch.Tensor): [N] SDF values for sampled points
        k (int): Number of nearest neighbors to consider
        
    Returns:
        vertex_sdf (torch.Tensor): [V] SDF values for each vertex
    """
    # Convert to numpy for KDTree
    verts_np = verts.cpu().numpy()
    sample_points_np = sample_points.cpu().numpy()
    diameters_np = smoothed_diameters.cpu().numpy()
    
    # Build KD-tree for sampled points
    tree = cKDTree(sample_points_np)
    
    # Find k nearest neighbors for each vertex
    distances, indices = tree.query(verts_np, k=k)
    
    # Compute weighted average of SDF values
    # Use inverse distance weighting
    weights = 1.0 / (distances + 1e-6)  # Add small epsilon to avoid division by zero
    weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize weights
    
    # Get SDF values of neighbors and compute weighted average
    neighbor_sdf = diameters_np[indices]
    vertex_sdf = (neighbor_sdf * weights).sum(axis=1)
    
    # Normalize to [0, 1] range
    min_sdf = vertex_sdf.min()
    max_sdf = vertex_sdf.max()
    if max_sdf > min_sdf:
        vertex_sdf = (vertex_sdf - min_sdf) / (max_sdf - min_sdf)
    else:
        vertex_sdf = np.zeros_like(vertex_sdf)
    
    return torch.from_numpy(vertex_sdf).to(verts.device)

def visualize_vertex_sdf(mesh: Meshes, vertex_sdf: torch.Tensor, output_path: str, title: str = "Vertex SDF"):
    """
    Visualize the mesh with vertices colored by their SDF values.
    
    Args:
        mesh (Meshes): PyTorch3D mesh object
        vertex_sdf (torch.Tensor): [V] SDF values for each vertex
        output_path (str): Path to save the visualization
        title (str): Plot title
    """
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get mesh data
    verts = mesh.verts_padded()[0]  # [V, 3]
    faces = mesh.faces_padded()[0]  # [F, 3]
    
    # Convert to numpy for plotting
    verts_np = verts.cpu().numpy()
    faces_np = faces.cpu().numpy()
    sdf_np = vertex_sdf.cpu().numpy()
    
    # Create scatter plot with custom colormap
    scatter = ax.scatter(verts_np[:, 0], verts_np[:, 1], verts_np[:, 2],
                        c=sdf_np, cmap='RdYlGn_r', s=50)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Normalized SDF Value')
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Equalize axes to preserve aspect ratio
    X, Y, Z = verts_np[:, 0], verts_np[:, 1], verts_np[:, 2]
    equal_3d_axes(ax, X, Y, Z, zoom=1.0)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_obj_file(obj_path: str = None, output_dir: str = "sdf_output", num_samples: int = 1000, num_rays: int = 30,
                    debug_mode: bool = False, seed: int = 0):
    """
    Process a single OBJ file or SMAL model and compute its SDF visualization.
    
    Args:
        obj_path (str, optional): Path to the input OBJ file. If None, uses SMAL_FILE from config.py
        output_dir (str): Directory to save the visualization
        num_samples (int): Number of points to sample on the mesh. If -1, samples all faces.
        num_rays (int): Number of rays to cast per point
        debug_mode (bool): If True, run in debug mode with single vertex visualization
        seed (int): Random seed for reproducible testing
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load the mesh
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading mesh...")
    if obj_path is None:
        # Use SMAL_FILE from config
        from config import SMAL_FILE
        print(f"Using SMAL model from: {SMAL_FILE}")
        
        # Load SMAL model
        with open(SMAL_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            smal_data = u.load()
            
        # Get vertices and faces from SMAL model
        verts = torch.FloatTensor(smal_data['v_template']).to(device)
        faces = torch.LongTensor(smal_data['f']).to(device)
        basename = os.path.splitext(os.path.basename(SMAL_FILE))[0]
    else:
        # Load OBJ file
        verts, faces, _ = load_obj(obj_path)
        faces = faces.verts_idx
        basename = os.path.splitext(os.path.basename(obj_path))[0]
    
    # Move to device
    verts = verts.to(device)
    faces = faces.to(device)
    
    # Create mesh object
    mesh = Meshes(verts=[verts], faces=[faces])
    
    if debug_mode:
        debug_single_vertex(mesh, num_rays, output_dir, seed)
        return
    
    # Normal processing continues here...
    # Compute initial SDF
    print(f"\nComputing SDF with {num_samples} samples and {num_rays} rays per sample...")
    sample_points, diameters = compute_sdf(mesh, num_samples, num_rays)
    
    # Smooth the distances
    print("\nSmoothing distances...")
    smoothed_diameters = smooth_distances(sample_points, diameters, k=50)
    
    # Create output filename for sampled points visualization
    output_path = os.path.join(output_dir, f"{basename}_sdf.png")
    
    # Visualize results for sampled points
    print("\nGenerating visualization for sampled points...")
    visualize_sdf(mesh, sample_points, smoothed_diameters, output_path,
                 title=f"Spatial Diameter Function - {basename}")
    print(f"Sampled points visualization saved to {output_path}")
    
    # Assign SDF values to all vertices
    print("\nAssigning SDF values to all vertices...")
    vertex_sdf = assign_vertex_sdf(verts, sample_points, smoothed_diameters, k=10)
    
    # Create output filename for vertex visualization
    vertex_output_path = os.path.join(output_dir, f"{basename}_vertex_sdf.png")
    
    # Visualize results for vertices
    print("\nGenerating visualization for vertices...")
    visualize_vertex_sdf(mesh, vertex_sdf, vertex_output_path,
                        title=f"Vertex SDF - {basename}")
    print(f"Vertex visualization saved to {vertex_output_path}")
    
    # Save results to file
    results = {
        'sample_points': sample_points.cpu(),
        'smoothed_diameters': smoothed_diameters.cpu(),
        'vertex_sdf': vertex_sdf.cpu(),
        'verts': verts.cpu(),
        'faces': faces.cpu(),
        'num_samples': num_samples,
        'num_rays': num_rays,
        'seed': seed
    }
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Save results
    results_path = os.path.join(data_dir, f"{basename}_sdf.pkl")
    with open(results_path, 'wb') as f:
        pkl.dump(results, f)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute and visualize Spatial Diameter Function for a 3D mesh")
    parser.add_argument("--obj_path", type=str, default=None,
                       help="Path to the input OBJ file. If not provided, uses SMAL_FILE from config.py")
    parser.add_argument("--output_dir", type=str, default="sdf_output",
                       help="Directory to save the visualization")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of points to sample on the mesh. If -1, samples all faces.")
    parser.add_argument("--num_rays", type=int, default=30,
                       help="Number of rays to cast per sampled point")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode with single vertex visualization")
    parser.add_argument("--seed", type=int, default=123,
                       help="Random seed for reproducible testing")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    try_mkdir(args.output_dir)
    
    # Process the mesh
    process_obj_file(args.obj_path, args.output_dir,
                    num_samples=args.num_samples,
                    num_rays=args.num_rays,
                    debug_mode=args.debug,
                    seed=args.seed)
