import trimesh
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

# Function to get user input with a default value
def get_user_input(prompt, default=None, cast_type=str):
    user_input = input(f"{prompt} [Default: {default}]: ") or default
    if user_input == default:
        return default
    try:
        return cast_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default value: {default}")
        return default

def load_stl(file_path):
    return trimesh.load(file_path)

def bounding_box_check(mesh, bed_dimensions):
    bounding_box = mesh.bounds
    box_size = bounding_box[1] - bounding_box[0]
    return np.all(box_size <= bed_dimensions)

def translate_and_rotate_3x3(mesh, rotation_angles):
    theta_x, theta_y, theta_z = rotation_angles
    rot_x = trimesh.transformations.rotation_matrix(theta_x, [1, 0, 0], point=None)[:3, :3]
    rot_y = trimesh.transformations.rotation_matrix(theta_y, [0, 1, 0], point=None)[:3, :3]
    rot_z = trimesh.transformations.rotation_matrix(theta_z, [0, 0, 1], point=None)[:3, :3]
    rotation_matrix = np.dot(np.dot(rot_x, rot_y), rot_z)
    mesh.apply_transform(np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]), [0, 0, 0, 1]]))
    return mesh

def voxelize_mesh(mesh, voxel_size=1.0):
    voxelized = mesh.voxelized(pitch=voxel_size)
    voxelized_mesh = voxelized.as_boxes()
    return voxelized_mesh

def estimate_print_area_voxel(mesh, layer_height, voxel_size=1.0):
    voxelized_mesh = voxelize_mesh(mesh, voxel_size)
    z_coords = voxelized_mesh.vertices[:, 2]
    layer_areas = []
    for z in np.arange(z_coords.min(), z_coords.max(), layer_height):
        layer_voxels = np.logical_and(z_coords >= z, z_coords < z + layer_height)
        area_estimate = np.sum(layer_voxels) * voxel_size ** 2
        layer_areas.append(area_estimate)
    return np.sum(layer_areas)

def vertices_intersecting_horizontal(mesh):
    vertices = mesh.vertices
    return np.sum(vertices[:, 2] <= 0.01)

def cost_function_rotation(params, original_mesh, bed_dimensions, layer_height, voxel_size=1.0):
    rotation_angles = params[:3]
    mesh = original_mesh.copy()
    mesh = translate_and_rotate_3x3(mesh, rotation_angles)
    if not bounding_box_check(mesh, bed_dimensions):
        return np.inf
    layer_area_estimate = estimate_print_area_voxel(mesh, layer_height, voxel_size)
    intersecting_vertices = vertices_intersecting_horizontal(mesh)
    cost = intersecting_vertices + layer_area_estimate
    return cost

def print_bounding_box_comparison(original_mesh, optimized_mesh):
    original_box = original_mesh.bounds
    optimized_box = optimized_mesh.bounds
    original_dimensions = original_box[1] - original_box[0]
    optimized_dimensions = optimized_box[1] - optimized_box[0]
    print("\nBounding Box Comparison:")
    print(f"Original Bounding Box (X, Y, Z): {original_dimensions}")
    print(f"Optimized Bounding Box (X, Y, Z): {optimized_dimensions}")

def minimize_orientation_with_bounding_box(mesh, bed_dimensions, layer_height=0.1, voxel_size=1.0):
    """
    Use scipy.optimize.minimize to find the best rotation for minimizing the print area and intersecting vertices.
    """
    initial_guess = np.random.uniform(0, 2 * np.pi, size=3)  # Initial guess for the rotation angles
    
    result = minimize(
        cost_function_rotation, 
        initial_guess, 
        args=(mesh, bed_dimensions, layer_height, voxel_size), 
        method='L-BFGS-B',  # Use L-BFGS-B for bounded optimization
        bounds=[(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)]  # Bounds for rotation angles
    )
    
    if result.success:
        best_params = result.x
        print(f"\nOptimal rotation found: {best_params} with cost: {result.fun}")
        optimized_mesh = translate_and_rotate_3x3(mesh.copy(), best_params)
        print_bounding_box_comparison(mesh, optimized_mesh)
        return best_params
    else:
        print("Optimization failed!")
        return None

def save_stl(mesh, output_path):
    mesh.export(output_path)

# Main Script
if __name__ == "__main__":
    stl_file = input("Enter the STL file path: ")
    
    default_bed_size = [200, 200, 200]
    default_layer_height = 0.1
    default_voxel_size = 1.0

    bed_size_input = get_user_input("Enter the printer bed size (x, y, z) in mm (comma-separated)", default_bed_size)
    if isinstance(bed_size_input, str):
        bed_size = [float(i) for i in bed_size_input.split(',')]
    else:
        bed_size = bed_size_input

    layer_height = get_user_input("Enter the layer height in mm", default_layer_height, float)
    voxel_size = get_user_input("Enter the voxel size", default_voxel_size, float)

    # Load the STL file
    mesh = load_stl(stl_file)

    # Perform minimization using scipy.optimize.minimize
    best_params = minimize_orientation_with_bounding_box(mesh, bed_size, layer_height, voxel_size)

    if best_params is not None:
        optimized_mesh = translate_and_rotate_3x3(mesh.copy(), best_params)
        output_stl_file = "optimized_model_3x3.stl"
        save_stl(optimized_mesh, output_stl_file)
        print(f"\nOptimized model saved to {output_stl_file}")
