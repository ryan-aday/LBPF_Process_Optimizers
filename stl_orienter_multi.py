import io
from tqdm import tqdm
import trimesh
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
import os
import random

# Global evaluation counter stored in a dictionary for easy access
info = {'Nfeval': 0}

# Callback function for progress update during minimization
def callbackF(Xi):
    """
    This function is used as a callback during the optimization process to print updates
    every 50 function evaluations. It also updates the tqdm progress bar.

    :param Xi: Current parameter values (rotation angles) in the optimization process.
    """
    global info
    if info['Nfeval'] % 50 == 0:
        current_cost = cost_function_rotation(Xi, mesh, bed_size, layer_height, voxel_size, parallel_workers)
        print(f"Iter: {info['Nfeval']} - X: {Xi} - Cost: {current_cost}")
    
    # Update tqdm progress bar
    tqdm_bar.update(1)
    info['Nfeval'] += 1

# Function to get user input with a default value
def get_user_input(prompt, default=None, cast_type=str):
    """
    Helper function to get user input with a default value. Returns the user input cast to the desired type.

    :param prompt: The prompt message to display to the user.
    :param default: The default value if the user doesn't provide input.
    :param cast_type: The type to cast the input to (e.g., int, float, str).
    :return: User input cast to the specified type, or the default value if input is invalid.
    """
    user_input = input(f"{prompt} [Default: {default}]: ") or default
    if user_input == default:
        return default
    try:
        return cast_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default value: {default}")
        return default

def load_stl(file_path):
    """
    Load the STL file from the provided file path with a progress bar for larger files.

    :param file_path: Path to the STL file.
    :return: A trimesh object representing the loaded STL file, and its file size.
    """
    file_size = os.path.getsize(file_path)  # Get the file size in bytes

    # Read the STL file in chunks with a progress bar
    file_data = bytearray()
    with open(file_path, 'rb') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading STL") as pbar:
            for chunk in iter(lambda: f.read(4096), b''):  # Read in 4KB chunks
                file_data.extend(chunk)
                pbar.update(len(chunk))

    file_like_object = io.BytesIO(file_data)

    try:
        mesh = trimesh.load(file_obj=file_like_object, file_type='stl')
    except Exception as e:
        print(f"Failed to load STL file: {e}")
        return None
    
    return mesh, file_size

def bounding_box_check(mesh, bed_dimensions):
    """
    Check if the mesh's bounding box fits within the printer's bed dimensions.

    :param mesh: The 3D mesh object.
    :param bed_dimensions: Dimensions of the printer bed.
    :return: Boolean indicating whether the mesh fits within the printer's bed.
    """
    bounding_box = mesh.bounds
    if bounding_box is None or len(bounding_box) < 2:
        return False  # Return False if bounding_box is None or invalid
    box_size = bounding_box[1] - bounding_box[0]  # Calculate size of the bounding box
    return np.all(box_size <= bed_dimensions)

def translate_and_rotate_3x3(mesh, rotation_angles):
    """
    Apply 3x3 rotation matrix (x, y, z) to the mesh.

    :param mesh: The 3D mesh object to be transformed.
    :param rotation_angles: A list or array containing rotation angles [theta_x, theta_y, theta_z] in radians.
    :return: The transformed mesh object.
    """
    theta_x, theta_y, theta_z = rotation_angles
    rot_x = trimesh.transformations.rotation_matrix(theta_x, [1, 0, 0], point=None)[:3, :3]
    rot_y = trimesh.transformations.rotation_matrix(theta_y, [0, 1, 0], point=None)[:3, :3]
    rot_z = trimesh.transformations.rotation_matrix(theta_z, [0, 0, 1], point=None)[:3, :3]
    rotation_matrix = np.dot(np.dot(rot_x, rot_y), rot_z)
    mesh.apply_transform(np.vstack([np.hstack([rotation_matrix, np.zeros((3, 1))]), [0, 0, 0, 1]]))
    return mesh

def voxelize_mesh(mesh, voxel_size=1.0):
    """
    Voxelize the mesh based on the specified voxel size.

    :param mesh: The 3D mesh object to voxelize.
    :param voxel_size: The size of each voxel cube.
    :return: A voxelized mesh.
    """
    voxelized = mesh.voxelized(pitch=voxel_size)
    voxelized_mesh = voxelized.as_boxes()
    return voxelized_mesh

def estimate_print_area_voxel(mesh, layer_height, voxel_size=1.0):
    """
    Estimate the print area per layer using voxelization.

    :param mesh: The 3D mesh object to voxelize and estimate the area.
    :param layer_height: The height of each layer.
    :param voxel_size: The size of the voxel for voxelization.
    :return: The total estimated print area.
    """
    voxelized_mesh = voxelize_mesh(mesh, voxel_size)
    z_coords = voxelized_mesh.vertices[:, 2]
    layer_areas = []
    for z in np.arange(z_coords.min(), z_coords.max(), layer_height):
        layer_voxels = np.logical_and(z_coords >= z, z_coords < z + layer_height)
        area_estimate = np.sum(layer_voxels) * voxel_size ** 2
        layer_areas.append(area_estimate)
    return np.sum(layer_areas)

def vertices_intersecting_horizontal(mesh):
    """
    Calculate the number of vertices intersecting with the horizontal plane (z = 0).

    :param mesh: The 3D mesh object.
    :return: The number of vertices intersecting with the horizontal plane.
    """
    vertices = mesh.vertices
    return np.sum(vertices[:, 2] <= 0.01)

def cost_function_rotation(params, original_mesh, bed_dimensions, layer_height, voxel_size=1.0, parallel_workers=4):
    """
    Cost function that applies 3x3 rotation and estimates the print area and intersecting vertices.

    :param params: Rotation angles for optimization.
    :param original_mesh: The original mesh before transformations.
    :param bed_dimensions: The dimensions of the printer bed.
    :param layer_height: The height of each layer.
    :param voxel_size: The size of the voxel for voxelization.
    :param parallel_workers: Number of parallel workers for voxelization and vertex intersection calculations.
    :return: The cost value combining intersecting vertices and print area.
    """
    rotation_angles = params[:3]
    mesh = original_mesh.copy()
    mesh = translate_and_rotate_3x3(mesh, rotation_angles)

    if not bounding_box_check(mesh, bed_dimensions):
        return np.inf

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = []
        futures.append(executor.submit(estimate_print_area_voxel, mesh, layer_height, voxel_size))
        futures.append(executor.submit(vertices_intersecting_horizontal, mesh))
        layer_area_estimate = futures[0].result()
        intersecting_vertices = futures[1].result()

    cost = intersecting_vertices + layer_area_estimate

    return cost

# Additional function for filling print bed with parts
def populate_print_bed(mesh, bed_dimensions, spacing=5.0):
    """
    Populate the print bed with as many parts as possible within the given dimensions.
    
    :param mesh: The optimized 3D mesh to duplicate.
    :param bed_dimensions: Dimensions of the printer bed.
    :param spacing: Space between parts in mm.
    :return: A combined mesh with as many parts as can fit in the print bed.
    """
    # Get bounding box size of the rotated mesh
    bounding_box = mesh.bounds
    part_size = bounding_box[1] - bounding_box[0] + spacing  # Add spacing between parts

    # Calculate how many parts fit along each dimension
    num_x = int(bed_dimensions[0] // part_size[0])
    num_y = int(bed_dimensions[1] // part_size[1])
    num_z = int(bed_dimensions[2] // part_size[2])

    parts = []
    for x in range(num_x):
        for y in range(num_y):
            for z in range(num_z):
                # Calculate translation for each part
                translation = [
                    x * part_size[0],
                    y * part_size[1],
                    z * part_size[2]
                ]
                part_copy = mesh.copy()
                part_copy.apply_translation(translation)
                parts.append(part_copy)
    
    # Combine all parts into a single mesh
    return trimesh.util.concatenate(parts)

def minimize_orientation_with_bounding_box(mesh, bed_dimensions, layer_height=0.1, voxel_size=1.0, parallel_workers=4, file_size=None):
    """
    Use L-BFGS-B optimization to find the optimal rotation for minimizing the print area and intersecting vertices.
    Apply quadratic decimation if the file size exceeds 5000 KB.

    :param mesh: The 3D mesh object.
    :param bed_dimensions: The dimensions of the printer bed.
    :param layer_height: The height of each layer.
    :param voxel_size: The size of each voxel.
    :param parallel_workers: Number of parallel workers for voxelization and intersection calculations.
    :param file_size: The size of the STL file (in bytes).
    :return: The best rotation parameters found by the optimizer.
    """
    global tqdm_bar
    initial_guess = np.random.uniform(0, 2 * np.pi, size=3)
    bounds = [(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)]

    if file_size > 5000 * 1024:
        print("Applying Quadratic Edge Collapse Decimation due to large file size.")
        mesh = apply_quadratic_decimation(mesh, target_reduction=0.5)

    tqdm_bar = tqdm(total=1000, desc="L-BFGS-B Minimization Progress", unit="iter")

    result = minimize(
        cost_function_rotation, 
        initial_guess, 
        args=(mesh, bed_dimensions, layer_height, voxel_size, parallel_workers), 
        method='L-BFGS-B',
        bounds=bounds,  
        callback=callbackF,  
        options={'maxiter': 1000}
    )
    tqdm_bar.close()

    if result.success:
        best_params = result.x
        print(f"\nOptimal rotation found: {best_params} with cost: {result.fun}")
    else:
        print("Optimization failed!")
        return None

    return best_params

def save_stl(mesh, output_path):
    """
    Save the mesh to an STL file.

    :param mesh: The 3D mesh object to save.
    :param output_path: The path to the output STL file.
    """
    mesh.export(output_path)

# Main Script
if __name__ == "__main__":
    stl_file = input("Enter the STL file path: ")

    default_bed_size = [200, 200, 200]
    default_layer_height = 0.1
    default_voxel_size = 1.0
    default_parallel_workers = 4  
    default_overhang_angle = 45  
    default_branch_density = 1  

    bed_size_input = get_user_input("Enter the printer bed size (x, y, z) in mm (comma-separated)", default_bed_size)
    if isinstance(bed_size_input, str):
        bed_size = [float(i) for i in bed_size_input.split(',')]
    else:
        bed_size = bed_size_input

    layer_height = get_user_input("Enter the layer height in mm", default_layer_height, float)
    voxel_size = get_user_input("Enter the voxel size", default_voxel_size, float)
    parallel_workers = get_user_input("Enter the number of parallel workers", default_parallel_workers, int)
    overhang_angle = np.radians(get_user_input("Enter the maximum overhang angle in degrees", default_overhang_angle, float))
    branch_density = get_user_input("Enter the branch density for supports (higher values create more supports)", default_branch_density, int)
    
    add_supports = get_user_input("Would you like to add supports to the optimized model? (yes/no)", 'yes', str).lower() == 'yes'
    populate_choice = get_user_input("Would you like to populate the print bed with as many parts as possible? (yes/no)", 'no', str).lower() == 'yes'

    mesh, file_size = load_stl(stl_file)
    if mesh is None:
        print("Failed to load the STL file.")
        exit(1)

    best_params = minimize_orientation_with_bounding_box(mesh, bed_size, layer_height, voxel_size, parallel_workers, file_size)

    if best_params is not None:
        optimized_mesh = translate_and_rotate_3x3(mesh.copy(), best_params)
        
        if populate_choice:
            optimized_mesh = populate_print_bed(optimized_mesh, bed_size, spacing=5.0)

        if add_supports:
            mesh_with_supports = add_supports_to_model(optimized_mesh, max_overhang_angle=overhang_angle, branch_density=branch_density)
            output_stl_file = "optimized_model_with_supports.stl"
            save_stl(mesh_with_supports, output_stl_file)
            print(f"\nOptimized model with supports saved to {output_stl_file}")
        else:
            output_stl_file = "optimized_model_no_supports.stl"
            save_stl(optimized_mesh, output_stl_file)
            print(f"\nOptimized model without supports saved to {output_stl_file}")
