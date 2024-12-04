import trimesh
import numpy as np
from scipy.optimize import minimize


def preprocess_mesh(mesh, max_vertices=10000, voxel_size_factor=0.01):
    """
    Preprocess the mesh by decimating and/or voxelizing it.
    """
    if len(mesh.vertices) > max_vertices:
        decimated_mesh = mesh.simplify_quadratic_decimation(max_vertices)
    else:
        decimated_mesh = mesh

    bounding_box_size = decimated_mesh.bounding_box.extents
    voxel_size = voxel_size_factor * max(bounding_box_size)
    voxelized_mesh = decimated_mesh.voxelized(voxel_size).as_boxes()
    return voxelized_mesh


def repair_mesh(mesh):
    """
    Ensure the mesh is valid by repairing it.
    """
    if not mesh.is_watertight:
        mesh = mesh.fill_holes()
    
    # Replace deprecated methods
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def calculate_support_volume(mesh, overhang_angle=43):
    """
    Calculate the volume of support material needed for a given mesh orientation.
    """
    overhang_angle_rad = np.radians(overhang_angle)
    face_normals = mesh.face_normals
    z_axis = np.array([0, 0, -1])
    cos_theta = np.dot(face_normals, z_axis)
    overhang_faces = np.where(np.arccos(cos_theta) > overhang_angle_rad)[0]
    overhang_vertices = mesh.vertices[mesh.faces[overhang_faces]]
    projected_area = np.sum([np.abs(np.cross(v[1] - v[0], v[2] - v[0])) for v in overhang_vertices]) / 2
    return projected_area


def calculate_slice_area(mesh):
    """
    Calculate the total area per slice by slicing the mesh along the Z-axis.
    """
    slices = mesh.section_multiplane(
        plane_origin=[0, 0, mesh.bounds[0][2]],
        plane_normal=[0, 0, 1],
        heights=np.linspace(mesh.bounds[0][2], mesh.bounds[1][2], 50),
    )
    slice_areas = []
    for s in slices:
        if s is not None:
            try:
                slice_areas.append(s.area)
            except Exception:
                # Skip problematic slices
                slice_areas.append(0)
        else:
            slice_areas.append(0)
    return sum(slice_areas)


def evaluate_orientation(rotation_vector, mesh, overhang_angle=43):
    """
    Evaluate the quality of an orientation based on support volume and slice area.
    """
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.linalg.norm(rotation_vector),
        direction=rotation_vector / np.linalg.norm(rotation_vector),
        point=None,
    )
    rotated_mesh = mesh.copy()
    rotated_mesh.apply_transform(rotation_matrix)
    support_volume = calculate_support_volume(rotated_mesh, overhang_angle)
    slice_area = calculate_slice_area(rotated_mesh)
    return support_volume + slice_area


def optimize_orientation(mesh, overhang_angle=43):
    """
    Optimize the orientation of a mesh.
    """
    initial_guess = np.array([0, 0, 1e-4])
    result = minimize(
        evaluate_orientation,
        x0=initial_guess,
        args=(mesh, overhang_angle),
        method="BFGS",
        options={"disp": True},
    )
    optimal_rotation = result.x
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.linalg.norm(optimal_rotation),
        direction=optimal_rotation / np.linalg.norm(optimal_rotation),
        point=None,
    )
    optimal_mesh = mesh.copy()
    optimal_mesh.apply_transform(rotation_matrix)
    return optimal_mesh, optimal_rotation


if __name__ == "__main__":
    file_path = "Knob.stl"
    mesh = trimesh.load(file_path)
    print("Original Mesh Vertex Count:", len(mesh.vertices))

    # Repair the mesh
    repaired_mesh = repair_mesh(mesh)

    # Preprocess the mesh (decimation + voxelization)
    preprocessed_mesh = preprocess_mesh(repaired_mesh)
    print("Preprocessed Mesh Vertex Count:", len(preprocessed_mesh.vertices))

    # Use the original mesh if preprocessing increases vertex count
    if len(preprocessed_mesh.vertices) > len(repaired_mesh.vertices):
        print("Preprocessed mesh has more vertices than the original. Using the original.")
        preprocessed_mesh = repaired_mesh

    # Center the mesh at its center of mass
    preprocessed_mesh.apply_translation(-preprocessed_mesh.center_mass)

    # Optimize orientation
    optimized_mesh, optimal_rotation = optimize_orientation(preprocessed_mesh)
    print("Optimal Rotation Vector (radians):", optimal_rotation)

    # Save the optimized mesh
    optimized_mesh.export("optimized_mesh.stl")
