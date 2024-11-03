import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from stl import mesh
from sklearn.decomposition import PCA

def load_stl(filename):
    """Load an STL file and return the mesh object and its vertices as a numpy array."""
    model = mesh.Mesh.from_file(filename)
    vertices = np.unique(model.vectors.reshape(-1, 3), axis=0)
    return model, vertices

def apply_transformation(vertices, rotation_matrix, translation_vector):
    """Apply rotation and translation to the vertices."""
    return np.dot(vertices, rotation_matrix) + translation_vector

def rotation_matrix_from_angles(angles):
    """Generate a rotation matrix from Euler angles (in radians)."""
    rx, ry, rz = angles
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def alignment_error(params, source_vertices, target_vertices):
    """Calculate the error between transformed target vertices and source vertices."""
    angles = params[:3]
    translation = params[3:]
    rotation = rotation_matrix_from_angles(angles)
    transformed_target = apply_transformation(target_vertices, rotation, translation)

    # Find closest source vertex for each transformed target vertex
    tree = cKDTree(source_vertices)
    distances, _ = tree.query(transformed_target)
    return np.sum(distances)

def align_meshes(source_vertices, target_vertices):
    """Align target vertices to source vertices using rotation and translation optimization."""
    initial_params = np.zeros(6)  # 3 for rotation angles and 3 for translation vector
    result = minimize(
        alignment_error, initial_params,
        args=(source_vertices, target_vertices),
        method='L-BFGS-B'
    )

    # Extract the optimized rotation and translation
    angles = result.x[:3]
    translation = result.x[3:]
    rotation_matrix = rotation_matrix_from_angles(angles)
    
    # Apply transformation
    aligned_target_vertices = apply_transformation(target_vertices, rotation_matrix, translation)
    return aligned_target_vertices

def combine_meshes(source_model, source_vertices, aligned_target_vertices):
    """Combine two meshes by merging vertices and removing duplicates."""
    combined_vertices = np.vstack((source_vertices, aligned_target_vertices))
    unique_vertices, unique_indices = np.unique(combined_vertices, axis=0, return_index=True)

    # Create combined STL mesh with unique vertices
    combined_model = mesh.Mesh(np.zeros(unique_vertices.shape[0] // 3, dtype=mesh.Mesh.dtype))
    combined_model.vectors = unique_vertices[unique_indices].reshape(-1, 3, 3)
    return combined_model

# Load source and target STL models
source_filename = 'source.stl'  # STL to use as global coordinate system
target_filename = 'target.stl'  # STL to rotate and translate
source_model, source_vertices = load_stl("LowpolyStanfordBunnyUprightEars.stl")
target_model, target_vertices = load_stl("stanford-bunny.obj")

# Align the target vertices to the source vertices
aligned_target_vertices = align_meshes(source_vertices, target_vertices)

# Combine the two models with minimal overlap
combined_model = combine_meshes(source_model, source_vertices, aligned_target_vertices)

# Save the combined STL model
combined_filename = 'combined_model.stl'
combined_model.save(combined_filename)
print(f"Combined model saved as {combined_filename}")
