import numpy as np
import trimesh

def load_stl_as_point_cloud(file_path):
    """
    Load an STL file and extract vertices as a point cloud.
    :param file_path: Path to the STL file.
    :return: Point cloud (Nx3 array of vertices).
    """
    mesh = trimesh.load_mesh(file_path)
    return np.array(mesh.vertices)

def closest_point(src, dst):
    distances = np.linalg.norm(src[:, np.newaxis, :] - dst[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)

def icp(src, dst, max_iterations=50, tolerance=1e-6):
    src_hom = np.hstack((src, np.ones((src.shape[0], 1))))
    prev_error = float('inf')

    for i in range(max_iterations):
        indices = closest_point(src, dst)
        dst_closest = dst[indices]

        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst_closest, axis=0)

        src_centered = src - src_mean
        dst_centered = dst_closest - dst_mean

        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = dst_mean - R @ src_mean
        src = (R @ src.T).T + t

        mean_error = np.mean(np.linalg.norm(src - dst_closest, axis=1))
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    return transformation

# Example usage
if __name__ == "__main__":
    # Load STL files and convert to point clouds
    src_cloud = load_stl_as_point_cloud("source_model.stl")
    dst_cloud = load_stl_as_point_cloud("target_model.stl")

    # Perform ICP
    transformation_matrix = icp(src_cloud, dst_cloud)
    print("Transformation matrix:\n", transformation_matrix)
