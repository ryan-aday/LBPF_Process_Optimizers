import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def align_point_clouds_pca(pc1, pc2):
    """
    Aligns two point clouds using PCA to determine their principal axes.
    
    Parameters:
        pc1 (numpy.ndarray): First point cloud (N x 3).
        pc2 (numpy.ndarray): Second point cloud (M x 3).
        
    Returns:
        pc2_aligned (numpy.ndarray): Aligned version of pc2.
        transform (dict): Transformation details (rotation and translation).
    """
    # Center point clouds at their means
    pc1_centered = pc1 - np.mean(pc1, axis=0)
    pc2_centered = pc2 - np.mean(pc2, axis=0)
    
    # Compute PCA for each point cloud
    pca1 = PCA(n_components=3)
    pca2 = PCA(n_components=3)
    
    pca1.fit(pc1_centered)
    pca2.fit(pc2_centered)
    
    # Get the principal components
    axes1 = pca1.components_
    axes2 = pca2.components_
    
    # Align the axes using Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(np.dot(axes2.T, axes1))
    R = np.dot(U, Vt)
    
    # Apply rotation to the second point cloud
    pc2_rotated = np.dot(pc2_centered, R.T)
    
    # Align the centroids
    t = np.mean(pc1, axis=0) - np.mean(pc2_rotated, axis=0)
    pc2_aligned = pc2_rotated + t
    
    # Transformation details
    transform = {"rotation": R, "translation": t}
    
    return pc2_aligned, transform

# Example point clouds
np.random.seed(42)
pc1 = np.random.uniform(-10, 10, (100, 3))  # Random distribution for cloud 1
R_true = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0], [-0.48, 0.64, 0.6]])  # True rotation
t_true = np.array([5, -2, 3])  # True translation
pc2 = np.dot(pc1, R_true.T) + t_true  # Transformed version of pc1

# Align pc2 to pc1
pc2_aligned, transform = align_point_clouds_pca(pc1, pc2)

# Visualize the alignment
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], label="Point Cloud 1", alpha=0.5)
ax1.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], label="Point Cloud 2 (Unaligned)", alpha=0.5)
ax1.set_title("Before Alignment")
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], label="Point Cloud 1", alpha=0.5)
ax2.scatter(pc2_aligned[:, 0], pc2_aligned[:, 1], pc2_aligned[:, 2], label="Point Cloud 2 (Aligned)", alpha=0.5)
ax2.set_title("After Alignment")
ax2.legend()

plt.show()

# Print transformation details
print("Rotation Matrix:\n", transform["rotation"])
print("Translation Vector:\n", transform["translation"])
