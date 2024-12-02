import pyvista as pv
import numpy as np

# Step 1: Load or Create Your Mesh
# Replace this with your actual mesh
mesh = pv.Sphere(radius=5)  # Example mesh; replace with your extracted surface mesh

# Step 2: Calculate the Average Normal
normals = mesh.point_normals
avg_normal = np.mean(normals, axis=0)
avg_normal /= np.linalg.norm(avg_normal)  # Normalize the vector

# Step 3: Project the Mesh onto a Plane
centroid = mesh.center  # Center of the mesh
flattened_points = mesh.points - np.outer(np.dot(mesh.points - centroid, avg_normal), avg_normal)

# Create a new mesh from the flattened points
flattened_mesh = pv.PolyData(flattened_points, mesh.faces)

# Step 4: Optional Smoothing
# Uncomment if the surface is noisy or irregular
# flattened_mesh = flattened_mesh.smooth(n_iter=20, relaxation_factor=0.1)

# Step 5: Compute the Inertial Axes
inertia_tensor = flattened_mesh.inertia_tensor
eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

# The eigenvector with the largest eigenvalue is the major axis
major_axis = eigenvectors[:, np.argmax(eigenvalues)]

# Step 6: Visualize the Flattened Mesh and Major Axis
plotter = pv.Plotter()
plotter.add_mesh(flattened_mesh, color='white', show_edges=True, opacity=0.6)
plotter.add_arrows(flattened_mesh.center, major_axis, mag=1.0, color="red", label="Major Axis")
plotter.add_text("Flattened Surface and Major Axis", font_size=12)
plotter.show()

# Step 7: Use Major Axis for Label Alignment
# Use the `major_axis` to determine the direction of your labels
print("Major Axis for Label Alignment:", major_axis)
