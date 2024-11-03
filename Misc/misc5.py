import trimesh
import pymeshlab
import sys
from easygui import fileopenbox

def load_mesh(file_path):
    try:
        mesh = trimesh.load(file_path)
        return mesh
    except Exception as e:
        print(f"Error loading mesh: {e}")
        sys.exit(1)

def decimate_mesh(mesh, target_faces=10000, ratio=0.7):
    mesh.fill_holes()
    initial_face_count = len(mesh.faces)
    decimated_mesh = mesh.simplify_quadric_decimation(int(len(mesh.faces) * ratio))
    print(f"Decimation reduced from {initial_face_count} to {len(decimated_mesh.faces)} faces.")
    return decimated_mesh

'''
def smooth_and_make_isometric(mesh, iterations=2, featuredeg=2.0):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))

    filters = pymeshlab.filter_list()
    print(filters)

'''
'''
    # Laplacian smoothing
    ms.apply_filter("apply_coord_laplacian_smoothing", stepsmoothnum=smoothing_iterations)
    print(f"Applied {smoothing_iterations} iterations of Laplacian smoothing.")

    # Make the mesh isometric by scaling uniformly
    ms.apply_filter("meshing_isotropic_explicit_remeshing", targetlen=target_edge_length)
    print("Mesh is scaled to be isometric.")

    # Get the smoothed and isometric mesh
    smoothed_mesh = ms.current_mesh()
'''
'''
    # Laplacian smoothing and isotropic remeshing
    ms.apply_filter("meshing_isotropic_explicit_remeshing", adaptive=True, featuredeg=featuredeg, smoothflag=True, iterations=iterations)
    print(f"Applied {iterations} iterations of Laplacian smoothing and isotropic remeshing.")
    smoothed_mesh = ms.current_mesh()
    return smoothed_mesh

def clean_and_export_mesh(mesh, output_path):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    #ms = pymeshlab.Mesh(mesh.vertices, mesh.faces)

    # Fill holes and remove duplicate triangles
    ms.meshing_close_holes()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()

    try:
        ms.save_current_mesh(output_path)
        print(f"Mesh cleaned and saved to {output_path}")
    except Exception as e:
        print(f"Error exporting mesh: {e}")
        sys.exit(1)

'''

def mod_mesh(mesh, output_path, iterations=2, featuredeg=2.0):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    
    # Fill holes and remove duplicate triangles
    ms.meshing_close_holes(maxholesize=100)
    print("Closed holes.")
    ms.meshing_remove_duplicate_faces()
    print("Removed duplicate faces.")
    ms.meshing_remove_duplicate_vertices()
    print("Removed duplicate vertices.")

    
    # Laplacian smoothing and isotropic remeshing
    #ms.apply_filter("meshing_isotropic_explicit_remeshing", adaptive=True, featuredeg=featuredeg, smoothflag=True, iterations=iterations)
    ms.apply_filter("generate_surface_reconstruction_vcg", smoothnum=2, geodesic=3, simplification=False, normalsmooth=4)

    print(f"Applied {iterations} iterations of Laplacian smoothing and isotropic remeshing.")


    try:
        ms.save_current_mesh(output_path)
        
        # Fill in holes if the mod_mesh failed to do so earlier
        check_mesh = trimesh.load(output_path)
        print(f"Mesh cleaned and saved to {output_path}")           
        print(f"Final mesh watertight: {trimesh.load(output_path).is_watertight}")
    except Exception as e:
        print(f"Error exporting mesh: {e}")
        sys.exit(1)
    

    '''
    filters = pymeshlab.filter_list()
    print(filters)
    ''' 


def main(input_file, output_file, target_faces=10000, ratio=0.7, iterations=2, featuredeg=1.0):
    mesh = load_mesh(input_file)
    decimated_mesh = decimate_mesh(mesh, target_faces, ratio)
    
    # Smooth and make the mesh isometric
    #smoothed_mesh = smooth_and_make_isometric(decimated_mesh, iterations, featuredeg)
    
    # Smooth and make the mesh isometric, then clean and export the final mesh
    mod_mesh(decimated_mesh, output_file, iterations, featuredeg)

if __name__ == "__main__":
    input_file = fileopenbox(title='Select input file.', default='*', filetypes=["*.stl", "*.obj"])
    output_file = "output_mesh.stl"
    target_faces = 10000
    main(input_file, output_file, target_faces)
