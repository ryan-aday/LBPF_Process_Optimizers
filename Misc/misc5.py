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

def decimate_mesh(mesh, ratio=0.5):
    mesh.fill_holes()
    initial_face_count = len(mesh.faces)
    decimated_mesh = mesh.simplify_quadric_decimation(int(len(mesh.faces) * ratio))
    #decimated_mesh = mesh.simplify_vertex_clustering()

    print(f"Decimation reduced from {initial_face_count} to {len(decimated_mesh.faces)} faces.")
    decimated_mesh.process(validate=True)
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

def mod_mesh(mesh, output_path, iterations=2, featuredeg=15.5):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    

    #ms.meshing_repair_non_manifold_edges()
    #ms.meshing_repair_non_manifold_vertices()
    # Fill holes and remove duplicate triangles
    ms.meshing_close_holes(maxholesize=100)
    print("Closed holes.")
    ms.meshing_remove_duplicate_faces()
    print("Removed duplicate faces.")
    ms.meshing_remove_duplicate_vertices()
    print("Removed duplicate vertices.")

    
    # Laplacian smoothing and isotropic remeshing
    ms.apply_filter("meshing_isotropic_explicit_remeshing", adaptive=True, featuredeg=featuredeg, smoothflag=False, iterations=2)
    #ms.apply_filter("generate_surface_reconstruction_vcg", smoothnum=4, widenum=5, geodesic=2.5, simplification=, normalsmooth=5)
    #ms.apply_filter("generate_surface_reconstruction_screened_poisson", visiblelayer=True, cgdepth=2, scale=1.2, samplespernode=5.0)
    #ms.apply_filter("compute_iso_parametrization", stopcriteria='Area + Angle')
    #print(f"Applied {iterations} iterations of Laplacian smoothing and isotropic remeshing.")
    #ms.apply_filter("generate_iso_parametrization_remeshing")
    #ms.apply_filter("generate_resampled_uniform_mesh")

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


def main(input_file, output_file, ratio=0.4, iterations=2, featuredeg=2.0):
    mesh = load_mesh(input_file)
    decimated_mesh = decimate_mesh(mesh, ratio)
    
    # Smooth and make the mesh isometric
    #smoothed_mesh = smooth_and_make_isometric(decimated_mesh, iterations, featuredeg)
    
    # Smooth and make the mesh isometric, then clean and export the final mesh
    mod_mesh(decimated_mesh, output_file, iterations, featuredeg)

if __name__ == "__main__":
    input_file = fileopenbox(title='Select input file.', default='*', filetypes=["*.stl", "*.obj"])
    output_file = "output_mesh.stl"
    main(input_file, output_file)
