import bpy
import zipfile
import os
import shutil

# Path to the folder containing multiple ZIP files
zip_folder_path = "/media/fabi/WOLO_DRIVE/SMAL/AKIRA_ANTS/ORIGINAL"
top_level_dir = os.path.dirname(zip_folder_path)
output_dir = os.path.join(top_level_dir, "PROCESSED")
max_vertices = 20000


# Custom function to export a mesh object to an OBJ file
def export_mesh_to_obj(obj, filepath):
    if obj.type != 'MESH':
        raise TypeError("The selected object is not a mesh.")

    # Convert mesh to triangles
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = obj.data
    vertices = []
    faces = []

    for vert in mesh.vertices:
        vertices.append(vert.co)

    for poly in mesh.polygons:
        if len(poly.vertices) == 3:
            faces.append(poly.vertices)
        else:
            raise ValueError(f"Face with vertices {poly.vertices} is not a triangle and will be skipped.")

    with open(filepath, 'w') as file:
        for vert in vertices:
            file.write(f"v {vert.x} {vert.y} {vert.z}\n")

        for face in faces:
            file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    return filepath


# Function to extract files from a ZIP
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


# Function to find the nested ZIP file within the extracted directory
def find_nested_zip(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".zip"):
                return os.path.join(root, file)
    return None


# Function to get the path of the extracted STL or PLY file
def get_mesh_file_path(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".stl") or file.lower().endswith(".ply"):
                return os.path.join(root, file)
    return None


# Function to apply simplification modifiers
def apply_modifiers(obj):
    # Apply Edge Split Modifier
    edge_split = obj.modifiers.new(name="EdgeSplit", type='EDGE_SPLIT')
    edge_split.split_angle = 1.5708  # 90 degrees
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

    # Apply Weld Modifier
    weld = obj.modifiers.new(name="Weld", type='WELD')
    weld.merge_threshold = 0.01  # Adjust threshold as needed
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Weld")

    # Apply Limited Dissolve in Edit Mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.dissolve_limited(angle_limit=0.01)  # Adjust angle limit as needed
    bpy.ops.object.mode_set(mode='OBJECT')


# Create the top-level output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each ZIP file in the provided folder
for zip_file_name in os.listdir(zip_folder_path):
    if zip_file_name.lower().endswith(".zip"):
        zip_file_path = os.path.join(zip_folder_path, zip_file_name)
        extracted_dir = os.path.join(top_level_dir, "source")

        # Create the extracted_dir if it doesn't exist
        os.makedirs(extracted_dir, exist_ok=True)

        # Extract the main ZIP file
        print(f"Extracting the main ZIP file: {zip_file_path}")
        extract_zip(zip_file_path, top_level_dir)
        print("Main ZIP file extracted.")

        # Find the nested ZIP file in the "source" directory
        print("Searching for the nested ZIP file...")
        nested_zip_path = find_nested_zip(extracted_dir)
        if not nested_zip_path:
            print(f"No nested ZIP file found in the 'source' directory for {zip_file_name}. Skipping...")
            shutil.rmtree(extracted_dir)
            continue
        print(f"Nested ZIP file found: {nested_zip_path}")

        # Extract the nested ZIP file within the same "source" directory
        print("Extracting the nested ZIP file...")
        extract_zip(nested_zip_path, extracted_dir)
        print("Nested ZIP file extracted.")

        # Get the path of the extracted STL or PLY file
        print("Searching for the mesh file (STL or PLY)...")
        mesh_file_path = get_mesh_file_path(extracted_dir)
        if not mesh_file_path:
            print(f"No STL or PLY file found in the 'source' directory for {zip_file_name}. Skipping...")
            shutil.rmtree(extracted_dir)
            continue
        print(f"Mesh file found: {mesh_file_path}")

        # Import the mesh file into Blender
        print("Importing the mesh file into Blender...")
        file_extension = os.path.splitext(mesh_file_path)[1].lower()
        if file_extension == ".stl":
            bpy.ops.import_mesh.stl(filepath=mesh_file_path)
        elif file_extension == ".ply":
            bpy.ops.wm.ply_import(filepath=mesh_file_path)
        else:
            print(f"Unsupported file format for {mesh_file_path}. Skipping...")
            shutil.rmtree(extracted_dir)
            continue

        # Get the imported object
        obj = bpy.context.selected_objects[0]

        # Rotate the mesh around the X axis by 90 degrees
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.objects.active = obj
        bpy.ops.transform.rotate(value=1.5708, orient_axis='X')

        # Rotate the mesh around the Z axis by 90 degrees
        bpy.ops.transform.rotate(value=1.5708, orient_axis='Z')

        # Scale the mesh so that its X dimension is one Blender unit long
        bbox = obj.bound_box
        x_length = max(bbox[i][0] for i in range(8)) - min(bbox[i][0] for i in range(8))
        scale_factor = 1.0 / x_length
        bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))

        # Apply all transforms
        try:
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        except RuntimeError as e:
            if "Cannot apply to a multi user" in str(e):
                print("Making mesh data single-user and retrying...")
                bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
            else:
                raise

        # Apply simplification modifiers
        print("Applying simplification modifiers...")
        apply_modifiers(obj)

        # Apply a decimate modifier iteratively if necessary with Preserve Volume
        print("Applying the decimate modifier...")
        initial_vertices = len(obj.data.vertices)
        current_vertices = initial_vertices

        while current_vertices > max_vertices:
            modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
            modifier.decimate_type = 'COLLAPSE'
            modifier.use_symmetry = False
            modifier.use_collapse_triangulate = True
            modifier.use_dissolve_boundaries = False
            modifier.use_limit_surface = True  # Preserve volume

            # Apply decimation with a ratio of 0.5 if more than twice the target vertices
            if current_vertices / max_vertices > 2:
                modifier.ratio = 0.5
            else:
                modifier.ratio = max_vertices / current_vertices

            bpy.context.view_layer.objects.active = obj
            try:
                bpy.ops.object.modifier_apply(modifier="Decimate")
            except RuntimeError as e:
                if "Modifiers cannot be applied to multi-user data" in str(e):
                    print("Making mesh data single-user and retrying...")
                    bpy.ops.object.make_single_user(object=True, obdata=True, material=False, animation=False)
                    bpy.ops.object.modifier_apply(modifier="Decimate")
                else:
                    raise

            current_vertices = len(obj.data.vertices)
            print(f"Current vertices after decimation: {current_vertices}")

        # Derive the export file name from the original ZIP file name
        original_zip_name = os.path.splitext(zip_file_name)[0]
        export_path = os.path.join(output_dir, f"{original_zip_name}.obj")

        # Export the mesh as a .obj file using the custom function
        print(f"Exporting the mesh to {export_path}...")
        export_mesh_to_obj(obj, export_path)
        print("Mesh exported successfully.")

        # Delete the imported mesh object
        bpy.data.objects.remove(obj)

        # Delete the extracted "source" folder and its contents
        print(f"Deleting the extracted 'source' folder and its contents: {extracted_dir}")
        shutil.rmtree(extracted_dir)
        print("Source folder deleted.")

        print(f"Completed processing for {zip_file_name}.\n")

print("Batch processing completed successfully.")
