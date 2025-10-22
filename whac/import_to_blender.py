"""
Import WHAC inference results into Blender.
This script imports the dynamic SMPL-X mesh sequence and camera trajectory.

=============================================================================
USAGE FROM BLENDER'S SCRIPTING INTERFACE:
=============================================================================

1. Open Blender
2. Switch to "Scripting" workspace (top menu bar)
3. Open this file in the text editor (or paste the entire script)
4. EITHER:

   Option A - Edit the default path (line 366) and run directly:
   - Change: output_folder = "/mnt/share/dev-lbringer/markerless_mocap/WHAC/demo/Barnab"
   - To: output_folder = "/YOUR/PATH/TO/demo/YourVideo"
   - Click "Run Script" or press Alt+P

   Option B - Run with custom parameters from the console:
   - Click "Run Script" or press Alt+P (to load the functions)
   - Then in the Python Console at the bottom, type:
     >>> main(output_folder="/mnt/share/dev-lbringer/markerless_mocap/WHAC/demo/Barnab")
   - Or with additional options:
     >>> main(output_folder="/path/to/demo/YourVideo", frame_skip=2, camera_scale=0.15)

=============================================================================
USAGE FROM COMMAND LINE:
=============================================================================

blender --python whac/import_to_blender.py -- --output_folder demo/Barnab

Options:
  --output_folder PATH    Path to WHAC output folder (required)
  --frame_skip N         Import every Nth frame (default: 1)
  --camera_scale SCALE   Camera visualization scale (default: 0.1)

=============================================================================
"""

import os
import sys
import argparse
import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector, Euler
import torch
import joblib


def import_obj_file(filepath):
    """
    Import OBJ file - handles both Blender 3.x and 4.x API
    """
    try:
        # Blender 3.x and earlier
        bpy.ops.import_scene.obj(filepath=filepath)
    except AttributeError:
        try:
            # Blender 4.0+
            bpy.ops.wm.obj_import(filepath=filepath)
        except AttributeError:
            raise RuntimeError(
                "Could not find OBJ importer. Please ensure OBJ import addon is enabled:\n"
                "Edit > Preferences > Add-ons > Search for 'Wavefront' > Enable checkbox"
            )


# Optional: if running from command line, parse args
def parse_args():
    """Parse command line arguments when running from Blender CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the WHAC output folder (e.g., demo/Barnab)')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Import every Nth frame (default: 1 = all frames)')
    parser.add_argument('--camera_scale', type=float, default=0.1,
                        help='Scale factor for camera visualization (default: 0.1)')

    # Blender passes args after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    args = parser.parse_args(argv)
    return args


def coordinate_transform_matrix():
    """
    Create transformation matrix to convert from renderer coordinates to Blender.

    WHAC applies R = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]] to flip X and Y axes
    Then we need to convert Y-up to Z-up for Blender

    Combined transformation:
    1. Flip X and Y: R = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    2. Y-up to Z-up: [[1, 0, 0], [0, 0, 1], [0, -1, 0]]

    Result: [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
    """
    # Combined transformation: flip XY then Y-up to Z-up
    transform = Matrix([
        [-1,  0,  0, 0],
        [ 0,  0, -1, 0],
        [ 0,  1,  0, 0],
        [ 0,  0,  0, 1]
    ])
    return transform


def load_whac_data(output_folder):
    """Load the WHAC inference results"""

    # Check for required files
    slam_path = os.path.join(output_folder, 'slam_results.pth')
    detection_path = os.path.join(output_folder, 'detection_results.pth')
    whac_scale_path = os.path.join(output_folder, 'whac_scale.pth')

    if not os.path.exists(slam_path):
        raise FileNotFoundError(f"slam_results.pth not found in {output_folder}")

    # Check if mesh files exist
    mesh_folder = os.path.join(output_folder, 'mesh')
    if not os.path.exists(mesh_folder):
        raise FileNotFoundError(
            f"Mesh folder not found. Please run inference.py with --save_mesh flag first"
        )

    # Load SLAM results (camera trajectory)
    slam_results = joblib.load(slam_path)

    # Load WHAC scale if available
    whac_scale = None
    if os.path.exists(whac_scale_path):
        whac_scale_data = joblib.load(whac_scale_path)
        whac_scale = whac_scale_data.get('whac_scale', None)
        if whac_scale is not None:
            print(f"Loaded WHAC scale from file: {whac_scale:.4f}")

    # Get list of mesh files
    mesh_files = sorted([f for f in os.listdir(mesh_folder) if f.endswith('.obj')])

    if len(mesh_files) == 0:
        raise FileNotFoundError(f"No mesh files found in {mesh_folder}")

    return {
        'slam_results': slam_results,
        'mesh_folder': mesh_folder,
        'mesh_files': mesh_files,
        'whac_scale': whac_scale,
        'detection_results': joblib.load(detection_path) if os.path.exists(detection_path) else None
    }


def quaternion_to_rotation_matrix(quaternion_wxyz):
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.

    Args:
        quaternion_wxyz: Array of shape (..., 4) with quaternions in (w, x, y, z) format

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    w, x, y, z = quaternion_wxyz[..., 0], quaternion_wxyz[..., 1], quaternion_wxyz[..., 2], quaternion_wxyz[..., 3]

    # Compute rotation matrix elements
    R = np.zeros(quaternion_wxyz.shape[:-1] + (3, 3))

    R[..., 0, 0] = 1 - 2*(y**2 + z**2)
    R[..., 0, 1] = 2*(x*y - w*z)
    R[..., 0, 2] = 2*(x*z + w*y)

    R[..., 1, 0] = 2*(x*y + w*z)
    R[..., 1, 1] = 1 - 2*(x**2 + z**2)
    R[..., 1, 2] = 2*(y*z - w*x)

    R[..., 2, 0] = 2*(x*z - w*y)
    R[..., 2, 1] = 2*(y*z + w*x)
    R[..., 2, 2] = 1 - 2*(x**2 + y**2)

    return R


def process_slam_result(slam_results, whac_scale=None):
    """
    Process SLAM results from raw format to camera extrinsics matrices.
    Replicates the process_slam_result function from inference_utils.py

    Args:
        slam_results: numpy array of shape (num_frames, 7) containing [tx, ty, tz, qx, qy, qz, qw]
        whac_scale: Optional scale factor to apply to translations (matches WHAC processing)

    Returns:
        numpy array of shape (num_frames, 4, 4) with camera-to-world transformation matrices
    """
    translation = slam_results[:, :3]  # (N, 3)
    quaternion_xyzw = slam_results[:, 3:]  # (N, 4) in [x, y, z, w] format

    # Convert to [w, x, y, z] format
    quaternion_wxyz = quaternion_xyzw[:, [3, 0, 1, 2]]

    # Convert quaternions to rotation matrices
    rotation = quaternion_to_rotation_matrix(quaternion_wxyz)  # (N, 3, 3)

    # Apply WHAC scale to translations if provided
    if whac_scale is not None:
        translation = translation * whac_scale

    # Apply the same transformation that WHAC applies: R = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    R_flip = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rotation = np.einsum('ij,bjk->bik', R_flip, rotation)  # Apply flip to rotations
    translation = np.einsum('ij,bj->bi', R_flip, translation)  # Apply flip to translations

    # Build 4x4 homogeneous transformation matrices
    N = rotation.shape[0]
    RT = np.zeros((N, 4, 4))
    RT[:, :3, :3] = rotation
    RT[:, :3, 3] = translation
    RT[:, 3, 3] = 1.0

    return RT


def estimate_whac_scale_from_meshes(mesh_folder, mesh_files, slam_results, sample_size=10):
    """
    Estimate the WHAC scale factor by comparing mesh positions with raw SLAM data.

    The WHAC scale is applied to SLAM translations to match the world-space mesh coordinates.
    We can estimate it by comparing distances in mesh space vs SLAM space.

    Args:
        mesh_folder: Path to mesh files
        mesh_files: List of mesh filenames
        slam_results: Raw SLAM data (N, 7)
        sample_size: Number of meshes to sample for estimation

    Returns:
        Estimated scale factor
    """
    import trimesh

    # Sample mesh files evenly
    num_files = len(mesh_files)
    indices = np.linspace(0, num_files - 1, min(sample_size, num_files), dtype=int)

    mesh_positions = []
    slam_positions = []

    for idx in indices:
        # Load mesh and compute centroid
        mesh_path = os.path.join(mesh_folder, mesh_files[idx])
        try:
            mesh = trimesh.load(mesh_path, process=False)
            centroid = mesh.vertices.mean(axis=0)
            mesh_positions.append(centroid)

            # Get corresponding SLAM translation
            slam_positions.append(slam_results[idx, :3])
        except:
            continue

    if len(mesh_positions) < 2:
        print("Warning: Could not estimate scale from meshes, using default scale=1.0")
        return 1.0

    mesh_positions = np.array(mesh_positions)
    slam_positions = np.array(slam_positions)

    # Compute pairwise distances
    mesh_distances = np.linalg.norm(mesh_positions[1:] - mesh_positions[:-1], axis=1)
    slam_distances = np.linalg.norm(slam_positions[1:] - slam_positions[:-1], axis=1)

    # Estimate scale as ratio of distances (with some robustness)
    valid_mask = slam_distances > 1e-6
    if valid_mask.sum() > 0:
        scale_ratios = mesh_distances[valid_mask] / slam_distances[valid_mask]
        estimated_scale = np.median(scale_ratios)
        print(f"Estimated WHAC scale: {estimated_scale:.4f}")
        return float(estimated_scale)
    else:
        print("Warning: Could not estimate scale, using default scale=1.0")
        return 1.0


def get_camera_extrinsics_from_slam(slam_results, whac_scale=None):
    """
    Extract camera extrinsics from SLAM results.

    Args:
        slam_results: Either dict with 'slam_res_rotmat' or numpy array (N, 7)
        whac_scale: Optional scale factor to apply to translations

    Returns:
        numpy array of shape (num_frames, 4, 4) with camera-to-world transforms
    """
    if isinstance(slam_results, dict) and 'slam_res_rotmat' in slam_results:
        # Already processed format (from future versions or cached results)
        # Note: scale should already be applied in this case
        return slam_results['slam_res_rotmat']
    elif isinstance(slam_results, np.ndarray) and slam_results.shape[1] == 7:
        # Raw SLAM format: (num_frames, 7) with [tx, ty, tz, qx, qy, qz, qw]
        return process_slam_result(slam_results, whac_scale=whac_scale)
    else:
        raise ValueError(
            f"Unexpected slam_results format. "
            f"Expected dict with 'slam_res_rotmat' or numpy array of shape (N, 7), "
            f"got {type(slam_results)} with shape {slam_results.shape if hasattr(slam_results, 'shape') else 'N/A'}"
        )


def create_mesh_sequence(mesh_folder, mesh_files, frame_skip=1, coord_transform=None):
    """
    Import mesh sequence as shape keys on a single mesh object.

    Args:
        mesh_folder: Path to folder containing mesh OBJ files
        mesh_files: List of mesh filenames
        frame_skip: Import every Nth frame
        coord_transform: Matrix to transform coordinates

    Returns:
        Blender mesh object with shape keys
    """

    # Clear existing mesh data
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import first mesh as base
    first_mesh_path = os.path.join(mesh_folder, mesh_files[0])
    import_obj_file(first_mesh_path)
    mesh_obj = bpy.context.selected_objects[0]
    mesh_obj.name = "SMPL_Sequence"

    # Apply coordinate transformation to base mesh
    if coord_transform:
        mesh_obj.matrix_world = coord_transform
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    mesh = mesh_obj.data

    # Add basis shape key
    mesh_obj.shape_key_add(name='Basis')

    # Import remaining meshes as shape keys
    print(f"Importing {len(mesh_files[::frame_skip])} frames...")

    for i, mesh_file in enumerate(mesh_files[::frame_skip]):
        # Import mesh temporarily
        mesh_path = os.path.join(mesh_folder, mesh_file)
        import_obj_file(mesh_path)
        temp_obj = bpy.context.selected_objects[0]

        # Apply coordinate transformation
        if coord_transform:
            temp_obj.matrix_world = coord_transform
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Create shape key from this mesh
        shape_key = mesh_obj.shape_key_add(name=f'Frame_{i:04d}')

        # Copy vertex positions
        for v_idx, vert in enumerate(temp_obj.data.vertices):
            shape_key.data[v_idx].co = vert.co

        # Delete temporary mesh
        bpy.data.objects.remove(temp_obj)

        if i % 10 == 0:
            print(f"  Processed {i}/{len(mesh_files[::frame_skip])} frames")

    print(f"Mesh import complete: {len(mesh_obj.data.shape_keys.key_blocks)} shape keys")

    return mesh_obj


def animate_shape_keys(mesh_obj, start_frame=1, frame_skip=1):
    """
    Animate shape keys to play back the sequence.
    Sets up drivers or keyframes for shape key animation.
    """
    shape_keys = mesh_obj.data.shape_keys.key_blocks

    # Animate each shape key
    for i, sk in enumerate(shape_keys[1:]):  # Skip basis
        # Set value to 1.0 at its frame, 0.0 elsewhere
        frame_num = start_frame + i * frame_skip

        # Keyframe at previous frame (value = 0)
        if i > 0:
            sk.value = 0.0
            sk.keyframe_insert(data_path="value", frame=frame_num - 1)

        # Keyframe at current frame (value = 1.0)
        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=frame_num)

        # Keyframe at next frame (value = 0)
        sk.value = 0.0
        sk.keyframe_insert(data_path="value", frame=frame_num + 1)

    # Set animation to start at first frame
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = start_frame + len(shape_keys) * frame_skip
    bpy.context.scene.frame_current = start_frame

    print(f"Animation range: {bpy.context.scene.frame_start} - {bpy.context.scene.frame_end}")


def create_camera_empty(name, location, rotation_matrix, scale=0.1):
    """Create an empty object to represent a camera position"""
    bpy.ops.object.empty_add(type='CONE', location=location)
    empty = bpy.context.active_object
    empty.name = name
    empty.scale = (scale, scale, scale)

    # Set rotation from matrix
    empty.rotation_euler = rotation_matrix.to_euler()

    return empty


def import_camera_trajectory(extrinsics, coord_transform, frame_skip=1, scale=0.1):
    """
    Import camera trajectory as animated camera object.

    Args:
        extrinsics: numpy array of shape (num_frames, 4, 4)
        coord_transform: Matrix to transform from renderer to Blender coords
        frame_skip: Import every Nth frame
        scale: Scale factor for camera visualization

    Returns:
        Blender camera object
    """

    # Create camera
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.name = "SLAM_Camera"

    # Sample frames
    num_frames = extrinsics.shape[0]
    sampled_frames = list(range(0, num_frames, frame_skip))

    print(f"Importing camera trajectory: {len(sampled_frames)} frames...")

    # Animate camera position and rotation
    for i, frame_idx in enumerate(sampled_frames):
        frame_num = i + 1

        # Get camera-to-world transform (4x4 matrix)
        cam_to_world = Matrix(extrinsics[frame_idx].tolist())

        # Apply coordinate transformation
        cam_to_world = coord_transform @ cam_to_world

        # Blender cameras point along -Z axis, but our camera points along +Z
        # We need to rotate 180 degrees around X to fix this
        camera_correction = Matrix.Rotation(np.pi, 4, 'X')
        cam_to_world = cam_to_world @ camera_correction

        # Set camera transform
        camera.matrix_world = cam_to_world

        # Insert keyframes
        camera.keyframe_insert(data_path="location", frame=frame_num)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame_num)

    # Set as active camera
    bpy.context.scene.camera = camera

    print(f"Camera trajectory imported: {len(sampled_frames)} keyframes")

    return camera


def create_trajectory_visualization(extrinsics, coord_transform, frame_skip=1, name="CameraPath", color=(0.0, 1.0, 0.0)):
    """
    Create a curve object showing a trajectory path.

    Args:
        extrinsics: numpy array of shape (num_frames, 4, 4)
        coord_transform: Matrix to transform coordinates
        frame_skip: Sample every Nth frame
        name: Name for the curve object
        color: RGB color for the path (default: green)

    Returns:
        Blender curve object
    """

    # Extract positions
    num_frames = extrinsics.shape[0]
    sampled_frames = list(range(0, num_frames, frame_skip))

    positions = []
    for frame_idx in sampled_frames:
        cam_to_world = Matrix(extrinsics[frame_idx].tolist())
        cam_to_world = coord_transform @ cam_to_world
        positions.append(cam_to_world.translation)

    # Create curve
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2

    # Create spline
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(positions) - 1)

    for i, pos in enumerate(positions):
        polyline.points[i].co = (pos.x, pos.y, pos.z, 1)

    # Create object
    curve_obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Set curve properties
    curve_data.bevel_depth = 0.02
    curve_data.bevel_resolution = 4

    # Create material with color
    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    curve_obj.data.materials.append(mat)

    # Set color
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (*color, 1.0)
        # Try to set emission (not available in all Blender versions/configs)
        try:
            bsdf.inputs['Emission'].default_value = (*color, 1.0)
            bsdf.inputs['Emission Strength'].default_value = 0.5
        except KeyError:
            pass  # Emission not available, use base color only

    print(f"{name} visualization created with {len(positions)} points")

    return curve_obj


def create_human_trajectory_visualization(mesh_obj, frame_skip=1, vertex_index=4292):
    """
    Create a curve showing the human trajectory path.

    WHAC tracks vertex 4292 (pelvis region) to visualize human movement path.

    Args:
        mesh_obj: SMPL mesh object with shape keys
        frame_skip: Sample every Nth frame
        vertex_index: Which vertex to track (default: 4292, pelvis region)

    Returns:
        Blender curve object showing human path
    """

    # Check if mesh has enough vertices
    if not mesh_obj.data.shape_keys or len(mesh_obj.data.shape_keys.key_blocks) < 2:
        print(f"Warning: Mesh has no shape keys, skipping human trajectory")
        return None

    num_verts = len(mesh_obj.data.shape_keys.key_blocks[1].data)
    if vertex_index >= num_verts:
        print(f"Warning: Mesh only has {num_verts} vertices, cannot track vertex {vertex_index}")
        print(f"Using mesh centroid instead")
        vertex_index = None

    # Extract positions of the tracked vertex across all frames
    positions = []
    shape_keys = mesh_obj.data.shape_keys.key_blocks[1:]  # Skip basis

    for i, shape_key in enumerate(shape_keys[::frame_skip]):
        if vertex_index is not None:
            # Get the position of the tracked vertex in this frame
            vert_pos = shape_key.data[vertex_index].co.copy()
        else:
            # Use centroid if vertex index is invalid
            centroid = Vector((0, 0, 0))
            for vert_data in shape_key.data:
                centroid += vert_data.co
            vert_pos = centroid / len(shape_key.data)

        positions.append(vert_pos)

    # Create curve
    curve_data = bpy.data.curves.new('HumanPath', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2

    # Create spline
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(len(positions) - 1)

    for i, pos in enumerate(positions):
        polyline.points[i].co = (pos.x, pos.y, pos.z, 1)

    # Create object
    curve_obj = bpy.data.objects.new('HumanPath', curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Set curve properties - thinner than camera path
    curve_data.bevel_depth = 0.015
    curve_data.bevel_resolution = 4

    # Create material with light gray color (matching WHAC visualization)
    mat = bpy.data.materials.new(name="HumanPath_Material")
    mat.use_nodes = True
    curve_obj.data.materials.append(mat)

    # Set color to light gray (0.8, 0.8, 0.8)
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.3
        bsdf.inputs['Roughness'].default_value = 0.7

    print(f"Human trajectory path created with {len(positions)} points (tracking vertex {vertex_index})")

    return curve_obj


def create_checkerboard_ground(mesh_obj, length=None, tile_width=0.5, offset=0.0):
    """
    Create a checkerboard ground plane matching WHAC's visualization.

    WHAC uses dynamic ground sizing based on mesh trajectory:
    - Calculates XZ extents of mesh centroids across all frames
    - Ground size = max(sx, sz) * 1.5, with minimum of 10 meters
    - Tile width = 0.5 meters (24 tiles for 12m ground)

    Args:
        mesh_obj: The SMPL mesh object (used to determine ground position and size)
        length: Size of the ground plane (if None, calculated from mesh trajectory like WHAC)
        tile_width: Size of each checker tile (default: 0.5 meters to match WHAC)
        offset: Vertical offset (default: at the feet of the mesh)
    """
    # Get the minimum Z coordinate from the mesh to place ground at feet
    # In Blender's Z-up coordinate system, Z is the vertical axis
    if offset == 0.0:
        # Find the lowest point of the mesh (feet) - this is minimum Z in Blender Z-up
        min_z = float('inf')
        for shape_key in mesh_obj.data.shape_keys.key_blocks[1:]:  # Skip basis
            for vert_data in shape_key.data:
                if vert_data.co.z < min_z:
                    min_z = vert_data.co.z
        offset = min_z
        print(f"Detected ground level at Z={offset:.3f} (mesh feet position)")

    # Calculate ground size dynamically like WHAC if not provided
    if length is None:
        # Calculate centroid positions for all frames (shape keys)
        centroids = []
        for shape_key in mesh_obj.data.shape_keys.key_blocks[1:]:  # Skip basis
            # Calculate centroid of this frame
            centroid = Vector((0, 0, 0))
            for vert_data in shape_key.data:
                centroid += vert_data.co
            centroid /= len(shape_key.data)
            centroids.append(centroid)

        # Get min/max in X and Y (horizontal plane in Blender Z-up)
        centroids = np.array([[c.x, c.y, c.z] for c in centroids])
        min_pos = centroids.min(axis=0)
        max_pos = centroids.max(axis=0)

        # Calculate extents in X and Y (horizontal)
        sx = max_pos[0] - min_pos[0]  # X extent
        sy = max_pos[1] - min_pos[1]  # Y extent (horizontal in Blender)

        # Calculate center position
        cx = (max_pos[0] + min_pos[0]) / 2.0
        cy = (max_pos[1] + min_pos[1]) / 2.0

        # Ground size: max extent * 1.5, minimum 10 (matching WHAC logic)
        length = max(max(sx, sy) * 1.5, 10.0)

        print(f"Calculated ground size: {length:.2f}m (trajectory extent: X={sx:.2f}m, Y={sy:.2f}m)")
        print(f"Ground centered at: X={cx:.3f}, Y={cy:.3f}")
    else:
        cx, cy = 0.0, 0.0

    # Create plane - in Blender Z-up, XY plane is horizontal
    bpy.ops.mesh.primitive_plane_add(size=length, location=(cx, cy, offset))
    ground = bpy.context.active_object
    ground.name = "CheckerboardGround"

    # Plane is already horizontal in XY (no rotation needed for Z-up)
    ground.rotation_euler = (0, 0, 0)

    # Create material with checkerboard texture
    mat = bpy.data.materials.new(name="CheckerboardMaterial")
    mat.use_nodes = True
    ground.data.materials.append(mat)

    # Get material nodes
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create nodes for checkerboard pattern
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (400, 0)

    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_node.location = (0, 0)
    principled_node.inputs['Metallic'].default_value = 0.0
    principled_node.inputs['Roughness'].default_value = 0.5

    # Texture coordinate - using Generated for proper UV mapping on plane
    texcoord_node = nodes.new(type='ShaderNodeTexCoord')
    texcoord_node.location = (-800, 0)

    # Mapping node to scale the texture coordinates properly
    mapping_node = nodes.new(type='ShaderNodeMapping')
    mapping_node.location = (-600, 0)
    # Calculate number of tiles per side
    num_tiles = length / tile_width
    # Scale the UV coordinates to get the right number of tiles
    # Generated coords go from 0 to 1, we need to scale them to get num_tiles/2 periods
    mapping_node.inputs['Scale'].default_value = (num_tiles / 2.0, num_tiles / 2.0, 1.0)

    # Checker texture
    checker_node = nodes.new(type='ShaderNodeTexChecker')
    checker_node.location = (-400, 0)
    checker_node.inputs['Scale'].default_value = 1.0  # Scale is handled by mapping node

    # Colors (light gray and darker gray) - matching WHAC visualization
    checker_node.inputs['Color1'].default_value = (0.8, 0.9, 0.9, 1.0)  # Light
    checker_node.inputs['Color2'].default_value = (0.6, 0.7, 0.7, 1.0)  # Dark

    # Connect nodes
    links.new(texcoord_node.outputs['Generated'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], checker_node.inputs['Vector'])
    links.new(checker_node.outputs['Color'], principled_node.inputs['Base Color'])
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    num_tiles = int(length / tile_width)
    print(f"Checkerboard ground: {length:.2f}m × {length:.2f}m with {tile_width}m tiles ({num_tiles}×{num_tiles} grid)")

    return ground


def setup_scene(mesh_obj=None):
    """Setup basic scene lighting and rendering settings"""

    # Add a sun light
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.data.energy = 2.0

    # Set up rendering (handle both Blender 3.x and 4.x)
    try:
        # Blender 4.x
        bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        # Blender 3.x
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    # Set EEVEE options (available in both versions)
    try:
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.eevee.use_bloom = False
    except AttributeError:
        pass  # EEVEE settings not available or named differently

    # Set frame rate
    bpy.context.scene.render.fps = 30

    # Add checkerboard ground
    if mesh_obj is not None:
        create_checkerboard_ground(mesh_obj)


def main(output_folder=None, frame_skip=1, camera_scale=0.1):
    """
    Main function to import WHAC data into Blender

    Args:
        output_folder: Path to WHAC output folder (e.g., "demo/Barnab" or "/full/path/to/demo/Barnab")
        frame_skip: Import every Nth frame (default: 1 = all frames)
        camera_scale: Scale factor for camera visualization (default: 0.1)
    """

    # If no arguments provided, try to parse from command line
    if output_folder is None:
        try:
            args = parse_args()
            output_folder = args.output_folder
            frame_skip = args.frame_skip
            camera_scale = args.camera_scale
        except:
            # Default fallback - you can change this when running from script editor
            output_folder = "/mnt/share/dev-lbringer/markerless_mocap/WHAC/demo/Barnab"
            print("\n" + "!"*60)
            print("WARNING: Using default path!")
            print(f"Path: {output_folder}")
            print("To use a different path, call: main(output_folder='your/path')")
            print("!"*60 + "\n")

    print(f"\n{'='*60}")
    print(f"WHAC to Blender Importer")
    print(f"{'='*60}")
    print(f"Output folder: {output_folder}")
    print(f"Frame skip: {frame_skip}")
    print(f"Camera scale: {camera_scale}")
    print(f"{'='*60}\n")

    # Get coordinate transformation matrix
    coord_transform = coordinate_transform_matrix()

    # Load WHAC data
    print("Loading WHAC data...")
    data = load_whac_data(output_folder)

    # Get WHAC scale - prefer saved value, otherwise estimate from meshes
    print("\nLoading WHAC scale...")
    whac_scale = data.get('whac_scale', None)

    if whac_scale is None and isinstance(data['slam_results'], np.ndarray) and data['slam_results'].shape[1] == 7:
        print("WHAC scale not found in saved data, estimating from mesh positions...")
        whac_scale = estimate_whac_scale_from_meshes(
            data['mesh_folder'],
            data['mesh_files'],
            data['slam_results'],
            sample_size=10
        )
        print(f"WARNING: Using estimated scale. For accurate results, re-run inference.py to save the exact scale.")
    elif whac_scale is None:
        print("SLAM results already processed, using existing scale")
    else:
        print(f"Using saved WHAC scale: {whac_scale:.4f}")

    # Import mesh sequence
    print("\nImporting mesh sequence...")
    mesh_obj = create_mesh_sequence(
        data['mesh_folder'],
        data['mesh_files'],
        frame_skip=frame_skip,
        coord_transform=coord_transform
    )

    # Animate shape keys
    print("\nSetting up animation...")
    animate_shape_keys(mesh_obj, start_frame=1, frame_skip=frame_skip)

    # Import camera trajectory
    print("\nImporting camera trajectory...")
    extrinsics = get_camera_extrinsics_from_slam(data['slam_results'], whac_scale=whac_scale)

    camera = import_camera_trajectory(
        extrinsics,
        coord_transform,
        frame_skip=frame_skip,
        scale=camera_scale
    )

    # Create camera path visualization
    print("\nCreating camera path visualization...")
    create_trajectory_visualization(extrinsics, coord_transform, frame_skip=max(5, frame_skip))

    # Create human trajectory path visualization
    print("\nCreating human trajectory path...")
    create_human_trajectory_visualization(mesh_obj, frame_skip=max(5, frame_skip))

    # Setup scene
    print("\nSetting up scene...")
    setup_scene(mesh_obj=mesh_obj)

    print(f"\n{'='*60}")
    print("Import complete!")
    print(f"{'='*60}")
    print(f"- Mesh object: {mesh_obj.name}")
    print(f"- Camera: {camera.name}")
    print(f"- Frame range: {bpy.context.scene.frame_start} - {bpy.context.scene.frame_end}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()


# =============================================================================
# QUICK REFERENCE FOR BLENDER CONSOLE
# =============================================================================
#
# After running this script in Blender (Alt+P), you can use these commands
# in the Python Console:
#
# Import with default settings:
#   >>> main(output_folder="/path/to/demo/YourVideo")
#
# Import every 2nd frame (faster):
#   >>> main(output_folder="/path/to/demo/YourVideo", frame_skip=2)
#
# Import with larger camera visualization:
#   >>> main(output_folder="/path/to/demo/YourVideo", camera_scale=0.2)
#
# Import multiple videos in sequence:
#   >>> main(output_folder="/path/to/demo/Video1")
#   >>> main(output_folder="/path/to/demo/Video2")
#   >>> main(output_folder="/path/to/demo/Video3")
#
# =============================================================================
