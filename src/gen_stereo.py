import bpy, random, json, os, sys, argparse
from math import radians, tan
from mathutils import Vector, Euler

def parse_args():
    argv = sys.argv
    # splits script arguments from blender engine arguments
    argv = argv[argv.index("--")+1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True, help="output path")
    p.add_argument("--n", type=int, default=100, help="number of stereo pairs")
    p.add_argument("--start", type=int, default=0, help="start index")
    p.add_argument("--width", type=int, default=512, help="width of images")
    p.add_argument("--height", type=int, default=512, help="height of images")
    p.add_argument("--baselines", type=str, default="0.06,0.08,0.10,0.12,0.14", help="comma-separated baselines")
    p.add_argument("--fov_min", type=float, default=45.0, help="minimum allowd fov")
    p.add_argument("--fov_max", type=float, default=70.0, help="maximum allowd baseline")
    p.add_argument("--fixed_fov", type=float, default=60.0, help="fixed fov")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    return p.parse_args(argv)

A = parse_args()
os.makedirs(A.out, exist_ok=True)
random.seed(A.seed)
_baselines = [float(x) for x in A.baselines.split(",") if x.strip()]

# start with a clean scene
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = A.width
scene.render.resolution_y = A.height
scene.render.film_transparent = False

vl = scene.view_layers[0]
vl.use_pass_z = True
vl_name = vl.name  

# Depth
scene.use_nodes = True
tree = scene.node_tree
for n in list(tree.nodes):
    tree.nodes.remove(n)

rl   = tree.nodes.new("CompositorNodeRLayers")
comp = tree.nodes.new("CompositorNodeComposite")  
tree.links.new(rl.outputs["Image"], comp.inputs["Image"])

file_depth = tree.nodes.new("CompositorNodeOutputFile")
file_depth.label = "DepthOut"
file_depth.format.file_format = "OPEN_EXR"
tree.links.new(rl.outputs["Depth"], file_depth.inputs["Image"])


# world and lighting
scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
wnodes = scene.world.node_tree.nodes
wnodes["Background"].inputs[1].default_value = 1.0
wnodes["Background"].inputs[0].default_value = (0.9, 0.9, 0.95, 1.0)

light_data = bpy.data.lights.new(name="Key", type='AREA')
light_data.energy = 3000
light_obj = bpy.data.objects.new("Key", light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (2.0, -2.0, 3.0)
light_obj.rotation_euler = Euler((radians(60), 0.0, radians(35)), 'XYZ')

# helpers
def add_random_mesh():
    num = random.randint(3, 6)
    for i in range(num):
        t = random.choice(["CUBE", "UV_SPHERE", "CONE", "CYLINDER", "TORUS"])
        if t == "CUBE":
            bpy.ops.mesh.primitive_cube_add(size=random.uniform(0.3, 1.0))
        elif t == "UV_SPHERE":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=random.uniform(0.3, 0.9), segments=32, ring_count=16)
        elif t == "CONE":
            bpy.ops.mesh.primitive_cone_add(radius1=random.uniform(0.2, 0.7), depth=random.uniform(0.5,1.2))
        elif t == "CYLINDER":
            bpy.ops.mesh.primitive_cylinder_add(radius=random.uniform(0.2,0.6), depth=random.uniform(0.5,1.2))
        elif t == "TORUS":
            bpy.ops.mesh.primitive_torus_add(major_radius=random.uniform(0.3,0.8), minor_radius=random.uniform(0.1,0.25))
        ob = bpy.context.active_object
        ob.location = (random.uniform(-1.2, 1.2), random.uniform(1.8, 3.5), random.uniform(-0.2, 1.0))
        ob.rotation_euler = (random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14))
        mat = bpy.data.materials.new(f"Mat_{i}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        col = (random.random(), random.random(), random.random(), 1.0)
        bsdf.inputs["Base Color"].default_value = col
        bsdf.inputs["Roughness"].default_value = random.uniform(0.2, 0.9)
        ob.data.materials.append(mat)

def look_at(obj, target):
    direction = (target - obj.location).normalized()
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

from math import radians
from mathutils import Vector
def camera_to_world(obj):
    return obj.matrix_world.copy()

def intrinsics(width, height, fov_deg):
    from math import tan, radians
    fx = (width / 2.0) / tan(radians(fov_deg) / 2.0)
    return fx, fx, width/2.0, height/2.0

"""
create two cameras:
left camera shifted left with baseline/2
right camera shifted right with baseline/2
"""
def create_cameras(baseline_m, fov_deg):
    # Left Camera
    camL = bpy.data.cameras.new("CamLeft")
    camL.lens_unit = 'FOV'; camL.angle = radians(fov_deg)
    obL = bpy.data.objects.new("CamLeft", camL); scene.collection.objects.link(obL)

    # Right Camera
    camR = bpy.data.cameras.new("CamRight")
    camR.lens_unit = 'FOV'; camR.angle = radians(fov_deg)
    obR = bpy.data.objects.new("CamRight", camR); scene.collection.objects.link(obR)

    # camera direction 
    target = Vector((0.0, 2.5, 0.6))
    obL.location = Vector((-baseline_m/2.0, 0.0, 1.2))
    obR.location = Vector(( baseline_m/2.0, 0.0, 1.2))
    look_at(obL, target); look_at(obR, target)
    return obL, obR

def remove_object(obj):
    bpy.data.objects.remove(obj, do_unlink=True)

def remove_camera(cam_obj):
    cam_data = cam_obj.data
    remove_object(cam_obj)
    bpy.data.cameras.remove(cam_data, do_unlink=True)

# main generation loop
for i in range(A.start, A.start + A.n):
    # create subdirectory
    idx = f"{i:03d}"
    sample_dir = os.path.join(A.out, idx)
    os.makedirs(sample_dir, exist_ok=True)

    file_depth.base_path = sample_dir
    file_depth.file_slots[0].path = f"{idx}_depth_left_"

    # delete all meshes in scene
    for obj in [o for o in list(scene.objects) if o.type == "MESH"]:
        remove_object(obj)

    add_random_mesh()

    baseline = random.choice(_baselines)
    fov_deg  = A.fixed_fov if A.fixed_fov is not None else random.uniform(A.fov_min, A.fov_max)
    camL, camR = create_cameras(baseline, fov_deg)

    fx, fy, cx, cy = intrinsics(A.width, A.height, fov_deg)

    # only one depth map
    file_depth.base_path = sample_dir
    file_depth.file_slots[0].path = f"{idx}_depth_left_"
    file_depth.mute = False

    # Left
    scene.camera = camL
    scene.render.filepath = os.path.join(sample_dir, f"{idx}_left.png")
    bpy.ops.render.render(write_still=True)

    # Right
    file_depth.mute = True
    scene.camera = camR
    scene.render.filepath = os.path.join(sample_dir, f"{idx}_right.png")
    bpy.ops.render.render(write_still=True)

    # activate for next sample depth
    file_depth.mute = False

    # meta data
    meta = {
        "index": i,
        "width": A.width, "height": A.height,
        "baseline_m": baseline, "fov_deg": fov_deg,
        # focal length fx and fy
        # optical center cx and cy
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "cam_left":  {"matrix_world": [list(row) for row in camera_to_world(camL)], "name": "CamLeft"},
        "cam_right": {"matrix_world": [list(row) for row in camera_to_world(camR)], "name": "CamRight"},
    }
    with open(os.path.join(sample_dir, f"{idx}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    remove_camera(camL); remove_camera(camR)

print("Done.")
