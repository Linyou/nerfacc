import taichi as ti
import numpy as np
import tqdm
from taichi.math import vec3
import torch
# from kernel_utils import init_m_transforms, pack_transforms, ray_at
from datasets.dnerf_3d_video import SubjectLoader

ti.init(arch=ti.vulkan)

grid_size = 256
grid_size2 = grid_size**2
grid_size3 = grid_size**3
grid_size4index = grid_size - 1

@ti.kernel
def init_m_transforms(
    gx: ti.types.ndarray(), 
    gy: ti.types.ndarray(), 
    gz: ti.types.ndarray(),
    transforms: ti.template(),
    indicator: ti.template(),
    occ_grid: ti.template()
) -> ti.i32:
    bit_sum = 0
    for i, j, k in ti.ndrange(grid_size, grid_size, grid_size):
        n = i*grid_size2 + j*grid_size + k

        transforms[n][0, 0] = 1
        transforms[n][1, 1] = 1
        transforms[n][2, 2] = 1
        transforms[n][3, 3] = 1

        occ = occ_grid[i, grid_size4index-k, j]
        
        if occ:
            curr_bit_sum = ti.atomic_add(bit_sum, 1)
            if curr_bit_sum % 100 == 0:
                indicator[n] = 1
            transforms[n][0, 3] = gx[i, j, k]
            transforms[n][1, 3] = gy[i, j, k]
            transforms[n][2, 3] = gz[i, j, k]

    return bit_sum


@ti.kernel
def pack_transforms(
    num_instance: ti.i32, 
    indicator: ti.template(), 
    packed_transforms: ti.template(), 
    transforms: ti.template()):
    n = 0
    for i in range(num_instance):
        if indicator[i] == 1:
            index = ti.atomic_add(n, 1)
            packed_transforms[index] = transforms[i]

@ti.kernel
def ray_at(
    rays_o: ti.types.ndarray(), 
    rays_d: ti.types.ndarray(), 
    t: ti.f32, 
    points_pos: ti.template()):
    for i in ti.ndrange(points_pos.shape[0]):
        ray_o = vec3(rays_o[i, 0], rays_o[i, 1], rays_o[i, 2])
        ray_d = vec3(rays_d[i, 0], rays_d[i, 1], rays_d[i, 2])
        points_pos[i] = ray_o + t * ray_d


N = 10
cube_scale = 2.
cube_offset = 1.5
num_instance = grid_size**3

# generate one cube for grid
def generate_cube():
    n_grid_sample = 2
    n_grid_sample_nor = n_grid_sample - 1
    cube_np = (np.array([
        # front
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
        # back]
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0]
    ])) / cube_scale
    # Normalize to [0, 1].
    cube_np = (cube_np / n_grid_sample_nor * cube_scale - np.array([0,0,1])[None,:]) / grid_size
    print(cube_np.shape)
    cube_elements = np.array([
        # front
        0, 1, 2, 2, 3, 0,
        # right
        1, 5, 6, 6, 2, 1,
        # back
        7, 6, 5, 5, 4, 7,
        # left
        4, 0, 3, 3, 7, 4,
        # bottom
        4, 5, 1, 1, 0, 4,
        # top
        3, 2, 6, 6, 7, 3
    ])
    return cube_np.astype(np.float32), cube_elements.astype(np.int32)

cube_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
ijk_base = ti.field(shape=12*3, dtype=ti.i32)
cube_np, cube_elements = generate_cube()
cube_vertices.from_numpy(cube_np)
ijk_base.from_numpy(cube_elements.astype(np.int32))

# generate bounding box cube
def get_line_bounding_box():
    v = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ]).astype(np.float32) * 2.0 - 1.0

    v = v * cube_scale - np.array([0,0,1.0])[None,:]

    i = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]).astype(np.int32).reshape(-1)

    return v, i

vertices = ti.Vector.field(3, dtype=ti.f32, shape = 8)
lines_indices = ti.field(dtype=ti.i32, shape = 24)
vertices_np, lines_indices_np = get_line_bounding_box()
vertices.from_numpy(vertices_np)
lines_indices.from_numpy(lines_indices_np)

# get occupancy grid
path_root = '/home/loyot/workspace/code/training_results/nerfacc/checkpoints/'
model_path = path_root + 'ngp_dnerf_lr_0-01_hl2_nopf_nopw_l-huber_distor_op_coffee_martini_20000.pth'

def get_occ_from_model(path):
    print('Loading model from {}'.format(model_path))
    model_state = torch.load(path, map_location='cpu')
    print(model_state.keys())
    occ_bool = model_state['occupancy_grid']['_binary']
    occ = occ_bool.to(torch.uint8).numpy()
    return occ

density_grid = ti.field(ti.uint8, shape=(grid_size, grid_size, grid_size))
density_grid.from_numpy(get_occ_from_model(model_path))

# generate instance transforms
def get_grid_coord():
    xs, ys, zs = np.meshgrid(
        np.arange(grid_size, dtype=np.int32),
        np.arange(grid_size, dtype=np.int32),
        np.arange(grid_size, dtype=np.int32),
        indexing='ij'
    )

    normalize_grid = lambda grid: (grid.astype(np.float32) / grid_size4index * 2 - 1) * cube_scale
    xs = normalize_grid(xs)
    ys = normalize_grid(ys)
    zs = normalize_grid(zs) - 1
    return xs, ys, zs

m_transforms = ti.Matrix.field(4, 4, dtype = ti.f32, shape = num_instance)
indicator_tran = ti.field(dtype = ti.i32, shape = num_instance)    
grid_x, grid_y, grid_z = get_grid_coord()
bit_sum = init_m_transforms(
    grid_x, grid_y, grid_z, 
    m_transforms, indicator_tran, density_grid
)
print(f"bit_sum: {bit_sum}")
final_m_transforms = ti.Matrix.field(4, 4, dtype = ti.f32, shape = bit_sum)

pack_transforms(num_instance, indicator_tran, final_m_transforms, m_transforms)

def get_lrtr():
    scene = 'coffee_martini'
    data_root_fp = "/home/loyot/workspace/Datasets/NeRF/3d_vedio_datasets/"
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 2}
    test_dataset_kwargs = {"factor": 2}
    train_dataset_test = SubjectLoader(
        subject_id=scene,
        root_fp=data_root_fp,
        split='train',
        num_rays=None,
        read_image=False,
        **train_dataset_kwargs
    )
    train_dataset_test.training = False

    origins = []
    rays_o = []
    rays_d = []
    i = []
    c = []
    timestamps = []
    c_base = np.array([
        # [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [1.0, 0.5, 0.5],
    ], dtype=np.float32)
    i_base = np.array([0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1, 1, 3, 2, 4])
    for ind in range(0, len(train_dataset_test)):
        
        ray_o, ray_d, timestamp = train_dataset_test.get_ltrb(ind)
        rays_o.append(ray_o)
        rays_d.append(ray_d)
        i.append(i_base + (ind * 5))
        c.append(c_base)
        print(f"timestamp: {timestamp}")
        timestamps.append(timestamp)
        origins.append(ray_o[0:1])

    rays_o = np.concatenate(rays_o, axis=0)
    rays_d = np.concatenate(rays_d, axis=0)
    i = np.concatenate(i, axis=0)
    c = np.concatenate(c, axis=0)
    # timestamps = np.array(timestamps, dtype=np.float32)
    origins = np.concatenate(origins, axis=0)
    # origins = origins[np.argsort(timestamps)]

    temps = []
    for d in range(origins.shape[0]):
        tem_ = np.array([0, 1]) + d
        temps.append(tem_)
    temps = np.concatenate(temps, axis=0)

    return rays_o, rays_d, i, c, origins.reshape(-1, 3), temps

rays_o, rays_d, lrtf_indices_np, llrtf_colors_np, origins_np, origin_indice_np = get_lrtr()
print("origins_np.shape: ", origins_np.shape)
lrtf_vertices = ti.Vector.field(3, dtype=ti.f32, shape = rays_d.shape[0])
lrtf_colors = ti.Vector.field(3, dtype=ti.f32, shape = llrtf_colors_np.shape[0])
lrtf_indices = ti.field(dtype=ti.i32, shape = lrtf_indices_np.shape[0])
origins = ti.Vector.field(3, dtype=ti.f32, shape = origins_np.shape[0])
origin_indice = ti.field(dtype=ti.i32, shape = origin_indice_np.shape[0])
lrtf_colors.from_numpy(llrtf_colors_np)
lrtf_indices.from_numpy(lrtf_indices_np)
origins.from_numpy(origins_np)
origin_indice.from_numpy(origin_indice_np)


window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(2.34681249, 2.57935544, -9.94782073)
# camera.up(0, 0, 1)
# camera.lookat(0, 0, 0)

canvas.set_background_color((0, 0, 0))
t = 0.4
speed = 0.04
while window.running:

    gui = window.get_gui()
    with gui.sub_window("vis config", 0.05, 0.3, 0.2, 0.1) as w:
        t = w.slider_float("ltrb t", t, 0.1, 10)
        speed = w.slider_float("movement_speed", speed, 0.02, 0.5)


    camera.track_user_inputs(window, movement_speed=speed, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(2.5, 2.5, 2.5), color=(1, 1, 1))
    # scene.point_light(pos=(1.5, -1.5, 1.5), color=(1, 1, 1))
    # scene.point_light(pos=(-2.5, -2.5, -2.5), color=(1, 1, 1))
    ray_at(rays_o, rays_d, t, lrtf_vertices)
    scene.lines(vertices=lrtf_vertices, indices=lrtf_indices, per_vertex_color=lrtf_colors, width = 5.0)

    scene.lines(vertices=vertices, indices=lines_indices, color = (1., 1., 1.), width = 5.0)

    # scene.lines(vertices=origins, indices=origin_indice, color = (1., 0., 1.), width = 100.0)

    # Draw mesh instances (from the 1st instance)
    scene.mesh_instance(cube_vertices, ijk_base, transforms = final_m_transforms, instance_offset = 1, color=(0.9, 0.5, 0.5), show_wireframe=False)

    canvas.scene(scene)
    window.show()