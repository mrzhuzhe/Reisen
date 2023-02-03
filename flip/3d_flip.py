from matplotlib.pyplot import grid
from sklearn.preprocessing import KernelCenterer
import taichi as ti
import taichi.math as tm

f32 = ti.f32
i32 = ti.i32
ti.init(arch=ti.gpu) 

n = 100
numSteps = 1
particleRadius = 0.05
dt = 0.01
g = ti.Vector((0, 0, -9.81), ti.f32)
rho = 1000.0 # density
grid_size = (64, 64, 64)
reconstruction_resolution = (100, 100, 100)
reconstruction_threshold = 0.75
reconstruct_radius = 0.1
num_iter = 100

FLUID = 0
AIR = 1
SOLID = 2

mu = 0.6 # friction 
b_mu = 0.8 # boundary friction

# 

pressure = ti.field(f32, shape=grid_size)
cell_type = ti.field(i32, grid_size)
cell_type.fill(AIR)
#   create solid boundary 
for i, j in ti.ndrange(grid_size[0], grid_size[1]):
    cell_type[i, j, 0] = SOLID
    cell_type[i, j, grid_size[2]-1] = SOLID

for j, k in ti.ndrange(grid_size[1], grid_size[2]):
    cell_type[0, j, k] = SOLID
    cell_type[grid_size[0]-1, j, k] = SOLID

for i, k in ti.ndrange(grid_size[0], grid_size[2]):
    cell_type[i, 0, k] = SOLID
    cell_type[i, grid_size[1]-1, k] = SOLID    

# grid velocity
grid_v_x = ti.field(f32, shape=(grid_size[0]+1, grid_size[1], grid_size[2]))
grid_v_y = ti.field(f32, shape=(grid_size[0], grid_size[1]+1, grid_size[2]))
grid_v_z = ti.field(f32, shape=(grid_size[0], grid_size[1], grid_size[2]+1))

# grid weight 
grid_w_x = ti.field(f32, shape=(grid_size[0]+1, grid_size[1], grid_size[2]))
grid_w_y = ti.field(f32, shape=(grid_size[0], grid_size[1]+1, grid_size[2]))
grid_w_z = ti.field(f32, shape=(grid_size[0], grid_size[1], grid_size[2]+1))

divergence = ti.field(f32, shape=grid_size)
new_pressure = ti.field(f32, shape=grid_size)


#

# number of particle
n = 100000
pos = ti.Vector.field(3, ti.f32, shape=(n))
vel = ti.Vector.field(3, ti.f32, shape=(n))


@ti.kernel
def init():
    range_min = ti.Vector([16, 16, 0])
    range_max = ti.Vector([48, 48, 40])
    particle_init_size = range_max - range_min
    for p in pos:
        pl = p // 4 # 4 particle per grid
        k = pl % particle_init_size[2] + range_min[2]
        j = pl // particle_init_size[2] % particle_init_size[1] + range_min[1]
        i = pl // (particle_init_size[2] * particle_init_size[1]) % particle_init_size[0] + range_min[0]
        if cell_type[i, j, k] != SOLID:
            cell_type[i, j, k] = FLUID
        pos[p] = (ti.Vector([i, j, k]) + ti.Vector([ti.random(), ti.random(), ti.random()])) * 0.5

@ti.kernel
def clear_grid():
    grid_v_x.fill(0.0)
    grid_v_y.fill(0.0)
    grid_v_z.fill(0.0)

    grid_w_x.fill(0.0)
    grid_w_y.fill(0.0)
    grid_w_z.fill(0.0)

@ti.kernel
def p2g():
    for p in pos:
        x = pos[p]
        v = vel[p]

    
@ti.kernel
def apply_force():
    for i, j, k in grid_v_z:
        if k > 1:
            grid_v_z[i, j, k] -= 9.81* dt

@ti.kernel
def boundary_condition():
    for i, j in ti.ndrange(grid_size[0], grid_size[1]):
        grid_v_z[i, j, 0] = 0
        grid_v_z[i, j, 1] = 0
        grid_v_z[i, j, grid_size[2]-1] = 0
        grid_v_z[i, j, grid_size[2]] = 0

    for j, k in ti.ndrange(grid_size[1], grid_size[2]):
        grid_v_z[0, j, k] = 0
        grid_v_z[1, j, k] = 0
        grid_v_z[grid_size[0]-1, j, k] = 0
        grid_v_z[grid_size[0], j, k] = 0

    for i, k in ti.ndrange(grid_size[0], grid_size[1]):
        grid_v_z[i, 0, k] = 0
        grid_v_z[i, 1, k] = 0
        grid_v_z[i, grid_size[1]-1, k] = 0
        grid_v_z[i, grid_size[1], k] = 0

@ti.kernel 
def advection_particle():
    for p in pos:
        pos[p] += vel[p]


def update():
    apply_force()
    boundary_condition()
    #p2g()
    #solve_imcompressive()
    #g2p()
    advection_particle()



win_x = 640
win_y = 640

window = ti.ui.Window("flip 3d", 
(win_x, win_y), vsync=True
)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(30, 30, 30)
camera.lookat(0, 0, 0)
scene.ambient_light((0.5, 0.5, 0.5))
scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))


init()
while window.running:
    ti.deactivate_all_snodes()  
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
 
    scene.set_camera(camera)
    
    for s in range(numSteps):
        update()    
    
    scene.particles(pos, color = (0, 1, 1), radius = particleRadius)

    canvas.scene(scene)
    window.show()
