import taichi as ti
import taichi.math as tm

f32 = ti.f32
i32 = ti.i32
ti.init(arch=ti.vulkan)

numSteps = 1
particleRadius = 0.005
dt = 0.01
#g = ti.Vector((0, 0, -9.81), ti.f32)
rho = 1000.0 # density
grid_size = (64, 64, 64)
dx = 0.1 # grid quantitle size
#reconstruction_resolution = (100, 100, 100)
#reconstruction_threshold = 0.75
#reconstruct_radius = 0.1

FLUID = 0
AIR = 1
SOLID = 2

#mu = 0.6 # friction 
#b_mu = 0.8 # boundary friction

# 


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

pressure = ti.field(f32, shape=grid_size)
new_pressure = ti.field(f32, shape=grid_size)

#

# number of particle
n = 160000 #39936
pos = ti.Vector.field(3, ti.f32, shape=(n))
vel = ti.Vector.field(3, ti.f32, shape=(n))


@ti.kernel
def init():
    range_min = ti.Vector([16, 16, 16])
    range_max = ti.Vector([48, 48, 48])
    particle_init_size = range_max - range_min
    for p in pos:
        pl = p // 4 # 4 particle per grid
        k = pl % particle_init_size[2] + range_min[2]
        j = pl // particle_init_size[2] % particle_init_size[1] + range_min[1]
        i = pl // (particle_init_size[2] * particle_init_size[1]) % particle_init_size[0] + range_min[0]
        if cell_type[i, j, k] != SOLID:
            cell_type[i, j, k] = FLUID
        pos[p] = (ti.Vector([i, j, k]) + ti.Vector([ti.random(), ti.random(), ti.random()])) * dx

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
        idx = x/dx
        base = ti.cast(ti.floor(idx), i32)
        frac = idx - base         
        interp_grid(base, frac, v)
    
    for i, j, k in grid_v_x:
        v = grid_v_x[i, j, k]
        w = grid_w_x[i, j, k]
        grid_v_x[i, j, k] = v / w if w > 0.0 else 0.0

    for i, j, k in grid_v_y:
        v = grid_v_y[i, j, k]
        w = grid_w_y[i, j, k]
        grid_v_y[i, j, k] = v / w if w > 0.0 else 0.0

    for i, j, k in grid_v_z:
        v = grid_v_z[i, j, k]
        w = grid_w_z[i, j, k]
        grid_v_z[i, j, k] = v / w if w > 0.0 else 0.0

"""

#quadratic B-spline 
0.75-x^2                |x| in [0, 0.5]
0.5*(1.5-|x|)^2         |x| in [0.5, 1.5]
0                       |x| above 1.5

"""

@ti.func
def quadratic_kernel(x):
    w = ti.Vector([0.0 for _ in range(3)])
    for i in ti.static(range(3)): 
        if x[i] < 0.5:
            w[i] = 0.75 - x[i]**2
        elif x[i] < 1.5:
            w[i] = 0.5 * (1.5-x[i])**2
        else:
            w[i] = 0.0
    return w

@ti.func
def interp_grid(base, frac, vp):
    
    # Index on sides
    #idx_side = ti.Vector([base-1, base, base+1, base+2])
    idx_side = [base-1, base, base+1, base+2]

    # Weight on sides
    #w_side = ti.Vector([quadratic_kernel(1.0+frac), quadratic_kernel(frac), quadratic_kernel(1.0-frac), quadratic_kernel(2.0-frac)])
    w_side = [quadratic_kernel(1.0+frac), quadratic_kernel(frac), quadratic_kernel(1.0-frac), quadratic_kernel(2.0-frac)]
    
    # Index on center 
    #idx_center = ti.Vector([base-1, base, base+1])
    idx_center = [base-1, base, base+1]

    # weight on center 
    #w_center= ti.Vector([quadratic_kernel(0.5+frac), quadratic_kernel(ti.abs(0.5-frac)), quadratic_kernel(ti.abs(0.5-frac)), quadratic_kernel(1.5-frac)])
    w_center= [quadratic_kernel(0.5+frac), quadratic_kernel(ti.abs(0.5-frac)), quadratic_kernel(1.5-frac)]

    #for i, j, k in ti.ndrange(4, 3, 3):
    for i in ti.static(range(4)):
        for j in ti.static(range(3)):
            for k in ti.static(range(3)):
                w = w_side[i].x * w_center[j].y * w_center[k].z
                idx = (idx_side[i].x, idx_center[j].y, idx_center[k].z)
                grid_v_x[idx] += vp.x * w
                grid_w_x[idx] += w
        

    #for i, j, k in ti.ndrange(3, 4, 3):
    for i in ti.static(range(3)):
        for j in ti.static(range(4)):
            for k in ti.static(range(3)):
                w = w_center[i].x * w_side[j].y * w_center[k].z
                idx = (idx_center[i].x, idx_side[j].y, idx_center[k].z)
                grid_v_y[idx] += vp.y *w
                grid_w_y[idx] += w

    #for i, j, k in ti.ndrange(3, 3, 4):
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            for k in ti.static(range(4)):
                w = w_center[i].x * w_center[j].y * w_side[k].z
                idx = (idx_center[i].x, idx_center[j].y, idx_side[k].z)
                grid_v_z[idx] += vp.z * w
                grid_w_z[idx] += w
    

@ti.kernel
def g2p():
    for p in pos:
        x = pos[p]
        idx = x/dx
        base = ti.cast(ti.floor(idx), i32)
        frac = idx - base           
        interp_particle(base, frac, p)

@ti.func
def interp_particle(base, frac, p):
    # Index on sides
    #idx_side = ti.Vector([base-1, base, base+1, base+2])
    idx_side = [base-1, base, base+1, base+2]

    # Weight on sides
    #w_side = ti.Vector([quadratic_kernel(1.0+frac), quadratic_kernel(frac), quadratic_kernel(1.0-frac), quadratic_kernel(2.0-frac)])
    w_side = [quadratic_kernel(1.0+frac), quadratic_kernel(frac), quadratic_kernel(1.0-frac), quadratic_kernel(2.0-frac)]
    
    # Index on centers
    #idx_center = ti.Vector([base-1, base, base+1])
    idx_center = [base-1, base, base+1]

    # Weight on centers
    #w_center = ti.Vector([quadratic_kernel(0.5+frac), quadratic_kernel(ti.abs(0.5-frac)), quadratic_kernel(ti.abs(0.5-frac)), quadratic_kernel(1.5-frac)])
    w_center= [quadratic_kernel(0.5+frac), quadratic_kernel(ti.abs(0.5-frac)), quadratic_kernel(1.5-frac)]

    wx = 0.0
    wy = 0.0
    wz = 0.0
    vx = 0.0
    vy = 0.0
    vz = 0.0
    
    # ti.static complier time unroll  https://docs.taichi-lang.org/blog/ast-refactoring#dealing-with-break-and-continue-in-compile-time-loop-unrolling
    #for i, j, k in ti.static(ti.ndrange(4, 3, 3)):
    for i in ti.static(range(4)):
        for j in ti.static(range(3)):
            for k in ti.static(range(3)):
                w = w_side[i].x * w_center[j].y * w_center[k].z
                idx = (idx_side[i].x, idx_center[j].y, idx_center[k].z)                
                vtemp = grid_v_x[idx] * w
                vx += vtemp
                wx += w

    #for i, j, k in ti.ndrange(3, 4, 3):
    for i in ti.static(range(3)):
        for j in ti.static(range(4)):
            for k in ti.static(range(3)):
                w = w_center[i].x * w_side[j].y * w_center[k].z
                idx = (idx_center[i].x, idx_side[j].y, idx_center[k].z)
                vtemp = grid_v_y[idx] * w
                vy += vtemp
                wy += w

    
    #for i, j, k in ti.ndrange(3, 3, 4):
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            for k in ti.static(range(4)):
                w = w_center[i].x * w_center[j].y * w_side[k].z
                idx = (idx_center[i].x, idx_center[j].y, idx_side[k].z)
                vtemp = grid_v_z[idx] * w
                vz += vtemp
                wz += w

    vel[p] = ti.Vector([vx/wx, vy/wy, vz/wz])


@ti.kernel
def apply_force():
    for i, j, k in grid_v_z:
        if j > 1:
            grid_v_y[i, j, k] -= 9.81* dt / 10

@ti.kernel
def boundary_condition():
    for i, j in ti.ndrange(grid_size[0], grid_size[1]):
        grid_v_z[i, j, 0] = 0
        grid_v_z[i, j, 1] = 0
        grid_v_z[i, j, grid_size[2]-1] = 0
        grid_v_z[i, j, grid_size[2]] = 0

    for j, k in ti.ndrange(grid_size[1], grid_size[2]):
        grid_v_x[0, j, k] = 0
        grid_v_x[1, j, k] = 0
        grid_v_x[grid_size[0]-1, j, k] = 0
        grid_v_x[grid_size[0], j, k] = 0

    for i, k in ti.ndrange(grid_size[0], grid_size[2]):
        grid_v_y[i, 0, k] = 0
        grid_v_y[i, 1, k] = 0
        grid_v_y[i, grid_size[1]-1, k] = 0
        grid_v_y[i, grid_size[1], k] = 0

@ti.kernel 
def advection_particle():
    for p in pos:
        pos[p] += vel[p]

    for p in pos:
        _p = pos[p]
        _v = vel[p]
        #"""
        for i in ti.static(range(3)):
            if _p[i] <= dx:
                _p[i] = dx
                _v[i] = 0
            if _p[i] >= grid_size[i] * dx - dx:
                _p[i] = grid_size[i] * dx - dx
                _v[i] = 0
        #"""        
        pos[p] = _p
        vel[p] = _v

@ti.kernel
def compute_divergence():
    for i, j, k in divergence:
        if not is_solid(i, j, k):
            div = grid_v_x[i+1, j, k] - grid_v_x[i, j, k]
            div += grid_v_y[i, j+1, k] - grid_v_y[i, j, k]
            div += grid_v_z[i, j, k+1] - grid_v_z[i, j, k]
            divergence[i, j, k] = div
        else:
            divergence[i, j, k] = 0
        divergence[i, j, k] /= dx

bound = 0
@ti.func
def is_valid(i, j, k):
    return bound <= i < grid_size[0] - bound and bound <= j < grid_size[1] - bound and bound <= k < grid_size[2] - bound

@ti.func
def is_solid(i, j, k):
    return is_valid(i, j, k) and cell_type[i, j, k] == SOLID

@ti.func
def is_fluid(i, j, k):
    return is_valid(i, j, k) and cell_type[i, j, k] == FLUID

damped_jacobi_weight = 1
@ti.kernel
def jacobi_iter():
    for i, j, k in pressure:
        if is_fluid(i, j, k):
            div = divergence[i, j, k]
            p_x1 = pressure[i-1, j, k]
            p_x2 = pressure[i+1, j, k]
            p_y1 = pressure[i, j-1, k]
            p_y2 = pressure[i, j+1, k]
            p_z1 = pressure[i, j, k-1]
            p_z2 = pressure[i, j, k+1]
            n = 6 
            if is_solid(i-1, j, k):
                p_x1 = 0.0
                n -=1
            if is_solid(i+1, j, k):
                p_x2 = 0.0
                n -=1
            if is_solid(i, j-1, k):
                p_y1 = 0.0
                n -=1
            if is_solid(i, j+1, k):
                p_y2 = 0.0
                n -=1
            if is_solid(i, j, k-1):
                p_z1 = 0.0
                n -=1
            if is_solid(i, j, k+1):
                p_z2 = 0.0
                n -=1
            
            #？？？？ 此处需要改成隐式求解
            new_pressure[i, j, k] = (1 - damped_jacobi_weight) * pressure[i, j, k] + damped_jacobi_weight * ( p_x1 + p_x2 + p_y1 + p_y2 + p_z1 + p_z2 - div * rho / dt * dx ** 2 ) / n
            #new_pressure[i, j, k] = ( p_x1 + p_x2 + p_y1 + p_y2 + p_z1 + p_z2 - div * rho / dt * dx**2 ) / n
        else:
            new_pressure[i, j, k] = 0.0


num_jacobian_iter = 100
def solve_pressure():
    for i in range(num_jacobian_iter):
        jacobi_iter()
        pressure.copy_from(new_pressure)

@ti.kernel
def project_velocity():     
    scale = dt / rho / dx
    for i, j, k in ti.ndrange(grid_size[0], grid_size[1], grid_size[2]):
        if is_fluid(i-1, j, k) or is_fluid(i, j, k):
            if is_solid(i-1, j, k) or is_solid(i, j, k):
                grid_v_x[i, j, k] = 0
            else:
                grid_v_x[i, j, k] -= scale * (pressure[i, j, k] - pressure[i-1, j, k])
        
        if is_fluid(i, j-1, k) or is_fluid(i, j, k):
            if is_solid(i, j-1, k) or is_solid(i, j, k):
                grid_v_y[i, j, k] = 0
            else:                
                grid_v_y[i, j, k] -= scale * (pressure[i, j, k]- pressure[i, j-1, k])

        if is_fluid(i, j, k-1) or is_fluid(i, j, k):
            if is_solid(i, j, k-1) or is_solid(i, j, k):
                grid_v_z[i, j, k] = 0
            else:                
                grid_v_z[i, j, k] -= scale * (pressure[i, j, k]- pressure[i, j, k-1])

@ti.kernel
def mark_celltype():
    for i, j, k in cell_type:
        if not is_solid(i, j, k):
            cell_type[i, j, k] = AIR
    
    for p in pos:
        x = pos[p]
        idx = ti.cast(ti.floor(x/dx), i32)
        if not is_solid(idx[0], idx[1], idx[2]):
            cell_type[idx] = FLUID

def update():
    clear_grid()
    boundary_condition()
    p2g()
    apply_force()

    compute_divergence()
    solve_pressure()
    project_velocity()

    g2p()
    advection_particle()
    mark_celltype()



win_x = 640
win_y = 640

window = ti.ui.Window("flip 3d", 
(win_x, win_y), vsync=True
)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(10, 1, 10)
camera.lookat(0, 1, 0)
scene.ambient_light((0.5, 0.5, 0.5))
scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))


init()


while window.running:
    ti.deactivate_all_snodes()  
    camera.track_user_inputs(window, movement_speed=0.5, hold_key=ti.ui.RMB)
 
    scene.set_camera(camera)
    
    #for s in range(numSteps):
    update()    
    
    scene.particles(pos, color = (0, 1, 1), radius = particleRadius)

    canvas.scene(scene)
    window.show()
