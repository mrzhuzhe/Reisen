import taichi as ti
import taichi.math as tm

f32 = ti.f32
i32 = ti.i32
ti.init(arch=ti.vulkan)

numSteps = 50
particleRadius = 0.01
dt = 2e-4
g = ti.Vector((0, -9.81, 0), ti.f32)
#g = ti.Vector((0, 0, 0), ti.f32)
bound = 3
#dx = 0.1 # grid quantitle size
dx = 1 / 128
rho = 1.0 # density
p_vol = (dx * 0.5)**2
p_mass = p_vol * rho
E = 400


grid_size = (128, 128, 128)

# grid velocity
grid_v = ti.Vector.field(3, f32, shape=(grid_size[0], grid_size[1], grid_size[2]))
grid_m = ti.field(float, (grid_size[0], grid_size[1], grid_size[2]))
#

# number of particle
n = 10000
pos = ti.Vector.field(3, ti.f32, shape=(n))
vel = ti.Vector.field(3, ti.f32, shape=(n))
C = ti.Matrix.field(3, 3, ti.f32, shape=(n))
J = ti.field(float, n)

@ti.kernel
def init():
    for i in range(n):
        pos[i] = ti.Vector([ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2])
        vel[i] = [0, -1, 0]
        J[i] = 1

@ti.kernel
def clear_grid():
    grid_v.fill(0.0)
    grid_m.fill(0.0)


@ti.kernel
def p2g():
    for p in pos:
        x = pos[p]
        v = vel[p]
        idx = x/dx
        #base = ti.cast(ti.floor(idx), i32)
        base = int(idx - 0.5)
        frac = idx - base
        cp = C[p]
        jp = J[p]     
        interp_grid(base, frac, v, cp, jp)
        """
        w = [0.5 * (1.5 - frac)**2, 0.75 - (frac - 1)**2, 0.5 * (frac - 0.5)**2]

        stress = -dt * 4 * E * p_vol * (jp - 1) / dx**2
        affine = ti.Matrix([[stress, 0, 0], [0, stress, 0], [0, 0, stress]]) + p_mass * cp

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # [simplify.cpp:visit@568] Nested struct-fors are not supported for now. Please try to use range-fors for inner loops        
            offset = ti.Vector([i, j, k])
            dpos = (offset - frac) * dx
            weight = w[i].x * w[j].y * w[k].z
            grid_v[base + offset] += weight * (p_mass * vel[p] + affine @ dpos)
            #print(p_mass * vel[p], affine @ dpos)
            #print(p_mass * vel[p] + affine @ dpos)
            #print(weight)
            #grid_v[base + offset] += weight * (p_mass * vel[p])
            #val = weight * ( affine @ dpos)
            #grid_v[base + offset] += weight * (p_mass * vel[p]) + val
            grid_m[base + offset] += weight * p_mass
        """
        
                
    for i, j, k in grid_m:
    #for i, j, k in ti.ndrange(grid_size[0], grid_size[1], grid_size[2]):
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] /= grid_m[i, j, k]
#"""
@ti.func
def interp_grid(base, frac, vp, cp, jp):
    
    w = [0.5 * (1.5 - frac)**2, 0.75 - (frac - 1)**2, 0.5 * (frac - 0.5)**2]

    stress = -dt * 4 * E * p_vol * (jp - 1) / dx**2
    affine = ti.Matrix([[stress, 0, 0], [0, stress, 0], [0, 0, stress]]) + p_mass * cp

    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # [simplify.cpp:visit@568] Nested struct-fors are not supported for now. Please try to use range-fors for inner loops
    #for i, j, k in ti.ndrange(3, 3, 3):
    #for i in ti.static(range(3)):
    #    for j in ti.static(range(3)):
    #        for k in ti.static(range(3)):
        offset = ti.Vector([i, j, k])
        dpos = (offset - frac) * dx
        weight = w[i].x * w[j].y * w[k].z
        grid_v[base + offset] += weight * (p_mass * vp) + affine @ dpos
        grid_m[base + offset] += weight * p_mass            

#"""

    

@ti.kernel
def g2p():
    for p in pos:
        x = pos[p]
        idx = x/dx
        #base = ti.cast(ti.floor(idx), i32)
        base = int(idx - 0.5)
        frac = idx - base           
        interp_particle(base, frac, p)

@ti.func
def interp_particle(base, frac, p):
    w = [0.5 * (1.5 - frac)**2, 0.75 - (frac - 1)**2, 0.5 * (frac - 0.5)**2]
    new_v = ti.Vector.zero(float, 3)
    new_c = ti.Matrix.zero(float, 3, 3)
    #for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
    #for i in ti.static(range(3)):
    #   for j in ti.static(range(3)):
    #        for k in ti.static(range(3)):
    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): 
        offset = ti.Vector([i, j, k])
        dpos = (offset - frac) * dx
        weight = w[i].x * w[j].y * w[k].z
        g_v = grid_v[base + offset]      
        new_v += weight * g_v
        # 4 need to be changed 
        new_c += 4 * weight * g_v.outer_product(dpos) / dx**2    
      
    vel[p] = new_v
    
    J[p] *= 1 + dt * new_c.trace()
    C[p] = new_c
    #advection 
    #pos[p] += dt * vel[p]


@ti.kernel
def apply_force():
    for i, j, k in grid_m:
    #for i, j, k in ti.ndrange(grid_size[0], grid_size[1], grid_size[2]):
        grid_v[i, j, k] += g * dt

@ti.kernel
def boundary_condition():
    for i, j, k in ti.ndrange(grid_size[0], grid_size[1], grid_size[2]):
        if i < bound and grid_v[i, j, k].x < 0:
            grid_v[i, j, k].x = 0
        if i > grid_size[0] - bound and grid_v[i, j, k].x > 0:
            grid_v[i, j, k].x = 0

        if j < bound and grid_v[i, j, k].y < 0:
            grid_v[i, j, k].y = 0
        if j > grid_size[1] - bound and grid_v[i, j, k].y > 0:
            grid_v[i, j, k].y = 0

        if j < bound and grid_v[i, j, k].z < 0:
            grid_v[i, j, k].z = 0
        if j > grid_size[2] - bound and grid_v[i, j, k].z > 0:
            grid_v[i, j, k].z = 0
        

@ti.kernel 
def advection_particle():    
    for p in pos:
    #for p in ti.ndrange(n):
        pos[p] += dt * vel[p] 


def update():
    clear_grid()
    #p2g()
    apply_force()
    boundary_condition()
    g2p()
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
camera.position(1, 1, 1)
camera.lookat(0, 0, 0)
scene.ambient_light((0.5, 0.5, 0.5))
scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))


init()


while window.running:
    ti.deactivate_all_snodes()  
    camera.track_user_inputs(window, movement_speed=0.5, hold_key=ti.ui.RMB)
 
    scene.set_camera(camera)
    
    for s in range(numSteps):
        update()    
    
    scene.particles(pos, color = (0, 1, 1), radius = particleRadius)

    canvas.scene(scene)
    window.show()
