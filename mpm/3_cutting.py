# reffer https://github.com/taichi-dev/taichi_elements
# 2d description https://github.com/taichi-dev/mls_mpm_88_extensions
# why 0.5 offset https://forum.taichi-lang.cn/t/topic/1584/4
# ray traycing https://github.com/HK-SHAO/RayTracingPBR/tree/taichi


# aot https://github.com/taichi-dev/taichi-aot-demo
# aot https://docs.taichi-lang.org/blog/taichi-aot-the-solution-for-deploying-kernels-in-mobile-devices
# https://docs.taichi-lang.org/docs/taichi_core#load-and-destroy-a-taichi-aot-module


import taichi as ti
import taichi.math as tm

f32 = ti.f32
i32 = ti.i32
ti.init(arch=ti.gpu)

numSteps = 25
particleRadius = 0.002
dt = 1e-4 # 2e-4 not move
g = ti.Vector((0, -9.81, 0), ti.f32)
bound = 3
#dx = 0.1 # grid quantitle size
grid_n = 32
dx = 1 / grid_n
dx_inv = float(grid_n)
rho = 1.0 # density
p_vol = (dx * 0.5)**2
p_mass = p_vol * rho
E = 100 #400  # checkborar pattern
nu = 0.2
mu_0 = E / (2 *( 1 + nu ))
lambda_0 = E * nu / ((1+nu)*(1-2*nu)) # lame parameters

grid_size = (grid_n, grid_n, grid_n)

# grid velocity
grid_v = ti.Vector.field(3, f32, shape=(grid_size[0], grid_size[1], grid_size[2]))
grid_m = ti.field(float, (grid_size[0], grid_size[1], grid_size[2]))
#

# number of particle
n = int(1 * 1e5)#100000
pos = ti.Vector.field(3, ti.f32, shape=(n))
vel = ti.Vector.field(3, ti.f32, shape=(n))
C = ti.Matrix.field(3, 3, ti.f32, shape=(n)) # affine velocity
J = ti.field(float, n) # plastic deformation
F = ti.Matrix.field(3, 3, ti.f32, shape=(n)) # defomation gradient
material = ti.field(int, shape=(n))
color = ti.Vector.field(3, ti.f32, shape=(n))


@ti.kernel
def init():
    group = 9
    group_size = n // group
    colorArr = ti.Vector([
        [0, 1, 1], [1, 1, 0], [1, 0, 1], 
        [1, 0, 0], [0, 1, 0], [0, 0, 1], 
        [0.6, 0.3, 0.2], [0.4, 0.5, 0.3], [0.3, 0.5, 0.7]
        ])    
    
    for i in ti.ndrange(n):
        _gid = i // group_size
        _col = _gid//3
        _row = _gid%3
        pos[i] = ti.Vector([
            ti.random() * 0.1 + 0.3 * _col, 
            ti.random() * 0.3 + 0.5 , 
            ti.random() * 0.1 + 0.2 * _row])
        vel[i] = [0, 0, 0]
        J[i] = 1
        F[i] = ti.Matrix.identity(float, 3)
        #material[i] = i // group_size % 3
        material[i] = 0
        color[i] = (colorArr[_gid, 0], 
                    colorArr[_gid, 1], 
                    colorArr[_gid, 2])

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
        frac = idx - base.cast(float)
        #cp = C[p]
        #jp = J[p]

        # F[p]: deformation gradient update
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]
        # h: Hardening coefficient: snow gets harder when compressed
        h = ti.exp(10 * (1.0 - J[p]))
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        
        U, sig, V = ti.svd(F[p])

        jc = 1.0
        for d in ti.static(range(3)):
            #new_sig = sig[d, d, d]
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                #new_sig = ti.min(ti.max(sig[d, d, d], 1 - 2.5e-2),
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2),
                                 1 + 4.5e-3)  # Plasticity
            #jp *= sig[d, d, d] / new_sig
            J[p] *= sig[d, d] / new_sig
            #sig[d, d, d] = new_sig
            sig[d, d] = new_sig
            jc *= new_sig
        if material[p] == 0:
            # Reset deformation gradient to avoid numerical instability
            #new_F = ti.Matrix.identity(float, 3) * ti.sqrt(jc)
            new_F = ti.Matrix.identity(float, 3) # this is borrow from taichi element
            new_F[0, 0] = jc
            F[p] = new_F

        elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U @ sig @ V.transpose()
        
        #F[p] = fp
        #J[p] = jp

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * jc * (jc - 1)
        #stress = (-dt * p_vol * 4 ) * stress / dx**2
        stress = (-dt * p_vol * 4 * dx_inv **2 ) * stress
        affine = stress + p_mass * C[p]

        interp_grid(base, frac, v, affine)        
                        
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] /= grid_m[i, j, k]


@ti.func
def interp_grid(base, frac, vp, affine):
    # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
    w = [0.5 * (1.5 - frac)**2, 0.75 - (frac - 1)**2, 0.5 * (frac - 0.5)**2]
    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): # [simplify.cpp:visit@568] Nested struct-fors are not supported for now. Please try to use range-fors for inner loops   
        offset = ti.Vector([i, j, k])
        dpos = (offset - frac) * dx  
        weight = w[i].x * w[j].y * w[k].z
        grid_v[base + offset] += weight * (p_mass * vp + affine @ dpos)
        grid_m[base + offset] += weight * p_mass            

    

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
    for i, j, k in ti.static(ti.ndrange(3, 3, 3)): 
        offset = ti.Vector([i, j, k])
        #dpos = (offset - frac) * dx 
        dpos = (offset - frac)
        weight = w[i].x * w[j].y * w[k].z
        g_v = grid_v[base + offset]      
        new_v += weight * g_v
        # 4 need to be changed 
        #new_c += 4 * weight * g_v.outer_product(dpos) / dx**2  
        new_c += 4 * weight * g_v.outer_product(dpos) * dx_inv
    vel[p] = new_v
    
    #J[p] *= 1 + dt * new_c.trace()
    C[p] = new_c


@ti.kernel
def apply_force():
    for i, j, k in grid_m:
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

        if k < bound and grid_v[i, j, k].z < 0:
            grid_v[i, j, k].z = 0
        if k > grid_size[2] - bound and grid_v[i, j, k].z > 0:
            grid_v[i, j, k].z = 0
        
        # wall
        """        
        x_bd = 16
        bd_w = 1
        j_bd= 6
        if i >= x_bd and i <= x_bd + bd_w and j <= j_bd:
            grid_v[i, j, k].x = 0
            grid_v[i, j, k].y = 0
            grid_v[i, j, k].z = 0
        """



@ti.kernel 
def advection_particle():    
    for p in pos:       
        pos[p] += dt * vel[p]        
        #   cutting
        edge_x = 0.35
        edge_y = 0.7
        edge_w = 0.01
        if pos[p].y > 0 and pos[p].y  < edge_y and abs(pos[p].x - edge_x) < edge_w:
            if pos[p].x  >= edge_x:
                vel[p].x = (edge_w - abs(pos[p].x - edge_x))*10000
            else:
                vel[p].x = -(edge_w - abs(pos[p].x - edge_x))*10000

def setBoundary():
    box_anchors[0] = ti.Vector([0.35, 0.5, 0.0])
    box_anchors[1] = ti.Vector([0.35, 0.5, 1.0])
    box_anchors[2] = ti.Vector([0.35, 0.0, 0.0])
    box_anchors[3] = ti.Vector([0.35, 0.0, 1.0])
    box_lines_indices[0] = 0
    box_lines_indices[1] = 1
    box_lines_indices[2] = 0
    box_lines_indices[3] = 2
    box_lines_indices[4] = 1
    box_lines_indices[5] = 3
    box_lines_indices[6] = 2
    box_lines_indices[7] = 3

def update():
    clear_grid()
    p2g()
    apply_force()
    boundary_condition()
    g2p()
    advection_particle()



win_x = 1280
win_y = 1280

window = ti.ui.Window("mpm 3d", 
(win_x, win_y), vsync=True
)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(2, 1, 2)
camera.lookat(0, 0, 0)
#scene.ambient_light((0.5, 0.5, 0.5))
scene.ambient_light((0.75, 0.75, 0.75))
#scene.point_light(pos=(0.1, 0.1, 0.1), color=(1, 1, 1))


init()

box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
box_lines_indices = ti.field(int, shape=(2 * 12))
setBoundary()

while window.running:
    ti.deactivate_all_snodes()  
    camera.track_user_inputs(window, movement_speed=0.5, hold_key=ti.ui.RMB)
 
    scene.set_camera(camera)
    
    for s in range(numSteps):
        update()    
    
    #scene.particles(pos, color = (0, 1, 1), radius = particleRadius)
    scene.particles(pos, 
                    #palette=[0x068587, 0xED553B, 0xEEEEF0],
                    #palette_indices=material, 
                    #color = (0, 1, 1),
                    per_vertex_color=color,
                    radius = particleRadius
                    )
    scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
    canvas.scene(scene)
    window.show()
