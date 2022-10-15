from math import ceil
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu) 

dim = 2

vec3f = ti.types.vector(dim, ti.f32)
if dim == 2:
    gravity = vec3f(0, -9.8)
else:
    gravity = vec3f(0, -9.8, 0)
dt = 0.01
numSteps = 5
sdt = dt / numSteps
n = 900

minX = 0.5
minZ = 0.5
particleRadius = 0.01
maxVel = 0.4 * particleRadius
kernelRadius = 3.0 * particleRadius
particleDiameter = 2 * particleRadius
restDensity = 1.0 / (particleDiameter * particleDiameter)
# 2d poly6 (SPH based shallow water simulation

viscosity = 3
h = kernelRadius
h2 = h * h
PI = 3.14
kernelScale = 4.0 / (PI * h2 * h2 * h2 * h2)
	

pos = ti.Vector.field(dim, dtype=ti.f32, shape=n)
prepos = ti.Vector.field(dim, dtype=ti.f32, shape=n)
vel = ti.Vector.field(dim, dtype=ti.f32, shape=n)
grads = ti.Vector.field(dim, dtype=ti.f32, shape=n)

grid_size = kernelRadius * 1.5
grid_n = ceil(0.5 / grid_size)
"""
grid_n = 16
grid_size = 0.5/ grid_n  # Simulation domain of size [0, 1]
"""
print(f"Grid size: {grid_n}x{grid_n}")

assert particleRadius * 2 < grid_size

list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


@ti.func
def findNeighbors():
    grain_count.fill(0)

    # count grain O(n)
    for i in range(n):
        grid_idx = ti.floor(pos[i] * grid_n, int)
        grain_count[grid_idx] += 1

    # count every horizon column O(1)
    for i in range(grid_n):
        sum = 0
        for j in range(grid_n):
            sum += grain_count[i, j]
        column_sum[i] = sum

    prefix_sum[0, 0] = 0

    #   O(1)
    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]
    
    #   O(1)
    for i in range(grid_n):
        for j in range(grid_n):
            if j == 0:
                prefix_sum[i, j] += grain_count[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]

            linear_idx = i * grid_n + j

            list_head[linear_idx] = prefix_sum[i, j] - grain_count[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    #   O(n)
    for i in range(n):
        grid_idx = ti.floor(pos[i] * grid_n, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i

@ti.func
def solveBoundaries():
    for i in range(n):
        if pos[i][1] < 0:
            pos[i][1] = 0
        if (pos[i][0] < 0): 
            pos[i][0] = 0
        if (pos[i][0] > minX):
            pos[i][0] = minX; 
        """
        
        if (pos[i][2] < 0): 
            pos[i][2] = 0
        if (pos[i][2] > minZ):
            pos[i][2] = minZ; 
        """

@ti.func
def applyViscosity(i, sdt):
    #avgVel = vec3f(0, 0, 0)
    avgVel = vec3f(0, 0)
    num = n
    for j in range(n):			
        avgVel += vel[j]
		
				
    avgVel /= num
    
    _delta = avgVel -  vel[i]
    
    vel[i] += viscosity * _delta

@ti.func
def solveFluid():

    avgRho = 0.0
    for i in range(n):
        rho = 0.0
        sumGrad2 = 0.0        
        #_gradient = vec3f(0, 0, 0)
        _gradient = vec3f(0, 0)

        grid_idx = ti.floor(pos[i] * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(list_head[neigh_linear_idx],
                                list_tail[neigh_linear_idx]):
                    j = particle_id[p_idx]
                    if i != j:
                        _dist = pos[j] - pos[i]
                        _norm = _dist.norm()
                        if _norm > 0:
                            _dist = _dist.normalized()

                        if _norm > h:
                            grads[j] = vec3f(0, 0)
                        else:
                            r2 = _norm * _norm
                            w = h2 - r2
                            rho += kernelScale * w * w * w
                            _grad = (kernelScale * 3.0 * w * w * (-2.0 * _norm)) / restDensity;	
                            grads[j] = _dist * _grad
                            _gradient -= _dist * _grad
                            sumGrad2 += _grad * _grad

        
            
        
        sumGrad2 += _gradient.norm_sqr()
        avgRho += rho
        
        C = rho / restDensity - 1.0
        if C < 0.0:
            continue
        _lambda = -C / (sumGrad2 + 0.0001)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(list_head[neigh_linear_idx],
                                list_tail[neigh_linear_idx]):
                    j = particle_id[p_idx]
                    if (j == i) :
                        pos[j] += _lambda * _gradient
                    else:
                        pos[j] += _lambda * grads[j]
                    

@ti.kernel
def init():
    _w = 10 
    _h = 10
    for i in range(n):
        #_y = i // (_h * _w)
        #_cur = i % (_h * _w)
        _cur = i
        #pos[i] = 0.03 * vec3f(_cur%_w, _y, _cur//_w)
        pos[i] = 0.03 * vec3f(_cur%_w, _cur//_w)


@ti.kernel
def update():

    findNeighbors()

    # predict 
    for i in range(n):
        vel[i] += gravity * sdt
        prepos[i] = pos[i]
        pos[i] += vel[i] * sdt

    # solve
    solveBoundaries()
    solveFluid()

    # derive velocities
    for i in range(n):
        deltaV = pos[i] - prepos[i]

        # CFL
        #"""
        _Vnorm = deltaV.norm()
        if _Vnorm > maxVel:
            deltaV *= maxVel / _Vnorm
            pos[i] = prepos[i] + deltaV
        #"""
        vel[i] = deltaV / sdt
        
        #applyViscosity(i, sdt)

win_x = 640
win_y = 640

"""
window = ti.ui.Window("simple pendulum", (win_x, win_y))
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(1, 1, 1)
camera.lookat(0, 0, 0)
scene.ambient_light((0.5, 0.5, 0.5))
scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
"""
gui = ti.GUI('Taichi DEM', (win_x, win_y))

step = 0
init()
"""
while window.running:

    ti.deactivate_all_snodes()  
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
 
    scene.set_camera(camera)
    
    for s in range(numSteps):
        update()    
    
    scene.particles(pos, color = (0, 1, 1), radius = particleRadius)

    canvas.scene(scene)
    window.show()
    """


while gui.running:
    for s in range(numSteps):
        update()
    _pos = pos.to_numpy()
    gui.circles(_pos, radius=particleRadius* win_x)
    
    gui.show()
    step +=1 
