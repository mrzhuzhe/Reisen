import math
import taichi as ti
import taichi.math as tm
_fp  = ti.f32
ti.init(arch=ti.gpu, default_fp=_fp) 
vec3f = ti.types.vector(2, _fp)
gravity = vec3f([0, -9.8])
dt = 0.01
numSteps = 5
sdt = dt / numSteps
n = 400
epsilon = 1e-5

minX = 64
screen_to_world_ratio = 640 / minX
particleRadius = 0.3
particleRadius_show = particleRadius * screen_to_world_ratio
maxVel = 0.4 * particleRadius
kernelRadius = 3.0 * particleRadius
particleDiameter = 2 * particleRadius
restDensity = 1.0 / (particleDiameter * particleDiameter)
# 2d poly6 (SPH based shallow water simulation

viscosity = 3
h = kernelRadius 
h2 = h * h
_PI = math.pi
kernelScale = 4.0 / (_PI * h2 * h2 * h2 * h2)
	

pos = ti.Vector.field(2, dtype=_fp, shape=n)
prepos = ti.Vector.field(2, dtype=_fp, shape=n)
vel = ti.Vector.field(2, dtype=_fp, shape=n)
grads = ti.Vector.field(2, dtype=_fp, shape=n)
grid_size = kernelRadius * 1.5
grid_n = math.ceil(minX / grid_size)
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

# new 

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi
@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result

@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def findNeighbors():
    grain_count.fill(0)

    # count grain O(n)
    for i in range(n):
        grid_idx = ti.floor(pos[i]/minX * grid_n, int)
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
        grid_idx = ti.floor(pos[i]/minX * grid_n, int)      
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i

@ti.func
def solveBoundaries():
    for i in range(n):
        if pos[i][1] <= 0:
            pos[i][1] = 0
        if (pos[i][0] <= 0): 
            pos[i][0] = 0
        if (pos[i][0] >= minX):
            pos[i][0] = minX

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
    ti.loop_config(serialize=True)
    for i in range(n):
        rho = 0.0
        sumGrad2 = 0.0        
        _gradient = vec3f([0.0, 0.0])
        _pos = pos[i]

        grid_idx = ti.floor(pos[i]/minX * grid_n, int)
        #print(grid_idx)
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
                    _dist = pos[j] - _pos
                    _norm = _dist.norm(eps=0)
                    #"""
                    # over there
                    _grad = spiky_gradient(-_dist, h)
                    _gradient += _grad
                    sumGrad2 += _grad.dot(_grad)
                    rho += poly6_value(_norm, h)
                    #"""
                    """          

                    if _norm <=0 or _norm >= h:
                        grads[j] = vec3f(0, 0)              
                    else:
                        w = (h - _norm) /h/h/h
                        rho += kernelScale * w * w * w
                        g_factor = (kernelScale * 3.0 * w * w * (-2.0)) / restDensity;	
                        _grad = g_factor * _dist / _norm
                        grads[j] = _grad
                        _gradient -= _grad
                        sumGrad2 += _grad.dot(_grad)
                    """

                   
                
        sumGrad2 += _gradient.dot(_gradient)
        avgRho += rho
        _C = rho / restDensity - 1.0        
        if _C < 0:
            continue
        #if (sumGrad2 < 10): print(sumGrad2)
        _lambda = -_C / (sumGrad2 + epsilon)
        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(list_head[neigh_linear_idx],
                                list_tail[neigh_linear_idx]):                    
                    j = particle_id[p_idx]	   
                    """
                    if (j == i):
                        pos[j] += _lambda * _gradient
                    else:
                        pos[j] += _lambda * grads[j]
                    """
                    _dist = pos[j] - _pos
                    pos[j] += _lambda * spiky_gradient(_dist, h)


@ti.kernel
def init():
    _w = 10
    _h = 10
    for i in range(n):
        #_y = i // (_h * _w)
        #_cur = i % (_h * _w)
        _cur = i
        #pos[i] = 0.03 * vec3f(_cur%_w, _y, _cur//_w)
        pos[i] = 0.03 * vec3f(_cur%_w + ti.random(), _cur//_w + ti.random())

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
        _Vnorm = deltaV.norm()
        if _Vnorm > maxVel:
            deltaV *= maxVel / _Vnorm
            pos[i] = prepos[i] + deltaV        
        vel[i] = deltaV / sdt
        
        #applyViscosity(i, sdt)
   
win_x = 640
win_y = 640

gui = ti.GUI('Taichi DEM', (win_x, win_y))

step = 0
init()

while gui.running:
    for s in range(numSteps):
        update()
    _pos = pos.to_numpy()
    gui.circles(_pos/minX, radius=particleRadius_show)
    
    gui.show()
