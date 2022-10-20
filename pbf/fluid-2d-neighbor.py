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
n = 3000
epsilon = 1e-5

maxX = 64.0
screen_to_world_ratio = 640.0 / maxX
particleRadius = 0.3
particleRadius_show = particleRadius * screen_to_world_ratio
maxVel = 0.4 * particleRadius
kernelRadius = 3.0 * particleRadius
particleDiameter = 2.0 * particleRadius
restDensity = 1.0 / (particleDiameter * particleDiameter)
# 2d poly6 (SPH based shallow water simulation

viscosity = 0.3
h = kernelRadius 
h2 = h * h
_PI = math.pi
kernelScale = 4.0 / (_PI * h2 * h2 * h2 * h2)
	

pos = ti.Vector.field(2, dtype=_fp, shape=n)
prepos = ti.Vector.field(2, dtype=_fp, shape=n)
vel = ti.Vector.field(2, dtype=_fp, shape=n)
grads = ti.Vector.field(2, dtype=_fp, shape=n)

grid_size = 4.0
grid_n = math.ceil(maxX / grid_size)

print(f"Grid size: {grid_n}x{grid_n}")

assert particleRadius * 2 < grid_size

list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


@ti.func
def findNeighbors():

    grain_count.fill(0)
    for i in range(n):
        grid_idx = ti.floor(pos[i]/maxX * grid_n, int)
        grain_count[grid_idx] += 1
    
    column_sum.fill(0)
    # kernel comunicate with global variable ???? this is a bit amazing 
    for i, j in ti.ndrange(grid_n, grid_n):        
        ti.atomic_add(column_sum[i], grain_count[i, j])

    # this is because memory mapping can be out of order
    _prefix_sum_cur = 0
    
    for i in ti.ndrange(grid_n):
        prefix_sum[i] = ti.atomic_add(_prefix_sum_cur, column_sum[i])
    
    for i, j in ti.ndrange(grid_n, grid_n):        
        # we cannot visit prefix_sum[i,j] in this loop
        pre = ti.atomic_add(prefix_sum[i], grain_count[i, j])        
        linear_idx = i * grid_n  + j
        list_head[linear_idx] = pre
        list_cur[linear_idx] = list_head[linear_idx]
        # only pre pointer is useable
        list_tail[linear_idx] = pre + grain_count[i, j]       

    for i in range(n):
        grid_idx = ti.floor(pos[i]/maxX * grid_n, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i
        

@ti.func
def solveBoundaries():
    for i in range(n):
        if pos[i][1] <= 1:
            pos[i][1] = 1
        if pos[i][1] >= maxX - 1:
            pos[i][1] = maxX - 1
        if (pos[i][0] <= 1): 
            pos[i][0] = 1
        if (pos[i][0] >= maxX - 1 ):
            pos[i][0] = maxX -1 

@ti.func
def applyViscosity(i, sdt):
    avgVel = vec3f(0, 0)
    _count = 0

    grid_idx = ti.floor(pos[i]/maxX * grid_n, int)
    x_begin = max(grid_idx[0] - 1, 0)
    x_end = min(grid_idx[0] + 2, grid_n)
    y_begin = max(grid_idx[1] - 1, 0)
    y_end = min(grid_idx[1] + 2, grid_n)  
    for neigh_i, neigh_j in ti.ndrange((x_begin, x_end), (y_begin, y_end)):
            neigh_linear_idx = neigh_i * grid_n + neigh_j
            for p_idx in range(list_head[neigh_linear_idx],
                            list_tail[neigh_linear_idx]):                    
                j = particle_id[p_idx]
                _dist = pos[i] - pos[j]
                if _dist.norm() < h:			
                    avgVel += vel[j]
                    _count += 1
    
    if _count > 0:                
        avgVel /= _count        
        _delta = avgVel -  vel[i]        
        vel[i] += viscosity * _delta

@ti.func
def getDensityAndNormal(_norm: float, dist: ti.template()):
    r2 = _norm * _norm 
    w = (h2 - r2) 
    if _norm > 0:
        dist = dist.normalized()
    return w, dist
    
@ti.func
def calculateGrad(w: float, _norm: float):    
    return  (kernelScale * 3.0 * w * w * (-2.0 * _norm)) / restDensity


@ti.func
def solveFluid():

    avgRho = 0.0
    for i in range(n):
        rho = 0.0
        sumGrad2 = 0.0        
        _gradient = vec3f([0.0, 0.0])
        _pos = pos[i]

        grid_idx = ti.floor(pos[i]/maxX * grid_n, int)

        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)  
        for neigh_i, neigh_j in ti.ndrange((x_begin, x_end), (y_begin, y_end)):
            neigh_linear_idx = neigh_i * grid_n + neigh_j
            for p_idx in range(list_head[neigh_linear_idx],
                            list_tail[neigh_linear_idx]):                    
                j = particle_id[p_idx]
                _dist = pos[j] - _pos
                _norm = _dist.norm()
                if _norm < h:
                    w, _dist = getDensityAndNormal(_norm, _dist)
                    _grad = calculateGrad(w, _norm)                
                    _gradient -= _grad * _dist 
                    sumGrad2 += _grad * _grad
                    rho += kernelScale * w * w * w                   
        #print(grid_idx, _count)
        sumGrad2 += _gradient.dot(_gradient)
        avgRho += rho
        _C = rho / restDensity - 1.0        
        if _C < 0:
            continue
        _lambda = -_C / (sumGrad2 + epsilon)
        for neigh_i, neigh_j in ti.ndrange((x_begin, x_end), (y_begin, y_end)):
            neigh_linear_idx = neigh_i * grid_n + neigh_j
            for p_idx in range(list_head[neigh_linear_idx],
                            list_tail[neigh_linear_idx]):                    
                j = particle_id[p_idx]	   
                if (j == i):
                    pos[j] += _lambda * _gradient
                else:
                    _dist = pos[j] - _pos
                    _norm = _dist.norm()
                    _grad = 0.0
                    if _norm < h:
                        w, _dist = getDensityAndNormal(_norm, _dist)   
                        _grad = calculateGrad(w, _norm)                
                    pos[j] += _lambda * _grad * _dist

@ti.kernel
def init():
    _w = 50
    for i in range(n):
        _cur = i
        pos[i] = vec3f(_cur%_w + 0.3 * ti.random(), _cur//_w  + 0.3 * ti.random()) + vec3f(1, 1)
        #pos[i] = vec3f(_cur%_w , _cur//_w  ) + vec3f(25, 0)

@ti.kernel
def update():
        

    # predict 
    for i in range(n):
        vel[i] += gravity * sdt
        prepos[i] = pos[i]
        pos[i] += vel[i] * sdt

    # solve
    solveBoundaries()

    findNeighbors()

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
        
        applyViscosity(i, sdt)
   
win_x = 640
win_y = 640

gui = ti.GUI('pbf 2d', (win_x, win_y))

step = 0
init()

while gui.running:
    for s in range(numSteps):
        update()
    _pos = pos.to_numpy()
    gui.circles(_pos/maxX, radius=particleRadius_show)
    
    gui.show()
