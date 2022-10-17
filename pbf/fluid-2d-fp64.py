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
n = 4000
epsilon = 1e-5

minX = 64
screen_to_world_ratio = 10
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
    pass

@ti.func
def solveBoundaries():
    for i in range(n):
        if pos[i][1] <= 0:
            pos[i][1] = 0
        if (pos[i][0] <= 0): 
            pos[i][0] = 0
        if (pos[i][0] >= minX):
            pos[i][0] = 0

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
        _gradient = vec3f([0.0, 0.0])
        _pos = pos[i]
        for j in range(n):		        
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
        for j in range(n):	
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
        pos[i] = 0.5 * vec3f(_cur%_w + 50 + ti.random(), _cur//_w + ti.random())

@ti.kernel
def update():

    # predict 
    for i in range(n):
        vel[i] += gravity * sdt
        prepos[i] = pos[i]
        pos[i] += vel[i] * sdt

    # solve
    solveBoundaries()
    solveFluid()

    # derive velocities
    """
    for i in range(n):
        deltaV = pos[i] - prepos[i]

        # CFL
        "" "
        _Vnorm = deltaV.norm()
        if _Vnorm > maxVel:
            deltaV *= maxVel / _Vnorm
            pos[i] = prepos[i] + deltaV
        "" "
        vel[i] = deltaV / sdt
        
        #applyViscosity(i, sdt)
    """
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
