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
n = 300
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

viscosity = 0.03
h = kernelRadius 
h2 = h * h
_PI = math.pi
kernelScale = 4.0 / (_PI * h2 * h2 * h2 * h2)
	

pos = ti.Vector.field(2, dtype=_fp, shape=n)
prepos = ti.Vector.field(2, dtype=_fp, shape=n)
vel = ti.Vector.field(2, dtype=_fp, shape=n)

@ti.func
def findNeighbors():
    pass

@ti.func
def solveBoundaries():
    for i in range(n):
        if pos[i][1] <= 1:
            pos[i][1] = 1
        if (pos[i][0] <= 1): 
            pos[i][0] = 1
        if (pos[i][0] >= minX-1):
            pos[i][0] = minX-1

@ti.func
def applyViscosity(i, sdt):
    avgVel = vec3f(0, 0)
    _count = 0
    for j in range(n):
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
    #avgRho = 0.0
    for i in range(n):
        rho = 0.0
        sumGrad2 = 0.0        
        _gradient = vec3f([0.0, 0.0])
        _pos = pos[i]
        for j in range(n):
            _dist = pos[j] - _pos
            _norm = _dist.norm()
            if _norm < h:
                w, _dist = getDensityAndNormal(_norm, _dist)
                _grad = calculateGrad(w, _norm)                
                _gradient -= _grad * _dist 
                sumGrad2 += _grad * _grad
                rho += kernelScale * w * w * w
                         
        sumGrad2 += _gradient.dot(_gradient)
        #avgRho += rho
        _C = rho / restDensity - 1.0        
        if _C < 0:
            continue
        _lambda = -_C / (sumGrad2 + epsilon)

        for j in range(n):
            if (j == i):
               pos[j] += _lambda * _gradient
            else:
                _dist = pos[j] - _pos
                _norm = _dist.norm(eps=0)
                _grad = 0.0
                if _norm < h:
                    w, _dist = getDensityAndNormal(_norm, _dist)   
                    _grad = calculateGrad(w, _norm)                
                pos[j] += _lambda * _grad * _dist
        

@ti.kernel
def init():
    _w = 10
    for i in range(n):
        _cur = i
        pos[i] = vec3f(_cur%_w + 0.5 * ti.random(), _cur//_w  + 0.5 * ti.random()) + vec3f(10, 5)

@ti.kernel
def update():

    # predict 
    for i in ti.ndrange(n):
        vel[i] += gravity * sdt
        prepos[i] = pos[i]
        pos[i] += vel[i] * sdt

    # solve
    solveBoundaries()
    solveFluid()

    # derive velocities
    for i in ti.ndrange(n):
        deltaV = pos[i] - prepos[i]

        # CFL
        _Vnorm = deltaV.norm()
        if _Vnorm > maxVel:
            deltaV *= maxVel / _Vnorm
            pos[i] = prepos[i] + deltaV
        vel[i] = deltaV / sdt
        
        applyViscosity(i, sdt)
    #"""
win_x = 640
win_y = 640

gui = ti.GUI('fluid-2D', (win_x, win_y))

step = 0
init()

while gui.running:
    for s in range(numSteps):
        update()
    _pos = pos.to_numpy()
    gui.circles(_pos/minX, radius=particleRadius_show)
    
    gui.show()
