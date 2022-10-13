import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu) 

vec3f = ti.types.vector(3, ti.f32)
gravity = vec3f(0, -9.8, 0)
dt = 0.01
numSteps = 10
sdt = dt / numSteps
n = 9000

minX = 3
minZ = 3
particleRadius = 0.1
maxVel = 0.4 * particleRadius
kernelRadius = 3.0 * particleRadius
particleDiameter = 2 * particleRadius
restDensity = 1.0 / (particleDiameter * particleDiameter)

viscosity = 3
h = kernelRadius
h2 = h * h
PI = 3.14
kernelScale = 4.0 / (PI * h2 * h2 * h2 * h2);		

pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
prepos = ti.Vector.field(3, dtype=ti.f32, shape=n)
vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
grads = ti.Vector.field(3, dtype=ti.f32, shape=n)

@ti.func
def findNeighbors():
    pass

@ti.func
def solveBoundaries():
    for i in range(n):
        if pos[i][1] < 0:
            pos[i][1] = 0
        if (pos[i][0] < -minX): 
            pos[i][0] = -minX
        if (pos[i][0] > minX):
            pos[i][0] = minX; 
        if (pos[i][2] < -minZ): 
            pos[i][2] = -minZ
        if (pos[i][2] > minZ):
            pos[i][2] = minZ; 

@ti.func
def applyViscosity(i, sdt):
    avgVel = vec3f(0, 0, 0)
			
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
        _gradient = vec3f(0, 0, 0)

        # all Neighbors
        for j in range(n):
            _dist = pos[j] - pos[i]
            _norm = _dist.norm()
            if _norm > 0:
                _dist = _dist.normalized()

            if _norm > h:
                grads[j] = vec3f(0, 0, 0)
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

        for j in range(n):			
            if (j == i) :
                pos[j] += _lambda * _gradient
            else:
                pos[j]  += _lambda * grads[j]
                    

@ti.kernel
def init():
    _w = 10 
    _h = 10
    for i in range(n):
        _y = i // (_h * _w)
        _cur = i % (_h * _w)
        pos[i] = 0.3 * vec3f(_cur%_w, _y, _cur//_w)


@ti.kernel
def update():

    #findNeighbors()

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

window = ti.ui.Window("simple pendulum", (win_x, win_y))
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(15, 15, 15)
camera.lookat(0, 0, 0)
scene.ambient_light((0.5, 0.5, 0.5))
scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

step = 0
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
    step +=1 
