import taichi as ti
import math

ti.init(arch=ti.gpu)

#gravity = -9.81
gravity = 0
dt = 1.0 / 120.0
numSteps = 100
radius = 0.15
res = 100
h = 1 / res

numX = 200
numY = 200
numCells = numX * numY

U = ti.Vector.field(1, dtype=ti.f32, shape=numCells)
V = ti.Vector.field(1, dtype=ti.f32, shape=numCells)
newU = ti.Vector.field(1, dtype=ti.f32, shape=numCells)
newV = ti.Vector.field(1, dtype=ti.f32, shape=numCells)
p = ti.Vector.field(1, dtype=ti.f32, shape=numCells)
s = ti.Vector.field(1, dtype=ti.f32, shape=numCells)
M = ti.Vector.field(1, dtype=ti.f32, shape=numCells)
newM = ti.Vector.field(1, dtype=ti.f32, shape=numCells) 

# external force	
def integrate(dt, gravity):
    pass

# project
def solveIncompressibility(numIters, dt):
    pass

def extrapolate():
    pass

def advectVel(dt):
    pass

def advectSmoke(dt):
    pass

"""
def setObstacle(x, y, reset):

		var vx = 0.0;
		var vy = 0.0;

		if (!reset) {
			vx = (x - scene.obstacleX) / scene.dt;
			vy = (y - scene.obstacleY) / scene.dt;
		}

		scene.obstacleX = x;
		scene.obstacleY = y;
		var r = scene.obstacleRadius;
		var f = scene.fluid;
		var n = f.numY;
		var cd = Math.sqrt(2) * f.h;

		for (var i = 1; i < f.numX-2; i++) {
			for (var j = 1; j < f.numY-2; j++) {

				f.s[i*n + j] = 1.0;

				dx = (i + 0.5) * f.h - x;
				dy = (j + 0.5) * f.h - y;

				if (dx * dx + dy * dy < r * r) {
					f.s[i*n + j] = 0.0;
					if (scene.sceneNr == 2) 
						f.m[i*n + j] = 0.5 + 0.5 * Math.sin(0.1 * scene.frameNr)
					else 
						f.m[i*n + j] = 1.0;
					f.u[i*n + j] = vx;
					f.u[(i+1)*n + j] = vx;
					f.v[i*n + j] = vy;
					f.v[i*n + j+1] = vy;
				}
			}
		}
		
		scene.showObstacle = true;
	
"""

def init():
    n = numY
    inVel = 2.0
    for i in range(numX):
        for j in range(numY):
            s = 1.0;	# fluid
            if (i == 0 | j == 0 | j == numY-1):
                s = 0.0;	# solid
            s[i*n + j] = s

            if (i == 1):
                u[i*n + j] = inVel
            


    pipeH = 0.1 * numY
    minJ = math.floor(0.5 * numY - 0.5*pipeH)
    maxJ = math.floor(0.5 * numY + 0.5*pipeH)

    for j in range(minJ, maxJ):
        M[j] = 0.0

    #setObstacle(0.4, 0.5, true)

    #pass

def update():
    pass

gui = ti.GUI('langrange-2D', (480, 480))

while gui.running:
    #for s in range(numSteps):
    #    update()
    #_pos = pos.to_numpy()
    #gui.circles(_pos/minX, radius=particleRadius_show)
    
    gui.show()