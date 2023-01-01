import taichi as ti

ti.init(arch=ti.gpu)

gravity = -9.81
dt = 1.0 / 120.0
numSteps = 100

numX = 102
numY = 202
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

def update():
    pass

gui = ti.GUI('langrange-2D', (480, 480))

while gui.running:
    #for s in range(numSteps):
    #    update()
    #_pos = pos.to_numpy()
    #gui.circles(_pos/minX, radius=particleRadius_show)
    
    gui.show()