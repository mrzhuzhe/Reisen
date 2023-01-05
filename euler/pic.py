from pyparsing import col
import taichi as ti

ti.init(arch="gpu")

c_width = 600
c_height = 600

FLUID_CELL = 0
AIR_CELL = 1
SOLID_CELL = 2

density = 1000
dt = 1 / 60
res = 100
h = 1 / res
numPressureIters = 50
numParticleIters = 2


r =  0.3 * h
dx = 2.0 * r 
dy = ti.sqrt(3.0) / 2.0 * dx

numX = 100
numY = 100

fnumX = 1 / h + 1 
fnumY = 1 / h + 1
fInvSpacing = 1 / h
fNumCells = fnumX * fnumY

U = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))
V = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))
dU = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))
dV = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))
preU = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))
preV = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))

P = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))
S = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))

cellType = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))
cellColor = ti.Vector.field(3, dtype=ti.f32, shape=(fnumX, fnumY))



numParticles = numX * numY
particlePos = ti.Vector.field(2, dtype=ti.f32, shape=numParticles)
particleColor = ti.Vector.field(3, dtype=ti.f32, shape=numParticles)
# init color 

particleVel = ti.Vector.field(2, dtype=ti.f32, shape=numParticles)


pixels = ti.field(dtype=ti.f32, shape=(c_width, c_height))

@ti.func
def clamp(x: float, min: float, max: float):
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x

@ti.kernel
def init():
    cnt = 0
    for i, j in ti.ndrange(numX, numY):        
        particlePos[cnt][0] = h + r + dx * i + (0 if j % 2 == 0 else r)
        particlePos[cnt][1] = h + r + dy * j
        cnt +=1

def update():
	pass

def draw():
    pass

gui = ti.GUI('PIC-2D', (c_width, c_height))

init()


while gui.running:
    update()
    draw()

    pos = particlePos.to_numpy()
    #print(pos * c_width)
    gui.circles(pos , radius=2)

    #gui.set_image(pixels)
    gui.show()