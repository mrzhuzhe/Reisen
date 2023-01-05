from xml.etree.ElementTree import PI
from psutil import pid_exists
import taichi as ti

ti.init(arch=ti.gpu)

c_width = 600
c_height = 600
width = 1
height = 1

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

fnumX = int(1 / h + 1) 
fnumY = int(1 / h + 1)
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

particleDensity = ti.field(dtype=ti.f32, shape=fNumCells)
particleRestDensity = 0

particleRadius = r
pInvSpacing = 1.0 / (2.2 * particleRadius)

print("pInvSpacing", pInvSpacing)


pNumX = ti.floor(width * pInvSpacing) + 1
pNumY = ti.floor(height * pInvSpacing) + 1

pNumCells = pNumX * pNumY
numCellParticles = ti.field(dtype=ti.f32, shape=pNumCells)
firstCellParticle = ti.field(dtype=ti.f32, shape=(pNumCells+1))

cellParticleIds =  ti.field(dtype=ti.f32, shape=numParticles)

numParticles = 0
print("pNumX pNumY", pNumX, pNumY)

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

@ti.kernel
def integrateParticles(dt: float, gravity: float):
    for i in ti.ndrange(numParticles):
        particleVel[i][1] += dt * gravity
        particlePos[i] += particleVel[i] * dt        

@ti.kernel
def pushParticlesApart(numIters):
    colorDiffusionCoeff = 0.001
    
    # count particles per cell
    numCellParticles.fill(0)

    for i in ti.ndrange(numParticles):
        x, y = particlePos[i]

        xi = clamp(ti.floor(x * pInvSpacing), 0, pNumX-1)
        yi = clamp(ti.floor(y * pInvSpacing), 0, pNumY-1) 
        numCellParticles[xi, yi] += 1

    # partial sums 
    first = 0
    for i in ti.ndrange(pNumCells):
        first += numCellParticles[i]
        firstCellParticle[i] = first
    firstCellParticle[pNumCells] = first

    # fill particles into cells
    for i in ti.ndrange(numParticles):
        x, y = particlePos[i]
        xi = clamp(ti.floor(x* pInvSpacing), 0, pNumX-1)
        yi = clamp(ti.floor(y* pInvSpacing), 0, pNumY-1)
        firstCellParticle[xi, yi] -= 1
        cellParticleIds[firstCellParticle[xi, yi]] = i

    minDist = 2 * particleRadius
    minDist2 = minDist * minDist

    for iter in range(numIters):
        for i in ti.ndrange(numParticles):
            px, py = particlePos[i]
            pxi = ti.floor(px * pInvSpacing)
            pyi = ti.floor(py * pInvSpacing)
            x0 = ti.max(pxi-1, 0)
            y0 = ti.max(pyi-1, 0)
            x1 = ti.min(pxi+1, pNumX-1)
            y1 = ti.min(pyi+1, pNumY-1)

            for xi, yi in ti.ndrange((x0, x1), (y0, y1)):
                first = firstCellParticle[xi, yi]
                last = firstCellParticle[xi, yi+1]
                for j in range(first, last):
                    id = cellParticleIds[j]
                    if id == i:
                        continue
                    qx, qy = particlePos[id]
                    
                    dx = qx - px 
                    dy = qy - py
                    d2 = dx * dx + dy * dy
                    if (d2 > minDist2 or d2 == 0):
                        continue
                    d = ti.sqrt(d2)
                    s = 0.5 * (minDist -d) / d
                    dx *= S
                    dy *= S
                    particlePos[i] -= [dx, dy]
                    particlePos[id] += [dx, dy]



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