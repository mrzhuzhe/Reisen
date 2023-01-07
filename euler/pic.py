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
h2 = 0.5 * h
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

@ti.kernel
def handleParticleCollisions(obstacleX, obstacleY, obstacleRadius):
    h = 1 / fInvSpacing
    r = particleRadius
	
    minX = h + r
    maxX = (fnumX - 1) * h - r
    minY = h + r
    maxY = (fnumY - 1) * h - r

    for i in ti.ndrange(numParticles):
        x, y = particlePos[i]

        if (x < minX):
            x = minX
            particleVel[i][0] = 0
        
        if (x > maxX):
            x = maxX
            particleVel[i][0] = 0

        if (y < minY):
            y = minY
            particleVel[i][1] = 0

        if (y > maxY):
            y = maxY
            particleVel[i][1] = 0

        particlePos[i][0] = x
        particlePos[i][1] = y

@ti.kernel
def updateParticleDensity():
    h1 = fInvSpacing
    h2 = 0.5 * h

    d = particleDensity
    d.fill(0)
    
    for i in ti.ndrange(numParticles):
        x, y = particlePos[i]

        x = clamp(x, h, (fnumX-1)*h)
        y = clamp(y, h, (fnumY-1)*h)

        x0 = ti.floor((x-h2)*h1)
        tx = ((x-h2)-x0*h)*h1
        x1 = ti.min(x0+1, fnumX-2)

        y0 = ti.floor((y-h2)*h1)
        ty = ((y - h2)-y0*h)*h1
        y1 = ti.min(y0+1, fnumY-2)
        
        sx = 1 - tx
        sy = 1 - ty

        if x0 < fnumX and y0 < fnumY: d[x0, y0] += sx * sy
        if x1 < fnumX and y0 < fnumY: d[x1, y0] += tx * sy
        if x1 < fnumX and y1 < fnumY: d[x1, y1] += tx * ty
        if x0 < fnumX and y1 < fnumY: d[x0, y1] += sx * ty
    
    if particleRestDensity == 0:
        sum = 0
        numFluidCells = 0

        for i in ti.ndrange(fNumCells):
            if cellType[i] == FLUID_CELL:
                sum +=d[i]
                numFluidCells += 1

        if numFluidCells > 0:
            particleRestDensity = sum / numFluidCells

def transferVelocities(toGrid, flipRatio):
    h1 = fInvSpacing
    if toGrid:
        for i, j in ti.ndrange(fnumX, fnumY):
            preU[i, j] = U[i, j]
            preV[i, j] = V[i, j]
        dU.fill(0)
        dV.fill(0)
        U.fill(0)
        V.fill(0)

        for i in ti.ndrange(fNumCells):
            cellType[i] = SOLID_CELL if S[i] == 0 else AIR_CELL

def solveIncompressibility(numIters, dt, overRelaxation, compensateDrift = True):
    
    pass

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