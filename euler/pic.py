from xml.etree.ElementTree import PI
from numpy import int32
from psutil import pid_exists
import taichi as ti

ti.init(arch=ti.gpu)

c_width = 1200
c_height = 1200
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

gravity = -9.81

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

#UV = ti.Vector.field(2, dtype=ti.f32, shape=(fnumX, fnumY))
#dUV = ti.Vector.field(2, dtype=ti.f32, shape=(fnumX, fnumY))
#preUV = ti.Vector.field(2, dtype=ti.f32, shape=(fnumX, fnumY))

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
#numCellParticles = ti.field(dtype=ti.i32, shape=(pNumX, pNumY))
numCellParticles = ti.field(dtype=ti.i32, shape=(pNumCells))
firstCellParticle = ti.field(dtype=ti.i32, shape=(pNumCells+1))

cellParticleIds =  ti.field(dtype=ti.i32, shape=numParticles)

print("pNumX pNumY", pNumX, pNumY)

pixels = ti.field(dtype=ti.f32, shape=(c_width, c_height))

@ti.func
def clamp(x: float, min: float, max: float):
	ret = x
	if x < min:
		ret = min
	if x > max:
		ret = max
	return ti.cast(ret, int) 

@ti.kernel
def init():
	dx = 2.0 * r
	dy = ti.sqrt(3.0) / 2.0 * dx
	for i, j in ti.ndrange(numX, numY):
		cnt = i* numY + j	
		particlePos[cnt][0] = h + r + dx * i + (0 if j % 2 == 0 else r)
		particlePos[cnt][1] = 10 * h + r + dy * j
		

@ti.kernel
def integrateParticles(dt: float, gravity: float):
	for i in ti.ndrange(numParticles):
		particleVel[i][1] += dt * gravity
		particlePos[i] += particleVel[i] * dt        

@ti.kernel
def pushParticlesApart(numIters: int):    
    # count particles per cell
	numCellParticles.fill(0)
	
	"""
	for i in ti.ndrange(numParticles):
		x, y = particlePos[i]

		xi = clamp(ti.floor(x * pInvSpacing), 0, pNumX-1)
		yi = clamp(ti.floor(y * pInvSpacing), 0, pNumY-1)		
		numCellParticles[xi, yi] += 1
	"""
	#ti.loop_config(serialize=True)
	for i in ti.ndrange(numParticles):
		x, y = ti.floor(particlePos[i] * pInvSpacing)
		xi = clamp(x, 0, pNumX-1)
		yi = clamp(y, 0, pNumY-1)
		numCellParticles[xi*pNumY+yi] += 1 # auto atomic add
		
		
	# partial sums 
	first = 0
	# [TODO] series
	#ti.loop_config(serialize=True)
	for i in ti.ndrange(pNumCells):
		#first += numCellParticles[i]
		#firstCellParticle[i] = first
		firstCellParticle[i] = ti.atomic_add(first, numCellParticles[i])
	firstCellParticle[pNumCells] = first
	
	"""
	ti.loop_config(serialize=True)
	for i in range(pNumCells+1):
		if firstCellParticle[i] < 10000:
			#print(numCellParticles[i])
			print(i, firstCellParticle[i])
	"""

	# fill particles into cells  
	#ti.loop_config(serialize=True)
	for i in ti.ndrange(numParticles):
		#x, y = particlePos[i]
		x, y = ti.floor(particlePos[i] * pInvSpacing)
		xi = clamp(x, 0, pNumX-1)
		yi = clamp(y, 0, pNumY-1)
		_ind = xi*pNumY+yi
		firstCellParticle[_ind] -= 1 #	auto atomic
		cellParticleIds[firstCellParticle[_ind]] = i

	
	minDist = 2 * particleRadius
	minDist2 = minDist * minDist

	for iter in range(numIters):
		#ti.loop_config(serialize=True)
		for i in ti.ndrange(numParticles):
			pxi, pyi = ti.cast(ti.floor(particlePos[i]* pInvSpacing), int)
			# TODO reduce cast type
			x0 = ti.max(pxi-1, 0)
			y0 = ti.max(pyi-1, 0)
			x1 = ti.min(pxi+1, pNumX-1)
			y1 = ti.min(pyi+1, pNumY-1)
			#ti.loop_config(serialize=True)
			for xi, yi in ti.ndrange((x0, x1), (y0, y1)):
				first = firstCellParticle[xi*pNumY+yi]
				last = firstCellParticle[xi*pNumY+yi+1]				
				for j in range(first, last):
					id = cellParticleIds[j]					
					if id == i:
						continue
					#qx, qy = particlePos[id]
					
					#dx = qx - px 
					#dy = qy - py
					delta = particlePos[id] - particlePos[i]
					#d2 = dx * dx + dy * dy
					d2 = delta.dot(delta)
					if (d2 > minDist2 or d2 == 0):
						continue
					#d = ti.sqrt(d2)
					d = delta.norm()
					s = 0.5 * (minDist -d) / d
					#dx *= s
					#dy *= s
					delta *= s
					#particlePos[i] -= [dx, dy]
					#particlePos[id] += [dx, dy]
					particlePos[i] -= delta
					particlePos[id] += delta
					
@ti.kernel
def handleParticleCollisions():
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

@ti.func
def project(x, y, dx, dy):
	h1 = fInvSpacing
	x0 = ti.min(ti.floor((x - dx)*h1), fnumX-2)
	tx = (x - dx  - x0*h) * h1
	x1 = ti.min(x0+1, fnumX-2)

	y0 = ti.min(ti.floor((y-dy)*h1), fnumY-2)
	ty = (y - dy - y0*h) * h1
	y1 = ti.min(y0+1, fnumY-2)

	sx = 1.0 - tx
	sy = 1.0 - ty

	d0 = sx * sy
	d1 = tx * sy
	d2 = tx * ty
	d3 = sx * ty
	return x0, x1, y0, y1, d0, d1, d2, d3

@ti.kernel
def p2g():
	h1 = fInvSpacing
    
	for i, j in ti.ndrange(fnumX, fnumY):
		preU[i, j] = U[i, j]
		preV[i, j] = V[i, j]
		#preUV[i, j] = UV[i, j]

	#dUV.fill(0)
	dU.fill(0)
	dV.fill(0)

	#UV.fill(0)
	U.fill(0)
	V.fill(0)

	for i in ti.ndrange(fNumCells):
		cellType[i] = SOLID_CELL if S[i] == 0 else AIR_CELL

	for i in ti.ndrange(numParticles):
		x, y =ti.floor(particlePos[i]* h1)
		xi = clamp(x, 0, fnumX-1)
		yi = clamp(y, 0, fnumY-1)
		_ind = xi * fnumY + yi
		cellType[_ind] = FLUID_CELL

	for i in range(numParticles):
		x, y = particlePos[i]
		x = clamp(x, h, (fnumX-1)*h)
		y = clamp(y, h, (fnumY-1)*h)
		
		Ux0, Ux1, Uy0, Uy1, Ud0, Ud1, Ud2, Ud3 = project(x, y, 0, h2)
		Vx0, Vx1, Vy0, Vy1, Vd0, Vd1, Vd2, Vd3 = project(x, y, h2, 0)
		
		pv0 = particleVel[i][0]		
		U[Ux0, Uy0] += pv0 * Ud0
		U[Ux1, Uy0] += pv0 * Ud1
		U[Ux1, Uy1] += pv0 * Ud2
		U[Ux0, Uy1] += pv0 * Ud3

		preU[Ux0, Uy0] += Ud0
		preU[Ux1, Uy0] += Ud1
		preU[Ux1, Uy1] += Ud2
		preU[Ux0, Uy1] += Ud3

		pv1 = particleVel[i][1]
		V[Vx0, Vy0] += pv1 * Vd0
		V[Vx1, Vy0] += pv1 * Vd1
		V[Vx1, Vy1] += pv1 * Vd2
		V[Vx0, Vy1] += pv1 * Vd3

		preV[Vx0, Vy0] += Vd0
		preV[Vx1, Vy0] += Vd1
		preV[Vx1, Vy1] += Vd2
		preV[Vx0, Vy1] += Vd3

	for i, i in ti.ndrange(fnumX, fnumY):
		if dU[i, j] > 0:
			U[i, j] /=  dU[i, j]
		if dV[i, j] > 0:
			V[i, j] /=  dV[i, j]

	for i,j in ti.ndrange(fnumX, fnumY):
		solid = cellType[i, j] == SOLID_CELL
		if (solid or (i > 0 and cellType[i-1, j] == SOLID_CELL)):
			U[i, j] = preU[i, j]
		if (solid or (j > 0 and cellType[i, j-1] == SOLID_CELL)):
			V[i, j] = preV[i, j]
	

@ti.kernel
def g2p():
	h1 = fInvSpacing
	for i in range(numParticles):
		x, y = particlePos[i]
		x = clamp(x, h, (fnumX-1)*h)
		y = clamp(y, h, (fnumY-1)*h)
		
		Ux0, Ux1, Uy0, Uy1, Ud0, Ud1, Ud2, Ud3 = project(x, y, 0, h2)
		Vx0, Vx1, Vy0, Vy1, Vd0, Vd1, Vd2, Vd3 = project(x, y, h2, 0)
		
		

@ti.kernel
def solveIncompressibility(numIters, dt, overRelaxation, compensateDrift = True):
    
    pass

def update():
	numSubSteps = 1
	sdt = dt / numSubSteps
	
	for step in range(numSubSteps):
		integrateParticles(sdt, gravity)
		#pushParticlesApart(numParticleIters)
		handleParticleCollisions()
		p2g()
		#updateParticleDensity()
		#solveIncompressibility(numPressureIters, sdt)
		#g2p()

gui = ti.GUI('PIC-2D', (c_width, c_height))

init()


while gui.running:
	update()

	pos = particlePos.to_numpy()
	#print(pos * c_width)
	gui.circles(pos , radius=2)

	#gui.set_image(pixels)
	gui.show()