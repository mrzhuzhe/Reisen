# to 3d https://github.com/ethz-pbs21/SandyFluid

import taichi as ti
import taichi.math as tm


ti.init(arch=ti.vulkan)

c_width = 640
c_height = 640
width = 1
height = 1

FLUID_CELL = 0
AIR_CELL = 1
SOLID_CELL = 2

density = 1000
#dt = 1 / 60 # [TODO] this may because smaller step have less dissipation in energe
dt = 1/60
res = 100
numPressureIters = 50
numParticleIters = 2

gravity = -9.81

# num of particle
numX = 100
numY = 100

# num of feild(grid)
spacing = 1 / res
fnumX = int(1 / spacing + 1) 
fnumY = int(1 / spacing + 1)
h = 1 / fnumX 
h2 = 0.5 * h
fInvSpacing = fnumX

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
#cellColor = ti.Vector.field(3, dtype=ti.f32, shape=(fnumX, fnumY))

numParticles = numX * numY
particlePos = ti.Vector.field(2, dtype=ti.f32, shape=numParticles)
#particleColor = ti.Vector.field(3, dtype=ti.f32, shape=numParticles)
# init color 

particleVel = ti.Vector.field(2, dtype=ti.f32, shape=numParticles)

particleDensity = ti.field(dtype=ti.f32, shape=(fnumX, fnumY))
#particleRestDensity = 0

r =  0.3 * h
print("r", r)
particleRadius = r
pInvSpacing = 1.0 / (2.2 * particleRadius)

print("pInvSpacing", pInvSpacing)
# used for nearby search
pNumX = ti.floor(width * pInvSpacing) + 1
pNumY = ti.floor(height * pInvSpacing) + 1

pNumCells = pNumX * pNumY
#numCellParticles = ti.field(dtype=ti.i32, shape=(pNumX, pNumY))
numCellParticles = ti.field(dtype=ti.i32, shape=(pNumCells))
firstCellParticle = ti.field(dtype=ti.i32, shape=(pNumCells+1))

cellParticleIds =  ti.field(dtype=ti.i32, shape=numParticles)

print("pNumX pNumY", pNumX, pNumY)

minDist = 2 * particleRadius
minDist2 = minDist * minDist

#"""
# [TODO] there is a huge bug if-else in return will be eliminate
@ti.func
def clamp(x: float, min: float, max: float):
	ret = x
	if x < min:
		ret = min
	if x > max:
		ret = max
	return ti.cast(ret, int) 
#"""
"""
@ti.func
def clamp(x: float, min: float, max: float):
	#print("bef", x, min, max)
	ti.math.clamp(x, min, max)
	#print("after", x, min, max)
	return ti.cast(x, int)
"""

@ti.kernel
def init():
	dx = 2.0 * r
	dy = ti.sqrt(3.0) / 2.0 * dx
	for i, j in ti.ndrange(numX, numY):
		cnt = i* numY + j
		#
		particlePos[cnt][0] = h + r + dx * i + (0 if j % 2 == 0 else r)
		particlePos[cnt][1] = 10 * h + r + dy * j

	for i, j in ti.ndrange(fnumX, fnumY):
		s = 1.0	# fluid
		if (i == 0 or i == fnumX-1 or j == 0):
			s = 0.0;	# solid
		S[i, j] = s
	
@ti.kernel
def integrateParticles(dt: float, gravity: float):
	for i in ti.ndrange(numParticles):
		particleVel[i][1] += dt * gravity
		particlePos[i] += particleVel[i] * dt        

@ti.kernel
def countParticleInGrid():
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
		#xi = clamp(x, 0, pNumX-1)
		#yi = clamp(y, 0, pNumY-1)
		tm.clamp(ti.Vector([x, y]), 0, ti.Vector([pNumX-1, pNumX-1]))
		#print(xi, yi)
		_ind = ti.cast(x*pNumY+y, int)
		numCellParticles[_ind] += 1 # auto atomic add
		
		
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
		#xi = clamp(x, 0, pNumX-1)
		#yi = clamp(y, 0, pNumY-1)
		#_ind = xi*pNumY+yi
		tm.clamp(ti.Vector([x, y]), 0, ti.Vector([pNumX-1, pNumX-1]))
		_ind = ti.cast(x*pNumY+y, int)
		firstCellParticle[_ind] -= 1 #	auto atomic
		cellParticleIds[firstCellParticle[_ind]] = i

@ti.kernel
def pushParticlesApart():        		
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
    #h = 1 / fInvSpacing
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

        particlePos[i] = ti.Vector([x, y])

@ti.kernel
def updateParticleDensity(d: ti.template()):
	h1 = fInvSpacing

	#particleDensity.fill(0)
	d.fill(0)


	for i in ti.ndrange(numParticles):
		x, y = particlePos[i]

		#x = clamp(x, h, (fnumX-1)*h)
		#y = clamp(y, h, (fnumY-1)*h)
		tm.clamp(ti.Vector([x, y]), 0, ti.Vector([(fnumX-1)*h, (fnumY-1)*h]))		

		x0 = int(ti.floor((x-h2)*h1))
		tx = ((x-h2)-x0*h)*h1
		x1 = int(ti.min(x0+1, fnumX-2))

		y0 = int(ti.floor((y-h2)*h1))
		ty = ((y - h2)-y0*h)*h1
		y1 = int(ti.min(y0+1, fnumY-2))

		sx = 1 - tx
		sy = 1 - ty

		#if x0 < fnumX and y0 < fnumY: particleDensity[x0, y0] += sx * sy
		#if x1 < fnumX and y0 < fnumY: particleDensity[x1, y0] += tx * sy
		#if x1 < fnumX and y1 < fnumY: particleDensity[x1, y1] += tx * ty
		#if x0 < fnumX and y1 < fnumY: particleDensity[x0, y1] += sx * ty

		if x0 < fnumX and y0 < fnumY: d[x0, y0] += sx * sy
		if x1 < fnumX and y0 < fnumY: d[x1, y0] += tx * sy
		if x1 < fnumX and y1 < fnumY: d[x1, y1] += tx * ty
		if x0 < fnumX and y1 < fnumY: d[x0, y1] += sx * ty
    
	"""
	if particleRestDensity == 0:
		sum = 0.0
		numFluidCells = 0

		for i, j in ti.ndrange(fnumX, fnumY):
			if cellType[i, j] == FLUID_CELL:
				#sum += particleDensity[i, j]
				sum += d[i, j]
				numFluidCells += 1

		if numFluidCells > 0:
			particleRestDensity = sum / numFluidCells
	"""
	

@ti.func
def project(x, y, dx, dy):
	h1 = fInvSpacing
	x0 = ti.min(ti.floor((x - dx)*h1), fnumX-2)
	#print(x, dx)
	tx = (x - dx  - x0*h) * h1
	x1 = ti.min(x0+1, fnumX-2)

	y0 = ti.min(ti.floor((y-dy)*h1), fnumY-2)
	ty = (y - dy - y0*h) * h1
	y1 = ti.min(y0+1, fnumY-2)

	sx = 1.0 - tx
	sy = 1.0 - ty

	# interpolation
	d0 = sx * sy
	d1 = tx * sy
	d2 = tx * ty
	d3 = sx * ty
	return ti.cast(x0, int), ti.cast(x1, int), ti.cast(y0, int), ti.cast(y1, int), d0, d1, d2, d3

#	https://www.youtube.com/watch?v=XmzBREkK8kY

@ti.kernel
def p2g():
	# TODO overhere
	h1 = fInvSpacing
    
	for i, j in ti.ndrange(fnumX, fnumY):
		preU[i, j] = U[i, j]
		preV[i, j] = V[i, j]
		#preUV[i, j] = UV[i, j]

	#dUV.fill(0)
	dU.fill(0.0)
	dV.fill(0.0)

	#UV.fill(0)
	U.fill(0.0)
	V.fill(0.0)

	# intial all to air 
	for i, j in ti.ndrange(fnumX, fnumY):
		cellType[i, j] = SOLID_CELL if S[i, j] == 0.0 else AIR_CELL

	# to fluid
	for i in ti.ndrange(numParticles):
		# which grid
		x, y = ti.floor(particlePos[i]* h1)
		#x = clamp(x, 0, fnumX-1)
		#y = clamp(y, 0, fnumY-1)
		tm.clamp(ti.Vector([x, y]), 0, ti.Vector([(fnumX-1), (fnumY-1)]))	
		if cellType[int(x), int(y)]  == AIR_CELL:	# seems no need
			cellType[int(x), int(y)] = FLUID_CELL

	# project
	for i in ti.ndrange(numParticles):
		x, y = particlePos[i]
		#x = clamp(x, h, (fnumX-1)*h)
		#y = clamp(y, h, (fnumY-1)*h)

		tm.clamp(ti.Vector([x, y]), h, ti.Vector([(fnumX-1)*h, (fnumY-1)*h]))			

		#if x> 0.0 or y > 0.0: print(x, y)

		#if x != 0 or y != 0 :print(x, y)

		Ux0, Ux1, Uy0, Uy1, Ud0, Ud1, Ud2, Ud3 = project(x, y, 0, h2)
		Vx0, Vx1, Vy0, Vy1, Vd0, Vd1, Vd2, Vd3 = project(x, y, h2, 0)
		#print(Vx0, Vx1, Vy0, Vy1)

		#"""
		pv0 = particleVel[i][0]
		U[Ux0, Uy0] += pv0 * Ud0
		U[Ux1, Uy0] += pv0 * Ud1
		U[Ux1, Uy1] += pv0 * Ud2
		U[Ux0, Uy1] += pv0 * Ud3

		dU[Ux0, Uy0] += Ud0
		dU[Ux1, Uy0] += Ud1
		dU[Ux1, Uy1] += Ud2
		dU[Ux0, Uy1] += Ud3
		
		#"""
		pv1 = particleVel[i][1]
		V[Vx0, Vy0] += pv1 * Vd0
		V[Vx1, Vy0] += pv1 * Vd1
		V[Vx1, Vy1] += pv1 * Vd2
		V[Vx0, Vy1] += pv1 * Vd3

		dV[Vx0, Vy0] += Vd0
		dV[Vx1, Vy0] += Vd1
		dV[Vx1, Vy1] += Vd2
		dV[Vx0, Vy1] += Vd3
		#if particleVel[i][1] < 0.0:print(Vd0, Vd1, Vd2, Vd3)
		#if particleVel[i][1] < 0.0: 
		#	print(V[Vx0, Vy0], V[Vx1, Vy0], V[Vx1, Vy1], V[Vx0, Vy1])
		#	print(dV[Vx0, Vy0], dV[Vx1, Vy0], dV[Vx1, Vy1], dV[Vx0, Vy1])
		#print(Vd0, Vd1, Vd2, Vd3)
		#if particleVel[i][1] < 0.0: print(particleVel[i])
		#if V[Vx0, Vy0] != 0.0: print(Vx0, Vy0, V[Vx0, Vy0])
		#if V[Vx1, Vy0] != 0.0: print(Vx1, Vy0, V[Vx1, Vy0])
		#if V[Vx1, Vy1] != 0.0: print(Vx1, Vy1, V[Vx1, Vy1])
		#if V[Vx0, Vy1] != 0.0: print(Vx0, Vy1, V[Vx0, Vy1])
		#"""
		

	for i, j in ti.ndrange(fnumX, fnumY):
		#if V[i, j] != 0.0 : print(i, j, U[i, j], V[i, j])
		if dU[i, j] > 0.0:
			U[i, j] = U[i, j] / dU[i, j]
		if dV[i, j] > 0.0:
			V[i, j] = V[i, j] / dV[i, j]		
		#if V[i, j] != 0.0 : print(i, j, U[i, j], V[i, j])
			
	# no solid
	for i,j in ti.ndrange(fnumX, fnumY):
		solid = cellType[i, j] == SOLID_CELL
		if (solid or (i > 0 and cellType[i-1, j] == SOLID_CELL)):
			U[i, j] = preU[i, j]
		if (solid or (j > 0 and cellType[i, j-1] == SOLID_CELL)):
			V[i, j] = preV[i, j]
	

@ti.kernel
def g2p():
	for i in ti.ndrange(numParticles):
		x, y = particlePos[i]
		#x = clamp(x, h, (fnumX-1)*h)
		#y = clamp(y, h, (fnumY-1)*h)
		
		tm.clamp(ti.Vector([x, y]), h, ti.Vector([(fnumX-1)*h, (fnumY-1)*h]))	

		Ux0, Ux1, Uy0, Uy1, Ud0, Ud1, Ud2, Ud3 = project(x, y, 0, h2)
		Vx0, Vx1, Vy0, Vy1, Vd0, Vd1, Vd2, Vd3 = project(x, y, h2, 0)
		
		Uvalid0 = 1.0 if (cellType[Ux0, Uy0] != AIR_CELL or cellType[Ux0-1 , Uy0] != AIR_CELL) else 0.0
		Uvalid1 = 1.0 if (cellType[Ux1, Uy0] != AIR_CELL or cellType[Ux1-1 , Uy0] != AIR_CELL) else 0.0
		Uvalid2 = 1.0 if (cellType[Ux1, Uy1] != AIR_CELL or cellType[Ux1-1 , Uy1] != AIR_CELL) else 0.0
		Uvalid3 = 1.0 if (cellType[Ux0, Uy1] != AIR_CELL or cellType[Ux0-1 , Uy1] != AIR_CELL) else 0.0

		Ud = Uvalid0 * Ud0 + Uvalid1 * Ud1 + Uvalid2 * Ud2 + Uvalid3 * Ud3

		if (Ud > 0.0):
			particleVel[i][0] = (Uvalid0 * Ud0 * U[Ux0, Uy0] 
			+ Uvalid1 * Ud1 * U[Ux1, Uy0] 
			+ Uvalid2 * Ud2 * U[Ux1, Uy1] 
			+ Uvalid3 * Ud3 * U[Ux0, Uy1]) / Ud
			

		Vvalid0 = 1.0 if (cellType[Vx0, Vy0] != AIR_CELL or cellType[Vx0 , Vy0-1] != AIR_CELL) else 0.0
		Vvalid1 = 1.0 if (cellType[Vx1, Vy0] != AIR_CELL or cellType[Vx1 , Vy0-1] != AIR_CELL) else 0.0
		Vvalid2 = 1.0 if (cellType[Vx1, Vy1] != AIR_CELL or cellType[Vx1 , Vy1-1] != AIR_CELL) else 0.0
		Vvalid3 = 1.0 if (cellType[Vx0, Vy1] != AIR_CELL or cellType[Vx0 , Vy1-1] != AIR_CELL) else 0.0

		Vd = Vvalid0 * Vd0 + Vvalid1 * Vd1 + Vvalid2 * Vd2 + Vvalid3 * Vd3		

		if (Vd > 0.0):
			particleVel[i][1] = (Vvalid0 * Vd0 * V[Vx0, Vy0] 
			+ Vvalid1 * Vd1 * V[Vx1, Vy0] 
			+ Vvalid2 * Vd2 * V[Vx1, Vy1] 
			+ Vvalid3 * Vd3 * V[Vx0, Vy1]) / Vd
			#if particleVel[i][1] < 0.0: print("after", V[Vx0, Vy0], V[Vx1, Vy0], V[Vx1, Vy1], V[Vx0, Vy1])
		

@ti.kernel
def solveIncompressibility(cp: float):	
	for i, j in ti.ndrange((1, fnumX-1), (1, fnumY-1)):
		center = [i, j]			
		if cellType[center] != FLUID_CELL:
			continue
		left = [i-1, j]
		right = [i+1, j]
		bottom = [i, j-1]
		top = [i, j+1]

		sx0 = S[left]
		sx1 = S[right]
		sy0 = S[bottom]
		sy1 = S[top]
		s = sx0 + sx1 +sy0 + sy1
		if s == 0.0:
			continue
		div = U[right] - U[center] + V[top] - V[center]
		#print(s)
		#if (particleRestDensity > 0.0)
		p = -div/s
		#p = p * 1.9

		P[center] += cp * p
		U[center] -= sx0 * p
		U[right] += sx1 * p
		V[center] -= sy0 * p
		V[top] += sy1 * p
		#print(U[center], U[right], V[center], V[top])



def update():
	numSubSteps = 10
	sdt = dt / numSubSteps
	
	for step in range(numSubSteps):
		integrateParticles(sdt, gravity)
		countParticleInGrid()
		# [ok] 性能问题
		for iter in range(numParticleIters):
			pushParticlesApart()
		handleParticleCollisions()
		
		p2g()
		
		#updateParticleDensity(particleDensity)
		P.fill(0.0)
		cp  = density * h / sdt 
		#print(cp)
		# [TODO] time step and grid size bug
		for iter in range(numPressureIters):
			solveIncompressibility(cp)
		
		g2p()

gui = ti.GUI('PIC-2D', (c_width, c_height))

init()


while gui.running:
	update()

	pos = particlePos.to_numpy()
	#print(pos * c_width)
	gui.circles(pos , radius=3)

	#gui.set_image(pixels)
	gui.show()