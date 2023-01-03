from matplotlib.pyplot import flag
import taichi as ti
import math

ti.init(arch=ti.gpu)

c_w = 400
c_h = 400
sim_w = 1
sim_h = 1
scale = c_w / sim_w

#gravity = -9.81
gravity = 0
dt = 1.0 / 120.0
numSteps = 100
radius = 0.15
res = 200
density = 1000
h = 1 / res
h2 = 0.5 * h

numX = 200
numY = 200

U = ti.field(dtype=ti.f32, shape=(numX, numY))
V = ti.field(dtype=ti.f32, shape=(numX, numY))
newU = ti.field(dtype=ti.f32, shape=(numX, numY))
newV = ti.field(dtype=ti.f32, shape=(numX, numY))
P = ti.field(dtype=ti.f32, shape=(numX, numY))
S = ti.field(dtype=ti.f32, shape=(numX, numY))
M = ti.field(dtype=ti.f32, shape=(numX, numY))
M.fill(1)
newM = ti.field(dtype=ti.f32, shape=(numX, numY)) 

pixels = ti.field(dtype=ti.f32, shape=(c_w, c_h))
pixels.fill(1)

U_FIELD = 0
V_FIELD = 1
S_FIELD = 2

@ti.func
def cX(x):
	return x * scale
	
@ti.func
def cY(y):
	return c_h - y * scale
	
@ti.func 
def avgU(i, j):
	return (U[i, j-1] + U[i, j] + U[i+1, j-1] + U[i+1, j]) * 0.25

@ti.func
def avgV(i, j):
	return (V[i-1, j] + U[i, j] + U[i-1, j+1] + U[i,j+1]) * 0.25

@ti.func 
def sampleField(x, y , field: int):
	h1 = 1 / h
	x = ti.max(ti.min(x, numX * h), h)
	y = ti.max(ti.min(y, numY * h), h)
	dx = 0
	dy = 0
	
	if field == U_FIELD:
		#f = U
		dy = h2
	elif field == V_FIELD:
		#f = V
		dx = h2
	elif field == S_FIELD:
		#f = M
		dx = h2 
		dy = h2 

	x0 = int(ti.min(ti.floor((x-dx)*h1), numX-1))
	tx = ((x - dx) - x0*h) * h1
	x1 = int(ti.min(x0 +1, numX-1))

	y0 = int(ti.min(ti.floor((y-dy)*h1), numY-1))
	ty = ((y - dy)- y0*h)*h1
	y1 = int(ti.min(y0+1, numY-1))

	sx = 1 - tx 
	sy = 1 - ty
	
	#print(x0, x1, y0, y1)
	
	_coeff = ti.Vector([sx*sy, tx*sy, tx*ty, sx*ty])
	#print(sx*sy, tx*sy, tx*ty, sx*ty)
	_vars = ti.Vector([0, 0 , 0, 0])
	if field == U_FIELD:
		_vars = ti.Vector([U[x0, y0], U[x1, y0], U[x1, y1], U[x0, y1]])		
	elif field == V_FIELD:
		_vars = ti.Vector([V[x0, y0], V[x1, y0], V[x1, y1], V[x0, y1]])		
	elif field == S_FIELD:		
		_vars = ti.Vector([M[x0, y0], M[x1, y0], M[x1, y1], M[x0, y1]])	
		
	return _coeff.dot(_vars)

@ti.func 
def sampleFieldM(x, y , field: int):
	h1 = 1 / h
	x = ti.max(ti.min(x, numX * h), h)
	y = ti.max(ti.min(y, numY * h), h)
	dx = 0
	dy = 0
	if field == U_FIELD:
		#f = U
		dy = h2
	elif field == V_FIELD:
		#f = V
		dx = h2
	elif field == S_FIELD:
		#f = M
		dx = h2 
		dy = h2 

	x0 = int(ti.min(ti.floor((x-dx)*h1), numX-1))
	tx = ((x - dx) - x0*h) * h1
	x1 = int(ti.min(x0 +1, numX-1))

	y0 = int(ti.min(ti.floor((y-dy)*h1), numY-1))
	ty = ((y - dy)- y0*h)*h1
	y1 = int(ti.min(y0+1, numY-1))

	sx = 1 - tx 
	sy = 1 - ty


	#print(S_FIELD == field)

	"""
	_vars = 0
	if field == U_FIELD:
		#print("is me U_FIELD ????")
		_vars = sx*sy * U[x0, y0] + tx*sy * U[x1, y0] + tx*ty * U[x1, y1] + sx*ty * U[x0, y1]	
	elif field == V_FIELD:
		#print("is me V_FIELD ????")
		_vars = sx*sy * V[x0, y0] + tx*sy * V[x1, y0] + tx*ty * V[x1, y1] + sx*ty * V[x0, y1]
	elif field == S_FIELD:		
		#print("is me???")
		_vars = sx*sy * M[x0, y0] + tx*sy * M[x1, y0] + tx*ty * M[x1, y1] + sx*ty * M[x0, y1]
		
	return _vars 
	"""
	return sx*sy * M[x0, y0] + tx*sy * M[x1, y0] + tx*ty * M[x1, y1] + sx*ty * M[x0, y1]
	


# external force	
@ti.kernel
def integrate(dt: float, gravity: float):    
	for i, j in ti.ndrange(numX, numY-1):
		if S[i, j] != 0 and S[i, j-1] !=0:
			V[i, j] += gravity * dt


# project
@ti.kernel
def solveIncompressibility(substep: int, dt: float):
	cp = density * h / dt
	for i, j in ti.ndrange((1, numX-1), (1, numY-1)):
		if S[i, j] == 0:
			continue			
		Sx0 = S[i-1, j]
		Sx1 = S[i+1, j]
		Sy0 = S[i, j-1]
		Sy1 = S[i, j+1]
		_s = Sx0 + Sx1 + Sy0 + Sy1
		if _s == 0:
			continue
		
		_div = U[i+1, j] - U[i, j] + V[i, j+1] - V[i, j] 
		_p = - _div / _s
		# todo sor here
		P[i, j] += cp * _p
		U[i, j] -= Sx0 * _p
		U[i+1, j] += Sx1 * _p
		V[i, j] -= Sy0 * _p
		V[i, j+1] += Sy1 * _p

@ti.kernel
def extrapolate():
	for i in range(numX):
		U[i, 0] = U[i, 1]
		U[i, numY-1] = U[i, numY-2] 

	for j in range(numY):
		V[0, j] = U[1, j]
		V[numX-1, j] = U[numX-2, j]

@ti.kernel
def advectVel(dt: float):
	for i, j in ti.ndrange(numX, numY):
		newU[i, j] = U[i, j]
		newV[i, j] = V[i, j]

	for i, j in ti.ndrange((1, numX), (1, numY)):
		if S[i, j] != 0 and S[i-1, j] != 0 and j < numY -1 :
			x = i * h
			y = j * h + h2
			_u = U[i, j]
			_v = avgV(i, j)
			x = x - dt * _u
			y = y - dt * _v
			_u = sampleField(x, y, U_FIELD)
			newU[i, j] = _u
		if S[i, j] != 0 and S[i, j-1] !=0 and i < numX -1:
			x = i * h + h2
			y = j * h
			_u = avgU(i, j)
			_v = V[i, j]
			x = x - dt * _u
			y = y - dt * _v 
			_v = sampleField(x, y, V_FIELD)
			newV[i, j] = _v 

	for i, j in ti.ndrange(numX, numY):
		U[i, j] = newU[i, j]
		V[i, j] = newV[i, j]

@ti.kernel
def advectSmoke(dt: float):
	for i, j in ti.ndrange(numX, numY):
		newM[i, j] = M[i, j]

	for i, j in ti.ndrange((1, numX-1), (1, numY-1)):
		if S[i, j] != 0:
			_u = (U[i, j] + U[i+1, j]) * 0.5
			_v = (V[i, j] + V[i, j+1]) * 0.5
			x = i * h + h2 - dt * _u 
			y = j * h + h2 - dt * _v 
			#newM[i, j] = sampleField(x, y, S_FIELD)
			newM[i, j] = sampleFieldM(x, y, S_FIELD)

	for i, j in ti.ndrange(numX, numY):
		M[i, j] = newM[i, j]


@ti.kernel
def draw():	
	for i, j in ti.ndrange(numX, numY):
		_s = M[i, j]			
		#x = ti.floor(cX(i * h))
		x = ti.floor(cX(i * h))
		#y = ti.floor(cY((j+1) * h))
		y = ti.floor(cY((j+1) * h))
		#cx = ti.floor(scale * h) + 1
		#cy = ti.floor(scale * h) + 1
		#p = 4 * (y * c_w + x)
		pixels[int(x), int(y)] = 255*_s
			

@ti.kernel
def init():
	inVel = 2.0
	for i, j in ti.ndrange(numX, numY):
		_s = 1.0	
		if (i == 0 | j == 0 | j == numY-1):
			_s = 0.0
		S[i, j] = _s

		if (i == 1):
			U[i, j] = inVel
            

	pipeH = 0.1 * numY
	minJ = ti.floor(0.5 * numY - 0.5*pipeH)
	maxJ = ti.floor(0.5 * numY + 0.5*pipeH)

	for j in range(int(minJ), int(maxJ)):
		M[0, j] = 0.0
	

    #setObstacle(0.4, 0.5, true)

    #pass

def update():
	integrate(dt, gravity)
	P.fill(0)
	for substep in range(numSteps):
		solveIncompressibility(substep, dt)

	extrapolate()
	advectVel(dt)
	advectSmoke(dt)

gui = ti.GUI('langrange-2D', (c_w, c_h))

init()

while gui.running:
	update()
	draw()
	gui.set_image(pixels)
	gui.show()