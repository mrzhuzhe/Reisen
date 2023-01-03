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

numX = 200
numY = 200

U = ti.field(dtype=ti.f32, shape=(numX, numY))
V = ti.field(dtype=ti.f32, shape=(numX, numY))
newU = ti.field(dtype=ti.f32, shape=(numX, numY))
newV = ti.field(dtype=ti.f32, shape=(numX, numY))
P = ti.field(dtype=ti.f32, shape=(numX, numY))
S = ti.field(dtype=ti.f32, shape=(numX, numY))
M = ti.field(dtype=ti.f32, shape=(numX, numY))
newM = ti.field(dtype=ti.f32, shape=(numX, numY)) 

pixels = ti.field(dtype=float, shape=(c_w, c_h))

@ti.func
def cX(x):
	return x * scale
	
@ti.func
def cY(y):
	return c_h - y * scale
	
# external force	
@ti.kernel
def integrate(dt, gravity):    
	for i, j in ti.ndrange(numX, numY-1):
		if S[i, j] != 0 and S[i, j-1] !=0:
			V[i, j] += gravity * dt


# project
@ti.kernel
def solveIncompressibility(substep, dt):
	cp = density * h / dt
	for i, j in ti.ndrange(numX-1, numY-1):
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

def extrapolate():
	for i in range(numX):
		U[i, 0] = U[i, 1]
		U[i, numY-1] = U[i, numY-2] 

	for j in range(numY):
		V[0, j] = U[1, j]
		V[numX-1, j] = U[numX-2, j]

def advectVel(dt):
    pass

def advectSmoke(dt):
    pass

@ti.kernel
def draw():	
	for i, j in ti.ndrange(numX, numY):
		_s = M[i, j]			
		#x = math.floor(cX(i * h))
		x = ti.floor(cX(i * h))
		#y = math.floor(cY((j+1) * h))
		y = ti.floor(cY((j+1) * h))
		#cx = math.floor(scale * h) + 1
		#cy = math.floor(scale * h) + 1
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
		M[0, j] = 1.0
	

    #setObstacle(0.4, 0.5, true)

    #pass

def update():
    integrate(dt, gravity)
	P.fill(0)
	for substep in range(numSteps):
		solveIncompressibility(substep, dt)

	extrapolate();
	advectVel(dt);
	advectSmoke(dt);

gui = ti.GUI('langrange-2D', (c_w, c_h))

init()

while gui.running:
    #for s in range(numSteps):
	update()
    #_pos = pos.to_numpy()
    #gui.circles(_pos/minX, radius=particleRadius_show)
	draw()
	gui.set_image(pixels)
	gui.show()