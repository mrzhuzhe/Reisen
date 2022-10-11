from numpy import float64
import taichi as ti
import taichi.math as tm
from torch import double

ti.init(arch=ti.gpu)


vec3f = ti.types.vector(3, ti.f64)
gravity = vec3f(0, -0.98, 0)
dt = 1.0 / 60.0
numSteps = 10
sdt = dt / numSteps
wireRadius = 10
n = 6

pos = ti.Vector.field(3, dtype=float, shape=n)
prepos = ti.Vector.field(3, dtype=float, shape=n)
vel = ti.Vector.field(3, dtype=float, shape=n)
centers  = ti.Vector.field(3, dtype=float, shape=n)

# init position 
centers[0] = vec3f(0 , 0 , 0)
centers[1] = vec3f(15 , 0 , 0)
pos[0] = vec3f(10, 0, 0)
pos[1] = vec3f(25, 0, 0)


centers[2] = vec3f(0 , 0 , 5)
centers[3] = vec3f(15 , 0 , 5)
pos[2] = vec3f(10, 0, 5)
pos[3] = vec3f(25, 0, 5)


centers[4] = vec3f(-15 , 0 , 0)
centers[5] = vec3f(-15 , 0 , 5)
pos[4] = vec3f(-5, 0, 0)
pos[5] = vec3f(-5, 0, 5)


print(sdt)
@ti.kernel
def update():
    for index in range(n):
        _pos = pos[index]
        _center = centers[index]
        _vel = vel[index]
        for step in range(numSteps):

            _vel +=  sdt * gravity
            
            # previous position 
            prepos[index] = _pos
            _pos +=  sdt * _vel
            # constraint 
            _dir = _pos - _center
            _norm = _dir.norm()
            
            _normalized = _dir / _norm

            _lambda = wireRadius - _norm
            _pos += _normalized * _lambda
                            
            # project back 
            _vel = (_pos - prepos[index]) / sdt

        pos[index] = _pos
        vel[index] = _vel

win_x = 640
win_y = 640

window = ti.ui.Window("simple pendulum", (win_x, win_y))
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(50, 50, 50)
camera.lookat(0, 0, 0)



#
step = 0
while window.running:
    ti.deactivate_all_snodes()  
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
 
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    
    #if step < 10:
    #    print(step)
    update()
    
    scene.particles(pos, color = (0, 1, 1), radius = 1)
    scene.particles(centers, color = (1, 1, 0), radius = 1)

    canvas.scene(scene)
    window.show()
    step +=1 
