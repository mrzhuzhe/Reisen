from numpy import float64
import taichi as ti
import taichi.math as tm
from torch import double

ti.init(arch=ti.gpu, default_fp=ti.f64) 

vec3f = ti.types.vector(3, ti.f64)
gravity = vec3f(0, -9.8, 0)
dt = 1.0 / 60.0
# numSteps > 100 will broken
numSteps = 1000
sdt = dt / numSteps
wireRadius = 50
n = 6

pos = ti.Vector.field(3, dtype=ti.f64, shape=n)
pos32 = ti.Vector.field(3, dtype=ti.f32, shape=n)
vel = ti.Vector.field(3, dtype=ti.f64, shape=n)
centers  = ti.Vector.field(3, dtype=ti.f64, shape=n)
centers32  = ti.Vector.field(3, dtype=ti.f32, shape=n)

# init position 
centers[0] = vec3f(0 , 0 , 0)
centers[1] = vec3f(50 , 0 , 0)
pos[0] = vec3f(50, 0, 0)
pos[1] = vec3f(100, 0, 0)


centers[2] = vec3f(0 , 0 , 50)
centers[3] = vec3f(50 , 0 , 50)
pos[2] = vec3f(50, 0, 50)
pos[3] = vec3f(100, 0, 50)


centers[4] = vec3f(-50 , 0 , 0)
centers[5] = vec3f(-50 , 0 , 50)
pos[4] = vec3f(0, 0, 0)
pos[5] = vec3f(0, 0, 50)


@ti.kernel
def update():
    for index in range(n):
        _pos = pos[index]
        _center = centers[index]
        _vel = vel[index]
        centers32[index] = centers[index]
        for step in range(numSteps):

            _vel +=  sdt * gravity
            
            # previous position 
            prepos = _pos
            _pos +=  sdt * _vel
            # constraint 
            _dir = _pos - _center
            _norm = _dir.norm()
            
            _normalized = _dir / _norm

            _lambda = wireRadius - _norm
            _pos += _normalized * _lambda
                            
            # project back 
            #print(_pos - prepos)
            _vel = (_pos - prepos) / sdt

        pos[index] = _pos
        vel[index] = _vel
        pos32[index] = _pos

win_x = 640
win_y = 640

window = ti.ui.Window("simple pendulum", (win_x, win_y))
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(500, 500, 500)
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
    scene.particles(pos32, color = (0, 1, 1), radius = 10)
    scene.particles(centers32, color = (1, 1, 0), radius = 10)

    canvas.scene(scene)
    window.show()
    step +=1 
