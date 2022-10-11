import taichi as ti
import taichi.math as tm
import numpy as np

ti.init(arch=ti.gpu, default_fp=ti.f64) 

vec3f = ti.types.vector(3, ti.f64)
gravity = vec3f(0, -0.98, 0)
dt = 0.01
numSteps = 100
sdt = dt / numSteps
n = 4
pi = 3.14

pos = ti.Vector.field(3, dtype=ti.f64, shape=n)
prePos = ti.Vector.field(3, dtype=ti.f64, shape=n)
pos32 = ti.Vector.field(3, dtype=ti.f32, shape=n)
vel = ti.Vector.field(3, dtype=ti.f64, shape=n)


lengths = ti.field(ti.f64, shape=n)
masses = ti.field(ti.f64, shape=n)
angles = ti.field(ti.f64, shape=n)

_lengths = [0, 0.4, 0.3, 0.2]
_masses =  [0, 1.0, 1.0, 1.0]
_angles =  [0, pi * 0.5, pi, pi]


# init position
#@ti.kernel
def init():
    x = 0
    y = 0
    for i in range(1, n):
        lengths[i] = _lengths[i]
        masses[i] = _masses[i]
        angles[i] = _angles[i]
   
        x += lengths[i] * ti.sin(angles[i])
        y += lengths[i] * -ti.cos(angles[i])
        z = 0
        pos[i] = vec3f(x, y, z)
        prePos[i] = vec3f(x, y, z)
        pos32[i] = pos[i]
    

@ti.kernel
def update():
    for index in range(1, n):
        _pos = pos[index]
        _vel = vel[index]
        for step in range(numSteps):                    
            _vel +=  sdt * gravity
            
            # previous position 
            prepos = _pos
            _pos +=  sdt * _vel
            
            #"""
            # constraint 
            delta = _pos - pos[index-1]
            norm = delta.norm()
            w0 = 1 / masses[index-1] if masses[index-1]  > 0.0 else 0.0 
            w1 = 1 / masses[index]
            corr = (lengths[index] - norm) / norm / (w0 + w1)
            pos[index-1] -= w0 * corr * delta
            pos32[index-1] = pos[index-1]
            _pos += w1 * corr * delta

            # project back
            _vel = (_pos - prepos) / sdt           

        pos[index] = _pos
        vel[index] = _vel
        pos32[index] = _pos
        
        

win_x = 640
win_y = 640

window = ti.ui.Window("simple pendulum", (win_x, win_y), vsync=True)
#window = ti.ui.Window("simple pendulum", (win_x, win_y))
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.Camera()
camera.position(1, 1, 1)
camera.lookat(0, 0, 0)



init()
step = 0
while window.running:
    ti.deactivate_all_snodes() 
    camera.track_user_inputs(window, movement_speed=0.01, hold_key=ti.ui.RMB)


    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    
    update()
    scene.particles(pos32, color = (0, 1, 1), radius = 0.01)
    
    canvas.scene(scene)
    window.show()
    step +=1 
