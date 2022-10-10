import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)


vec3f = ti.types.vector(3, ti.f32)
gravity = vec3f(0, -0.98, 0)
dt = 1.0 / 60.0
numSteps = 10
sdt = dt / numSteps
wireRadius = 100
n = 2

pendulum = ti.Vector.field(3, dtype=float, shape=n)
prepos = ti.Vector.field(3, dtype=float, shape=n)
vel = ti.Vector.field(3, dtype=float, shape=n)
centers  = ti.Vector.field(3, dtype=float, shape=n)

# init position 

#pendulum[0] += tm.vec3(1.0, -1.5, 0)
#centers[0] += tm.vec3(-0.5 , 0 , 0)

centers[0] = vec3f(-5 , 0 , 0)
centers[1] = vec3f(-6 , 0 , 0)
pendulum[0] = vec3f(10, -15, 0)
pendulum[1] = vec3f(10, -16, 0)


@ti.kernel
def update():
    for index in range(n):
        _pos = pendulum[index]
        _centers = centers[index]
        _vel = vel[index]
        for step in range (numSteps):
            
            _vel +=  sdt * gravity
            
            # previous position 
            prepos[index] = pendulum[index]        
            _pos +=  sdt * _vel
            #"""
            # constraint 
            _dir = _pos - _centers
            _norm = _dir.norm()
            _normalized = _dir / _norm
            _lambda = wireRadius - _norm
            
            _pos += _normalized * _lambda
                        
            # project back 
            _vel = (_pos - prepos[index]) / sdt
            #"""
            
                        
        pendulum[index] = _pos
        vel[index] = _vel

win_x = 640
win_y = 640

window = ti.ui.Window("simple pendulum", (win_x, win_y), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(150, 150, 150)
camera.lookat(0, 0, 0)



#update()
#"""
while window.running:
    ti.deactivate_all_snodes()  
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
 
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
    
    update()
    scene.particles(pendulum, color = (0, 1, 1), radius = 5)

    canvas.scene(scene)
    window.show()
#"""
