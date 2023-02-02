from matplotlib.pyplot import grid
import taichi as ti
import taichi.math as tm

f32 = ti.f32
i32 = ti.i32
ti.init(arch=ti.gpu) 

n = 100
numSteps = 1
particleRadius = 0.05
dt = 0.01
g = ti.Vector((0, 0, -9.81), ti.f32)


pos = ti.Vector.field(3, ti.f32, shape=(n))

@ti.kernel
def init():
    pass


@ti.kernel
def update():
    pass



win_x = 640
win_y = 640

window = ti.ui.Window("flip 3d", 
(win_x, win_y), vsync=True
)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(2.5, 1, 2)
camera.lookat(0, 0, 0)
scene.ambient_light((0.5, 0.5, 0.5))
scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))


init()
while window.running:
    ti.deactivate_all_snodes()  
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
 
    scene.set_camera(camera)
    
    for s in range(numSteps):
        update()    
    
    scene.particles(pos, color = (0, 1, 1), radius = particleRadius)

    canvas.scene(scene)
    window.show()
