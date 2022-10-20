import taichi as ti


_fp  = ti.f32
ti.init(arch=ti.gpu, default_fp=_fp)

x = ti.field(ti.i32)
block = ti.root.dense(ti.i, 5)
pixel = block.dynamic(ti.j, 5)
pixel.place(x)
l = ti.field(ti.i32)
ti.root.dense(ti.i, 5).place(l)

@ti.kernel
def make_lists():
    for i in range(5):
        for j in range(i):
            ti.append(x.parent(), i, j * j)  # ti.append(pixel, i, j * j)
        l[i] = ti.length(x.parent(), i)  # [0, 1, 2, 3, 4]

@ti.kernel
def read_lists():
    for i in range(5):
        for j in range(l[i]):
            print(i, x[i, j])
    print(int(6.0*ti.random()))
    print(int(6.0*ti.random()))
    

make_lists()
read_lists()