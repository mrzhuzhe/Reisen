import taichi as ti


_fp  = ti.f32
ti.init(arch=ti.gpu, default_fp=_fp)

n = 2
x = ti.field(ti.f32)
block = ti.root.dense(ti.ij, (n, n))
pixel = block.dynamic(ti.k, n)
pixel.place(x)
l = ti.field(ti.i32)
ti.root.dense(ti.ij, (n, n)).place(l)

@ti.kernel
def make_lists():
    for i in range(n):
        for j in range(n):
            _count = 0
            for k in range(int(6.0*ti.random())):
                ti.append(x.parent(), (i, j), i + j)
                _count += 1
            print("------", i, j, _count)
            l[i, j] = ti.length(x.parent(), (i, j))  
            print(i, j, ti.length(x.parent(), (i, j)))

@ti.kernel
def read_lists():
    for i in range(n):
        for j in range(n):
            print("***", i, j, l[i, j])


make_lists()
read_lists()