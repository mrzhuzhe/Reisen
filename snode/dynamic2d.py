import taichi as ti


_fp  = ti.f32
ti.init(arch=ti.gpu, default_fp=_fp)

n = 2
m = 50
x = ti.field(ti.i32)
block = ti.root.dense(ti.ij, (n, n))
pixel = block.dynamic(ti.k, m)
pixel.place(x)

l = ti.field(ti.i32)
ti.root.dense(ti.ij, (n, n)).place(l)

@ti.kernel
def make_lists():
    for i in range(n):
        for j in range(n):
            _count = 0
            for k in range(int(6.0*ti.random())):
                ti.append(x.parent(), (i, j), int(6.0*ti.random()))
                _count += 1
            print("------", i, j, _count)
            l[i, j] = ti.length(x.parent(), (i, j))  
            #print(i, j, ti.length(x.parent(), (i, j)))

@ti.kernel
def read_lists(d: int):
    for i in range(n * d):
        for j in range(n * d):
            print("***", i, j, l[i, j])
            for k in range(l[i, j]):
                print("x[i,j,k]", i, j, k, ":", x[i, j, k])

#ti.deactivate_all_snodes()  
make_lists()
read_lists(d=1)
ti.deactivate_all_snodes()
print("read 2")
read_lists(d=2)
print("read 3")
make_lists()
read_lists(d=2)