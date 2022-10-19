import math
import taichi as ti
import taichi.math as tm


_fp  = ti.f32
ti.init(arch=ti.gpu, default_fp=_fp)
n = 10
grads_j = ti.Vector.field(2, dtype=_fp, shape=n)

# because write is async
@ti.func
def run1():
    for i in range(n):
        for j in range(n):
            if j > 0:
                grads_j[j] = ti.Vector([i, j]) 

        for j in range(n):
            print("run1", grads_j[j])


@ti.func
def run2():
    for i in range(n):
        for j in range(n):
            print("run2", grads_j[j])


@ti.kernel
def main():
    run1()
    run2()

main()