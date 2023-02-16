#   https://docs.taichi-lang.org/docs/differentiable_programming

import taichi as ti 
ti.init(arch=ti.cpu)

x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_y():    
    y[None] = ti.sin(x[None])
    z[None] = ti.cos(x[None])   # dispear ? because only tape y in next line
with ti.ad.Tape(y):
    compute_y()

print('dy/dx =', x.grad[None], ' at x =', x[None])