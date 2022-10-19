import math
import taichi as ti

ti.init(arch=ti.gpu)

screen_res = (640, 640)
screen_to_world_ratio = 10.0
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio)

dim = 2
num_particles_x = 100
num_particles = num_particles_x * 20
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 3
particle_radius_in_world = particle_radius / screen_to_world_ratio


# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

old_positions = ti.Vector.field(dim, float, num_particles)
positions = ti.Vector.field(dim, float, num_particles)
velocities = ti.Vector.field(dim, float, num_particles)

lambdas = ti.field(float, num_particles)
position_deltas = ti.Vector.field(dim, float, num_particles)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float, ())

bg_color = 0x112f41
particle_color = 0x068587

grid_size = 2
grid_n = math.ceil(boundary[0] / grid_size)
"""
grid_n = 16
grid_size = 0.5/ grid_n  # Simulation domain of size [0, 1]
"""
print(f"Grid size: {grid_n}x{grid_n}")

assert particle_radius_in_world * 2 < grid_size

list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=num_particles, name="particle_id")

@ti.func
def findNeighbors():
    grain_count.fill(0)
    for i in range(num_particles):
        grid_idx = ti.floor(positions[i]/boundary[0] * grid_n, int)
        grain_count[grid_idx] += 1
    
    column_sum.fill(0)
    # kernel comunicate with global variable ???? this is a bit amazing 
    for i, j in ti.ndrange(grid_n, grid_n):        
        ti.atomic_add(column_sum[i], grain_count[i, j])

    # this is because memory mapping can be out of order
    _prefix_sum_cur = 0
    
    for i in ti.ndrange(grid_n):
        prefix_sum[i] = ti.atomic_add(_prefix_sum_cur, column_sum[i])
    
    for i, j in ti.ndrange(grid_n, grid_n):        
        # we cannot visit prefix_sum[i,j] in this loop
        pre = ti.atomic_add(prefix_sum[i], grain_count[i, j])        
        linear_idx = i * grid_n  + j
        list_head[linear_idx] = pre
        list_cur[linear_idx] = list_head[linear_idx]
        # only pre pointer is useable
        list_tail[linear_idx] = pre + grain_count[i, j]       

    for i in range(num_particles):
        grid_idx = ti.floor(positions[i]/boundary[0]  * grid_n, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i
        

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result

@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_,
                                                     h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]
                      ]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p



@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.8,])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)
    
    findNeighbors()

@ti.kernel
def substep():
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in range(num_particles):
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        grid_idx = ti.floor(pos_i/boundary[0] * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i , neigh_j in ti.ndrange((x_begin, x_end), (y_begin, y_end)):
            neigh_linear_idx = neigh_i * grid_n + neigh_j
            for p_idx in range(list_head[neigh_linear_idx],
                            list_tail[neigh_linear_idx]):                    
                j = particle_id[p_idx]	 
                pos_j = positions[j]                
                pos_ji = pos_i - pos_j
                grad_j = spiky_gradient(pos_ji, h_)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                # Eq(2)
                density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in range(num_particles):
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        grid_idx = ti.floor(pos_i/boundary[0] * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)
        pos_delta_i = ti.Vector([0.0, 0.0])
        for neigh_i , neigh_j in ti.ndrange((x_begin, x_end), (y_begin, y_end)):
            neigh_linear_idx = neigh_i * grid_n + neigh_j
            for p_idx in range(list_head[neigh_linear_idx],
                            list_tail[neigh_linear_idx]):                    
                j = particle_id[p_idx]
                pos_j = positions[j]
                lambda_j = lambdas[j]
                pos_ji = pos_i - pos_j
                scorr_ij = compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                    spiky_gradient(pos_ji, h_)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i

        positions[p_i] += position_deltas[p_i]
        
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]

@ti.kernel
def epilogue():
    
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    # no vorticity/xsph because we cannot do cross product in 2D...


def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()


def render(gui):
    gui.clear(bg_color)
    pos_np = positions.to_numpy()
    for j in range(dim):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    
    gui.show()


@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5,
                          boundary[1] * 0.02])
        positions[i] = ti.Vector([i % num_particles_x, i // num_particles_x
                                  ]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])



def main():
    init_particles()
    gui = ti.GUI('PBF2D', screen_res)
    while gui.running and not gui.get_event(gui.ESCAPE):
        
        run_pbf()
        
        render(gui)


if __name__ == '__main__':
    main()
