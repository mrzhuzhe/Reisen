import taichi as ti
import numpy as np
from config import _config

@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = np.array([0.0, -9.81, 0.0])  # Gravity

        self.viscosity = 0.01  # viscosity

        self.density_0 = _config["Configuration"]["density0"]

        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4

      
    def initialize(self):
        self.ps.initialize_particle_system()
        #for r_obj_id in self.ps.object_id_rigid_body:
        #    self.compute_rigid_rest_cm(r_obj_id)
        #self.compute_static_boundary_volume()
        #self.compute_moving_boundary_volume()

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_static_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            self.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            self.ps.m_V[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta: ti.template()):
        delta += self.cubic_kernel((self.ps.x[p_i] - self.ps.x[p_j]).norm())

    def compute_moving_boundary_volume(self):
        pass
    def substep(self):
        pass
    def solve_rigid_body(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v[p_i] -= (
            1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_3D(self, particle_type:int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    self.ps.x[p_i][2] = self.ps.domain_size[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    self.ps.x[p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)
    def step(self):
        self.ps.initialize_particle_system()

        #self.compute_moving_boundary_volume()
        self.substep()
        #self.solve_rigid_body()       
        self.enforce_boundary_3D(self.ps.material_fluid)
    
    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res