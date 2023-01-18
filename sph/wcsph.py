import taichi as ti
from base import SPHBase


class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)

        # Pressure state function parameters(WCSPH)
        self.exponent = 7.0

        self.stiffness = 50000.0
        
        self.surface_tension = 0.01
        self.dt[None] =  0.0004
        
    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            self.ps.acceleration[p_i] = d_v
            #if self.ps.material[p_i] == self.ps.material_fluid:
            #    self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
            #    self.ps.acceleration[p_i] = d_v
    @ti.kernel
    def advect(self):
        # [TODO] Symplectic Euler
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    def substep(self):
        self.compute_non_pressure_forces()       
        self.advect()