import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
      
    def initialize(self):
        self.ps.initialize_particle_system()
        #for r_obj_id in self.ps.object_id_rigid_body:
        #    self.compute_rigid_rest_cm(r_obj_id)
        #self.compute_static_boundary_volume()
        #self.compute_moving_boundary_volume()

    def compute_moving_boundary_volume(self):
        pass
    def substep(self):
        pass
    def solve_rigid_body(self):
        pass
    def enforce_boundary_3D(self, particle_type:int):
        pass
    def step(self):
        self.ps.initialize_particle_system()

        self.compute_moving_boundary_volume()
        self.substep()
        self.solve_rigid_body()       
        self.enforce_boundary_3D(self.ps.material_fluid)