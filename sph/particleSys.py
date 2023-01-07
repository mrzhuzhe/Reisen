import taichi as ti
import numpy as np

@ti.data_oriented
class ParticleSystem:
    def __init__(self):
        self.domain_start = np.array([0, 0, 0])
        #self.domain_end = np.array([1, 1, 1])
        self.domain_end = np.array([5, 3, 2])
        self.domain_size = self.domain_end - self.domain_start

        self.material_solid = 0
        self.material_fluid = 1
        
        self.particleRadius = 0.01
        self.particleDiameter = 2 * self.particleRadius

        self.support_radius = 4 * self.particleRadius # eq to grid size 
        pass