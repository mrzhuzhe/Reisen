import taichi as ti
from base import SPHBase


class WCSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
    def substep(self):
        pass