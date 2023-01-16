import taichi as ti
import numpy as np
from functools import reduce

_config = {
    "FluidBlocks": [
		{
            "objectId": 0,
			"start": [0.1, 0.1, 0.5],
			"end": [1.2, 2.9, 1.6],
			"translation": [0.2, 0.0, 0.2],
			"scale": [1, 1, 1],
			"velocity": [0.0, -1.0, 0.0],
			"density": 1000.0,
			"color": [50, 100, 200]
			
		}
	]
}

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
        self.m_V0 = 0.8 * self.particleDiameter ** 3 # seems a mass constant
        
        self.particle_num = ti.field(int, shape=()) # a constant

        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)

        self.padding = self.grid_size

        self.object_collection = dict()
        self.object_id_rigid_body = set()

        fluid_particle_num = 0
        for fluid in _config.FluidBlocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num

        self.particle_max_num = fluid_particle_num # + rigid_particle_num
        # no rigid block and rigid bodies

        # particle num of each grid
        self.grid_particles_num = ti.field(int, shape=(int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2])))
        self.grid_particles_num_temp = ti.field(int, shape=(int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2])))
        
        # particle related properties
        self.object_id = ti.field(dtype=int, shape=self.particle_max_num)
        self.x = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        

        self.v = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.acceleration = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.m_V = ti.field(dtype=float, shape=self.particle_max_num)
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.presure = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=float, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.is_dynamic = ti.field(dtype=float, shape=self.particle_max_num)

        """
        change this to new prefix sort 
        """
        # buffer for sort
        self.object_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.x_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.x_0_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.v_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.acceleration_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.m_V_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.density_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.material_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.color_buffer = ti.Vector.field(3, dtype=int, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)

        # Grid for each particle
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)

        self.x_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)

        # initial particles
        # fluid blocks
        fluid_blocks = _config.FluidBlocks
        for fluid in fluid_blocks:
            obj_id = fluid["objectId"]
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]
            self.add_cube(
                object_id=obj_id,
                lower_corner=start,
                cube_size=(end-start)*scale,
                velocity=velocity,
                density=density,
                is_dynamic=1,
                color=color,
                material=1
            )

    def add_particles(self,
        object_id: int,
        new_particles_num: int,
        new_particles_positions: ti.types.ndarray(),
        new_particles_velocity: ti.types.ndarray(),
        new_particles_density: ti.types.ndarray(),
        new_particles_pressures: ti.types.ndarray(),
        new_particles_materials: ti.types.ndarray(),
        new_particles_is_dynamic: ti.types.ndarray(),
        new_particles_color: ti.types.ndarray()
    ):
        self._add_particles(
            object_id,
            new_particles_num,
            new_particles_positions,
            new_particles_velocity,
            new_particles_density,
            new_particles_pressures,
            new_particles_materials,
            new_particles_is_dynamic,
            new_particles_color
        )

    def _add_particles(self,
        object_id: int,
        new_particles_num: int,
        new_particles_positions: ti.types.ndarray(),
        new_particles_velocity: ti.types.ndarray(),
        new_particles_density: ti.types.ndarray(),
        new_particles_pressures: ti.types.ndarray(),
        new_particles_materials: ti.types.ndarray(),
        new_particles_is_dynamic: ti.types.ndarray(),
        new_particles_color: ti.types.ndarray()
    ):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, 3)
            x = ti.Vector.zero(float, 3)
            for d in ti.static(range(3)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(
                p, object_id, x, v,
                new_particles_density[p - self.particle_num[None]],
                new_particles_pressures[p - self.particle_num[None]],
                new_particles_materials[p - self.particle_num[None]],
                new_particles_is_dynamic[p - self.particle_num[None]],
                ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
            )
        self.particle_num[None] += new_particles_num        

    def initialize_particle_sysytem(self):
        self.update_grid_id()
        # [TODO] change this part to grid search rather than cuda simt code
        #parallel_prefix_sum_inclusive_implace(self.grid_particles_num, self.grid_particles_num.shape[0])
        #self.count_sort()
    
    # ti.group https://docs.taichi-lang.org/docs/meta#dimensionality-independent-programming-using-grouped-indices
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for I in ti.grouped(self.x):
            grid_index = self.get_flatten_grid_index(self.x[I])
            self.grid_ids[I] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]
    
    #   get grid id for each partical position
    def get_flatten_grid_index(self, pos):
        grid_index = (pos / self.grid_size).cast(int)
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]

    def count_sort(self):
        pass

    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.object_id[p] = obj_id
        self.x[p] = x
        self.x_0[p] = x
        self.v[p] = v
        self.density[p] = density
        self.m_V[p] = self.m_V0
        self.m[p] = self.m_V0 * density
        self.pressure[p] = pressure
        self.material[p] = material
        self.is_dynamic[p] = is_dynamic
        self.color[p] = color



    def compute_cube_particle_num(self, start, end):
        num_dim = []
        for i in range(3):
            num_dim.append(np.arrange(start[i], end[i], self.particleDiameter))
        return reduce(lambda x, y: x*y, [len(n) for n in 3])

    def add_cube(self, object_id, lower_corner, cube_size, material, is_dynamic, color=(0, 0, 0),
    density=None, pressure=None, velocity=None):
        num_dim = []
        for i in range(3):
            num_dim.append(np.arrange(lower_corner[i], lower_corner[i]+cube_size[i], self.particleDiameter))
        num_new_particles = reduce(lambda x, y: x*y, [len(n) for n in num_dim])

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)

        velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)
   
    