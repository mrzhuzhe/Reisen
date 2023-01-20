import taichi as ti
import numpy as np
from functools import reduce
from config import _config
from wcsph import WCSPHSolver

@ti.data_oriented
class ParticleSystem:
    def __init__(self):
        self.domain_start = np.array([0, 0, 0])
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

        # config 
        self.particle_radius = _config["Configuration"]["particleRadius"]

        fluid_particle_num = 0
        for fluid in _config["FluidBlocks"]:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num

        #### Process Rigid Blocks ####
        rigid_blocks = _config["RigidBlocks"]
        rigid_particle_num = 0
        for rigid in rigid_blocks:
            particle_num = self.compute_cube_particle_num(rigid["start"], rigid["end"])
            rigid["particleNum"] = particle_num
            self.object_collection[rigid["objectId"]] = rigid
            rigid_particle_num += particle_num

        self.particle_max_num = fluid_particle_num + rigid_particle_num
        # no rigid block and rigid bodies
        
        # particle related properties
        self.object_id = ti.field(dtype=int, shape=self.particle_max_num)
        self.x = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        

        self.v = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.acceleration = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.m_V = ti.field(dtype=float, shape=self.particle_max_num)   # TODO ？
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.density = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)
        self.color = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)
        
        ##
        total_grid_num = self.grid_num[0]*self.grid_num[1]*self.grid_num[2]

        self.list_head = ti.field(dtype=ti.i32, shape=total_grid_num)
        self.list_cur = ti.field(dtype=ti.i32, shape=total_grid_num)
        self.list_tail = ti.field(dtype=ti.i32, shape=total_grid_num)

        self.grain_count = ti.field(dtype=ti.i32,
                            shape=(self.grid_num[0], self.grid_num[1], self.grid_num[2]),
                            name="grain_count")
        self.column_sum = ti.field(dtype=ti.i32, shape=(self.grid_num[0], self.grid_num[1]), name="column_sum")
        self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.grid_num[0], self.grid_num[1]), name="prefix_sum")
        self.particle_id = ti.field(dtype=ti.i32, shape=self.particle_max_num, name="particle_id")
        ##

        """
        self.x_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
        self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)          
        """

        # initial particles
        # fluid blocks
        #"""
        fluid_blocks = _config["FluidBlocks"]
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
        #"""
        
            
        for rigid_block in _config["RigidBlocks"]:                   
            obj_id = rigid_block["objectId"]
            offset = np.array(rigid_block["translation"])
            start = np.array(rigid_block["start"]) + offset            
            end = np.array(rigid_block["end"]) + offset
            scale = np.array(rigid_block["scale"])           
            velocity = rigid_block["velocity"]
            density = rigid_block["density"]
            color = rigid_block["color"]
            self.object_id_rigid_body.add(obj_id)
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=1, # enforce fluid dynamic
                          color=color,
                          material=0) # 1 indicates fluid

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
    @ti.kernel
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

    def initialize_particle_system(self):        
        self.findNeighbors()
    
    @ti.kernel
    def findNeighbors(self):
        self.grain_count.fill(0)

        for i in range(self.particle_max_num):
            #grid_idx = ti.floor(pos[i]* grid_n, int)
            grid_idx = self.get_flatten_grid_index(self.x[i])            
            self.grain_count[grid_idx] += 1
        
        self.column_sum.fill(0)
        # kernel comunicate with global variable ???? this is a bit amazing 
        for i, j, k in ti.ndrange(self.grid_num[0], self.grid_num[1], self.grid_num[2]):        
            ti.atomic_add(self.column_sum[i, j], self.grain_count[i, j, k])

        # this is because memory mapping can be out of order
        _prefix_sum_cur = 0    
        for i, j in ti.ndrange(self.grid_num[0], self.grid_num[1]):
            self.prefix_sum[i, j] = ti.atomic_add(_prefix_sum_cur, self.column_sum[i, j])
        
            
        for i, j, k in ti.ndrange(self.grid_num[0], self.grid_num[1], self.grid_num[2]): 
            # we cannot visit prefix_sum[i,j] in this loop
            pre = ti.atomic_add(self.prefix_sum[i,j], self.grain_count[i, j, k])        
            linear_idx = i * self.grid_num[1] * self.grid_num[2] + j * self.grid_num[2] + k
            self.list_head[linear_idx] = pre
            self.list_cur[linear_idx] = self.list_head[linear_idx]
            # only pre pointer is useable
            self.list_tail[linear_idx] = pre + self.grain_count[i, j, k]       

        for i in range(self.particle_max_num):
            grid_idx = self.get_flatten_grid_index(self.x[i])
            linear_idx = grid_idx[0] * self.grid_num[1] * self.grid_num[2] + grid_idx[1] * self.grid_num[2] + grid_idx[2]
            grain_location = ti.atomic_add(self.list_cur[linear_idx], 1)
            self.particle_id[grain_location] = i

    
    #   get grid id for each partical position
    @ti.func
    def get_flatten_grid_index(self, pos):
        grid_index = (pos / self.grid_size).cast(int)
        #return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]
        return (grid_index[0], grid_index[1] , grid_index[2])

    @ti.func
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
            num_dim.append(np.arange(start[i], end[i], self.particleDiameter))
        return reduce(lambda x, y: x*y, [len(n) for n in num_dim])

    def add_cube(self, object_id, lower_corner, cube_size, material, is_dynamic, color=(0, 0, 0),
    density=None, pressure=None, velocity=None):
        num_dim = []
        for i in range(3):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i]+cube_size[i], self.particleDiameter))
        num_new_particles = reduce(lambda x, y: x*y, [len(n) for n in num_dim])

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)

        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()

        #print("new_positions.shape", new_positions.shape)
        
        velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)
   
    def build_solver(self):        
        return WCSPHSolver(self)

    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == self.material_solid and (not self.is_dynamic[p])
    
    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]
    
    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = (self.x[p_i] / self.grid_size).cast(int)
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            _neigh = center_cell + offset
            neigh_linear_idx = _neigh[0] * self.grid_num[1] * self.grid_num[2] + _neigh[1] * self.grid_num[2] + _neigh[2]            
            for p_j in range(self.list_head[neigh_linear_idx],
                            self.list_tail[neigh_linear_idx]): 
                j = self.particle_id[p_j]	               
                if p_i[0] != j and (self.x[p_i] - self.x[j]).norm() < self.support_radius:
                    task(p_i, j, ret)
