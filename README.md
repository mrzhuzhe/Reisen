# Reisen

> Test playground for Taichi project 

## 
```python 

# /euler

# euler grid smoke 
python3 test.py

# /pbf 

# 3d position base fluid
python3 fluid.py

# taichi multithread for loop 
python3 case_study.py

# /pendulum

# position base dynamic pendulum float point underflow bug reproduce
python3 simple.py # compare to python3 simple-fp64.py

# /snode

# taichi bitmask and dynamic snode test , logic same as pdf

#  /sph 

# sph fluid rigid 
python3 main.py 


```


## References

0. ten minutes physics https://matthias-research.github.io/pages/tenMinutePhysics/
1. https://github.com/matthias-research/pages/blob/master/challenges/fluid2d.html
2. https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/contribs/PBFBoundary.html
3. taichi fp64 https://docs.taichi-lang.org/docs/type
4. other sph resources https://interactivecomputergraphics.github.io/physics-simulation/
5. snodes https://docs.taichi-lang.org/docs/internal#data-structure-organization
6. dynamic snode https://docs.taichi-lang.org/docs/sparse#dynamic-snode


# Roadmap

1. compliant control 
2. autodiff RL
3. fluid fem sph dem
4. geometry 
5. optimize 
6. matrix solver



# TODOS
1. In sph project original sph taichi use cuda instruction shfl_up_i32 https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/ 