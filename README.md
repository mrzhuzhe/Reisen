# Reisen
PBD PBF

## 
```python 
# /pbf 

# 3d position base fluid
python3 fluid.py

# taichi multithread for loop 
python3 case_study.py

# /pendulum

# position base dynamic pendulum float point underflow bug reproduce
python3 simple.py # compare to python3 simple-fp64.py

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
2. HPC MPI
3. complier LLVM 
4. c++ 11
5. autodiff RL
6. fluid fem sph dem
7. geometry 
8. optimize 
9. matrix solver