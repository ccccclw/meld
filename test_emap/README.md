### Grid force feature 
---
<!-- - [x] linear interpolation (reference)
- [x] linear interpolation test(reference) -->

##### Description
The map should contain the following information:

- origin coordinate 
- each grid size and map dimension
- density value on grid points

First, the potential is converted from density values from the following to scale density values lower than threshold to constant and other potential values are scaled inversely with their density.

```python
def map_potential(emap, threshold, scale_factor):
    emap_0 = scale_factor*((emap - threshold)/(emap.max() - threshold))
    emap_where=np.where(emap_0<=0)
    emap_pot = scale_factor*(1-(emap - threshold)/(emap.max() - threshold))
    emap_pot[emap_where[0],emap_where[1],emap_where[2]] = scale_factor  
    return emap_pot
```

Currently, the potential values for all replicas are created during setup, e.g., if the grid potential on all 8 replicas is the same, we will use the following to add grid force.

```python
    s.restraints.create_restraint('emap',map_scaler,atom_res_index=list(range(1,n_res+1)),
    atom_name=emap_atoms, cubic=0,mu=np.array([np.matrix.flatten(map_pot).tolist()]*8),
    map_origin=[origin_x,origin_y,origin_z],map_dimension=[dimension_x,dimension_y,dimension_z]],
    map_gridLength=[length_x,length_y,length_z])
```

when ```cubic=0```, ```mu``` should be provided with map potential; when ```cubic=1```, ```cubic_mu``` should be provided with calculated coefficients for cubic interpolation based on map potential.

 - [x] multiple map restraints can be used selectively in linear interpolation.
 - [x] linear interpolation (reference/cuda)
 - [x] tricubic interpolation (cuda)
 - [x] grid force reference test





