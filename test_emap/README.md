## Grid force feature 



#### Description
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

the blurring can be done with

```python
tmp_pot =  np.matrix.flatten(scipy.ndimage.gaussian_filter(map_pot,1))
tmp_pot = (tmp_pot-tmp_pot.min())*0.3/(tmp_pot.max()-tmp_pot.min())
```

#### Speed
The speed is tested with and without density restraint, which shows it has little computation cost.

With density restraint
```
14:10:35 INFO meld.remd.leader: Running replica exchange step 2 of 2000.
14:10:38 INFO meld.remd.leader: Running replica exchange step 3 of 2000.
14:10:41 INFO meld.remd.leader: Running replica exchange step 4 of 2000.
14:10:45 INFO meld.remd.leader: Running replica exchange step 5 of 2000.
14:10:48 INFO meld.remd.leader: Running replica exchange step 6 of 2000.
14:10:51 INFO meld.remd.leader: Running replica exchange step 7 of 2000.
14:10:54 INFO meld.remd.leader: Running replica exchange step 8 of 2000.
14:10:57 INFO meld.remd.leader: Running replica exchange step 9 of 2000.
14:11:00 INFO meld.remd.leader: Running replica exchange step 10 of 2000.
```
Without density restraint
```
14:07:00 INFO meld.remd.leader: Running replica exchange step 2 of 2000.
14:07:03 INFO meld.remd.leader: Running replica exchange step 3 of 2000.
14:07:06 INFO meld.remd.leader: Running replica exchange step 4 of 2000.
14:07:09 INFO meld.remd.leader: Running replica exchange step 5 of 2000.
14:07:12 INFO meld.remd.leader: Running replica exchange step 6 of 2000.
14:07:15 INFO meld.remd.leader: Running replica exchange step 7 of 2000.
14:07:18 INFO meld.remd.leader: Running replica exchange step 8 of 2000.
14:07:20 INFO meld.remd.leader: Running replica exchange step 9 of 2000.
14:07:23 INFO meld.remd.leader: Running replica exchange step 10 of 2000.
```

Currently, we have
 - [x] multiple map restraints can be used selectively in linear interpolation.
 - [x] linear interpolation (reference/cuda)
 - [x] tricubic interpolation for one map restraint (cuda)
 - [x] grid force reference test





