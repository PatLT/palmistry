import numpy as np
from copy import copy 

# %% ASE shell radii
def shell_radii(ase_cell,rc,atom=0):
    a1,a2,a3=ase_cell.get_cell()[:] # Lattice vectors
    p1,p2,p3 = ase_cell.get_pbc() # Booleans for periodic b.c.s
    # Work out max number of indices to go up to.
    i_max = 0
    for i,ai in enumerate([a1,a2,a3]):
        for j,aj in enumerate([a1,a2,a3]):
            for k,ak in enumerate([a1,a2,a3]):
                if i!=j and i!=k and j!=k:
                    ind_2_plane = int(np.ceil(np.abs((rc*np.linalg.norm(np.cross(ai,aj))/np.dot(np.cross(ai,aj),ak)))))
                    if ind_2_plane > i_max:
                        i_max = copy(ind_2_plane)
    dists = []
    r0 = ase_cell.get_positions()[atom]
    for p in ase_cell.get_positions():
        for i in range(-i_max-1,i_max+2):
            for j in range(-i_max-1,i_max+2):
                for k in range(-i_max-1,i_max+2):
                    r = p+p1*i*a1+p2*j*a2+p3*k*a3 - r0
                    R = np.linalg.norm(r)
                    if R<=rc and R>1.e-7:
                        dists.append(R)
    dists = np.array(dists)
    return np.unique(dists.round(decimals=7),return_counts=True)
