# test_lrs_parallel.py
import numpy as np
from mpi4py import MPI
from dolfinx import mesh as dmesh

from Simulation_Framework import read_mesh

comm = MPI.COMM_WORLD
data = read_mesh("Usecases/cornea_3D/cornea_3D_hex_coarse.msh", comm)
dom = data.domain
tdim = dom.topology.dim
n = dom.topology.index_map(tdim).size_local
mid = dmesh.compute_midpoints(dom, tdim, np.arange(n))

cols = [mid]
for f in (data.e_1, data.e_2, data.e_3):
    dofs = f.function_space.dofmap.list[:n].ravel()
    cols.append(f.x.array.reshape(-1, 3)[dofs])
rows = comm.gather(np.hstack(cols), root=0)

if comm.rank == 0:
    R = np.vstack(rows)
    R = R[np.lexsort(np.round(R[:, :3], 8).T[::-1])]     # sort by midpoint
    np.save(f"lrs_{comm.size}.npy", R)
    print(comm.size, "rank(s):", R.shape[0], "cells")