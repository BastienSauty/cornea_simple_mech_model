# Helpers to build case-specific scalar outputs for the OutputManager.

import numpy as np
from mpi4py import MPI
from dolfinx import fem, geometry


def point_probe(domain, function, point):
    """
    Return a zero-argument function evaluating `function` (a fem.Function) at a fixed
    point given in the reference configuration. Works in 2D and 3D: `point` has
    domain.geometry.dim coordinates. The returned array holds all components of
    `function`, e.g. probe()[1] is u_z for a displacement. Collective: call on all ranks.
    """
    pts = np.zeros((1, 3))
    pts[0, :len(point)] = point
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, pts)
    cells = geometry.compute_colliding_cells(domain, candidates, pts).links(0)
    comm = domain.comm

    if comm.allreduce(int(len(cells) > 0), op=MPI.SUM) == 0:
        raise ValueError(f"point {tuple(point)} is not inside the mesh")

    def probe():
        local = np.asarray(function.eval(pts, cells[:1])).reshape(-1) if len(cells) else None
        values = [v for v in comm.allgather(local) if v is not None]
        return values[0]

    return probe


def average(numerator, denominator, comm):
    """
    Zero-argument function returning (integral of numerator) / (integral of denominator),
    both ufl.Form, summed over ranks. For a mean over a surface or a volume:
    average(mech.surface_integral(f, tag), mech.surface_integral(one, tag), comm).
    """
    num, den = fem.form(numerator), fem.form(denominator)
    return lambda: (comm.allreduce(fem.assemble_scalar(num), op=MPI.SUM)
                    / comm.allreduce(fem.assemble_scalar(den), op=MPI.SUM))