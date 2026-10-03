# Generic output manager (dolfinx 0.12).
#
# Writes named fields to an XDMF file (one time step per call to write) and named
# scalars to a CSV file (one row per call to write). It knows nothing about the
# physics, the dimension or the mesh tags: the quantities are defined by the caller
# (framework.quantities for the standard ones, the main file for the case-specific ones).
#
# fields  : {name: expression}  or  {name: (expression, (family, degree))}
#           expression = UFL expression or fem.Function.
#           Default interpolation space: ("Lagrange", 1) for a fem.Function,
#           ("DG", 0) for any other UFL expression.
# scalars : {name: ufl.Form}      integral, compiled once and summed over ranks
#           {name: callable}      zero-argument function returning a float
#                                 (called on all ranks, must be collective-safe)
# meshtags: {name: dolfinx MeshTags}, written to the XDMF file (optional)
# meta    : dict written to <basename>_meta.json for the post-processing, normally
#           framework.output_meta: {"axes": coordinate labels, "axisymmetric": bool}.
#           gdim and tdim are added from the domain. Default: Cartesian x, y, z.
#
# Files: <basename>.xdmf (+ .h5), <basename>_scalars.csv, <basename>_meta.json
#
# Usage:
#   with OutputManager("results/run", domain, fields, scalars, meshtags, meta) as out:
#       out.write(t)

import csv
import json
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, io


def _field_spec(obj):
    """obj -> (expression, (family, degree))"""
    if isinstance(obj, tuple):
        expr, space = obj
        return expr, tuple(space)
    return obj, (("Lagrange", 1) if isinstance(obj, fem.Function) else ("DG", 0))


def _scalar_evaluator(obj, comm, name):
    """obj (ufl.Form or callable) -> zero-argument function returning a float on all ranks."""
    if isinstance(obj, ufl.Form):
        form = fem.form(obj)
        return lambda: comm.allreduce(fem.assemble_scalar(form), op=MPI.SUM)
    if callable(obj):
        return obj
    raise TypeError(f"scalar '{name}' must be a ufl.Form or a callable, got {type(obj).__name__}")


class OutputManager:
    def __init__(self, basename, domain, fields=None, scalars=None, meshtags=None, meta=None):
        fields, scalars, meshtags = fields or {}, scalars or {}, meshtags or {}
        self.domain = domain
        self.comm = domain.comm
        base = Path(basename)
        if self.comm.rank == 0:
            base.parent.mkdir(parents=True, exist_ok=True)
        self.comm.barrier()

        # ---- metadata sidecar, read by the post-processing ----
        meta = {"axes": ["x", "y", "z"], "axisymmetric": False, **(meta or {}),
                "gdim": domain.geometry.dim, "tdim": domain.topology.dim}
        if self.comm.rank == 0:
            with open(f"{base}_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

        # ---- fields: one output Function and one compiled Expression each ----
        self.fields = []
        self.xdmf = None
        spaces = {}
        for name, obj in fields.items():
            expr, space = _field_spec(obj)
            shape = expr.ufl_shape
            key = space if shape == () else (*space, shape)
            if key not in spaces:                       # share spaces between fields
                spaces[key] = fem.functionspace(domain, key)
            V = spaces[key]
            f = fem.Function(V, name=name)
            self.fields.append((f, fem.Expression(expr, V.element.interpolation_points)))

        if fields or meshtags:
            self.xdmf = io.XDMFFile(self.comm, f"{base}.xdmf", "w")
            self.xdmf.write_mesh(domain)
            for name, tags in meshtags.items():
                domain.topology.create_connectivity(tags.dim, domain.topology.dim)
                tags.name = name
                self.xdmf.write_meshtags(tags, domain.geometry)

        # ---- scalars: one compiled evaluator each, CSV written on rank 0 ----
        self.scalar_names = list(scalars)
        self.scalars = [_scalar_evaluator(obj, self.comm, name) for name, obj in scalars.items()]
        self.history = {name: [] for name in ["t", *self.scalar_names]}
        self._csv = None
        if scalars and self.comm.rank == 0:
            self._csv_file = open(f"{base}_scalars.csv", "w", newline="")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(["t", *self.scalar_names])

    def write(self, t):
        """Write all outputs at time (or load) t. Must be called on all ranks."""
        for f, expr in self.fields:
            f.interpolate(expr)
            self.xdmf.write_function(f, t)

        values = [float(evaluate()) for evaluate in self.scalars]
        for name, value in zip(["t", *self.scalar_names], [t, *values]):
            self.history[name].append(value)
        if self._csv is not None:
            self._csv.writerow([t, *values])
            self._csv_file.flush()                      # rows are on disk even if a later step fails

    def get_history(self):
        """Scalar outputs written so far, as numpy arrays (same on all ranks)."""
        return {name: np.array(v) for name, v in self.history.items()}

    def close(self):
        if self.xdmf is not None:
            self.xdmf.close()
            self.xdmf = None
        if self._csv is not None:
            self._csv_file.close()
            self._csv = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()