# Reading of the results written by OutputManager for one basename:
#   <basename>_scalars.csv   scalar histories
#   <basename>_meta.json     coordinate labels, axisymmetric flag, dimensions
#   <basename>.xdmf (+ .h5)  fields and mesh (layout written by dolfinx io.XDMFFile)
#
# numpy and h5py only: no dolfinx, no MPI, no matplotlib. Works for any dimension;
# the plotting restrictions (2D only) belong to plots_manager.

import csv
import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

DEFAULT_AXES = ("x", "y", "z")


# --------------------------------------------------------------------------- #
# Scalars and metadata
# --------------------------------------------------------------------------- #
def read_scalars(base):
    """{column name: array} from <base>_scalars.csv, or {} if the file does not exist."""
    path = Path(f"{base}_scalars.csv")
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = np.array([[float(v) for v in r] for r in reader if r]).reshape(-1, len(header))
    return {n: rows[:, i] for i, n in enumerate(header)}


def read_meta(base, gdim=None):
    """
    Metadata of the run: {"axes": 3 labels, "axisymmetric": bool, "gdim": int}.
    Read from <base>_meta.json; without this file, Cartesian axes (x, y, z) are
    assumed and gdim is the value given (inferred from the mesh by the caller).
    """
    meta = {"axes": DEFAULT_AXES, "axisymmetric": False, "gdim": gdim}
    path = Path(f"{base}_meta.json")
    if path.exists():
        with open(path) as f:
            meta.update(json.load(f))
    meta["axes"] = tuple(meta["axes"])
    return meta


# --------------------------------------------------------------------------- #
# XDMF / HDF5
# --------------------------------------------------------------------------- #
class XDMFResults:
    def __init__(self, path):
        self.path = Path(path)
        domain = ET.parse(self.path).getroot().find("Domain")
        grids = domain.findall("Grid")

        # the mesh is the first uniform grid (write_mesh is called first)
        mesh = next(g for g in grids
                    if g.get("GridType", "Uniform") == "Uniform" and g.find("Topology") is not None)
        self.x = self._read(mesh.find("Geometry/DataItem"))             # (n_nodes, 3)
        topo = mesh.find("Topology")
        self.cell_type = topo.get("TopologyType")
        self.cells = self._read(topo.find("DataItem")).astype(np.int64)
        self.num_cells = len(self.cells)

        # time series: name -> {"center", "times", "items"}
        self.fields = {}
        for g in grids:
            if g.get("GridType") != "Collection":
                continue
            for step in g.findall("Grid"):
                time = step.find("Time")
                t = float(time.get("Value")) if time is not None else 0.0
                for att in step.findall("Attribute"):
                    e = self.fields.setdefault(att.get("Name"),
                                               {"center": att.get("Center"), "times": [], "items": []})
                    e["times"].append(t)
                    e["items"].append(att.find("DataItem"))
        for e in self.fields.values():
            e["times"] = np.array(e["times"])

    @property
    def dim(self):
        """Dimension of the mesh cells (2 or 3), from the cell type."""
        ct = self.cell_type.lower()
        if ct.startswith(("triangle", "quadrilateral")):
            return 2
        if ct.startswith(("tetrahedron", "hexahedron", "wedge", "pyramid")):
            return 3
        return 1

    def _read(self, item):
        dims = tuple(int(d) for d in item.get("Dimensions").split())
        text = item.text.strip()
        if item.get("Format", "XML") == "HDF":
            fname, dset = text.rsplit(":", 1)
            with h5py.File(self.path.parent / fname, "r") as h5:
                return np.asarray(h5[dset]).reshape(dims)
        return np.array(text.split(), dtype=float).reshape(dims)

    def step_index(self, name, t):
        times = self.fields[name]["times"]
        if t is None:
            return len(times) - 1
        i = int(np.argmin(np.abs(times - t)))
        if not np.isclose(times[i], t, rtol=1e-8, atol=1e-12):
            warnings.warn(f"{name}: no output at t={t:g}, using nearest t={times[i]:g}")
        return i

    def values(self, name, t=None):
        """(values of shape (n, ncomp), center 'Node' or 'Cell', actual time)."""
        e = self.fields[name]
        i = self.step_index(name, t)
        v = self._read(e["items"][i])
        return v.reshape(len(v), -1), e["center"], e["times"][i]


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def extract_component(values, name, component, axes, gdim):
    """
    One component of a field, as an array of length n.

    values    : (n, ncomp) as returned by XDMFResults.values
    component : scalars None; vectors an axis label ('r', 'z', ...) or 'magnitude'
                (default); tensors two axis labels ('rr', 'zt', 'xy', ...).
    axes      : the three coordinate labels of the run, e.g. ('r', 'z', 't') or ('x', 'y', 'z')
    gdim      : geometric dimension; the magnitude of a vector uses its first gdim components
                (dolfinx pads 2D vectors to 3 components).
    Tensors are stored row-major.
    """
    n = values.shape[1]
    if n == 1:
        if component is not None:
            raise ValueError(f"{name} is scalar, no component {component!r}")
        return values[:, 0]
    if n in (2, 3):
        if component in (None, "magnitude"):
            return np.linalg.norm(values[:, :gdim], axis=1)
        table = {a: i for i, a in enumerate(axes[:gdim])}
        if component not in table:
            raise ValueError(f"{name}: component must be one of {list(table)} or 'magnitude'")
        return values[:, table[component]]
    if n in (4, 9):
        m = 2 if n == 4 else 3
        table = {axes[i] + axes[j]: m * i + j for i in range(m) for j in range(m)}
        if component not in table:
            raise ValueError(f"{name} is a tensor, choose component from {list(table)}")
        return values[:, table[component]]
    raise ValueError(f"{name}: unexpected number of components {n}")
