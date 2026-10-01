# Post-processing plots for results written by outputs.OutputManager.
#
# Reads <basename>_scalars.csv and <basename>.xdmf (+ .h5) directly, so it only
# needs numpy, matplotlib and h5py: no dolfinx, no MPI. Plotting scripts can run
# on a laptop, on results copied from a cluster, while or after the solve.
#
# Usage:
#   from plots import PlotManager, compare_scalars
#   pm = PlotManager("results/cornea_IOP")
#   pm.info()
#   pm.plot_scalars(["apex_uz_anterior", "apex_uz_posterior"])
#   pm.plot_scalars("limbus_reaction_z", x="apex_uz_anterior")
#   pm.plot_field("von_mises", deformed=True, mirror=True)
#   pm.plot_field("cauchy_stress", component="tt", symmetric=True)
#   pm.plot_profile("green_lagrange", component="rr")       # along the axis r = 0
#   pm.animate_field("von_mises", "vm.gif", deformed=True)
#   plt.show()
#
# Every plot method takes an optional `ax` and returns it, so plots can be
# composed into your own figures. Nothing calls plt.show() for you.
#
# Quick look from the command line:  python plots.py results/cornea_IOP

import csv
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Pretty axis labels for the names in outputs.FIELDS / outputs.SCALARS.
# Unknown names fall back to the raw name, so new outputs work without edits here.
LABELS = {
    "t": "t",
    "apex_uz_anterior": r"anterior apex $u_z$",
    "apex_uz_posterior": r"posterior apex $u_z$",
    "central_thickness": "central thickness",
    "limbus_reaction_r": r"limbus reaction $R_r$",
    "limbus_reaction_z": r"limbus reaction $R_z$",
    "volume": "volume",
    "strain_energy": "strain energy",
    "displacement": r"$u$",
    "cauchy_stress": r"$\sigma$",
    "PK1_stress": r"$P$",
    "green_lagrange": r"$E$",
    "J": r"$J$",
    "hoop_stretch": r"$\lambda_\theta$",
    "psi": r"$\psi$",
    "von_mises": r"$\sigma_{vM}$",
    "hydrostatic_pressure": r"$p$",
}

# Components, (r, z, theta) ordering as in outputs.py; tensors are flattened row-major.
_VECTOR = {"r": 0, "z": 1}
_TENSOR3 = {"rr": 0, "rz": 1, "rt": 2, "zr": 3, "zz": 4, "zt": 5, "tr": 6, "tz": 7, "tt": 8}
_TENSOR2 = {"rr": 0, "rz": 1, "zr": 2, "zz": 3}


def _label(name, component=None):
    base = LABELS.get(name, name)
    if component is None:
        return base
    if component == "magnitude":
        return f"|{base}|"
    sub = component.replace("t", r"\theta")
    return f"{base} [{sub}]" if not base.startswith("$") else base[:-1] + f"_{{{sub}}}$"


# --------------------------------------------------------------------------- #
# XDMF / HDF5 reader (layout written by dolfinx io.XDMFFile)
# --------------------------------------------------------------------------- #
def _to_triangles(cell_type, cells):
    """Triangles for matplotlib + index of the original cell of each triangle."""
    ct = cell_type.lower()
    if ct.startswith("triangle"):
        return cells[:, :3], np.arange(len(cells))
    if ct.startswith("quadrilateral"):
        q = cells[:, :4]
        tris = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        return tris, np.tile(np.arange(len(q)), 2)
    raise ValueError(f"unsupported cell type {cell_type!r} for 2D plotting")


class _XDMFResults:
    def __init__(self, path):
        self.path = Path(path)
        domain = ET.parse(self.path).getroot().find("Domain")
        grids = domain.findall("Grid")

        # the mesh is the first uniform grid (write_mesh is called first)
        mesh = next(g for g in grids
                    if g.get("GridType", "Uniform") == "Uniform" and g.find("Topology") is not None)
        self.x = self._read(mesh.find("Geometry/DataItem"))[:, :2]
        topo = mesh.find("Topology")
        cells = self._read(topo.find("DataItem")).astype(np.int64)
        self.triangles, self.tri_cell = _to_triangles(topo.get("TopologyType"), cells)
        self.num_cells = len(cells)

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


def _component(values, name, component):
    n = values.shape[1]
    if n == 1:
        if component not in (None,):
            raise ValueError(f"{name} is scalar, no component {component!r}")
        return values[:, 0]
    if n in (2, 3):                                   # vectors (dolfinx pads 2D to 3)
        if component in (None, "magnitude"):
            return np.linalg.norm(values[:, :2], axis=1)
        if component not in _VECTOR:
            raise ValueError(f"{name}: component must be one of {list(_VECTOR)} or 'magnitude'")
        return values[:, _VECTOR[component]]
    table = _TENSOR3 if n == 9 else _TENSOR2 if n == 4 else None
    if table is None:
        raise ValueError(f"{name}: unexpected number of components {n}")
    if component not in table:
        raise ValueError(f"{name} is a tensor, choose component from {list(table)}")
    return values[:, table[component]]


# --------------------------------------------------------------------------- #
# Plot manager
# --------------------------------------------------------------------------- #
class PlotManager:
    """Plots for one run, i.e. one `basename` given to OutputManager."""

    def __init__(self, basename, label=None):
        base = Path(basename)
        self.label = label or base.name
        self.scalars = {}
        self._res = None

        csv_path = Path(f"{base}_scalars.csv")
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = np.array([[float(v) for v in r] for r in reader if r]).reshape(-1, len(header))
            self.scalars = {n: rows[:, i] for i, n in enumerate(header)}

        xdmf_path = base.with_suffix(".xdmf")
        if xdmf_path.exists():
            self._res = _XDMFResults(xdmf_path)

        if not self.scalars and self._res is None:
            raise FileNotFoundError(f"neither {csv_path} nor {xdmf_path} exists")

    # ---- inspection ------------------------------------------------------- #
    @property
    def field_names(self):
        return list(self._res.fields) if self._res else []

    def times(self, field):
        self._need_field(field)
        return self._res.fields[field]["times"].copy()

    def info(self):
        print(f"Run '{self.label}'")
        if self.scalars:
            n = len(next(iter(self.scalars.values())))
            print(f"  scalars ({n} rows): {[k for k in self.scalars if k != 't']}")
        for name in self.field_names:
            e = self._res.fields[name]
            ts = e["times"]
            print(f"  field {name:22s} {e['center']:4s}  {len(ts)} steps, "
                  f"t in [{ts.min():g}, {ts.max():g}]")

    def _need_field(self, name):
        if self._res is None or name not in self._res.fields:
            raise KeyError(f"field {name!r} not in results; available: {self.field_names}")

    # ---- scalar histories ------------------------------------------------- #
    def plot_scalars(self, y, x="t", ax=None, label=None, **plot_kw):
        """Plot one or several scalar outputs against `x` (t or another scalar)."""
        names = [y] if isinstance(y, str) else list(y)
        for n in [x, *names]:
            if n not in self.scalars:
                raise KeyError(f"scalar {n!r} not in results; available: {list(self.scalars)}")
        if ax is None:
            ax = plt.subplots()[1]
        plot_kw.setdefault("marker", "o")
        plot_kw.setdefault("markersize", 3)
        for n in names:
            if label is not None:
                lab = label if len(names) == 1 else f"{label}: {_label(n)}"
            else:
                lab = _label(n)
            ax.plot(self.scalars[x], self.scalars[n], label=lab, **plot_kw)
        ax.set_xlabel(_label(x))
        ax.set_ylabel(_label(names[0]) if len(names) == 1 else "")
        if len(names) > 1 or label is not None:
            ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # ---- fields on the (r, z) section ------------------------------------- #
    def _field_on_mesh(self, name, component, t, deformed, scale, mirror):
        """Triangulation, values, 'Node'/'Cell', actual t."""
        self._need_field(name)
        res = self._res
        raw, center, t_act = res.values(name, t)
        vals = _component(raw, name, component)

        xy = res.x.copy()
        if deformed:
            self._need_field("displacement")
            u, uc, _ = res.values("displacement", t_act)
            if uc != "Node" or len(u) != len(xy):
                raise ValueError("displacement is not nodal on the mesh geometry; cannot deform")
            xy = xy + scale * u[:, :2]

        tris = res.triangles
        if center == "Cell":
            vals = vals[res.tri_cell]                 # one value per plotted triangle
        elif len(vals) != len(xy):
            raise ValueError(f"{name}: {len(vals)} nodal values for {len(xy)} mesh nodes")

        if mirror:                                    # show the full meridian section
            n = len(xy)
            xy = np.vstack([xy, xy * [-1.0, 1.0]])
            tris = np.vstack([tris, tris[:, ::-1] + n])
            vals = np.concatenate([vals, vals])       # cylindrical components are symmetric

        return mtri.Triangulation(xy[:, 0], xy[:, 1], tris), vals, center, t_act

    def plot_field(self, name, component=None, t=None, deformed=False, scale=1.0,
                   mirror=False, show_mesh=False, symmetric=False, cmap=None,
                   vmin=None, vmax=None, colorbar=True, ax=None, title=None):
        """
        Colour map of a field at time t (None = last step) on the (r, z) section.

        component : vectors 'r', 'z', 'magnitude' (default); tensors 'rr', 'zz',
                    'tt' (hoop), 'rz', ...; scalars None.
        deformed  : draw on the deformed configuration (needs the 'displacement'
                    field in the file), displacement multiplied by `scale`.
        mirror    : also draw the r < 0 half.
        symmetric : colour range symmetric about 0 with a diverging colormap
                    (useful for signed stresses).
        """
        tri, vals, center, t_act = self._field_on_mesh(name, component, t, deformed, scale, mirror)
        if ax is None:
            ax = plt.subplots()[1]
        if symmetric and vmin is None and vmax is None:
            m = np.nanmax(np.abs(vals))
            vmin, vmax = -m, m
        cmap = cmap or ("RdBu_r" if symmetric else "viridis")

        if center == "Cell":
            pc = ax.tripcolor(tri, facecolors=vals, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            pc = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
        if show_mesh:
            ax.triplot(tri, color="k", lw=0.2, alpha=0.4)

        ax.set_aspect("equal")
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title(title or f"{_label(name, component)}   t = {t_act:g}")
        if colorbar:
            ax.figure.colorbar(pc, ax=ax, shrink=0.8)
        return ax

    # ---- line profiles ---------------------------------------------------- #
    def _trifinder(self):
        """Reference-mesh triangulation and point locator, built once."""
        if not hasattr(self, "_ref_tri"):
            res = self._res
            self._ref_tri = mtri.Triangulation(res.x[:, 0], res.x[:, 1], res.triangles)
            self._ref_finder = self._ref_tri.get_trifinder()
        return self._ref_tri, self._ref_finder

    def sample(self, name, points, component=None, t=None):
        """
        Values of a field at reference points (array (n, 2) of (r, z)); NaN outside
        the mesh. DG0 fields give the value of the containing cell (piecewise
        constant), nodal fields are interpolated linearly. Returns (values, actual t).
        """
        self._need_field(name)
        res = self._res
        raw, center, t_act = res.values(name, t)
        vals = _component(raw, name, component)
        pts = np.asarray(points, dtype=float)
        tri, finder = self._trifinder()
        if center == "Cell":
            idx = finder(pts[:, 0], pts[:, 1])
            out = np.full(len(pts), np.nan)
            ok = idx >= 0
            out[ok] = vals[res.tri_cell[idx[ok]]]
        else:
            interp = mtri.LinearTriInterpolator(tri, vals, trifinder=finder)
            out = np.ma.filled(interp(pts[:, 0], pts[:, 1]).astype(float), np.nan)
        return out, t_act

    def _axis_line(self, n):
        """Points along the symmetry axis r = 0, between the lowest and highest axis nodes."""
        x = self._res.x
        extent = np.ptp(x, axis=0).max()
        on_axis = np.isclose(x[:, 0], 0.0, atol=1e-10 * extent)
        if not on_axis.any():
            raise ValueError("no mesh node on r = 0; pass start and end explicitly")
        eps = 1e-9 * extent                           # stay just inside the mesh
        z = np.linspace(x[on_axis, 1].min() + eps, x[on_axis, 1].max() - eps, n)
        return np.c_[np.full(n, eps), z]

    def plot_profile(self, name, component=None, t=None, start=None, end=None, n=400,
                     x_axis="z", deformed=False, ax=None, label=None, **plot_kw):
        """
        Plot a field along a straight line of material points.

        start, end : (r, z) end points in the reference configuration. Default:
                     the symmetry axis r = 0, from posterior to anterior apex.
        t          : a time, a list of times, or "all" (one curve per step).
                     Default: last step.
        x_axis     : abscissa, "z", "r" or "s" (distance from start).
        deformed   : abscissa in the deformed configuration (x + u), so the curve
                     follows the material line as it moves. Needs 'displacement'.
        """
        self._need_field(name)
        if start is None and end is None:
            pts = self._axis_line(n)
        elif start is None or end is None:
            raise ValueError("give both start and end, or neither (axis r = 0)")
        else:
            s = np.linspace(0.0, 1.0, n)[:, None]
            pts = (1 - s) * np.asarray(start, float) + s * np.asarray(end, float)
        if x_axis not in ("r", "z", "s"):
            raise ValueError("x_axis must be 'r', 'z' or 's'")

        if t is None:
            times = [None]
        elif isinstance(t, str) and t == "all":
            times = list(self.times(name))
        else:
            times = list(np.atleast_1d(t))

        if ax is None:
            ax = plt.subplots()[1]
        cmap = plt.get_cmap("viridis")
        many = len(times) > 8
        t_actual = []
        for k, tk in enumerate(times):
            vals, t_act = self.sample(name, pts, component, tk)
            t_actual.append(t_act)
            xy = pts
            if deformed:
                ur, _ = self.sample("displacement", pts, "r", t_act)
                uz, _ = self.sample("displacement", pts, "z", t_act)
                xy = pts + np.c_[ur, uz]
            if x_axis == "s":
                xs = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            else:
                xs = xy[:, 0 if x_axis == "r" else 1]

            kw = dict(plot_kw)
            if len(times) > 1:
                kw.setdefault("color", cmap(k / (len(times) - 1)))
                lab = None if many else f"t = {t_act:g}"
                if label is not None and lab is not None:
                    lab = f"{label}, {lab}"
            else:
                lab = label
            ax.plot(xs, vals, label=lab, **kw)

        conf = "deformed" if deformed else "reference"
        ax.set_xlabel(f"{x_axis} ({conf})")
        ax.set_ylabel(_label(name, component))
        if len(times) == 1:
            ax.set_title(f"{_label(name, component)}   t = {t_actual[0]:g}")
        if many:
            sm = ScalarMappable(Normalize(min(t_actual), max(t_actual)), cmap)
            ax.figure.colorbar(sm, ax=ax, label="t")
        elif len(times) > 1 or label is not None:
            ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def animate_field(self, name, filename, component=None, fps=10, deformed=False,
                      scale=1.0, mirror=False, symmetric=False, **plot_kw):
        """Animate a field over all its time steps; .gif via Pillow, .mp4 needs ffmpeg."""
        times = self.times(name)
        frames = [self._field_on_mesh(name, component, t, deformed, scale, mirror) for t in times]

        # fixed colour range and axis limits over all frames
        allv = np.concatenate([f[1] for f in frames])
        vmin, vmax = plot_kw.pop("vmin", np.nanmin(allv)), plot_kw.pop("vmax", np.nanmax(allv))
        if symmetric:
            m = max(abs(vmin), abs(vmax))
            vmin, vmax = -m, m
        cmap = plot_kw.pop("cmap", None) or ("RdBu_r" if symmetric else "viridis")
        xs = np.concatenate([f[0].x for f in frames])
        ys = np.concatenate([f[0].y for f in frames])
        pad = 0.03 * max(np.ptp(xs), np.ptp(ys))
        xlim, ylim = (xs.min() - pad, xs.max() + pad), (ys.min() - pad, ys.max() + pad)

        fig, ax = plt.subplots()
        fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), ax=ax, shrink=0.8)

        def update(i):
            ax.clear()
            self.plot_field(name, component, times[i], deformed, scale, mirror,
                            cmap=cmap, vmin=vmin, vmax=vmax, colorbar=False, ax=ax, **plot_kw)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        anim = FuncAnimation(fig, update, frames=len(times))
        if str(filename).endswith(".gif"):
            anim.save(filename, writer=PillowWriter(fps=fps))
        else:
            anim.save(filename, fps=fps)
        plt.close(fig)
        return filename


def compare_scalars(managers, y, x="t", ax=None, **plot_kw):
    """Overlay the same scalar output from several runs (e.g. a parameter study)."""
    if ax is None:
        ax = plt.subplots()[1]
    for pm in managers:
        pm.plot_scalars(y, x=x, ax=ax, label=pm.label, **plot_kw)
    return ax


# --------------------------------------------------------------------------- #
# Quick look:  python plots.py results/cornea_IOP
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python plots.py <basename>")
    pm = PlotManager(sys.argv[1])
    pm.info()
    names = [n for n in pm.scalars if n != "t"]
    if names:
        cols = min(3, len(names))
        rows = -(-len(names) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
        for ax, n in zip(axes.flat, names):
            pm.plot_scalars(n, ax=ax)
        for ax in axes.flat[len(names):]:
            ax.set_visible(False)
        fig.tight_layout()
    plt.show()