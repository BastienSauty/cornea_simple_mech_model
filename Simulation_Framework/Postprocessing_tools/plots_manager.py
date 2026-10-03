# Post-processing plots for results written by OutputManager.
#
# Reads <basename>_scalars.csv, <basename>_meta.json and <basename>.xdmf (+ .h5) through
# results_reader, so it only needs numpy, matplotlib and h5py: no dolfinx, no MPI.
#
# Scope: scalar histories (any run), and fields / profiles on 2D meshes (planar or
# axisymmetric). For 3D results open the .xdmf file in ParaView.
#
# Usage:
#   pm = PlotManager("results/run", labels={"apex_uz": r"apex $u_z$"})
#   pm.info()
#   pm.plot_scalars("apex_uz", save="apex.png")                  # stand-alone figure, saved
#   pm.plot_scalars("force", x="apex_uz", ax=ax)                 # composed into your own figure
#   pm.plot_field("von_mises", deformed=True, mirror=True, save="vm.png")
#   pm.plot_field("cauchy_stress", component="tt", symmetric=True, save="stt.png")
#   pm.plot_profile("green_lagrange", component="zz", save="Ezz_axis.png")
#   pm.animate_field("von_mises", "vm.gif", deformed=True)
#
# Every plot method takes an optional `ax` and returns it. With `save=<file>` the method
# creates its own figure, saves it and closes it (do not combine with `ax`). Nothing
# calls plt.show() for you.
#
# Components are named with the coordinate labels of the run (<basename>_meta.json):
# 'r', 'z' and 'tt' (hoop) for an axisymmetric run, 'x', 'y', 'xy', ... otherwise.

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .results_reader import XDMFResults, extract_component, read_meta, read_scalars

# Labels of the standard quantities (framework.quantities) and of generic scalars.
# Case-specific names are given to PlotManager(labels=...); unknown names fall back to
# the raw name.
DEFAULT_LABELS = {
    "t": "t",
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


def _axes(ax, save, figsize):
    """The axes to draw on: the given one, or a new stand-alone figure."""
    if ax is not None:
        if save is not None:
            raise ValueError("save is for stand-alone figures: do not pass ax, "
                             "or save the figure yourself")
        return ax
    return plt.subplots(figsize=figsize or (7, 5))[1]


def _finish(ax, save, dpi=200):
    """Save and close the figure if save was given."""
    if save is not None:
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(save, dpi=dpi)
        plt.close(fig)


class PlotManager:
    """Plots for one run, i.e. one `basename` given to OutputManager."""

    def __init__(self, basename, label=None, labels=None, meta=None):
        """
        label  : name of the run in legends (default: the file name)
        labels : {output name: matplotlib label}, merged with DEFAULT_LABELS
        meta   : overrides the content of <basename>_meta.json (or of its defaults),
                 e.g. {"axes": ("r", "z", "t"), "axisymmetric": True}
        """
        base = Path(basename)
        self.label = label or base.name
        self.labels = {**DEFAULT_LABELS, **(labels or {})}
        self.scalars = read_scalars(base)

        xdmf_path = Path(f"{base}.xdmf")
        self._res = XDMFResults(xdmf_path) if xdmf_path.exists() else None
        if not self.scalars and self._res is None:
            raise FileNotFoundError(f"neither {base}_scalars.csv nor {xdmf_path} exists")

        self.meta = {**read_meta(base, self._res.dim if self._res else None), **(meta or {})}
        self.axes = tuple(self.meta["axes"])
        self.axisymmetric = bool(self.meta["axisymmetric"])
        self.gdim = self.meta["gdim"]

    # ---- labels and checks ------------------------------------------------ #
    def _lab(self, name, component=None):
        base = self.labels.get(name, name)
        if component is None:
            return base
        if component == "magnitude":
            return f"|{base}|"
        sub = component.replace("t", r"\theta") if "t" in self.axes else component
        return f"{base} [{sub}]" if not base.startswith("$") else base[:-1] + f"_{{{sub}}}$"

    def _need_field(self, name):
        if self._res is None or name not in self._res.fields:
            raise KeyError(f"field {name!r} not in results; available: {self.field_names}")

    def _need_2d(self):
        if self.gdim != 2:
            raise NotImplementedError("field plots and profiles are 2D only: "
                                      "open the .xdmf file in ParaView for 3D results")

    @property
    def _triangles(self):
        if not hasattr(self, "_tri_cache"):
            self._tri_cache = _to_triangles(self._res.cell_type, self._res.cells)
        return self._tri_cache

    def _component(self, raw, name, component):
        return extract_component(raw, name, component, self.axes, self.gdim)

    # ---- inspection ------------------------------------------------------- #
    @property
    def field_names(self):
        return list(self._res.fields) if self._res else []

    def times(self, field):
        self._need_field(field)
        return self._res.fields[field]["times"].copy()

    def info(self):
        print(f"Run '{self.label}'")
        kind = "axisymmetric" if self.axisymmetric else "Cartesian"
        print(f"  {self.gdim}D, {kind}, coordinates {self.axes}")
        if self.scalars:
            n = len(next(iter(self.scalars.values())))
            print(f"  scalars ({n} rows): {[k for k in self.scalars if k != 't']}")
        for name in self.field_names:
            e = self._res.fields[name]
            ts = e["times"]
            print(f"  field {name:22s} {e['center']:4s}  {len(ts)} steps, "
                  f"t in [{ts.min():g}, {ts.max():g}]")

    # ---- scalar histories ------------------------------------------------- #
    def plot_scalars(self, y, x="t", ax=None, label=None, save=None, figsize=None, **plot_kw):
        """Plot one or several scalar outputs against `x` (t or another scalar)."""
        names = [y] if isinstance(y, str) else list(y)
        for n in [x, *names]:
            if n not in self.scalars:
                raise KeyError(f"scalar {n!r} not in results; available: {list(self.scalars)}")
        ax = _axes(ax, save, figsize)
        plot_kw.setdefault("marker", "o")
        plot_kw.setdefault("markersize", 3)
        for n in names:
            if label is not None:
                lab = label if len(names) == 1 else f"{label}: {self._lab(n)}"
            else:
                lab = self._lab(n)
            ax.plot(self.scalars[x], self.scalars[n], label=lab, **plot_kw)
        ax.set_xlabel(self._lab(x))
        ax.set_ylabel(self._lab(names[0]) if len(names) == 1 else "")
        if len(names) > 1 or label is not None:
            ax.legend()
        ax.grid(True, alpha=0.3)
        _finish(ax, save)
        return ax

    # ---- fields on a 2D mesh ---------------------------------------------- #
    def _field_on_mesh(self, name, component, t, deformed, scale, mirror):
        """Triangulation, values, 'Node'/'Cell', actual t."""
        self._need_2d()
        self._need_field(name)
        if mirror and not self.axisymmetric:
            raise ValueError("mirror is only meaningful for an axisymmetric run")
        res = self._res
        raw, center, t_act = res.values(name, t)
        vals = self._component(raw, name, component)

        xy = res.x[:, :2].copy()
        if deformed:
            self._need_field("displacement")
            u, uc, _ = res.values("displacement", t_act)
            if uc != "Node" or len(u) != len(xy):
                raise ValueError("displacement is not nodal on the mesh geometry; cannot deform")
            xy = xy + scale * u[:, :2]

        tris, tri_cell = self._triangles
        if center == "Cell":
            vals = vals[tri_cell]                     # one value per plotted triangle
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
                   vmin=None, vmax=None, colorbar=True, ax=None, title=None,
                   save=None, figsize=None):
        """
        Colour map of a field at time t (None = last step) on a 2D mesh.

        component : vectors: an axis label ('r', 'z', 'x', ...) or 'magnitude' (default);
                    tensors: two labels ('rr', 'zz', 'tt' hoop, 'rz', 'xy', ...); scalars None.
        deformed  : draw on the deformed configuration (needs the 'displacement' field in
                    the file), displacement multiplied by `scale`.
        mirror    : also draw the r < 0 half (axisymmetric runs).
        symmetric : colour range symmetric about 0 with a diverging colormap
                    (useful for signed stresses).
        save      : file name; creates a stand-alone figure, saves it and closes it.
        """
        tri, vals, center, t_act = self._field_on_mesh(name, component, t, deformed, scale, mirror)
        ax = _axes(ax, save, figsize)
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
        ax.set_xlabel(self.axes[0])
        ax.set_ylabel(self.axes[1])
        ax.set_title(title or f"{self._lab(name, component)}   t = {t_act:g}")
        if colorbar:
            ax.figure.colorbar(pc, ax=ax, shrink=0.8)
        _finish(ax, save)
        return ax

    # ---- line profiles ---------------------------------------------------- #
    def _trifinder(self):
        """Reference-mesh triangulation and point locator, built once."""
        if not hasattr(self, "_ref_tri"):
            res = self._res
            self._ref_tri = mtri.Triangulation(res.x[:, 0], res.x[:, 1], self._triangles[0])
            self._ref_finder = self._ref_tri.get_trifinder()
        return self._ref_tri, self._ref_finder

    def sample(self, name, points, component=None, t=None):
        """
        Values of a field at reference points (array (n, 2)); NaN outside the mesh.
        DG0 fields give the value of the containing cell (piecewise constant), nodal
        fields are interpolated linearly. Returns (values, actual t).
        """
        self._need_2d()
        self._need_field(name)
        res = self._res
        raw, center, t_act = res.values(name, t)
        vals = self._component(raw, name, component)
        pts = np.asarray(points, dtype=float)
        tri, finder = self._trifinder()
        if center == "Cell":
            idx = finder(pts[:, 0], pts[:, 1])
            out = np.full(len(pts), np.nan)
            ok = idx >= 0
            out[ok] = vals[self._triangles[1][idx[ok]]]
        else:
            interp = mtri.LinearTriInterpolator(tri, vals, trifinder=finder)
            out = np.ma.filled(interp(pts[:, 0], pts[:, 1]).astype(float), np.nan)
        return out, t_act

    def _axis_line(self, n):
        """Points along the symmetry axis (first coordinate = 0), lowest to highest axis node."""
        x = self._res.x
        extent = np.ptp(x[:, :2], axis=0).max()
        on_axis = np.isclose(x[:, 0], 0.0, atol=1e-10 * extent)
        if not on_axis.any():
            raise ValueError(f"no mesh node on {self.axes[0]} = 0; pass start and end explicitly")
        eps = 1e-9 * extent                           # stay just inside the mesh
        z = np.linspace(x[on_axis, 1].min() + eps, x[on_axis, 1].max() - eps, n)
        return np.c_[np.full(n, eps), z]

    def plot_profile(self, name, component=None, t=None, start=None, end=None, n=400,
                     x_axis=None, deformed=False, ax=None, label=None,
                     save=None, figsize=None, **plot_kw):
        """
        Plot a field along a straight line of material points.

        start, end : end points in the reference configuration. Default (axisymmetric
                     runs only): the symmetry axis, from lowest to highest node.
        t          : a time, a list of times, or "all" (one curve per step).
                     Default: last step.
        x_axis     : abscissa, an axis label or "s" (distance from start).
                     Default: the second coordinate.
        deformed   : abscissa in the deformed configuration (x + u), so the curve
                     follows the material line as it moves. Needs 'displacement'.
        """
        self._need_2d()
        self._need_field(name)
        x_axis = x_axis or self.axes[1]
        if x_axis not in (self.axes[0], self.axes[1], "s"):
            raise ValueError(f"x_axis must be '{self.axes[0]}', '{self.axes[1]}' or 's'")
        if start is None and end is None:
            if not self.axisymmetric:
                raise ValueError("give start and end (this run has no symmetry axis)")
            pts = self._axis_line(n)
        elif start is None or end is None:
            raise ValueError("give both start and end, or neither")
        else:
            s = np.linspace(0.0, 1.0, n)[:, None]
            pts = (1 - s) * np.asarray(start, float) + s * np.asarray(end, float)

        if t is None:
            times = [None]
        elif isinstance(t, str) and t == "all":
            times = list(self.times(name))
        else:
            times = list(np.atleast_1d(t))

        ax = _axes(ax, save, figsize)
        cmap = plt.get_cmap("viridis")
        many = len(times) > 8
        t_actual = []
        for k, tk in enumerate(times):
            vals, t_act = self.sample(name, pts, component, tk)
            t_actual.append(t_act)
            xy = pts
            if deformed:
                u0, _ = self.sample("displacement", pts, self.axes[0], t_act)
                u1, _ = self.sample("displacement", pts, self.axes[1], t_act)
                xy = pts + np.c_[u0, u1]
            if x_axis == "s":
                xs = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            else:
                xs = xy[:, self.axes.index(x_axis)]

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
        ax.set_ylabel(self._lab(name, component))
        if len(times) == 1:
            ax.set_title(f"{self._lab(name, component)}   t = {t_actual[0]:g}")
        if many:
            sm = ScalarMappable(Normalize(min(t_actual), max(t_actual)), cmap)
            ax.figure.colorbar(sm, ax=ax, label="t")
        elif len(times) > 1 or label is not None:
            ax.legend()
        ax.grid(True, alpha=0.3)
        _finish(ax, save)
        return ax

    # ---- animation -------------------------------------------------------- #
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


def quick_look(basename):
    """One panel per scalar output of a run. Call plt.show() afterwards."""
    pm = PlotManager(basename)
    pm.info()
    names = [n for n in pm.scalars if n != "t"]
    if not names:
        return pm
    cols = min(3, len(names))
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    for ax, n in zip(axes.flat, names):
        pm.plot_scalars(n, ax=ax)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.tight_layout()
    return pm
