# Simulation_Framework : overlay on fenicsx to run simulations with various
# mechanical contexts in a systematic way, adapted for the cornea.
#
# Imports are lazy: a module is loaded only when one of its names is used, so
# `from Simulation_Framework import PlotManager` works without dolfinx installed.

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "Hyperelastic_axisymmetric_framework":  ".hyperelastic_framework",
    "Hyperelastic_3D_framework":  ".hyperelastic_framework",
    "read_mesh":               ".mesh_io",
    "MeshData":                ".mesh_io",
    "write_mesh":              ".mesh_io",
    "top_cell_centroids":      ".mesh_io",
    "MechParams":              ".parameters_class",

    "OutputManager":           ".Postprocessing_tools.outputs_manager",
    "point_probe":             ".Postprocessing_tools.output_utils",
    "average":                 ".Postprocessing_tools.output_utils",

    "PlotManager":             ".Postprocessing_tools.plots_manager",
    "compare_scalars":         ".Postprocessing_tools.plots_manager",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value          # cache: next access is a normal lookup
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:                        # lets IDEs / linters see the names
    from .hyperelastic_framework import Hyperelastic_axisymmetric_framework, Hyperelastic_3D_framework
    from .mesh_io import read_mesh, MeshData, top_cell_centroids
    from .parameters_class import MechParams

    from .Postprocessing_tools.outputs_manager import OutputManager
    from .Postprocessing_tools.output_utils import point_probe, average
    from .Postprocessing_tools.plots_manager import PlotManager, compare_scalars