# Simulation_Framework : overlay on fenicsx to run simulations with various
# mechanical contexts in a systematic way, adapted for the cornea.
#
# Imports are lazy: a module is loaded only when one of its names is used, so
# `from Simulation_Framework import PlotManager` works without dolfinx installed.

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "Hyperelastic_framework": ".mechanical_axisymmetric_framework",
    "OutputManager":          ".outputs_manager",
    "available_outputs":      ".outputs_manager",
    "MechParams":             ".parameters_class",
    "PlotManager":            ".plots_manager",
    "compare_scalars":        ".plots_manager",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value          # cache: next access is a normal lookup
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:                        # lets IDEs / linters see the names
    from .mechanical_axisymmetric_framework import Hyperelastic_framework
    from .outputs_manager import OutputManager, available_outputs
    from .parameters_class import MechParams
    from .plot_manager import PlotManager, compare_scalars