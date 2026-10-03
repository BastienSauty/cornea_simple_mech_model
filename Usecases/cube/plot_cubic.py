# Plotting script for the cube test. Runs without dolfinx:
#   python plot_cubic.py
# 3D fields are not plotted here: open results/<name>.xdmf in ParaView.
import matplotlib.pyplot as plt
from Simulation_Framework import PlotManager

name = "Usecases/cube/cubetest"
out = f"results/{name}"

# labels of the case-specific scalars; "t" is the applied pressure in this run
labels = {
    "t": "pressure [MPa]",
    "uz_Z1": r"mean $u_z$ on $Z_1$",
    "sigma_zz_mean": r"mean $\sigma_{zz}$ [MPa]",
}

pm = PlotManager(out, labels=labels)
pm.info()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
pm.plot_scalars("uz_Z1", ax=axes[0])
pm.plot_scalars("sigma_zz_mean", ax=axes[1])
fig.tight_layout()
fig.savefig(f"{out}_scalars.png", dpi=200)

plt.show()