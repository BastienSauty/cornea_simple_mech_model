# author : B. Sauty ; 28 Sept 2026
"""
Read an axisymmetric (r, z) section .msh file into a dolfinx mesh, its
physical tags, and -- if present -- its local reference system (LRS): three
per-element orthonormal unit vector fields (e_circ, e_normal, e_theta),
matching the circ/normal element data written by
cornea_meshing_package.mesh_io.write_msh (circ, normal: 3-component element
data, one vector per quad, out-of-plane component 0; e_theta is the
hoop/out-of-plane direction (0, 0, 1), the same everywhere).

Deliberately independent of any particular physics framework (hyperelastic,
poroelastic, ...): every single-material axisymmetric framework needs the
same mesh, tags, and LRS, so this lives on its own and is called once per
framework instance.
"""
from dataclasses import dataclass
from typing import Optional

import gmsh
import numpy as np
from dolfinx import fem
from dolfinx.io import gmsh as gmshio


@dataclass
class MeshData:
    domain: object       # dolfinx.mesh.Mesh
    facet_tag: object    # dolfinx.mesh.MeshTags
    cell_tags: object    # dolfinx.mesh.MeshTags
    e_circ: Optional[object] = None     # fem.Function, DG0, 3 components (r, z, theta)
    e_normal: Optional[object] = None   # fem.Function, DG0, 3 components (r, z, theta)
    e_theta: Optional[object] = None    # fem.Function, DG0, 3 components (r, z, theta) = (0, 0, 1)

    @property
    def has_lrs(self):
        return self.e_circ is not None


def read_axisymmetric_mesh(mesh_file, comm, rank=0) -> MeshData:
    """
    Open mesh_file once with gmsh (on `rank`), extract the (r, z) mesh, its
    physical tags, and, if the file carries $ElementData "circ" and
    "normal" (as written by cornea_meshing_package.mesh_io.write_msh), the
    per-cell local reference system as three DG0 vector fields.

    A mesh without an LRS is not an error here -- e_circ/e_normal/e_theta
    are simply left as None (MeshData.has_lrs is False); anything that
    actually needs fibre directions should raise on that itself (see
    Hyperelastic_framework._fibre_orientation_field for an example), not
    this function, so a purely isotropic run can use an LRS-less mesh.
    """
    element_data = None
    if comm.rank == rank:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(mesh_file))
        mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)

        element_data = {}
        for tag in gmsh.view.getTags():
            name = gmsh.option.getString(f"View[{gmsh.view.getIndex(tag)}].Name")
            data_type, elem_tags, values, _, num_components = gmsh.view.getModelData(tag, 0)
            if data_type != "ElementData":
                continue                                    # e.g. an old-style NodeData view; ignore
            elem_tags = np.asarray(elem_tags, dtype=np.int64)
            values = np.asarray(values).reshape(-1, num_components)
            table = np.full((int(elem_tags.max()), num_components), np.nan)
            table[elem_tags - 1] = values                    # indexed by (gmsh element tag - 1)
            element_data[name] = table
        gmsh.finalize()
    else:
        mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)
    element_data = comm.bcast(element_data, root=rank)

    domain, facet_tag, cell_tags = mesh_data.mesh, mesh_data.facet_tags, mesh_data.cell_tags

    e_circ = e_normal = e_theta = None
    if "circ" in element_data and "normal" in element_data:
        e_circ, e_normal, e_theta = _build_lrs(domain, element_data)

    return MeshData(domain, facet_tag, cell_tags, e_circ, e_normal, e_theta)


def _build_lrs(domain, element_data):
    """
    Turn the per-gmsh-element-tag "circ"/"normal" tables into three DG0
    vector Functions (e_circ, e_normal, e_theta), 3 components (r, z,
    theta) each, one value per cell of `domain`, in the mesh's own local
    cell order.

    gmsh numbers ElementData by the element's tag over the WHOLE file --
    the boundary lines, written first by write_msh, then the quads -- not
    by position among cells of the mesh's own top dimension. dolfinx keeps
    only the top-dimension cells (the quads) in `domain`, and
    domain.topology.original_cell_index gives, for each local cell, its
    0-based position among just those top-dimension elements as read from
    the file -- i.e. exactly `n_lines` lower than its gmsh tag. n_lines is
    recovered from the element-data table itself (which covers every
    element in the file) minus the mesh's own global quad count, rather
    than assumed from Nx/Ny, so this doesn't depend on how the mesh was
    built.

    NOTE: this is the one part of the port that most needs checking against
    a real dolfinx run, in particular in parallel (ghost cells): it assumes
    domain.topology.original_cell_index and V.dofmap.list are both indexed
    over exactly the same (local, ghosts included) cell range, in the same
    order -- true for a single rank, and expected to hold in general, but
    not exercised here since dolfinx isn't available in this environment.
    """
    n_quads_global = domain.topology.index_map(domain.topology.dim).size_global
    n_total_elements = len(element_data["circ"])              # lines + quads, from write_msh
    n_lines = n_total_elements - n_quads_global

    original = domain.topology.original_cell_index             # 0-based position among quads only
    gmsh_tags0 = original + n_lines                             # -> matches table's (tag - 1) index

    V = fem.functionspace(domain, ("DG", 0, (3,)))
    e_circ, e_normal, e_theta = fem.Function(V), fem.Function(V), fem.Function(V)

    dofmap = V.dofmap.list.reshape(-1)                          # 1 dof-block per local cell (DG0)
    assert len(dofmap) == len(gmsh_tags0), (
        f"{len(dofmap)} local DG0 dofs but {len(gmsh_tags0)} cells in original_cell_index"
    )

    for name, f in (("circ", e_circ), ("normal", e_normal)):
        vals = element_data[name][gmsh_tags0]                   # (n_local_cells, 3), (r, z, 0)
        f.x.array.reshape(-1, 3)[dofmap] = vals
        f.x.scatter_forward()

    e_theta.x.array.reshape(-1, 3)[dofmap] = np.array([0.0, 0.0, 1.0])
    e_theta.x.scatter_forward()

    return e_circ, e_normal, e_theta
