# author : B. Sauty ; 28 Sept 2026 ; Modified on 1 Oct
"""
Mesh input/output: reads a .msh file into a dolfinx mesh, its physical tags
and, if present, its Local Reference System (LRS): three per-cell orthonormal
unit vector fields (e_1, e_2, e_3), stored as DG0 functions with 3 components.

One reader for both cases (axisymmetric and 3D), the dimension being inferred from the file:
  - 2D (r, z) section, axisymmetric case: e_1, e_2 are read from the file
    (out-of-plane component 0); e_3 is the hoop direction (0, 0, 1) everywhere.
  - 3D (x, y, z) mesh: e_1, e_2 are read from the file; e_3 is read from the
    file if present, otherwise computed as e_1 x e_2.

Element data names accepted in the file: "e_1" or "e1", "e_2" or "e2",
"e_3" or "e3" (3D only).

Independent of any physics framework: every framework needs the same mesh,
tags and LRS, so this is called once per framework instance.
"""
from dataclasses import dataclass
from typing import Optional

import gmsh
import numpy as np
from dolfinx import fem
from dolfinx.io import gmsh as gmshio

import itertools
from pathlib import Path

LRS_NAMES = {
    "e_1": ("e_1", "e1"),
    "e_2": ("e_2", "e2"),
    "e_3": ("e_3", "e3"),
}


@dataclass
class MeshData:
    domain: object       # dolfinx.mesh.Mesh
    facet_tag: object    # dolfinx.mesh.MeshTags
    cell_tags: object    # dolfinx.mesh.MeshTags
    e_1: Optional[object] = None   # fem.Function, DG0, 3 components: (r, z, theta) in 2D, (x, y, z) in 3D
    e_2: Optional[object] = None   # same
    e_3: Optional[object] = None   # same; (0, 0, 1) in 2D

    @property
    def has_lrs(self):
        return self.e_1 is not None


def read_mesh(mesh_file, comm, rank=0) -> MeshData:
    """
    Open mesh_file once with gmsh (on `rank`) and build the dolfinx mesh, its
    physical tags and, if the file carries ElementData for e_1 and e_2, the
    per-cell LRS. The dimension of the mesh (2 or 3) is taken from the file.

    A mesh without LRS is not an error here: e_1/e_2/e_3 are left as None
    (MeshData.has_lrs is False). Code that needs fibre directions should raise
    on that itself, so that an isotropic run can use an LRS-less mesh.
    """
    dim = None
    if comm.rank == rank:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(mesh_file))
        dim = gmsh.model.getDimension()
    dim = comm.bcast(dim, root=rank)

    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=dim)

    payload = None
    if comm.rank == rank:
        payload = (_read_element_data(), _top_dim_element_tags(dim))
        gmsh.finalize()
    element_data, top_tags = comm.bcast(payload, root=rank)

    domain, facet_tag, cell_tags = mesh_data.mesh, mesh_data.facet_tags, mesh_data.cell_tags

    e_1 = e_2 = e_3 = None
    if _find_name(element_data, "e_1") and _find_name(element_data, "e_2"):
        e_1, e_2, e_3 = _build_lrs(domain, element_data, top_tags)

    return MeshData(domain, facet_tag, cell_tags, e_1, e_2, e_3)


def _read_element_data():
    """
    {view name: (gmsh element tags, values (n, n_components))} for every
    ElementData view of the gmsh model. Only the elements listed in the view
    are stored, so views covering all elements (2D case, boundary lines
    included) and views covering the cells only (3D case) are both handled.
    """
    element_data = {}
    for tag in gmsh.view.getTags():
        name = gmsh.option.getString(f"View[{gmsh.view.getIndex(tag)}].Name")
        data_type, elem_tags, values, _, num_components = gmsh.view.getModelData(tag, 0)
        if data_type != "ElementData":
            continue                                    # e.g. an old-style NodeData view; ignore
        elem_tags = np.asarray(elem_tags, dtype=np.int64)
        values = np.asarray(values, dtype=float).reshape(-1, num_components)
        element_data[name] = (elem_tags, values)
    return element_data


def _top_dim_element_tags(dim):
    """
    gmsh tags of the top-dimension elements, in the order in which dolfinx
    receives them (physical groups of dimension `dim`, then their entities).
    domain.topology.original_cell_index indexes into this array.
    For a mesh with a single volume (or surface) physical group this is the
    ascending tag order.
    """
    tags = []
    for _, pg in gmsh.model.getPhysicalGroups(dim):
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, pg):
            _, elem_tags, _ = gmsh.model.mesh.getElements(dim, ent)
            for t in elem_tags:
                tags.extend(t)
    return np.asarray(tags, dtype=np.int64)


def _find_name(element_data, key):
    """Name under which the LRS vector `key` is stored in the file, or None."""
    for name in LRS_NAMES[key]:
        if name in element_data:
            return name
    return None

def _lookup(element_data, name, cell_gmsh_tags):
    """Values of view `name` for the given gmsh element tags, shape (n, 3)."""
    tags, values = element_data[name]
    if values.shape[1] != 3:
        raise ValueError(f"ElementData '{name}' has {values.shape[1]} components, expected 3")
    order = np.argsort(tags)
    tags, values = tags[order], values[order]
    idx = np.minimum(np.searchsorted(tags, cell_gmsh_tags), len(tags) - 1)
    if not np.array_equal(tags[idx], cell_gmsh_tags):
        raise ValueError(f"ElementData '{name}' does not cover every cell of the mesh")
    return values[idx]


def _build_lrs(domain, element_data, top_tags):
    """
One reader for both cases, the dimension being inferred from the file:

    Build the three DG0 vector Functions (e_1, e_2, e_3), 3 components each,
    one value per cell of `domain`.

    gmsh numbers ElementData by element tag over the whole file (boundary
    elements first, cells last). domain.topology.original_cell_index gives for
    each local cell its position among the top-dimension elements as passed to
    dolfinx; top_tags maps that position to the gmsh tag, which is then looked
    up in the views. No assumption is made on the number of lower-dimension
    elements, nor on whether a view covers all elements or only the cells.

    e_1, e_2: read from the file.
    e_3: (0, 0, 1) if the mesh is 2D; in 3D read from the file if present,
         otherwise e_1 x e_2.

    NOTE: parallel behaviour (ghost cells) is not verified. The fields are
    filled on owned cells and ghost values are obtained by scatter_forward.
    This assumes original_cell_index and V.dofmap.list are indexed identically
    over the owned cells, which holds on a single rank.
    """
    tdim = domain.topology.dim
    n_owned = domain.topology.index_map(tdim).size_local

    original = np.asarray(domain.topology.original_cell_index)[:n_owned]
    cell_gmsh_tags = top_tags[original]

    V = fem.functionspace(domain, ("DG", 0, (3,)))
    e_1, e_2, e_3 = fem.Function(V), fem.Function(V), fem.Function(V)
    dofs = V.dofmap.list[:n_owned].reshape(-1)                  # 1 dof block per cell (DG0)

    v_1 = _lookup(element_data, _find_name(element_data, "e_1"), cell_gmsh_tags)
    v_2 = _lookup(element_data, _find_name(element_data, "e_2"), cell_gmsh_tags)

    if tdim == 2: # axisymmetric case : the e_3 is given as the out of plane direction
        v_3 = np.tile([0.0, 0.0, 1.0], (n_owned, 1))
    else: # 3D case : e_3 is given directly within the mesh
        name_3 = _find_name(element_data, "e_3")
        v_3 = _lookup(element_data, name_3, cell_gmsh_tags) if name_3 else np.cross(v_1, v_2)

    for f, vals in ((e_1, v_1), (e_2, v_2), (e_3, v_3)):
        f.x.array.reshape(-1, 3)[dofs] = vals
        f.x.scatter_forward()

    return e_1, e_2, e_3

def write_mesh(mesh_file, e_1=None, e_2=None, e_3=None,
               msh_version=4.1, binary=False, tol=1e-8):
    """
    Write the CURRENT GMSH MODEL to `mesh_file` (.msh), with an optional LRS.
    Serial use only.

    Input: a gmsh model, already meshed (gmsh.model.mesh.generate), with at
    least one physical group of the top dimension. Triangles, quads, tetrahedra
    and hexahedra are supported. Only elements belonging to physical groups are
    written (gmsh default). gmsh.initialize()/finalize() are left to the caller.

    LRS (optional): e_1, e_2 (and e_3, 3D only) are arrays of shape (n_cells, 3),
    ordered like the top-dimension elements of the model, i.e. as in
    _top_dim_element_tags(dim). They are stored as ElementData views named
    "e_1", "e_2", "e_3", which read_mesh accepts.
      - e_1 and e_2 must be given together.
      - 2D: e_3 must not be given (read_mesh sets it to (0, 0, 1)); the
        out-of-plane component of e_1 and e_2 must be zero.
      - 3D: if e_3 is omitted, read_mesh computes e_1 x e_2.
    The vectors must be unit and mutually orthogonal within `tol`.
    """
    mesh_file = Path(mesh_file)
    if mesh_file.suffix != ".msh":
        raise ValueError(f"mesh_file must have a .msh extension, got '{mesh_file.name}'")

    dim = gmsh.model.getDimension()
    if dim not in (2, 3):
        raise ValueError(f"model dimension is {dim}, expected 2 or 3")
    top_tags = _top_dim_element_tags(dim)
    if top_tags.size == 0:
        raise ValueError(f"no mesh elements in a physical group of dimension {dim}")

    vectors = _check_lrs(dim, len(top_tags), e_1, e_2, e_3, tol)

    if vectors and binary:
        raise ValueError("an LRS can only be appended to an ASCII file (binary=False)")

    gmsh.option.setNumber("Mesh.MshFileVersion", msh_version)
    gmsh.option.setNumber("Mesh.Binary", int(binary))
    gmsh.write(str(mesh_file))

    # ElementData: 1 string tag (name), 1 real tag (time), 3 integer tags
    # (time step, number of components, number of elements).
    with open(mesh_file, "a") as f:
        for name, values in vectors.items():
            f.write(f'$ElementData\n1\n"{name}"\n1\n0.0\n3\n0\n3\n{len(top_tags)}\n')
            for tag, v in zip(top_tags, values):
                f.write(f"{tag} {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n")
            f.write("$EndElementData\n")


def _check_lrs(dim, n_cells, e_1, e_2, e_3, tol):
    """Validate the LRS arrays; return {name: (n_cells, 3) array} (empty if no LRS)."""
    if e_1 is None and e_2 is None and e_3 is None:
        return {}
    if e_1 is None or e_2 is None:
        raise ValueError("e_1 and e_2 must be given together")
    if dim == 2 and e_3 is not None:
        raise ValueError("e_3 is not accepted in 2D: read_mesh sets it to (0, 0, 1)")

    given = {"e_1": e_1, "e_2": e_2}
    if e_3 is not None:
        given["e_3"] = e_3

    vectors = {}
    for name, v in given.items():
        v = np.asarray(v, dtype=float)
        if v.shape != (n_cells, 3):
            raise ValueError(f"{name} has shape {v.shape}, expected {(n_cells, 3)}")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{name} contains non-finite values")
        err = np.abs(np.linalg.norm(v, axis=1) - 1.0).max()
        if err > tol:
            raise ValueError(f"{name} is not unit norm (max deviation {err:.2e})")
        if dim == 2 and np.abs(v[:, 2]).max() > tol:
            raise ValueError(f"{name} has a non-zero out-of-plane component in 2D")
        vectors[name] = v

    for a, b in itertools.combinations(vectors, 2):
        err = np.abs(np.einsum("ij,ij->i", vectors[a], vectors[b])).max()
        if err > tol:
            raise ValueError(f"{a} and {b} are not orthogonal (max |dot| {err:.2e})")
    return vectors

def top_cell_centroids(dim):
    """Centroids of the top-dimension cells, in the order of _top_dim_element_tags."""
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.zeros((int(node_tags.max()) + 1, 3))
    xyz[node_tags.astype(int)] = coords.reshape(-1, 3)
    out = []
    for _, pg in gmsh.model.getPhysicalGroups(dim):
        for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, pg):
            types, _, conn = gmsh.model.mesh.getElements(dim, ent)
            for t, nodes in zip(types, conn):
                n = gmsh.model.mesh.getElementProperties(t)[3]
                out.append(xyz[nodes.reshape(-1, n).astype(int)].mean(axis=1))
    return np.vstack(out)