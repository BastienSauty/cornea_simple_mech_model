"""Unit-cube test meshes (hexa and tetra) in Gmsh MSH 2.2 ASCII, matching cornea.msh conventions.

Outputs: cube_hexa.msh, cube_tetra.msh
Each has physical volume "cube" (tag 10) and physical surfaces for the six faces,
plus three $ElementData blocks (e1, e2, e3), 3 components, one value per volume cell.
"""
import sys
import gmsh

E = {"e1": (1.0, 0.0, 0.0), "e2": (0.0, 1.0, 0.0), "e3": (0.0, 0.0, 1.0)}
FACES = ["x0", "x1", "y0", "y1", "z0", "z1"]


def build(kind, fname, n=5, h=0.2):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.model.add(kind)
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, tag=1)
    gmsh.model.occ.synchronize()

    # Faces: identify by center of mass
    face_tag = {}
    for _, t in gmsh.model.getEntities(2):
        c = gmsh.model.occ.getCenterOfMass(2, t)
        for i, ax in enumerate("xyz"):
            for v in (0, 1):
                if abs(c[i] - v) < 1e-9:
                    face_tag[f"{ax}{v}"] = t
    for i, name in enumerate(FACES):
        gmsh.model.addPhysicalGroup(2, [face_tag[name]], 1 + i)
        gmsh.model.setPhysicalName(2, 1 + i, name)
    gmsh.model.addPhysicalGroup(3, [1], 10)
    gmsh.model.setPhysicalName(3, 10, "cube")

    if kind == "hexa":
        for _, t in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(t, n + 1)
        for _, t in gmsh.model.getEntities(2):
            gmsh.model.mesh.setTransfiniteSurface(t)
            gmsh.model.mesh.setRecombine(2, t)
        gmsh.model.mesh.setTransfiniteVolume(1)
        gmsh.model.mesh.setRecombine(3, 1)
    else:
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)

    gmsh.model.mesh.generate(3)
    types, tags, _ = gmsh.model.mesh.getElements(3)
    cell_tags = sorted(int(t) for tt in tags for t in tt)
    ctype = {5: "hexahedron", 4: "tetrahedron"}[int(types[0])]
    gmsh.write(fname)
    gmsh.finalize()

    with open(fname, "a") as f:
        for name, v in E.items():
            f.write(f'$ElementData\n1\n"{name}"\n1\n0.0\n3\n0\n3\n{len(cell_tags)}\n')
            for t in cell_tags:
                f.write(f"{t} {v[0]} {v[1]} {v[2]}\n")
            f.write("$EndElementData\n")
    print(f"{fname}: {len(cell_tags)} {ctype} cells")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    build("hexa", f"{out}/cube_hexa.msh")
    build("tetra", f"{out}/cube_tetra.msh")
