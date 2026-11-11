"""Blender: a generated model (Tripo FBX, or GLB/OBJ) to a clean GLB that
faces forward, ready to rig.

    blender -b --factory-startup -P tools/tripo_to_glb.py -- IN OUT.glb
        [--kind auto|biped|quadruped] [--convention +x|-x|+y|-y|none]
        [--tris N] [--height M] [--texture PIXELS] [--holes SHARE] [--report R.json]

One mesh, transforms applied, standing on the ground at the origin, facing
glTF +Z (Blender -Y) with +Y up. Which way the model faces is measured
(`facing.py`), with the source's convention as the fallback: Tripo's
exports face +X. `--tris` cuts the mesh to a triangle budget, keeping its
UVs and texture; `--height` scales it to stand that tall (m); textures
leave as PNGs no larger than `--texture` a side (1024). Holes whose rim is
no longer than `--holes` of the model's largest side (0.08; 0 only counts
them) are filled; the report says how many it found, filled and left. The last
line printed is `TRIPO_TO_GLB {json}`; the same JSON goes to `--report`.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facing  # noqa: E402


def options():
    a = sys.argv[sys.argv.index("--") + 1:]
    o = {"kind": "auto", "tris": "0", "height": "0", "report": None, "convention": "+x", "texture": "1024", "holes": "0.08"}
    pos = []
    i = 0
    while i < len(a):
        if a[i].startswith("--"):
            o[a[i][2:]] = a[i + 1]
            i += 2
        else:
            pos.append(a[i])
            i += 1
    if len(pos) != 2:
        raise SystemExit(__doc__)
    o["in"], o["out"] = pos
    o["tris"] = int(o["tris"])
    o["height"] = float(o["height"])
    o["texture"] = int(o["texture"])
    o["holes"] = float(o["holes"])
    return o


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == "obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise SystemExit(f"unknown mesh type: {path}")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no mesh in " + path)
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
        # Out from under any armature or empty, keeping its place; an old
        # rig's weights and modifiers go.
        m = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = m
        for mod in list(o.modifiers):
            o.modifiers.remove(mod)
        o.vertex_groups.clear()
        if o.data.shape_keys:
            o.shape_key_clear()
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in list(bpy.context.scene.objects):
        if o != obj:
            bpy.data.objects.remove(o)
    return obj


def png_textures(limit):
    """Every texture as a PNG no larger than `limit` a side: Tripo's are
    JPEGs, which the glTF exporter passes through and the game's Bevy build
    doesn't decode (rfcheck B_IMAGE_CODEC). Returns their sizes."""
    import tempfile
    out = []
    folder = tempfile.mkdtemp(prefix="tripo_to_glb_")
    for i, img in enumerate(bpy.data.images):
        if img.type != "IMAGE" or not img.has_data or img.size[0] == 0:
            continue
        w, h = img.size
        k = max(w, h) / float(limit) if limit > 0 else 1.0
        if k > 1.0:
            img.scale(max(1, int(round(w / k))), max(1, int(round(h / k))))
        path = os.path.join(folder, f"texture_{i}.png")
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        if img.packed_file:
            img.unpack(method="REMOVE")
        img.source = "FILE"
        img.reload()
        out.append([img.size[0], img.size[1]])
    return out


def boundary_loops(bm):
    """The mesh's holes: each a list of edges with a face on one side
    only, joined end to end."""
    open_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    at = {}
    for e in open_edges:
        for v in e.verts:
            at.setdefault(v.index, []).append(e)
    seen, loops = set(), []
    for e in open_edges:
        if e.index in seen:
            continue
        seen.add(e.index)
        stack, loop = [e], []
        while stack:
            a = stack.pop()
            loop.append(a)
            for v in a.verts:
                for b in at[v.index]:
                    if b.index not in seen:
                        seen.add(b.index)
                        stack.append(b)
        loops.append(loop)
    return loops


def mend_holes(obj, limit):
    """Generated meshes come with holes: gaps a few edges round where the
    surface never closed, which show as see-through cracks in the game.
    Each hole whose rim is no longer than `limit` (a share of the model's
    largest side) is filled, and the new faces take the colour beside
    them; a longer rim is taken for an opening that is meant (a hood, a
    sleeve, a hollow cloak) and left, and said. Returns the counts."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    size = max(obj.dimensions) or 1.0
    loops = boundary_loops(bm)
    found = len(loops)
    open_edges = sum(len(lp) for lp in loops)
    small = [lp for lp in loops if sum(e.calc_length() for e in lp) <= limit * size]
    filled = 0
    if small and limit > 0:
        uv = bm.loops.layers.uv.active
        old = set(bm.faces)
        bmesh.ops.holes_fill(bm, edges=[e for lp in small for e in lp], sides=0)
        new = [f for f in bm.faces if f not in old]
        for f in new:
            f.smooth = True
            # The colour beside it: each corner takes the UV its vertex has
            # on a face that was already there.
            if uv:
                for corner in f.loops:
                    for other in corner.vert.link_loops:
                        if other.face in old:
                            corner[uv].uv = other[uv].uv
                            break
            near = next((e.link_faces[0] for e in f.edges if e.link_faces and e.link_faces[0] in old), None)
            if near:
                f.material_index = near.material_index
        bmesh.ops.recalc_face_normals(bm, faces=new)
    left = boundary_loops(bm)
    # What closed, counted after the fact: a rim that touches itself or
    # another rim doesn't always fill.
    filled = max(0, found - len(left))
    out = {
        "found": found,
        "open_edges": open_edges,
        "filled": filled,
        "left": len(left),
        "largest_left": round(max((sum(e.calc_length() for e in lp) for lp in left), default=0.0) / size, 3),
        "edges_with_three_faces": sum(1 for e in bm.edges if len(e.link_faces) > 2),
    }
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return out


def tris(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def main():
    o = options()
    obj = load(o["in"])
    holes = mend_holes(obj, o["holes"])
    source_tris = tris(obj)
    m = facing.measure([tuple(v.co) for v in obj.data.vertices], kind=o["kind"])
    fwd, decided_by, off = facing.decide(m, o["convention"])
    obj.matrix_world = Matrix.Rotation(facing.turn_to_minus_y(fwd), 4, "Z") @ obj.matrix_world
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if o["tris"] and source_tris > o["tris"]:
        mod = obj.modifiers.new("budget", "DECIMATE")
        mod.ratio = o["tris"] / source_tris
        mod.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
    vs = obj.data.vertices
    lo = Vector((min(v.co.x for v in vs), min(v.co.y for v in vs), min(v.co.z for v in vs)))
    hi = Vector((max(v.co.x for v in vs), max(v.co.y for v in vs), max(v.co.z for v in vs)))
    scale = o["height"] / (hi.z - lo.z) if o["height"] > 0 else 1.0
    shift = Vector((-(lo.x + hi.x) / 2, -(lo.y + hi.y) / 2, -lo.z))
    obj.matrix_world = Matrix.Scale(scale, 4) @ Matrix.Translation(shift)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.name = obj.data.name = "Body"
    textures = png_textures(o["texture"])
    info = {
        "kind": m["kind"],
        "forward": [round(fwd[0], 3), round(fwd[1], 3)],
        "measured_forward": [round(m["forward"][0], 3), round(m["forward"][1], 3)],
        "decided_by": decided_by,
        "off_convention_deg": None if off is None else round(off, 1),
        "how": m["how"],
        "confidence": round(m["confidence"], 4),
        "mirror_contrast": round(m["mirror_contrast"], 2),
        "source_size": [round(m["width"], 4), round(m["length"], 4), round(m["height"], 4)],
        "source_tris": source_tris,
        "tris": tris(obj),
        "verts": len(obj.data.vertices),
        "scale": round(scale, 5),
        "height": round((hi.z - lo.z) * scale, 4),
        "textures": textures,
        "holes": holes,
    }
    bpy.ops.export_scene.gltf(filepath=o["out"], export_format="GLB", use_selection=False,
                              export_apply=True, export_yup=True, export_animations=False,
                              export_skins=False, export_morph=False)
    if o["report"]:
        with open(o["report"], "w") as f:
            json.dump(info, f, indent=1)
    print("TRIPO_TO_GLB " + json.dumps(info))


main()
