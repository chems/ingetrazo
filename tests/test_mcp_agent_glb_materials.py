# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
# Copyright (C) 2026 Maher (fork chems/ingetrazo, branch mcp-agent).
"""A GLB brings its material NAMES and render FINISHES into the document
(fork mcp-agent): models made in Blender (blender-objet) used to keep their
colours only, so the render showed brass and glass as matte paint.

Covered: names on faces and in the registry; the finish read from glTF's
physical parameters (and from IngeTrazo's own extras); a name already used
by another recipe; import_model over the bridge and its undo; the plain
File ▸ Import path (SnapshotImport) and its undo/redo; the finish reaching
the GLB the Blender render reads."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

if QApplication.instance() is None:
    QApplication(sys.argv[:1])

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from test_mcp_agent_tools import _ask, _ok, bridge, win  # noqa: E402,F401

#: (name, glTF material) — the four kinds of surface a Blender object has.
MATERIALS = [
    {"name": "Laiton", "pbrMetallicRoughness": {
        "baseColorFactor": [0.69, 0.55, 0.34, 1], "metallicFactor": 1.0,
        "roughnessFactor": 0.3}},
    {"name": "Chêne clair", "pbrMetallicRoughness": {
        "baseColorFactor": [0.78, 0.64, 0.45, 1], "metallicFactor": 0.0,
        "roughnessFactor": 0.5}},
    {"name": "Vitrage", "alphaMode": "BLEND", "pbrMetallicRoughness": {
        "baseColorFactor": [0.8, 0.9, 1.0, 0.25], "metallicFactor": 0.0,
        "roughnessFactor": 0.05}},
    {"name": "Chêne huilé", "pbrMetallicRoughness": {
        "baseColorFactor": [0.78, 0.64, 0.45, 1], "metallicFactor": 0.0,
        "roughnessFactor": 0.5500000119}},
    {"name": "Lin", "pbrMetallicRoughness": {
        "baseColorFactor": [0.93, 0.9, 0.84, 1], "metallicFactor": 0.0,
        "roughnessFactor": 0.9}},
    {"name": "Laque", "pbrMetallicRoughness": {
        "baseColorFactor": [0.95, 0.94, 0.92, 1], "metallicFactor": 0.0,
        "roughnessFactor": 0.15}},
    {"name": "verre trempé", "extras": {"ingetrazo_finish": "glass"},
     "pbrMetallicRoughness": {"baseColorFactor": [0.7, 0.8, 0.8, 1],
                              "metallicFactor": 0.0}},
    {"pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.2, 0.2, 1],
                              "metallicFactor": 0.0}},          # unnamed
]


def _glb(path: Path, materials=MATERIALS) -> Path:
    """One small upright square (two triangles) per material, side by side
    (Y up, as glTF)."""
    pos, idx, prims = [], [], []
    for i, _m in enumerate(materials):
        x = i * 0.3
        base = len(pos)
        pos += [(x, 0, 0), (x + 0.2, 0, 0), (x + 0.2, 0.5, 0), (x, 0.5, 0)]
        start = len(idx)
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
        prims.append((start, i))
    pbin = b"".join(struct.pack("<3f", *p) for p in pos)
    ibin = b"".join(struct.pack("<I", i) for i in idx)
    blob = pbin + ibin
    js = {
        "asset": {"version": "2.0", "generator": "Khronos glTF Blender I/O"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Objet"}],
        "materials": materials,
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pbin)},
            {"buffer": 0, "byteOffset": len(pbin), "byteLength": len(ibin)}],
        "accessors": [{"bufferView": 0, "componentType": 5126,
                       "count": len(pos), "type": "VEC3",
                       "min": [0, 0, 0],
                       "max": [max(p[0] for p in pos), 0.5, 0]}],
        "meshes": [{"primitives": []}],
    }
    for start, mi in prims:
        js["accessors"].append({"bufferView": 1, "byteOffset": start * 4,
                                "componentType": 5125, "count": 6,
                                "type": "SCALAR"})
        js["meshes"][0]["primitives"].append(
            {"attributes": {"POSITION": 0},
             "indices": len(js["accessors"]) - 1, "material": mi})
    jbin = json.dumps(js).encode()
    jbin += b" " * (-len(jbin) % 4)
    blob += b"\0" * (-len(blob) % 4)
    out = (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(jbin) + 8
                       + len(blob))
           + struct.pack("<II", len(jbin), 0x4E4F534A) + jbin
           + struct.pack("<II", len(blob), 0x004E4942) + blob)
    path.write_bytes(out)
    return path


def _faces_by_mat(group):
    from core.group import iter_placements
    out = {}
    for kid, _m in iter_placements(group):
        for f in kid.mesh.faces:
            out.setdefault((f.attrs or {}).get("mat"), []).append(f)
    return out


EXPECTED = {"Laiton": "metal", "Chêne clair": "satin", "Vitrage": "glass",
            "Chêne huilé": "satin", "Lin": "matte", "Laque": "gloss",
            "verre trempé": "glass"}


def test_load_glb_keeps_names_and_finishes(tmp_path):
    from core.scene import Scene
    from formats.glb import load_glb
    s = Scene()
    load_glb(s, _glb(tmp_path / "objet.glb"))
    faces = _faces_by_mat(s.groups[0])
    assert set(faces) == set(EXPECTED) | {None}        # unnamed: no name
    assert len(faces["Laiton"]) == 2
    assert {n: s.materials[n].finish for n in EXPECTED} == EXPECTED
    laiton = s.materials["Laiton"]
    assert laiton.color == (0.69, 0.55, 0.34)
    assert faces["Laiton"][0].attrs["color"] == pytest.approx([0.69, 0.55, 0.34])
    assert s.materials["Vitrage"].opacity == pytest.approx(0.25)
    assert faces["Vitrage"][0].attrs["opacity"] == pytest.approx(0.25)


def test_a_name_used_by_another_recipe_gets_its_own(tmp_path):
    from core.materials import Material
    from core.scene import Scene
    from formats.glb import load_glb
    s = Scene()
    s.materials["Laiton"] = Material("Laiton", (0.1, 0.1, 0.1), finish="matte")
    load_glb(s, _glb(tmp_path / "objet.glb"))
    faces = _faces_by_mat(s.groups[0])
    assert "Laiton (2)" in faces and "Laiton" not in faces
    assert s.materials["Laiton"].color == (0.1, 0.1, 0.1)   # untouched
    assert s.materials["Laiton (2)"].finish == "metal"       # finish kept


def test_the_same_file_twice_reuses_its_materials(tmp_path):
    from core.scene import Scene
    from formats.glb import load_glb
    s = Scene()
    p = _glb(tmp_path / "objet.glb")
    load_glb(s, p)
    load_glb(s, p)
    assert sorted(s.materials) == sorted(EXPECTED)


def test_import_model_brings_the_materials_and_undo_takes_them_away(
        win, bridge, tmp_path):
    vp = win.viewport
    res = _ok(_ask(bridge, "import_model", {
        "path": str(_glb(tmp_path / "lampe.glb")), "position": [1, 1, 0],
        "units": "m", "name": "Lampe Blender"}))
    assert sorted(res["materials_added"]) == sorted(EXPECTED)
    g = next(x for x in vp.scene.groups if x.name == "Lampe Blender")
    assert set(_faces_by_mat(g)) >= set(EXPECTED)
    assert vp.scene.materials["Laiton"].finish == "metal"
    assert vp.history.undo()
    assert not set(EXPECTED) & set(vp.scene.materials)
    assert vp.history.redo()
    assert vp.scene.materials["Vitrage"].finish == "glass"


def test_file_import_undo_and_redo_the_materials(tmp_path):
    from core.history import History, SnapshotImport
    from core.scene import Scene
    from formats.glb import load_glb
    s = Scene()
    h = History(s)
    p = _glb(tmp_path / "objet.glb")
    cmd = SnapshotImport(lambda scene: load_glb(scene, p))
    h.execute(cmd)
    assert cmd.changed and sorted(cmd.added_materials) == sorted(EXPECTED)
    assert set(EXPECTED) <= set(s.materials)
    h.undo()
    assert not s.materials and not s.groups
    h.redo()
    assert s.materials["Laque"].finish == "gloss" and len(s.groups) == 1


def test_a_failed_import_leaves_no_material_behind(tmp_path):
    from core.history import SnapshotImport
    from core.materials import Material
    from core.scene import Scene

    def broken(scene):
        scene.materials["Fantôme"] = Material("Fantôme", (1, 0, 0))
        raise RuntimeError("fichier abîmé")
    s = Scene()
    with pytest.raises(RuntimeError):
        SnapshotImport(broken).do(s)
    assert "Fantôme" not in s.materials


def test_the_render_glb_carries_the_finishes(tmp_path):
    """What Render with Blender reads: formats.gltf.save_glb writes
    extras.ingetrazo_finish from the registry, found by the face's name."""
    from core.scene import Scene
    from formats.glb import load_glb, parse_container
    from formats.gltf import save_glb
    s = Scene()
    load_glb(s, _glb(tmp_path / "objet.glb"))
    out = tmp_path / "rendu.glb"
    save_glb(s, out)
    js, _bufs = parse_container(out)
    got = {m.get("name"): (m.get("extras") or {}).get("ingetrazo_finish")
           for m in js["materials"]}
    for name, finish in EXPECTED.items():
        # The exporter writes « Chêne_clair » for « Chêne clair ».
        assert any(k and k.startswith(name.replace(" ", "_")) and v == finish
                   for k, v in got.items()), (name, got)
    # And back: IngeTrazo's own export re-imports with the same finishes.
    s2 = Scene()
    load_glb(s2, out)
    finishes = {m.finish for m in s2.materials.values()}
    assert {"metal", "glass", "gloss", "satin"} <= finishes


def test_three_tuple_readers_still_unpack_the_recipes(tmp_path):
    """scripts/glb_to_component.py unpacks (texture, rgba, opacity)."""
    from formats.glb import read_triangles
    batches, _js = read_triangles(_glb(tmp_path / "objet.glb"))
    for _tris, _uvs, recipe in batches:
        tex, rgba, opacity = recipe
        assert len(recipe) == 3
    assert [r.name for _t, _u, r in batches][:2] == ["Laiton", "Chêne clair"]
    assert batches[0][2].finish == "metal"
