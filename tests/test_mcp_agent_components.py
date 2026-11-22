# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
# Copyright (C) 2026 Maher (fork chems/ingetrazo, branch mcp-agent).
"""Lot 2 of the MCP tools: real-size library components, 3D model files as
one component, and the render finish of a named material — driven over the
bridge as an agent drives them, against a library published on disk (no
network)."""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtGui import QVector3D  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

if QApplication.instance() is None:
    QApplication(sys.argv[:1])

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from test_mcp_agent_tools import _ask, _box_group, _ok, _png  # noqa: E402

V = QVector3D


def _box_obj(w, h, d) -> str:
    """An OBJ box in centimetres, Y up (as Sweet Home 3D writes them)."""
    xs, ys, zs = (0, w), (0, h), (0, d)
    verts = [(x, y, z) for x in xs for y in ys for z in zs]
    faces = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7),
             (1, 3, 7, 5), (2, 6, 8, 4)]
    return ("mtllib m.mtl\nusemtl bois\n"
            + "".join("v %g %g %g\n" % v for v in verts)
            + "".join("f %d %d %d %d\n" % f for f in faces))


@pytest.fixture
def library_on_disk(tmp_path, monkeypatch):
    src = tmp_path / "sitio"
    (src / "modelos").mkdir(parents=True)
    models = [
        ("cama-doble", "Cama", "Bed", "Dormitorio", (160, 45, 200)),
        ("cama-simple", "Cama", "Bed", "Dormitorio", (90, 45, 190)),
        ("radiador", "Radiador", "Radiator", "Varios", (80, 60, 10)),
    ]
    entries = []
    for ident, nombre, en, cat, (w, h, d) in models:
        entries.append({"id": ident, "nombre": nombre, "nombre_en": en,
                        "categoria": cat, "obj": "m.obj",
                        "cm": [str(w), str(d), str(h)],
                        "licencia": "CC-BY-4.0",
                        "licencia_nombre": "Creative Commons BY",
                        "autor": "Autor de prueba"})
        with zipfile.ZipFile(src / "modelos" / f"{ident}.zip", "w") as z:
            z.writestr("m.obj", _box_obj(w, h, d))
            z.writestr("m.mtl", "newmtl bois\nKd 0.6 0.4 0.2\n")
    (src / "index.json").write_text(json.dumps(
        {"version": 1, "unidad": "cm", "modelos": entries}), encoding="utf-8")
    monkeypatch.setenv("INGETRAZO_LIBRARY", str(src))
    monkeypatch.setenv("INGETRAZO_TEXTURE_CACHE",
                       str(tmp_path / "cache" / "tex"))
    return src


@pytest.fixture
def win(monkeypatch):
    monkeypatch.setenv("INGETRAZO_AI_PORT", "0")
    from views.main_window import MainWindow
    w = MainWindow()
    yield w
    w._saved_version = w.viewport.scene.version
    b = getattr(w, "_ai_bridge", None)
    if b is not None:
        b.stop()
    w.close()


@pytest.fixture
def bridge(win):
    from plugins.ai_bridge import _bridge_of
    b = _bridge_of(win.viewport)
    b.start()
    yield b
    b.stop()


# ---- search_components ---------------------------------------------------------

def test_search_finds_spanish_english_and_ranks_by_size(win, bridge,
                                                       library_on_disk):
    res = _ok(_ask(bridge, "search_components", {"query": "cama"}))
    assert res["total"] == 2
    assert {r["id"] for r in res["results"]} == {"cama-doble", "cama-simple"}
    res = _ok(_ask(bridge, "search_components",
                   {"query": "bed", "size_cm": [160, 200, None]}))
    assert res["results"][0]["id"] == "cama-doble"
    assert res["results"][0]["size_cm"] == [160.0, 200.0, 45.0]
    assert res["results"][0]["licence"] == "Creative Commons BY"
    res = _ok(_ask(bridge, "search_components", {"query": "RADIADOR"}))
    assert [r["name_en"] for r in res["results"]] == ["Radiator"]
    res = _ok(_ask(bridge, "search_components", {"category": "dormitorio"}))
    assert res["total"] == 2 and "Dormitorio" in res["categories"]
    res = _ok(_ask(bridge, "search_components", {"query": "sofa"}))
    assert res["results"][0]["id"] == "bundled:sofa"


# ---- insert_component ------------------------------------------------------------

def test_insert_component_at_real_size_base_on_the_point(win, bridge,
                                                        library_on_disk):
    vp = win.viewport
    r = _ask(bridge, "insert_component", {"query": "cama"})
    assert not r["ok"] and "cama-doble" in r["error"] and "size_cm" in r["error"]
    depth = len(vp.history.undo_stack)
    res = _ok(_ask(bridge, "insert_component", {
        "query": "Bed", "size_cm": [160, 200, None],
        "position": [2.5, 2.0, 0.0], "name": "Lit chambre 1",
        "layer": "A-MOBIL", "ifc": "IfcFurniture"}))
    assert res["id"] == "cama-doble" and res["name"] == "Lit chambre 1"
    assert res["size_m"] == pytest.approx([1.6, 2.0, 0.45], abs=1e-3)
    lo, hi = res["bounds"]["min"], res["bounds"]["max"]
    assert (lo[0] + hi[0]) / 2 == pytest.approx(2.5, abs=1e-4)
    assert (lo[1] + hi[1]) / 2 == pytest.approx(2.0, abs=1e-4)
    assert lo[2] == pytest.approx(0.0, abs=1e-4)
    assert res["licence"] == "Creative Commons BY"
    assert res["author"] == "Autor de prueba"
    assert res["layer"] == "A-MOBIL" and res["ifc"] == "IfcFurniture"
    g = next(x for x in vp.scene.groups if x.name == "Lit chambre 1")
    assert g.xform is not None and g.is_component()
    assert len(vp.history.undo_stack) == depth + 1      # one step
    assert _ok(_ask(bridge, "undo"))["ok"]
    assert g not in vp.scene.groups
    assert _ok(_ask(bridge, "redo"))["ok"]
    assert g in vp.scene.groups


def test_insert_component_turns_and_shares_the_definition(win, bridge,
                                                         library_on_disk):
    a = _ok(_ask(bridge, "insert_component", {"id": "cama-simple",
                                              "position": [0, 0, 0]}))
    b = _ok(_ask(bridge, "insert_component", {"id": "cama-simple",
                                              "position": [4, 0, 0],
                                              "rotation": 90}))
    assert a["shared_definition"] is False and b["shared_definition"] is True
    assert a["size_m"] == pytest.approx([0.9, 1.9, 0.45], abs=1e-3)
    assert b["size_m"] == pytest.approx([1.9, 0.9, 0.45], abs=1e-3)
    beds = [g for g in win.viewport.scene.groups if g.name == "Cama"]
    assert len(beds) == 2 and beds[0].mesh is beds[1].mesh


def test_insert_a_bundled_starter_offline(win, bridge, monkeypatch):
    monkeypatch.setenv("INGETRAZO_LIBRARY", "/nowhere")
    res = _ok(_ask(bridge, "insert_component", {"id": "bundled:banco",
                                                "position": [1, 1, 0]}))
    assert res["faces"] > 10 and res["bounds"]["min"][2] == \
        pytest.approx(0.0, abs=1e-3)
    r = _ask(bridge, "insert_component", {"id": "no-such-model"})
    assert not r["ok"]


# ---- import_model ----------------------------------------------------------------

def _model_files(tmp_path):
    """A textured box exported by IngeTrazo itself as GLB, OBJ and IGZ."""
    from core.scene import Scene
    from formats.gltf import save_glb
    from formats.igz import save_scene
    from formats.obj import save_obj
    s = Scene()
    g = _box_group(s, "Caisson", origin=(10, 10, 0))
    img = _png(tmp_path / "bois.png")
    for f in g.mesh.faces:
        f.attrs["texture"] = {"path": str(img), "sw": 0.5, "sh": 0.5}
    save_glb(s, tmp_path / "caisson.glb")
    save_obj(s, tmp_path / "caisson.obj")
    save_scene(s, tmp_path / "caisson.igz")
    return tmp_path


def test_import_a_glb_where_asked_in_one_step(win, bridge, tmp_path):
    files = _model_files(tmp_path)
    vp = win.viewport
    depth = len(vp.history.undo_stack)
    res = _ok(_ask(bridge, "import_model", {
        "path": str(files / "caisson.glb"), "position": [3, -1, 0],
        "rotation": 30, "name": "Caisson importé", "ifc": "IfcFurniture"}))
    assert res["format"] == "glb" and res["faces"] >= 6
    assert res["name"] == "Caisson importé"
    lo, hi = res["bounds"]["min"], res["bounds"]["max"]
    assert (lo[0] + hi[0]) / 2 == pytest.approx(3.0, abs=1e-3)
    assert lo[2] == pytest.approx(0.0, abs=1e-3)
    g = next(x for x in vp.scene.groups if x.name == "Caisson importé")
    assert any(f.attrs.get("texture") for kid, _m in
               __import__("core.group", fromlist=["x"]).iter_placements(g)
               for f in kid.mesh.faces)
    assert len(vp.history.undo_stack) == depth + 1
    assert vp.history.undo() and g not in vp.scene.groups


def test_import_without_position_keeps_the_file_coordinates(win, bridge,
                                                            tmp_path):
    files = _model_files(tmp_path)
    res = _ok(_ask(bridge, "import_model", {"path": str(files / "caisson.igz")}))
    assert res["bounds"]["min"][0] == pytest.approx(10.0, abs=1e-3)
    res = _ok(_ask(bridge, "import_model", {
        "path": str(files / "caisson.igz"), "position": [0, 0, 0],
        "anchor": "origin"}))
    assert res["bounds"]["min"][0] == pytest.approx(10.0, abs=1e-3)


def test_import_an_obj_in_the_unit_given(win, bridge, tmp_path):
    files = _model_files(tmp_path)
    res = _ok(_ask(bridge, "import_model", {"path": str(files / "caisson.obj"),
                                            "units": "m",
                                            "position": [0, 0, 0]}))
    assert res["units"] == "m"
    assert res["size_m"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-3)
    res = _ok(_ask(bridge, "import_model", {"path": str(files / "caisson.obj"),
                                            "units": "cm",
                                            "position": [5, 0, 0]}))
    assert res["size_m"] == pytest.approx([0.01, 0.01, 0.01], abs=1e-4)


def test_import_model_says_what_is_wrong(win, bridge, tmp_path):
    r = _ask(bridge, "import_model", {"path": str(tmp_path / "x.fbx")})
    assert not r["ok"] and ".glb" in r["error"]
    r = _ask(bridge, "import_model", {"path": str(tmp_path / "absent.glb")})
    assert not r["ok"] and "no such file" in r["error"]
    r = _ask(bridge, "import_model", {"path": str(tmp_path / "x.dxf")})
    assert not r["ok"] and "import_dxf" in r["error"]


# ---- finish ------------------------------------------------------------------------

def test_paint_sets_the_render_finish_of_a_named_material(win, bridge,
                                                          tmp_path):
    vp = win.viewport
    _box_group(vp.scene, "Vitrage")
    r = _ask(bridge, "paint", {"target": "Vitrage", "material":
                               {"color": [0.7, 0.85, 1.0], "finish": "glass"}})
    assert not r["ok"] and "name" in r["error"]
    r = _ask(bridge, "paint", {"target": "Vitrage", "material":
                               {"name": "Verre", "color": [1, 1, 1],
                                "finish": "shiny"}})
    assert not r["ok"] and "glass" in r["error"]
    res = _ok(_ask(bridge, "paint", {"target": "Vitrage", "faces": "all",
                                     "material": {"color": [0.7, 0.85, 1.0],
                                                  "opacity": 0.3,
                                                  "name": "Verre clair",
                                                  "finish": "glass"}}))
    assert res["changed"] and res["finish"] == "glass"
    assert vp.scene.materials["Verre clair"].finish == "glass"
    # The finish reaches the GLB the Blender render reads.
    from formats.gltf import save_glb
    save_glb(vp.scene, tmp_path / "rendu.glb")
    blob = (tmp_path / "rendu.glb").read_bytes()
    assert b'"ingetrazo_finish":"glass"' in blob.replace(b" ", b"")
    # The finish alone, on the material by its name: one undoable step.
    res = _ok(_ask(bridge, "paint", {"target": "Vitrage", "faces": "all",
                                     "material": {"name": "Verre clair",
                                                  "finish": "gloss"}}))
    assert res["changed"] and vp.scene.materials["Verre clair"].finish == "gloss"
    assert all(f.attrs.get("opacity") == 0.3 for f in
               next(g for g in vp.scene.groups if g.name == "Vitrage").mesh.faces)
    assert _ok(_ask(bridge, "undo"))["ok"]
    assert vp.scene.materials["Verre clair"].finish == "glass"
    r = _ask(bridge, "paint", {"target": "Vitrage",
                               "material": {"name": "Inconnu"}})
    assert not r["ok"] and "no material named" in r["error"]


def test_client_lists_the_lot_2_tools():
    import ingetrazo_mcp as client
    names = {t["name"] for t in client.TOOLS}
    assert {"search_components", "insert_component", "import_model"} <= names
    paint = next(t for t in client.TOOLS if t["name"] == "paint")
    assert "finish" in paint["description"]
    desc = next(t for t in client.TOOLS
                if t["name"] == "run_python")["description"]
    assert "insert_component" in desc and "finish" in desc


# ---- Fixes found by the lot-2 acceptance run (22-11-2026) ----------------------------

def test_paint_faces_of_every_group_with_that_name(win, bridge):
    vp = win.viewport
    frames = [_box_group(vp.scene, "A-FENETRE-120 cadre", origin=(i * 2, 0, 0))
              for i in range(4)]
    res = _ok(_ask(bridge, "paint", {
        "target": "A-FENETRE-120 cadre", "faces": "all",
        "material": {"color": [0.22, 0.23, 0.25], "name": "Alu anthracite",
                     "finish": "metal"}}))
    assert res["changed"] and res["groups"] == 4 and res["faces"] == 24
    assert all(f.attrs.get("mat") == "Alu anthracite"
               for g in frames for f in g.mesh.faces)
    assert vp.history.undo()
    assert not any(f.attrs.get("mat") for g in frames for f in g.mesh.faces)


def test_run_python_material_registry_is_undone(win, bridge):
    vp = win.viewport
    res = _ok(_ask(bridge, "run_python", {"code": (
        "from core.materials import Material\n"
        "scene.materials['Inox'] = Material('Inox', (0.7, 0.7, 0.72), "
        "finish='metal')")}))
    assert res["changed"] is True and "Inox" in vp.scene.materials
    assert vp.history.undo() and "Inox" not in vp.scene.materials
    assert vp.history.redo()
    assert vp.scene.materials["Inox"].finish == "metal"
    res = _ok(_ask(bridge, "run_python", {"code":
                                          "scene.materials['Inox'].finish = 'gloss'"}))
    assert res["changed"] is True
    assert vp.history.undo() and vp.scene.materials["Inox"].finish == "metal"


def _two_part_igz(tmp_path):
    """A piece of furniture made of two parts (as library models often are)."""
    from core.scene import Scene
    from formats.igz import save_scene
    s = Scene()
    _box_group(s, "plateau", origin=(0, 0, 1))
    _box_group(s, "pied", origin=(0, 0, 0))
    save_scene(s, tmp_path / "meuble.igz")
    return tmp_path / "meuble.igz"


def test_ifc_and_csv_take_a_component_with_its_parts_where_it_stands(
        win, bridge, tmp_path):
    path = _two_part_igz(tmp_path)
    res = _ok(_ask(bridge, "import_model", {
        "path": str(path), "position": [20, 30, 0], "name": "Meuble",
        "ifc": "IfcFurniture"}))
    assert res["children"] == 2
    out = _ok(_ask(bridge, "export_ifc", {"path": str(tmp_path / "m.ifc")}))
    text = Path(out["path"]).read_text("ascii")
    assert out["elements"] == 1 and "'Meuble'" in text
    assert "IFCFURNITURE(" in text and "IFCBUILDINGELEMENTPROXY" not in text
    # Its points are in world space: around x = 20, y = 30 — not at the
    # parts' local origin.
    import re
    xs = [float(m.group(1)) for m in re.finditer(
        r"IFCCARTESIANPOINT\(\(([-0-9.]+),([-0-9.]+),", text)]
    assert any(19.0 < x < 21.0 for x in xs)
    csv = _ok(_ask(bridge, "quantities_csv", {"path": str(tmp_path / "q.csv")}))
    line = [l for l in Path(csv["path"]).read_text("utf-8").splitlines()
            if "Meuble" in l][0]
    assert line.split(",")[2:4] == ["1.0000", "und"]   # one piece


# ---- Lot 3: choosing a model from its picture ----------------------------------

def _thumb(path: Path, rgb) -> Path:
    from PySide6.QtGui import QColor, QImage
    path.parent.mkdir(parents=True, exist_ok=True)
    img = QImage(128, 128, QImage.Format_RGB32)
    img.fill(QColor(*rgb))
    img.save(str(path))
    return path


def test_search_preview_draws_a_numbered_contact_sheet(win, bridge,
                                                       library_on_disk,
                                                       tmp_path):
    from PySide6.QtGui import QImage
    _thumb(library_on_disk / "miniaturas" / "cama-doble.png", (200, 40, 40))
    out = tmp_path / "choix-lit.png"
    res = _ok(_ask(bridge, "search_components",
                   {"query": "cama", "size_cm": [160, 200, None],
                    "sheet_path": str(out)}))
    assert [r["n"] for r in res["results"]] == [1, 2]
    assert res["sheet"] == str(out) and res["sheet_count"] == 2
    first, second = res["results"]
    assert first["id"] == "cama-doble" and Path(first["thumbnail"]).is_file()
    assert second["thumbnail"] is None and res["without_picture"] == [2]
    img = QImage(str(out))
    assert (img.width(), img.height()) == (2 * 210, 266)
    # The first cell shows the model's picture, the second says it has none.
    c = img.pixelColor(105, 120)
    assert (c.red(), c.green(), c.blue()) == (200, 40, 40)
    c = img.pixelColor(210 + 105, 120)
    assert c.red() == c.green() == c.blue() != 255


def test_search_preview_shows_the_bundled_pictures(win, bridge, monkeypatch,
                                                   tmp_path):
    monkeypatch.setenv("INGETRAZO_LIBRARY", str(tmp_path / "offline"))
    res = _ok(_ask(bridge, "search_components",
                   {"query": "sofa", "preview": True}))
    assert res["results"][0]["id"] == "bundled:sofa"
    assert Path(res["results"][0]["thumbnail"]).name == "sofa.png"
    assert Path(res["sheet"]).is_file() and res["without_picture"] == []


def test_search_without_preview_draws_nothing(win, bridge, library_on_disk):
    res = _ok(_ask(bridge, "search_components", {"query": "radiador"}))
    assert "sheet" not in res and "thumbnail" not in res["results"][0]
    assert res["results"][0]["n"] == 1


def test_search_sheet_path_in_a_missing_folder_is_refused(win, bridge,
                                                          library_on_disk,
                                                          tmp_path):
    reply = _ask(bridge, "search_components",
                 {"query": "cama", "sheet_path": str(tmp_path / "nope" / "a")})
    assert not reply["ok"] and "folder does not exist" in reply["error"]


def test_client_returns_the_sheet_as_an_image(monkeypatch, tmp_path):
    import ingetrazo_mcp as client
    sheet = _thumb(tmp_path / "s.png", (10, 20, 30))

    def fake_bridge(name, args):
        assert name == "search_components" and args["preview"] is True
        return {"ok": True, "result": {"total": 1, "sheet": str(sheet),
                                       "results": [{"n": 1, "id": "x"}]}}
    monkeypatch.setattr(client, "_bridge", fake_bridge)
    res = client._call("search_components", {"query": "x", "preview": True})
    kinds = [c["type"] for c in res["content"]]
    assert kinds == ["text", "image"] and not res["isError"]
    assert '"n": 1' in res["content"][0]["text"]
    spec = next(t for t in client.TOOLS if t["name"] == "search_components")
    assert {"preview", "sheet_path"} <= set(spec["inputSchema"]["properties"])


def test_size_ranking_puts_models_without_a_size_last(win, bridge,
                                                      library_on_disk):
    """Found on Windows: the bundled «Sofa» (no size) came first for
    size_cm [200, 90] — a distance of nothing beat every real model."""
    from core.agent_tools import search_components
    entries = [
        {"id": "bundled:sofa", "name": "Sofa", "name_en": "Sofa",
         "category": "Starter", "size_cm": []},
        {"id": "a", "name": "Sofá", "name_en": "Sofa", "category": "Salón",
         "size_cm": [230.0, 95.0, 80.0]},
        {"id": "b", "name": "Sofá", "name_en": "Sofa", "category": "Salón",
         "size_cm": [199.0, 84.0, 80.0]},
    ]
    res = search_components("sofa", size_cm=[200, 90, None], _entries=entries)
    assert [r["id"] for r in res["results"]] == ["b", "a", "bundled:sofa"]
    res = search_components("sofa", size_cm=[None, None, None],
                            _entries=entries)
    assert res["results"][0]["id"] == "bundled:sofa"     # no size asked


def test_sheet_card_fits_all_four_lines(tmp_path):
    from PySide6.QtGui import QImage
    from core.agent_tools import contact_sheet
    r = {"n": 1, "id": "x", "name": "Sofá", "name_en": "Couch",
         "size_cm": [201, 91, 84], "licence": "Creative Commons Attribution",
         "source": "library"}
    out = contact_sheet([r], {}, tmp_path / "s.png")
    img = QImage(str(out))
    # The border of the card is drawn below the last text line: the 4th
    # line (licence) ends above it, with white between them.
    bottom = img.height() - 4
    row = [img.pixelColor(x, bottom - 2) for x in range(20, 190)]
    assert all(c.red() == 255 for c in row)
