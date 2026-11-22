# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
# Copyright (C) 2026 Maher (fork chems/ingetrazo, branch mcp-agent).
"""The MCP tools of the «mcp-agent» fork, one block per point of the
22-11-2026 list: files (save/open/IFC/CSV), the Blender render waited for,
the DXF import, paint, render lights, UTF-8 over stdio, hidden objects out
of pictures, query_model counting groups, and run_python's description.

Driven the way an agent drives them: requests over the bridge's socket
while this (Qt) thread pumps events, and the stdio client's own functions
where the client does the work (render waiting, encoding)."""
from __future__ import annotations

import io
import json
import os
import socket
import sys
import threading
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtGui import QVector3D  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

if QApplication.instance() is None:
    QApplication(sys.argv[:1])

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

V = QVector3D
ACCENT = "Allège séjour"          # the name that came back as «AllÃ¨ge»


# ---- Plumbing ------------------------------------------------------------------

def _rpc(sock, tool, args=None):
    sock.sendall((json.dumps({"id": 1, "tool": tool,
                              "args": args or {}}) + "\n").encode())
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        assert chunk, "bridge closed"
        buf += chunk
    return json.loads(buf.split(b"\n", 1)[0])


def _in_worker(fn):
    """Run ``fn`` on a worker thread while this thread pumps Qt events —
    the bridge executes on the main thread through a queued signal."""
    app = QApplication.instance()
    out: dict = {}

    def run():
        try:
            out["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            out["error"] = exc

    t = threading.Thread(target=run)
    t.start()
    while t.is_alive():
        app.processEvents()
        t.join(timeout=0.01)
    if "error" in out:
        raise out["error"]
    return out["value"]


def _ask(bridge, tool, args=None):
    def call():
        s = socket.socket()
        s.settimeout(30.0)
        s.connect(("127.0.0.1", bridge.port))
        try:
            return _rpc(s, tool, args)
        finally:
            s.close()
    return _in_worker(call)


def _ok(reply):
    assert reply["ok"], reply.get("error")
    return reply["result"]


@pytest.fixture
def win(monkeypatch):
    from core import extensions
    from core.paths import app_root
    # Only the bundled plugins: the Render tab must come from them.
    monkeypatch.setattr(extensions, "plugin_dirs",
                        lambda: [app_root() / "plugins"])
    monkeypatch.setenv("INGETRAZO_AI_PORT", "0")
    from views.main_window import MainWindow
    w = MainWindow()
    yield w
    for d in list(getattr(w, "_documents", [])):
        try:
            w._saved_version = d.viewport.scene.version
        except Exception:  # noqa: BLE001
            pass
    w._saved_version = w.viewport.scene.version
    bridge = getattr(w, "_ai_bridge", None)
    if bridge is not None:
        bridge.stop()
    w.close()


@pytest.fixture
def bridge(win):
    from plugins.ai_bridge import _bridge_of
    b = _bridge_of(win.viewport)
    b.start()
    yield b
    b.stop()


def _box_group(scene, name, origin=(0, 0, 0)):
    from core.group import Group
    from core.mesh import Mesh
    x, y, z = origin
    m = Mesh()
    p = [V(x, y, z), V(x + 1, y, z), V(x + 1, y + 1, z), V(x, y + 1, z),
         V(x, y, z + 1), V(x + 1, y, z + 1), V(x + 1, y + 1, z + 1),
         V(x, y + 1, z + 1)]
    for idx in ((3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
                (2, 3, 7, 6), (3, 0, 4, 7)):
        m.add_face([p[i] for i in idx])
    g = Group(m, name=name)
    scene.groups.append(g)
    scene.version += 1
    return g


def _png(path: Path) -> Path:
    from PySide6.QtGui import QColor, QImage
    img = QImage(8, 8, QImage.Format_RGB32)
    img.fill(QColor(200, 120, 60))
    img.save(str(path))
    return path


# ---- 1. save / open / export_ifc / quantities_csv ---------------------------------

def test_save_then_open_keeps_accented_names(win, bridge, tmp_path):
    _box_group(win.viewport.scene, ACCENT)
    res = _ok(_ask(bridge, "save", {"path": str(tmp_path / "maison")}))
    out = Path(res["path"])
    assert out.suffix == ".igz" and out.is_file() and res["bytes"] > 0
    assert not win._is_dirty()
    assert win._doc.path == out
    # Saved again: the older file is kept as .bak, as File ▸ Save does.
    _ok(_ask(bridge, "save", {"path": str(out)}))
    assert out.with_name(out.name + ".bak").is_file()

    win._new_document()
    res = _ok(_ask(bridge, "open", {"path": str(out)}))
    assert res["opened"] == str(out)
    assert ACCENT in res["group_names"]
    assert win.viewport.scene is not None
    assert any(g.name == ACCENT for g in win.viewport.scene.groups)


def test_open_refuses_a_dxf_and_a_missing_file(win, bridge, tmp_path):
    r = _ask(bridge, "open", {"path": str(tmp_path / "plan.dxf")})
    assert not r["ok"] and "import_dxf" in r["error"]
    r = _ask(bridge, "open", {"path": str(tmp_path / "nope.igz")})
    assert not r["ok"] and "no such file" in r["error"]


def test_open_asks_before_dropping_an_autosaved_copy(win, bridge, tmp_path,
                                                     monkeypatch):
    _box_group(win.viewport.scene, "Mur")
    saved = Path(_ok(_ask(bridge, "save",
                          {"path": str(tmp_path / "a.igz")}))["path"])
    import shutil
    path = tmp_path / "copie.igz"          # a document NOT open in a tab
    shutil.copy(saved, path)
    from core import autosave
    win._new_document()
    real = autosave.pending
    monkeypatch.setattr(autosave, "pending",
                        lambda p, **kw: path if p == path else real(p, **kw))
    r = _ask(bridge, "open", {"path": str(path)})
    assert not r["ok"] and "recover" in r["error"]
    res = _ok(_ask(bridge, "open", {"path": str(path), "recover": False}))
    assert res["recovered"] is False


def test_open_does_not_ask_about_the_slot_of_a_document_already_open(
        win, bridge, tmp_path, monkeypatch):
    _box_group(win.viewport.scene, "Mur")
    path = Path(_ok(_ask(bridge, "save",
                         {"path": str(tmp_path / "ouvert.igz")}))["path"])
    from core import autosave
    real = autosave.pending
    monkeypatch.setattr(autosave, "pending",
                        lambda p, **kw: path if p == path else real(p, **kw))
    res = _ok(_ask(bridge, "open", {"path": str(path)}))
    assert res["document"] == str(path) and res["recovered"] is False


def test_export_ifc_and_quantities_csv(win, bridge, tmp_path):
    from core import bim
    scene = win.viewport.scene
    r = _ask(bridge, "export_ifc", {"path": str(tmp_path / "m.ifc")})
    assert not r["ok"] and "bim.tag_group" in r["error"]
    g = _box_group(scene, ACCENT)
    bim.tag_group(g, "IfcWall", ACCENT)
    res = _ok(_ask(bridge, "export_ifc", {"path": str(tmp_path / "m.ifc"),
                                          "project_name": "Maison"}))
    text = Path(res["path"]).read_text("ascii")
    assert res["elements"] == 1 and "IFCWALL" in text
    # STEP strings carry the accent encoded (\X2\00E8\X0\ for è).
    assert "\\X2\\00E8\\X0\\" in text
    res = _ok(_ask(bridge, "quantities_csv",
                   {"path": str(tmp_path / "metrado")}))
    csv = Path(res["path"]).read_text("utf-8")
    assert res["path"].endswith(".csv") and res["rows"] == 1
    assert csv.startswith("class,name,metrado,unit,area_m2,volume_m3")
    assert f'IfcWall,"{ACCENT}"' in csv


def test_file_tools_want_an_absolute_path_in_an_existing_folder(win, bridge,
                                                                tmp_path):
    r = _ask(bridge, "save", {"path": "relatif.igz"})
    assert not r["ok"] and "absolute" in r["error"]
    r = _ask(bridge, "save", {"path": str(tmp_path / "no" / "x.igz")})
    assert not r["ok"] and "folder does not exist" in r["error"]


# ---- 1. render (Blender started, waited for) --------------------------------------

_FAKE_BLENDER = r'''#!{python}
import json, sys, time
job = json.load(open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8"))
open(job["output"].replace("render.png", "seen.json"), "w").write(
    json.dumps({{"lights": job["lights"], "camera": job["camera"],
                "engine": job["engine"], "w": job["width"],
                "h": job["height"]}}))
for i in range(1, 4):
    print("Sample %d/3" % i, flush=True)
    time.sleep(0.05)
data = open({png!r}, "rb").read()
open(job["output"], "wb").write(data)
'''


@pytest.fixture
def fake_blender(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("the fake Blender is a POSIX script")
    png = _png(tmp_path / "src.png")
    exe = tmp_path / "blender"
    exe.write_text(_FAKE_BLENDER.format(python=sys.executable, png=str(png)))
    exe.chmod(0o755)
    monkeypatch.setenv("INGETRAZO_BLENDER", str(exe))
    from core import render_blender as rb
    monkeypatch.setattr(rb, "work_base", lambda: tmp_path / "work")
    return exe


def test_render_waits_for_blender_and_delivers_the_png(win, bridge, tmp_path,
                                                       fake_blender,
                                                       monkeypatch):
    import ingetrazo_mcp as client
    from core.saved_views import SavedView
    scene = win.viewport.scene
    _box_group(scene, "Volume")
    scene.saved_views.append(SavedView.capture("Façade sud", scene,
                                               win.viewport.camera))
    _ok(_ask(bridge, "add_light", {"position": [0.5, 0.5, 3.0],
                                   "name": "Plafonnier"}))
    monkeypatch.setattr(client, "PORT", bridge.port)
    monkeypatch.setattr(client, "RENDER_POLL_S", 0.05)
    monkeypatch.setattr(client, "_sock", None)
    out = tmp_path / "rendu.png"
    res = _in_worker(lambda: client._call("render", {
        "path": str(out), "scene": "Façade sud", "engine": "cycles",
        "width": 640, "height": 360, "quality": 0}))
    assert not res["isError"], res
    body = json.loads(res["content"][0]["text"])
    assert body["state"] == "done" and body["path"] == str(out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert body["started"]["scene"] == "Façade sud"
    seen = json.loads(next((tmp_path / "work").glob("*/seen.json"))
                      .read_text())
    assert seen["engine"] == "cycles" and (seen["w"], seen["h"]) == (640, 360)
    assert [lt["name"] for lt in seen["lights"]] == ["Plafonnier"]


def test_render_names_a_missing_scene_and_a_missing_blender(win, bridge,
                                                            tmp_path,
                                                            fake_blender,
                                                            monkeypatch):
    r = _ask(bridge, "render_start", {"path": str(tmp_path / "a.png"),
                                      "scene": "Inconnue"})
    assert not r["ok"] and "no scene named" in r["error"]
    from core import render_blender as rb
    monkeypatch.setenv("INGETRAZO_BLENDER", "")
    monkeypatch.setattr(rb, "find_blender", lambda saved=None: None)
    r = _ask(bridge, "render_start", {"path": str(tmp_path / "a.png")})
    assert not r["ok"] and "Blender not found" in r["error"]


# ---- 1. import_dxf -------------------------------------------------------------

def _dxf(path: Path, units: int = 4) -> Path:
    import ezdxf
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = units
    doc.layers.add("Murs")
    doc.layers.add(ACCENT)
    msp = doc.modelspace()
    # A 10 m × 8 m outline in millimetres, and a sill line.
    msp.add_lwpolyline([(0, 0), (10000, 0), (10000, 8000), (0, 8000)],
                       close=True, dxfattribs={"layer": "Murs"})
    msp.add_line((2000, 0), (3200, 0), dxfattribs={"layer": ACCENT})
    doc.saveas(path)
    return path


def test_import_dxf_returns_the_groups_it_made(win, bridge, tmp_path):
    vp = win.viewport
    path = _dxf(tmp_path / "test-transfert.dxf")
    depth = len(vp.history.undo_stack)
    res = _ok(_ask(bridge, "import_dxf", {"path": str(path),
                                          "units": "mm",
                                          "create_faces": True}))
    names = {g["name"] for g in res["groups"]}
    assert names == {"Murs", ACCENT}
    assert res["units"] == "mm" and res["scale_m_per_unit"] == 0.001
    murs = next(g for g in res["groups"] if g["name"] == "Murs")
    assert murs["layer"] == "Murs" and murs["faces"] == 1
    assert res["bounds"]["max"][0] == pytest.approx(10.0)
    assert len(vp.history.undo_stack) == depth + 1      # ONE step
    assert vp.history.undo()
    assert not any(g.name == "Murs" for g in vp.scene.groups)


def test_import_dxf_auto_units_and_errors(win, bridge, tmp_path):
    path = _dxf(tmp_path / "plan.dxf")
    res = _ok(_ask(bridge, "import_dxf", {"path": str(path)}))
    assert res["units"] == "mm"
    r = _ask(bridge, "import_dxf", {"path": str(path), "units": "pouces"})
    assert not r["ok"] and "units" in r["error"]
    r = _ask(bridge, "import_dxf", {"path": str(tmp_path / "absent.dxf")})
    assert not r["ok"] and "no such DXF" in r["error"]


# ---- 2. paint -------------------------------------------------------------------

def test_paint_a_texture_on_faces_inside_a_group(win, bridge, tmp_path):
    vp = win.viewport
    g = _box_group(vp.scene, ACCENT)
    img = _png(tmp_path / "brique.png")
    g.mesh._attrs_dirty = False
    depth = len(vp.history.undo_stack)
    mat = {"texture": {"path": str(img), "sw": 0.5, "sh": 0.25, "rot": 90}}
    res = _ok(_ask(bridge, "paint", {"target": ACCENT, "faces": "all",
                                     "material": mat}))
    assert res["changed"] is True and res["faces"] == 6
    assert all(f.attrs["texture"] == {"path": str(img), "sw": 0.5,
                                      "sh": 0.25, "rot": 90.0}
               for f in g.mesh.faces)
    assert g.mesh._attrs_dirty is True               # the display re-reads it
    assert len(vp.history.undo_stack) == depth + 1
    # The same paint again changes nothing, and says so.
    res = _ok(_ask(bridge, "paint", {"target": ACCENT, "faces": "all",
                                     "material": mat}))
    assert res["changed"] is False
    assert len(vp.history.undo_stack) == depth + 1
    assert _ok(_ask(bridge, "undo"))["ok"]
    assert not any("texture" in f.attrs for f in g.mesh.faces)
    assert _ok(_ask(bridge, "redo"))["ok"]
    assert all("texture" in f.attrs for f in g.mesh.faces)


def test_paint_a_group_a_colour_by_normal_and_the_library(win, bridge):
    vp = win.viewport
    g = _box_group(vp.scene, "Toit")
    res = _ok(_ask(bridge, "paint", {"target": "Toit",
                                     "material": {"color": [204, 51, 51]}}))
    assert res["changed"] and g.material == {"color": (0.8, 0.2, 0.2)}
    res = _ok(_ask(bridge, "paint", {
        "target": {"group": "Toit", "faces": {"normal": [0, 0, 1]}},
        "material": {"library": "brick_red", "name": "Brique rouge"}}))
    assert res["faces"] == 1 and res["changed"]
    top = [f for f in g.mesh.faces if f.normal().z() > 0.9][0]
    assert top.attrs["texture"]["path"].endswith("brick_red.png")
    assert top.attrs["mat"] == "Brique rouge"
    assert "Brique rouge" in vp.scene.materials
    vp.history.undo()
    assert "Brique rouge" not in vp.scene.materials
    assert "texture" not in top.attrs


def test_paint_says_what_is_wrong(win, bridge, tmp_path):
    _box_group(win.viewport.scene, "Mur")
    r = _ask(bridge, "paint", {"target": "Murr", "material": {"color": [1, 0, 0]}})
    assert not r["ok"] and "no group named" in r["error"] and "Mur" in r["error"]
    r = _ask(bridge, "paint", {"target": "Mur", "material":
                               {"texture": {"path": str(tmp_path / "x.png")}}})
    assert not r["ok"] and "not found" in r["error"]
    r = _ask(bridge, "paint", {"target": "Mur", "material":
                               {"library": "papier_peint_inexistant"}})
    assert not r["ok"] and "brick_red" in r["error"]


def test_run_python_paint_inside_a_group_is_seen_and_undone(win, bridge,
                                                           tmp_path):
    """The old trap: f.attrs["texture"] written by script in a group gave
    changed: False and showed only after a vertex moved."""
    vp = win.viewport
    g = _box_group(vp.scene, "Bloc")
    img = _png(tmp_path / "t.png")
    g.mesh._attrs_dirty = False
    res = _ok(_ask(bridge, "run_python", {"code": (
        "g = [x for x in groups if x.name == 'Bloc'][0]\n"
        f"g.mesh.faces[0].attrs['texture'] = {{'path': {str(img)!r}, "
        "'sw': 1.0, 'sh': 1.0}")}))
    assert res["changed"] is True
    assert g.mesh._attrs_dirty is True
    assert vp.history.undo()
    assert "texture" not in g.mesh.faces[0].attrs


def test_run_python_hide_is_one_undoable_step_and_failures_roll_back(win,
                                                                    bridge):
    vp = win.viewport
    g = _box_group(vp.scene, "Bloc")
    res = _ok(_ask(bridge, "run_python",
                   {"code": "[x for x in groups if x.name == 'Bloc'][0]"
                            ".hidden = True"}))
    assert res["changed"] is True and g.hidden is True
    assert vp.history.undo() and g.hidden is False
    # A recipe that edits a group then raises leaves the group untouched.
    n = len(g.mesh.faces)
    res = _ok(_ask(bridge, "run_python", {"code": (
        "g = [x for x in groups if x.name == 'Bloc'][0]\n"
        "g.mesh.add_face([QVector3D(5,5,0), QVector3D(6,5,0), "
        "QVector3D(6,6,0)])\n"
        "g.name = 'Renommé'\n"
        "raise RuntimeError('boom')")}))
    assert res["error"] and len(g.mesh.faces) == n and g.name == "Bloc"


# ---- 3. render lights --------------------------------------------------------------

def _render_panel(w):
    for dock in w._extension_docks:
        if type(dock.widget()).__name__ == "RenderPanel":
            return dock.widget()
    raise AssertionError("no Render tab")


def test_lights_are_added_listed_removed_and_shown_in_the_panel(win, bridge):
    panel = _render_panel(win)
    vp = win.viewport
    res = _ok(_ask(bridge, "add_light", {
        "position": [1, 2, 2.5], "kind": "spot", "target": [1, 2, 0],
        "power": 900, "kelvin": 3000, "angle": 45, "name": "Spot îlot",
        "ambience": "night"}))
    assert res["index"] == 0 and res["kind"] == "spot"
    assert res["dir"] == [0.0, 0.0, -2.5] and res["kelvin"] == 3000
    _ok(_ask(bridge, "add_light", {"position": [0, 0, 2.4]}))
    QApplication.instance().processEvents()
    assert panel._lights.count() == 2
    assert panel._lights.item(0).text().startswith("Spot îlot")
    listed = _ok(_ask(bridge, "list_lights"))
    assert [lt["name"] for lt in listed["lights"]] == ["Spot îlot", ""]
    assert listed["ambience"] == "night"
    assert vp.scene.plugin_data["render_blender"]["lights"][1]["power"] == 400
    _ok(_ask(bridge, "remove_light", {"name": "Spot îlot"}))
    QApplication.instance().processEvents()
    assert panel._lights.count() == 1
    assert _ok(_ask(bridge, "undo"))["ok"]            # the removal, undone
    QApplication.instance().processEvents()
    assert panel._lights.count() == 2
    r = _ask(bridge, "remove_light", {"index": 9})
    assert not r["ok"] and "out of range" in r["error"]
    r = _ask(bridge, "add_light", {"position": [0, 0]})
    assert not r["ok"]


# ---- 4. UTF-8 on stdio ------------------------------------------------------------

def test_the_client_reads_utf8_whatever_the_console_code_page(monkeypatch):
    import ingetrazo_mcp as client
    seen = {}

    def fake_bridge(tool, args):
        seen["code"] = args.get("code")
        return {"ok": True, "result": {"stdout": "ok", "changed": True}}

    monkeypatch.setattr(client, "_bridge", fake_bridge)
    req = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
        "name": "run_python",
        "arguments": {"code": "g.name = 'Allège'  # «séjour»"}}}
    raw = (json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8")
    # What Windows hands a console program: text streams in cp1252.
    stdin = io.TextIOWrapper(io.BytesIO(raw), encoding="cp1252")
    out_bytes = io.BytesIO()
    stdout = io.TextIOWrapper(out_bytes, encoding="cp1252")
    ins, outs = client._streams(stdin, stdout)
    client.serve(ins, outs)
    outs.flush()
    assert seen["code"] == "g.name = 'Allège'  # «séjour»"
    reply = json.loads(out_bytes.getvalue().decode("utf-8"))
    assert reply["id"] == 7 and not reply["result"]["isError"]


def test_accented_names_reach_the_document_intact(win, bridge):
    res = _ok(_ask(bridge, "run_python", {"code": (
        "g = box((0,0,0), (1,1,1), name='Allège')\n"
        "h = box((2,0,0), (1,1,1), name='S\\u00e9jour')\n"
        "print(g.name, h.name)")}))
    assert res["stdout"].strip() == "Allège Séjour"
    names = {g.name for g in win.viewport.scene.groups}
    assert {"Allège", "Séjour"} <= names


# ---- 5. hidden objects stay out of pictures ----------------------------------------

def _rgb(img):
    import numpy as np
    from PySide6.QtGui import QImage
    img = img.convertToFormat(QImage.Format_RGBA8888)
    buf = np.frombuffer(img.constBits(), np.uint8)
    px = buf.reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4]
    return px.reshape(img.height(), img.width(), 4)[..., :3].astype(int)


def test_a_hidden_scale_figure_is_not_in_the_screenshot(win, bridge):
    import numpy as np
    from PySide6.QtGui import QImage
    win.show()
    app = QApplication.instance()
    for _ in range(3):
        app.processEvents()
    vp = win.viewport
    if getattr(vp, "_gl", None) is None or not vp.isValid():
        pytest.skip("no OpenGL context on this platform")
    fig = [g for g in vp.scene.groups if g.billboard]
    assert fig, "a new document has its scale figure"
    vp.camera.place(V(-0.65, -6.0, 1.2), V(-0.65, -0.6, 0.9))
    shot = lambda: _rgb(QImage(_ok(_ask(bridge, "screenshot",       # noqa: E731
                                         {"width": 320, "height": 240}))
                               ["path"]))
    with_figure = shot()
    # Hidden by a script WITHOUT any version bump: the stale-memo case
    # that kept the figure in the picture.
    fig[0].hidden = True
    hidden = shot()
    groups = list(vp.scene.groups)
    vp.scene.groups[:] = [g for g in groups if not g.billboard]
    vp.scene.version += 1
    removed = shot()
    vp.scene.groups[:] = groups
    assert int((np.abs(with_figure - removed).sum(-1) > 30).sum()) > 100
    assert int((np.abs(hidden - removed).sum(-1) > 30).sum()) == 0


def test_fresh_view_forgets_the_placement_memo():
    from core.agent_tools import fresh_view

    class VP:
        pass
    vp = VP()
    vp._epoch_memo, vp._epoch_same, vp._billboard_groups = 1, 2, 3
    vp._tick = 5
    fresh_view(vp)
    assert (vp._epoch_memo, vp._epoch_same, vp._billboard_groups) == \
        (None, None, None)
    assert vp._tick == 6


# ---- 6. query_model ----------------------------------------------------------------

def test_query_model_counts_group_faces_and_pages_long_lists(win, bridge):
    scene = win.viewport.scene
    scene.groups[:] = []
    for i in range(130):
        _box_group(scene, f"Pièce {i:03d}", origin=(2 * i, 0, 0))
    scene.mesh.add_face([V(0, -5, 0), V(1, -5, 0), V(1, -4, 0)])
    res = _ok(_ask(bridge, "query_model"))
    assert res["faces_loose"] == 1 and res["faces"] == 1
    assert res["faces_in_groups"] == 130 * 6
    assert res["faces_total"] == 130 * 6 + 1
    assert res["edges_in_groups"] == 130 * 12
    assert res["groups_total"] == 130 and len(res["group_names"]) == 100
    assert res["group_names_truncated"] is True
    assert res["group_names_next_offset"] == 100
    more = _ok(_ask(bridge, "query_model", {"offset": 100}))
    assert more["group_names"][0] == "Pièce 100"
    assert len(more["group_names"]) == 30
    assert more["group_names_truncated"] is False
    assert more["group_names_next_offset"] is None


def test_query_model_counts_a_component_once_per_definition(win, bridge):
    from core.group import Group
    from PySide6.QtGui import QMatrix4x4
    scene = win.viewport.scene
    scene.groups[:] = []
    proto = _box_group(scene, "Chaise")
    for i in range(3):
        inst = Group(proto.mesh, name="Chaise")
        m = QMatrix4x4()
        m.translate(3 + i, 0, 0)
        inst.xform = m
        scene.groups.append(inst)
    scene.version += 1
    res = _ok(_ask(bridge, "query_model"))
    assert res["faces_in_groups"] == 6          # one mesh
    assert res["faces_placed"] == 24            # four placements drawn


# ---- 7. run_python's description and the tool list ---------------------------------

def test_the_client_lists_every_tool_and_documents_the_api():
    import ingetrazo_mcp as client
    names = {t["name"] for t in client.TOOLS}
    assert names >= {"run_python", "query_model", "screenshot", "undo",
                     "redo", "save", "open", "export_ifc", "quantities_csv",
                     "render", "import_dxf", "paint", "add_light",
                     "list_lights", "remove_light"}
    desc = next(t for t in client.TOOLS
                if t["name"] == "run_python")["description"]
    for term in ("f.normal()", "f.centroid()", "cam.place(", "perspective",
                 "fov_deg", "render_image", '"sw"', '"rot"',
                 "texture_library", "bim.tag_group", "bim.quantities_csv",
                 "show_axes", "import_dxf", "paint(", "render("):
        assert term in desc, term
    for tool in client.TOOLS:
        assert tool["inputSchema"]["type"] == "object"
        for req in tool["inputSchema"].get("required", []):
            assert req in tool["inputSchema"]["properties"], (tool["name"], req)


def test_documented_api_facts_hold(win):
    """What MCP_NOTES says, checked against the code it describes."""
    from core import bim
    from core.agent_tools import texture_library
    vp = win.viewport
    g = _box_group(vp.scene, "Bloc")
    f = g.mesh.faces[0]
    assert isinstance(f.normal(), QVector3D)
    assert isinstance(f.centroid(), QVector3D)
    assert vp.camera.place(V(5, -5, 3), V(0, 0, 0))
    assert isinstance(vp.camera.perspective, bool)
    assert isinstance(vp.camera.fov_deg, float)
    assert callable(vp.render_image)
    bim.tag_group(g, "IfcWall", "Mur")
    assert "IfcWall" in bim.quantities_csv(vp.scene)
    assert hasattr(vp.scene, "show_axes")
    lib = texture_library()
    assert any(t["name"] == "brick_red" for t in lib)
    assert all(Path(t["path"]).is_file() for t in lib)

