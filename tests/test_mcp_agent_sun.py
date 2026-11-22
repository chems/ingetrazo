# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
# Copyright (C) 2026 Maher (fork chems/ingetrazo, branch mcp-agent).
"""The sun of a render series (salon d'exercice, 22/11/2026):

- a saved scene remembers its sun and puts it back when recalled, so a sun
  set on the document alone fell back to the scene's noon at the first
  ``render(scene=…)`` — ``set_sun`` now sets the scenes too, and ``render``
  says when a scene changed the sun (or keeps the document's with
  ``keep_sun``);
- without a location the sun is computed for the default site (Arequipa):
  ``set_sun`` takes the real site and ``render`` / ``sun`` say which site
  was used;
- a glass pane stopped the sun in the Blender render: shadow rays now see
  it as transparent (checked here on the generated material nodes when
  Blender's ``bpy`` module is available)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

if QApplication.instance() is None:
    QApplication(sys.argv[:1])

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from test_mcp_agent_tools import (  # noqa: E402,F401
    _ask, _box_group, _in_worker, _ok, bridge, fake_blender, win)


def _scene_with_views(win, names=("Vue A", "Vue B")):
    from core.saved_views import SavedView
    scene = win.viewport.scene
    _box_group(scene, "Volume")
    scene.shadows.hour, scene.shadows.minute = 12, 0
    for n in names:
        scene.saved_views.append(SavedView.capture(n, scene,
                                                   win.viewport.camera))
    return scene


def test_set_sun_sets_the_document_and_every_scene_in_one_undo(win, bridge):
    scene = _scene_with_views(win)
    vp = win.viewport
    depth = len(vp.history.undo_stack)
    res = _ok(_ask(bridge, "set_sun", {"month": 7, "day": 15, "hour": 20,
                                       "minute": 30}))
    assert (scene.shadows.month, scene.shadows.day,
            scene.shadows.hour, scene.shadows.minute) == (7, 15, 20, 30)
    assert res["scenes_set"] == ["Vue A", "Vue B"]
    for v in scene.saved_views:
        assert (v.shadows["hour"], v.shadows["minute"]) == (20, 30)
    # Recalling a scene now keeps the sun that was set.
    scene.saved_views[1].apply(scene, vp.camera)
    assert (scene.shadows.hour, scene.shadows.minute) == (20, 30)
    assert res["time"] == "20:30" and res["date"] == "15/07"
    assert res["site"]["default"] is True and "warning" in res
    assert len(vp.history.undo_stack) == depth + 1
    _ok(_ask(bridge, "undo"))
    assert (scene.shadows.hour, scene.shadows.minute) == (12, 0)
    assert all(v.shadows["hour"] == 12 for v in scene.saved_views)


def test_set_sun_can_leave_the_scenes_alone_or_set_some(win, bridge):
    scene = _scene_with_views(win)
    _ok(_ask(bridge, "set_sun", {"hour": 18, "scenes": "none"}))
    assert scene.shadows.hour == 18
    assert all(v.shadows["hour"] == 12 for v in scene.saved_views)
    res = _ok(_ask(bridge, "set_sun", {"hour": 19, "scenes": ["Vue B"]}))
    assert res["scenes_set"] == ["Vue B"]
    assert [v.shadows["hour"] for v in scene.saved_views] == [12, 19]
    r = _ask(bridge, "set_sun", {"hour": 19, "scenes": ["Inconnue"]})
    assert not r["ok"] and "no scene named" in r["error"]


def test_set_sun_takes_the_site_and_reports_the_sun(win, bridge):
    scene = _scene_with_views(win, names=())
    # Brussels, 21 June, 21:00 summer time: the sun is low in the north-west.
    res = _ok(_ask(bridge, "set_sun", {"lat": 50.85, "lon": 4.35,
                                       "month": 6, "day": 21, "hour": 21,
                                       "minute": 0, "utc_offset": 2}))
    assert abs(scene.georef.lat - 50.85) < 1e-9
    assert res["site"]["default"] is False and "warning" not in res
    assert res["utc_offset"] == 2
    assert 0 < res["elevation_deg"] < 10
    assert 290 < res["azimuth_deg"] < 320
    assert _ok(_ask(bridge, "sun"))["time"] == "21:00"
    _ok(_ask(bridge, "undo"))
    assert scene.georef is None


def test_set_sun_reads_a_time_zone_with_its_summer_time(win, bridge):
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo("Europe/Brussels")
    except Exception:  # noqa: BLE001
        pytest.skip("no time zone data in this Python")
    _scene_with_views(win, names=())
    summer = _ok(_ask(bridge, "set_sun", {"timezone": "Europe/Brussels",
                                          "month": 7, "day": 1}))
    winter = _ok(_ask(bridge, "set_sun", {"timezone": "Europe/Brussels",
                                          "month": 1, "day": 15}))
    assert (summer["utc_offset"], winter["utc_offset"]) == (2, 1)


def test_set_sun_refuses_nonsense(win, bridge):
    _scene_with_views(win, names=())
    for args, words in (({"hour": 25}, "hour 0-23"),
                        ({"lat": 50.8}, "lat and lon together"),
                        ({"lat": 95, "lon": 4}, "lat -90..90"),
                        ({"north_deg": 10}, "north_deg goes with lat/lon"),
                        ({"utc_offset": 20}, "-14..14")):
        r = _ask(bridge, "set_sun", args)
        assert not r["ok"] and words in r["error"], (args, r)


def _render(bridge, tmp_path, **kw):
    import ingetrazo_mcp as client
    args = {"path": str(tmp_path / "r.png"), "engine": "eevee",
            "width": 64, "height": 48, "quality": 0}
    args.update(kw)
    try:
        res = _in_worker(lambda: client._call("render", args))
    finally:
        if client._sock is not None:     # one client at a time on the bridge
            client._sock.close()
            client._sock = None
    assert not res["isError"], res
    return json.loads(res["content"][0]["text"])


def test_render_says_when_a_scene_put_its_own_sun_back(win, bridge, tmp_path,
                                                        fake_blender,
                                                        monkeypatch):
    import ingetrazo_mcp as client
    monkeypatch.setattr(client, "PORT", bridge.port)
    monkeypatch.setattr(client, "RENDER_POLL_S", 0.05)
    monkeypatch.setattr(client, "_sock", None)
    scene = _scene_with_views(win)
    _ok(_ask(bridge, "set_sun", {"hour": 17, "minute": 10,
                                 "scenes": "none"}))
    body = _render(bridge, tmp_path, scene="Vue A")
    started = body["started"]
    assert any("put back the sun it remembers" in w
               for w in started["warnings"])
    assert started["sun"]["time"] == "12:00"
    # keep_sun: the scene's camera, the document's sun.
    _ok(_ask(bridge, "set_sun", {"hour": 17, "minute": 10,
                                 "scenes": "none"}))
    body = _render(bridge, tmp_path, scene="Vue A", keep_sun=True)
    started = body["started"]
    assert started["sun"]["time"] == "17:10"
    assert any("kept the document's sun" in w for w in started["warnings"])
    assert (scene.shadows.hour, scene.shadows.minute) == (17, 10)
    # Default site said in the render result too.
    assert any("default site" in w for w in started["warnings"])


def test_render_is_quiet_once_the_scenes_share_the_sun(win, bridge, tmp_path,
                                                       fake_blender,
                                                       monkeypatch):
    import ingetrazo_mcp as client
    monkeypatch.setattr(client, "PORT", bridge.port)
    monkeypatch.setattr(client, "RENDER_POLL_S", 0.05)
    monkeypatch.setattr(client, "_sock", None)
    _scene_with_views(win)
    _ok(_ask(bridge, "set_sun", {"lat": 50.85, "lon": 4.35, "hour": 19,
                                 "utc_offset": 1}))
    body = _render(bridge, tmp_path, scene="Vue B")
    assert "warnings" not in body["started"]
    assert body["started"]["sun"]["site"]["default"] is False


# ---- Glass lets the sun through (Blender material nodes) -----------------------

def _render_scene_ns():
    src = (Path(__file__).resolve().parent.parent / "resources" / "blender"
           / "render_scene.py").read_text(encoding="utf-8")
    ns = {"__name__": "render_scene_test"}
    exec(compile(src.rsplit("\nmain()", 1)[0], "render_scene.py", "exec"), ns)
    return ns


def test_glass_shows_shadow_rays_a_transparent_surface():
    bpy = pytest.importorskip("bpy")
    ns = _render_scene_ns()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    glass = bpy.data.materials.new("Vidrio claro")
    glass.use_nodes = True
    glass["ingetrazo_finish"] = "glass"
    wood = bpy.data.materials.new("Roble")
    wood.use_nodes = True
    wood["ingetrazo_finish"] = "satin"
    ns["apply_finishes"]()
    nodes = glass.node_tree.nodes
    out = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    mix = out.inputs["Surface"].links[0].from_node
    assert mix.type == "MIX_SHADER"
    assert mix.inputs["Fac"].links[0].from_socket.name == "Is Shadow Ray"
    assert mix.inputs[2].links[0].from_node.type == "BSDF_TRANSPARENT"
    assert mix.inputs[1].links[0].from_node.type == "BSDF_PRINCIPLED"
    if hasattr(glass, "use_transparent_shadow"):
        assert glass.use_transparent_shadow is True
    # Other finishes keep their plain surface.
    out_w = next(n for n in wood.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
    assert out_w.inputs["Surface"].links[0].from_node.type == "BSDF_PRINCIPLED"
