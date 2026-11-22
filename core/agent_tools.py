# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
# Copyright (C) 2026 Maher (fork chems/ingetrazo, branch mcp-agent).
"""What the MCP bridge offers an agent beyond ``run_python``.

``run_python`` can reach everything, but a few jobs kept going wrong in real
sessions with an architecture plugin driving IngeTrazo (22-11-2026):

- **Paint** written straight into ``f.attrs`` inside a group was not seen by
  the transaction (``changed: False``, nothing to undo) and did not show
  until something moved a vertex: the group's display cache never heard of
  it. :func:`paint` goes through one command that the history undoes, flags
  the meshes it touched, and says exactly whether anything changed.
- **Lights** for the Blender render live in the Render panel's document data
  (``scene.plugin_data["render_blender"]``); :func:`add_light` and friends
  write it through the same undoable command the panel uses, so the panel
  shows them at once.
- **Files**: saving, opening, IFC and the quantities table, and the DXF
  import, without the dialogs nobody can answer over the bridge.

Everything here takes the scene / history / viewport it works on; nothing
reaches for a window except the few functions that say so. Qt-free where it
can be, so the tests drive it with a bare scene.
"""
from __future__ import annotations

import json
import math
import os
import unicodedata
from pathlib import Path

from core.history import Command

#: The Render panel's document-data key (its plugin file's stem).
RENDER_KEY = "render_blender"

#: DXF units an agent may name, metres per drawing unit.
DXF_UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "ft": 0.3048}


# ---- Names --------------------------------------------------------------------

def norm(text) -> str:
    """Names compared the way a person reads them: NFC, outer spaces off.
    «allège» typed with a combining accent is the same word."""
    return unicodedata.normalize("NFC", str(text or "")).strip()


def walk_groups(groups, depth: int = 0):
    """Every group of a list and the ones nested in them, depth first."""
    for g in groups or ():
        if not hasattr(g, "mesh"):
            continue
        yield g
        if depth < 32:
            yield from walk_groups(getattr(g, "children", None) or (),
                                   depth + 1)


def find_groups(scene, name: str, nested: bool = True) -> list:
    """The groups named ``name`` (exact, NFC), top-level first."""
    want = norm(name)
    pool = walk_groups(scene.groups) if nested else iter(scene.groups)
    return [g for g in pool if norm(getattr(g, "name", "")) == want]


def group_meshes(scene) -> list:
    """The distinct meshes the groups hold (a component's definition once,
    however many placements), the loose mesh left out."""
    seen, out = {id(scene.mesh)}, []
    for g in walk_groups(scene.groups):
        m = getattr(g, "mesh", None)
        if m is not None and id(m) not in seen:
            seen.add(id(m))
            out.append(m)
    return out


def flag_paint(mesh) -> None:
    """Tell the display a mesh's materials changed (not its shape)."""
    if mesh is None:
        return
    mesh._chunk_dirty = True
    mesh._attrs_dirty = True
    mesh._mut_serial = getattr(mesh, "_mut_serial", 0) + 1


# ---- Model overview -------------------------------------------------------------

def model_counts(scene) -> dict:
    """Faces and edges, loose and inside groups. Groups count once per
    distinct mesh (``*_in_groups``) and once per placement
    (``*_placed``: what is drawn, components repeated)."""
    loose_f, loose_e = len(scene.mesh.faces), len(scene.mesh.edges)
    in_f = in_e = 0
    for m in group_meshes(scene):
        in_f += len(m.faces)
        in_e += len(m.edges)
    placed_f = placed_e = 0
    for g in walk_groups(scene.groups):
        m = getattr(g, "mesh", None)
        if m is not None and m is not scene.mesh:
            placed_f += len(m.faces)
            placed_e += len(m.edges)
    return {
        "faces_loose": loose_f, "edges_loose": loose_e,
        "faces_in_groups": in_f, "edges_in_groups": in_e,
        "faces_placed": placed_f, "edges_placed": placed_e,
        "faces_total": loose_f + in_f, "edges_total": loose_e + in_e,
    }


def page(items: list, offset: int = 0, limit: int = 100) -> dict:
    """A slice of a long list, saying how long it is and where the rest
    starts (``next_offset`` None at the end)."""
    total = len(items)
    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 100), 1000))
    chunk = items[offset:offset + limit]
    end = offset + len(chunk)
    return {"items": chunk, "total": total, "offset": offset,
            "truncated": end < total,
            "next_offset": end if end < total else None}


# ---- Paint ------------------------------------------------------------------

_PAINT_KEYS = ("color", "texture", "opacity", "mat")


def texture_library() -> list:
    """``[{name, path, sw, sh, category}]`` — the textures shipped in
    ``resources/textures/library`` (the Materials panel's library)."""
    from core.paths import app_root
    root = app_root() / "resources" / "textures"
    try:
        data = json.loads((root / "library.json").read_text("utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for cat in data.get("categories", []):
        for it in cat.get("items", []):
            out.append({"name": it.get("name"), "category": cat.get("id"),
                        "path": str(root / "library" / it.get("file", "")),
                        "sw": float(it.get("sw", 1.0)),
                        "sh": float(it.get("sh", 1.0))})
    return out


def _color(raw) -> tuple:
    vals = [float(c) for c in raw]
    if len(vals) not in (3, 4):
        raise ValueError("color takes 3 or 4 numbers (r, g, b[, a])")
    if any(v > 1.0 for v in vals[:3]):          # 0–255 given
        vals = [v / 255.0 for v in vals[:3]] + vals[3:]
    if not all(0.0 <= v <= 1.0 and math.isfinite(v) for v in vals):
        raise ValueError(f"color out of range: {raw!r}")
    return tuple(vals[:3])


def resolve_material(spec) -> dict:
    """The face attrs a material spec stands for.

    ``spec``: ``{"color": [r, g, b]}`` (0–1, or 0–255), and/or
    ``{"texture": {"path", "sw", "sh", "rot"}}`` (tile size in metres,
    rotation in degrees) or ``{"library": "brick_red"}`` (a texture of the
    shipped library; ``sw``/``sh``/``rot`` beside it override its tile),
    ``"opacity"`` 0–1 and ``"name"`` (a named material of the document).
    ``None`` / ``{}`` = back to the default material."""
    if spec is None or spec == {}:
        return {}
    if isinstance(spec, (list, tuple)):
        spec = {"color": spec}
    if not isinstance(spec, dict):
        raise ValueError("material must be an object, e.g. "
                         '{"color": [0.8, 0.2, 0.2]}')
    out: dict = {}
    if spec.get("color") is not None:
        out["color"] = _color(spec["color"])
    tex = None
    if spec.get("library"):
        want = norm(spec["library"]).lower()
        for it in texture_library():
            if norm(it["name"]).lower() == want:
                tex = {"path": it["path"], "sw": it["sw"], "sh": it["sh"]}
                break
        if tex is None:
            names = ", ".join(sorted(i["name"] for i in texture_library()))
            raise ValueError(f"no texture {spec['library']!r} in the "
                             f"library. Known: {names}")
        for k in ("sw", "sh", "rot"):
            if spec.get(k) is not None:
                tex[k] = float(spec[k])
    elif spec.get("texture") is not None:
        raw = spec["texture"]
        if isinstance(raw, str):
            raw = {"path": raw}
        tex = {"path": str(raw["path"]),
               "sw": float(raw.get("sw", 1.0)),
               "sh": float(raw.get("sh", 1.0))}
        if raw.get("rot") is not None:
            tex["rot"] = float(raw["rot"])
    if tex is not None:
        from core.texture import is_image_file
        if not Path(tex["path"]).is_file():
            raise ValueError(f"texture image not found: {tex['path']}")
        if not is_image_file(tex["path"]):
            raise ValueError(f"not an image: {tex['path']}")
        if tex["sw"] <= 0 or tex["sh"] <= 0:
            raise ValueError("texture sw/sh are the tile size in metres, > 0")
        if not tex.get("rot"):
            tex.pop("rot", None)
        out["texture"] = tex
    if spec.get("opacity") is not None:
        op = float(spec["opacity"])
        if not 0.0 <= op <= 1.0:
            raise ValueError("opacity is 0..1")
        if op < 1.0:
            out["opacity"] = op
    if spec.get("name"):
        out["mat"] = norm(spec["name"])
    if spec.get("finish") is not None:
        from core.finish import FINISHES
        fin = str(spec["finish"]).lower()
        if fin not in FINISHES:
            raise ValueError(f"finish is one of {FINISHES}")
        if not out.get("mat"):
            raise ValueError('a finish belongs to a NAMED material: give '
                             '"name" too, e.g. {"color": [...], '
                             '"name": "Verre clair", "finish": "glass"}')
        out["_finish"] = fin
    return out


class PaintCommand(Command):
    """Paint faces and/or a group's own material as ONE undo step.

    ``face_attrs``: ``[(face, mesh, new_paint)]`` where ``new_paint`` holds
    the paint keys to set (absent keys are removed — painting replaces the
    old recipe, as the Paint tool does). ``group``/``group_paint``: the
    container material, ``None`` to unpaint. ``material``: a
    :class:`core.materials.Material` to add to the document's registry."""

    def __init__(self, face_attrs=(), group=None, group_paint=None,
                 set_group: bool = False, material=None,
                 finish=None) -> None:
        self.face_attrs = list(face_attrs)
        self.group = group
        self.group_paint = dict(group_paint) if group_paint else None
        self.set_group = set_group
        self.material = material
        #: (name, finish) to set on a material already in the registry.
        self.finish = finish
        self._old_finish = None
        self._old_faces = None
        self._old_group = None
        self._added_mat = None

    @staticmethod
    def _put(face, paint: dict) -> None:
        for k in _PAINT_KEYS:
            face.attrs.pop(k, None)
        for k, v in paint.items():
            face.attrs[k] = dict(v) if isinstance(v, dict) else v

    def _flag(self, scene) -> None:
        for m in {id(m): m for _f, m, _p in self.face_attrs}.values():
            flag_paint(m)
        if self.group is not None:
            flag_paint(self.group.mesh)
        from core.history import _dirty_group_chunks
        _dirty_group_chunks(scene)

    def do(self, scene) -> None:
        if self._old_faces is None:
            self._old_faces = [{k: f.attrs[k] for k in _PAINT_KEYS
                                if k in f.attrs}
                               for f, _m, _p in self.face_attrs]
            self._old_group = getattr(self.group, "material", None)
        for f, _m, paint in self.face_attrs:
            self._put(f, paint)
        if self.set_group and self.group is not None:
            self.group.material = (dict(self.group_paint)
                                   if self.group_paint else None)
        if self.material is not None:
            reg = scene.materials
            if self.material.name not in reg:
                reg[self.material.name] = self.material
                self._added_mat = self.material.name
        if self.finish is not None:
            mat = scene.materials.get(self.finish[0])
            if mat is not None:
                self._old_finish = mat.finish
                mat.finish = self.finish[1]
        self._flag(scene)
        scene.version += 1

    def undo(self, scene) -> None:
        if self.finish is not None:
            mat = scene.materials.get(self.finish[0])
            if mat is not None:
                mat.finish = self._old_finish
        for (f, _m, _p), old in zip(self.face_attrs, self._old_faces or []):
            self._put(f, old)
        if self.set_group and self.group is not None:
            self.group.material = self._old_group
        if self._added_mat is not None:
            scene.materials.pop(self._added_mat, None)
            self._added_mat = None
        self._flag(scene)
        scene.version += 1


def _select_faces(mesh, faces) -> list:
    """The faces of ``mesh`` a ``faces`` argument names: ``"all"``, a list
    of indices, or ``{"normal": [x, y, z], "tol_deg": 10}`` (the faces
    looking that way)."""
    pool = list(mesh.faces)
    if faces in (None, "all"):
        return pool
    if isinstance(faces, dict) and faces.get("normal") is not None:
        from PySide6.QtGui import QVector3D
        n = QVector3D(*[float(c) for c in faces["normal"]]).normalized()
        tol = math.cos(math.radians(float(faces.get("tol_deg", 10.0))))
        return [f for f in pool
                if QVector3D.dotProduct(f.normal().normalized(), n) >= tol]
    if isinstance(faces, (list, tuple)):
        out = []
        for i in faces:
            i = int(i)
            if not 0 <= i < len(pool):
                raise ValueError(f"face index {i} out of range "
                                 f"(0..{len(pool) - 1})")
            out.append(pool[i])
        return out
    raise ValueError('faces is "all", a list of indices, or '
                     '{"normal": [x, y, z], "tol_deg": 10}')


def plan_paint(scene, target, material, faces=None) -> dict:
    """What :func:`paint` would do, without doing it: ``{"command" or None,
    "report"}``. Split out so a test can check the plan."""
    from core.materials import Material, register  # noqa: F401
    paint = resolve_material(material)
    finish = paint.pop("_finish", None)
    reg = scene.materials
    only_name = isinstance(material, dict) and not any(
        material.get(k) is not None for k in
        ("color", "texture", "library", "opacity"))
    if only_name and paint.get("mat") in reg:
        # A named material of the document, applied by its name alone.
        paint = dict(reg[paint["mat"]].face_attrs())
    elif only_name and paint.get("mat"):
        raise ValueError(f"no material named {paint['mat']!r} in the "
                         f"document; give its color/texture too. Known: "
                         f"{sorted(reg)[:40]}")
    if isinstance(target, dict):
        group_name = target.get("group")
        faces = target.get("faces", faces)
    else:
        group_name = target
    if group_name in (None, "", "@loose"):
        group, mesh = None, scene.mesh
    else:
        found = find_groups(scene, group_name)
        if not found:
            names = sorted({norm(g.name) for g in walk_groups(scene.groups)})
            raise ValueError(f"no group named {norm(group_name)!r}. "
                             f"Groups: {names[:60]}")
        group, mesh = found[0], found[0].mesh
    mat = None
    fin_change = None
    if paint.get("mat"):
        mat = Material(paint["mat"], paint.get("color"),
                       paint.get("texture"), paint.get("opacity"),
                       finish=finish)
        if paint["mat"] in scene.materials:
            mat = None                       # already in the registry
            if finish is not None and \
                    scene.materials[paint["mat"]].finish != finish:
                fin_change = (paint["mat"], finish)
    report = {"target": "loose geometry" if group is None
              else norm(group.name), "material": paint}
    if finish is not None:
        report["finish"] = finish
    if group is not None and faces is None:
        # The group as a whole: every face on the default material shows
        # it; faces painted themselves keep their own (the classic rule).
        targets = find_groups(scene, group_name)
        old = [getattr(g, "material", None) or None for g in targets]
        new = paint or None
        if all((o or None) == new for o in old) and mat is None \
                and fin_change is None:
            report.update(changed=False, groups=len(targets), faces=0)
            return {"command": None, "report": report}
        if len(targets) == 1:
            cmd = PaintCommand(group=group, group_paint=new, set_group=True,
                               material=mat, finish=fin_change)
        else:
            from core.history import CompoundCommand
            cmd = CompoundCommand([PaintCommand(group=g, group_paint=new,
                                                set_group=True,
                                                material=mat if i == 0
                                                else None,
                                                finish=fin_change if i == 0
                                                else None)
                                   for i, g in enumerate(targets)])
        own = sum(1 for g in targets for f in g.mesh.faces
                  if f.attrs.get("color") is not None
                  or f.attrs.get("texture") is not None)
        report.update(changed=True, groups=len(targets), faces=0,
                      faces_keeping_own_paint=own)
        return {"command": cmd, "report": report}
    # Faces of EVERY group of that name (four window frames named alike
    # are painted together); a mesh shared by copies is painted once.
    if group is None:
        meshes = [scene.mesh]
    else:
        meshes, seen = [], set()
        for g in find_groups(scene, group_name):
            if id(g.mesh) not in seen:
                seen.add(id(g.mesh))
                meshes.append(g.mesh)
    chosen, todo = [], []
    for m in meshes:
        for f in _select_faces(m, faces):
            chosen.append(f)
            cur = {k: f.attrs[k] for k in _PAINT_KEYS if k in f.attrs}
            if cur != paint:
                todo.append((f, m, dict(paint)))
    report.update(faces=len(chosen), groups=len(meshes),
                  changed=bool(todo or mat or fin_change))
    if not todo and mat is None and fin_change is None:
        return {"command": None, "report": report}
    return {"command": PaintCommand(todo, material=mat, finish=fin_change),
            "report": report}


def paint(viewport, target, material, faces=None) -> dict:
    """Paint through the history (one undo step); see :func:`plan_paint`."""
    plan = plan_paint(viewport.scene, target, material, faces)
    cmd = plan["command"]
    if cmd is not None:
        hist = viewport.history
        hist.execute(cmd)
        if hist.last_error:
            raise RuntimeError(hist.last_error)
    _refresh(viewport)
    return plan["report"]


def _refresh(viewport) -> None:
    notify = getattr(viewport, "notify_scene_changed", None)
    if callable(notify):
        notify()


# ---- Lights for the Blender render ---------------------------------------------

def render_state(scene) -> dict:
    """The Render panel's data: ``{"ambience", "sun_scale", "lights"}``."""
    from core import render_blender as rb
    raw = (getattr(scene, "plugin_data", {}) or {}).get(RENDER_KEY) or {}
    if not isinstance(raw, dict):
        raw = {}
    amb = raw.get("ambience", "day")
    try:
        sun = max(0.1, min(2.0, float(raw.get("sun_scale", 1.0))))
    except (TypeError, ValueError):
        sun = 1.0
    return {"ambience": amb if amb in rb.AMBIENCES else "day",
            "sun_scale": sun, "lights": rb.clean_lights(raw.get("lights", []))}


def _store_render(viewport, st: dict) -> None:
    from core.history import SetPluginDataCommand
    viewport.history.execute(SetPluginDataCommand(RENDER_KEY, {
        "ambience": st["ambience"], "sun_scale": st["sun_scale"],
        "lights": st["lights"]}))
    if viewport.history.last_error:
        raise RuntimeError(viewport.history.last_error)
    _refresh(viewport)


def _vec(raw, what: str) -> list:
    vals = [float(c) for c in raw]
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{what} takes three numbers (x, y, z) in metres")
    return vals


def list_lights(scene) -> dict:
    st = render_state(scene)
    lights = [dict(lt, index=i) for i, lt in enumerate(st["lights"])]
    return {"lights": lights, "ambience": st["ambience"],
            "sun_scale": st["sun_scale"]}


def add_light(viewport, position, kind: str = "point", direction=None,
              target=None, power=None, kelvin=None, angle=None, name="",
              on: bool = True, ambience=None) -> dict:
    """A point light or a spot for the Blender render (Render panel).
    ``position`` in metres; a spot aims along ``direction`` or at
    ``target`` (straight down by default). ``power`` in W, ``kelvin``
    1800–10000 (2700 warm by default), ``angle`` the spot's cone (°)."""
    from core import render_blender as rb
    if kind not in rb.LIGHT_KINDS:
        raise ValueError(f"kind is one of {rb.LIGHT_KINDS}")
    pos = _vec(position, "position")
    if target is not None:
        t = _vec(target, "target")
        d = [t[i] - pos[i] for i in range(3)]
    elif direction is not None:
        d = _vec(direction, "direction")
    else:
        d = [0.0, 0.0, -1.0]
    if math.hypot(*d) < 1e-9:
        raise ValueError("the light's direction is zero")
    st = render_state(viewport.scene)
    light = {"kind": kind, "pos": pos, "dir": d,
             "kelvin": int(kelvin if kelvin is not None else rb.DEFAULT_KELVIN),
             "power": float(power if power is not None
                            else rb.DEFAULT_POWER[kind]),
             "angle": float(angle if angle is not None else 60.0),
             "on": bool(on), "name": norm(name)}
    st["lights"].append(light)
    if ambience is not None:
        if ambience not in rb.AMBIENCES:
            raise ValueError(f"ambience is one of {tuple(rb.AMBIENCES)}")
        st["ambience"] = ambience
    st["lights"] = rb.clean_lights(st["lights"])
    _store_render(viewport, st)
    index = len(st["lights"]) - 1
    return dict(st["lights"][index], index=index)


def remove_light(viewport, index=None, name=None) -> dict:
    st = render_state(viewport.scene)
    lights = st["lights"]
    if name is not None:
        hits = [i for i, lt in enumerate(lights) if norm(lt["name"]) == norm(name)]
        if not hits:
            raise ValueError(f"no light named {norm(name)!r}")
        index = hits[0]
    if index is None:
        raise ValueError("give the light's index or name (see list_lights)")
    index = int(index)
    if not 0 <= index < len(lights):
        raise ValueError(f"light index {index} out of range "
                         f"(0..{len(lights) - 1})")
    gone = lights.pop(index)
    _store_render(viewport, st)
    return {"removed": dict(gone, index=index), "left": len(lights)}


# ---- DXF -----------------------------------------------------------------------

def import_dxf(viewport, path, units: str = "auto", create_faces: bool = False,
               name=None) -> dict:
    """Import a DXF as one undo step, no dialog: ``units`` is ``"auto"``
    (the drawing measured, header checked — the dialog's preselection) or
    one of :data:`DXF_UNITS`. Returns the groups it created."""
    from core.history import SnapshotImport
    from formats.dxf_in import load_dxf, open_document, suggest_unit_scale
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such DXF: {path}")
    doc = open_document(path)
    if units in (None, "", "auto"):
        scale, _code = suggest_unit_scale(doc)
        scale = float(scale or 1.0)
        used = next((k for k, v in DXF_UNITS.items()
                     if abs(v - scale) < 1e-12), f"x{scale:g}")
    else:
        u = str(units).lower()
        if u not in DXF_UNITS:
            raise ValueError(f"units is 'auto' or one of {tuple(DXF_UNITS)}")
        scale, used = DXF_UNITS[u], u
    stats: dict = {}
    scene = viewport.scene

    def load(sc):
        stats.update(load_dxf(sc, path, scale=scale, doc=doc,
                              name=norm(name) if name else path.stem,
                              make_faces=bool(create_faces)))

    cmd = SnapshotImport(load)
    viewport.history.execute(cmd)
    if viewport.history.last_error:
        raise RuntimeError(viewport.history.last_error)
    made = []
    for g in cmd.added_groups:
        made.append({"name": g.name, "layer": getattr(g, "layer", None),
                     "component": getattr(g, "xform", None) is not None,
                     "edges": len(g.mesh.edges), "faces": len(g.mesh.faces)})
    _refresh(viewport)
    lo, hi = scene.bounds()
    return {"units": used, "scale_m_per_unit": scale,
            "groups": made, "group_count": len(made),
            "edges": stats.get("edges", 0), "faces": stats.get("faces", 0),
            "outline_faces": stats.get("outline_faces", 0),
            "skipped": stats.get("skipped", {}),
            "offset": stats.get("offset"),
            "bounds": None if lo is None else
            {"min": [lo.x(), lo.y(), lo.z()], "max": [hi.x(), hi.y(), hi.z()]}}


# ---- Files ---------------------------------------------------------------------

def check_out_path(path, suffix: str) -> Path:
    """An output path: absolute, its folder existing, the suffix right."""
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        raise ValueError(f"give an absolute path, not {path!r}")
    if p.suffix.lower() != suffix:
        p = p.with_suffix(suffix) if p.suffix else Path(str(p) + suffix)
    if not p.parent.is_dir():
        raise FileNotFoundError(f"folder does not exist: {p.parent}")
    return p


def export_ifc(scene, path, project_name: str = "IngeTrazo project") -> dict:
    from core.bim import collect_objects
    from formats.ifc import save_ifc
    p = check_out_path(path, ".ifc")
    if not collect_objects(scene):
        raise ValueError("nothing is tagged for BIM: tag groups first, e.g. "
                         "bim.tag_group(g, 'IfcWall', 'Mur Sud') in run_python")
    n = save_ifc(scene, p, project_name=norm(project_name) or "IngeTrazo project")
    return {"path": str(p), "elements": n, "bytes": p.stat().st_size}


def quantities_csv(scene, path) -> dict:
    from core.bim import quantities_csv as table
    p = check_out_path(path, ".csv")
    text = table(scene)
    p.write_text(text, encoding="utf-8", newline="\n")
    rows = text.count("\n") - 1
    return {"path": str(p), "rows": rows, "bytes": p.stat().st_size,
            "preview": text if len(text) <= 4000 else text[:4000] + "…"}


# ---- Pictures --------------------------------------------------------------------

def fresh_view(viewport) -> None:
    """Forget the viewport's memo of what is placed and visible before a
    picture is taken for an agent: a ``hidden`` flag or a material set by a
    script without a version bump left the memo stale, and the picture kept
    showing what was hidden (the scale figure, 22-11-2026)."""
    for attr in ("_epoch_memo", "_epoch_same", "_billboard_groups",
                 "_epoch_frozen_base"):
        if attr in viewport.__dict__:
            viewport.__dict__[attr] = None
    viewport._tick = getattr(viewport, "_tick", 0) + 1


# ---- Blender render, started here and waited for by the MCP client --------------

class RenderJobs:
    """Blender renders an agent started. :meth:`start` writes the job (the
    model as GLB, the camera, the sun, the Render panel's lights) and
    starts Blender in the background; :meth:`status` says how far it is
    and, once Blender is done, copies the image to where the agent asked.
    The MCP client polls ``status`` until the image is there, so the UI
    thread is never blocked by a render that takes minutes."""

    def __init__(self) -> None:
        self.jobs: dict = {}
        self._next = 1

    @staticmethod
    def blender(override=None):
        """The Blender to use: ``override``, ``$INGETRAZO_BLENDER``, the
        one chosen in the Render panel, then the usual places."""
        from core import render_blender as rb
        saved = override or os.environ.get("INGETRAZO_BLENDER")
        if not saved:
            try:
                from PySide6.QtCore import QSettings
                saved = QSettings().value(rb.SETTINGS_KEY) or None
            except Exception:  # noqa: BLE001 — no settings: search
                saved = None
        return rb.find_blender(str(saved) if saved else None)

    def start(self, viewport, path, scene_name=None, engine="eevee",
              width=1600, height=900, quality=1, ground=True,
              blender=None) -> dict:
        import subprocess
        import time
        from core import render_blender as rb
        out = check_out_path(path, ".png")
        engine = str(engine or "eevee").lower()
        if engine not in rb.ENGINES:
            raise ValueError(f"engine is one of {rb.ENGINES}")
        width = max(16, min(int(width), 8192))
        height = max(16, min(int(height), 8192))
        found = self.blender(blender)
        if found is None:
            raise RuntimeError(
                "Blender not found. Install it (winget install --id "
                "BlenderFoundation.Blender -e) or pick it once in the Render "
                "panel (Render tab ▸ Blender…).")
        scene = viewport.scene
        applied = None
        if scene_name:
            views = [v for v in getattr(scene, "saved_views", []) or []
                     if norm(v.name) == norm(scene_name)]
            if not views:
                names = [norm(v.name) for v in scene.saved_views]
                raise ValueError(f"no scene named {norm(scene_name)!r}. "
                                 f"Scenes: {names}")
            views[0].apply(scene, viewport.camera)
            _refresh(viewport)
            applied = norm(views[0].name)
        st = render_state(scene)
        work = rb.work_base() / time.strftime("agent-%Y%m%d-%H%M%S")
        n = 1
        while work.exists():
            n += 1
            work = work.with_name(work.name.split("~")[0] + f"~{n}")
        job = rb.write_job(scene, viewport.camera, work, engine=engine,
                           quality=max(0, min(2, int(quality))),
                           width=width, height=height, ground=bool(ground),
                           ambience=st["ambience"], lights=st["lights"],
                           sun_scale=st["sun_scale"])
        argv = rb.command(found, job)
        log = open(work / "blender.log", "w", encoding="utf-8",
                   errors="replace")
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=rb.clean_env(),
                                **kw)
        jid = str(self._next)
        self._next += 1
        self.jobs[jid] = {"proc": proc, "log": log, "work": work, "out": out,
                          "started": time.time(), "engine": engine,
                          "scene": applied}
        return {"job": jid, "work": str(work), "blender": found.where,
                "engine": engine, "width": width, "height": height,
                "scene": applied, "lights": sum(1 for lt in st["lights"]
                                                if lt["on"])}

    def status(self, job) -> dict:
        import shutil
        import time
        from core import render_blender as rb
        j = self.jobs.get(str(job))
        if j is None:
            raise ValueError(f"no render job {job!r}")
        code = j["proc"].poll()
        try:
            j["log"].flush()
            lines = (j["work"] / "blender.log").read_text(
                "utf-8", "replace").splitlines()
        except OSError:
            lines = []
        progress = None
        for line in lines:
            p = rb.progress_of(line)
            if p is not None:
                progress = p
        res = {"job": str(job), "elapsed_s": round(time.time() - j["started"], 1),
               "progress": progress}
        if code is None:
            res["state"] = "running"
            return res
        if not j["log"].closed:
            j["log"].close()
        image = j["work"] / "render.png"
        if code == 0 and image.is_file():
            shutil.copyfile(image, j["out"])
            res.update(state="done", path=str(j["out"]),
                       bytes=j["out"].stat().st_size, progress=1.0)
        else:
            tail = [ln for ln in lines[-15:] if ln.strip()]
            res.update(state="failed", exit_code=code, log_tail=tail)
        return res

    def cancel(self, job) -> dict:
        j = self.jobs.get(str(job))
        if j is None:
            raise ValueError(f"no render job {job!r}")
        if j["proc"].poll() is None:
            j["proc"].kill()
        return {"job": str(job), "state": "cancelled"}


# ---- Components and models (lot 2) -------------------------------------------------

def _fold(text) -> str:
    """Lower case without accents: «Lámpara» matches «lampara»."""
    s = unicodedata.normalize("NFKD", str(text or "")).lower()
    return "".join(c for c in s if not unicodedata.combining(c)).strip()


def _cm(entry) -> list:
    try:
        return [float(v) for v in entry.get("cm") or []][:3]
    except (TypeError, ValueError):
        return []


def bundled_components() -> list:
    """The starter components shipped with the app (resources/components)."""
    from core.paths import app_root
    base = app_root() / "resources" / "components"
    try:
        items = json.loads((base / "components.json").read_text("utf-8"))
    except (OSError, ValueError):
        return []
    return [{"id": "bundled:" + it["key"], "name": it.get("name", it["key"]),
             "name_en": it.get("name", it["key"]), "category": "Starter",
             "size_cm": [], "tip": it.get("tip", ""), "licence": "see "
             "resources/components/SOURCES.md", "source": "bundled"}
            for it in items if it.get("key")]


def _library_entries() -> list:
    from core import library
    out = []
    for e in library.index():
        out.append({"id": e["id"], "name": e.get("nombre", ""),
                    "name_en": e.get("nombre_en", ""),
                    "category": e.get("categoria", ""),
                    "size_cm": _cm(e), "licence": e.get("licencia_nombre")
                    or e.get("licencia", ""), "author": e.get("autor", ""),
                    "source": "library", "_entry": e})
    return out


def _size_distance(have: list, want) -> float:
    if not want or not have:
        return 0.0
    d = 0.0
    for h, w in zip(have, list(want) + [None] * 3):
        if w is not None:
            d += abs(float(h) - float(w))
    return d


def search_components(query: str = "", category=None, limit: int = 20,
                      size_cm=None, _entries=None) -> dict:
    """Models of the online library (Sweet Home 3D collections: names in
    Spanish with an English translation, sizes in cm) and the bundled
    starters whose name, English name, category or id contains every word
    of ``query`` (accents and case ignored). ``size_cm`` [w, d, h] (any may
    be null) ranks the closest sizes first."""
    entries = _entries if _entries is not None else (
        bundled_components() + _library_entries())
    words = [w for w in _fold(query).split() if w]
    cat = _fold(category) if category else None
    hits = []
    for e in entries:
        hay = " ".join(_fold(e.get(k, "")) for k in
                       ("name", "name_en", "category", "id"))
        if cat and cat not in _fold(e.get("category", "")):
            continue
        if not all(w in hay for w in words):
            continue
        exact = (_fold(e.get("name")) == _fold(query)
                 or _fold(e.get("name_en")) == _fold(query))
        hits.append((0 if exact else 1, _size_distance(e["size_cm"], size_cm),
                     e))
    hits.sort(key=lambda t: (t[0], t[1], _fold(t[2].get("name"))))
    limit = max(1, min(int(limit or 20), 100))
    shown = [{k: v for k, v in e.items() if not k.startswith("_")}
             for _x, _d, e in hits[:limit]]
    cats = sorted({e.get("category", "") for e in entries if e.get("category")})
    return {"total": len(hits), "results": shown,
            "truncated": len(hits) > limit, "categories": cats}


class ImportStep(Command):
    """Put a ready group in the scene as ONE undo step, with whatever the
    import added to the layers and the material registry: undo takes all
    of it back, redo puts it again."""

    def __init__(self, group, materials_before: dict, layers_before: list):
        self.group = group
        self._mat_before = dict(materials_before)
        self._lay_before = list(layers_before)
        self._mat_after = None
        self._lay_after = None

    def do(self, scene) -> None:
        if self._mat_after is None:
            self._mat_after = dict(scene.materials)
            self._lay_after = list(scene.layers)
        else:
            scene.materials.clear()
            scene.materials.update(self._mat_after)
            scene.layers[:] = self._lay_after
        if self.group not in scene.groups:
            scene.groups.append(self.group)
        scene.version += 1

    def undo(self, scene) -> None:
        if self.group in scene.groups:
            scene.groups.remove(self.group)
        scene.selection.discard(self.group)
        scene.materials.clear()
        scene.materials.update(self._mat_before)
        scene.layers[:] = self._lay_before
        scene.version += 1


def _world_box(group):
    from core.group import iter_placements
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for g, m in iter_placements(group):
        for v in g.mesh.vertices:
            p = m.map(v.position) if m is not None else v.position
            for i, c in enumerate((p.x(), p.y(), p.z())):
                lo[i] = min(lo[i], c)
                hi[i] = max(hi[i], c)
    if lo[0] == float("inf"):
        return None
    return lo, hi


def _pose(group, position, rotation_deg: float = 0.0,
          anchor: str = "base") -> None:
    """Make ``group`` a placed instance: its anchor (the centre of its base,
    or the file's origin) at ``position``, turned ``rotation_deg`` about
    the vertical through the anchor."""
    from PySide6.QtGui import QMatrix4x4, QVector3D
    if getattr(group, "xform", None) is None:
        group.xform = QMatrix4x4()
    if anchor == "origin":
        a = group.xform.map(QVector3D(0.0, 0.0, 0.0))
    else:
        box = _world_box(group)
        a = (QVector3D(0.0, 0.0, 0.0) if box is None else
             QVector3D((box[0][0] + box[1][0]) / 2.0,
                       (box[0][1] + box[1][1]) / 2.0, box[0][2]))
    p = QVector3D(*[float(c) for c in position])
    m = QMatrix4x4()
    m.translate(p)
    m.rotate(float(rotation_deg or 0.0), 0.0, 0.0, 1.0)
    m.translate(-a)
    group.xform = m * group.xform


def _insert(viewport, group, mats: dict, lays: list, name=None, layer=None,
            ifc=None) -> dict:
    """``group`` into the scene as one undo step, named, tagged and on its
    layer. ``mats``/``lays``: the registry and layers BEFORE the import, so
    the undo takes back what it added."""
    from core.group import iter_placements
    scene = viewport.scene
    if name:
        group.name = norm(name)
    if layer:
        from core.layers import Layer
        layer = norm(layer)
        if scene.layer(layer) is None:
            scene.layers.append(Layer(layer))
        group.layer = layer
    if ifc:
        from core import bim
        bim.tag_group(group, ifc, group.name)
    viewport.history.execute(ImportStep(group, mats, lays))
    if viewport.history.last_error:
        raise RuntimeError(viewport.history.last_error)
    _refresh(viewport)
    out = {"name": group.name,
           "faces": sum(len(g.mesh.faces) for g, _m in iter_placements(group)),
           "layer": group.layer,
           "ifc": (getattr(group, "ifc", None) or {}).get("class"),
           "materials_added": sorted(set(scene.materials) - set(mats))}
    box = _world_box(group)
    if box is not None:
        out["bounds"] = {"min": box[0], "max": box[1]}
        out["size_m"] = [round(box[1][i] - box[0][i], 4) for i in range(3)]
    return out


def _shared_copy(viewport, source: str):
    """A new placement of a definition already in the model (the window's
    shared components, the cache the UI's placement fills too), or None."""
    from PySide6.QtGui import QMatrix4x4
    from core.group import copy_group
    window = viewport.window() if hasattr(viewport, "window") else None
    shared = getattr(window, "__dict__", {}).get("_shared_components") \
        if window is not None else None
    known = (shared or {}).get(source)
    if not known:
        return None
    mesh, base = known
    live = next((g for g in viewport.scene.groups
                 if g.mesh is mesh and g.is_component()), None)
    if live is None:
        return None
    group = copy_group(live)
    group.xform = QMatrix4x4(base)
    group.hidden = group.locked = False
    group.material = group.layer = group.ifc = None
    return group


def _remember_shared(viewport, source: str, group) -> None:
    from PySide6.QtGui import QMatrix4x4
    window = viewport.window() if hasattr(viewport, "window") else None
    if window is None or not hasattr(window, "__dict__"):
        return
    shared = window.__dict__.setdefault("_shared_components", {})
    shared[source] = (group.mesh, QMatrix4x4(group.xform))


def _build_library(entry: dict):
    """A library model as a component group at its real size, Z up."""
    from core import library
    from core.group import Group
    from core.scene import Scene as _Scene
    from formats import obj as _obj
    from formats.fuse import soften_smooth_edges
    obj = library.model_file(entry)
    if obj is None:
        raise RuntimeError(f"could not download {entry.get('nombre')!r} "
                           "(no network, or the library is unreachable)")
    temp = _Scene()
    _obj.load_obj(temp, obj, matrix=library.model_matrix(entry, obj))
    if temp.groups and temp.groups[0].children:
        group = temp.groups[0]
        for kid in group.children:
            soften_smooth_edges(kid.mesh, cos_threshold=0.55)
    else:
        mesh = temp.mesh
        if not mesh.faces and temp.groups:
            mesh = temp.groups[0].mesh
        if not mesh.faces:
            raise RuntimeError(f"{entry.get('nombre')!r} has no geometry")
        soften_smooth_edges(mesh, cos_threshold=0.55)
        group = Group(mesh, name=entry.get("nombre", ""))
    group.name = entry.get("nombre", "") or group.name
    return group, temp


def _build_bundled(key: str):
    from core.group import Group
    from core.paths import app_root
    from core.scene import Scene as _Scene
    base = app_root() / "resources" / "components"
    temp = _Scene()
    for ext in ("igz", "glb", "obj"):
        p = base / f"{key}.{ext}"
        if p.exists():
            break
    else:
        raise ValueError(f"no bundled component {key!r}")
    if ext == "igz":
        from formats import igz as _igz
        _igz.load_into(temp, p)
    elif ext == "glb":
        from formats.glb import load_glb
        load_glb(temp, p)
    else:
        from formats import obj as _obj
        _obj.load_obj(temp, p)
    groups = [g for g in temp.groups if not getattr(g, "billboard", False)]
    if not groups:
        raise RuntimeError(f"bundled component {key!r} has no geometry")
    g = groups[0]
    return Group(g.mesh, name=g.name or key.capitalize()), temp


def insert_component(viewport, id=None, query=None, position=(0, 0, 0),
                     rotation=0.0, name=None, layer=None, ifc=None,
                     size_cm=None, category=None) -> dict:
    """Place a library or bundled model at its real size, as a component:
    the centre of its base at ``position`` (m), turned ``rotation``° about
    the vertical. ``id`` from search_components, or ``query`` (+ ``size_cm``,
    ``category``) when it names one model. Copies of a model already in the
    document share its definition."""
    if id is None:
        if not query:
            raise ValueError("give the model's id (search_components) or a "
                             "query")
        found = search_components(query, category=category, limit=12,
                                  size_cm=size_cm)
        res = found["results"]
        if not res:
            raise ValueError(f"no model matches {query!r}. Categories: "
                             f"{found['categories']}")
        exact = [r for r in res if _fold(r["name"]) == _fold(query)
                 or _fold(r["name_en"]) == _fold(query)]
        pick = None
        if size_cm and res:
            pick = res[0]
        elif len(exact) == 1:
            pick = exact[0]
        elif len(res) == 1:
            pick = res[0]
        if pick is None:
            raise ValueError(
                f"{found['total']} models match {query!r}: give id= or "
                "size_cm=. Candidates: " + json.dumps(
                    [{k: r.get(k) for k in ("id", "name", "name_en",
                                            "category", "size_cm")}
                     for r in res], ensure_ascii=False))
        id = pick["id"]
    id = str(id)
    source = ("bundled:" + id.split(":", 1)[1] if id.startswith("bundled:")
              else "library:" + id)
    if hasattr(viewport, "end_group_edit"):
        viewport.end_group_edit()
    mats, lays = dict(viewport.scene.materials), list(viewport.scene.layers)
    info = {"id": id}
    group = _shared_copy(viewport, source)
    shared = group is not None
    if group is None:
        if source.startswith("bundled:"):
            group, temp = _build_bundled(source.split(":", 1)[1])
            info.update(licence="see resources/components/SOURCES.md")
        else:
            from core import library
            entry = next((e for e in library.index() if e["id"] == id), None)
            if entry is None:
                raise ValueError(f"no library model with id {id!r}")
            group, temp = _build_library(entry)
            info.update(licence=entry.get("licencia_nombre")
                        or entry.get("licencia"), author=entry.get("autor"),
                        catalogue_size_cm=_cm(entry))
            from core.insert import merge_materials
            merge_materials(getattr(temp, "materials", {}) or {},
                            viewport.scene.materials, group)
        from PySide6.QtGui import QMatrix4x4
        if group.xform is None:
            group.xform = QMatrix4x4()
        group.component = True
        _remember_shared(viewport, source, group)
    _pose(group, position, rotation, "base")
    out = _insert(viewport, group, mats, lays, name=name, layer=layer,
                  ifc=ifc)
    out.update(info, shared_definition=shared, position=list(position),
               rotation=float(rotation or 0.0))
    return out


#: What import_model reads, by suffix.
MODEL_SUFFIXES = (".glb", ".gltf", ".obj", ".dae", ".skp", ".stl", ".igz")


def import_model(viewport, path, position=None, rotation=0.0, units="auto",
                 name=None, layer=None, ifc=None, anchor="base") -> dict:
    """A 3D model file as ONE component, one undo step: GLB/glTF, OBJ, DAE,
    SKP, STL or IGZ. ``position`` (m) puts its anchor there — the centre of
    its base, or the file's origin with ``anchor="origin"`` — turned
    ``rotation``° about the vertical; without it the model stays where its
    file puts it. ``units`` for OBJ/STL (no unit in the file): auto, m, cm,
    mm, in, ft."""
    from core.insert import import_document_as_component
    from core.scene import Scene as _Scene
    p = Path(os.path.expanduser(str(path)))
    suffix = p.suffix.lower()
    if suffix not in MODEL_SUFFIXES:
        raise ValueError(f"import_model reads {MODEL_SUFFIXES}; DXF goes "
                         "through import_dxf")
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    if anchor not in ("base", "origin"):
        raise ValueError("anchor is 'base' or 'origin'")
    temp = _Scene()
    used_units = None
    offset = None
    if suffix in (".glb", ".gltf"):
        from formats.glb import load_glb
        offset = load_glb(temp, p)
    elif suffix == ".dae":
        from formats.dae import load_dae
        offset = load_dae(temp, p)
    elif suffix == ".skp":
        from formats.skp import load_skp
        load_skp(temp, p)
    elif suffix == ".igz":
        from formats import igz as _igz
        _igz.load_into(temp, p)
    else:
        from formats.obj import OBJ_UNITS, suggest_unit
        u = str(units or "auto").lower()
        if u == "auto":
            u = suggest_unit(p) if suffix == ".obj" else "mm"
        if u not in OBJ_UNITS:
            raise ValueError(f"units is 'auto' or one of {tuple(OBJ_UNITS)}")
        used_units = u
        if suffix == ".obj":
            from formats.obj import load_obj
            load_obj(temp, p, scale=OBJ_UNITS[u], require_geometry=True)
        else:
            from formats.stl import load_stl
            load_stl(temp, p, scale=OBJ_UNITS[u])
    scene = viewport.scene
    mats, lays = dict(scene.materials), list(scene.layers)
    group = import_document_as_component(scene, temp, norm(name) if name
                                         else p.stem)
    if group is None:
        scene.materials.clear()
        scene.materials.update(mats)
        scene.layers[:] = lays
        raise RuntimeError(f"{p.name} holds no geometry")
    if hasattr(viewport, "end_group_edit"):
        viewport.end_group_edit()
    from PySide6.QtGui import QMatrix4x4
    if group.xform is None:
        group.xform = QMatrix4x4()
    group.component = True
    if position is not None:
        _pose(group, position, rotation, anchor)
    out = _insert(viewport, group, mats, lays, name=name, layer=layer,
                  ifc=ifc)
    out.update(format=suffix.lstrip("."), units=used_units,
               survey_offset=offset,
               children=len(getattr(group, "children", []) or []))
    return out
