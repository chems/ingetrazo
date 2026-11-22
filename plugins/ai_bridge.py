# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""AI Bridge plugin — let an AI coding agent drive the live document (MCP).

The realisation of invariant #5 (AI-native): the agent generates RECIPES of
actions over the deterministic engine, never raw meshes. The «AI» tab of
the side tray (section "AI bridge (MCP)", or Extensions ▸ AI Bridge (MCP))
starts a localhost-only TCP server; the companion
``scripts/ingetrazo_mcp.py`` bridges it to Claude Code / Claude Desktop as
an MCP server. The usual pattern for app MCP bridges (a TCP server inside
the app + an MCP process outside), with three legs up:

- ``run_python`` executes through the Python Console's transactional
  machinery: every AI action is ONE undo step, and a script that raises is
  rolled back whole — the agent can never leave the document half-mutated.
- ``screenshot`` renders the real viewport (``render_image``), closing the
  "the agent can't see what it built" gap: describe → build → LOOK → fix.
- The hermeticity guard stays in the loop: recipes that would commit a
  broken solid are refused by the engine itself.

Protocol (framed for the bridge, not for humans): newline-delimited JSON on
127.0.0.1:4763 (``INGETRAZO_AI_PORT`` overrides). Request
``{"id": n, "tool": name, "args": {...}}`` → reply ``{"id": n, "ok": bool,
"result": ... | "error": str}``. One client at a time; everything the tools
touch runs on the Qt MAIN thread (queued-signal relay to a bound method —
the documented PySide6 threading gotcha).
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QVBoxLayout, QWidget)

from core.i18n import tr
from views.fold_section import FoldSection, narrow

DEFAULT_PORT = 4763
_REFUSED = b'{"id": null, "ok": false, "error": "not a bridge request; closing"}\n'


class _NoDialogs(QObject):
    """While an agent's code runs, nobody is at the screen to answer a
    dialog it opens (#694): «Delete sheet» asks Yes/No, a sheet the
    document cannot lose says so in a box. ``exec()`` then waited on the UI
    thread for good — the agent got «timed out», the app looked hung, and
    the next call ran INSIDE that dialog's event loop, nested in the first
    call's undo step. Every dialog shown meanwhile is answered «Cancel» at
    once (as Escape would) and named in the reply, so the agent knows."""

    def __init__(self) -> None:
        super().__init__()
        self.refused: list[str] = []

    def eventFilter(self, obj, event):  # noqa: N802 — Qt override
        if event.type() == QEvent.Show:
            from PySide6.QtWidgets import QDialog, QMessageBox
            if isinstance(obj, QDialog):
                text = obj.windowTitle()
                if isinstance(obj, QMessageBox) and obj.text():
                    text = f"{text}: {obj.text()}" if text else obj.text()
                self.refused.append(text or type(obj).__name__)
                # Not from inside the Show: exec() only enters its loop
                # after show() returns, and a reject before that is lost.
                QTimer.singleShot(0, obj, obj.reject)
        return False


class _Bridge(QObject):
    """The in-app half: accepts one agent connection and executes its tool
    calls on the Qt main thread."""

    # Signal(object), NOT Signal(dict): a dict payload on a queued
    # connection may be marshalled into a QVariantMap — the slot then gets a
    # COPY, sets the copied Event, and the worker times out forever. object
    # passes the PyObject by reference. (Cost us a hunt; now documented.)
    _dispatch = Signal(object)

    @property
    def _viewport(self):
        """The viewport of the project in front (projects in tabs, #1066):
        the one handed in at setup may be a tab in the back, or closed."""
        return self._live_viewport()

    def __init__(self, viewport) -> None:
        # The window's, not the viewport's: a project tab closed must not
        # take the bridge with it (#1066).
        super().__init__(viewport.window() or viewport)
        from views.documents import live_viewport
        self._live_viewport = live_viewport(viewport)
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._scope: dict = {"__name__": "__ai__"}
        self.port: int | None = None
        self._busy = False
        # Queued to a BOUND method of this main-thread QObject — connecting a
        # lambda would run the slot on the worker thread (CLAUDE.md gotcha).
        self._dispatch.connect(self._run_on_main, Qt.QueuedConnection)

    # ---- Lifecycle ----------------------------------------------------------
    def start(self) -> int:
        port = int(os.environ.get("INGETRAZO_AI_PORT", DEFAULT_PORT))
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(1)
        srv.settimeout(0.5)
        self._server = srv
        self.port = srv.getsockname()[1]
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._serve, name="ingetrazo-ai-bridge", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        self.port = None

    @property
    def running(self) -> bool:
        return self._server is not None

    # ---- Worker side ---------------------------------------------------------
    def _serve(self) -> None:
        srv = self._server
        while not self._stop.is_set() and srv is not None:
            try:
                conn, _addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with conn:
                conn.settimeout(0.5)
                buf = b""
                while not self._stop.is_set():
                    try:
                        chunk = conn.recv(65536)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    if not self._drain(conn, buf):
                        break
                    buf = buf[buf.rfind(b"\n") + 1:]

    def _drain(self, conn, buf: bytes) -> bool:
        """Answer every complete line in ``buf``. False when the connection
        must close: a line that is not a JSON request ends it, and nothing
        after it runs.

        Loopback is not a wall against the browser on this machine. A web page
        can ``fetch("http://127.0.0.1:4763", {method: "POST", mode: "no-cors",
        body})`` — a text/plain POST needs no CORS preflight, so the browser
        sends it and only hides the reply. Read line by line, its request line
        and headers failed as bad JSON and its body, a tool call, ran. The
        first line of any HTTP request is ``POST / HTTP/1.1``, so refusing the
        connection there stops the body from ever being read. The MCP client
        sends nothing but JSON lines and never meets this."""
        lines = buf.split(b"\n")[:-1]
        for line in lines:
            if not line.strip():
                continue
            reply = self._handle_line(line)
            try:
                conn.sendall(reply if reply is not None else _REFUSED)
            except OSError:
                return False
            if reply is None:
                return False
        return True

    def _handle_line(self, line: bytes) -> bytes | None:
        """The reply to one request line, or None if it is not a request."""
        try:
            req = json.loads(line)
        except ValueError:
            return None
        if not isinstance(req, dict):
            return None
        job = {"req": req, "done": threading.Event(), "reply": None}
        self._dispatch.emit(job)
        job["done"].wait(timeout=120.0)
        reply = job["reply"] or {"id": req.get("id"), "ok": False,
                                 "error": "timed out on the UI thread"}
        return (json.dumps(reply) + "\n").encode()

    # ---- Main-thread side ----------------------------------------------------
    def _run_on_main(self, job: dict) -> None:
        req = job["req"]
        tool = req.get("tool", "")
        args = req.get("args") or {}
        if self._busy:
            # A call delivered by an event loop nested in the previous one
            # (something it ran is still waiting): never run inside it.
            job["reply"] = {"id": req.get("id"), "ok": False, "error":
                            "IngeTrazo is still running the previous call"}
            job["done"].set()
            return
        self._busy = True
        from PySide6.QtWidgets import QApplication
        guard = _NoDialogs()
        QApplication.instance().installEventFilter(guard)
        try:
            handler = getattr(self, f"_tool_{tool}", None)
            if handler is None:
                raise ValueError(f"unknown tool {tool!r}")
            result = handler(**args)
            if guard.refused and isinstance(result, dict):
                result["dialogs_cancelled"] = list(guard.refused)
                result["stderr"] = (result.get("stderr") or "") + "".join(
                    f"A dialog nobody can answer over the bridge was "
                    f"cancelled: {t!r}\n" for t in guard.refused)
            job["reply"] = {"id": req.get("id"), "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001 — reported to the agent
            job["reply"] = {"id": req.get("id"), "ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}
        finally:
            QApplication.instance().removeEventFilter(guard)
            self._busy = False
            job["done"].set()

    # ---- Tools ---------------------------------------------------------------
    def _tool_run_python(self, code: str = "") -> dict:
        """Execute ``code`` via the shared transactional executor (core.ai):
        one undoable step, whole-rollback on error, no undo entry when
        nothing changed."""
        from core.ai import run_transactional
        return run_transactional(self._viewport, code, self._scope)

    def _tool_query_model(self, offset: int = 0, limit: int = 100) -> dict:
        """Counts include the faces and edges INSIDE groups (the loose mesh
        alone read 0 on a model built of groups); long lists come in pages
        that say they are cut (``*_total``, ``*_next_offset``)."""
        from core.agent_tools import model_counts, page, render_state
        vp = self._viewport
        scene = vp.scene
        lo, hi = scene.bounds()
        insts = sum(1 for g in scene.groups
                    if getattr(g, "xform", None) is not None)
        counts = model_counts(scene)
        groups = page([g.name for g in scene.groups], offset, limit)
        mats = page(sorted(getattr(scene, "materials", {}) or {}), 0, limit)
        doc = getattr(getattr(vp.window(), "_doc", None), "path", None)
        out = {
            # «faces»/«edges» keep their old meaning (the loose mesh) for
            # callers that read them; the totals say what the model holds.
            "faces": counts["faces_loose"],
            "edges": counts["edges_loose"],
            **counts,
            "groups": len(scene.groups) - insts,
            "component_instances": insts,
            "groups_total": groups["total"],
            "group_names": groups["items"],
            "group_names_offset": groups["offset"],
            "group_names_truncated": groups["truncated"],
            "group_names_next_offset": groups["next_offset"],
            "materials": mats["items"],
            "materials_total": mats["total"],
            "materials_truncated": mats["truncated"],
            "layers": [ly.name for ly in scene.layers],
            "scenes": [v.name for v in getattr(scene, "saved_views", []) or []],
            "render_lights": len(render_state(scene)["lights"]),
            "dimensions": len(getattr(scene, "dimensions", []) or []),
            "section_planes": len(getattr(scene, "section_planes", []) or []),
            "selection": len(scene.selection),
            "document": None if doc is None else str(doc),
            "bounds": (None if lo is None else
                       {"min": [lo.x(), lo.y(), lo.z()],
                        "max": [hi.x(), hi.y(), hi.z()]}),
        }
        return out

    def _tool_screenshot(self, width: int = 1024, height: int = 768) -> dict:
        from core.agent_tools import fresh_view
        width = max(64, min(int(width), 4096))
        height = max(64, min(int(height), 4096))
        vp = self._viewport
        fresh_view(vp)
        image = vp.render_image(width, height)
        if image is None:
            raise RuntimeError("the viewport has no OpenGL context yet "
                               "(window minimised or not shown)")
        out_dir = Path(tempfile.gettempdir()) / "ingetrazo-ai"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "viewport.png"
        image.save(str(out))
        return {"path": str(out), "width": width, "height": height}

    # ---- Files (no dialog: the bridge has nobody to answer one) ------------
    def _window(self):
        win = self._viewport.window()
        if not hasattr(win, "open_path"):
            raise RuntimeError("no IngeTrazo main window around the viewport")
        return win

    def _tool_save(self, path: str) -> dict:
        """Save the project in front as .igz at ``path`` — what File ▸ Save
        As does after its dialog: camera and views stored, a .bak of the
        previous file kept, the tab renamed, the document clean."""
        import shutil
        from core import autosave
        from core.agent_tools import check_out_path
        from formats import igz as igz_format
        win = self._window()
        p = check_out_path(path, ".igz")
        win._end_group_edits()
        doc = win._doc
        vp = doc.viewport
        if p.is_file():
            try:
                shutil.copy2(p, p.with_name(p.name + ".bak"))
            except OSError:
                pass
        vp.scene.camera_home = win._camera_dict(vp)
        vp.scene.views_home = win._views_dict(doc)
        stats = igz_format.save_scene(vp.scene, p) or {}
        autosave.clear(p)
        autosave.clear(doc.path, untitled=doc.untitled_no)
        doc.path = p
        win._remember_recent(p)
        win._saved_version = vp.scene.version
        win._update_title()
        return {"path": str(p), "bytes": p.stat().st_size,
                "textures_embedded": int(stats.get("embedded", 0)),
                "textures_missing": int(stats.get("missing", 0))}

    def _tool_open(self, path: str, recover=None) -> dict:
        """Open an .igz in its own tab (or in the untouched one in front).
        A DXF goes through ``import_dxf``. An auto-saved copy waiting for
        this file is never thrown away silently: say ``recover`` true or
        false."""
        from core import autosave
        win = self._window()
        p = Path(os.path.expanduser(str(path)))
        if p.suffix.lower() in (".dxf", ".dwg"):
            raise ValueError("use import_dxf for a DXF (open asks for units "
                             "in a dialog)")
        if p.suffix.lower() != ".igz":
            raise ValueError("open takes an .igz document")
        if not p.is_file():
            raise FileNotFoundError(f"no such file: {p}")
        # A document already open in a tab is only brought to the front:
        # its own session's recovery slot is no copy to choose (seen in the
        # acceptance run, 22-11-2026, right after a save).
        already = any(d.same_file(p) for d in getattr(win, "_documents", ()))
        pending = not already and autosave.pending(p) is not None
        if pending and recover is None:
            raise RuntimeError(
                "an auto-saved copy of this document exists (a session ended "
                "without saving it). Call open again with recover=true to "
                "take the auto-saved copy, or recover=false to open the file "
                "on disk and discard the copy.")
        import views.main_window as mw
        real = mw.QMessageBox.question
        answer = mw.QMessageBox.Yes if recover else mw.QMessageBox.No
        mw.QMessageBox.question = staticmethod(lambda *a, **k: answer)
        try:
            ok = win.open_path(p)
        finally:
            mw.QMessageBox.question = real
        if not ok:
            raise RuntimeError(f"IngeTrazo could not open {p}")
        out = self._tool_query_model()
        out["opened"] = str(p)
        out["recovered"] = bool(pending and recover)
        return out

    def _tool_export_ifc(self, path: str,
                         project_name: str = "IngeTrazo project") -> dict:
        from core.agent_tools import export_ifc
        self._viewport.end_group_edit()
        return export_ifc(self._viewport.scene, path, project_name)

    def _tool_quantities_csv(self, path: str) -> dict:
        from core.agent_tools import quantities_csv
        self._viewport.end_group_edit()
        return quantities_csv(self._viewport.scene, path)

    def _tool_import_dxf(self, path: str, units: str = "auto",
                         create_faces: bool = False, name=None) -> dict:
        from core.agent_tools import import_dxf
        vp = self._viewport
        vp.end_group_edit()
        out = import_dxf(vp, path, units=units, create_faces=create_faces,
                         name=name)
        try:
            self._window()._on_zoom_extents()
        except Exception:  # noqa: BLE001 — framing is a nicety
            pass
        return out

    # ---- Paint and render lights -------------------------------------------
    def _tool_paint(self, target=None, material=None, faces=None) -> dict:
        from core.agent_tools import paint
        return paint(self._viewport, target, material, faces)

    def _tool_add_light(self, position, kind: str = "point", direction=None,
                        target=None, power=None, kelvin=None, angle=None,
                        name: str = "", on: bool = True,
                        ambience=None) -> dict:
        from core.agent_tools import add_light
        return add_light(self._viewport, position, kind=kind,
                         direction=direction, target=target, power=power,
                         kelvin=kelvin, angle=angle, name=name, on=on,
                         ambience=ambience)

    def _tool_list_lights(self) -> dict:
        from core.agent_tools import list_lights
        return list_lights(self._viewport.scene)

    def _tool_remove_light(self, index=None, name=None) -> dict:
        from core.agent_tools import remove_light
        return remove_light(self._viewport, index=index, name=name)

    def _tool_set_sun(self, month=None, day=None, hour=None, minute=None,
                      utc_offset=None, timezone=None, lat=None, lon=None,
                      north_deg=None, scenes="all") -> dict:
        from core.agent_tools import set_sun
        return set_sun(self._viewport, month=month, day=day, hour=hour,
                       minute=minute, utc_offset=utc_offset,
                       timezone=timezone, lat=lat, lon=lon,
                       north_deg=north_deg, scenes=scenes)

    def _tool_sun(self) -> dict:
        from core.agent_tools import sun_report
        return sun_report(self._viewport.scene)

    # ---- Components and 3D models -------------------------------------------
    def _tool_search_components(self, query: str = "", category=None,
                                limit: int = 20, size_cm=None,
                                preview: bool = False,
                                sheet_path=None) -> dict:
        from core.agent_tools import search_components
        return search_components(query, category=category, limit=limit,
                                 size_cm=size_cm, preview=preview,
                                 sheet_path=sheet_path)

    def _tool_insert_component(self, id=None, query=None,
                               position=(0, 0, 0), rotation=0.0, name=None,
                               layer=None, ifc=None, size_cm=None,
                               category=None) -> dict:
        from core.agent_tools import insert_component
        return insert_component(self._viewport, id=id, query=query,
                                position=position, rotation=rotation,
                                name=name, layer=layer, ifc=ifc,
                                size_cm=size_cm, category=category)

    def _tool_import_model(self, path: str, position=None, rotation=0.0,
                           units: str = "auto", name=None, layer=None,
                           ifc=None, anchor: str = "base") -> dict:
        from core.agent_tools import import_model
        return import_model(self._viewport, path, position=position,
                            rotation=rotation, units=units, name=name,
                            layer=layer, ifc=ifc, anchor=anchor)

    # ---- Blender render (started here, waited for by the MCP client) --------
    def _renders(self):
        jobs = getattr(self, "_render_jobs", None)
        if jobs is None:
            from core.agent_tools import RenderJobs
            jobs = self._render_jobs = RenderJobs()
        return jobs

    def _tool_render_start(self, path: str, scene=None, engine="eevee",
                           width=1600, height=900, quality=1, ground=True,
                           blender=None, keep_sun=False) -> dict:
        from core.agent_tools import fresh_view
        vp = self._viewport
        vp.end_group_edit()
        fresh_view(vp)
        return self._renders().start(vp, path, scene_name=scene,
                                     engine=engine, width=width,
                                     height=height, quality=quality,
                                     ground=ground, blender=blender,
                                     keep_sun=bool(keep_sun))

    def _tool_render_status(self, job) -> dict:
        return self._renders().status(job)

    def _tool_render_cancel(self, job) -> dict:
        return self._renders().cancel(job)

    def _tool_undo(self) -> dict:
        ok = self._viewport.history.undo()
        # notify, not only update: the panels that follow the document
        # (Render lights, Scenes…) hear of it at once.
        self._viewport.notify_scene_changed()
        return {"ok": bool(ok)}

    def _tool_redo(self) -> dict:
        ok = self._viewport.history.redo()
        self._viewport.notify_scene_changed()
        return {"ok": bool(ok)}


def mcp_command(platform: str | None = None, frozen: bool | None = None,
                executable: str | None = None, root: Path | None = None,
                env: dict | None = None) -> list[str]:
    """The command an MCP client must run to reach this IngeTrazo — the
    packaged app carries the server, so nobody needs Python installed:
    ``ingetrazo-mcp.exe`` beside the app on Windows, ``<ingetrazo> --mcp``
    for the Linux/macOS packages, and the script itself from a checkout.

    The Linux packages need the command that is valid FROM THE HOST and
    OUTLIVES this run, not ``sys.executable``: an AppImage mounts itself
    under a fresh ``/tmp/.mount_*`` each launch (the path the dialog used
    to print died with the session), a Flatpak's executable lives inside
    the sandbox (``flatpak run <id> --mcp`` is the door), and a snap is
    reached through ``/snap/bin``. Marco found the gap setting Antigravity
    up for the tutorial (2026-09-21)."""
    from pathlib import PurePosixPath, PureWindowsPath
    platform = platform or sys.platform
    frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    executable = executable or sys.executable
    env = os.environ if env is None else env
    windows = platform.startswith("win")
    PathOf = PureWindowsPath if windows else PurePosixPath   # host-agnostic
    if not windows:
        if env.get("APPIMAGE"):
            return [env["APPIMAGE"], "--mcp"]
        if env.get("FLATPAK_ID"):
            return ["flatpak", "run", env["FLATPAK_ID"], "--mcp"]
        if env.get("SNAP_NAME"):
            return [f"/snap/bin/{env['SNAP_NAME']}", "--mcp"]
    if frozen:
        exe = PathOf(executable)
        if windows:
            return [str(exe.with_name("ingetrazo-mcp.exe"))]
        return [str(exe), "--mcp"]
    root = PathOf(str(root or Path(__file__).resolve().parents[1]))
    python = "python" if windows else "python3"
    return [python, str(root / "scripts" / "ingetrazo_mcp.py")]


def desktop_config_path(platform: str | None = None) -> str:
    """Where Claude Desktop reads its MCP servers on this platform."""
    platform = platform or sys.platform
    if platform.startswith("win"):
        return r"%APPDATA%\\Claude\\claude_desktop_config.json"
    if platform == "darwin":
        return "~/Library/Application Support/Claude/claude_desktop_config.json"
    return "~/.config/Claude/claude_desktop_config.json"


#: Where the common MCP clients read a ``mcpServers`` block like the one the
#: dialog prints (the same JSON, or the equivalent, works in every one:
#: MCP is an open standard, not a Claude feature).
OTHER_CLIENTS = (
    ("Cursor", "~/.cursor/mcp.json"),
    ("VS Code (Copilot)", ".vscode/mcp.json  (key \"servers\")"),
    ("Windsurf", "~/.codeium/windsurf/mcp_config.json"),
    # Google retired Gemini CLI's free Google-account login on 2026-06-18;
    # its successor for individuals is Antigravity CLI (`agy`), which reads
    # this file (Marco hit the shutdown notice on 2026-09-21).
    ("Antigravity CLI (Google)", "~/.gemini/config/mcp_config.json"),
    ("Codex CLI", "~/.codex/config.toml  ([mcp_servers.ingetrazo])"),
)


def connect_instructions(port: int, platform: str | None = None, **kw) -> str:
    """Copy-and-paste text for the dialog the Extensions entry shows once
    the bridge is up: Claude Code's one-liner, the JSON every other MCP
    client takes, and where each of them reads it."""
    cmd = mcp_command(platform, **kw)
    quoted = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    config = {"mcpServers": {"ingetrazo": {"command": cmd[0], "args": cmd[1:]}}}
    others = "\n".join(f"    {name}: {path}" for name, path in OTHER_CLIENTS)
    return (
        tr("The AI bridge is listening on 127.0.0.1:{port}. Connect ANY MCP "
           "client — MCP is an open standard: Claude Code, Claude Desktop, "
           "Cursor, VS Code, Windsurf, Antigravity CLI, Codex CLI…", port=port)
        + "\n\n"
        + tr("Claude Code (in a terminal):") + "\n"
        + f"    claude mcp add ingetrazo -- {quoted}\n\n"
        + tr("Antigravity CLI (Google; in a terminal):") + "\n"
        + f"    agy mcp add ingetrazo -- {quoted}\n\n"
        + tr("Claude Desktop: add this to {path} and restart Claude Desktop:",
             path=desktop_config_path(platform)) + "\n"
        + json.dumps(config, indent=2) + "\n\n"
        + tr("Other clients take the same block in their own file:") + "\n"
        + others + "\n\n"
        + tr("Keep IngeTrazo open with the bridge on; the tools answer only while it runs.")
    )


def _bridge_of(viewport) -> _Bridge:
    window = viewport.window()
    bridge = getattr(window, "_ai_bridge", None)
    if bridge is None:
        bridge = _Bridge(viewport)
        window._ai_bridge = bridge
    return bridge


class BridgeSection(QWidget):
    """The bridge's part of the «AI» tab: on/off, where it listens, and
    the exact lines to paste into the MCP client — the user who reached
    this is rarely the one who knows them by heart."""

    @property
    def _viewport(self):
        """The viewport of the project in front (projects in tabs, #1066):
        the one handed in at setup may be a tab in the back, or closed."""
        return self._live_viewport()

    def __init__(self, viewport, parent=None) -> None:
        super().__init__(parent)
        from views.documents import live_viewport
        self._live_viewport = live_viewport(viewport)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 4, 4)
        self.section = FoldSection(tr("AI bridge (MCP)"), "ia/open_bridge",
                                   default_open=False)
        lay.addWidget(self.section)
        body = QVBoxLayout(self.section.body)
        self._status = QLabel()
        self._status.setWordWrap(True)
        body.addWidget(self._status)
        row = QHBoxLayout()
        self._toggle = QPushButton()
        self._toggle.clicked.connect(self.toggle)
        row.addWidget(self._toggle)
        self._copy = QPushButton(tr("Copy"))
        self._copy.setToolTip(tr("Copy the connection instructions"))
        self._copy.clicked.connect(self._on_copy)
        row.addWidget(self._copy)
        body.addLayout(row)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self._text.setMinimumHeight(140)
        body.addWidget(self._text)
        narrow(self._toggle, self._copy)
        self._refresh()

    @property
    def running(self) -> bool:
        bridge = getattr(self._viewport.window(), "_ai_bridge", None)
        return bridge is not None and bridge.running

    def start(self) -> bool:
        bridge = _bridge_of(self._viewport)
        if bridge.running:
            return True
        try:
            port = bridge.start()
        except OSError as exc:
            self._viewport.flash_status(tr(
                "AI bridge could not start: {err}", err=str(exc)))
            self._refresh(error=str(exc))
            return False
        self._port = port
        self._viewport.flash_status(tr(
            "AI bridge listening on 127.0.0.1:{port}", port=port), 8000)
        self._refresh()
        return True

    def stop(self) -> None:
        bridge = getattr(self._viewport.window(), "_ai_bridge", None)
        if bridge is not None and bridge.running:
            bridge.stop()
            self._viewport.flash_status(tr("AI bridge stopped"))
        self._refresh()

    def toggle(self) -> None:
        if self.running:
            self.stop()
        else:
            self.start()

    def _refresh(self, error: str | None = None) -> None:
        on = self.running
        self._toggle.setText(tr("Stop bridge") if on else tr("Start bridge"))
        self._copy.setVisible(on)
        self._text.setVisible(on)
        if on:
            port = getattr(self, "_port", DEFAULT_PORT)
            self._status.setText(tr(
                "Listening on 127.0.0.1:{port} — connect your MCP client "
                "with the lines below.", port=port))
            self._text.setPlainText(connect_instructions(port))
        elif error:
            self._status.setText(tr(
                "AI bridge could not start: {err}", err=error))
        else:
            self._status.setText(tr(
                "Off. Start it to let an AI agent (Claude, Cursor, "
                "Antigravity…) model in this document over MCP."))

    def _on_copy(self) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._text.toPlainText())
        self._viewport.flash_status(tr("Copied"))


def setup(app) -> None:
    """The bridge joins the assistant in the side tray's «AI» tab; the
    Extensions entry opens that section and starts the bridge."""
    section = BridgeSection(app.viewport)
    dock = app.add_panel(tr("AI"), section, panel="ai", icon="panel_ai")
    app.window._ai_bridge_section = section

    def summon() -> None:
        app.show_panel(dock)
        section.section.set_open(True)
        section.start()

    app.add_menu_action(tr("AI Bridge (MCP)"), summon, tip=tr(
        "Start the local bridge through which an AI agent (MCP) drives "
        "the open document."))
