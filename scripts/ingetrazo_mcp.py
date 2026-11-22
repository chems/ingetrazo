#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""IngeTrazo MCP server — connect Claude (Code / Desktop) to the live app.

Stdlib-only stdio MCP server: it bridges the Model Context Protocol to the
AI Bridge plugin's localhost TCP port (start it first: IngeTrazo ▸
Extensiones ▸ "AI Bridge (MCP)").

Register it with Claude Code:

    claude mcp add ingetrazo -- python3 /path/to/app/scripts/ingetrazo_mcp.py

Tools: run_python (transactional, one undo step per call), query_model,
screenshot (the agent SEES the viewport), undo, redo — and, in the
chems/ingetrazo «mcp-agent» fork: save, open, export_ifc, quantities_csv,
render (Blender, waited for), import_dxf, paint, add_light, list_lights,
remove_light, search_components, insert_component, import_model. Every mutation goes through IngeTrazo's command engine — the
hermeticity guard validates the agent's recipes, and Ctrl+Z always works.

Text is UTF-8 both ways, whatever the console code page: on Windows a
piped stdin was decoded as cp1252, and «allège» reached the app as
«allÃ¨ge».
"""
from __future__ import annotations

import base64
import io
import json
import os
import socket
import sys
import time
from pathlib import Path

PORT = int(os.environ.get("INGETRAZO_AI_PORT", 4763))
#: Where the bridge is. It only ever listens on the app machine's loopback
#: (no authentication, and run_python runs code in the app); a client in
#: a container or on another machine reaches it through a tunnel the user
#: sets up, and names the tunnel's end here (issue #130).
HOST = os.environ.get("INGETRAZO_AI_HOST", "").strip() or "127.0.0.1"
PROTOCOL = "2024-11-05"

# The recipe book the model needs, from the app's own copy — the SAME text
# the in-app assistant teaches (core/ai_recipes.py explains what its absence
# cost). This process is stdlib-only and starts from four layouts (checkout,
# PyInstaller bundle, AppImage, Flatpak), so the app root goes on the path
# first; if that import ever fails the server still answers, with the short
# reference it can state truthfully.
_APP_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _APP_ROOT)
try:
    from core.ai_recipes import reference as _reference
    REFERENCE = _reference("llamada")
    try:
        from core.ai_recipes import MCP_NOTES
    except ImportError:                         # pragma: no cover
        MCP_NOTES = ""
except ImportError:                             # pragma: no cover - packaging
    REFERENCE = ("IngeTrazo: Z arriba, unidades en METROS. En el scope: "
                 "scene, mesh, selection, groups, layers, viewport, "
                 "QVector3D, Mesh, Group, Edge, Face, bim. Recetario: "
                 "revolve / extrude / prism / wall / house.")
    MCP_NOTES = ""
finally:
    if sys.path and sys.path[0] == _APP_ROOT:   # leave the process as found
        del sys.path[0]

#: MCP's ``initialize`` hands this to the client before any tool is listed —
#: the stance, with the recipes themselves on ``run_python`` where every
#: client is sure to put them in front of the model.
INSTRUCTIONS = (
    "Estas herramientas operan el documento ABIERTO de IngeTrazo, en vivo: "
    "lo que escribas aparece en la pantalla del usuario y entra en su "
    "historial de deshacer.\n" + REFERENCE + "\nEmpieza por query_model "
    "para ver qué hay, construye con run_python y MIRA el resultado con "
    "screenshot. Para archivos, pintura, luces y render usa las "
    "herramientas dedicadas (save, open, import_dxf, import_model, "
    "search_components, insert_component, export_ifc, quantities_csv, "
    "paint, add_light, render), no run_python.")

_OBJ = {"type": "object"}
_PATH = {"type": "string",
         "description": "Absolute path on the machine running IngeTrazo."}

TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute Python against the LIVE IngeTrazo document (Z-up, "
            "metres). Returns stdout/stderr. The reference below is "
            "everything you need — read it instead of exploring the API.\n\n"
            + REFERENCE + ("\n\n" + MCP_NOTES if MCP_NOTES else "")),
        "inputSchema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "query_model",
        "description": (
            "Model overview: faces/edges loose AND inside groups "
            "(faces_total, faces_in_groups, faces_placed…), group/component "
            "names (paged: groups_total, group_names_truncated, "
            "group_names_next_offset — call again with offset to read on), "
            "materials, layers, scenes, render lights, bounds, selection, "
            "the document's path."),
        "inputSchema": {"type": "object", "properties": {
            "offset": {"type": "integer", "description":
                       "First group name to list (default 0)."},
            "limit": {"type": "integer", "description":
                      "How many group names (default 100, max 1000)."}}},
    },
    {
        "name": "screenshot",
        "description": ("Render the current viewport and LOOK at it — use "
                        "after building to verify and iterate."),
        "inputSchema": {
            "type": "object",
            "properties": {"width": {"type": "integer"},
                           "height": {"type": "integer"}},
        },
    },
    {
        "name": "undo",
        "description": "Undo the last step in IngeTrazo.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "redo",
        "description": "Redo the last undone step in IngeTrazo.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "save",
        "description": ("Save the open project as an IngeTrazo document "
                        "(.igz) at path — like File ▸ Save As: textures "
                        "packed in, a .bak of an older file kept, the "
                        "document marked clean."),
        "inputSchema": {"type": "object", "properties": {"path": _PATH},
                        "required": ["path"]},
    },
    {
        "name": "open",
        "description": ("Open an .igz document (its own tab unless the one "
                        "in front is an untouched new drawing). For DXF use "
                        "import_dxf. If an auto-saved copy exists the call "
                        "asks you to repeat it with recover true/false."),
        "inputSchema": {"type": "object", "properties": {
            "path": _PATH,
            "recover": {"type": "boolean", "description":
                        "Only when asked: true = take the auto-saved copy, "
                        "false = the file on disk."}},
            "required": ["path"]},
    },
    {
        "name": "import_dxf",
        "description": ("Import a DXF (CadCraft plans…) as ONE undo step, "
                        "no dialog: one tagged group per CAD layer, blocks "
                        "as components. Returns the groups created (name, "
                        "layer, edges, faces), the unit used and the "
                        "bounds."),
        "inputSchema": {"type": "object", "properties": {
            "path": _PATH,
            "units": {"type": "string",
                      "enum": ["auto", "mm", "cm", "m", "in", "ft"],
                      "description": "Drawing unit; auto = measured from "
                                     "the drawing (header checked)."},
            "create_faces": {"type": "boolean", "description":
                             "Faces on closed planar outlines (default "
                             "false)."},
            "name": {"type": "string", "description":
                     "Name for the layer-0 group (default: file name)."}},
            "required": ["path"]},
    },
    {
        "name": "export_ifc",
        "description": ("Export the BIM-tagged objects as IFC4 to path "
                        "(tag first: bim.tag_group(g, 'IfcWall', 'Mur') in "
                        "run_python). Returns the element count."),
        "inputSchema": {"type": "object", "properties": {
            "path": _PATH,
            "project_name": {"type": "string"}},
            "required": ["path"]},
    },
    {
        "name": "quantities_csv",
        "description": ("Write the BIM quantity takeoff (class, name, "
                        "metrado, unit, area_m2, volume_m3) as UTF-8 CSV to "
                        "path; returns the rows and a preview."),
        "inputSchema": {"type": "object", "properties": {"path": _PATH},
                        "required": ["path"]},
    },
    {
        "name": "paint",
        "description": (
            "Paint a group or faces — one undo step, shown at once, "
            "'changed' exact. target: a group NAME (nested ones too), "
            "\"@loose\" for loose geometry, or {\"group\": name, "
            "\"faces\": ...}. Without faces the GROUP's own material is "
            "set (faces painted themselves keep theirs). faces: \"all\", "
            "[indices], or {\"normal\": [x,y,z], \"tol_deg\": 10} (faces "
            "looking that way). material: {\"color\": [r,g,b]} (0-1 or "
            "0-255), {\"texture\": {\"path\", \"sw\", \"sh\", \"rot\"}} "
            "(tile size m, rotation °), {\"library\": \"brick_red\"} (a "
            "shipped texture; sw/sh/rot beside it override), \"opacity\" "
            "0-1, \"name\" (named document material; {\"name\": n} alone "
            "applies that material), \"finish\" matte|satin|gloss|metal|"
            "glass|water (how the Blender render treats it; needs \"name\"; "
            "otherwise guessed from Spanish/English names only). {} = "
            "default material."),
        "inputSchema": {"type": "object", "properties": {
            "target": {"description": "Group name, \"@loose\", or "
                                      "{group, faces}."},
            "material": _OBJ,
            "faces": {"description": "\"all\", [indices] or "
                                     "{normal, tol_deg}."}},
            "required": ["target", "material"]},
    },
    {
        "name": "add_light",
        "description": ("Add a light for the Blender render (shown in the "
                        "Render panel, saved in the document, undoable). "
                        "kind point|spot; position [x,y,z] m; a spot aims "
                        "along direction or at target (default straight "
                        "down); power W (point 400, spot 1500); kelvin "
                        "1800-10000 (2700 warm); angle = spot cone °; "
                        "ambience day|night|overcast optionally set too."),
        "inputSchema": {"type": "object", "properties": {
            "position": {"type": "array", "items": {"type": "number"}},
            "kind": {"type": "string", "enum": ["point", "spot"]},
            "direction": {"type": "array", "items": {"type": "number"}},
            "target": {"type": "array", "items": {"type": "number"}},
            "power": {"type": "number"},
            "kelvin": {"type": "number"},
            "angle": {"type": "number"},
            "name": {"type": "string"},
            "on": {"type": "boolean"},
            "ambience": {"type": "string",
                         "enum": ["day", "night", "overcast"]}},
            "required": ["position"]},
    },
    {
        "name": "list_lights",
        "description": "The render lights (index, kind, pos, dir, power, "
                       "kelvin, angle, on, name) and the ambience.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "remove_light",
        "description": "Remove a render light by index or name (undoable).",
        "inputSchema": {"type": "object", "properties": {
            "index": {"type": "integer"}, "name": {"type": "string"}}},
    },
    {
        "name": "search_components",
        "description": (
            "Search IngeTrazo's component library (~1500 real-size models "
            "from the Sweet Home 3D collections: names in Spanish with an "
            "English translation, e.g. 'Cama'/'Bed', 'Silla'/'Chair', "
            "'Lampara'/'Lamp'; categories Dormitorio, Salón, Cocina, "
            "Cuarto de Baño, Oficina, Iluminación, Exterior, Puertas y "
            "Ventanas…) and the bundled starters. Every word of query "
            "must match; size_cm [w, d, h] (nulls allowed) ranks the "
            "closest first. Returns n (number), id, name, name_en, "
            "category, size_cm [w, d, h], licence, author. preview true "
            "also returns a numbered contact sheet of the first 12 "
            "(picture, name, size, licence) as an image, and each "
            "result's thumbnail path; sheet_path saves that sheet as a PNG "
            "where you say (e.g. the project folder) so the person can see "
            "it and choose by number."),
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string"},
            "category": {"type": "string"},
            "limit": {"type": "integer"},
            "size_cm": {"type": "array",
                        "items": {"type": ["number", "null"]}},
            "preview": {"type": "boolean",
                        "description": "Contact sheet of the candidates "
                                       "(pictures), for choosing."},
            "sheet_path": {"type": "string",
                           "description": "Absolute .png path on the "
                                          "IngeTrazo machine to keep the "
                                          "sheet (implies preview)."}}},
    },
    {
        "name": "insert_component",
        "description": (
            "Place a library model at its real size as a component (one "
            "undo step): the centre of its base at position [x,y,z] m, "
            "turned rotation degrees about the vertical. id from "
            "search_components, or query (+ size_cm/category) naming one "
            "model. Optional name, layer, ifc (e.g. 'IfcFurniture'). "
            "Copies of a model share one definition. Downloads the model "
            "the first time (needs the network on the IngeTrazo machine). "
            "Returns name, size_m, bounds, licence and author (credit "
            "CC-BY models when publishing images)."),
        "inputSchema": {"type": "object", "properties": {
            "id": {"type": "string"},
            "query": {"type": "string"},
            "position": {"type": "array", "items": {"type": "number"}},
            "rotation": {"type": "number"},
            "name": {"type": "string"},
            "layer": {"type": "string"},
            "ifc": {"type": "string"},
            "size_cm": {"type": "array",
                        "items": {"type": ["number", "null"]}},
            "category": {"type": "string"}}},
    },
    {
        "name": "import_model",
        "description": (
            "Import a 3D model file as ONE component (one undo step): "
            "GLB/glTF (best: textures and PBR materials), OBJ, DAE, SKP, "
            "STL or IGZ. position [x,y,z] m puts its anchor there (anchor "
            "'base' = centre of its base, default; 'origin' = the file's "
            "origin), turned rotation degrees; without position it stays "
            "where the file puts it. units for OBJ/STL only (no unit in "
            "the file): auto, m, cm, mm, in, ft. Optional name, layer, "
            "ifc. Returns faces, size_m, bounds, materials added."),
        "inputSchema": {"type": "object", "properties": {
            "path": _PATH,
            "position": {"type": "array", "items": {"type": "number"}},
            "rotation": {"type": "number"},
            "units": {"type": "string",
                      "enum": ["auto", "m", "cm", "mm", "in", "ft"]},
            "anchor": {"type": "string", "enum": ["base", "origin"]},
            "name": {"type": "string"},
            "layer": {"type": "string"},
            "ifc": {"type": "string"}},
            "required": ["path"]},
    },
    {
        "name": "render",
        "description": (
            "Render with the user's Blender and WAIT for the PNG at path "
            "(minutes with Cycles). scene = a saved scene's name to render "
            "from (it is applied to the view, as a click in Scenes does); "
            "without it, the current view. engine eevee|cycles, width/"
            "height px, quality 0 draft / 1 medium / 2 high. Uses the "
            "render lights, the ambience and the sun of the Shadows panel; "
            "hidden objects stay out."),
        "inputSchema": {"type": "object", "properties": {
            "path": _PATH,
            "scene": {"type": "string"},
            "engine": {"type": "string", "enum": ["eevee", "cycles"]},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
            "quality": {"type": "integer", "enum": [0, 1, 2]},
            "ground": {"type": "boolean"},
            "timeout_s": {"type": "integer", "description":
                          "Give up waiting after this (default 1800)."}},
            "required": ["path"]},
    },
]

_sock: socket.socket | None = None
_req_id = 0


def _bridge(tool: str, args: dict) -> dict:
    """One request to the in-app bridge (reconnecting once if it dropped)."""
    global _sock, _req_id
    for attempt in (0, 1):
        if _sock is None:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(180.0)
            s.connect((HOST, PORT))
            _sock = s
        _req_id += 1
        try:
            _sock.sendall((json.dumps(
                {"id": _req_id, "tool": tool, "args": args}) + "\n").encode())
            buf = b""
            while b"\n" not in buf:
                chunk = _sock.recv(65536)
                if not chunk:
                    raise OSError("bridge closed the connection")
                buf += chunk
            return json.loads(buf.split(b"\n", 1)[0])
        except OSError:
            try:
                _sock.close()
            except OSError:
                pass
            _sock = None
            if attempt:
                raise
    raise OSError("unreachable")


def _tool_result(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


#: How often the client asks the bridge how a render is going (s).
RENDER_POLL_S = float(os.environ.get("INGETRAZO_RENDER_POLL", 2.0))


def _render(args: dict) -> dict:
    """Start Blender through the bridge and wait for the image: the bridge
    never blocks its UI thread on a render, the client does the waiting."""
    args = dict(args)
    timeout = float(args.pop("timeout_s", 1800) or 1800)
    reply = _bridge("render_start", args)
    if not reply.get("ok"):
        return _tool_result(str(reply.get("error")), is_error=True)
    started = reply.get("result") or {}
    job = started.get("job")
    deadline = time.monotonic() + timeout
    while True:
        time.sleep(RENDER_POLL_S)
        st = _bridge("render_status", {"job": job})
        if not st.get("ok"):
            return _tool_result(str(st.get("error")), is_error=True)
        res = st.get("result") or {}
        if res.get("state") != "running":
            res["started"] = started
            return _tool_result(json.dumps(res, indent=2, ensure_ascii=False),
                                is_error=res.get("state") != "done")
        if time.monotonic() > deadline:
            _bridge("render_cancel", {"job": job})
            return _tool_result(
                "Render still running after %ds: cancelled. Try quality 0, "
                "eevee or a smaller size." % timeout, is_error=True)


def _call(name: str, args: dict) -> dict:
    try:
        if name == "render":
            return _render(args)
        reply = _bridge(name, args)
    except OSError as exc:
        return _tool_result(
            "Cannot reach IngeTrazo's AI bridge on %s:%d (%s). "
            "In IngeTrazo: the AI tab of the side tray > Start bridge (or Extensions > AI Bridge (MCP))."
            % (HOST, PORT, exc), is_error=True)
    if not reply.get("ok"):
        return _tool_result(str(reply.get("error")), is_error=True)
    result = reply.get("result") or {}
    if name == "screenshot" and result.get("path"):
        try:
            with open(result["path"], "rb") as fh:
                data = base64.b64encode(fh.read()).decode()
            return {"content": [{"type": "image", "data": data,
                                 "mimeType": "image/png"}]}
        except OSError as exc:
            return _tool_result(f"screenshot unreadable: {exc}",
                                is_error=True)
    if name == "search_components" and result.get("sheet"):
        content = [{"type": "text",
                    "text": json.dumps(result, indent=2, ensure_ascii=False)}]
        try:
            with open(result["sheet"], "rb") as fh:
                content.append({"type": "image",
                                "data": base64.b64encode(fh.read()).decode(),
                                "mimeType": "image/png"})
        except OSError:
            pass
        return {"content": content, "isError": False}
    if name == "run_python":
        parts = []
        if result.get("stdout"):
            parts.append(result["stdout"].rstrip())
        if result.get("stderr"):
            parts.append("stderr:\n" + result["stderr"].rstrip())
        if result.get("error"):
            parts.append("ROLLED BACK — the document is unchanged: "
                         + str(result["error"]))
        parts.append("(changed: %s)" % result.get("changed"))
        return _tool_result("\n".join(p for p in parts if p),
                            is_error=bool(result.get("error")))
    return _tool_result(json.dumps(result, indent=2, ensure_ascii=False))


def handle(msg: dict) -> dict | None:
    """One JSON-RPC message → the reply dict (None for notifications)."""
    method = msg.get("method")
    msg_id = msg.get("id")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "ingetrazo", "version": "1.1.0-mcp-agent"},
            "instructions": INSTRUCTIONS,
        }}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id,
                "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        result = _call(params.get("name", ""),
                       params.get("arguments") or {})
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if msg_id is None:          # notification (initialized, cancelled, ...)
        return None
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": -32601, "message": f"unknown {method}"}}


def _streams(stdin=None, stdout=None):
    """stdin/stdout as UTF-8 text with \\n line ends, whatever the platform
    made of them. MCP is UTF-8; a Windows console program reads its piped
    stdin in the ANSI code page (cp1252), which turned every accent the
    client sent into two characters before json.loads ever saw it."""
    stdin = stdin if stdin is not None else sys.stdin
    stdout = stdout if stdout is not None else sys.stdout
    raw_in = getattr(stdin, "buffer", stdin)
    raw_out = getattr(stdout, "buffer", stdout)
    if isinstance(raw_in, io.TextIOBase):           # already text (tests)
        text_in = raw_in
    else:
        text_in = io.TextIOWrapper(raw_in, encoding="utf-8",
                                   errors="replace", newline=None)
    if isinstance(raw_out, io.TextIOBase):
        text_out = raw_out
    else:
        text_out = io.TextIOWrapper(raw_out, encoding="utf-8",
                                    newline="\n", write_through=True)
    return text_in, text_out


def serve(stdin, stdout) -> None:
    for line in stdin:
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        reply = handle(msg)
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def main() -> None:
    serve(*_streams())


if __name__ == "__main__":
    main()
