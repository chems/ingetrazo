# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marco Sumari Tellez and IngeTrazo contributors.
"""The recipe book the model reads — ONE text behind the two AI doors.

IngeTrazo has two of them: the in-app Asistente IA (``plugins/ai_assistant``,
which pastes this into its system prompt) and the MCP bridge
(``scripts/ingetrazo_mcp``, which pastes it into ``run_python``'s description
and the handshake's ``instructions``). They drifted: the assistant's prompt
taught ``revolve``/``extrude``/``house`` and ``f.attrs["color"]``, the MCP
description taught ``mesh.add_face`` and nothing else.

What that cost, measured (2026-09-21, Antigravity CLI + Gemini 3.8 Flash,
"una mesa de comedor de 1,60 × 0,90 × 0,75 con cuatro sillas"): the model
built it correctly — exact sizes, 462 faces, not one normal inward — in
**81 tool calls, 46 of them pure introspection**. It ran ``dir()``,
``inspect.signature``, and ``dis`` over ``material_sig``'s bytecode to find
out that a colour is ``f.attrs["color"]``, and spent 34 s reasoning about
the winding of a box that ``prism()`` returns in one line. Nine calls
actually built something.

So: stdlib-only, no imports, no Qt — the MCP server is a bare stdio process
that must import this from a checkout, a PyInstaller bundle, an AppImage and
a Flatpak alike.
"""
from __future__ import annotations

#: What is bound in the execution scope. Same names through both doors.
SCOPE = (
    "En el scope tienes: scene, mesh, selection, groups, layers, "
    "viewport, QVector3D, Mesh, Group, Edge, Face, Layer, bim.")

#: The one-line builders. Hand-rolling these is the single biggest waste of
#: turns we have measured — every model that was not told reinvented a
#: ring-of-quads lathe, faceted and wrongly wound.
RECIPES = """Recetario (usa PRIMERO estas funciones; existen ya, no las \
definas):
- CAJA / CUBO: g = box((x,y,z), (dx,dy,dz), name="Cubo", color=(r,g,b,1.0)) \
— esquina inferior y medidas. Un cubo de 6 m es box((0,0,0), (6,6,6)). \
Crea el grupo y lo agrega (igual que revolve, extrude, prism, wall y \
house): NO hagas groups.append.
- CASA COMPLETA: ver house(...) más abajo — para «una casa» úsala SIEMPRE \
en vez de armar muros y techo a mano.
- TORNO (prefiérelo para toda pieza redonda): g = revolve([(radio,z), ...], \
name="Columna", color=(r,g,b,1.0), segments=32, scallop=None, closed=False) \
— perfil de abajo a arriba; abierto se tapa solo; closed=True si el perfil \
es una sección cerrada (p.ej. la pared de una taza); scallop=(profundidad, \
lóbulos) talla festones/gallones en el borde. Crea el grupo y lo agrega.
- PRISMA: g = extrude([(x,y), ...], z0, z1, name="Base", color=...) — \
contorno en planta extruido; crea el grupo y lo agrega. Una caja \
(tablero, pata, larguero, peldaño) es box(...): no la armes cara por cara.
- LOSA / PLANCHA: g = prism([puntos 3D de un polígono plano], (dx,dy,dz), \
holes=[[...]], name=..., color=...) — barre el polígono (con agujeros) \
a lo largo del vector.
- MURO: g = wall((x,y), (x,y), height=3, thickness=0.2, openings=[(offset, \
alféizar, ancho, alto)], peak=None) — crece a la IZQUIERDA de a→b; \
alféizar 0 = puerta (muesca), >0 = ventana (agujero).
- CASA COMPLETA: gs = house(width=6, depth=4, wall_height=3, thickness=0.2, \
roof="gable"|"hip"|"flat", ridge_height=None, overhang=0.4, ridge="x", \
doors=[("S", offset, ancho, alto)], windows=[("S", offset, alféizar, ancho, \
alto)], origin=(0,0), name="Casa") — muros con espesor, puertas con hoja, \
ventanas con vidrio y techo, en grupos «Casa · Paredes/Techo/Carpintería». \
Lados S (frente, y=y0), E, N, W; offset a lo largo del muro en sentido \
antihorario desde su primera esquina. Medidas típicas: puerta 0.9×2.1, \
ventana 1.2×1.0 con alféizar 1.0, pendiente 30°.
- Cara suelta: f = mesh.add_face([QVector3D(x,y,z), ...])  (lazo \
antihorario visto desde afuera); f.attrs["color"] = (r, g, b, 1.0)  (0..1)
- Arista: mesh.add_edge(QVector3D(...), QVector3D(...))
- Grupo manual: m = Mesh(); m.add_face([...]); g = Group(m, name="..."); \
groups.append(g)
- Pintar un grupo entero: g.material = {"color": (r,g,b), "opacity": 1.0} \
— NO toques scene.materials (es el registro de materiales del documento).
- Mover un grupo: for v in list(g.mesh.vertices): g.mesh.move_vertex(v, \
QVector3D(dx,dy,dz))  — y si g.xform is not None (instancia de \
componente), compón la traslación en g.xform en vez de tocar vértices.
- Cámara: viewport.camera.target/distance/yaw/pitch (RADIANES); \
viewport.update()
- LÁMINA (hoja del compositor para imprimir; «lámina»/«sheet»/«layout», \
NO es una capa): c = sheet("Planta", paper="A3", landscape=True) — la crea \
con un marco de vista y aparece sola en las pestañas de láminas. Borrar: \
scene.compositions.remove(c) (queda al menos una si había).
- CAPA / ETIQUETA: ly = Layer("Muros"); layers.append(ly); g.layer = \
"Muros". En layers solo van objetos Layer, nunca textos.
- print(...) para reportar datos (breve: el resultado viaja cada turno)."""

#: What the engine guarantees, and what the model should not waste turns on.
#: ``{unit}`` is what one execution is called at this door — a ``bloque`` of
#: code for the in-app assistant, a tool ``llamada`` over MCP.
HOW_IT_RUNS = """Cada {unit} es UN paso de undo: si lanza una excepción se \
revierte POR COMPLETO, y el Ctrl+Z del usuario deshace tu paso entero. El scope \
PERSISTE entre {unit}s: las variables y funciones que definas siguen \
disponibles — no las redefinas.
NO explores la API con dir(), inspect ni dis: lo que necesitas está en este \
recetario, y cada sondeo es un turno perdido. Si algo falta, escríbelo \
directo y lee el error.
Un pedido sencillo (una casa, una mesa, un poste) va COMPLETO de una sola \
vez, con detalles razonables aunque no te los pidan; los grandes, por \
pasos. Verifica con las capturas del viewport e itera.
La figura de persona de un documento nuevo se llama «Ingeniero» \
(«Engineer» si la interfaz está en inglés) y es la ESCALA (1,70 m, un \
billboard): no la borres ni la muevas salvo que te lo pidan, y no la tomes \
por el modelo."""

#: Model conventions. First line of everything the model reads.
UNITS = ("IngeTrazo es un modelador 3D libre de empujar/tirar: Z arriba, "
         "unidades en METROS, ángulos en RADIANES.")


#: What the MCP door adds (fork chems/ingetrazo, branch mcp-agent): the
#: API facts agents kept getting wrong (22-11-2026), and which dedicated
#: tool to use instead of run_python. Only the MCP server reads this — the
#: in-app assistant's prompt stays as short as small local models need.
MCP_NOTES = """API notes (MCP):
- Faces: f.normal() and f.centroid() are METHODS (QVector3D), f.area() too; \
f.vertices (list of QVector3D), f.attrs (dict). Loose geometry is mesh; a \
group's own is g.mesh (g.mesh.faces). Components: g.xform (QMatrix4x4) is \
not None and the mesh is shared by every copy.
- Paint: use the paint tool (undo, display refresh and an exact 'changed'). \
In run_python a face colour is f.attrs["color"] = (r, g, b) 0..1; a texture \
f.attrs["texture"] = {"path": abs_path, "sw": tile_w_m, "sh": tile_h_m, \
"rot": degrees}; a whole group g.material = {"color": (r,g,b)} or \
{"texture": {...}}. Shipped texture library: from core.agent_tools import \
texture_library; texture_library() -> [{name, category, path, sw, sh}] \
(names like brick_red, concrete_smooth, stone_ashlar…).
- Hide / show: g.hidden = True (undoable, and pictures leave it out — the \
scale figure «Ingeniero»/«Engineer» included).
- Camera: cam = viewport.camera; cam.place(QVector3D(eye), \
QVector3D(target)) puts the eye exactly there; cam.perspective (bool); \
cam.fov_deg (vertical field of view, degrees); viewport.update().
- Picture from code: img = viewport.render_image(width_px, height_px) \
(QImage; img.save(path)); prefer the screenshot tool to look.
- BIM: bim.tag_group(g, "IfcWall", "Mur Sud") (IfcSlab, IfcRoof, IfcDoor, \
IfcWindow, IfcColumn, IfcBeam, IfcStair, IfcCovering…), bim.untag_group(g); \
bim.quantities_csv(scene) -> CSV text. Files: quantities_csv / export_ifc \
tools.
- Axes: model.show_axes = False / True, then viewport.update().
- Names are Unicode: write accents as they are («Allège», «Séjour»).
Dedicated tools (prefer them to run_python): save(path .igz), open(path), \
import_dxf(path, units, create_faces) -> groups created, export_ifc(path), \
quantities_csv(path), paint(target, material, faces), add_light / \
list_lights / remove_light (Blender render lights, Render panel), \
render(path, scene, engine, width, height, quality) -> waits for the PNG, \
query_model(offset, limit) -> counts incl. groups, paged names, \
search_components(query, category, size_cm) + insert_component(id, \
position, rotation) -> real-size library models (names in Spanish/English: \
Cama/Bed, Silla/Chair, Lampara/Lamp, Radiador/Radiator…), import_model(path, \
position, rotation, units) -> GLB/OBJ/DAE/SKP/STL/IGZ as one component.
- Quality: replace placeholder boxes by library models; textures at real \
tile size (sw/sh m); name materials and set their finish with paint \
({"name": "Vidrio", "finish": "glass"}) — the render guesses finishes from \
Spanish/English/Portuguese names only (a French «verre» renders matte)."""


def reference(unit: str = "llamada") -> str:
    """Everything, in reading order — what a door hands the model."""
    return "\n".join((UNITS, SCOPE, RECIPES, HOW_IT_RUNS.format(unit=unit)))
