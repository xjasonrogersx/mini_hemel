"""Local control panel for the mini_hemel viewer."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import html
import logging
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import unquote, urlparse


WEB_HOST = "127.0.0.1"
WEB_PORT = 8765
LOGGER = logging.getLogger(__name__)


class ViewerWebServer:
    def __init__(
        self,
        captures_path: Path,
        artifacts_path: Path,
        state_callback: Callable[[], dict[str, Any]],
        command_callback: Callable[[dict[str, Any]], None],
        config_callback: Callable[[dict[str, Any]], None],
        host: str = WEB_HOST,
        port: int = WEB_PORT,
    ) -> None:
        self.captures_path = captures_path.resolve()
        self.artifacts_path = artifacts_path.resolve()
        self.state_callback = state_callback
        self.command_callback = command_callback
        self.config_callback = config_callback
        self.server = ThreadingHTTPServer(
            (host, port), self._handler_class()
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
            name="viewer-web-server",
        )

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_json(self, payload: Any, status: int = 200) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_bytes(self, data: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                LOGGER.info("Web GET %s", parsed.path)
                if parsed.path in {"/", "/index.html"}:
                    self._send_bytes(CONTROL_PANEL_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/state":
                    self._send_json(owner.state_callback())
                    return
                if parsed.path == "/api/artifacts":
                    self._send_json(owner._load_artifacts())
                    return
                if parsed.path.startswith("/artifact/"):
                    owner._send_artifact_page(self, parsed.path.removeprefix("/artifact/"))
                    return
                if parsed.path.startswith("/captures/"):
                    self._send_capture(parsed.path.removeprefix("/captures/"))
                    return
                self._send_json({"error": "not found"}, 404)

            def _send_capture(self, relative_path: str) -> None:
                capture_path = (owner.captures_path / unquote(relative_path)).resolve()
                if owner.captures_path not in capture_path.parents or not capture_path.is_file():
                    self._send_json({"error": "capture not found"}, 404)
                    return
                content_type = "image/png" if capture_path.suffix.lower() == ".png" else "application/octet-stream"
                self._send_bytes(capture_path.read_bytes(), content_type)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/control":
                    self._send_json({"error": "not found"}, 404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("request must be a JSON object")
                    LOGGER.info("Web command received: %s", payload.get("action"))
                    if payload.get("action") == "set_options":
                        result = owner.config_callback(payload)
                    else:
                        result = owner.command_callback(payload)
                    if isinstance(result, dict) and not result.get("ok", True):
                        LOGGER.warning("Web command failed: %s", result.get("error"))
                        self._send_json(result, 400)
                        return
                    LOGGER.info("Web command completed: %s", payload.get("action"))
                    self._send_json(result if isinstance(result, dict) else {"ok": True})
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, 400)

        return Handler

    def _load_artifacts(self) -> list[dict[str, Any]]:
        if not self.artifacts_path.is_file():
            return []
        try:
            data = json.loads(self.artifacts_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def _send_artifact_page(self, handler: BaseHTTPRequestHandler, value: str) -> None:
        try:
            index = int(value)
        except ValueError:
            handler.send_error(404, "invalid artifact")
            return
        artifacts = self._load_artifacts()
        if index < 0 or index >= len(artifacts):
            handler.send_error(404, "artifact not found")
            return
        artifact = artifacts[index]
        image_fields = (
            ("texture_render", "Texture input"),
            ("result_render", "Generated result"),
            ("depth_render", "Renderer depth"),
            ("generated_depth_render", "Depth Anything"),
            ("segmentation_color_map", "SegFormer color map"),
            ("segmentation_label_map", "SegFormer label map"),
            ("mask2former_color_map", "Mask2Former color map"),
            ("mask2former_label_map", "Mask2Former label map"),
            ("building_regularization_lines", "Building M-LSD lines"),
        )
        available = [(field, label, artifact[field]) for field, label in image_fields if artifact.get(field)]
        available.extend(
            (f"segmentation_mask_{mask_index}", "SegFormer mask", filename)
            for mask_index, filename in enumerate(artifact.get("segmentation_masks", []))
            if filename
        )
        available.extend(
            (f"mask2former_mask_{mask_index}", "Mask2Former mask", filename)
            for mask_index, filename in enumerate(artifact.get("mask2former_masks", []))
            if filename
        )
        raw_detections = artifact.get("grounding_dino_detections", {})
        if isinstance(raw_detections, list):
            grouped_legacy = {}
            for detection in raw_detections:
                prompt = str(detection.get("prompt") or detection.get("label") or "legacy").strip().lower().rstrip(".,;:!? ")
                grouped_legacy.setdefault(prompt, []).append(detection)
            detection_groups = list(grouped_legacy.items())
        elif isinstance(raw_detections, dict):
            detection_groups = [
                (prompt, detections)
                for prompt, detections in raw_detections.items()
                if isinstance(detections, list)
            ]
        else:
            detection_groups = []
        detection_entries = [
            (prompt, detection_index, dict(detection))
            for prompt, group in detection_groups
            for detection_index, detection in enumerate(group)
            if isinstance(detection, dict)
        ]
        available.extend(
            (f"sam2_dino_box_{entry_index}", f"SAM2 {prompt} box {detection_index + 1}", detection["sam2_mask"])
            for entry_index, (prompt, detection_index, detection) in enumerate(detection_entries)
            if detection.get("sam2_mask")
        )
        options = "".join(
            f'<option value="{html.escape(field)}">{html.escape(label)}</option>'
            for field, label, _filename in available
        )
        images = "".join(
            f'<figure><img src="/captures/{html.escape(filename)}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}<br>{html.escape(filename)}</figcaption></figure>'
            for _field, label, filename in available
        )
        detections = [
            {**detection, "prompt": detection.get("prompt", prompt)}
            for prompt, _detection_index, detection in detection_entries
        ]
        detection_rows = "".join(
            f'<li><strong>{html.escape(prompt)}: {html.escape(str(detection.get("label", "detection")))}</strong> '
            f'confidence {float(detection.get("score", 0)):.3f} '
            f'box [{", ".join(f"{float(value):.1f}" for value in detection.get("box_xyxy", []))}] '
            f'<button onclick="sam2Box({entry_index})">Run / rerun SAM2</button>'
            f'<button onclick="deleteDino({entry_index})">Delete</button>'
            f'{" &middot; mask: " + html.escape(str(detection["sam2_mask"])) if detection.get("sam2_mask") else ""}</li>'
            for entry_index, (prompt, _detection_index, detection) in enumerate(detection_entries)
        )
        detection_report = (
            f'<ul>{detection_rows}</ul>' if detection_rows else '<p class="muted">No detections yet.</p>'
        )
        dino_image = artifact.get("result_render") or ""
        refined_asset = artifact.get("refined_asset")
        refinement_status = artifact.get("refinement_status")
        refined_view_button = (
            f'<button onclick="viewRefined()">View refined texture</button>'
            if refined_asset and artifact.get("result_render") else ""
        )
        refinement_section = (
            f'<section class="operation"><h2>Mesh refinement</h2>'
            f'<p class="muted">Uses depth, segmentation, and generated-image structure to adjust the mesh geometry and save a refined mesh.</p>'
            f'<p id="refinementStatus" class="muted">Refinement status: {html.escape(str(refinement_status or "not started"))}</p>'
            f'<p><button onclick="refineMesh()">Run / rerun mesh refinement</button></p></section>'
            f'<section class="operation"><h2>Texture reprojection</h2>'
            f'<p class="muted">Projects the generated result image back onto the refined mesh for viewing. This is separate from geometry refinement.</p>'
            f'<p>{refined_view_button or "No reprojected texture available yet."}</p></section>'
        )
        road_result = artifact.get("road_smoothing_asset")
        building_result = artifact.get("building_regularization_asset")
        road_section = (
            f'<section class="operation"><h2>Road-smoothed mesh</h2><p class="muted">Smooths road geometry using the detected road mask. Deploying it loads the modified mesh only and does not apply the generated texture.</p><p class="muted">Status: {html.escape(str(artifact.get("road_smoothing_status")))}</p><button onclick="smoothRoad()">Run road smoothing</button> <button onclick="loadRoadSmoothed()">Deploy mesh to viewer</button></section>'
            if road_result else
            '<section class="operation"><h2>Road-smoothed mesh</h2><p class="muted">Smooths road geometry using the detected road mask. Deployment loads the modified mesh only and does not apply the generated texture.</p><p class="muted">No road-smoothed mesh yet.</p><button onclick="smoothRoad()">Run road smoothing</button></section>'
        )
        building_section = (
            f'<section class="operation"><h2>Building-regularized mesh</h2><p class="muted">Aligns and simplifies building surfaces using building masks and M-LSD structural lines. Deployment loads the modified mesh only and does not apply the generated texture.</p><p class="muted">Status: {html.escape(str(artifact.get("building_regularization_status")))}</p><p class="muted">M-LSD lines: {html.escape(str(artifact.get("building_regularization_lines_status", "not generated")))}</p><label for="buildingAggression">Aggression</label> <input id="buildingAggression" class="range" type="range" min="0" max="1" step="0.05" value="0.5" oninput="document.getElementById(\'buildingAggressionValue\').textContent=Number(this.value).toFixed(2)"> <output id="buildingAggressionValue">0.50</output><p><button onclick="runMlsd()">Run / rerun M-LSD</button> <button onclick="regularizeBuildings()">Run building regularization</button> <button onclick="loadBuildingsRegularized()">Deploy mesh to viewer</button></p></section>'
            if building_result else
            f'<section class="operation"><h2>Building-regularized mesh</h2><p class="muted">Aligns and simplifies building surfaces using building masks and M-LSD structural lines. Deployment loads the modified mesh only and does not apply the generated texture.</p><p class="muted">No building-regularized mesh yet. M-LSD lines: {html.escape(str(artifact.get("building_regularization_lines_status", "not generated")))}</p><label for="buildingAggression">Aggression</label> <input id="buildingAggression" class="range" type="range" min="0" max="1" step="0.05" value="0.5" oninput="document.getElementById(\'buildingAggressionValue\').textContent=Number(this.value).toFixed(2)"> <output id="buildingAggressionValue">0.50</output><p><button onclick="runMlsd()">Run / rerun M-LSD</button> <button onclick="regularizeBuildings()">Run building regularization</button></p></section>'
        )
        pose = artifact.get("camera_pose", {})
        page = ARTIFACT_PAGE_HTML.format(
            index=index,
            title=html.escape(f"Artifact {index + 1}"),
            created_at=html.escape(str(artifact.get("created_at", ""))),
            generator=html.escape(str(artifact.get("generator", "generated"))),
            yaw=html.escape(str(pose.get("yaw", "?"))),
            pitch=html.escape(str(pose.get("pitch", "?"))),
            distance=html.escape(str(pose.get("distance", "?"))),
            options=options,
            images=images,
            first_image=html.escape(available[1][2] if len(available) > 1 else (available[0][2] if available else "")),
            second_image=html.escape(available[0][2] if available else ""),
            fields=json.dumps({field: filename for field, _label, filename in available}),
            detection_report=detection_report,
            dino_image=html.escape(dino_image),
            detections_json=json.dumps(detections),
            detection_targets=json.dumps([
                {"prompt": prompt, "detection_index": detection_index}
                for prompt, detection_index, _detection in detection_entries
            ]),
            refinement_summary=refinement_section + road_section + building_section,
        )
        data = page.encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


ARTIFACT_PAGE_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} | Mini Hemel</title>
<style>
:root {{ color-scheme:dark; --ink:#e9edf4; --muted:#93a0b4; --panel:#18202b; --line:#2b384a; --accent:#f0a35b; --cyan:#75d0c5; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#0d131b; color:var(--ink); font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }}
main {{ max-width:1200px; margin:auto; padding:24px clamp(18px,4vw,56px) 60px; }} a,button,select,input {{ color:inherit; }} a {{ color:var(--cyan); }}
.head {{ display:flex; justify-content:space-between; gap:18px; align-items:start; border-bottom:1px solid var(--line); padding-bottom:18px; }} h1,h2 {{ font-family:Georgia,serif; }} h1 {{ margin:0; font-size:30px; }} h2 {{ margin:28px 0 12px; }} .muted,figcaption {{ color:var(--muted); }}
button,select,input {{ border:1px solid var(--line); background:var(--panel); border-radius:5px; padding:9px 12px; font:inherit; }} button {{ cursor:pointer; }} button:hover {{ border-color:var(--accent); }}
.images {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }} figure {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; }} figure img {{ display:block; width:100%; aspect-ratio:4/3; object-fit:contain; background:#080b10; }} figcaption {{ padding:9px; overflow-wrap:anywhere; }}
.controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:14px 0; }} .stage {{ position:relative; aspect-ratio:4/3; max-width:900px; background:#080b10; overflow:hidden; }} .stage img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }} .top {{ clip-path:inset(0 50% 0 0); }} .divider {{ position:absolute; top:0; bottom:0; left:50%; width:2px; background:var(--accent); pointer-events:none; }} .range {{ width:min(900px,100%); accent-color:var(--accent); }} .status {{ color:var(--cyan); }}
.dino-stage {{ position:relative; width:min(900px,100%); background:#080b10; }} .dino-stage img {{ display:block; width:100%; height:auto; }} .dino-stage canvas {{ position:absolute; inset:0; width:100%; height:100%; }}
.operation {{ margin:18px 0; padding:16px; background:var(--panel); border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:6px; }} .operation h2 {{ margin:0 0 8px; }} .operation .range {{ width:min(420px,100%); vertical-align:middle; }} .operation output {{ display:inline-block; min-width:3em; color:var(--cyan); }}
</style></head><body><main>
<div class="head"><div><a href="/">&larr; All artifacts</a><h1>{title}</h1><div class="muted">{generator} &middot; {created_at}</div></div><div class="muted">yaw {yaw} &middot; pitch {pitch} &middot; distance {distance}</div></div>
<p><button onclick="navigate()">Navigate display to this pose</button> <button onclick="segment()">Run / rerun SegFormer</button> <button onclick="mask2former()">Run / rerun Mask2Former</button> <button onclick="depthAnything()">Run / rerun Depth Anything on generated result</button> <span id="status" class="status"></span></p>{refinement_summary}
<h2>Artifact images</h2><div class="images">{images}</div>
<h2>Compare images from this artifact</h2><div class="controls"><label for="imageA">Image A</label><select id="imageA" onchange="updateCompare()">{options}</select><label for="imageB">Image B</label><select id="imageB" onchange="updateCompare()">{options}</select></div>
<div class="stage"><img id="bottom" src="/captures/{first_image}" alt="Image B"><img id="top" class="top" src="/captures/{second_image}" alt="Image A"><div id="divider" class="divider"></div></div><label for="slider">Swipe position</label><br><input id="slider" class="range" type="range" min="0" max="100" value="50" oninput="updateCompare()">
<h2>Grounding DINO</h2><div class="controls"><label for="dinoPrompt">Text prompt</label><input id="dinoPrompt" value="door"><button onclick="groundingDino()">Run / rerun Grounding DINO</button><button onclick="sam2All()">Run SAM2 on all boxes</button></div>{detection_report}<div class="dino-stage"><img id="dinoImage" src="/captures/{dino_image}" alt="Generated result image for Grounding DINO" onerror="this.alt='Generated result image unavailable'"><canvas id="dinoCanvas"></canvas></div>
<script>
async function navigate() {{ const status=document.getElementById('status'); status.textContent='Navigating...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'navigate',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'Display moved to this pose':data.error; }}
async function viewRefined() {{ const status=document.getElementById('status'); status.textContent='Loading refined texture...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'view_refined',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'Refined texture loaded in viewer':data.error; }}
async function loadRoadSmoothed() {{ const status=document.getElementById('status'); status.textContent='Loading road-smoothed mesh...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'load_road_smoothed',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'Road-smoothed mesh loaded':data.error; }}
async function loadBuildingsRegularized() {{ const status=document.getElementById('status'); status.textContent='Loading building-regularized mesh...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'load_buildings_regularized',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'Building-regularized mesh loaded':data.error; }}
async function segment() {{ const status=document.getElementById('status'); status.textContent='Starting SegFormer...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'segment',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'SegFormer started; refresh this page when complete':data.error; }}
async function mask2former() {{ const status=document.getElementById('status'); status.textContent='Starting Mask2Former...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'mask2former',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'Mask2Former started; refresh this page when complete':data.error; }}
async function runMlsd() {{ const status=document.getElementById('status'); status.textContent='Starting M-LSD...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'mlsd',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'M-LSD started; refresh this page when complete':data.error; }}
async function depthAnything() {{ const status=document.getElementById('status'); status.textContent='Starting Depth Anything on generated result...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'depth_anything',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'Depth Anything started on generated result; refresh this page when complete':data.error; }}
async function refineMesh() {{ const status=document.getElementById('refinementStatus'); status.textContent='Refinement status: starting...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'refine_mesh',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'Refinement status: started; refresh this page when complete':'Refinement status: '+data.error; }}
async function smoothRoad() {{ const status=document.getElementById('status'); status.textContent='Starting road smoothing...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'smooth_road',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'Road smoothing started; refresh this page when complete':data.error; }}
async function regularizeBuildings() {{ const status=document.getElementById('status'); const aggression=Number(document.getElementById('buildingAggression').value); status.textContent='Starting building regularization...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'regularize_buildings',index:{index},aggression:aggression}})}}); const data=await response.json(); status.textContent=data.ok?'Building regularization started; refresh this page when complete':data.error; }}
async function groundingDino() {{ const status=document.getElementById('status'); status.textContent='Starting Grounding DINO...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'grounding_dino',index:{index},prompt:document.getElementById('dinoPrompt').value}})}}); const data=await response.json(); status.textContent=data.ok?'Grounding DINO started; refresh this page when complete':data.error; }}
async function sam2All() {{ const status=document.getElementById('status'); status.textContent='Starting SAM2 for all boxes...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'sam2_all',index:{index}}})}}); const data=await response.json(); status.textContent=data.ok?'SAM2 started for all boxes; refresh this page when complete':data.error; }}
async function deleteDino(entryIndex) {{ const target=detectionTargets[entryIndex]; if(!target || !window.confirm('Delete this Grounding DINO detection?')) return; const status=document.getElementById('status'); status.textContent='Deleting detection...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'delete_dino_detection',index:{index},prompt:target.prompt,detection_index:target.detection_index}})}}); const data=await response.json(); if(data.ok) window.location.reload(); else status.textContent=data.error; }}
const detectionTargets={detection_targets};
async function sam2Box(entryIndex) {{ const target=detectionTargets[entryIndex]; const status=document.getElementById('status'); status.textContent='Starting SAM2...'; const response=await fetch('/api/control',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{action:'sam2_box',index:{index},prompt:target.prompt,detection_index:target.detection_index}})}}); const data=await response.json(); status.textContent=data.ok?'SAM2 started; refresh this page when complete':data.error; }}
function updateCompare() {{ const a=document.getElementById('imageA').value,b=document.getElementById('imageB').value,p=Number(document.getElementById('slider').value); const fields={fields}; document.getElementById('bottom').src='/captures/'+encodeURIComponent(fields[b]); document.getElementById('top').src='/captures/'+encodeURIComponent(fields[a]); document.getElementById('top').style.clipPath='inset(0 '+(100-p)+'% 0 0)'; document.getElementById('divider').style.left=p+'%'; }}
const dinoDetections={detections_json};
function promptColor(prompt) {{ let hash=0; for(let index=0; index<prompt.length; index++) hash=((hash<<5)-hash)+prompt.charCodeAt(index); const hue=Math.abs(hash)%360; return 'hsl('+hue+',85%,62%)'; }}
function drawDetections() {{ const image=document.getElementById('dinoImage'); const canvas=document.getElementById('dinoCanvas'); if(!image || !canvas || !image.naturalWidth) return; const scale=image.clientWidth/image.naturalWidth; canvas.width=image.clientWidth; canvas.height=image.clientHeight; const context=canvas.getContext('2d'); context.clearRect(0,0,canvas.width,canvas.height); dinoDetections.forEach((detection,index)=>{{ const box=detection.box_xyxy.map(value=>value*scale); const color=promptColor(detection.prompt||'default'); context.strokeStyle=color; context.lineWidth=2; context.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]); context.fillStyle=color; context.font='bold 14px sans-serif'; context.fillText((index+1)+': '+detection.label+' '+Number(detection.score).toFixed(3),box[0]+4,Math.max(16,box[1]-5)); }}); }}
document.getElementById('dinoImage').addEventListener('load',drawDetections); window.addEventListener('resize',drawDetections); drawDetections();
</script></main></body></html>"""


CONTROL_PANEL_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini Hemel Control</title>
<style>
:root { color-scheme: dark; --ink:#e9edf4; --muted:#93a0b4; --panel:#18202b; --line:#2b384a; --accent:#f0a35b; --cyan:#75d0c5; }
* { box-sizing:border-box; } body { margin:0; background:#0d131b; color:var(--ink); font:14px/1.45 ui-sans-serif,system-ui,sans-serif; }
header { padding:24px clamp(18px,4vw,56px) 18px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:20px; align-items:end; }
h1 { margin:0; font:700 30px/1.05 Georgia,serif; letter-spacing:.01em; } header p { margin:7px 0 0; color:var(--muted); }
main { max-width:1400px; margin:auto; padding:22px clamp(18px,4vw,56px) 60px; }
.toolbar { display:flex; flex-wrap:wrap; gap:10px; align-items:end; padding-bottom:22px; border-bottom:1px solid var(--line); }
.group { display:flex; gap:8px; align-items:center; } label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em; }
button, select, input { border:1px solid var(--line); background:var(--panel); color:var(--ink); border-radius:5px; padding:9px 12px; font:inherit; } button { cursor:pointer; } button:hover { border-color:var(--accent); color:#fff; } .primary { background:#8c512d; border-color:#bd7040; }
input { min-width:260px; } .status { margin-left:auto; color:var(--cyan); min-height:20px; }
h2 { margin:28px 0 12px; font:600 20px Georgia,serif; } .assets { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; }
article { background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; } .meta { padding:13px 14px; } .meta strong { display:block; margin-bottom:5px; } .meta small { color:var(--muted); }
.images { display:grid; grid-template-columns:repeat(3,1fr); background:#0a0e14; gap:1px; } figure { margin:0; min-width:0; } figure img { display:block; width:100%; aspect-ratio:4/3; object-fit:cover; background:#090c11; } figcaption { padding:6px 7px 8px; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }
.actions { display:flex; gap:8px; padding:0 14px 14px; } .empty { color:var(--muted); padding:24px 0; }
.modal { position:fixed; inset:0; z-index:5; display:none; place-items:center; padding:20px; background:rgba(3,7,12,.82); }
.modal.open { display:grid; } .dialog { width:min(1100px,100%); max-height:calc(100vh - 40px); overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:18px; }
.dialog-head { display:flex; justify-content:space-between; align-items:center; gap:12px; } .dialog h2 { margin:0; }
.compare-controls { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:15px 0; } .compare-controls select { min-width:220px; }
.compare-stage { position:relative; width:100%; aspect-ratio:4/3; overflow:hidden; background:#080b10; } .compare-stage img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }
.compare-top { clip-path:inset(0 50% 0 0); } .compare-divider { position:absolute; top:0; bottom:0; left:50%; width:2px; background:var(--accent); pointer-events:none; }
.compare-range { width:100%; accent-color:var(--accent); }
@media (max-width:700px) { header { display:block; } .status { margin:10px 0 0; } .toolbar { align-items:stretch; } input { min-width:0; width:100%; } }
</style>
</head>
<body>
<header><div><h1>Mini Hemel</h1><p>Viewer control and generated asset archive</p></div><div id="state">Connecting...</div></header>
<main>
<section class="toolbar">
 <div class="group"><label for="mode">Display</label><select id="mode"><option value="textured">Textured</option><option value="depth">Depth</option></select><button onclick="setMode()">Apply</button></div>
 <div class="group"><label for="navigation">Camera</label><select id="navigation"><option value="orbit">Orbit</option><option value="walk">Walk</option></select><button onclick="setNavigation()">Apply</button></div>
 <button class="primary" onclick="generate('configured')">Generate configured texture</button><button onclick="generate('sdxl')">Generate SDXL</button><button onclick="toggleEdges()">Toggle triangle edges</button><button onclick="toggleVisibleFaces()">Toggle in-view triangles</button><button onclick="toggleTexture()">Toggle texture / flat colors</button><button onclick="saveView()">Save view</button>
 <span class="status" id="status"></span>
</section>
<section><h2>Texture options</h2><div class="toolbar"><label for="prompt">Prompt</label><input id="prompt"><label for="resolution">Resolution</label><select id="resolution"><option>1k</option><option>2k</option><option>4k</option></select><label for="aspect">Aspect</label><select id="aspect"><option>4:3</option><option>16:9</option><option>1:1</option></select><button onclick="saveOptions()">Save options</button></div></section>
<section><h2>Generated assets</h2><div class="assets" id="assets"><div class="empty">Loading artifacts...</div></div></section>
</main>
<script>
const $ = id => document.getElementById(id);
async function post(body) { const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const data=await r.json(); if(!data.ok) throw Error(data.error); return data; }
function notice(text) { $('status').textContent=text; setTimeout(()=>{$('status').textContent=''},3500); }
async function setMode(){ try { await post({action:'set_mode',mode:$('mode').value}); notice('Display mode updated'); } catch(e){notice(e.message)} }
async function setNavigation(){ try { await post({action:'set_navigation',mode:$('navigation').value}); notice('Camera mode updated'); } catch(e){notice(e.message)} }
async function toggleEdges(){ try { await post({action:'toggle_edges'}); notice('Triangle edges toggled'); load(); } catch(e){notice(e.message)} }
async function toggleVisibleFaces(){ try { await post({action:'toggle_visible_faces'}); notice('In-view triangle overlay toggled'); load(); } catch(e){notice(e.message)} }
async function toggleTexture(){ try { await post({action:'toggle_texture'}); notice('Texture / flat colors toggled'); load(); } catch(e){notice(e.message)} }
async function generate(generator){ try { await post({action:'generate',generator}); notice('Generation started'); } catch(e){notice(e.message)} }
async function saveView(){ try { await post({action:'save_view'}); notice('View capture started'); } catch(e){notice(e.message)} }
async function saveOptions(){ try { await post({action:'set_options',options:{prompt:$('prompt').value,resolution:$('resolution').value,aspect_ratio:$('aspect').value}}); notice('Options saved'); } catch(e){notice(e.message)} }
function image(name,label){ if(!name) return '<figure><div style="aspect-ratio:4/3"></div><figcaption>'+label+' unavailable</figcaption></figure>'; return '<figure><img loading="lazy" src="/captures/'+encodeURIComponent(name)+'" alt="'+label+'"><figcaption>'+name+'</figcaption></figure>'; }
let artifacts=[];
function renderAssets(items){ artifacts=items; const root=$('assets'); if(!items.length){root.innerHTML='<div class="empty">No artifact records yet.</div>';return;} root.innerHTML=items.slice().reverse().map((a,i)=>'<article><div class="meta"><strong>'+((a.generator||'generated')+' · '+(a.created_at||''))+'</strong><small>Camera yaw '+Number(a.camera_pose?.yaw||0).toFixed(3)+' · pitch '+Number(a.camera_pose?.pitch||0).toFixed(3)+' · distance '+Number(a.camera_pose?.distance||0).toFixed(2)+'</small></div><div class="images">'+image(a.texture_render,'Texture input')+image(a.result_render,'Generated result')+image(a.depth_render,'Renderer depth')+image(a.generated_depth_render,'Depth Anything')+'</div><div class="actions"><a href="/artifact/'+items.indexOf(a)+'">Open artifact</a><button onclick="navigateAsset('+items.indexOf(a)+')">Navigate display</button></div></article>').join(''); }
async function navigateAsset(index){ try { await post({action:'navigate',index}); notice('Display moved to asset camera'); } catch(e){notice(e.message)} }
async function load(){ try { const [state,items]=await Promise.all([fetch('/api/state').then(r=>r.json()),fetch('/api/artifacts').then(r=>r.json())]); $('state').textContent=state.mode+' · '+(state.walk_mode?'walk':'orbit'); $('mode').value=state.mode; $('navigation').value=state.walk_mode?'walk':'orbit'; const t=state.texture_generator||{}; $('prompt').value=t.prompt||''; $('resolution').value=t.resolution||'1k'; $('aspect').value=t.aspect_ratio||'4:3'; renderAssets(items); } catch(e){$('state').textContent='Offline';} }
load(); setInterval(load,5000);
</script>
</body></html>"""
