"""Local control panel for the mini_hemel viewer."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import unquote, urlparse


WEB_HOST = "127.0.0.1"
WEB_PORT = 8765


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
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_bytes(self, data: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path in {"/", "/index.html"}:
                    self._send_bytes(CONTROL_PANEL_HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/state":
                    self._send_json(owner.state_callback())
                    return
                if parsed.path == "/api/artifacts":
                    self._send_json(owner._load_artifacts())
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
                    if payload.get("action") == "set_options":
                        result = owner.config_callback(payload)
                    else:
                        result = owner.command_callback(payload)
                    if isinstance(result, dict) and not result.get("ok", True):
                        self._send_json(result, 400)
                        return
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

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


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
 <button class="primary" onclick="generate('configured')">Generate configured texture</button><button onclick="generate('sdxl')">Generate SDXL</button><button onclick="saveView()">Save view</button><button onclick="openCompare()">Compare artifacts</button>
 <span class="status" id="status"></span>
</section>
<section><h2>Texture options</h2><div class="toolbar"><label for="prompt">Prompt</label><input id="prompt"><label for="resolution">Resolution</label><select id="resolution"><option>1k</option><option>2k</option><option>4k</option></select><label for="aspect">Aspect</label><select id="aspect"><option>4:3</option><option>16:9</option><option>1:1</option></select><button onclick="saveOptions()">Save options</button></div></section>
<section><h2>Generated assets</h2><div class="assets" id="assets"><div class="empty">Loading artifacts...</div></div></section>
</main>
<div class="modal" id="compareModal" onclick="closeCompare(event)"><div class="dialog" onclick="event.stopPropagation()"><div class="dialog-head"><h2>Compare generated images</h2><button onclick="closeCompare()">Close</button></div><div class="compare-controls"><label for="compareA">Image A</label><select id="compareA" onchange="updateCompare()"></select><label for="compareB">Image B</label><select id="compareB" onchange="updateCompare()"></select></div><div class="compare-stage"><img id="compareBottom" alt="Image A"><img id="compareTop" class="compare-top" alt="Image B"><div id="compareDivider" class="compare-divider"></div></div><label for="compareSlider">Swipe position</label><input id="compareSlider" class="compare-range" type="range" min="0" max="100" value="50" oninput="updateCompare()"></div></div>
<script>
const $ = id => document.getElementById(id);
async function post(body) { const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); const data=await r.json(); if(!data.ok) throw Error(data.error); return data; }
function notice(text) { $('status').textContent=text; setTimeout(()=>{$('status').textContent=''},3500); }
async function setMode(){ try { await post({action:'set_mode',mode:$('mode').value}); notice('Display mode updated'); } catch(e){notice(e.message)} }
async function setNavigation(){ try { await post({action:'set_navigation',mode:$('navigation').value}); notice('Camera mode updated'); } catch(e){notice(e.message)} }
async function generate(generator){ try { await post({action:'generate',generator}); notice('Generation started'); } catch(e){notice(e.message)} }
async function saveView(){ try { await post({action:'save_view'}); notice('View capture started'); } catch(e){notice(e.message)} }
async function saveOptions(){ try { await post({action:'set_options',options:{prompt:$('prompt').value,resolution:$('resolution').value,aspect_ratio:$('aspect').value}}); notice('Options saved'); } catch(e){notice(e.message)} }
function image(name,label){ if(!name) return '<figure><div style="aspect-ratio:4/3"></div><figcaption>'+label+' unavailable</figcaption></figure>'; return '<figure><img loading="lazy" src="/captures/'+encodeURIComponent(name)+'" alt="'+label+'"><figcaption>'+name+'</figcaption></figure>'; }
let artifacts=[];
function renderAssets(items){ artifacts=items; const root=$('assets'); if(!items.length){root.innerHTML='<div class="empty">No artifact records yet.</div>';return;} root.innerHTML=items.slice().reverse().map((a,i)=>'<article><div class="meta"><strong>'+((a.generator||'generated')+' · '+(a.created_at||''))+'</strong><small>Camera yaw '+Number(a.camera_pose?.yaw||0).toFixed(3)+' · pitch '+Number(a.camera_pose?.pitch||0).toFixed(3)+' · distance '+Number(a.camera_pose?.distance||0).toFixed(2)+'</small></div><div class="images">'+image(a.texture_render,'Texture input')+image(a.result_render,'Generated result')+image(a.depth_render,'Renderer depth')+image(a.generated_depth_render,'Depth Anything')+'</div><div class="actions"><button class="primary" onclick="navigateAsset(${items.indexOf(a)})">Navigate display</button></div></article>').join(''); }
async function navigateAsset(index){ try { await post({action:'navigate',index}); notice('Display moved to asset camera'); } catch(e){notice(e.message)} }
function imageUrl(asset){ return asset.result_render||asset.texture_render||asset.generated_depth_render||asset.depth_render; }
function openCompare(){ if(artifacts.length<2){notice('At least two artifacts are required');return;} const options=artifacts.map((a,i)=>'<option value="'+i+'">'+(i+1)+' · '+(a.result_render||a.texture_render||'artifact')+'</option>').join(''); $('compareA').innerHTML=options; $('compareB').innerHTML=options; $('compareB').value='1'; $('compareModal').classList.add('open'); updateCompare(); }
function closeCompare(event){ if(!event||event.target===$('compareModal')) $('compareModal').classList.remove('open'); }
function updateCompare(){ const a=artifacts[Number($('compareA').value)], b=artifacts[Number($('compareB').value)]; if(!a||!b)return; const position=Number($('compareSlider').value); $('compareBottom').src='/captures/'+encodeURIComponent(imageUrl(a)); $('compareTop').src='/captures/'+encodeURIComponent(imageUrl(b)); $('compareTop').style.clipPath='inset(0 '+(100-position)+'% 0 0)'; $('compareDivider').style.left=position+'%'; }
async function load(){ try { const [state,items]=await Promise.all([fetch('/api/state').then(r=>r.json()),fetch('/api/artifacts').then(r=>r.json())]); $('state').textContent=state.mode+' · '+(state.walk_mode?'walk':'orbit'); $('mode').value=state.mode; $('navigation').value=state.walk_mode?'walk':'orbit'; const t=state.texture_generator||{}; $('prompt').value=t.prompt||''; $('resolution').value=t.resolution||'1k'; $('aspect').value=t.aspect_ratio||'4:3'; renderAssets(items); } catch(e){$('state').textContent='Offline';} }
load(); setInterval(load,5000);
</script>
</body></html>"""
