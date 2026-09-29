#!/usr/bin/env python3
"""Verify the viewer's W/Q key bindings dispatch to the correct worker queue.

This does not require RabbitMQ, a GPU, or model weights: it builds the real
``ModelWindow`` against the bundled ``merged.gltf`` asset, monkeypatches the
RabbitMQ RPC call, and asserts that pressing `W` sends a request to the SDXL
ControlNet queue (using the SDXL worker's defaults) while `Q` continues to
use the original ControlNet queue. Run inside a virtual display, for example:

    xvfb-run -a python3 viewer_key_test.py
"""

from __future__ import annotations

import threading

import pyglet

import viewer


class _ImmediateThread:
    """Runs the target synchronously instead of spawning a real thread."""

    def __init__(self, target=None, args=(), daemon=None, name=None) -> None:
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


def main() -> None:
    render_scene, camera = viewer.load_scene(viewer.ASSET_PATH)
    window = viewer.ModelWindow(render_scene, camera)
    try:
        calls: list[tuple[str, str, dict]] = []

        def fake_request_worker(_view_path, _depth_path, *, queue, worker_label, request):
            calls.append((queue, worker_label, dict(request)))

        window._request_controlnet_worker = fake_request_worker
        window.save_current_view_files = lambda: ("view.png", "depth.png")

        original_thread = threading.Thread
        threading.Thread = _ImmediateThread
        try:
            window.on_key_press(pyglet.window.key.W, 0)
            window.on_key_press(pyglet.window.key.Q, 0)
        finally:
            threading.Thread = original_thread

        assert len(calls) == 2, f"expected 2 worker requests, got {calls!r}"

        w_queue, w_label, w_request = calls[0]
        assert w_queue == viewer.SDXL_CONTROLNET_QUEUE, w_queue
        assert w_label == "SDXL", w_label
        assert w_request["strength"] == viewer.SDXL_STRENGTH, w_request
        assert w_request["prompt"] == viewer.SDXL_PROMPT, w_request
        assert w_request["negative_prompt"] == viewer.SDXL_NEGATIVE_PROMPT, w_request

        q_queue, q_label, _q_request = calls[1]
        assert q_queue == viewer.CONTROLNET_QUEUE, q_queue
        assert q_label == "ControlNet", q_label
        assert q_queue != w_queue, "W and Q must use different worker queues"

        print("OK: W dispatches to the SDXL worker queue; Q is unaffected.")
    finally:
        window.close()


if __name__ == "__main__":
    main()
