#!/usr/bin/env python3
"""Verify the viewer no longer binds W or opens a worker picker."""

from __future__ import annotations

import pyglet

import viewer


def main() -> None:
	render_scene, camera = viewer.load_scene(viewer.ASSET_PATH)
	window = viewer.ModelWindow(render_scene, camera)
	closed = []
	try:
		window.close = lambda: closed.append(True)

		window.on_key_press(pyglet.window.key.W, 0)
		assert not closed, "W must not close the viewer"
		assert not hasattr(window, "worker_menu"), "worker menu state must be removed"

		window.on_key_press(pyglet.window.key.Q, 0)
		assert closed == [True], f"Q should close the viewer: {closed!r}"
		print("OK: W has no action and Q still closes the viewer.")
	finally:
		if not closed:
			window.close()


if __name__ == "__main__":
	main()
