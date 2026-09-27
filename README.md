# mini_hemel

---
Reality itself is not fully defined until it is observed.

The idea that the world is not concrete until observed lies at the heart of the famous thought experiment known as Schrödinger's cat. Until the box is opened, quantum mechanics suggests the system exists in a superposition of possibilities, with the cat both alive and dead in a mathematical sense. The act of observation appears to transform these possibilities into a single reality, raising the profound question of whether the world is fully defined before it is observed. This idea echoes the philosophy of George Berkeley, who argued that "to be is to be perceived," and resonates with John Archibald Wheeler's notion of a "participatory universe," in which observers play an active role in bringing reality into concrete existence. Whether observation creates reality or merely reveals it remains one of the deepest mysteries in both physics and philosophy
---

Setup 

```bash
# On the host
xhost +local:
docker run -it --name god \
  -v /home/jason/work:/workspace \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --device /dev/dri \
  ubuntu:24.04

# Inside the container
apt-get update && apt-get install -y \
  python3 python3-pip \
  libgl1 libglx0 libegl1 libgl1-mesa-dri \
  libglu1-mesa \
  libx11-6 libxext6 libxi6 libxfixes3 libxrandr2 libxxf86vm1
pip config set global.break-system-packages true
cd /workspace/3d_test
pip install -r requirements.txt
# pyrender 0.1.45 pins the incompatible PyOpenGL 3.1.0 release.
pip install --upgrade --force-reinstall \
 'PyOpenGL>=3.1.10' 'PyOpenGL_accelerate>=3.1.10'
 ```

## Orbit mode navigation

The viewer currently uses **Google Earth-style orbit controls**.

| Action | Control |
| --- | --- |
| Orbit / tilt | Shift + left mouse button drag |
| Pan | Left mouse button drag |
| Zoom | Scroll wheel (or right mouse button drag up/down) |

### Comparison matrix

| Action | SketchUp | Blender (Default) | Google Earth |
| --- | --- | --- | --- |
| Orbit / tilt | Middle mouse button drag | Middle mouse button drag | Shift + left mouse button drag (or middle mouse button drag) |
| Pan | Shift + middle mouse button drag | Shift + middle mouse button drag | Left mouse button drag |
| Zoom | Scroll wheel | Scroll wheel | Scroll wheel (or right mouse button drag up/down) |

Double-click the scene to enter walk mode. Press `G` to return to orbit mode
and `F` to toggle fullscreen.