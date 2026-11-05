"""Shared helper: render HTML strings to PIL images via the Node Playwright screenshot script."""
import os
import subprocess
import tempfile

from PIL import Image


def render(htmls, width=1100, height=640, dsf=1):
    tmp = tempfile.mkdtemp()
    paths = []
    for i, h in enumerate(htmls):
        paths.append(os.path.join(tmp, f"f{i:02d}.html"))
        with open(paths[-1], "w") as f:
            f.write(h)
    npm_root = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, NODE_PATH=os.environ.get("NODE_PATH", npm_root), SHOT_W=str(width), SHOT_H=str(height), SHOT_DSF=str(dsf))
    subprocess.run(["node", os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshot.js"), tmp] + paths, check=True, env=env)
    return [Image.open(p.replace(".html", ".png")).convert("RGB") for p in paths]


def save_gif(imgs, durations, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=durations, loop=0, optimize=True)
