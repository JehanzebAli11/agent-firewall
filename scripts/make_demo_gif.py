"""Render a demo GIF of the firewall blocking an exfiltration attempt (real judge decisions on a
held-out test-split episode). Usage: python scripts/make_demo_gif.py [--model models/random_tfidf_lr_full.joblib]"""
import argparse
import html
import subprocess
import tempfile
import os
import sys

import imageio.v2 as imageio
import joblib
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from afw.episode import build_episode, episode_records  # noqa
from afw.firewall import run_episode  # noqa

CSS = """
body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:#0f172a;color:#e2e8f0;width:1100px;height:640px}
.wrap{display:flex;gap:16px;padding:16px;height:608px;box-sizing:border-box}
.left{flex:1.1;background:#fff;color:#111;border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.bar{background:#e2e8f0;padding:8px 12px;font-size:13px;color:#334155}
.page{padding:6px 16px;font-size:14px;overflow:hidden}
.page h2{font-size:18px;margin:8px 0}
.right{flex:1;display:flex;flex-direction:column;gap:10px}
.card{background:#1e293b;border-radius:10px;padding:12px 14px;font-size:14px;line-height:1.35}
.k{color:#94a3b8;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.mono{font-family:Consolas,Menlo,monospace;font-size:12.5px;word-break:break-all}
.block{border:2px solid #ef4444}.allow{border:2px solid #22c55e}.ask{border:2px solid #f59e0b}
.v{font-weight:700;font-size:20px}.vb{color:#f87171}.va{color:#4ade80}
.reveal span,.reveal div,.reveal img{outline:3px dashed #ef4444}
.hl{background:#fecaca!important;color:#7f1d1d!important;font-size:13px!important;display:block!important}
.title{font-size:13px;color:#94a3b8;padding:0 16px}
"""


def page_panel(ep, reveal):
    h = ep.page_html.split("<body>")[1].split("</body>")[0]
    if reveal:
        frag = ep.injection.fragment
        h = h.replace(frag, f"<div class='hl'>⚠ hidden text the agent reads: {html.escape(ep.injection.text)}</div>")
    return f"<div class='left'><div class='bar'>📧 fake-mail.test — {html.escape(ep.task.page.source)}</div><div class='page'>{h}</div></div>"


def frame(ep, cards, reveal=False, caption=""):
    right = "".join(cards)
    return (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"
            f"<div class='title'>agent-firewall demo — {html.escape(caption)}</div>"
            f"<div class='wrap'>{page_panel(ep, reveal)}<div class='right'>{right}</div></div></body></html>")


def card(k, body, cls=""):
    return f"<div class='card {cls}'><div class='k'>{k}</div>{body}</div>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/random_tfidf_lr_full.joblib")
    ap.add_argument("--out", default="results/figures/demo_blocked_exfiltration.gif")
    a = ap.parse_args()
    fw = joblib.load(a.model)
    # first TEST-split seed (seed % 10 >= 8) where the naive agent falls for a hidden Roman-Urdu exfil
    for s in range(8, 100000, 10):
        ep = build_episode(s, "naive", True, task_id="T01_reply_offsite", goal="exfil", trick="white_text",
                           style="user_voice", lang="ur_rom", position="middle")
        if any(p.label for p in ep.proposals):
            break
    recs = episode_records(ep)
    dec = fw.decide(recs)
    base, guarded = run_episode(ep, None), run_episode(ep, [d.verdict for d in dec])
    task = card("user task", html.escape(ep.task.text))
    frames = [(frame(ep, [task], caption="the user asks the agent to do a routine task"), 1800),
              (frame(ep, [task, card("agent", "reads the email…")], caption="the page looks harmless to a human"), 1800),
              (frame(ep, [task, card("agent", "reads the email…")], reveal=True,
                     caption="…but it contains white-on-white instructions (Roman Urdu)"), 2600)]
    log = [task]
    for r, d in zip(recs, dec):
        prop = card("agent proposes", f"<div class='mono'>{html.escape(r['action_text'][:220])}</div>")
        frames.append((frame(ep, log + [prop], reveal=True, caption="firewall checks every consequential action"), 1800))
        vcls = "vb" if d.verdict == "block" else "va"
        vb = (f"<div class='v {vcls}'>{d.verdict.upper()}</div>"
              f"P(not from user's request) = {d.p_misaligned:.3f} &nbsp; risk = {d.risk:.2f}<br>"
              f"block threshold ({r['action_class']}, 1% false-block budget) = {d.threshold['block']:.3f}")
        log = log + [card("agent proposes", f"<div class='mono'>{html.escape(r['action_text'][:220])}</div>"),
                     card("firewall", vb, d.verdict)]
        frames.append((frame(ep, log[-4:] if len(log) > 4 else log, reveal=True, caption="calibrated judge + code policy"), 2600))
    summary = card("outcome", f"without firewall: attack succeeded = <b>{base['attack_success']}</b><br>"
                              f"with firewall: attack succeeded = <b>{guarded['attack_success']}</b>, "
                              f"user task completed = <b>{guarded['task_completed']}</b>")
    frames.append((frame(ep, [task, summary], reveal=True, caption=f"test-split episode seed {ep.seed}"), 4000))

    # screenshots via the Node Playwright install (scripts/screenshot.js)
    tmp = tempfile.mkdtemp()
    paths = []
    for i, (h, _) in enumerate(frames):
        paths.append(os.path.join(tmp, f"f{i:02d}.html"))
        open(paths[-1], "w").write(h)
    env = dict(os.environ, NODE_PATH=os.environ.get("NODE_PATH", subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()))
    subprocess.run(["node", os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshot.js"), tmp] + paths, check=True, env=env)
    imgs = [Image.open(p.replace(".html", ".png")).convert("RGB") for p in paths]
    durs = [ms for _, ms in frames]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True)
    print({"seed": ep.seed, "verdicts": [d.verdict for d in dec], "p": [round(d.p_misaligned, 4) for d in dec],
           "no_fw": base, "fw": guarded, "out": a.out})


if __name__ == "__main__":
    main()
