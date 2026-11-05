"""LinkedIn summary card (1200x627) and GitHub social preview (1280x640). Every number is read from results/*.json.
Usage: python scripts/make_summary_card.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_html import render  # noqa


def get(M, exp, model, view):
    return next(m for m in M if (m["experiment"], m["model"], m["view"]) == (exp, model, view))


M = json.load(open("results/metrics.json"))
W = json.load(open("results/wallet_metrics.json"))
r = get(M, "random", "tfidf_lr", "full")
ht = get(M, "heldout_task", "tfidf_lr", "full")
w = get(W, "random", "tfidf_lr", "wallet")
N = {
    "asr0": 100 * r["episodes"]["asr_no_firewall"], "asr1": 100 * r["episodes"]["asr_firewall"],
    "fb": 100 * r["overall"]["false_block"], "catch": 100 * r["overall"]["catch_block_or_ask"],
    "d0": 100 * w["episodes"]["drain_success_no_fw"], "d1": 100 * w["episodes"]["drain_success_fw"],
    "ht_fb": 100 * ht["overall"]["false_block"],
}

CSS = """
*{box-sizing:border-box}body{margin:0;width:%(W)spx;height:%(H)spx;background:#0b1020;color:#e5e7eb;font-family:Inter,Segoe UI,Arial,sans-serif;
padding:44px 56px;display:flex;flex-direction:column;gap:26px;overflow:hidden}
.top{display:flex;justify-content:space-between;align-items:flex-start}
h1{margin:0;font-size:46px;letter-spacing:-.02em;color:#fff}h1 span{color:#60a5fa}
.sub{color:#94a3b8;font-size:19px;margin-top:8px;max-width:820px}
.tag{font-size:14px;color:#94a3b8;border:1px solid #334155;border-radius:999px;padding:6px 14px;white-space:nowrap}
.flow{display:flex;align-items:stretch;gap:12px}
.step{flex:1;background:#131a2e;border:1px solid #23304d;border-radius:14px;padding:14px 16px;font-size:16px;line-height:1.3}
.step b{display:block;color:#fff;font-size:17px;margin-bottom:4px}.arrow{align-self:center;color:#475569;font-size:26px}
.v{display:flex;gap:8px;margin-top:6px}.pill{font-weight:800;font-size:13px;padding:4px 10px;border-radius:999px}
.al{background:#14532d;color:#86efac}.as{background:#713f12;color:#fde68a}.bl{background:#7f1d1d;color:#fca5a5}
.stats{display:flex;gap:16px}
.stat{flex:1;background:#131a2e;border:1px solid #23304d;border-radius:14px;padding:16px 18px}
.big{font-size:38px;font-weight:800;color:#fff;letter-spacing:-.02em}.big .ar{color:#64748b;font-weight:500}
.lab{color:#94a3b8;font-size:15px;margin-top:4px;line-height:1.3}
.foot{color:#64748b;font-size:14px;margin-top:auto}
.warn .big{color:#fbbf24}
"""

BODY = """
<div class="top"><div><h1>An <span>action firewall</span> for AI agents</h1>
<div class="sub">Before an agent sends, pays or signs, a calibrated judge asks: <i>does this action follow from the user's request?</i></div></div>
<div class="tag">open source · CPU-only study</div></div>
<div class="flow">
 <div class="step"><b>Agent proposes</b>send_email · transfer · buy · approve · permit</div><div class="arrow">→</div>
 <div class="step"><b>Judge + simulation</b>P(not from the request), code risk score, on-chain state diff</div><div class="arrow">→</div>
 <div class="step"><b>Code decides</b>threshold = 1%% false-block budget<div class="v"><span class="pill al">ALLOW</span><span class="pill as">ASK</span><span class="pill bl">BLOCK</span></div></div>
</div>
<div class="stats">
 <div class="stat"><div class="big">%(asr0).1f%% <span class="ar">→</span> %(asr1).1f%%</div><div class="lab">prompt-injection attack success on fake email / docs / shop / bank (scripted agents)</div></div>
 <div class="stat"><div class="big">%(d0).1f%% <span class="ar">→</span> %(d1).1f%%</div><div class="lab">wallet drains on a local chain, measured by running a fake drainer</div></div>
 <div class="stat warn"><div class="big">%(ht_fb).0f%%</div><div class="lab">false blocks on <b>unseen tasks</b> — the open problem I'm working on next</div></div>
</div>
<div class="foot">Synthetic attacks in 6 languages incl. Roman Urdu · 10 styles · 8 hiding tricks · %(fb).1f%% false blocks in distribution · github.com/JehanzebAli11/agent-firewall</div>
"""


def page(W_, H_):
    return f"<html><head><meta charset='utf-8'><style>{CSS % {'W': W_, 'H': H_}}</style></head><body>{BODY % N}</body></html>"


os.makedirs("results/figures", exist_ok=True)
for (wd, ht_, out) in [(1200, 627, "results/figures/linkedin_card.png"), (1280, 640, "results/figures/social_preview.png")]:
    img = render([page(wd, ht_)], width=wd, height=ht_, dsf=2)[0]
    img.save(out)
    print(out, img.size)
print(N)
