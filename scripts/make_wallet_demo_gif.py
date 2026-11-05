"""Wallet demo GIF: a fake airdrop site asks for an UNLIMITED token approval to a drainer. The firewall decodes
and simulates it on a local anvil chain, the calibrated judge blocks it, and the drain is measured by actually
running the drainer with and without the firewall. A legit airdrop claim from the test split is shown allowed.
Usage: python scripts/make_wallet_demo_gif.py"""
import gzip
import html
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from afw.calibration import budget_thresholds, fit_temperature, sigmoid  # noqa
from afw.firewall import Firewall  # noqa
from afw.judges.cpu import TfidfLR  # noqa
from afw.wallet.chain import Chain  # noqa
from afw.wallet.core import attack_execute, gen_scenario  # noqa
from render_html import render, save_gif  # noqa

CSS = """
body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:#0f172a;color:#e2e8f0;width:1100px;height:640px}
.title{font-size:13px;color:#94a3b8;padding:6px 16px 0}
.wrap{display:flex;gap:16px;padding:10px 16px 16px;height:600px;box-sizing:border-box}
.left{flex:1;background:linear-gradient(160deg,#4c1d95,#1e1b4b);border-radius:12px;padding:22px;position:relative;overflow:hidden}
.left h1{margin:0 0 6px;font-size:26px;color:#fff}.left p{color:#ddd6fe;font-size:14px;margin:4px 0}
.btn{display:inline-block;margin-top:18px;background:#facc15;color:#1e1b4b;font-weight:800;padding:12px 26px;border-radius:999px;font-size:16px}
.pop{position:absolute;right:18px;bottom:18px;width:300px;background:#fff;color:#111;border-radius:12px;padding:14px;box-shadow:0 10px 30px rgba(0,0,0,.5);font-size:13px}
.pop .h{font-weight:700;font-size:15px;margin-bottom:6px}.pop .hex{font-family:Consolas,monospace;font-size:10.5px;color:#475569;word-break:break-all;background:#f1f5f9;padding:6px;border-radius:6px}
.pop .row{display:flex;gap:8px;margin-top:10px}.pop .b{flex:1;text-align:center;padding:7px;border-radius:8px;background:#e2e8f0}.pop .ok{background:#2563eb;color:#fff}
.right{flex:1.05;display:flex;flex-direction:column;gap:9px}
.card{background:#1e293b;border-radius:10px;padding:10px 13px;font-size:13.5px;line-height:1.35}
.k{color:#94a3b8;font-size:11.5px;text-transform:uppercase;letter-spacing:.05em}
.mono{font-family:Consolas,Menlo,monospace;font-size:12px;word-break:break-all}
.block{border:2px solid #ef4444}.allow{border:2px solid #22c55e}
.v{font-weight:800;font-size:20px}.vb{color:#f87171}.va{color:#4ade80}
"""


def load():
    return [json.loads(l) for l in gzip.open("data/wallet_records.jsonl.gz", "rt")]


def train(recs):
    tr = [r for r in recs if r["seed"] % 10 < 6]
    va = [r for r in recs if r["seed"] % 10 in (6, 7)]
    j = TfidfLR(view="wallet").fit(tr, np.array([r["label"] for r in tr]))
    lv = j.logit(va)
    yv = np.array([r["label"] for r in va])
    T = fit_temperature(lv, yv)
    return Firewall(j, T, budget_thresholds(sigmoid(lv / T), yv, np.array(["sign"] * len(va))))


def short(a):
    return a[:8] + "…" + a[-6:]


def frame(left, cards, caption):
    return (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body><div class='title'>agent-firewall demo — {html.escape(caption)}</div>"
            f"<div class='wrap'><div class='left'>{left}</div><div class='right'>{''.join(cards)}</div></div></body></html>")


def card(k, body, cls=""):
    return f"<div class='card {cls}'><div class='k'>{k}</div>{body}</div>"


def main():
    recs = load()
    fw = train(recs)
    test = [r for r in recs if r["seed"] % 10 >= 8 and r["intent_type"] == "claim"]
    atk = next(r for r in test if r["attack"] == "approve_drainer" and "UNLIMITED" in r["text"] and not r["roman_urdu"])
    ok = next(r for r in test if r["attack"] == "none" and not r["roman_urdu"])
    with Chain() as ch:
        sc_atk = gen_scenario(ch, random.Random(atk["seed"]))
        sc_ok = gen_scenario(ch, random.Random(ok["seed"]))
        assert json.dumps(sc_atk["request"], default=str) == json.dumps(atk["request"], default=str)
        d_atk, d_ok = fw.decide([atk, ok])
        no_fw = attack_execute(ch, sc_atk, True)
        with_fw = attack_execute(ch, sc_atk, d_atk.verdict == "allow")
    site = ("<h1>🪂 LumenDrop</h1><p>Season 2 airdrop is live!</p><p>Connect your wallet and claim 10 TKB.</p>"
            "<div class='btn'>Claim airdrop</div>")
    data = atk["request"]["tx"]["data"]
    pop = (f"<div class='pop'><div class='h'>Wallet: confirm transaction</div>"
           f"to {short(atk['request']['tx']['to'])}<div class='hex'>{data[:120]}…</div>"
           "<div class='row'><div class='b'>Reject</div><div class='b ok'>Confirm</div></div></div>")
    intent = card("user intent", html.escape(atk["intent"]))
    dec = card("firewall: decoded", f"<div class='mono'>{html.escape(atk['decoded'][0])}</div>")
    sim = card("firewall: simulated on a local fork", f"<div class='mono'>{html.escape(atk['simulation'][0])}</div>")
    th = d_atk.threshold["block"]
    ver = card("firewall verdict", f"<div class='v vb'>{d_atk.verdict.upper()}</div>P(not what the user asked) = {d_atk.p_misaligned:.3f} "
                                   "<br>threshold calibrated so ≤1% of legit requests get blocked", "block" if d_atk.verdict == "block" else "allow")
    out = card("measured on anvil (fake drainer actually run)",
               f"without firewall: drainer took <b>{no_fw['attacker_gain']:,.0f}</b> test tokens/NFT units<br>"
               f"with firewall: drainer took <b>{with_fw['attacker_gain']:,.0f}</b>")
    legit = card("legit request (claim() on the verified airdrop contract)",
                 f"<div class='v va'>{d_ok.verdict.upper()}</div>P(not what the user asked) ≈ 0", "allow" if d_ok.verdict == "allow" else "block")
    frames = [(frame(site, [intent], "the user just wants to claim an airdrop"), 1800),
              (frame(site + pop, [intent], "the wallet shows opaque calldata"), 2200),
              (frame(site + pop, [intent, dec], "decode the request"), 2000),
              (frame(site + pop, [intent, dec, sim], "simulate what it changes"), 2200),
              (frame(site + pop, [intent, dec, sim, ver], "judge effect vs intent"), 2800),
              (frame(site, [intent, ver, out, legit], f"held-out test scenarios {atk['id']} / {ok['id']}"), 4500)]
    imgs = render([h for h, _ in frames])
    save_gif(imgs, [d for _, d in frames], "results/figures/demo_wallet_drainer_blocked.gif")
    print({"attack": atk["id"], "verdict": d_atk.verdict, "p": d_atk.p_misaligned, "no_fw_gain": no_fw["attacker_gain"],
           "fw_gain": with_fw["attacker_gain"], "legit": ok["id"], "legit_verdict": d_ok.verdict, "legit_p": d_ok.p_misaligned})


if __name__ == "__main__":
    main()
