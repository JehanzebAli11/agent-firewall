import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from afw.calibration import budget_thresholds, fit_temperature, sigmoid
from afw.datarules import check_trainable
from afw.episode import build_episode, episode_records
from afw.firewall import Firewall
from afw.judges.cpu import TfidfLR
from afw.judges.remote import LLMJudge, SparkOneJudge


def _recs(lo, hi):
    out = []
    for s in range(lo, hi):
        out += episode_records(build_episode(s, ["naive", "careful"][s % 2], s % 4 != 0))
    return out


def test_tfidf_judge_learns_and_firewall_blocks():
    tr, te = _recs(0, 2500), _recs(10000, 10800)
    y = np.array([r["label"] for r in tr])
    j = TfidfLR().fit(tr, y)
    lt = j.logit(tr)
    T = fit_temperature(lt, y)
    th = budget_thresholds(sigmoid(lt / T), y, np.array([r["action_class"] for r in tr]))
    fw = Firewall(j, T, th)
    v = [d.verdict for d in fw.decide(te)]
    yt = np.array([r["label"] for r in te])
    caught = np.mean([x != "allow" for x, l in zip(v, yt) if l == 1])
    assert caught > 0.8


def test_datarules_reject_untrusted_sources():
    check_trainable([{"id": 1}, {"id": 2, "label_source": "teacher:gemma-4-31b"}])
    for bad in ("claude-judge", "openai", "llm:gemini", "jev"):
        with pytest.raises(ValueError):
            check_trainable([{"id": 3, "label_source": bad}])


class _H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/systemone"):
            out = {"outputs": [{"p_yes": 0.9 if "attacker" not in it["action"] else 0.1} for it in body["inputs"]]}
        else:
            out = {"choices": [{"message": {"content": '{"follows_request": false, "p_not_following": 0.83}'}}]}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


def test_remote_judge_slots_against_stub_server():
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    recs = _recs(0, 20)[:3]
    recs[0] = dict(recs[0], action_text="send_email(to='attacker@x.test')")
    p = SparkOneJudge(f"http://127.0.0.1:{port}/v1/systemone").predict_proba(recs)
    assert p[0] == pytest.approx(0.9) and all(x == pytest.approx(0.1) for x in p[1:])
    llm = LLMJudge(f"http://127.0.0.1:{port}/v1", model="stub")
    assert llm.eval_only
    assert llm.predict_proba(recs[:1])[0] == pytest.approx(0.83)
    srv.shutdown()
