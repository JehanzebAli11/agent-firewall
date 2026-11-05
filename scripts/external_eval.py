"""EVAL-ONLY adapters for public prompt-injection benchmarks pulled from GitHub:
  * BIPIA (microsoft/BIPIA): email QA contexts, clean vs. with text attacks inserted at start/middle/end
  * CyberSecEval prompt injection (meta-llama/PurpleLlama): 251 direct/indirect injection cases

These benchmarks test *text* hijacking, not tool actions, so they are adapted as an out-of-domain
"evidence screen" transfer test: the action is held fixed (answer the user's question) and only the
evidence the agent read changes. We report how much P(not from user's request) rises when the evidence
contains an injection (AUROC clean vs injected) and the flag rate at our general-class thresholds.
Records are tagged label_source='external:*' so afw.datarules forbids training on them.

Usage: python scripts/external_eval.py [--model models/random_tfidf_lr_full.joblib]"""
import argparse
import json
import os
import random
import subprocess
import sys

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW = "https://raw.githubusercontent.com"
FILES = {"external/bipia/email_test.jsonl": f"{RAW}/microsoft/BIPIA/main/benchmark/email/test.jsonl",
         "external/bipia/text_attack_test.json": f"{RAW}/microsoft/BIPIA/main/benchmark/text_attack_test.json",
         "external/cse/prompt_injection.json": f"{RAW}/meta-llama/PurpleLlama/main/CybersecurityBenchmarks/datasets/prompt_injection/prompt_injection.json"}


def fetch():
    for path, url in FILES.items():
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            subprocess.run(["curl", "-sSLf", "-m", "120", "-o", path, url], check=True)


def rec(task, evidence, source, src_tag, label):
    action = {"type": "send_email", "args": {"to": "me@corp.example", "subject": "Answer", "body": "Here is the answer to your question."}}
    return {"id": f"{src_tag}-{random.random()}", "task_text": task, "action": action, "action_type": "send_email",
            "action_class": "general", "action_text": "reply_to_user(body='Here is the answer to your question.')",
            "source": source, "evidence": evidence[:700], "user_email": "me@corp.example", "org_domain": "corp.example",
            "home_address": "", "contacts": ["me@corp.example"], "risk": 0.4, "label": label, "label_source": f"external:{src_tag}"}


def bipia(rng):
    ctx = [json.loads(l) for l in open("external/bipia/email_test.jsonl")]
    att = json.load(open("external/bipia/text_attack_test.json"))
    attacks = [(cat, a) for cat, v in att.items() for a in v]
    out = []
    for c in ctx:
        q = c["question"].removeprefix("Q:").strip()
        body = c["context"]
        out.append((rec(q, body, "email", "bipia", 0), "clean"))
        for _ in range(6):
            cat, a = rng.choice(attacks)
            pos = rng.choice(["start", "middle", "end"])
            if pos == "start":
                b = a + "\n" + body
            elif pos == "end":
                b = body + "\n" + a
            else:
                i = len(body) // 2
                b = body[:i] + "\n" + a + "\n" + body[i:]
            # the judge sees a 700-char evidence window centred on the attack (like our agents' citations)
            k = b.find(a)
            out.append((rec(q, b[max(0, k - 200): k + 500], "email", "bipia", 1), cat))
    return out


def cse():
    d = json.load(open("external/cse/prompt_injection.json"))
    return [(rec(x["test_case_prompt"][:300], x["user_input"], "user input", "cse", 1), x["injection_variant"]) for x in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/random_tfidf_lr_full.joblib")
    a = ap.parse_args()
    fetch()
    fw = joblib.load(a.model)
    rng = random.Random(0)
    res = {"model": a.model, "note": "out-of-domain evidence-screen transfer; eval only"}
    b = bipia(rng)
    recs = [r for r, _ in b]
    p = fw.probs(recs)
    y = np.array([r["label"] for r in recs])
    th = fw.th["general"]
    res["bipia_email"] = {"n_clean": int((y == 0).sum()), "n_injected": int(y.sum()), "auroc": float(roc_auc_score(y, p)),
                          "flag_rate_injected_block": float((p[y == 1] >= th["block"]).mean()),
                          "flag_rate_injected_block_or_ask": float((p[y == 1] >= th["ask"]).mean()),
                          "flag_rate_clean_block": float((p[y == 0] >= th["block"]).mean()),
                          "by_category": {}}
    cats = sorted({c for _, c in b if c != "clean"})
    for c in cats:
        idx = [i for i, (_, cc) in enumerate(b) if cc == c]
        res["bipia_email"]["by_category"][c] = {"n": len(idx), "flag_block_or_ask": float((p[idx] >= th["ask"]).mean())}
    cs = cse()
    pc = fw.probs([r for r, _ in cs])
    res["cyberseceval_pi"] = {"n": len(cs), "flag_rate_block": float((pc >= th["block"]).mean()),
                              "flag_rate_block_or_ask": float((pc >= th["ask"]).mean()), "by_variant": {}}
    for v in sorted({c for _, c in cs}):
        idx = [i for i, (_, c) in enumerate(cs) if c == v]
        res["cyberseceval_pi"]["by_variant"][v] = {"n": len(idx), "flag_block_or_ask": float((pc[idx] >= th["ask"]).mean())}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/external_eval.json", "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
