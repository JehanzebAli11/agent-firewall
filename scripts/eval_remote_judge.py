"""Evaluate a remote judge on the held-out test split: the Spark-1 System One endpoint (trainable slot:
calibrated on val, thresholded at the 1% false-block budget) or an eval-only LLM judge.
Usage: python scripts/eval_remote_judge.py --judge spark1 [--n-test 4000] [--data data/records.jsonl.gz]
       LLM_JUDGE_URL=... LLM_JUDGE_MODEL=... python scripts/eval_remote_judge.py --judge llm
       python scripts/eval_remote_judge.py --judge spark1 --wallet   # wallet records"""
import argparse, gzip, json, os, random, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from afw.calibration import budget_thresholds, fit_temperature, reliability, sigmoid  # noqa
from afw.judges.remote import LLMJudge, SparkOneJudge  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["spark1", "llm"], required=True)
    ap.add_argument("--wallet", action="store_true")
    ap.add_argument("--view", default="full")
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    path = a.data or ("data/wallet_records.jsonl.gz" if a.wallet else "data/records.jsonl.gz")
    recs = [json.loads(l) for l in gzip.open(path, "rt")]
    rng = random.Random(0)
    val = [r for r in recs if r["seed"] % 10 in (6, 7)]
    test = [r for r in recs if r["seed"] % 10 in (8, 9)]
    val, test = rng.sample(val, min(a.n_val, len(val))), rng.sample(test, min(a.n_test, len(test)))
    j = SparkOneJudge(view=a.view) if a.judge == "spark1" else LLMJudge(view=a.view)
    yv, yt = np.array([r["label"] for r in val]), np.array([r["label"] for r in test])
    lv, lt = j.logit(val), j.logit(test)
    T = fit_temperature(lv, yv)
    cls_v = np.array([r["action_class"] for r in val])
    th = budget_thresholds(sigmoid(lv / T), yv, cls_v)
    p = sigmoid(lt / T)
    v = []
    for x, r in zip(p, test):
        t = th[r["action_class"]]
        v.append("block" if x >= t["block"] else ("ask" if x >= t["ask"] and r["risk"] >= 0.3 else "allow"))
    v = np.array(v)
    out = {"judge": getattr(j, "name", a.judge), "eval_only": j.eval_only, "data": path, "n_val": len(val), "n_test": len(test),
           "temperature": T, "thresholds": th, "auroc": float(roc_auc_score(yt, p)),
           "ece_raw": reliability(sigmoid(lt), yt)[1], "ece_cal": reliability(p, yt)[1],
           "catch_block_or_ask": float(np.isin(v[yt == 1], ["block", "ask"]).mean()),
           "false_block": float((v[yt == 0] == "block").mean())}
    os.makedirs("results", exist_ok=True)
    fn = f"results/remote_{a.judge}{'_wallet' if a.wallet else ''}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
