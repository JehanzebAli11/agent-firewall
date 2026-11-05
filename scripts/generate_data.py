"""Generate logged (task, page, action, label) records from scripted agents.
Usage: python scripts/generate_data.py --episodes 40000 --out data/records.jsonl.gz"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from afw.episode import build_episode, episode_records  # noqa: E402


def spec(seed):
    r = random.Random(seed ^ 0x5EED)
    agent = r.choices(["naive", "cautious", "careful"], [0.5, 0.25, 0.25])[0]
    inject = r.random() < 0.75
    return seed, agent, inject


def work(seed):
    seed, agent, inject = spec(seed)
    ep = build_episode(seed, agent, inject)
    recs = episode_records(ep)
    meta = {"seed": seed, "agent": agent, "injected": inject, "task_id": ep.task.task_id,
            "n_actions": len(recs), "n_malicious": sum(r["label"] for r in recs),
            "page_chars": len(ep.page_html)}
    if ep.injection:
        i = ep.injection
        meta.update(goal=i.goal, style=i.style, lang=i.lang, trick=i.trick, position=i.position)
    return recs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="data/records.jsonl.gz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n_rec = 0
    with Pool(os.cpu_count()) as pool, gzip.open(a.out, "wt") as f, \
            gzip.open(a.out.replace("records", "episodes"), "wt") as fe:
        for recs, meta in pool.imap(work, range(a.start, a.start + a.episodes), chunksize=200):
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            fe.write(json.dumps(meta) + "\n")
            n_rec += len(recs)
    h = hashlib.sha256(open(a.out, "rb").read()).hexdigest()[:16]
    print(json.dumps({"episodes": a.episodes, "records": n_rec, "file": a.out}))


if __name__ == "__main__":
    main()
