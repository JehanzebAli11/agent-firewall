"""Generate labelled wallet-request scenarios on local anvil chains (one per worker).
Each record: intent, request, decoded effects, simulated diffs, features, exact label, and the
on-chain attacker gain if the request were executed (measured by actually running the drainer).
Usage: python scripts/wallet_generate.py --n 12000 --workers 3"""
import argparse, gzip, json, os, random, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from afw.wallet.chain import Chain  # noqa
from afw.wallet.core import attack_execute, decode, describe, gen_scenario, simulate, wallet_features  # noqa


def work(args):
    lo, hi = args
    out = []
    with Chain() as ch:
        for seed in range(lo, hi):
            rng = random.Random(seed)
            sc = gen_scenario(ch, rng)
            dec = decode(ch, sc["request"])
            sim = simulate(ch, sc["request"])
            d = describe(ch, sc, dec, sim)
            ex = attack_execute(ch, sc, True)
            risk = min(1.0, 0.2 + sim["exposure"] / 1000.0)
            out.append({"domain": "wallet", "id": f"w{seed}", "seed": seed, "intent": sc["intent"],
                        "intent_type": sc["intent_type"], "attack": sc["attack"], "label": sc["label"],
                        "roman_urdu": sc["roman_urdu"], "unlisted_dapp": sc["unlisted_dapp"],
                        "request": sc["request"], "decoded": d["decoded_lines"], "simulation": d["sim_lines"],
                        "text": d["text"], "features": wallet_features(ch, sc, dec, sim), "risk": risk,
                        "action_class": "sign", "exposure": sim["exposure"], "sim_status": sim["status"],
                        "attacker_gain": ex["attacker_gain"], "attack_success_if_executed": ex["attack_success"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="data/wallet_records.jsonl.gz")
    a = ap.parse_args()
    step = 500
    chunks = [(i, min(i + step, a.n)) for i in range(0, a.n, step)]
    n = 0
    with Pool(a.workers) as pool, gzip.open(a.out, "wt") as f:
        for recs in pool.imap(work, chunks):
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
            n += len(recs)
            print(n, flush=True)
    print(json.dumps({"records": n, "file": a.out}))


if __name__ == "__main__":
    main()
