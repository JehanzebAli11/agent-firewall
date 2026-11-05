"""Run real LLM agents (OpenAI-compatible endpoint, e.g. local Gemma on the Spark) in the sandbox.
Writes code-labelled records (label_source marks whether they are trainable) and state-checked ASR.
Usage: AGENT_URL=http://localhost:8000/v1 AGENT_MODEL=<id> python scripts/run_llm_agents.py --episodes 2000 \
           [--firewall models/random_tfidf_lr_full.joblib] --out data/llm_records.jsonl.gz"""
import argparse, gzip, json, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from afw.episode import episode_records  # noqa
from afw.firewall import run_episode  # noqa
from afw.llm_agent import LLMAgent, build_llm_episode  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--start", type=int, default=5_000_000)  # disjoint from scripted seeds
    ap.add_argument("--inject-frac", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--firewall", default=None)
    ap.add_argument("--out", default="data/llm_records.jsonl.gz")
    a = ap.parse_args()
    agent = LLMAgent()
    fw = None
    if a.firewall:
        import joblib
        fw = joblib.load(a.firewall)

    def one(seed):
        inject = (seed % 100) < a.inject_frac * 100
        ep, unl = build_llm_episode(seed, agent, inject)
        recs = episode_records(ep)
        for r in recs:
            r["label_source"] = agent.label_source
        verdicts = [d.verdict for d in fw.decide(recs)] if (fw and recs) else None
        base = run_episode(ep, None)
        guarded = run_episode(ep, verdicts) if fw else None
        return recs, {"seed": seed, "injected": inject, "n_unlabeled": len(unl), "unlabeled": unl,
                      "attack_success_no_fw": base["attack_success"], "task_completed_no_fw": base["task_completed"],
                      "attack_success_fw": guarded["attack_success"] if guarded else None,
                      "task_completed_fw": guarded["task_completed"] if guarded else None,
                      **({k: getattr(ep.injection, k) for k in ("goal", "style", "lang", "trick", "position")} if ep.injection else {})}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    eps = []
    with ThreadPoolExecutor(a.workers) as ex, gzip.open(a.out, "wt") as f:
        for recs, meta in ex.map(one, range(a.start, a.start + a.episodes)):
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            eps.append(meta)
    inj = [e for e in eps if e["injected"]]
    summ = {"model": agent.model, "label_source": agent.label_source, "episodes": len(eps),
            "asr_no_fw": sum(e["attack_success_no_fw"] for e in inj) / max(1, len(inj)),
            "asr_fw": (sum(bool(e["attack_success_fw"]) for e in inj) / max(1, len(inj))) if fw else None,
            "clean_task_completion_no_fw": sum(e["task_completed_no_fw"] for e in eps if not e["injected"]) / max(1, len(eps) - len(inj)),
            "clean_task_completion_fw": (sum(bool(e["task_completed_fw"]) for e in eps if not e["injected"]) / max(1, len(eps) - len(inj))) if fw else None,
            "unlabeled_actions": sum(e["n_unlabeled"] for e in eps)}
    json.dump({"summary": summ, "episodes": eps}, open(a.out.replace(".jsonl.gz", "_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
