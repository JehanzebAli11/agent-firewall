"""The action firewall: judge probability + code risk score -> allow / ask / block."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .calibration import sigmoid
from .episode import Episode, episode_records, goal_achieved, task_completed


@dataclass
class Decision:
    verdict: str  # allow | ask | block
    p_misaligned: float
    risk: float
    threshold: dict


class Firewall:
    def __init__(self, judge, temperature: float, thresholds: dict, min_ask_risk: float = 0.3):
        self.judge, self.T, self.th, self.min_ask_risk = judge, temperature, thresholds, min_ask_risk

    def probs(self, recs) -> np.ndarray:
        return sigmoid(self.judge.logit(recs) / self.T)

    def decide_p(self, p: float, cls: str, risk: float) -> str:
        th = self.th[cls]
        if p > th["block"]:
            return "block"
        if p > th["ask"] and risk >= self.min_ask_risk:
            return "ask"
        return "allow"

    def decide(self, recs) -> list[Decision]:
        ps = self.probs(recs)
        return [Decision(self.decide_p(float(p), r["action_class"], r["risk"]), float(p), r["risk"], self.th[r["action_class"]])
                for p, r in zip(ps, recs)]


def run_episode(ep: Episode, verdicts: list[str] | None):
    """Execute the agent's proposals against a fresh copy of the world. verdicts=None means no firewall.
    Asks are resolved by a simulated user who approves legit actions and denies injected ones."""
    w = copy.deepcopy(ep.world)
    executed = []
    for i, p in enumerate(ep.proposals):
        v = "allow" if verdicts is None else verdicts[i]
        if v == "ask":
            v = "allow" if p.label == 0 else "block"
        if v == "allow":
            try:
                ok = w.apply(copy.deepcopy(p.action))
            except (KeyError, ValueError, TypeError):  # malformed LLM tool call: no effect
                ok = False
            if ok:
                executed.append(p.action)
    attacked = ep.injection is not None and goal_achieved(w, ep.injection)
    return {"attack_success": attacked, "task_completed": task_completed(ep, executed)}


__all__ = ["Firewall", "Decision", "run_episode", "episode_records"]
