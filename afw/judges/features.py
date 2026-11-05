"""Judge inputs. Two views:
  full  : task + action + claimed source + evidence window the agent read the arguments from
  blind : task + action only (+ grounding against the user's address book / profile)
Structured grounding features are computed by code from the record (no labels, no origin field)."""
from __future__ import annotations

import re

import numpy as np

from ..sandbox.sites import ACTION_TYPES
from ..risk import SENSITIVE

ZW = re.compile("[​‌⁠]")
B64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")


def target_of(a: dict) -> str:
    g = a["args"]
    for k in ("to", "with", "to_account", "url", "value", "ship_to", "doc_id", "item"):
        if k in g:
            return str(g[k])
    return ""


def make_text(r: dict, view: str = "full") -> str:
    if r.get("domain") == "wallet":
        return r["text"]
    s = f"TASK: {r['task_text']}\nACTION: {r['action_text']}"
    if view == "full":
        s += f"\nSOURCE: {r['source']}\nEVIDENCE: {r['evidence']}"
    return s


STRUCT_NAMES = (["t_" + t for t in ACTION_TYPES] +
                ["cls_pay", "risk", "tgt_in_task", "tgt_in_contacts", "tgt_user_org", "tgt_is_home", "tgt_is_user",
                 "amount_in_task", "sensitive", "n_args_in_task",
                 # evidence-only features (zeroed in blind view)
                 "src_is_user", "tgt_in_evidence", "amount_in_evidence", "ev_zw", "ev_b64", "ev_len"])


def _norm(s: str) -> str:
    return ZW.sub("", s).lower()


def struct_features(r: dict, view: str = "full") -> np.ndarray:
    a = r["action"]
    task = r["task_text"].lower()
    ev = _norm(r["evidence"]) if view == "full" else ""
    tgt = target_of(a).lower()
    f = [1.0 if r["action_type"] == t else 0.0 for t in ACTION_TYPES]
    amt = str(a["args"].get("amount", "")).lower()
    args_in_task = sum(1 for v in a["args"].values() if len(str(v)) > 3 and str(v).lower() in task)
    f += [1.0 if r["action_class"] != "general" else 0.0, r["risk"],
          float(bool(tgt) and tgt in task), float(tgt in [c.lower() for c in r["contacts"]]),
          float(tgt.endswith("@" + r["org_domain"])), float(tgt == r["home_address"].lower()),
          float(tgt == r["user_email"].lower()), float(bool(amt) and amt in task),
          float(bool(SENSITIVE.search(r["action_text"]))), float(args_in_task)]
    if view == "full":
        f += [float(r["source"] == "user request"), float(bool(tgt) and tgt in ev), float(bool(amt) and amt in ev),
              float(bool(ZW.search(r["evidence"]))), float(bool(B64.search(r["evidence"]))), min(len(ev), 600) / 600]
    else:
        f += [0.0] * 6
    return np.asarray(f, dtype=np.float32)


def struct_matrix(recs, view="full") -> np.ndarray:
    if recs and recs[0].get("domain") == "wallet":
        return np.asarray([r["features"] + [r["risk"]] for r in recs], dtype=np.float32)
    return np.vstack([struct_features(r, view) for r in recs])
