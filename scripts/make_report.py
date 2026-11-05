"""Figures + markdown tables from results/metrics.json (and results/wallet_metrics.json if present).
Usage: python scripts/make_report.py"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = "results/figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
MODELS = ["tfidf_lr", "mlp", "gbm"]
EXPS = ["random", "heldout_style", "heldout_lang", "heldout_trick", "heldout_task"]


def get(M, exp, model, view):
    for m in M:
        if (m["experiment"], m["model"], m["view"]) == (exp, model, view):
            return m
    return None


def pct(x):
    return "—" if x is None or x != x else f"{100 * x:.1f}%"


def reliability_fig(M, view="full", exp="random", name="reliability_agent.png", title="Agent-action judges"):
    ms = [get(M, exp, m, view) for m in MODELS]
    ms = [m for m in ms if m]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(4 * len(ms), 4), squeeze=False)
    for ax, m in zip(axes[0], ms):
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
        for key, col, lab in (("reliability_raw", S2, f"raw (ECE {m['ece_raw']:.4f})"), ("reliability_cal", S1, f"temperature-scaled (ECE {m['ece_cal']:.4f})")):
            xs, ys, ns = zip(*m[key])
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=4, label=lab)
        ax.set_title(f"{m['model']} (T={m['temperature']:.2f})", fontsize=10)
        ax.set_xlabel("predicted P(not from user's request)")
        ax.set_ylabel("observed fraction injected")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"{title}: reliability on test split ({exp}, {view} view)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


def asr_fig(M, model="tfidf_lr"):
    rows = []
    for exp in EXPS:
        for view in ("full", "blind"):
            m = get(M, exp, model, view)
            if m and "asr_no_firewall" in m["episodes"]:
                rows.append((f"{exp}\n{view}", m["episodes"]["asr_no_firewall"], m["episodes"]["asr_firewall"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rows)), 4))
    x = range(len(rows))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [r[1] * 100 for r in rows], w, color=S2, label="no firewall")
    ax.bar([i + w / 2 for i in x], [r[2] * 100 for r in rows], w, color=S1, label="with firewall")
    for i, r in enumerate(rows):
        ax.text(i + w / 2, r[2] * 100 + 0.4, f"{100 * r[2]:.1f}", ha="center", fontsize=8, color=INK)
        ax.text(i - w / 2, r[1] * 100 + 0.4, f"{100 * r[1]:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("attack success rate, % of injected episodes")
    ax.set_title(f"Attack success with and without the firewall ({model}; state-checked on the fake sites)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "asr_agent.png"), dpi=150)
    plt.close(fig)


def breakdown_fig(M, key, held_exp, fname, model="tfidf_lr"):
    series = []
    for exp, view, col, lab in (("random", "full", S1, "random split, full view"), (held_exp, "full", S2, f"{held_exp}, full view"),
                                ("random", "blind", S3, "random split, blind view")):
        m = get(M, exp, model, view)
        if m:
            series.append((m["breakdown"][key], col, lab))
    if not series:
        return
    cats = [c for c in series[0][0] if c != "none"]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(cats) * len(series) + 1.2))
    h = 0.8 / len(series)
    for j, (bd, col, lab) in enumerate(series):
        vals = [100 * bd.get(c, {}).get("catch_block_or_ask", float("nan")) for c in cats]
        ax.barh([i + j * h for i in range(len(cats))], vals, h * 0.92, color=col, label=lab)
    ax.set_yticks([i + h * (len(series) - 1) / 2 for i in range(len(cats))])
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlabel("injected actions stopped (block or ask), %")
    ax.set_xlim(0, 100)
    ax.set_title(f"Catch rate by {key} ({model}); held-out rows in orange are unseen in training")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=150)
    plt.close(fig)


def tables(M):
    L = ["| experiment | model | view | AUROC | ECE raw→cal | catch (block) | catch (block+ask) | false block | false ask | ASR no FW → FW | clean ep. w/ false block | task completion (clean, FW) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for exp in EXPS:
        for view in ("full", "blind"):
            for model in MODELS:
                m = get(M, exp, model, view)
                if not m:
                    continue
                o, e = m["overall"], m["episodes"]
                L.append(f"| {exp} | {model} | {view} | {m['auroc']:.4f} | {m['ece_raw']:.4f}→{m['ece_cal']:.4f} | {pct(o['catch_block'])} | "
                         f"{pct(o['catch_block_or_ask'])} | {pct(o['false_block'])} | {pct(o['false_ask'])} | "
                         f"{pct(e.get('asr_no_firewall'))} → {pct(e.get('asr_firewall'))} | {pct(e.get('clean_episode_false_block'))} | "
                         f"{pct(e.get('task_completion_clean_fw'))} |")
    return "\n".join(L)


def breakdown_table(M, key, exps, model="tfidf_lr", view="full"):
    ms = [(e, get(M, e, model, view)) for e in exps]
    ms = [(e, m) for e, m in ms if m]
    if not ms:
        return ""
    cats = sorted({c for _, m in ms for c in m["breakdown"][key]})
    L = [f"| {key} | " + " | ".join(f"{e}: catch / false block (n+ / n−)" for e, _ in ms) + " |", "|---|" + "---|" * len(ms)]
    for c in cats:
        cells = []
        for _, m in ms:
            r = m["breakdown"][key].get(c)
            cells.append("" if r is None else f"{pct(r['catch_block_or_ask'])} / {pct(r['false_block'])} ({r['n_pos']}/{r['n_neg']})")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def main():
    os.makedirs(FIG, exist_ok=True)
    out = ["# Results (auto-generated by scripts/make_report.py — do not edit by hand)\n"]
    if os.path.exists("results/metrics.json"):
        M = json.load(open("results/metrics.json"))
        reliability_fig(M)
        asr_fig(M)
        breakdown_fig(M, "style", "heldout_style", "catch_by_style.png")
        breakdown_fig(M, "lang", "heldout_lang", "catch_by_lang.png")
        breakdown_fig(M, "trick", "heldout_trick", "catch_by_trick.png")
        out += ["## Agent-action firewall\n", "Test split = seeds with seed % 10 ∈ {8, 9}. Thresholds set on the validation split "
                "(1% false-block budget per action class; ask band 2% general / 5% pay+sign). Asks are resolved by a simulated user "
                "who approves legitimate and denies injected actions.\n", tables(M), "\n### Breakdown by injection style (tfidf_lr, full view)\n",
                breakdown_table(M, "style", ["random", "heldout_style"]), "\n### By language\n", breakdown_table(M, "lang", ["random", "heldout_lang"]),
                "\n### By hiding trick\n", breakdown_table(M, "trick", ["random", "heldout_trick"]),
                "\n### By goal\n", breakdown_table(M, "goal", ["random"]), "\n### By position\n", breakdown_table(M, "position", ["random"]),
                "\n### By task (held-out tasks T16–T20 in the right column)\n", breakdown_table(M, "task_id", ["random", "heldout_task"]),
                "\n### Blind view (task + action only, no evidence) by style\n", breakdown_table(M, "style", ["random", "heldout_style"], view="blind")]
    if os.path.exists("results/wallet_metrics.json"):
        W = json.load(open("results/wallet_metrics.json"))
        out += ["\n## Wallet signature firewall\n", wallet_tables(W)]
        reliability_fig(W, view="wallet", exp="random", name="reliability_wallet.png", title="Wallet judges")
    open("results/RESULTS.md", "w").write("\n".join(out) + "\n")
    print("wrote results/RESULTS.md")


def wallet_tables(W):
    L = ["| experiment | model | AUROC | ECE raw→cal | catch (block) | catch (block+ask) | false block | drain success no FW → FW | attacker gain no FW → FW (test units) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in W:
        o, e = m["overall"], m["episodes"]
        L.append(f"| {m['experiment']} | {m['model']} | {m['auroc']:.4f} | {m['ece_raw']:.4f}→{m['ece_cal']:.4f} | {pct(o['catch_block'])} | "
                 f"{pct(o['catch_block_or_ask'])} | {pct(o['false_block'])} | {pct(e['drain_success_no_fw'])} → {pct(e['drain_success_fw'])} | "
                 f"{e['attacker_gain_no_fw']:,.0f} → {e['attacker_gain_fw']:,.0f} |")
    for m in W:
        if m["model"] == "tfidf_lr":
            L.append(f"\n**{m['experiment']} — by attack type (tfidf_lr)**\n\n| attack | catch (block+ask) | n |\n|---|---|---|")
            for k, r in m["breakdown"]["attack"].items():
                if k != "none":
                    L.append(f"| {k} | {pct(r['catch_block_or_ask'])} | {r['n_pos']} |")
            L.append(f"\n**{m['experiment']} — false blocks by intent (tfidf_lr)**\n\n| intent | false block | false ask | n legit |\n|---|---|---|---|")
            for k, r in m["breakdown"]["intent_type"].items():
                L.append(f"| {k} | {pct(r['false_block'])} | {pct(r['false_ask'])} | {r['n_neg']} |")
    return "\n".join(L)


if __name__ == "__main__":
    main()
