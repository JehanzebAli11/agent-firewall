# LinkedIn post — draft (edit freely)

Attach: `results/figures/demo_blocked_exfiltration.gif` (or the wallet GIF), or `results/figures/linkedin_card.png`.
Every number below comes from `results/metrics.json` / `results/wallet_metrics.json` / `results/external_eval.json`.

---

AI agents read untrusted pages and emails, then act on them: they send mail, move money, sign wallet requests.
One hidden line on a page can turn the agent against its user.

So I built an **action firewall** for agents. Before the agent sends, pays or signs, a calibrated judge asks one
question: *does this action follow from what the user actually asked?* Then plain code decides to allow, ask
or block. Thresholds come from a false-alarm budget: at most 1% of legitimate actions blocked.

What I tested it on:
🧪 Fake email, docs, shop and bank sites, plus a local blockchain with a fake wallet drainer
🌍 Injected instructions in 6 languages (including Roman Urdu), 10 styles and 8 hiding tricks (white text, zero-width characters, base64…)
⛓️ Wallet requests decoded and simulated before signing (approve, setApprovalForAll, EIP-2612 permits)

Results (scripted agents, synthetic attacks):
✅ Prompt-injection attack success: 22.4% → 0.0%, with 0.9% of legitimate actions blocked
✅ Wallet drains (measured by actually running the drainer): 97.9% → 0%
⚠️ The honest part: on tasks the judge never saw, it blocks 41% of legitimate actions, and it doesn't transfer to
public benchmarks like BIPIA yet. It has learned what normal looks like, not what "follows from the request" means.

Next: real LLM agents (local Gemma) in the harness, teacher-written tasks to fix that generalization gap, and a
stronger judge model.

Code, data generator and every result file: github.com/JehanzebAli11/agent-firewall

#AIsafety #AIagents #PromptInjection #LLMSecurity #Web3Security

---

Shorter alternative (≈60 words):

Built an action firewall for AI agents: before an agent sends, pays or signs, a calibrated judge checks "does
this follow from the user's request?" On synthetic attacks, injection success fell 22.4% → 0.0% with 0.9% false
blocks, and wallet drains fell 97.9% → 0%. Open problem: 41% false blocks on unseen tasks. Next: real Gemma agents.
github.com/JehanzebAli11/agent-firewall
