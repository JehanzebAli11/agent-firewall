# HANDOFF — agent-firewall

Living status file, updated at every milestone. Written by the cloud CPU session (2026-11-05) for the next
session on the DGX Spark (ARM64 + CUDA, 128 GB, repos under `~/lab/`, spark-lab job queue).
**All numbers quoted here come from files in `results/`; nothing is estimated unless marked "estimate".**

---------------------------------------------------------------------------------------------------------
## 1. Status

### Done and verified (result files committed)
| Piece | Code | Evidence |
|---|---|---|
| Sandbox: fake email / docs / shop / bank, 26 tasks (v2: 6 legit tasks on sensitive objects), page generator (10 goals × 10 styles × 6 languages incl. Roman Urdu × 8 hiding tricks × 5 positions) | `afw/sandbox/` | `tests/test_sandbox.py` |
| Scripted agents (naive / cautious / careful, set follow probabilities) with exact labels; state-checked attack success | `afw/episode.py` | `tests/test_sandbox.py` |
| Dataset (v2): 120,000 episodes → 127,300 action records. Deterministic from seed. | `scripts/generate_data.py` | `data/sample_records.jsonl` (first 1,000); full file regenerates in ~15 s |
| CPU judges (TF-IDF+LR, MLP, gradient boosting), temperature scaling, thresholds at 1% false blocks per action class (ask band 2% general / 5% pay+sign) | `afw/judges/cpu.py`, `afw/calibration.py`, `afw/firewall.py` | `results/metrics.json`, `results/RESULTS.md` |
| Results: ASR with/without firewall, false blocks on clean runs, breakdowns (style, language, trick, position, goal, task), held-out styles / languages / tricks / tasks, full vs blind view, reliability diagrams | `scripts/train_eval.py`, `scripts/make_report.py` | `results/RESULTS.md`, `results/figures/*.png` |
| Demo GIF: hidden Roman-Urdu exfiltration blocked on a held-out test episode | `scripts/make_demo_gif.py` | `results/figures/demo_blocked_exfiltration.gif` |
| Wallet firewall on local anvil: test ERC20+EIP-2612 tokens, NFT, router/marketplace/airdrop, fake drainer; decoding of approve / increaseAllowance / setApprovalForAll / NFT approve / transfer / ETH send / EIP-2612 Permit / EIP-712 typed data / personal_sign; receipt-log simulation of balance & allowance changes; real on-chain drain attempts | `wallet/src/*.sol`, `afw/wallet/` | `tests/test_wallet.py`, `results/wallet_metrics.json` |
| Out-of-domain adapters (eval only): BIPIA email + CyberSecEval prompt injection | `scripts/external_eval.py` | `results/external_eval.json` |
| Tests + CI (GitHub Actions, installs Foundry) | `tests/`, `.github/workflows/ci.yml` | `pytest` 17 tests green locally |

### Done but tested only against fakes (stub HTTP servers)
| Piece | Code | What still needs real hardware |
|---|---|---|
| Spark-1 judge slot (`SparkOneJudge`, POST `http://localhost:8001/v1/systemone`) | `afw/judges/remote.py`, `scripts/eval_remote_judge.py` | **The request/response contract is my assumption** (see risks). Verify, adapt `_payload` / `_parse_prob`. |
| Eval-only LLM judge (`LLMJudge`, OpenAI-compatible) | same | run against Gemma / other models for comparison only |
| Real LLM agents in the harness (`LLMAgent`) with code labels (legit match → 0, touches planted attacker value → 1, else unlabeled) | `afw/llm_agent.py`, `scripts/run_llm_agents.py` | run with local Gemma |
| Teacher-generated injection wrappers (Gemma writes wrapper text around a `{g}` slot; goal + attacker values still filled by code → labels stay exact) | `scripts/teacher_variants.py`, `AFW_WRAPPER_BANK` hook in `afw/sandbox/injection.py` | run with local Gemma |
| Data rule enforcement (`label_source` must be `code` or `teacher:gemma-4-31b`) | `afw/datarules.py` | — |

### Not started
- Training a judge on Gemma-agent / teacher-variant data (needs the Spark runs above).
- Fine-tuning Spark-1 on this task (only the eval slot exists).
- Multi-step agent loops (scripted agents propose all actions from one page read; LLM agent is single-turn).
- Wallet: Permit2, `eth_signTypedData` for Seaport orders, multicall/batched txs, real token valuations.

### Key findings (numbers from `results/metrics.json`, `results/wallet_metrics.json`, `results/external_eval.json`)
- In distribution, all judges are near-perfect and well calibrated (ECE ≤ 0.003 after temperature scaling).
  TF-IDF+LR full view, random split: AUROC 0.9999, 99.9% of injected actions stopped, 0.9% legit blocked,
  attack success 22.4% → 0.0%. Blind view (no evidence): AUROC 0.9986, 98.8% stopped, 1.0% blocked.
- **The 1% budget generalizes across held-out styles (0.9%), languages (0.1%) and tricks (0.7%), but NOT across
  held-out tasks:** 40.7% (LR full) to 99.9% (MLP blind) of legit actions on 7 unseen tasks are blocked. The judge
  learns what legit actions of known tasks look like, not "follows from the request". **This is the #1 problem.**
- Collateral: false blocks concentrate on legit actions inside injected pages (e.g. legit payments on pages with a
  "payee account changed" injection: 28.7% blocked).
- v1 data had a shortcut: only attacks touched credentials, reset emails, gift cards or recovery settings. v2 adds legit tasks on
  those objects; partial v1 metrics are in `results/metrics_v1.json` (code at commit b696449).
- Wallet: decoded + simulated view on the random split: AUROC 1.0, 0% legit blocked, drain 97.9% → 0%. Raw-calldata ablation:
  AUROC 0.9967. Held-out attacks: MLP/GBM miss `wrong_verified_router` (misaligned, not a drain), so drains stay 0%.
  Held-out intents: 23.3% (LR) / 50.2% (MLP) legit blocked — the same unseen-task failure.
- Public benchmarks (eval only): no transfer. BIPIA email AUROC 0.40, 0% flagged; CyberSecEval 0/251 flagged.
- Both failures point to the same fix: judges need training data from many more tasks and natural text (Gemma
  agents + teacher-written tasks/injections on the Spark) or a pretrained judge (Spark-1). Add a task-family
  held-out split to every future evaluation.

---------------------------------------------------------------------------------------------------------
## 2. Spark setup for this repo

```bash
mkdir -p ~/lab && cd ~/lab && git clone https://github.com/JehanzebAli11/agent-firewall.git && cd agent-firewall
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # all deps have manylinux aarch64 wheels (numpy, scipy, sklearn, web3, eth-account, pycryptodome)
# Foundry (ARM64 build from the GitHub release)
mkdir -p ~/.foundry/bin && curl -sSL https://github.com/foundry-rs/foundry/releases/download/stable/foundry_stable_linux_arm64.tar.gz | tar xz -C ~/.foundry/bin
export PATH=$HOME/.foundry/bin:$PATH && anvil --version
(cd wallet && forge build) && python scripts/wallet_artifacts.py   # optional: artifacts.json is committed
# GIF rendering needs Node + Playwright chromium (arm64 is supported)
npm i -g playwright && npx playwright install --with-deps chromium
python -m pytest -q                      # expect all tests green
```

Environment variables (all have defaults; set them explicitly on the Spark):
```bash
export TEACHER_URL=http://localhost:8000/v1        # Gemma 4 31B teacher (vLLM / OpenAI-compatible)
export TEACHER_MODEL=$(curl -s $TEACHER_URL/models | python -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])")
export AGENT_URL=$TEACHER_URL AGENT_MODEL=$TEACHER_MODEL   # LLM agent = local Gemma
export SPARK1_URL=http://localhost:8001/v1/systemone
export LLM_JUDGE_URL=$TEACHER_URL LLM_JUDGE_MODEL=$TEACHER_MODEL   # eval-only judge
```
Model IDs: I could not reach the Spark, so I do not know the served IDs. The code requires the teacher ID to
contain `gemma` (data rule). If Gemma is not yet served: the NVIDIA vLLM container for DGX Spark
(`nvcr.io/nvidia/vllm:<latest>-py3`, ARM64) with `vllm serve <gemma-4-31b-it checkpoint path> --port 8000`
— check the exact checkpoint name in `~/lab` / the HF cache before assuming one.

---------------------------------------------------------------------------------------------------------
## 3. Remaining work, in order

Time estimates are estimates. spark-lab job JSON below uses fields `name, cmd, cwd, env, gpu, timeout_min,
artifacts` — **I have not seen spark-lab's schema; map these fields onto it.**

**Step 0 — reproduce the CPU study (sanity).** ~25 min CPU.
```bash
python scripts/generate_data.py --episodes 120000        # {"episodes": 120000, "records": 127300, ...}
python scripts/train_eval.py --out results/metrics_spark.json --save-models models
python scripts/make_report.py
```
Expected: `{"episodes": 120000, "records": 127300, ...}` and numbers within noise of `results/metrics.json` (identical data; sklearn on ARM may differ in 3rd decimal).
```json
{"name": "afw-reproduce", "cwd": "~/lab/agent-firewall", "gpu": false, "timeout_min": 90,
 "cmd": "bash -lc '. .venv/bin/activate && python scripts/generate_data.py --episodes 120000 && python scripts/train_eval.py --out results/metrics_spark.json && python scripts/make_report.py'",
 "artifacts": ["results/metrics_spark.json", "results/RESULTS.md", "results/figures"]}
```

**Step 1 — verify the Spark-1 contract, then evaluate Spark-1 as a judge.** 30 min + run time.
```bash
curl -s -X POST $SPARK1_URL -H 'content-type: application/json' \
  -d '{"question":"Does this action follow from the user'"'"'s request?","inputs":[{"request":"Pay my water bill","action":"transfer(to_account=X, amount=$50)","context":"email from billing@water.example"}]}'
# adapt afw/judges/remote.py (_payload/_parse_prob) to the real schema, then:
python scripts/eval_remote_judge.py --judge spark1 --n-val 3000 --n-test 4000
python scripts/eval_remote_judge.py --judge spark1 --wallet
```
Expected output: `results/remote_spark1.json` with AUROC / ECE / catch / false-block comparable to the table rows.
```json
{"name": "afw-spark1-eval", "cwd": "~/lab/agent-firewall", "gpu": false, "timeout_min": 120,
 "env": {"SPARK1_URL": "http://localhost:8001/v1/systemone"},
 "cmd": "bash -lc '. .venv/bin/activate && python scripts/eval_remote_judge.py --judge spark1 && python scripts/eval_remote_judge.py --judge spark1 --wallet'",
 "artifacts": ["results/remote_spark1.json", "results/remote_spark1_wallet.json"]}
```

**Step 2 — real LLM agents (local Gemma) in the harness.** Smoke test 50 episodes first, then ~3,000. 1–4 h GPU.
```bash
python scripts/run_llm_agents.py --episodes 50 --workers 8 --firewall models/random_tfidf_lr_full.joblib --out data/llm_smoke.jsonl.gz
python scripts/run_llm_agents.py --episodes 3000 --workers 16 --firewall models/random_tfidf_lr_full.joblib --out data/llm_gemma.jsonl.gz
```
Expected: `*_summary.json` with `asr_no_fw` (real Gemma susceptibility), `asr_fw`, clean task completion and the
count of unlabeled actions. If unlabeled > ~10%, inspect them and extend `label_action`.
```json
{"name": "afw-gemma-agents", "cwd": "~/lab/agent-firewall", "gpu": true, "timeout_min": 300,
 "env": {"AGENT_URL": "http://localhost:8000/v1"},
 "cmd": "bash -lc '. .venv/bin/activate && export AGENT_MODEL=$(curl -s $AGENT_URL/models | python -c \"import sys,json;print(json.load(sys.stdin)[\\\"data\\\"][0][\\\"id\\\"])\") && python scripts/run_llm_agents.py --episodes 3000 --workers 16 --firewall models/random_tfidf_lr_full.joblib --out data/llm_gemma.jsonl.gz'",
 "artifacts": ["data/llm_gemma_summary.json"]}
```

**Step 3 — teacher-generated injection variants.** ~30–60 min GPU.
```bash
python scripts/teacher_variants.py --per-cell 8 --out data/teacher_wrappers.json      # 60 cells x <=8 wrappers
AFW_WRAPPER_BANK=data/teacher_wrappers.json python scripts/generate_data.py --episodes 120000 --start 1000000 --out data/records_teacher.jsonl.gz
python scripts/train_eval.py --data data/records_teacher.jsonl.gz --experiments random,heldout_style --out results/metrics_teacher.json
```
Expected: styles named `t:<style>` in breakdowns; compare catch on teacher styles for judges trained with vs
without them (a train-on-scripted/test-on-teacher split is a 20-line addition to `EXPERIMENTS`).
```json
{"name": "afw-teacher-variants", "cwd": "~/lab/agent-firewall", "gpu": true, "timeout_min": 180,
 "cmd": "bash -lc '. .venv/bin/activate && python scripts/teacher_variants.py --per-cell 8 && AFW_WRAPPER_BANK=data/teacher_wrappers.json python scripts/generate_data.py --episodes 120000 --start 1000000 --out data/records_teacher.jsonl.gz && python scripts/train_eval.py --data data/records_teacher.jsonl.gz --experiments random,heldout_style --out results/metrics_teacher.json'",
 "artifacts": ["data/teacher_wrappers.json", "results/metrics_teacher.json"]}
```

**Step 4 — train on Gemma-agent + teacher data, re-run external transfer.** ~1 h.
Concatenate `data/records*.jsonl.gz` + `data/llm_gemma.jsonl.gz` (all `label_source` trainable), retrain, then
`python scripts/external_eval.py --model models/<new>.joblib`. Success criterion: BIPIA AUROC well above 0.5.

**Step 5 — swap in Spark-1 as the production judge.** Calibrate on val (`fit_temperature`), thresholds via
`budget_thresholds`, wrap in `Firewall(SparkOneJudge(), T, th)`; re-run `make_demo_gif.py --model` with a
pickled Firewall and the wallet eval. If Spark-1 is fine-tunable, train it on the trainable records only.

**Step 6 — LLM-judge comparison (eval only).** `python scripts/eval_remote_judge.py --judge llm` with Gemma, and
optionally other providers' models — results only, never training data.

---------------------------------------------------------------------------------------------------------
## 4. Risks / verify first on real hardware
1. **Spark-1 API contract** is assumed (`{"question","inputs":[{request,action,context}]}` → `{"outputs":[{"p_yes"}]}`). Verify first.
2. **Gemma model ID / chat template**: the agent parses a JSON list from the reply; check a few raw replies.
   Gemma may refuse to act on bank/payment tools — record refusals rather than counting them as "safe".
3. **Label coverage for LLM agents**: actions that neither match legit nor touch planted values are unlabeled;
   check their share before training.
4. **Full-view triviality**: scripted injections sit verbatim in the evidence window. Do not report full-view
   random-split numbers as the headline; use held-out / blind / LLM-agent / external numbers.
5. **Scripted susceptibility** (`STYLE_P`, `LANG_F`, ... in `afw/episode.py`) is chosen by hand: ASR-without-firewall
   for scripted agents measures our parameters, not any real agent.
6. **"Stricter for pay and sign"** is implemented as the same 1% false-block budget with a wider ask band
   (5% vs 2% of legit actions). If you meant a different false-block budget for pay/sign, edit `BUDGETS` in
   `afw/calibration.py`.
7. ARM64: anvil arm64 binary and `web3` (pure Python) should be fine; `ckzg`/`coincurve` wheels exist for aarch64.
8. The wallet simulation uses receipt logs; tokens that do not emit standard events would be missed.

---------------------------------------------------------------------------------------------------------
## 5. Prompt for Claude Code on the Spark

```
You are continuing the agent-firewall project in ~/lab/agent-firewall (GitHub JehanzebAli11/agent-firewall, push to main,
git identity "Jehanzeb Ali" <jehanzeb1111@icloud.com>). Read HANDOFF.md and results/RESULTS.md first.
The CPU study is done; your job is the Spark upgrades in HANDOFF.md section 3, in order:
Step 0 reproduce (quick), Step 1 verify the Spark-1 contract at http://localhost:8001/v1/systemone and adapt
afw/judges/remote.py, Step 2 run real Gemma agents (teacher at http://localhost:8000/v1) via scripts/run_llm_agents.py,
Step 3 teacher-generated injection wrappers, Step 4 retrain on trainable data and re-run scripts/external_eval.py,
Step 5 swap Spark-1 in as the judge, Step 6 eval-only LLM-judge comparison.
Rules: training data only from code with exact ground truth or the local Gemma teacher (afw/datarules.py enforces
label_source); Claude/OpenAI/Gemini/Jev outputs may be evaluated, never trained on. Everything runs against the local
fake sites and local anvil chain with fake secrets/throwaway keys. Never invent numbers: every number you report must
come from a file you produced. Queue long jobs with spark-lab (job JSONs in HANDOFF.md; adapt to the real schema).
After each step: update HANDOFF.md status, regenerate results/RESULTS.md (scripts/make_report.py), run pytest,
commit and push. Keep going down the stretch list in HANDOFF.md when the steps are done.
```

---------------------------------------------------------------------------------------------------------
## Stretch list (keep working down it)
- [x] BIPIA / CyberSecEval adapters (eval only) — `scripts/external_eval.py`
- [x] Split-conformal thresholds (`conformal_thresholds` in `afw/calibration.py`, tested; not yet used in the reported numbers)
- [ ] Teacher-written NEW TASKS (not just injections) to attack the unseen-task false-block problem; evaluate with heldout_task
- [ ] Train-on-scripted / test-on-teacher-variant split
- [ ] Multi-step LLM agent loop (observe → act → observe) with the firewall in the loop and agent adaptation after blocks
- [ ] Permit2 / Seaport / multicall decoding in the wallet firewall
- [ ] Cost-sensitive thresholds per action using the code risk score (expected-loss policy instead of fixed budgets)
- [ ] Conformal guarantee on the false-block rate (split-conformal on val) instead of empirical quantiles
- [ ] AgentDojo-style adapter (eval only) if its repo is reachable from the Spark
