# HANDOFF — agent-firewall

Living status file. Updated after every milestone. (Session started 2026-11-05, cloud CPU box.)

## 1. Status
- [x] Sandbox harness (fake email/docs/shop/bank, 20 tasks, injection generator) — afw/sandbox/, tests/test_sandbox.py
- [x] Scripted agents + dataset generation — afw/episode.py, scripts/generate_data.py (120k episodes -> 123,876 records, 19,030 injected)
- [~] CPU judges + calibration + thresholds — code done (afw/judges/cpu.py, afw/calibration.py); full grid running (results/metrics.json)
- [ ] Results (ASR with/without firewall, false blocks, breakdowns, reliability, GIF)
- [~] Wallet signature firewall (anvil) — contracts + decode/simulate/drain done & tested (tests/test_wallet.py); 12k scenario generation running
- [ ] Tests, CI, README

## Stretch list
- BIPIA / CyberSecEval adapters (eval only)
