"""Extract ABI + bytecode from forge build output into afw/wallet/artifacts.json.
Run: (cd wallet && forge build) && python scripts/wallet_artifacts.py"""
import json, os
out = {}
for sol, names in {"TestToken.sol": ["TestToken"], "TestNFT.sol": ["TestNFT"],
                   "Apps.sol": ["SwapRouter", "Marketplace", "Airdrop", "Drainer"]}.items():
    for n in names:
        j = json.load(open(f"wallet/out/{sol}/{n}.json"))
        out[n] = {"abi": j["abi"], "bytecode": j["bytecode"]["object"]}
json.dump(out, open("afw/wallet/artifacts.json", "w"))
print({k: len(v["bytecode"]) for k, v in out.items()})
