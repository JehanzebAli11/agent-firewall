"""Local anvil chain with test tokens, test dapps and a fake drainer.
Only throwaway anvil dev keys are used; nothing here touches a real network."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time

from web3 import Web3

ART = json.load(open(os.path.join(os.path.dirname(__file__), "artifacts.json")))
MAX = 2**256 - 1
# anvil's well-known dev key #0 (public, throwaway)
USER_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
FRIEND_NAMES = ["Ayesha", "Bilal", "Dana", "Marco", "Yuki", "Priya", "Kofi"]


def anvil_bin() -> str:
    for c in (shutil.which("anvil"), os.path.expanduser("~/.foundry/bin/anvil")):
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError("anvil not found: install Foundry (see README)")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Chain:
    def __init__(self, port: int | None = None):
        self.port = port or _free_port()
        self.proc = subprocess.Popen([anvil_bin(), "--port", str(self.port), "--chain-id", "31337", "--accounts", "12",
                                      "--silent"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.w3 = Web3(Web3.HTTPProvider(f"http://127.0.0.1:{self.port}", request_kwargs={"timeout": 30}))
        for _ in range(100):
            try:
                if self.w3.is_connected():
                    break
            except Exception:
                pass
            time.sleep(0.1)
        acc = self.w3.eth.accounts
        self.user, self.deployer, self.attacker, self.relayer = acc[0], acc[1], acc[2], acc[3]
        self.friends = dict(zip(FRIEND_NAMES, acc[4:11]))
        self.c = {}  # name -> contract
        self.registry = {}  # address -> verified label (the wallet's known-contract list)
        self.drainers = []
        self.unlisted = []
        self._deploy()

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except Exception:
            self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ------------------------------------------------------------------ setup
    def deploy(self, kind, *args, sender=None):
        art = ART[kind]
        C = self.w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
        tx = C.constructor(*args).transact({"from": sender or self.deployer})
        addr = self.w3.eth.wait_for_transaction_receipt(tx, poll_latency=0.01)["contractAddress"]
        return self.w3.eth.contract(address=addr, abi=art["abi"])

    def _deploy(self):
        tokens = {"TKA": "Alpha Token", "TKB": "Beta Token", "TUSD": "Test Dollar"}
        for sym, name in tokens.items():
            t = self.deploy("TestToken", name, sym)
            self.c[sym] = t
            self.registry[t.address] = f"{name} ({sym})"
        self.c["NFT"] = self.deploy("TestNFT", "Test Critters")
        self.registry[self.c["NFT"].address] = "Test Critters (NFT)"
        for name in ["HarborSwap", "CedarDEX", "NovaSwap"]:
            r = self.deploy("SwapRouter")
            self.c[name] = r
            self.registry[r.address] = f"{name} router"
        for name in ["PixelMarket", "OrbitNFT"]:
            m = self.deploy("Marketplace")
            self.c[name] = m
            self.registry[m.address] = f"{name} marketplace"
        a = self.deploy("Airdrop", self.c["TKB"].address)
        self.c["LumenDrop"] = a
        self.registry[a.address] = "LumenDrop airdrop"
        for name in ["FreshSwap", "ZephyrDEX"]:  # legit but NOT in the wallet registry (new dapps)
            r = self.deploy("SwapRouter")
            self.c[name] = r
            self.unlisted.append((name, r.address))
        for _ in range(6):
            self.drainers.append(self.deploy("Drainer", sender=self.attacker))
        # balances
        for sym in tokens:
            self.c[sym].functions.mint(self.user, 1000 * 10**18).transact({"from": self.deployer})
            for rn in ["HarborSwap", "CedarDEX", "NovaSwap", "FreshSwap", "ZephyrDEX"]:
                self.c[sym].functions.mint(self.c[rn].address, 10**6 * 10**18).transact({"from": self.deployer})
        for i in range(1, 6):
            self.c["NFT"].functions.mint(self.user, i).transact({"from": self.deployer})
        # pre-existing modest approvals so "revoke" intents have something to revoke
        for rn in ["HarborSwap", "CedarDEX"]:
            for sym in ["TKA", "TKB"]:
                self.c[sym].functions.approve(self.c[rn].address, 100 * 10**18).transact({"from": self.user})

    # ------------------------------------------------------------------ helpers
    def rpc(self, method, params=None):
        r = self.w3.provider.make_request(method, params or [])
        if "error" in r:
            raise RuntimeError(r["error"])
        return r["result"]

    def snapshot(self):
        return self.rpc("evm_snapshot")

    def revert(self, sid):
        return self.rpc("evm_revert", [sid])

    def token_syms(self):
        return ["TKA", "TKB", "TUSD"]

    def label(self, addr: str) -> str:
        addr = Web3.to_checksum_address(addr)
        if addr in self.registry:
            return f"{self.registry[addr]}, verified"
        if addr == self.user:
            return "your wallet"
        for n, a in self.friends.items():
            if a == addr:
                return f"contact {n}"
        return "unknown address"

    def attacker_controlled(self) -> list[str]:
        return [self.attacker] + [d.address for d in self.drainers]
