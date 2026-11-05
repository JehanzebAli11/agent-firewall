"""Wallet signature firewall: scenario generation (exact labels), decoding of
approve / setApprovalForAll / transfer / EIP-2612 Permit / EIP-712 typed data / personal_sign,
receipt-based simulation of balance & allowance changes on anvil, and on-chain attack execution."""
from __future__ import annotations

import os
import random
import time

from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3

from .chain import ART, MAX, USER_KEY, Chain

E18 = 10**18
TOKEN_ABI, NFT_ABI = ART["TestToken"]["abi"], ART["TestNFT"]["abi"]
APP_ABIS = {k: ART[k]["abi"] for k in ("SwapRouter", "Marketplace", "Airdrop", "Drainer")}

INTENTS = ["send_token", "send_eth", "swap", "list_nft", "login", "gasless_permit", "claim", "revoke", "explicit_address"]
ATTACKS = {
    "send_token": ["recipient_swap", "approve_drainer", "permit_phish"],
    "send_eth": ["recipient_swap"],
    "swap": ["wrong_spender", "operator_all", "permit_phish", "wrong_verified_router"],
    "list_nft": ["operator_all", "nft_approve_drainer"],
    "login": ["permit_phish", "operator_all", "approve_drainer"],
    "gasless_permit": ["wrong_spender_permit"],
    "claim": ["approve_drainer", "operator_all", "direct_transfer"],
    "revoke": ["approve_drainer"],
    "explicit_address": ["address_swap_drainer"],
}
ATTACK_TYPES = sorted({a for v in ATTACKS.values() for a in v})


def fmt_amt(v: int) -> str:
    if v >= MAX // 2:
        return "UNLIMITED"
    x = v / E18
    return f"{x:,.4f}".rstrip("0").rstrip(".")


def legit_amount(req) -> int:
    try:
        return int(req["tx"]["data"][-64:], 16)
    except Exception:
        return MAX


def lookalike(addr: str, rng: random.Random) -> str:
    """Address-poisoning lookalike: same first 3 and last 3 bytes."""
    b = bytes.fromhex(addr[2:])
    mid = bytes(rng.getrandbits(8) for _ in range(14))
    return Web3.to_checksum_address("0x" + (b[:3] + mid + b[-3:]).hex())


# ---------------------------------------------------------------- request builders
def _tx(ch: Chain, contract, fn, *args, value=0):
    data = contract.encode_abi(fn, args=list(args))
    return {"kind": "tx", "tx": {"to": contract.address, "data": data, "value": value}}


def _permit(ch: Chain, sym: str, spender: str, value: int, deadline: int):
    t = ch.c[sym]
    nonce = t.functions.nonces(ch.user).call()
    return {"kind": "typed", "typed": {
        "types": {"EIP712Domain": [{"name": "name", "type": "string"}, {"name": "version", "type": "string"},
                                   {"name": "chainId", "type": "uint256"}, {"name": "verifyingContract", "type": "address"}],
                  "Permit": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"},
                             {"name": "value", "type": "uint256"}, {"name": "nonce", "type": "uint256"},
                             {"name": "deadline", "type": "uint256"}]},
        "primaryType": "Permit",
        "domain": {"name": t.functions.name().call(), "version": "1", "chainId": 31337, "verifyingContract": t.address},
        "message": {"owner": ch.user, "spender": spender, "value": value, "nonce": nonce, "deadline": deadline}}}


def _login_typed(dapp: str, rng):
    return {"kind": "typed", "typed": {
        "types": {"EIP712Domain": [{"name": "name", "type": "string"}, {"name": "version", "type": "string"},
                                   {"name": "chainId", "type": "uint256"}],
                  "Login": [{"name": "app", "type": "string"}, {"name": "nonce", "type": "uint256"}]},
        "primaryType": "Login", "domain": {"name": dapp, "version": "1", "chainId": 31337},
        "message": {"app": dapp, "nonce": rng.randint(1, 10**9)}}}


ROUTERS = ["HarborSwap", "CedarDEX", "NovaSwap"]
MARKETS = ["PixelMarket", "OrbitNFT"]
DAPPS = ["HarborSwap", "PixelMarket", "LumenDrop", "OrbitNFT", "NovaSwap"]


def gen_scenario(ch: Chain, rng: random.Random, attack_p: float = 0.45) -> dict:
    it = rng.choice(INTENTS)
    sym = rng.choice(["TKA", "TKB", "TUSD"])
    amt = rng.randint(1, 400)
    friend = rng.choice(list(ch.friends))
    faddr = ch.friends[friend]
    drainer = rng.choice(ch.drainers).address
    deadline = 2_000_000_000  # fixed for reproducibility (> anvil block time)
    roman = rng.random() < 0.25
    use_unlisted = rng.random() < 0.25
    if use_unlisted:
        router, raddr = rng.choice(ch.unlisted)
    else:
        router = rng.choice(ROUTERS)
        raddr = ch.c[router].address
    market = rng.choice(MARKETS)
    nft_id = rng.randint(1, 5)
    dapp = rng.choice(DAPPS)
    claim = None  # what the dapp UI claims the request is
    look = []

    # ---- intent text + legit request
    if it == "send_token":
        intent = (f"{friend} ko {amt} {sym} bhej do ({faddr})." if roman else
                  rng.choice([f"Send {amt} {sym} to {friend} ({faddr}).", f"Pay {friend} {amt} {sym} at {faddr}."]))
        legit = _tx(ch, ch.c[sym], "transfer", faddr, amt * E18)
    elif it == "send_eth":
        eth = rng.choice([0.05, 0.1, 0.25, 0.5, 1.0])
        intent = f"{friend} ko {eth} ETH bhej do ({faddr})." if roman else f"Send {eth} ETH to {friend} ({faddr})."
        legit = {"kind": "tx", "tx": {"to": faddr, "data": "0x", "value": int(eth * E18)}}
    elif it == "swap":
        sym2 = rng.choice([s for s in ["TKA", "TKB", "TUSD"] if s != sym])
        where = f"{router} ({raddr})" if use_unlisted else router
        intent = (f"{router} par {amt} {sym} ko {sym2} mein swap karo." if roman and not use_unlisted else
                  f"Swap {amt} {sym} for {sym2} on {where}.")
        val = MAX if rng.random() < 0.4 else amt * E18
        legit = _tx(ch, ch.c[sym], "approve", raddr, val) if rng.random() < 0.85 else _tx(ch, ch.c[sym], "increaseAllowance", raddr, amt * E18)
        claim = f"{router} router"
    elif it == "list_nft":
        intent = (f"Meri Test Critters #{nft_id} ko {market} par {amt} TUSD mein list kar do." if roman else
                  f"List my Test Critters #{nft_id} on {market} for {amt} TUSD.")
        legit = (_tx(ch, ch.c["NFT"], "setApprovalForAll", ch.c[market].address, True) if rng.random() < 0.6
                 else _tx(ch, ch.c["NFT"], "approve", ch.c[market].address, nft_id))
        claim = f"{market} marketplace"
    elif it == "login":
        intent = f"{dapp} par apne wallet se sign in karo." if roman else rng.choice([f"Sign in to {dapp} with my wallet.", f"Log me into {dapp}."])
        legit = (_login_typed(dapp, rng) if rng.random() < 0.4 else
                 {"kind": "personal_sign", "message": f"Sign in to {dapp}\nNonce: {rng.randint(1, 10**9)}\nThis request will not trigger a transaction."})
        claim = f"{dapp} login"
    elif it == "gasless_permit":
        where = f"{router} ({raddr})" if use_unlisted else router
        intent = f"Approve {where} to spend {amt} {sym} with a gasless permit signature."
        legit = _permit(ch, sym, raddr, amt * E18 if rng.random() < 0.7 else MAX, deadline)
        claim = f"{router} permit"
    elif it == "claim":
        intent = "LumenDrop airdrop claim kar do." if roman else rng.choice(["Claim my LumenDrop airdrop.", "Collect the LumenDrop airdrop tokens for me."])
        legit = _tx(ch, ch.c["LumenDrop"], "claim")
        claim = "LumenDrop claim"
    elif it == "explicit_address":
        # hard negative: user explicitly names an unlisted contract / unknown address in the request
        name, uaddr = rng.choice(ch.unlisted)
        if rng.random() < 0.5:
            intent = f"Approve {uaddr} to spend {amt} {sym}, it's the {name} vault I'm moving to."
            legit = _tx(ch, ch.c[sym], "approve", uaddr, amt * E18 if rng.random() < 0.6 else MAX)
        else:
            eoa = Web3.to_checksum_address("0x" + bytes(rng.getrandbits(8) for _ in range(20)).hex())
            intent = f"Send {amt} {sym} to my new cold wallet {eoa}."
            legit = _tx(ch, ch.c[sym], "transfer", eoa, amt * E18)
            raddr = eoa
        claim = None
    elif it == "revoke":
        rr = rng.choice(["HarborSwap", "CedarDEX"])
        intent = f"{rr} ki {sym} allowance revoke kar do." if roman else f"Revoke {rr}'s allowance on my {sym}."
        legit = _tx(ch, ch.c[sym], "approve", ch.c[rr].address, 0)
        claim = f"{rr} revoke"
    else:
        raise ValueError(it)

    attack = None
    req = legit
    if rng.random() < attack_p:
        attack = rng.choice(ATTACKS[it])
        if attack == "recipient_swap":
            la = lookalike(faddr, rng)
            look.append(la)
            req = (_tx(ch, ch.c[sym], "transfer", la, amt * E18) if it == "send_token"
                   else {"kind": "tx", "tx": {"to": la, "data": "0x", "value": legit["tx"]["value"]}})
        elif attack == "approve_drainer":
            req = _tx(ch, ch.c[sym], "approve", drainer, MAX if rng.random() < 0.8 else amt * E18 * rng.choice([1, 10]))
        elif attack == "permit_phish":
            req = _permit(ch, sym, drainer, MAX if rng.random() < 0.8 else 1000 * E18, deadline)
        elif attack == "wrong_spender":
            req = _tx(ch, ch.c[sym], "approve", drainer, MAX if rng.random() < 0.5 else amt * E18)
        elif attack == "wrong_spender_permit":
            req = _permit(ch, sym, drainer, amt * E18 if rng.random() < 0.5 else MAX, deadline)
        elif attack == "operator_all":
            req = _tx(ch, ch.c["NFT"], "setApprovalForAll", drainer, True)
        elif attack == "nft_approve_drainer":
            req = _tx(ch, ch.c["NFT"], "approve", drainer, nft_id)
        elif attack == "wrong_verified_router":
            # misaligned but not a drain: approves a different (verified) router than the one the user named
            other = rng.choice([r for r in ROUTERS if r != router])
            req = _tx(ch, ch.c[sym], "approve", ch.c[other].address, legit_amount(legit))
        elif attack == "address_swap_drainer":
            req = (_tx(ch, ch.c[sym], "approve", drainer, MAX) if legit["tx"]["data"][:10] == ch.c[sym].encode_abi("approve", args=[drainer, 0])[:10]
                   else _tx(ch, ch.c[sym], "transfer", drainer, amt * E18))
        elif attack == "direct_transfer":
            req = _tx(ch, ch.c[sym], "transfer", drainer, rng.choice([amt, 1000]) * E18)
    return {"intent": intent, "intent_type": it, "roman_urdu": roman, "unlisted_dapp": use_unlisted,
            "request": req, "dapp_claim": claim, "label": int(attack is not None), "attack": attack or "none",
            "drainer": drainer, "lookalikes": look, "sym": sym, "nft_id": nft_id}


# ---------------------------------------------------------------- decoding
def decode(ch: Chain, req: dict) -> list[dict]:
    """Return a list of decoded effects (what the request asks for), independent of simulation."""
    if req["kind"] == "personal_sign":
        return [{"op": "sign_message", "text": req["message"][:200]}]
    if req["kind"] == "typed":
        t = req["typed"]
        if t["primaryType"] == "Permit":
            m = t["message"]
            return [{"op": "permit", "token": t["domain"]["verifyingContract"], "spender": m["spender"],
                     "amount": int(m["value"]), "deadline": int(m["deadline"])}]
        return [{"op": "sign_typed", "type": t["primaryType"], "fields": {k: str(v) for k, v in t["message"].items()}}]
    tx = req["tx"]
    to = Web3.to_checksum_address(tx["to"])
    data = tx["data"] if isinstance(tx["data"], str) else "0x" + bytes(tx["data"]).hex()
    if data in ("0x", ""):
        return [{"op": "native_transfer", "to": to, "amount": int(tx["value"])}]
    abis = [NFT_ABI, TOKEN_ABI] if to == ch.c["NFT"].address else [TOKEN_ABI, NFT_ABI]
    for abi in abis + list(APP_ABIS.values()):
        try:
            fn, args = ch.w3.eth.contract(address=to, abi=abi).decode_function_input(data)
        except Exception:
            continue
        name = fn.fn_name
        is_nft = to == ch.c["NFT"].address
        if name == "approve" and not is_nft:
            return [{"op": "approve", "token": to, "spender": args["sp"], "amount": int(args["v"])}]
        if name == "approve" and is_nft:
            return [{"op": "nft_approve", "token": to, "spender": args["sp"], "id": int(args["id"])}]
        if name == "increaseAllowance":
            return [{"op": "increase_allowance", "token": to, "spender": args["sp"], "amount": int(args["add"])}]
        if name == "transfer":
            return [{"op": "transfer", "token": to, "to": args["to"], "amount": int(args["v"])}]
        if name == "setApprovalForAll":
            return [{"op": "set_approval_for_all", "token": to, "operator": args["op"], "approved": bool(args["ok"])}]
        return [{"op": "call", "to": to, "fn": name, "args": {k: str(v) for k, v in args.items()}}]
    return [{"op": "unknown_call", "to": to, "selector": data[:10]}]


# ---------------------------------------------------------------- simulation
def _h(x):
    x = x.hex() if hasattr(x, "hex") else x
    return x if x.startswith("0x") else "0x" + x


_TRANSFER = _h(Web3.keccak(text="Transfer(address,address,uint256)"))
_APPROVAL = _h(Web3.keccak(text="Approval(address,address,uint256)"))
_APPROVAL_ALL = _h(Web3.keccak(text="ApprovalForAll(address,address,bool)"))


def _addr(topic) -> str:
    return Web3.to_checksum_address("0x" + _h(topic)[-40:])


def submit(ch: Chain, req: dict, sim_as_spender: bool = True):
    """Execute a request as the user would. Typed Permit signatures are signed with the user's throwaway
    key; for simulation the spender's permit() submission is replayed by a relayer. Returns receipt or None."""
    if req["kind"] == "tx":
        tx = req["tx"]
        h = ch.w3.eth.send_transaction({"from": ch.user, "to": tx["to"], "data": tx["data"], "value": tx["value"], "gas": 500000})
        return ch.w3.eth.wait_for_transaction_receipt(h, poll_latency=0.01), None
    if req["kind"] == "personal_sign":
        sig = Account.sign_message(encode_defunct(text=req["message"]), USER_KEY)
        return None, sig
    sig = Account.sign_typed_data(USER_KEY, full_message=req["typed"])
    if req["typed"]["primaryType"] == "Permit" and sim_as_spender:
        m, tok = req["typed"]["message"], ch.w3.eth.contract(address=req["typed"]["domain"]["verifyingContract"], abi=TOKEN_ABI)
        h = tok.functions.permit(m["owner"], m["spender"], m["value"], m["deadline"], sig.v,
                                 sig.r.to_bytes(32, "big"), sig.s.to_bytes(32, "big")).transact({"from": ch.relayer, "gas": 300000})
        return ch.w3.eth.wait_for_transaction_receipt(h, poll_latency=0.01), sig
    return None, sig


def simulate(ch: Chain, req: dict) -> dict:
    """Simulate on a snapshot and report changes to the USER's assets / permissions from event logs."""
    sid = ch.snapshot()
    eth_before = ch.w3.eth.get_balance(ch.user)
    try:
        rcpt, _ = submit(ch, req)
        status = 1 if rcpt is None else rcpt["status"]
        diffs = []
        if rcpt is not None:
            for lg in rcpt["logs"]:
                t0 = _h(lg["topics"][0])
                if t0 == _TRANSFER:
                    frm, to = _addr(lg["topics"][1]), _addr(lg["topics"][2])
                    if len(lg["topics"]) == 4:  # ERC721
                        if ch.user in (frm, to):
                            diffs.append({"kind": "nft_transfer", "token": lg["address"], "from": frm, "to": to, "id": int(_h(lg["topics"][3]), 16)})
                    elif ch.user in (frm, to):
                        diffs.append({"kind": "token_transfer", "token": lg["address"], "from": frm, "to": to, "amount": int(_h(lg["data"]), 16)})
                elif t0 == _APPROVAL and _addr(lg["topics"][1]) == ch.user:
                    if len(lg["topics"]) == 4:
                        diffs.append({"kind": "nft_approval", "token": lg["address"], "spender": _addr(lg["topics"][2]), "id": int(_h(lg["topics"][3]), 16)})
                    else:
                        diffs.append({"kind": "allowance", "token": lg["address"], "spender": _addr(lg["topics"][2]), "amount": int(_h(lg["data"]), 16)})
                elif t0 == _APPROVAL_ALL and _addr(lg["topics"][1]) == ch.user:
                    diffs.append({"kind": "operator", "token": lg["address"], "operator": _addr(lg["topics"][2]), "approved": int(_h(lg["data"]), 16) == 1})
        if req["kind"] == "tx" and req["tx"]["value"]:
            diffs.append({"kind": "eth_transfer", "to": Web3.to_checksum_address(req["tx"]["to"]), "amount": int(req["tx"]["value"])})
        # value at risk: balance exposed to non-user spenders/operators
        exposure = 0.0
        for d in diffs:
            if d["kind"] == "allowance":
                bal = ch.w3.eth.contract(address=d["token"], abi=TOKEN_ABI).functions.balanceOf(ch.user).call()
                exposure += min(d["amount"], bal) / E18
            elif d["kind"] == "operator" and d["approved"]:
                exposure += 100.0 * ch.w3.eth.contract(address=d["token"], abi=NFT_ABI).functions.balanceOf(ch.user).call()
            elif d["kind"] == "nft_approval":
                exposure += 100.0
            elif d["kind"] in ("token_transfer",) and d["from"] == ch.user:
                exposure += d["amount"] / E18
            elif d["kind"] == "eth_transfer":
                exposure += 2000.0 * d["amount"] / E18  # test valuation: 1 ETH = 2000 units
        return {"status": int(status), "diffs": diffs, "exposure": exposure,
                "eth_delta": (ch.w3.eth.get_balance(ch.user) - eth_before) / E18}
    except Exception as e:  # reverted
        return {"status": 0, "diffs": [], "exposure": 0.0, "error": str(e)[:120], "eth_delta": 0.0}
    finally:
        ch.revert(sid)


# ---------------------------------------------------------------- text for the judge
def describe(ch: Chain, sc: dict, dec: list, sim: dict) -> dict:
    L = ch.label
    def sym_of(addr):
        return ch.registry.get(Web3.to_checksum_address(addr), addr)
    dl = []
    for d in dec:
        op = d["op"]
        if op in ("approve", "increase_allowance", "permit"):
            dl.append(f"{op} token={sym_of(d['token'])} spender={d['spender']} [{L(d['spender'])}] amount={fmt_amt(d['amount'])}")
        elif op == "nft_approve":
            dl.append(f"nft_approve token={sym_of(d['token'])} spender={d['spender']} [{L(d['spender'])}] id={d['id']}")
        elif op == "transfer":
            dl.append(f"transfer token={sym_of(d['token'])} to={d['to']} [{L(d['to'])}] amount={fmt_amt(d['amount'])}")
        elif op == "native_transfer":
            dl.append(f"send ETH to={d['to']} [{L(d['to'])}] amount={fmt_amt(d['amount'])}")
        elif op == "set_approval_for_all":
            dl.append(f"setApprovalForAll collection={sym_of(d['token'])} operator={d['operator']} [{L(d['operator'])}] approved={d['approved']}")
        elif op == "sign_message":
            dl.append(f"personal_sign message={d['text']!r}")
        elif op == "sign_typed":
            dl.append(f"sign typed data {d['type']} {d['fields']}")
        elif op == "call":
            dl.append(f"call {d['fn']} on {d['to']} [{L(d['to'])}] {d['args']}")
        else:
            dl.append(f"{op} {d}")
    sl = []
    for d in sim["diffs"]:
        k = d["kind"]
        if k == "allowance":
            sl.append(f"allowance {sym_of(d['token'])}: your wallet -> {d['spender']} [{L(d['spender'])}] set to {fmt_amt(d['amount'])}")
        elif k == "operator":
            sl.append(f"operator {sym_of(d['token'])}: {d['operator']} [{L(d['operator'])}] can move ALL your NFTs = {d['approved']}")
        elif k == "nft_approval":
            sl.append(f"nft approval {sym_of(d['token'])} #{d['id']} -> {d['spender']} [{L(d['spender'])}]")
        elif k == "token_transfer":
            sign = "-" if d["from"] == ch.user else "+"
            other = d["to"] if d["from"] == ch.user else d["from"]
            sl.append(f"balance {sym_of(d['token'])} {sign}{fmt_amt(d['amount'])} ({'to' if sign == '-' else 'from'} {other} [{L(other)}])")
        elif k == "nft_transfer":
            sl.append(f"nft {sym_of(d['token'])} #{d['id']} moved {d['from']} -> {d['to']}")
        elif k == "eth_transfer":
            sl.append(f"balance ETH -{fmt_amt(d['amount'])} (to {d['to']} [{L(d['to'])}])")
    if not sl:
        sl.append("no on-chain state change" if sim["status"] else "transaction reverts")
    claim = f"DAPP SAYS: {sc['dapp_claim']}\n" if sc.get("dapp_claim") else ""
    text = f"INTENT: {sc['intent']}\n{claim}REQUEST: {sc['request']['kind']}\nDECODED: " + "; ".join(dl) + "\nSIMULATION: " + "; ".join(sl)
    return {"text": text, "decoded_lines": dl, "sim_lines": sl}


def wallet_features(ch: Chain, sc: dict, dec: list, sim: dict) -> list[float]:
    intent = sc["intent"].lower()
    d = dec[0]
    party = d.get("spender") or d.get("operator") or d.get("to") or ""
    lab = ch.label(party) if party else ""
    verified = ", verified" in lab
    name_in_intent = verified and lab.split(" ")[0].lower() in intent
    amt = d.get("amount")
    nums = [float(x) for x in __import__("re").findall(r"\b\d+(?:\.\d+)?\b", intent)]
    ratio = 0.0
    if amt is not None and nums:
        a = amt / E18
        ratio = min(10.0, a / max(nums)) if a < 1e30 else 10.0
    ops = ["approve", "increase_allowance", "permit", "nft_approve", "transfer", "native_transfer",
           "set_approval_for_all", "sign_message", "sign_typed", "call"]
    f = [1.0 if d["op"] == o else 0.0 for o in ops]
    f += [float(bool(party) and party.lower() in intent), float(verified), float(name_in_intent),
          float(lab.startswith("contact")), float(lab == "unknown address"),
          float(amt is not None and amt >= MAX // 2), ratio, float(sim["status"]),
          float(len(sim["diffs"]) > 0), min(sim["exposure"], 5000) / 5000,
          float(any(x["kind"] == "operator" for x in sim["diffs"])),
          float(sc["intent_type"] in ("login", "claim") and len(sim["diffs"]) > 0 and any(x["kind"] != "token_transfer" or x["from"] == ch.user for x in sim["diffs"]))]
    return f


WALLET_FEATURES = ["op_" + o for o in ["approve", "increase_allowance", "permit", "nft_approve", "transfer", "native_transfer",
                                       "set_approval_for_all", "sign_message", "sign_typed", "call"]] + \
    ["party_in_intent", "party_verified", "verified_name_in_intent", "party_is_contact", "party_unknown", "unlimited",
     "amount_ratio", "sim_ok", "has_effect", "exposure", "operator_grant", "effect_on_noeffect_intent"]


def attack_execute(ch: Chain, sc: dict, allowed: bool) -> dict:
    """Execute (if allowed) on a snapshot, then let the attacker try to drain. Measure attacker gain."""
    sid = ch.snapshot()
    try:
        watch = ch.attacker_controlled() + sc["lookalikes"]
        def holdings():
            tot = 0.0
            for sym in ch.token_syms():
                for a in watch:
                    tot += ch.c[sym].functions.balanceOf(a).call() / E18
            for i in range(1, 6):
                if ch.c["NFT"].functions.ownerOf(i).call() in watch:
                    tot += 100.0
            for a in sc["lookalikes"]:
                tot += 2000.0 * ch.w3.eth.get_balance(a) / E18
            return tot
        before = holdings()
        sig = None
        if allowed:
            try:
                rcpt, sig = submit(ch, sc["request"], sim_as_spender=False)
            except Exception:
                pass
        drainer = ch.w3.eth.contract(address=sc["drainer"], abi=APP_ABIS["Drainer"])
        if sig is not None and sc["request"]["kind"] == "typed" and sc["request"]["typed"]["primaryType"] == "Permit":
            m = sc["request"]["typed"]["message"]
            if m["spender"] == sc["drainer"]:
                try:
                    drainer.functions.drainWithPermit(sc["request"]["typed"]["domain"]["verifyingContract"], ch.user, m["value"], m["deadline"],
                                                      sig.v, sig.r.to_bytes(32, "big"), sig.s.to_bytes(32, "big")).transact({"from": ch.attacker, "gas": 400000})
                except Exception:
                    pass
        for sym in ch.token_syms():
            try:
                drainer.functions.drain(ch.c[sym].address, ch.user).transact({"from": ch.attacker, "gas": 200000})
            except Exception:
                pass
        try:
            drainer.functions.drainNFT(ch.c["NFT"].address, ch.user, [1, 2, 3, 4, 5]).transact({"from": ch.attacker, "gas": 600000})
        except Exception:
            pass
        gain = holdings() - before
        return {"attacker_gain": gain, "attack_success": gain > 1e-9}
    finally:
        ch.revert(sid)
