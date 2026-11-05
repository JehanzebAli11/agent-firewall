import random

import pytest

try:
    from afw.wallet.chain import Chain, MAX, anvil_bin
    anvil_bin()
    HAVE_ANVIL = True
except Exception:  # pragma: no cover
    HAVE_ANVIL = False

pytestmark = pytest.mark.skipif(not HAVE_ANVIL, reason="anvil not installed")


@pytest.fixture(scope="module")
def ch():
    c = Chain()
    yield c
    c.close()


def test_decode_and_simulate_unlimited_approval_to_drainer(ch):
    from afw.wallet.core import _tx, attack_execute, decode, simulate
    d = ch.drainers[0].address
    req = _tx(ch, ch.c["TKA"], "approve", d, MAX)
    dec = decode(ch, req)
    assert dec[0]["op"] == "approve" and dec[0]["spender"] == d and dec[0]["amount"] == MAX
    sim = simulate(ch, req)
    assert any(x["kind"] == "allowance" and x["spender"] == d for x in sim["diffs"])
    assert ch.c["TKA"].functions.allowance(ch.user, d).call() == 0  # simulation reverted
    sc = {"request": req, "drainer": d, "lookalikes": []}
    assert attack_execute(ch, sc, True)["attack_success"]
    assert not attack_execute(ch, sc, False)["attack_success"]


def test_permit_and_operator_decoding(ch):
    from afw.wallet.core import _permit, _tx, attack_execute, decode, simulate
    d = ch.drainers[1].address
    req = _permit(ch, "TKB", d, MAX, 2_000_000_000)
    dec = decode(ch, req)
    assert dec[0]["op"] == "permit" and dec[0]["spender"] == d
    assert any(x["kind"] == "allowance" for x in simulate(ch, req)["diffs"])
    assert attack_execute(ch, {"request": req, "drainer": d, "lookalikes": []}, True)["attack_success"]
    req2 = _tx(ch, ch.c["NFT"], "setApprovalForAll", d, True)
    assert decode(ch, req2)[0]["op"] == "set_approval_for_all"
    assert any(x["kind"] == "operator" for x in simulate(ch, req2)["diffs"])


def test_legit_scenarios_never_profit_attacker(ch):
    from afw.wallet.core import attack_execute, gen_scenario
    rng = random.Random(0)
    for _ in range(25):
        sc = gen_scenario(ch, rng, attack_p=0.0)
        assert sc["label"] == 0
        assert not attack_execute(ch, sc, True)["attack_success"]
