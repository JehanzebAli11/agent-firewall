import numpy as np

from afw.calibration import budget_thresholds, fit_temperature, reliability, sigmoid


def test_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < sigmoid(z)).astype(int)
    T = fit_temperature(3 * z, y)  # over-confident logits
    assert 2.6 < T < 3.4
    _, ece_raw = reliability(sigmoid(3 * z), y)
    _, ece_cal = reliability(sigmoid(3 * z / T), y)
    assert ece_cal < ece_raw


def test_budget_thresholds_respect_false_block_budget():
    rng = np.random.default_rng(1)
    p = rng.random(10000)
    y = np.zeros(10000, int)
    cls = np.array(["general"] * 5000 + ["pay"] * 5000)
    th = budget_thresholds(p, y, cls)
    for c in ("general", "pay"):
        m = cls == c
        assert (p[m] > th[c]["block"]).mean() <= 0.0102
        assert th[c]["ask"] <= th[c]["block"]
    assert th["pay"]["ask"] < th["general"]["ask"]  # stricter: wider ask band for pay


def test_saturated_scores_do_not_block_everything():
    p = np.zeros(1000)
    th = budget_thresholds(p, np.zeros(1000, int), np.array(["general"] * 1000))
    from afw.firewall import Firewall
    fw = Firewall(None, 1.0, th)
    assert fw.decide_p(0.0, "general", 0.9) == "allow"
