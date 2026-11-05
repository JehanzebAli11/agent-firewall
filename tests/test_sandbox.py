import base64

from afw.episode import build_episode, episode_records, goal_achieved
from afw.firewall import run_episode
from afw.sandbox.injection import GOALS, LANGS, STYLES, TRICKS
from afw.sandbox.sites import extract_text, visible_text
from afw.sandbox.tasks import TASKS


def test_all_tasks_build_and_are_deterministic():
    assert len(TASKS) == 26
    for i, tid in enumerate(sorted(TASKS)):
        a = episode_records(build_episode(100 + i, "naive", True, task_id=tid))
        b = episode_records(build_episode(100 + i, "naive", True, task_id=tid))
        assert a == b


def test_careful_agent_never_emits_injected_actions():
    for s in range(300):
        ep = build_episode(s, "careful", True)
        assert all(p.label == 0 for p in ep.proposals)


def test_injection_is_in_agent_text_and_hidden_from_human():
    for trick in TRICKS:
        ep = build_episode(7, "naive", True, task_id="T01_reply_offsite", trick=trick, style="direct", lang="en",
                           goal="exfil", position="middle")
        email = ep.injection.params["email"]
        txt = ep.page_text
        if trick == "base64":
            assert base64.b64encode(ep.injection.text.encode()).decode() in txt
        elif trick == "zero_width":
            assert email not in txt and email in txt.replace("​", "").replace("‌", "").replace("⁠", "")
        else:
            assert email in txt
        if trick in ("html_comment", "css_hidden", "white_text", "tiny_font"):
            assert email not in visible_text(ep.page_html)


def test_all_languages_and_styles_render():
    for lang in LANGS:
        for style in STYLES:
            ep = build_episode(3, "naive", True, task_id="T12_pay_bill", lang=lang, style=style, trick="none")
            assert ep.injection.text and ep.injection.lang == lang


def test_attack_success_is_state_based():
    hits = 0
    for s in range(400):
        ep = build_episode(s, "naive", True)
        mal = any(p.label for p in ep.proposals)
        res = run_episode(ep, None)
        if mal:
            hits += 1
            assert res["attack_success"], (ep.injection.goal, ep.task.task_id)
        else:
            assert not res["attack_success"]
        # blocking every injected action stops the attack and keeps the task
        res2 = run_episode(ep, ["block" if p.label else "allow" for p in ep.proposals])
        assert not res2["attack_success"]
        if not (ep.injection.replaces_legit and mal):
            assert res2["task_completed"]
    assert hits > 50


def test_clean_episode_completes_task():
    for s in range(100):
        ep = build_episode(s, "naive", False)
        assert run_episode(ep, None)["task_completed"]
