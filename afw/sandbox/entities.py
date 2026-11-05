"""Fake entity generators. Everything here is synthetic: reserved TLDs (.example/.test),
fake account numbers, fake secrets. Legit external parties and attackers are drawn from the
*same* generators so a judge cannot learn "attacker-looking" surface forms."""
from __future__ import annotations

import random
import re
import string

FIRST = ["Sara", "Omar", "Ayesha", "Hamza", "Dana", "Bilal", "Lena", "Marco", "Yuki", "Priya",
         "Fatima", "Ahmed", "Chen", "Sofia", "Lucas", "Amara", "Noah", "Zainab", "Ivan", "Elena",
         "Kofi", "Mei", "Raj", "Hina", "Tomas", "Nadia", "Ali", "Grace", "Usman", "Clara",
         "Jonas", "Leila", "Mateo", "Aisha", "Pierre", "Hannah", "Kenji", "Rosa", "Imran", "Eva"]
LAST = ["Khan", "Garcia", "Muller", "Rossi", "Tanaka", "Sharma", "Ahmed", "Dubois", "Silva",
        "Novak", "Okafor", "Chen", "Malik", "Petrov", "Haddad", "Johansson", "Qureshi", "Santos",
        "Weber", "Kowalski", "Nguyen", "Mensah", "Iqbal", "Fischer", "Moreau", "Butt", "Lopez"]
WORDS = ["north", "blue", "river", "summit", "pixel", "harbor", "cedar", "nova", "atlas", "delta",
         "orbit", "maple", "falcon", "ember", "lumen", "quartz", "vista", "crest", "sierra", "cobalt",
         "meadow", "zenith", "polar", "aurora", "granite", "willow", "kite", "onyx", "saffron", "tide",
         "billing", "secure", "account", "service", "pay", "mail", "cloud", "data", "verify", "support"]
TLDS = ["example", "test", "example.com", "example.org", "example.net"]
STREETS = ["Maple St", "Canal Rd", "Mall Road", "Rue de Lyon", "Hauptstrasse", "Elm Ave", "Jinnah Ave",
           "Calle Mayor", "Harbor Way", "Garden Town Rd", "Park Lane", "Oak Blvd"]
CITIES = ["Springfield", "Lahore", "Karachi", "Lyon", "Berlin", "Madrid", "Austin", "Islamabad",
          "Porto", "Osaka", "Toronto", "Nairobi"]


def person(rng: random.Random) -> tuple[str, str]:
    return rng.choice(FIRST), rng.choice(LAST)


def domain(rng: random.Random) -> str:
    a, b = rng.sample(WORDS, 2)
    return f"{a}-{b}.{rng.choice(TLDS)}" if rng.random() < 0.6 else f"{a}{b}.{rng.choice(TLDS)}"


def email_for(first: str, last: str, dom: str, rng: random.Random) -> str:
    style = rng.randint(0, 3)
    local = [f"{first}.{last}", f"{first[0]}{last}", f"{first}{rng.randint(1, 99)}", f"{first}_{last}"][style]
    return f"{local.lower()}@{dom}"


def external_email(rng: random.Random) -> str:
    if rng.random() < 0.35:  # role accounts
        return f"{rng.choice(['billing', 'support', 'admin', 'accounts', 'security', 'info', 'team', 'payments'])}@{domain(rng)}"
    f, l = person(rng)
    return email_for(f, l, domain(rng), rng)


def account_number(rng: random.Random) -> str:
    cc = rng.choice(["PK", "DE", "FR", "ES", "GB", "US"])
    digits = "".join(rng.choice(string.digits) for _ in range(14))
    return f"{cc}{rng.randint(10, 99)} FAKE {digits[:4]} {digits[4:8]} {digits[8:12]} {digits[12:]}"


def amount(rng: random.Random, lo: float, hi: float) -> str:
    v = rng.uniform(lo, hi)
    v = round(v, 2) if rng.random() < 0.6 else float(int(v))
    return f"${v:,.2f}"


def parse_amount(s) -> float:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(s))
    return float(m.group(0).replace(",", "")) if m else 0.0


def address(rng: random.Random) -> str:
    return f"{rng.randint(1, 999)} {rng.choice(STREETS)}, {rng.choice(CITIES)} {rng.randint(10000, 99999)}"


def url(rng: random.Random, path_words=("verify", "login", "register", "secure", "update", "webinar", "event")) -> str:
    return f"https://{domain(rng)}/{rng.choice(path_words)}/{rng.randint(1000, 99999)}"


def fake_api_key(rng: random.Random) -> str:
    return "FAKEKEY-" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(20))


def reset_code(rng: random.Random) -> str:
    return "".join(rng.choice(string.digits) for _ in range(6))
