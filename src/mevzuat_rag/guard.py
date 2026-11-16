"""Input guardrails: mask personal data before it reaches retrieval or a model, and flag prompt injection."""
from __future__ import annotations

import re
from dataclasses import dataclass


def _valid_tckn(digits: str) -> bool:
    if len(digits) != 11 or digits[0] == "0":
        return False
    d = [int(c) for c in digits]
    tenth = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    return d[9] == tenth and d[10] == sum(d[:10]) % 10


def _valid_luhn(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def _valid_iban(raw: str) -> bool:
    iban = re.sub(r"\s", "", raw).upper()
    if len(iban) != 26:
        return False
    rearranged = iban[4:] + iban[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1


_PATTERNS = (
    ("IBAN", re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE), _valid_iban),
    ("KART", re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"), lambda raw: _valid_luhn(re.sub(r"\D", "", raw))),
    ("TCKN", re.compile(r"\b\d{11}\b"), _valid_tckn),
    ("EPOSTA", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), None),
    ("TELEFON", re.compile(r"(?<!\d)(?:\+90|0)\s?5\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"), None),
)

_INJECTION = re.compile(
    r"(önceki|yukarıdaki|tüm)\s+(talimat|kural|yönerge)\w*\s+(yok say|unut|görmezden gel)"
    r"|ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+(instructions|rules)"
    r"|sistem\s+(prompt|istem|talimat)\w*\s*(nedir|göster|yaz|paylaş)"
    r"|(reveal|show|print)\s+(your\s+)?system\s+prompt"
    r"|you\s+are\s+now\s+|artık\s+\w+\s+gibi\s+davran",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardResult:
    text: str
    masked: tuple[str, ...]
    injection: bool


def mask_pii(text: str) -> tuple[str, tuple[str, ...]]:
    found: list[str] = []
    for label, pattern, validate in _PATTERNS:

        def replace(match: re.Match[str], label: str = label, validate=validate) -> str:
            if validate is not None and not validate(match.group(0)):
                return match.group(0)
            found.append(label)
            return f"[{label}]"

        text = pattern.sub(replace, text)
    return text, tuple(found)


def check(text: str) -> GuardResult:
    masked_text, labels = mask_pii(text)
    return GuardResult(masked_text, labels, bool(_INJECTION.search(text)))
