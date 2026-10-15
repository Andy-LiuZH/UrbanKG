"""Japanese address normalisation and Area/Block key extraction.

Canonical address form (space separated)::

    <prefecture> <city> <town> [<N>丁目] <block>[-<rest>]

e.g. ``東京都 世田谷区 北沢 2丁目 24-3``. The Area key is the part up to the
chome, the Block key adds the first number after it.
"""

from __future__ import annotations

import re
from typing import Any

_ZEN_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
_DASHES = "−－ー‐―‑–—"
_KANJI_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

_PREF_RE = re.compile(r"^(東京都|北海道|京都府|大阪府|.{2,3}?県)")
_CITY_RE = re.compile(r"^(.+?郡.+?[町村]|.+?市.+?区|.+?[市区町村])")


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value) or str(value).strip() == ""


def kanji_to_int(text: str) -> int | None:
    """Strictly convert 一 .. 九十九 (or arabic digits) to an int; None if not a numeral."""
    if not text:
        return None
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"([一二三四五六七八九]?)(十?)([一二三四五六七八九]?)", text)
    if not match or (not match.group(2) and (match.group(1) and match.group(3))):
        return None
    tens_digit, ten, ones_digit = match.groups()
    if not ten:
        return _KANJI_NUM.get(ones_digit or tens_digit)
    return (_KANJI_NUM.get(tens_digit, 1)) * 10 + _KANJI_NUM.get(ones_digit, 0)


def split_chome(text: str) -> tuple[str, int] | None:
    """``与一一`` -> (``与一``, 1); ``北沢二`` -> (``北沢``, 2): the longest valid numeral suffix."""
    run = re.search(r"[一二三四五六七八九十\d]+$", text)
    if not run:
        return None
    digits = run.group(0)
    for start in range(len(digits)):
        number = kanji_to_int(digits[start:])
        town = text[: run.start() + start]
        if number and town:
            return town, number
    return None


def normalize_text(value: Any) -> str:
    """Half-width digits, unified dashes, single half-width spaces."""
    if _is_missing(value):
        return ""
    text = str(value).replace("　", " ").translate(_ZEN_DIGITS).replace("ヶ", "ケ").replace("ヵ", "カ")
    for dash in _DASHES:
        text = text.replace(dash, "-")
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", text)
    return re.sub(r"\s+", " ", text).strip()


def standardize_town(name: Any) -> str:
    """``北沢二丁目`` / ``北沢 2丁目`` -> ``北沢 2丁目``; other names unchanged."""
    text = normalize_text(name)
    if not text.endswith("丁目"):
        return text
    split = split_chome(text[:-2].rstrip())
    return f"{split[0].rstrip()} {split[1]}丁目" if split else text


def area_key(pref: Any, city: Any, town: Any) -> str:
    return " ".join(part for part in (normalize_text(pref), normalize_text(city), standardize_town(town)) if part)


def parse_address(address: Any, chome_towns: dict[str, set[int]] | None = None) -> dict[str, str | None] | None:
    """Split an address into prefecture/city/town/chome/block.

    Works for both the spaced Zenrin form (``東京都 世田谷区 北沢 2丁目 24-3``)
    and compact forms (``東京都世田谷区北沢二丁目24-3``). If the chome is not
    written (``北沢2-24-3``) and ``chome_towns["東京都 世田谷区 北沢"]`` contains 2
    (an existing chome of that town), the first number is read as the chome.
    """
    text = normalize_text(address)
    if not text:
        return None
    compact = text.replace(" ", "")

    pref_match = _PREF_RE.match(compact)
    if not pref_match:
        return None
    pref = pref_match.group(1)
    rest = compact[len(pref):]
    city_match = _CITY_RE.match(rest)
    if not city_match:
        return None
    city = city_match.group(1)
    rest = rest[len(city):]

    result: dict[str, str | None] = {"pref": pref, "city": city, "town": None, "chome": None, "block": None}

    chome_match = re.match(r"(.+?丁目)(.*)$", rest)
    split = split_chome(chome_match.group(1)[:-2]) if chome_match else None
    if split:
        result["town"], number = split
        result["chome"] = f"{number}丁目"
        numbers = re.findall(r"\d+", chome_match.group(2))
        result["block"] = numbers[0] if numbers else None
        return result

    plain_match = re.match(r"(.+?)(\d+(?:-\d+)*)", rest)
    if not plain_match:
        result["town"] = rest or None
        return result
    town = plain_match.group(1)
    numbers = plain_match.group(2).split("-")
    result["town"] = town
    town_key = f"{pref} {city} {town}"
    valid_chome = (chome_towns or {}).get(town_key, set())
    if len(numbers) >= 2 and int(numbers[0]) in valid_chome:
        result["chome"] = f"{int(numbers[0])}丁目"
        result["block"] = numbers[1]
    else:
        result["block"] = numbers[0]
    return result


def address_area_key(parsed: dict[str, str | None]) -> str | None:
    if not parsed or not parsed.get("town"):
        return None
    parts = [parsed["pref"], parsed["city"], parsed["town"], parsed.get("chome")]
    return " ".join(part for part in parts if part)


def address_block_key(parsed: dict[str, str | None]) -> str | None:
    area = address_area_key(parsed)
    if not area or not parsed.get("block"):
        return None
    return f"{area} {int(parsed['block'])}"


def chome_towns_from_area_keys(area_keys: list[str]) -> dict[str, set[int]]:
    """Town prefix (``pref city town``) -> chome numbers that exist as Areas."""
    towns: dict[str, set[int]] = {}
    for key in area_keys:
        match = re.fullmatch(r"(.+) (\d+)丁目", key)
        if match:
            towns.setdefault(match.group(1), set()).add(int(match.group(2)))
    return towns
