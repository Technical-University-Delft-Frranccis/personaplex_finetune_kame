"""Turn digits in spoken text into words, so a script is not rejected for "34C" or "10:30".

The scripts must contain no digits (the TTS and the word aligner read them unpredictably), and
the LLM keeps writing some anyway. Rejecting and regenerating costs a whole retry (~2 min on an
A40); converting costs nothing. Pure Python, no dependencies.

    >>> numbers_to_words("Seat 34C, boarding at 10:30, flight KL 1234, about 2.5 hours, 50%")
    'Seat thirty-four C, boarding at ten thirty, flight KL one two three four, about two point five hours, fifty percent'
"""

from __future__ import annotations

import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
         "fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD_IRREGULAR = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                  "nine": "ninth", "twelve": "twelfth"}


def cardinal(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else "")
    if n < 1000:
        return f"{_ONES[n // 100]} hundred" + (f" {cardinal(n % 100)}" if n % 100 else "")
    if n < 1_000_000:
        return f"{cardinal(n // 1000)} thousand" + (f" {cardinal(n % 1000)}" if n % 1000 else "")
    return digits(str(n))


def digits(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s if c.isdigit())


def ordinal(n: int) -> str:
    head, last = re.match(r"^(.*?)([a-z]+)$", cardinal(n)).groups()  # last word of "twenty-two"
    if last in _ORD_IRREGULAR:
        last = _ORD_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + last


def year(n: int) -> str:
    hi, lo = divmod(n, 100)
    if 2000 <= n <= 2009:
        return "two thousand" + (f" {cardinal(lo)}" if lo else "")
    if lo == 0:
        return f"{cardinal(hi)} hundred"
    return f"{cardinal(hi)} {'oh ' + cardinal(lo) if lo < 10 else cardinal(lo)}"


def numbers_to_words(text: str) -> str:
    # times: 10:30, 9:05 pm, 14:00
    def t(m: re.Match) -> str:
        h, mi, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower().replace(".", "")
        out = cardinal(h) + (" o'clock" if mi == 0 else (f" oh {cardinal(mi)}" if mi < 10 else f" {cardinal(mi)}"))
        return out + (f" {' '.join(ap)}" if ap else "")

    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?", t, text, flags=re.I)
    # money and percent
    cur = {"€": "euros", "$": "dollars", "£": "pounds"}
    text = re.sub(r"([€$£])\s?(\d+)(?:[.,](\d{2}))?",
                  lambda m: f"{cardinal(int(m.group(2)))} {cur[m.group(1)]}"
                            + (f" {cardinal(int(m.group(3)))}" if m.group(3) and m.group(3) != "00" else ""), text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: m.group(1) + " percent", text)
    # ordinals: 1st, 22nd
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: ordinal(int(m.group(1))), text, flags=re.I)
    # flight numbers: KL 1234 -> KL one two three four
    text = re.sub(r"\b([A-Z]{2})\s?(\d{2,4})\b", lambda m: f"{m.group(1)} {digits(m.group(2))}", text)
    # decimals
    text = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: f"{cardinal(int(m.group(1)))} point {digits(m.group(2))}", text)
    # years
    text = re.sub(r"\b(19[5-9]\d|20[0-3]\d)\b", lambda m: year(int(m.group(1))), text)
    # thousands separators 1,200 -> 1200
    text = re.sub(r"\b(\d{1,3})(?:,(\d{3}))+\b", lambda m: m.group(0).replace(",", ""), text)
    # everything else: plain cardinals, keep a space before an attached letter (seat 34C)
    def plain(m: re.Match) -> str:
        return cardinal(int(m.group(0)))

    text = re.sub(r"(?<=[A-Za-z])\d+", lambda m: " " + plain(m), text)  # gate B12 -> gate B twelve
    text = re.sub(r"\d+", plain, text)
    text = re.sub(r"([a-z])([A-Z])\b", r"\1 \2", text)  # "thirty-fourC" -> "thirty-four C"
    return text


if __name__ == "__main__":
    import doctest

    print(doctest.testmod())
