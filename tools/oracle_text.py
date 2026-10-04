"""Oracle text helpers shared by TRAINING oracles and the LIVE server.

Keep ONE copy of this file, byte-identical, in both repos:
    personaplex-finetune/tools/oracle_text.py            (imported by tools/generate_oracle_local.py)
    personaplex-klm/moshi/moshi/utils/oracle_text.py     (imported by moshi/server_oracle.py)
It is stdlib only with no relative imports, so the same file works in both. Anything that changes what the
back-end LLM sees or how its text is cleaned lives here, so training oracles and runtime oracles cannot drift.
"""
from __future__ import annotations

import re

MAX_WORDS = 30

# Appended to the LAST user message. The history ends in the crew member's unfinished sentence and a small model
# otherwise just continues it ("crew: I" -> "I can offer you earplugs..."), i.e. it writes the crew's words.
ORACLE_TAIL = (
    "\n\n(Reply with only what the passenger says next. The crew member's last sentence may be "
    "unfinished: do not complete it, and never write the crew's words.)"
)


def normalize_crew_text(text: str) -> str:
    """Make script text look like Vosk output (lower case, no punctuation, apostrophes kept)."""
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def build_user_message(conversation: str, hint_block: str = "") -> str:
    """conversation: lines 'passenger: ...' / 'crew: ...'. Trailing spaces are removed so the live server's
    'text ' + '\\n' bookkeeping and the training transcript render identically."""
    lines = [ln.rstrip() for ln in conversation.splitlines() if ln.strip()]
    return "\n".join(lines) + hint_block + ORACLE_TAIL


def append_chunk(conv: str, current: str | None, speaker: str, text: str, glue: bool = False) -> tuple[str, str | None]:
    """The server's add_to_conversation, as a pure function. glue=True joins a SentencePiece continuation piece
    (no leading '▁') to the previous piece instead of putting a space between them."""
    text = text.strip()
    if not text:
        return conv, current
    if speaker != current:
        if conv and not conv.endswith("\n"):
            conv += "\n"
        conv += f"{speaker}: "
        current = speaker
    elif glue and conv.endswith(" "):
        conv = conv[:-1]
    return conv + f"{text} ", current


# Phrases a passenger does not say but crew do. Tuned on the pilot: flags 34/118 predictions and 0 of 16 real
# passenger lines. "is there anything you can do" is a normal passenger line, "is there anything I can do" is not.
CREW_CUES = re.compile(
    r"\b(can i get you|would you like me to|is there anything (?:else )?(?:i|we) can|"
    r"i (?:can|could|will|'ll|\u2019ll) (?:also )?(?:offer|bring) you|i'?ll (?:also )?(?:get|bring) you|"
    r"i can (?:also )?(?:speak|talk) (?:with|to) (?:our|my)|our purser|my colleague|"
    r"are you okay|you seem (?:a bit|rather|quite)|"
    r"we can'?t (?:accommodate|move|upgrade|change)|we'?ve checked|"
    r"i understand (?:your|that|you'?re|it'?s)\b[^.?!]{0,40}\bbut\b|"
    r"let me see (?:if|what) (?:i|we) can|i'?ll (?:also )?get the captain)\b",
    re.I,
)


def looks_like_crew(text: str) -> bool:
    return bool(CREW_CUES.search(text))


_I_FORM = re.compile(r"^i(?=$|['\u2019])")
_ACRONYMS = {"OK", "KLM", "TV", "ID", "USB", "VIP", "AC", "UK", "USA", "US", "EU", "PM", "AM", "SOS", "WIFI", "PIN", "ETA"}


def _letters(word: str) -> str:
    return "".join(c for c in word if c.isalpha())


_LABEL = re.compile(r"^(?:(?:passenger|crew)[:\-]|[ab]:|speaker:?)$", re.I)


class OracleStreamCleaner:
    """Cleans back-end LLM output word by word, so streamed (live) and whole (training) text get identical
    treatment: drops a leading speaker label, quotes and stage directions ((..), [..], *..*), stops at the first
    newline after any word, and stops after MAX_WORDS words.

        feed(chunk) -> (clean_text, done).   When done is True stop reading the LLM stream."""

    def __init__(self, max_words: int = MAX_WORDS):
        self.max_words, self.n = max_words, 0
        self.depth, self.star, self.done, self.shout = 0, False, False, False

    def _fix_case(self, word: str, first: bool) -> str:
        """Live and training text both come out of this, so lowercase mirroring of the (lowercase) crew line and
        ALL-CAPS output are undone identically everywhere: once an all-caps word of 2+ letters shows the model is shouting
        (acronyms excepted), every all-caps word except 'I' is lowered; then the first word is capitalised and
        'i' / 'i'm' / 'i'll' become 'I' forms."""
        letters = _letters(word)
        if len(letters) >= 2 and letters.isupper() and letters not in _ACRONYMS:
            self.shout = True
        if self.shout and letters.isupper() and letters not in _ACRONYMS and letters != "I":
            word = word.lower()
        word = _I_FORM.sub("I", word)
        return word[:1].upper() + word[1:] if first and word[:1].islower() else word

    def _clean_word(self, w: str) -> str:
        out = []
        for ch in w:
            if ch in "([":
                self.depth += 1
            elif ch in ")]":
                self.depth = max(0, self.depth - 1)
            elif ch == "*":
                self.star = not self.star
            elif ch in '"\u201c\u201d':
                continue
            elif not self.depth and not self.star:
                out.append(ch)
        return "".join(out)

    def feed(self, text: str) -> tuple[str, bool]:
        if self.done:
            return "", True
        words: list[str] = []
        for m in re.finditer(r"\n|\S+", text):
            tok = m.group(0)
            if tok == "\n":
                if self.n or words:
                    self.done = True
                    break
                continue
            w = self._clean_word(tok)
            if not w:
                continue
            if self.n == 0 and not words and _LABEL.match(w):
                continue
            words.append(self._fix_case(w, first=self.n == 0 and not words))
            if self.n + len(words) >= self.max_words:
                self.done = True
                break
        self.n += len(words)
        return " ".join(words), self.done


def clean_full(text: str) -> str:
    """Non-streaming version for training oracles (same state machine as the live path)."""
    return OracleStreamCleaner().feed(text)[0]