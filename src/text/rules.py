from __future__ import annotations

VERBS = {
    "go", "come", "eat", "drink", "sit", "stand", "read", "write", "help",
    "thank", "play", "run", "walk", "sleep", "wake", "open", "close", "give",
    "take", "make", "meet", "call", "stop", "start", "learn", "teach",
}
WH_WORDS = {"what", "when", "where", "who", "which", "how", "why"}
TIME_WORDS = {"today", "yesterday", "tomorrow", "morning", "evening", "night",
              "week", "month", "year", "now", "later", "daily"}
PRONOUNS = {"i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them"}
COPULA_NEED = {"happy", "sad", "good", "bad", "hot", "cold", "big", "small",
               "tall", "short", "new", "old", "sick", "tired", "ready", "sorry"}


def gloss_to_sentence(glosses: list[str], flags: dict | None = None) -> str:
    flags = flags or {}
    words = [g.strip().lower() for g in glosses if g and g.strip()]
    if not words:
        return ""
    if len(words) == 1:
        w = words[0]
        if w.startswith("fs(") and w.endswith(")"):
            return w[3:-1].capitalize() + "."
        return w.capitalize() + "."
    if words[-1].startswith("fs(") and words[-1].endswith(")"):
        words[-1] = words[-1][3:-1].capitalize()
    q_wh = bool(flags.get("Q_WH")) or words[-1] in WH_WORDS
    time_front = [w for w in words if w in TIME_WORDS]
    rest = [w for w in words if w not in TIME_WORDS]
    wh_word = ""
    if q_wh and rest and rest[-1] in WH_WORDS:
        wh_word = rest.pop()
    if rest and rest[-1] in VERBS and len(rest) >= 3:
        verb = rest.pop()
        rest.insert(1, verb)
    neg = bool(flags.get("NEG")) or (rest and rest[-1] in {"no", "not", "never"})
    if rest and rest[-1] in {"no", "not", "never"}:
        rest.pop()
    if neg and rest:
        rest.insert(1, "not" if rest[0].lower() not in PRONOUNS else "do not")
        if rest[0].lower() in PRONOUNS and len(rest) > 2 and rest[2] in VERBS:
            pass
    if len(rest) == 2 and rest[1] in COPULA_NEED:
        rest.insert(1, "are" if rest[0].lower() in {"you", "we", "they"} else "is")
    out = ([wh_word] if wh_word else []) + time_front + rest
    text = " ".join(out)
    text = text[:1].upper() + text[1:] if text else text
    if flags.get("Q_POLAR") or q_wh:
        return text + "?"
    return text + "."


def detect_flags(words: list[str]) -> dict:
    low = [w.lower() for w in words]
    return {
        "Q_WH": any(w in WH_WORDS for w in low),
        "NEG": any(w in {"no", "not", "never", "nothing", "none"} for w in low),
        "Q_POLAR": False,
    }
