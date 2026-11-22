from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class SentenceConfig:
    dedup: bool = True
    auto_speak_timeout: float = 3.0
    max_words: int = 20
    min_words_for_speak: int = 1


class SentenceAssembler:
    def __init__(self, config: SentenceConfig | None = None):
        if config is None:
            config = SentenceConfig()
        self.config = config
        self._words: list[str] = []
        self._last_word_time: float = 0.0
        self._auto_speak_enabled: bool = True
        self._on_ready_callback: Callable[[str], None] | None = None

    def add_word(self, word: str) -> bool:
        word = word.strip()
        if not word:
            return False
        if self.config.dedup and self._words and self._words[-1] == word:
            return False
        self._words.append(word)
        self._last_word_time = time.monotonic()
        if len(self._words) >= self.config.max_words:
            return True
        return False

    def update(self) -> str | None:
        if not self._auto_speak_enabled:
            return None
        if not self._words:
            return None
        if len(self._words) < self.config.min_words_for_speak:
            return None
        elapsed = time.monotonic() - self._last_word_time
        if elapsed < self.config.auto_speak_timeout:
            return None
        sentence = self.get_sentence()
        self._fire_ready(sentence)
        self._words.clear()
        self._last_word_time = time.monotonic()
        return sentence

    def set_ready_callback(self, callback: Callable[[str], None]) -> None:
        self._on_ready_callback = callback

    def set_auto_speak(self, enabled: bool) -> None:
        self._auto_speak_enabled = enabled

    @property
    def auto_speak_enabled(self) -> bool:
        return self._auto_speak_enabled

    def get_time_since_last_word(self) -> float:
        if self._last_word_time == 0.0:
            return 0.0
        return time.monotonic() - self._last_word_time

    def is_ready_to_speak(self) -> bool:
        if not self._words:
            return False
        if len(self._words) < self.config.min_words_for_speak:
            return False
        elapsed = time.monotonic() - self._last_word_time
        return elapsed >= self.config.auto_speak_timeout

    def _fire_ready(self, sentence: str) -> None:
        if self._on_ready_callback is not None:
            self._on_ready_callback(sentence)

    def clear(self) -> None:
        self._words.clear()
        self._last_word_time = 0.0

    def backspace(self) -> None:
        if self._words:
            self._words.pop()

    def get_sentence(self) -> str:
        return " ".join(self._words)

    @property
    def words(self) -> list[str]:
        return list(self._words)

    @property
    def word_count(self) -> int:
        return len(self._words)

    def __repr__(self) -> str:
        return f"SentenceAssembler({self.get_sentence()!r})"
