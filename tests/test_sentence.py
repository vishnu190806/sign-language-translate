from __future__ import annotations

import time

import pytest

from src.infer.sentence import SentenceAssembler, SentenceConfig


class TestSentenceAssembler:
    def test_add_words(self):
        asm = SentenceAssembler()
        asm.add_word("hello")
        asm.add_word("world")
        assert asm.get_sentence() == "hello world"

    def test_dedup(self):
        asm = SentenceAssembler(SentenceConfig(dedup=True))
        asm.add_word("hello")
        asm.add_word("hello")
        assert asm.get_sentence() == "hello"

    def test_no_dedup(self):
        asm = SentenceAssembler(SentenceConfig(dedup=False))
        asm.add_word("hello")
        asm.add_word("hello")
        assert asm.get_sentence() == "hello hello"

    def test_clear(self):
        asm = SentenceAssembler()
        asm.add_word("hello")
        asm.clear()
        assert asm.get_sentence() == ""

    def test_backspace(self):
        asm = SentenceAssembler()
        asm.add_word("hello")
        asm.add_word("world")
        asm.backspace()
        assert asm.get_sentence() == "hello"

    def test_backspace_empty(self):
        asm = SentenceAssembler()
        asm.backspace()
        assert asm.get_sentence() == ""

    def test_empty_word_ignored(self):
        asm = SentenceAssembler()
        asm.add_word("")
        asm.add_word("  ")
        assert asm.get_sentence() == ""

    def test_word_count(self):
        asm = SentenceAssembler()
        assert asm.word_count == 0
        asm.add_word("hello")
        assert asm.word_count == 1

    def test_auto_speak_timeout(self):
        config = SentenceConfig(auto_speak_timeout=0.1, min_words_for_speak=1)
        asm = SentenceAssembler(config=config)
        triggered = []
        asm.set_ready_callback(lambda s: triggered.append(s))

        asm.add_word("hello")
        result = asm.update()
        assert result is None
        assert len(triggered) == 0

        time.sleep(0.15)
        result = asm.update()
        assert result == "hello"
        assert len(triggered) == 1
        assert triggered[0] == "hello"

    def test_max_words_forced_speak(self):
        config = SentenceConfig(max_words=3, auto_speak_timeout=10.0, min_words_for_speak=1)
        asm = SentenceAssembler(config=config)
        triggered = []
        asm.set_ready_callback(lambda s: triggered.append(s))

        asm.add_word("one")
        asm.add_word("two")
        result = asm.add_word("three")
        assert result is True
        assert len(triggered) == 0

    def test_auto_speak_disabled(self):
        config = SentenceConfig(auto_speak_timeout=0.05, min_words_for_speak=1)
        asm = SentenceAssembler(config=config)
        triggered = []
        asm.set_ready_callback(lambda s: triggered.append(s))

        asm.set_auto_speak(False)
        asm.add_word("hello")
        time.sleep(0.1)
        result = asm.update()
        assert result is None
        assert len(triggered) == 0

    def test_is_ready_to_speak(self):
        config = SentenceConfig(auto_speak_timeout=0.1, min_words_for_speak=1)
        asm = SentenceAssembler(config=config)

        assert asm.is_ready_to_speak() is False
        asm.add_word("hello")
        assert asm.is_ready_to_speak() is False

        time.sleep(0.15)
        assert asm.is_ready_to_speak() is True

    def test_get_time_since_last_word(self):
        config = SentenceConfig(auto_speak_timeout=1.0, min_words_for_speak=1)
        asm = SentenceAssembler(config=config)

        assert asm.get_time_since_last_word() == 0.0
        asm.add_word("hello")
        time.sleep(0.05)
        elapsed = asm.get_time_since_last_word()
        assert elapsed >= 0.04

    def test_clear_resets_timer(self):
        config = SentenceConfig(auto_speak_timeout=0.1, min_words_for_speak=1)
        asm = SentenceAssembler(config=config)

        asm.add_word("hello")
        time.sleep(0.05)
        asm.clear()
        assert asm.get_time_since_last_word() == 0.0
        assert asm.word_count == 0

    def test_add_word_returns_false_for_dedup(self):
        config = SentenceConfig(dedup=True)
        asm = SentenceAssembler(config=config)

        result1 = asm.add_word("hello")
        assert result1 is False
        result2 = asm.add_word("hello")
        assert result2 is False
        assert asm.word_count == 1

    def test_add_word_returns_false_for_empty(self):
        asm = SentenceAssembler()
        result = asm.add_word("")
        assert result is False
        result = asm.add_word("  ")
        assert result is False
