import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.text import detect_flags, gloss_to_sentence


class TestGlossToSentence:
    def test_empty(self):
        assert gloss_to_sentence([]) == ""

    def test_single_word(self):
        assert gloss_to_sentence(["hello"]) == "Hello."
        assert gloss_to_sentence(["FS(R-A-M)"]) == "R-a-m."

    def test_sov_to_svo(self):
        assert gloss_to_sentence(["i", "market", "go"]) == "I go market."

    def test_time_fronting(self):
        assert gloss_to_sentence(["i", "market", "go", "today"]) == "Today i go market."

    def test_wh_final(self):
        assert gloss_to_sentence(["you", "go", "where"]) == "Where you go?"

    def test_neg_final(self):
        assert gloss_to_sentence(["i", "go", "no"]) == "I do not go."

    def test_copula_insert(self):
        assert gloss_to_sentence(["you", "happy"]) == "You are happy."

    def test_fs_passthrough(self):
        assert gloss_to_sentence(["my", "name", "FS(RAM)"]) == "My name Ram."

    def test_detect_flags(self):
        f = detect_flags(["you", "go", "where"])
        assert f["Q_WH"] is True
        f = detect_flags(["i", "go", "no"])
        assert f["NEG"] is True
        f = detect_flags(["hello"])
        assert f == {"Q_WH": False, "NEG": False, "Q_POLAR": False}
