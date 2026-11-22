from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.infer.profiler import LatencyProfiler
from src.infer.replay import LandmarkReplayer


class TestLatencyProfiler:
    def test_initial_state(self):
        p = LatencyProfiler()
        assert p.total_ms == 0.0
        assert p.fps == 0.0

    def test_begin_start_stop(self):
        p = LatencyProfiler()
        p.begin()
        p.start("capture")
        p.stop("capture")
        p.end_frame()
        assert p.total_ms > 0.0
        assert p.stage_ms("capture") > 0.0

    def test_multiple_stages(self):
        p = LatencyProfiler()
        p.begin()
        p.start("capture")
        p.stop("capture")
        p.start("landmarks")
        p.stop("landmarks")
        p.start("infer")
        p.stop("infer")
        p.end_frame()
        summary = p.summary()
        assert "capture_ms" in summary
        assert "landmarks_ms" in summary
        assert "infer_ms" in summary
        assert "total_ms" in summary
        assert "fps" in summary

    def test_reset(self):
        p = LatencyProfiler()
        p.begin()
        p.start("capture")
        p.stop("capture")
        p.end_frame()
        p.reset()
        assert p.total_ms == 0.0
        assert p.stage_ms("capture") == 0.0

    def test_stage_ms_unknown(self):
        p = LatencyProfiler()
        assert p.stage_ms("nonexistent") == 0.0

    def test_fps_calculation(self):
        p = LatencyProfiler()
        p.begin()
        p.start("a")
        p.stop("a")
        p.end_frame()
        assert p.fps > 0.0


class TestLandmarkReplayer:
    def test_empty_dir(self, tmp_path):
        replayer = LandmarkReplayer(tmp_path, fps=30)
        assert len(replayer) == 0

    def test_loads_npz_files(self, tmp_path):
        for i in range(3):
            features = np.random.rand(10, 168).astype(np.float32)
            np.savez(tmp_path / f"sample_{i}.npz", features=features, label=f"word_{i}")

        replayer = LandmarkReplayer(tmp_path, fps=30)
        assert len(replayer) == 3

    def test_get_all_samples(self, tmp_path):
        for i in range(2):
            features = np.random.rand(8, 168).astype(np.float32)
            np.savez(tmp_path / f"s{i}.npz", features=features, label=f"w{i}")

        replayer = LandmarkReplayer(tmp_path, fps=30)
        samples = replayer.get_all_samples()
        assert len(samples) == 2
        for feat, label in samples:
            assert isinstance(feat, np.ndarray)
            assert feat.dtype == np.float32
            assert feat.shape == (8, 168)

    def test_get_samples_by_words(self, tmp_path):
        for i, word in enumerate(["hello", "thank", "yes"]):
            features = np.random.rand(5, 168).astype(np.float32)
            np.savez(tmp_path / f"{word}.npz", features=features, label=word)

        replayer = LandmarkReplayer(tmp_path, fps=30)
        samples = replayer.get_samples_by_words(["hello", "yes"])
        assert len(samples) == 2
        labels = {s[1] for s in samples}
        assert labels == {"hello", "yes"}

    def test_get_samples_by_words_no_match(self, tmp_path):
        features = np.random.rand(5, 168).astype(np.float32)
        np.savez(tmp_path / "hello.npz", features=features, label="hello")

        replayer = LandmarkReplayer(tmp_path, fps=30)
        samples = replayer.get_samples_by_words(["nonexistent"])
        assert len(samples) == 0

    def test_yield_frames(self, tmp_path):
        features = np.random.rand(10, 168).astype(np.float32)
        np.savez(tmp_path / "test.npz", features=features, label="test")

        replayer = LandmarkReplayer(tmp_path, fps=30)
        samples = replayer.get_all_samples()
        frames = replayer.yield_frames(samples[0][0])
        assert len(frames) == 10
        for frame in frames:
            assert frame.shape == (168,)

    def test_get_sample_info(self, tmp_path):
        features = np.random.rand(15, 168).astype(np.float32)
        np.savez(tmp_path / "test.npz", features=features, label="test")

        replayer = LandmarkReplayer(tmp_path, fps=30)
        samples = replayer.get_all_samples()
        info = replayer.get_sample_info(samples[0][0], "test")
        assert info["label"] == "test"
        assert info["num_frames"] == 15
        assert info["feature_dim"] == 168
        assert info["duration_s"] == 0.5

    def test_len(self, tmp_path):
        replayer = LandmarkReplayer(tmp_path, fps=30)
        assert len(replayer) == 0
        assert replayer.__len__() == 0


class TestCategoryAnalysisHelpers:
    def test_parse_label_numbered(self):
        from scripts.category_analysis import _parse_label

        prefix, clean = _parse_label("48. Hello")
        assert prefix == 48
        assert clean == "hello"

    def test_parse_label_numbered_multiword(self):
        from scripts.category_analysis import _parse_label

        prefix, clean = _parse_label("55. Thank you")
        assert prefix == 55
        assert clean == "thank you"

    def test_parse_label_no_prefix(self):
        from scripts.category_analysis import _parse_label

        prefix, clean = _parse_label("hello")
        assert prefix is None
        assert clean == "hello"

    def test_build_category_mapping(self):
        from scripts.category_analysis import _build_category_mapping

        label_map = {
            "48. Hello": 0,
            "55. Thank you": 1,
            "1. Dog": 2,
            "4. Bird": 3,
            "47. Red": 4,
            "54. Black": 5,
            "55. White": 6,
            "61. Father": 7,
            "3. happy": 8,
            "77. Boy": 9,
            "94. good": 10,
        }
        mapping = _build_category_mapping(label_map)

        assert "Greetings" in mapping
        assert 0 in mapping["Greetings"]

        assert "Animals" in mapping
        assert 2 in mapping["Animals"]
        assert 3 in mapping["Animals"]

        assert "Colors" in mapping
        assert 4 in mapping["Colors"]
        assert 5 in mapping["Colors"]
        assert 6 in mapping["Colors"]

        assert "Family" in mapping
        assert 7 in mapping["Family"]

        assert "People" in mapping
        assert 9 in mapping["People"]

        all_categorized = set()
        for indices in mapping.values():
            all_categorized.update(indices)
        assert len(all_categorized) == len(label_map)

    def test_build_category_mapping_all_labels_categorized(self):
        from scripts.category_analysis import _build_category_mapping

        label_map = {f"{i}. Word{i}": i for i in range(50)}
        mapping = _build_category_mapping(label_map)
        all_categorized = set()
        for indices in mapping.values():
            all_categorized.update(indices)
        assert all_categorized == set(range(50))


class TestEvaluateImports:
    def test_evaluate_module_compiles(self):
        import importlib
        mod = importlib.import_module("scripts.evaluate")
        assert hasattr(mod, "main")

    def test_category_analysis_module_compiles(self):
        import importlib
        mod = importlib.import_module("scripts.category_analysis")
        assert hasattr(mod, "main")
        assert hasattr(mod, "_parse_label")
        assert hasattr(mod, "_build_category_mapping")
