from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.htma import HTMAClassifier, HTMAConfig
from src.train.distiller import Distiller


def _make_tiny_config(**overrides) -> HTMAConfig:
    defaults = dict(
        input_dim=168,
        embedding_dim=64,
        num_blocks=1,
        num_heads=4,
        num_classes=10,
        max_seq_len=32,
    )
    defaults.update(overrides)
    return HTMAConfig(**defaults)


def _make_dataloader(n: int = 16, config: HTMAConfig | None = None) -> DataLoader:
    if config is None:
        config = _make_tiny_config()
    features = torch.randn(n, 32, config.input_dim)
    masks = torch.ones(n, 32, dtype=torch.bool)
    labels = torch.randint(0, config.num_classes, (n,))
    return DataLoader(TensorDataset(features, masks, labels), batch_size=4)


@pytest.fixture
def teacher():
    return HTMAClassifier(_make_tiny_config())


@pytest.fixture
def student():
    return HTMAClassifier(_make_tiny_config())


@pytest.fixture
def distiller(teacher, student):
    return Distiller(teacher, student, temperature=4.0, alpha=0.7)


class TestTeacherFrozen:
    def test_teacher_no_grad(self, distiller):
        for p in distiller.teacher.parameters():
            assert not p.requires_grad

    def test_teacher_eval_mode(self, distiller):
        assert not distiller.teacher.training

    def test_teacher_output_no_grad(self, distiller):
        x = torch.randn(2, 32, 168)
        out = distiller._teacher_forward(x)
        assert out.grad_fn is None


class TestStudentTrainable:
    def test_student_has_grad(self, distiller):
        has_grad = any(p.requires_grad for p in distiller.student.parameters())
        assert has_grad

    def test_student_train_mode(self, distiller):
        assert distiller.student.training


class TestKDLoss:
    def test_kd_loss_computes(self, distiller):
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        loss = distiller._compute_kd_loss(student_logits, teacher_logits)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_kd_loss_scales_with_temperature(self):
        s_logits = torch.randn(4, 10)
        t_logits = torch.randn(4, 10)
        d1 = Distiller(
            HTMAClassifier(_make_tiny_config()),
            HTMAClassifier(_make_tiny_config()),
            temperature=2.0,
        )
        d2 = Distiller(
            HTMAClassifier(_make_tiny_config()),
            HTMAClassifier(_make_tiny_config()),
            temperature=8.0,
        )
        l1 = d1._compute_kd_loss(s_logits, t_logits)
        l2 = d2._compute_kd_loss(s_logits, t_logits)
        assert l1.item() != l2.item()


class TestCombinedLoss:
    def test_combined_loss_sums(self, distiller):
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss, breakdown = distiller._compute_combined_loss(
            student_logits, teacher_logits, labels
        )
        assert loss.ndim == 0
        assert "kd_loss" in breakdown
        assert "ce_loss" in breakdown
        assert "total_loss" in breakdown

    def test_combined_loss_alpha_zero(self, teacher, student):
        d = Distiller(teacher, student, alpha=0.0)
        s = torch.randn(4, 10)
        t = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss, bd = d._compute_combined_loss(s, t, labels)
        assert pytest.approx(bd["total_loss"], abs=1e-5) == bd["ce_loss"]

    def test_combined_loss_alpha_one(self, teacher, student):
        d = Distiller(teacher, student, alpha=1.0)
        s = torch.randn(4, 10)
        t = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss, bd = d._compute_combined_loss(s, t, labels)
        assert pytest.approx(bd["total_loss"], abs=1e-5) == bd["kd_loss"]


class TestTrainEpoch:
    def test_train_epoch_runs(self, distiller):
        loader = _make_dataloader()
        device = torch.device("cpu")
        optimizer = torch.optim.AdamW(distiller.student.parameters(), lr=1e-3)
        metrics = distiller.train_epoch(loader, optimizer, device)
        assert "loss" in metrics
        assert "kd_loss" in metrics
        assert "ce_loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] >= 0

    def test_train_epoch_updates_weights(self, distiller):
        loader = _make_dataloader()
        device = torch.device("cpu")
        optimizer = torch.optim.AdamW(distiller.student.parameters(), lr=1e-3)
        before = {n: p.clone() for n, p in distiller.student.named_parameters()}
        distiller.train_epoch(loader, optimizer, device)
        changed = False
        for n, p in distiller.student.named_parameters():
            if not torch.equal(before[n], p):
                changed = True
                break
        assert changed


class TestValidate:
    def test_validate_runs(self, distiller):
        loader = _make_dataloader()
        device = torch.device("cpu")
        metrics = distiller.validate(loader, device)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] >= 0
        assert 0 <= metrics["accuracy"] <= 100


class TestExportStudent:
    def test_export_creates_files(self, distiller):
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller.export_student(tmpdir)
            assert (Path(tmpdir) / "student_model.pt").exists()
            assert (Path(tmpdir) / "model_config.json").exists()

    def test_export_config_content(self, distiller):
        with tempfile.TemporaryDirectory() as tmpdir:
            distiller.export_student(tmpdir)
            with open(Path(tmpdir) / "model_config.json") as f:
                cfg = json.load(f)
            assert cfg["input_dim"] == distiller.student.config.input_dim
            assert cfg["num_classes"] == distiller.student.config.num_classes
