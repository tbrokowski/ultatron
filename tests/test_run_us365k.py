"""Checks for the US-365K run without model downloads or CSCS data."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import yaml

from models.student.hiera_backbone import HieraStudentBackbone
from tests.dataset_adapters import student_training_smoke as runner


def test_ema_only_initialization(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "configs/run_us365k/train.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["pretrain"].update(ckpt_dir=str(tmp_path), tensorboard=False)
    monkeypatch.setattr(runner, "build_student_encoder", lambda *a, **kw: torch.nn.Linear(4, 4))
    forbidden = Mock(side_effect=AssertionError("Frozen teacher or unused head constructed"))
    for name in ("FrozenDINOTeacher", "FrozenVJEPATeacher",
                 "build_fusion_target_builder", "build_hierarchical_seg_head"):
        monkeypatch.setattr(runner, name, forbidden)
    trainer = runner.StudentSmokeTrainer(cfg, device="cpu")
    try:
        assert trainer.model_cfg.pretrained is False
        assert trainer.dino is None and trainer.vjepa is None
        for step in range(cfg["training"]["total_steps"]):
            assert runner._stage_for_step(step, trainer.total_steps, trainer.stage_fracs) == 4
        trainer.sync_teachers_for_stage(4)
        forbidden.assert_not_called()
        for student, teacher in zip(trainer.student.parameters(), trainer.ema_student.parameters()):
            assert torch.equal(student, teacher)
            assert student.data_ptr() != teacher.data_ptr()
            assert not teacher.requires_grad
    finally:
        trainer.metrics.close()


def test_random_backbone_never_loads_weights(monkeypatch):
    import transformers
    from models import hf_loading

    backbone = torch.nn.Linear(4, 4)
    read_config = Mock(return_value=object())
    build_model = Mock(return_value=SimpleNamespace(
        vision_encoder=SimpleNamespace(backbone=backbone)))
    forbidden = Mock(side_effect=AssertionError("Pretrained weights requested"))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", read_config)
    monkeypatch.setattr(transformers.AutoModel, "from_config", build_model)
    monkeypatch.setattr(hf_loading, "load_pretrained", forbidden)
    result, _ = HieraStudentBackbone._load_sam2_hiera(
        SimpleNamespace(pretrained=False), "/cache")
    assert result is backbone
    assert read_config.call_args.kwargs["local_files_only"] is True
    build_model.assert_called_once()
    forbidden.assert_not_called()
