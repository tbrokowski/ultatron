"""Single source of truth for finetune benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MetricSpec = tuple[str, str, bool]

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results" / "finetune"
DASHBOARD_DIR = RESULTS_ROOT / "_dashboard"
LOG_ROOT = REPO_ROOT / "logs" / "finetune"
STUDENT_COLOR = "#2ecc71"
REFERENCE_COLOR = "#e67e22"


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    sweep: str
    log_prefix: str
    primary_metric: str
    default_head: str
    chart_metrics: tuple[MetricSpec, ...]


SWEEPS: dict[str, Path] = {
    "representative": RESULTS_ROOT / "representative",
    "openus_seg": RESULTS_ROOT / "openus_seg",
    "fetal_planes": RESULTS_ROOT / "fetal_planes",
}

EXPERIMENTS: dict[str, ExperimentSpec] = {
    "busi": ExperimentSpec("busi", "BUSI — breast tumour segmentation", "representative", "ultatron_ft_busi_", "dice_tumor_mean", "dpt", (("dice_tumor_mean", "Tumour Dice", False), ("dice_mean", "Dice (all)", False), ("iou_tumor_mean", "Tumour IoU", False), ("precision_mean", "Precision", False), ("recall_mean", "Recall", False))),
    "busi_multitask": ExperimentSpec("busi_multitask", "BUSI — tumour seg + cls", "representative", "ultatron_ft_busi_multitask_", "dice_tumor_mean", "dpt", (("dice_tumor_mean", "Tumour Dice", False), ("cls_accuracy", "Cls accuracy", False), ("iou_tumor_mean", "Tumour IoU", False), ("dice_mean", "Dice (all)", False))),
    "camus": ExperimentSpec("camus", "CAMUS — cardiac LV segmentation", "representative", "ultatron_ft_camus_", "dice_lv_epi_mean", "dpt", (("dice_lv_epi_mean", "LVEpi Dice", False), ("dice_lv_epi_ed", "LVEpi ED", False), ("dice_lv_epi_es", "LVEpi ES", False), ("hd95_mean", "HD95", True))),
    "camus_lv_endo": ExperimentSpec("camus_lv_endo", "CAMUS — LVEndo binary", "representative", "ultatron_ft_camus_", "dice_lv_endo_mean", "dpt", (("dice_lv_endo_mean", "LVEndo Dice", False), ("dice_lv_endo_ed", "LVEndo ED", False), ("dice_lv_endo_es", "LVEndo ES", False), ("hd95_mean", "HD95", True))),
    "camus_lv_epi": ExperimentSpec("camus_lv_epi", "CAMUS — LVEpi binary", "representative", "ultatron_ft_camus_", "dice_lv_epi_mean", "dpt", (("dice_lv_epi_mean", "LVEpi Dice", False), ("dice_lv_epi_ed", "LVEpi ED", False), ("dice_lv_epi_es", "LVEpi ES", False), ("hd95_mean", "HD95", True))),
    "camus_la": ExperimentSpec("camus_la", "CAMUS — LA binary", "representative", "ultatron_ft_camus_", "dice_la_mean", "dpt", (("dice_la_mean", "LA Dice", False), ("dice_la_ed", "LA ED", False), ("dice_la_es", "LA ES", False), ("hd95_mean", "HD95", True))),
    "camus_multiclass": ExperimentSpec("camus_multiclass", "CAMUS — 4-class multiclass", "representative", "ultatron_ft_camus_", "dice_macro_fg_mean", "dpt", (("dice_macro_fg_mean", "Macro FG Dice", False), ("dice_class_1_mean", "Cavity Dice", False), ("dice_class_2_mean", "Myo Dice", False), ("dice_class_3_mean", "LA Dice", False))),
    "echonet": ExperimentSpec("echonet", "EchoNet-Dynamic — EF regression", "representative", "ultatron_ft_echonet_", "mae", "mlp", (("mae", "MAE", True), ("rmse", "RMSE", True), ("r2", "R²", False), ("pearson_r", "Pearson r", False))),
    "lus": ExperimentSpec("lus", "LUS — patient-level MIL", "representative", "ultatron_ft_lus_", "val_auc_tb", "mil", (("val_auc_tb", "AUC TB", False), ("val_loss", "Val Loss", True))),
    "lus_video": ExperimentSpec("lus_video", "LUS — video multilabel", "representative", "ultatron_ft_lus_video_", "val_auc_macro", "mlp", (("val_auc_a_line", "AUC A-line", False), ("val_auc_b_line", "AUC B-line", False), ("val_auc_confluent_b_line", "AUC Confluent B-line", False), ("val_auc_pleural_effusion", "AUC Pleural effusion", False), ("val_auc_large_consolidation", "AUC Large consolidation", False), ("val_auc_small_consolidation", "AUC Small consolidation", False), ("val_auc_pneumothorax", "AUC Pneumothorax", False), ("val_auc_macro", "AUC Macro", False))),
    "busbra": ExperimentSpec("busbra", "BUS-BRA — breast segmentation", "openus_seg", "ultatron_ft_busbra_", "dice_mean", "dpt", (("dice_mean", "Dice", False), ("iou_mean", "IoU", False), ("s_measure_mean", "S-measure", False), ("precision_mean", "Precision", False), ("recall_mean", "Recall", False))),
    "tn3k": ExperimentSpec("tn3k", "TN3K — thyroid segmentation", "openus_seg", "ultatron_ft_tn3k_", "dice_mean", "dpt", (("dice_mean", "Dice", False), ("iou_mean", "IoU", False), ("s_measure_mean", "S-measure", False))),
    "fetal_planes_db": ExperimentSpec("fetal_planes_db", "FETAL-PLANES-DB — classification", "fetal_planes", "ultatron_ft_fetal_planes_db_", "val_acc", "linear", (("val_acc", "Val accuracy", False), ("cls_accuracy", "Test accuracy", False))),
}

_ALIASES = {
    "busi_breast_segmentation": "busi", "busi_seg": "busi",
    "tn3k_thyroid_segmentation": "tn3k", "tn3k_segmentation": "tn3k",
    "busbra_breast_segmentation": "busbra", "camus_lv_segmentation": "camus",
    "fetal_planes_db_classification": "fetal_planes_db",
}

REFERENCE_SCORES: dict[str, dict[str, dict[str, float]]] = {
    "busi": {"reference": {"dice_tumor_mean": 0.7709, "dice_mean": 0.7709, "iou_tumor_mean": 0.6546, "precision_mean": 0.8029, "recall_mean": 0.7438}},
}


def normalize_experiment(name: str) -> str | None:
    return name if name in EXPERIMENTS else _ALIASES.get(name)


def all_sweep_dirs() -> list[Path]:
    return [SWEEPS[k] for k in ("representative", "openus_seg", "fetal_planes")]


def chart_metrics_for(experiment: str) -> list[MetricSpec]:
    spec = EXPERIMENTS.get(experiment)
    return list(spec.chart_metrics) if spec else []


def title_for(experiment: str) -> str:
    return EXPERIMENTS[experiment].title if experiment in EXPERIMENTS else experiment


def default_head_for(experiment: str) -> str:
    return EXPERIMENTS[experiment].default_head if experiment in EXPERIMENTS else "dpt"


def primary_metric_for(experiment: str) -> str:
    return EXPERIMENTS[experiment].primary_metric if experiment in EXPERIMENTS else "dice_mean"


def is_student_backbone(name: str) -> bool:
    return name.startswith("student_")
