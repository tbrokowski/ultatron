"""
ontology.py — Canonical anatomy/pathology taxonomy.
Maps dataset-specific label strings -> (anatomy_family, anatomy_fine).
"""
from __future__ import annotations
from typing import Dict, Tuple, List

ANATOMY_FAMILIES = [
    "cardiac","breast","thyroid","fetal","lung","liver","kidney","prostate",
    "ovarian","gallbladder","musculoskeletal","vascular","nerve","brain",
    "skin","ocular","abdomen","phantom","unknown",
]
ANATOMY_FINE_BY_FAMILY: Dict[str,list] = {
    "cardiac": ["lv_endo","lv_epi","rv_endo","rv_epi","la","ra","myocardium",
                "whole_heart","aortic_valve","mitral_valve"],
    "breast":  ["lesion_benign","lesion_malignant","lesion_unknown","mass","cyst","fibroadenoma"],
    "thyroid": ["nodule_benign","nodule_malignant","nodule_unknown","whole_thyroid"],
    "fetal":   ["fetal_head","fetal_brain","head_circumference","fetal_abdomen",
                "fetal_femur","fetal_heart","maternal_cervix","pubic_symphysis"],
    "lung":    [
        "b_line",
        "a_line",
        "consolidation",
        "pleural_effusion",
        "pleural_line",
        "whole_lung",
        "confluent_b_line",
        "large_consolidation",
        "small_consolidation",
        "pneumothorax",
        "not_measured",
    ],
    "liver":   ["whole_liver","liver_lesion","fatty_liver"],
    "kidney":  ["whole_kidney"],
    "prostate":["whole_prostate","prostate_cancer","peripheral_zone"],
    "ovarian": ["whole_ovary","polycystic_ovary","ovarian_cyst"],
    "gallbladder":["whole_gallbladder","gallbladder_cancer","appendix_inflamed"],
    "musculoskeletal":["muscle_fascicle","tendon","muscle_tendon_junction","lumbar_multifidus"],
    "vascular":["carotid_intima_media","carotid_plaque","vessel_lumen"],
    "nerve":   ["brachial_plexus","median_nerve"],
    "brain":   ["tumor_boundary","resection_cavity","white_matter"],
    "skin":    ["skin_lesion_benign","skin_lesion_malignant"],
    "ocular":  ["retinal_detachment","retina"],
    "abdomen": ["spleen","pancreas","colon_wall","lymph_node"],
    "phantom": ["phantom_structure"],
    "unknown": ["unknown"],
}

_RAW: Dict[str,Tuple[str,str]] = {}
def _add(labels,f,n):
    for r in ([labels] if isinstance(labels,str) else labels):
        _RAW[r.lower().strip()] = (f,n)

_add(["lv","left ventricle","endocardium","lv_endo"],"cardiac","lv_endo")
_add(["myocardium"],"cardiac","myocardium")
_add(["whole heart","heart","cardiac"],"cardiac","whole_heart")
_add(["benign","benign_lesion"],"breast","lesion_benign")
_add(["malignant","carcinoma","cancer"],"breast","lesion_malignant")
_add(["lesion","mass","tumor","breast_lesion"],"breast","lesion_unknown")
_add(["nodule","thyroid_nodule","tn"],"thyroid","nodule_unknown")
_add(["thyroid","whole_thyroid"],"thyroid","whole_thyroid")
_add(["fetal head","fetal_head"],"fetal","fetal_head")
_add(["head circumference","hc","head_circumference"],"fetal","head_circumference")
_add(["fetal brain"],"fetal","fetal_brain")
_add(["fetal abdomen","fetal_abdomen"],"fetal","fetal_abdomen")
_add(["maternal cervix","cervix"],"fetal","maternal_cervix")
_add(["b-line","b_line","bline","vertical_artifact"],"lung","b_line")
_add(["consolidation","covid","pneumonia"],"lung","consolidation")
_add(["confluent b-lines","confluent_b_lines","confluent_b_line"],"lung","confluent_b_line")
_add(["large consolidations","large_consolidation","large consolidation"],"lung","large_consolidation")
_add(["small consolidations or nodules","small_consolidation","small consolidation"],"lung","small_consolidation")
_add(["pattern a' (pneumothorax)","pneumothorax"],"lung","pneumothorax")
_add(["lung","whole_lung"],"lung","whole_lung")
_add(["liver","whole_liver"],"liver","whole_liver")
_add(["fatty_liver","nafld"],"liver","fatty_liver")
_add(["kidney","whole_kidney"],"kidney","whole_kidney")
_add(["prostate","whole_prostate"],"prostate","whole_prostate")
_add(["ovary","whole_ovary"],"ovarian","whole_ovary")
_add(["pcos","polycystic_ovary"],"ovarian","polycystic_ovary")
_add(["gallbladder"],"gallbladder","whole_gallbladder")
_add(["appendix","appendicitis"],"gallbladder","appendix_inflamed")
_add(["muscle","muscle_fascicle","fascicle"],"musculoskeletal","muscle_fascicle")
_add(["carotid","carotid_artery","cca"],"vascular","carotid_intima_media")
_add(["brachial_plexus","plexus"],"nerve","brachial_plexus")
_add(["brain_tumor","tumor"],"brain","tumor_boundary")
_add(["phantom","phantom_structure"],"phantom","phantom_structure")
_add(["not measured","not_measured"],"lung","not_measured")

def normalize_label(raw:str)->Tuple[str,str]:
    key=raw.lower().strip().replace("-","_").replace(" ","_")
    if key in _RAW: return _RAW[key]
    for k,v in _RAW.items():
        if k in key or key in k: return v
    return ("unknown","unknown")

def register_label_mapping(raw,family,fine): _add(raw,family,fine)
def get_family_index(f): return ANATOMY_FAMILIES.index(f) if f in ANATOMY_FAMILIES else -1
def get_fine_labels(f): return ANATOMY_FINE_BY_FAMILY.get(f,["unknown"])
