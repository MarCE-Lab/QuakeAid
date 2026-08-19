"""
main_pipeline/pipeline_config.py
================================
Single source of truth for the Tier-I END-TO-END pipeline
(image -> CV/EDL -> VLM -> KG -> GNN -> A/B/C + civilian report).

Mirrors the spirit of CV_modules/cv_config.py and validator/vlm/vlm_config.py:
every path, model-variant registry, default and business rule the main pipeline
needs lives here, so run_pipeline.py / train_gnn_models.py / report_builder.py
never drift from each other. Nothing here imports torch / cv2 / vllm, so it is
cheap to import anywhere.

Terminology used everywhere in this package
-------------------------------------------
gnn_input  : WHAT the GNN sees from the CV branch (the user-facing flag).
               "full"     -> the whole 21-dim CV block incl. the EDL grade
               "cv_noedl" -> CV observations only, EDL columns zeroed
               "nocv"     -> no CV block at all (all-zero, cv_present=0)
             This simultaneously selects (a) how the KG instance graph is built
             (cv_record / drop_cv_edl) and (b) WHICH trained GNN checkpoint is
             loaded — the two must always match, which is why one flag drives
             both.
gnn_arch   : "rgcn" (relation-typed, default) or "gin" (untyped topology).
variant    : the checkpoint family name, e.g. "fewshot_full". All production
             variants are few-shot (the VLM always runs --few-shot in this
             pipeline, per project decision).
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------- #
# Paths (resolved relative to this file: <repo>/main_pipeline/pipeline_config.py)
# --------------------------------------------------------------------------- #
PIPELINE_DIR = Path(__file__).resolve().parent            # <repo>/main_pipeline
REPO_ROOT = PIPELINE_DIR.parent                            # <repo>  (== "main/")

VALIDATOR_DIR = REPO_ROOT / "validator"
BUILT_KG_DIR = VALIDATOR_DIR / "built_kg"
GNN_DIR = VALIDATOR_DIR / "gnn"
CV_DIR = REPO_ROOT / "CV_modules"

# Deployable GNN checkpoints trained by main_pipeline/train_gnn_models.py.
GNN_MODELS_DIR = PIPELINE_DIR / "models" / "gnn"

# Per-run outputs of the end-to-end pipeline.
RUNS_DIR = PIPELINE_DIR / "runs"

# The static KG ontology (built once from validator/built_kg/csv_files/*.csv).
ONTOLOGY_PKL = BUILT_KG_DIR / "out" / "ontology.pkl"

# Existing training assets (produced during the formal eval; used to (re)build
# the three GNN training datasets when they are missing).
FEWSHOT_HANDOFF = BUILT_KG_DIR / "handoff" / "eval3_fewshot.jsonl"
CV_CACHE_EVAL = REPO_ROOT / "results" / "cv_cache_eval.jsonl"

# --------------------------------------------------------------------------- #
# Elements & grades (shared vocabulary with the whole repo)
# --------------------------------------------------------------------------- #
ELEMENTS = ("wall", "column", "beam")
ABC = ("A", "B", "C")
SEVERITY_RANK = {"A": 3, "B": 2, "C": 1}          # A = worst (project mandate)
IV_GRADES = ("I", "II", "III", "IV", "V")
IV_RANK = {g: i + 1 for i, g in enumerate(IV_GRADES)}


def worst_abc(grades) -> Optional[str]:
    """Most severe of a list of A/B/C values (None-tolerant)."""
    best = None
    for g in grades:
        if g in SEVERITY_RANK and (best is None or SEVERITY_RANK[g] > SEVERITY_RANK[best]):
            best = g
    return best


def iv_to_abc(g: Optional[str]) -> Optional[str]:
    """Member I~V -> building A/B/C (same mapping as ablation_core.iv_to_abc):
    IV/V (rebar buckled/ruptured, core crushed, capacity lost) -> A;
    III (cover spalling, bars intact) -> B; I/II -> C."""
    o = IV_RANK.get(g or "")
    if not o:
        return None
    return "A" if o >= 4 else ("B" if o == 3 else "C")


def placard_to_abc(p: Optional[str]) -> Optional[str]:
    """KG placard_suggestion -> A/B/C severity equivalent, used ONLY for the
    internal conservative cross-check (never shown as a placard to the user):
    紅=A, 黃=B, 綠=C, 不足=B (insufficient info can never clear as safe)."""
    if not p:
        return None
    head = p.split("_")[0]
    if "紅" in head:
        return "A"
    if "黃" in head:
        return "B"
    if "綠" in head:
        return "C"
    if "不足" in head:
        return "B"
    return None


# --------------------------------------------------------------------------- #
# GNN variant registry — ONE flag ("gnn_input") drives BOTH the KG graph build
# and the checkpoint choice, so they can never disagree.
#
#   kg_out_dir : where the matching TRAINING dataset lives / is rebuilt to
#                (validator/built_kg/<kg_out_dir>/{dataset_tensors.jsonl,
#                 feature_spec.json}).
#   cv_cache   : does the KG training build fuse the CV cache?
#   cv_mode    : build_kg.py --cv-mode ("full" | "no_edl"); None when no cache.
#   at inference: attach_cv + drop_cv_edl replicate the same condition for the
#                 single image's live CV record.
# --------------------------------------------------------------------------- #
GNN_VARIANTS: Dict[str, Dict[str, Any]] = {
    "fewshot_full": {
        "gnn_input": "full",
        "kg_out_dir": "out_fewshot",
        "cv_cache": True, "cv_mode": "full",
        "attach_cv": True, "drop_cv_edl": False,
        "desc": "GNN sees VLM graph + full CV block (YOLO/seg observations + EDL grade).",
    },
    "fewshot_cvnoedl_full": {
        "gnn_input": "cv_noedl",
        "kg_out_dir": "out_fewshot_cvnoedl",
        "cv_cache": True, "cv_mode": "no_edl",
        "attach_cv": True, "drop_cv_edl": True,
        "desc": "GNN sees VLM graph + CV observations; EDL grade/conf/uncertainty zeroed.",
    },
    "fewshot_nocv_full": {
        "gnn_input": "nocv",
        "kg_out_dir": "out_fewshot_nocv",
        "cv_cache": False, "cv_mode": None,
        "attach_cv": False, "drop_cv_edl": False,
        "desc": "GNN sees the VLM graph only (CV block all-zero, cv_present=0).",
    },
}
GNN_INPUT_TO_VARIANT = {v["gnn_input"]: k for k, v in GNN_VARIANTS.items()}
DEFAULT_GNN_INPUT = "full"
DEFAULT_GNN_ARCH = "rgcn"          # relation-typed model; "gin" also trained/available
GNN_ARCHS = ("rgcn", "gin")


def gnn_checkpoint_path(gnn_input: str, arch: str = DEFAULT_GNN_ARCH) -> Path:
    variant = GNN_INPUT_TO_VARIANT[gnn_input]
    return GNN_MODELS_DIR / variant / f"{arch}_final.pt"


def gnn_feature_spec_path(gnn_input: str) -> Path:
    variant = GNN_INPUT_TO_VARIANT[gnn_input]
    return GNN_MODELS_DIR / variant / "feature_spec.json"


# --------------------------------------------------------------------------- #
# VLM defaults for the production pipeline (per project decision: few-shot ON,
# element hint ON, single GPU, gemma4_12b_fp8).
# --------------------------------------------------------------------------- #
VLM_PROFILE = "gemma4_12b_fp8"
VLM_FEW_SHOT = True
VLM_ELEMENT_HINT = True
DEFAULT_GPU_ID = 0
# vLLM must leave VRAM headroom for the CV models on the SAME single GPU.
# 0.70 on a 24 GB card ≈ 16.8 GB for vLLM (12B-fp8 ≈ 13 GB + KV) and ~7 GB free
# for YOLO + 2 ViT + 2 U-Nets (~3-4 GB peak). serve_vlm.sh applies this.
VLM_GPU_MEMORY_UTILIZATION = 0.70

# run_meta stamped into the single-image tensor row must mirror the TRAINING
# rows of the fewshot segments so the GNN's input distribution is unchanged.
def inference_run_meta(profile: str = VLM_PROFILE) -> Dict[str, Any]:
    # v2 fix: schema_version was hardcoded "1.1" and silently drifted from the
    # canonical schema; read it from the single source of truth instead.
    from validator.vlm import vlm_config as vcfg   # import-cheap by design
    return {
        "profile": profile,
        "few_shot": True,                     # all deployed variants are few-shot
        "schema_version": vcfg.SCHEMA_VERSION,
        "prompt_versions": dict(vcfg.PROMPT_VERSIONS),
        "pipeline": "main_pipeline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Pipeline run naming & output layout
# --------------------------------------------------------------------------- #
_SLUG_RE = re.compile(r"[^0-9A-Za-z]+")


def make_run_id(image_path: str | Path) -> str:
    stem = _SLUG_RE.sub("_", Path(image_path).stem).strip("_")[:48]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1(str(Path(image_path).resolve()).encode()).hexdigest()[:6]
    return f"{ts}_{stem}__{h}"


# File names inside one run directory (fixed contract, referenced by guide.md).
OUT_CV = "cv_output.json"                      # CV modules structured output (+ derived)
OUT_VLM_EXTRACTION = "vlm_extraction.json"     # VLM pass-1
OUT_VLM_REASONING = "vlm_reasoning.json"       # VLM pass-2 (the KG input)
OUT_VALIDATION = "validation_report.json"      # Track-2 verdict for this image
OUT_KG_SUMMARY = "kg_reasoning.json"           # KG reasoning summary (placard, members…)
OUT_KG_GRAPH = "kg_graph.json"                 # reasoned subgraph (jsonable)
OUT_GNN = "gnn_result.json"                    # GNN grade + probabilities + provenance
OUT_ASSESSMENT = "assessment.json"             # machine-readable combined final result
OUT_REPORT_MD = "report.md"                    # zh-TW civilian summary report
OUT_LOG = "pipeline.log"

# --------------------------------------------------------------------------- #
# Assessment-basis labels used in assessment.json / the report
# --------------------------------------------------------------------------- #
BASIS_FULL = "gnn"                    # normal path: GNN over the fused KG graph
BASIS_CV_ONLY = "cv_only_fallback"    # VLM/KG/GNN unavailable -> CV/EDL fallback

PIPELINE_VERSION = "main_pipeline_1.1"


if __name__ == "__main__":
    print("REPO_ROOT          :", REPO_ROOT)
    print("GNN_MODELS_DIR     :", GNN_MODELS_DIR)
    print("ontology exists    :", ONTOLOGY_PKL.exists())
    print("variants           :", list(GNN_VARIANTS))
    print("ckpt (full,rgcn)   :", gnn_checkpoint_path("full", "rgcn"))
    print("run_id sample      :", make_run_id("/tmp/IMG_0007.jpg"))
