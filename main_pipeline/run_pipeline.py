#!/usr/bin/env python3
"""
main_pipeline/run_pipeline.py
=============================
The Tier-I END-TO-END orchestrator: ONE image in -> A/B/C + civilian report out.

    image ──▶ [1] CV branch (YOLO → crack-ViT → U-Net segs → cost, + ViT-EDL A/B/C)
          ──▶ [2] VLM branch (Gemma-4, two-pass, few-shot, element hint; ONE GPU)
          ──▶ [3] Validator verdict + KG grounding/reasoning (CPU)
          ──▶ [4] GNN over the fused graph (CPU, batch_size=1)
          ──▶ [5] assessment.json + TWO rule-based zh-TW reports:
                    report.md          — civilian view (3 sections, no model talk)
                    report_engineer.md — engineer view (all module detail, no advice)

The CV branch ALWAYS runs (its JSON is a pipeline output and feeds the report);
the --gnn-input flag only controls what the GNN sees from it — and thereby
which trained checkpoint is loaded:

    --gnn-input full      -> KG attaches the whole CV block  -> models/gnn/fewshot_full
    --gnn-input cv_noedl  -> CV attached, EDL columns zeroed -> models/gnn/fewshot_cvnoedl_full
    --gnn-input nocv      -> no CV in the graph              -> models/gnn/fewshot_nocv_full

Single-GPU deployment: start the vLLM server FIRST (it holds a reduced VRAM
reservation so the CV models fit beside it):

    bash main_pipeline/serve_vlm.sh --gpu 0          # once, stays up
    PYTHONPATH=. python3 -m main_pipeline.run_pipeline photo.jpg --element column

Degradation policy (project mandate: never silently clear a building):
  * CV fails            -> continue; GNN sees cv_present=0 (a training-time
                           join-miss condition), report says so.
  * VLM/KG/GNN fails    -> CV-only fallback: grade = worst(EDL, CV-rule table),
                           assessment_basis = "cv_only_fallback", loud banner.
  * validator "rejected"-> proceed on the repaired graph, reliability flagged.

Outputs land in main_pipeline/runs/<run_id>/ (see pipeline_config for names).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from main_pipeline import pipeline_config as pc
from main_pipeline import cv_infer, vlm_infer, kg_infer, gnn_infer, report_builder

log = logging.getLogger("main_pipeline")

# The civilian report keeps pc.OUT_REPORT_MD (report.md) so existing callers and
# paths do not move; the engineer view is written beside it. Define
# OUT_REPORT_ENGINEER_MD in pipeline_config to override the default name.
OUT_REPORT_ENGINEER_MD = getattr(pc, "OUT_REPORT_ENGINEER_MD", "report_engineer.md")


def _setup_logging(run_dir: Path, verbose: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(run_dir / pc.OUT_LOG, encoding="utf-8")]
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _save(run_dir: Path, name: str, obj: Any) -> None:
    (run_dir / name).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_pipeline(image: str | Path,
                 element: str,
                 gnn_input: str = pc.DEFAULT_GNN_INPUT,
                 gnn_arch: str = pc.DEFAULT_GNN_ARCH,
                 gpu: int = pc.DEFAULT_GPU_ID,
                 vlm_profile: str = pc.VLM_PROFILE,
                 vlm_backend: str = "vllm",
                 cv_device: str = "auto",
                 cv_ratio: Optional[float] = None,
                 out_dir: Optional[str | Path] = None,
                 debug_crops: bool = False,
                 reports: str = "both",
                 verbose: bool = False) -> Dict[str, Any]:
    """Programmatic entry point (the CLI is a thin wrapper). Returns the
    assessment dict; all artifacts are written into the run directory."""
    image = Path(image)
    if not image.exists():
        raise FileNotFoundError(f"image not found: {image}")
    if element not in pc.ELEMENTS:
        raise ValueError(f"--element must be one of {pc.ELEMENTS}")

    run_id = pc.make_run_id(image)
    run_dir = Path(out_dir) if out_dir else (pc.RUNS_DIR / run_id)
    _setup_logging(run_dir, verbose)
    t0 = time.time()
    log.info("run_id=%s  image=%s  element=%s  gnn_input=%s  arch=%s  gpu=%s",
             run_id, image, element, gnn_input, gnn_arch, gpu)

    bundle: Dict[str, Any] = {"run_id": run_id, "image_path": str(image),
                              "element": element, "gnn_input": gnn_input,
                              "gnn_arch": gnn_arch}

    # ---------------- [1] CV branch (always runs) ------------------------- #
    log.info("── stage 1/5: CV modules (YOLO / crack-ViT / segs / cost / EDL) ──")
    cv = cv_infer.run_cv_stage(
        image, element, device=cv_device, ratio=cv_ratio,
        debug_dir=str(run_dir / "cv_debug") if debug_crops else None,
        release_gpu=True)                       # free VRAM beside the vLLM server
    bundle["cv"] = cv
    _save(run_dir, pc.OUT_CV, cv)
    log.info("CV %s in %.1fs", "OK" if cv.get("ok") else f"FAILED ({cv.get('error')})",
             cv.get("seconds", 0))

    # ---------------- [2] VLM branch (single GPU) ------------------------- #
    log.info("── stage 2/5: VLM two-pass (few-shot=%s, hint=%s, gpu=%d) ──",
             pc.VLM_FEW_SHOT, pc.VLM_ELEMENT_HINT, gpu)
    vlm = vlm_infer.run_vlm_stage(image, element, gpu_id=gpu,
                                  profile=vlm_profile, backend_kind=vlm_backend)
    bundle["vlm"] = vlm
    if vlm.get("extraction"):
        _save(run_dir, pc.OUT_VLM_EXTRACTION, vlm["extraction"])
    if vlm.get("reasoning"):
        _save(run_dir, pc.OUT_VLM_REASONING, vlm["reasoning"])
    log.info("VLM %s in %.1fs", "OK" if vlm.get("ok") else f"FAILED ({vlm.get('error')})",
             vlm.get("seconds", 0))

    # ---------------- [3] Validator + KG reasoning ------------------------ #
    kg: Dict[str, Any] = {"ok": False, "error": "skipped: no VLM reasoning output"}
    if vlm.get("ok"):
        log.info("── stage 3/5: Validator + KG reasoning ──")
        kg = kg_infer.run_kg_stage(
            reasoning_json=vlm["reasoning"], image_id=vlm["image_id"],
            cv_record=cv.get("cv_record") if cv.get("ok") else None,
            gnn_input=gnn_input, run_meta=pc.inference_run_meta(vlm_profile))
        if kg.get("validation"):
            _save(run_dir, pc.OUT_VALIDATION, kg["validation"])
        if kg.get("kg_summary"):
            _save(run_dir, pc.OUT_KG_SUMMARY,
                  {"image_id": kg.get("image_id"), "verdict": kg.get("verdict"),
                   "conversion_flags": kg.get("conversion_flags"),
                   "kg_summary": kg["kg_summary"]})
        if kg.get("graph_jsonable"):
            _save(run_dir, pc.OUT_KG_GRAPH, kg["graph_jsonable"])
        log.info("KG %s (verdict=%s) in %.1fs",
                 "OK" if kg.get("ok") else f"FAILED ({kg.get('error')})",
                 kg.get("verdict"), kg.get("seconds", 0))
    else:
        log.warning("stage 3/5 skipped — VLM branch unavailable")
    bundle["kg"] = kg

    # ---------------- [4] GNN inference ----------------------------------- #
    gnn: Dict[str, Any] = {"ok": False, "error": "skipped: no KG tensor row"}
    if kg.get("ok") and kg.get("tensor_row"):
        log.info("── stage 4/5: GNN (%s / %s) ──", gnn_input, gnn_arch)
        gnn = gnn_infer.run_gnn_stage(kg["tensor_row"], gnn_input=gnn_input,
                                      arch=gnn_arch, device="cpu")
        log.info("GNN %s -> %s %s in %.1fs",
                 "OK" if gnn.get("ok") else f"FAILED ({gnn.get('error')})",
                 gnn.get("grade"), gnn.get("probs"), gnn.get("seconds", 0))
    else:
        log.warning("stage 4/5 skipped — falling back to CV-only assessment")
    bundle["gnn"] = gnn
    _save(run_dir, pc.OUT_GNN, gnn)

    # ---------------- [5] assessment + BOTH reports ------------------------ #
    log.info("── stage 5/5: assessment + reports (civilian + engineer) ──")
    assessment = report_builder.build_assessment(bundle)
    assessment["total_seconds"] = round(time.time() - t0, 1)
    _save(run_dir, pc.OUT_ASSESSMENT, assessment)
    if reports in ("both", "public"):
        (run_dir / pc.OUT_REPORT_MD).write_text(
            report_builder.render_public_report(assessment), encoding="utf-8")
    if reports in ("both", "engineer"):
        (run_dir / OUT_REPORT_ENGINEER_MD).write_text(
            report_builder.render_engineer_report(assessment), encoding="utf-8")

    log.info("=" * 64)
    log.info("FINAL GRADE: %s   (basis=%s%s)", assessment["final_grade"],
             assessment["assessment_basis"],
             f", conservative={assessment['conservative_grade']}"
             if assessment["escalation_needed"] else "")
    if reports in ("both", "public"):
        log.info("report (civilian) -> %s", run_dir / pc.OUT_REPORT_MD)
    if reports in ("both", "engineer"):
        log.info("report (engineer) -> %s", run_dir / OUT_REPORT_ENGINEER_MD)
    log.info("details -> %s", run_dir / pc.OUT_ASSESSMENT)
    log.info("total %.1fs", assessment["total_seconds"])
    return assessment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="path to ONE post-earthquake building photo")
    ap.add_argument("--element", required=True, choices=pc.ELEMENTS,
                    help="structural type the user picked (wall/column/beam)")
    ap.add_argument("--gnn-input", default=pc.DEFAULT_GNN_INPUT,
                    choices=sorted(pc.GNN_INPUT_TO_VARIANT),
                    help="what the GNN sees from the CV branch — also selects "
                         "the matching trained checkpoint (default: full)")
    ap.add_argument("--gnn-arch", default=pc.DEFAULT_GNN_ARCH, choices=pc.GNN_ARCHS,
                    help="GNN architecture (default: rgcn)")
    ap.add_argument("--gpu", type=int, default=pc.DEFAULT_GPU_ID,
                    help="the single GPU id; the vLLM server must be on port 8000+gpu")
    ap.add_argument("--vlm-profile", default=pc.VLM_PROFILE)
    ap.add_argument("--vlm-backend", default="vllm", choices=["vllm", "mock"],
                    help="'mock' exercises the whole pipeline without a GPU/server")
    ap.add_argument("--cv-device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="device for the CV models (cpu if VRAM beside vLLM is tight)")
    ap.add_argument("--cv-ratio", type=float, default=None,
                    help="cm-per-pixel override for the CV cost estimator")
    ap.add_argument("--out-dir", default=None,
                    help="override the run directory (default main_pipeline/runs/<run_id>)")
    ap.add_argument("--reports", default="both",
                    choices=["both", "public", "engineer"],
                    help="which report(s) to write: civilian report.md, the "
                         "engineer report_engineer.md, or both (default)")
    ap.add_argument("--debug-crops", action="store_true",
                    help="save YOLO crops + segmentation masks under <run>/cv_debug/")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    try:
        assessment = run_pipeline(
            a.image, a.element, gnn_input=a.gnn_input, gnn_arch=a.gnn_arch,
            gpu=a.gpu, vlm_profile=a.vlm_profile, vlm_backend=a.vlm_backend,
            cv_device=a.cv_device, cv_ratio=a.cv_ratio, out_dir=a.out_dir,
            debug_crops=a.debug_crops, reports=a.reports, verbose=a.verbose)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"\n損傷評估等級 / Damage grade: {assessment['final_grade']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())