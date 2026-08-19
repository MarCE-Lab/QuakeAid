"""
main_pipeline/cv_infer.py
=========================
Stage 1 — the CV branch on ONE image.

Thin, fault-isolated wrapper around CV_modules.Model_pipeLine.run_cv (YOLO
detection -> crack-ViT classification -> crack/spalling U-Net segmentation ->
rule-based cost, plus the ViT-EDL A/B/C severity head). Adds:

  * a `cv_record` shaped exactly like one results/cv_cache_eval.jsonl line
    ({image_path, element, structured_output}) so the KG fusion path at
    inference is byte-compatible with how the GNN training data was built;
  * derived, report-friendly numbers (crack/spalling area RATIOS + the same
    none/minor/moderate/extensive buckets export_pyg uses, total cost);
  * an independent RULE-BASED A/B/C from CV_modules.cv_rule_mapping — used by
    run_pipeline as the conservative cross-check and as the fallback grade
    when the Validator branch is unavailable;
  * optional GPU release after inference (single-GPU deployments share the
    card with the persistent vLLM server).

Heavy deps (torch/cv2/ultralytics) are imported lazily inside run_cv_stage so
the rest of the pipeline (report, KG on CPU) can import this module freely.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from main_pipeline import pipeline_config as pc

log = logging.getLogger("main_pipeline.cv")

# export_pyg.CV_AREA_BUCKETS — kept numerically identical so the report's
# wording matches what the GNN actually saw.
AREA_BUCKET_EDGES = [0.0, 0.01, 0.05, 0.15]
AREA_BUCKET_WORDS_ZH = ["未見", "少量", "中等", "大範圍"]


def area_bucket(ratio: float) -> int:
    b = 0
    for i, edge in enumerate(AREA_BUCKET_EDGES):
        if ratio >= edge:
            b = i
    return b


def _ratio(area_px, total_px) -> float:
    try:
        a, t = float(area_px), float(total_px)
        return max(0.0, min(1.0, a / t)) if t > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def derive_cv_extras(so: Dict[str, Any], element: str) -> Dict[str, Any]:
    """Report-friendly numbers derived deterministically from structured_output."""
    total_px = so.get("area")
    if not total_px and isinstance(so.get("size"), (list, tuple)) and len(so["size"]) == 2:
        try:
            total_px = float(so["size"][0]) * float(so["size"][1])
        except (TypeError, ValueError):
            total_px = None
    crack_ratio = _ratio(so.get("crack area"), total_px)
    spall_ratio = _ratio(so.get("spalling area"), total_px)
    cost_total = sum(int(so.get(k, 0) or 0) for k in (
        "estimated cost of rebar", "estimated cost of crack", "estimated cost of spalling"))

    # independent rule-based A/B/C (CV observations only, no EDL) — the same
    # tables the ablation's config-A used (CV_modules/cv_rule_mapping.py)
    rule_grade = rule_reasons = None
    try:
        from CV_modules.cv_rule_mapping import grade_structured_output
        rule_grade, rule_reasons, mapped = grade_structured_output(element, so)
        if not mapped:
            rule_grade = None            # nothing in the image mapped -> abstain
            rule_reasons = None
    except Exception as e:               # never let the extras kill the stage
        log.warning("cv_rule_mapping unavailable/failed: %s", e)

    return {
        "crack_area_ratio": round(crack_ratio, 5),
        "spalling_area_ratio": round(spall_ratio, 5),
        "crack_area_bucket": area_bucket(crack_ratio),
        "spalling_area_bucket": area_bucket(spall_ratio),
        "estimated_cost_total": cost_total,
        "cv_rule_grade": rule_grade,
        "cv_rule_reasons": rule_reasons,
    }


def run_cv_stage(image_path: str | Path,
                 element: str,
                 device: str = "auto",
                 ratio: Optional[float] = None,
                 debug_dir: Optional[str] = None,
                 release_gpu: bool = True) -> Dict[str, Any]:
    """Run the whole CV branch on one image.

    Returns {"ok", "seconds", "element", "image_path",
             "structured_output" | "error", "derived", "cv_record"}.
    Raises nothing — the orchestrator decides how to degrade on failure.
    """
    image_path = str(image_path)
    if element not in pc.ELEMENTS:
        return {"ok": False, "error": f"element must be one of {pc.ELEMENTS}, got {element!r}",
                "image_path": image_path, "element": element}

    t0 = time.time()
    out: Dict[str, Any] = {"image_path": image_path, "element": element}
    try:
        import torch
        from CV_modules.Model_pipeLine import run_cv
        from CV_modules import cv_config as cvcfg

        dev = None if device == "auto" else device
        kwargs: Dict[str, Any] = {"device": dev}
        if ratio is not None:
            kwargs["ratio"] = ratio
        if debug_dir:
            kwargs["debug_dir"] = debug_dir

        so = run_cv(image_path, element, **kwargs)
        out.update({
            "ok": True,
            "structured_output": so,
            "derived": derive_cv_extras(so, element),
            "cv_record": {"image_path": image_path, "element": element,
                          "structured_output": so},
            "cv_config": {"ratio": ratio if ratio is not None else cvcfg.DEFAULT_RATIO,
                          "yolo_conf": cvcfg.YOLO_CONF,
                          "edl_checkpoints": dict(cvcfg.EDL_CHECKPOINTS)},
        })
    except Exception as e:  # noqa: BLE001 — fault-isolated stage
        log.exception("CV stage failed on %s", image_path)
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    finally:
        if release_gpu:
            _release_cv_gpu()
        out["seconds"] = round(time.time() - t0, 2)
    return out


def _release_cv_gpu() -> None:
    """Drop the cached CV models and return their VRAM to the pool. On the
    single-GPU box the persistent vLLM server holds its (reduced) reservation
    regardless, but freeing keeps repeated pipeline runs from accumulating."""
    try:
        import torch
        from CV_modules import Model_pipeLine as MP
        MP._CACHE.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("CV models released (CUDA cache emptied)")
    except Exception:
        pass


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Run the CV branch on one image.")
    ap.add_argument("image")
    ap.add_argument("--element", required=True, choices=pc.ELEMENTS)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--debug-dir", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_cv_stage(a.image, a.element, device=a.device,
                                  debug_dir=a.debug_dir),
                     ensure_ascii=False, indent=2))
