"""
main_pipeline/vlm_infer.py
==========================
Stage 2 — the VLM branch on ONE image, on ONE GPU.

Production redesign of validator/vlm/run_vlm_extraction.py:
  * the batch orchestrator shards a dataset across N data-parallel vLLM
    replicas (GPUs 4-7). Here there is exactly ONE image and ONE server
    (started by main_pipeline/serve_vlm.sh on port 8000+gpu), so the sharding /
    ThreadPool machinery is dropped and a single sequential two-pass call
    remains — per-image behaviour is IDENTICAL (each test replica already was
    a single-GPU model; multi-GPU only ever bought throughput).
  * few-shot exemplars and the structural-type hint are ON by default (the
    production configuration selected by the formal eval).
  * everything semantic is REUSED from the validator package (prompts, schema
    loading, message building, post-generation normalisation + validation), so
    this file cannot drift from the trial pipeline.

Flow per image:
    PASS 1 extraction  (system + extraction prompt + hint + few-shot + image)
    PASS 2 reasoning   (same image + the pass-1 JSON appended to the prompt)
Both passes run under the server's enforced JSON schema; the stable image_id is
injected into image_context.id of both outputs (the KG join key).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from main_pipeline import pipeline_config as pc

log = logging.getLogger("main_pipeline.vlm")


def make_image_id(image_path: str | Path) -> str:
    """Stable id for a standalone production image: derived from the file name
    (dataset_root = the image's own folder), same scheme as the trial runs."""
    from validator.vlm import vlm_config as vcfg
    p = Path(image_path).resolve()
    return vcfg.make_image_id(p, p.parent)


def server_health(gpu_id: int, timeout: float = 3.0) -> bool:
    from validator.vlm import vlm_config as vcfg
    url = f"http://{vcfg.SERVER_HOST}:{vcfg.server_port(gpu_id)}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def run_vlm_stage(image_path: str | Path,
                  element: str,
                  gpu_id: int = pc.DEFAULT_GPU_ID,
                  profile: str = pc.VLM_PROFILE,
                  backend_kind: str = "vllm",
                  few_shot: bool = pc.VLM_FEW_SHOT,
                  element_hint: bool = pc.VLM_ELEMENT_HINT,
                  image_id: Optional[str] = None) -> Dict[str, Any]:
    """Two-pass VLM extraction+reasoning for one image on one server.

    backend_kind: "vllm" (production) | "mock" (no GPU; returns the repo
    example — lets the whole downstream KG/GNN/report path be exercised).

    Returns {"ok","image_id","extraction","reasoning","seconds", ...} or
    {"ok":False,"error":...}. Never raises.
    """
    from validator.vlm import vlm_config as vcfg
    from validator.vlm import backends as bk
    from validator.vlm import vlm_postprocess as post
    from validator.vlm.run_vlm_extraction import (
        build_messages, load_prompts, load_fewshot, load_generation_schema,
        make_full_schema_validator, finalize_generated_json, _generate_with_retry)

    image_path = str(image_path)
    iid = image_id or make_image_id(image_path)
    t0 = time.time()
    out: Dict[str, Any] = {"image_id": iid, "image_path": image_path,
                           "element": element, "profile": profile,
                           "few_shot": few_shot, "element_hint": element_hint,
                           "backend": backend_kind, "gpu_id": gpu_id}
    try:
        if backend_kind == "vllm" and not server_health(gpu_id):
            raise ConnectionError(
                f"vLLM server not reachable on GPU {gpu_id} "
                f"(http://{vcfg.SERVER_HOST}:{vcfg.server_port(gpu_id)}/health). "
                f"Start it first:  bash main_pipeline/serve_vlm.sh --gpu {gpu_id}")

        prompts = load_prompts()
        generation_schema = load_generation_schema(None)
        full_validator = make_full_schema_validator(vcfg.load_schema(), enabled=True)
        fewshot = ({"extraction": load_fewshot("extraction"),
                    "reasoning": load_fewshot("reasoning")} if few_shot else {})

        backend = bk.get_backend("mock" if backend_kind == "mock" else profile,
                                 kind=backend_kind,
                                 base_urls=[vcfg.server_base_url(gpu_id)]
                                 if backend_kind == "vllm" else None)
        hint = element if element_hint else None
        # v2: letterbox-pad to square (never stretch) + true-WxH metadata; the
        # meta drives the 【影像技術資訊】 prompt block AND the runner-injected
        # image_context.source_image, exactly as in the trial orchestrator.
        data_url, image_meta = bk.prepare_image_for_vlm(image_path)
        out["image_meta"] = image_meta

        # ---- PASS 1: extraction ------------------------------------------ #
        log.info("VLM pass 1/2 (extraction) on %s ...", iid)
        m1 = build_messages(prompts["system"], prompts["extraction"], data_url,
                            element_hint=hint, fewshot=fewshot.get("extraction"),
                            image_meta=image_meta)
        extraction = _generate_with_retry(backend, m1, generation_schema,
                                          "pass1/extraction", iid)
        extraction = finalize_generated_json(
            extraction, image_id=iid, pass_completed=vcfg.PASS_EXTRACTION,
            full_schema_validator=full_validator, validate_graph=True,
            image_meta=image_meta, element_hint=hint)

        # ---- PASS 2: reasoning (same image + the pass-1 JSON) ------------ #
        log.info("VLM pass 2/2 (reasoning) on %s ...", iid)
        m2 = build_messages(prompts["system"], prompts["reasoning"], data_url,
                            extraction_json=extraction, element_hint=hint,
                            fewshot=fewshot.get("reasoning"),
                            image_meta=image_meta)
        try:
            reasoning = _generate_with_retry(backend, m2, generation_schema,
                                             "pass2/reasoning", iid)
        except Exception as e2:  # noqa: BLE001 — v3: degrade, don't drop pass 1
            out["reasoning_synthesized"] = True
            out["pass2_error"] = f"{type(e2).__name__}: {e2}"
            log.error("pass 2 failed twice on %s — synthesizing reasoning from "
                      "pass 1 (%s)", iid, out["pass2_error"])
            reasoning = post.synthesize_reasoning_from_extraction(
                extraction, element_hint=hint, image_meta=image_meta)
        reasoning = finalize_generated_json(
            reasoning, image_id=iid, pass_completed=vcfg.PASS_REASONING,
            full_schema_validator=full_validator, validate_graph=True,
            image_meta=image_meta, element_hint=hint)

        out.update({"ok": True, "extraction": extraction, "reasoning": reasoning,
                    "run_meta": vcfg.run_meta(
                        "mock" if backend_kind == "mock" else profile,
                        few_shot=few_shot,
                        extra={"element_hint_provided": element_hint,
                               "pipeline": pc.PIPELINE_VERSION})})
    except Exception as e:  # noqa: BLE001 — fault-isolated stage
        log.exception("VLM stage failed on %s", iid)
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    out["seconds"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Two-pass VLM on one image (one GPU).")
    ap.add_argument("image")
    ap.add_argument("--element", required=True, choices=pc.ELEMENTS)
    ap.add_argument("--gpu", type=int, default=pc.DEFAULT_GPU_ID)
    ap.add_argument("--profile", default=pc.VLM_PROFILE)
    ap.add_argument("--backend", default="vllm", choices=["vllm", "mock"])
    ap.add_argument("--no-few-shot", dest="few_shot", action="store_false")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    r = run_vlm_stage(a.image, a.element, gpu_id=a.gpu, profile=a.profile,
                      backend_kind=a.backend, few_shot=a.few_shot)
    print(json.dumps({k: v for k, v in r.items() if k != "extraction"},
                     ensure_ascii=False, indent=2)[:4000])


