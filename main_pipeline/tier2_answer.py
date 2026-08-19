#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_pipeline/tier2_answer.py
=============================
The Tier-II END-TO-END orchestrator: ONE user question in -> grounded zh-TW
answer out.

    question ──▶ [1] query-type classifier (T1–T5, per rag_config `routing:`)
             ──▶ [2] routed 3-channel RAG retrieval (sparse + dense + graph
                     via Neo4j) + RRF fusion + cross-encoder rerank
             ──▶ [3] prompt assembly: user question + retrieved knowledge
                     + Tier-I assessment result + Tier-I civilian report
             ──▶ [4] local LLM (Breeze-7B, optional LoRA) -> answer

Prerequisites (start once, stay up):
    bash main_pipeline/serve_neo4j.sh start        # graph channel backend
    # (the Tier-I vLLM server is NOT needed for Tier-II)

Typical usage
-------------
# Answer against the most recent Tier-I run (assessment.json + report.md):
    PYTHONPATH=. python3 -m main_pipeline.tier2_answer \
        --query "牆上有斜向裂縫，這樣還能住嗎？" --latest-run

# Point at a specific Tier-I run directory:
    PYTHONPATH=. python3 -m main_pipeline.tier2_answer \
        --query "修這種裂縫大概要多少錢？" \
        --run-dir main_pipeline/runs/20260601_120000_IMG_0007__ab12cd

# Standalone domain Q&A (no image / no Tier-I context):
    PYTHONPATH=. python3 -m main_pipeline.tier2_answer \
        --query "什麼是短柱效應？"

# Interactive chat loop (models stay loaded between questions):
    PYTHONPATH=. python3 -m main_pipeline.tier2_answer --latest-run --interactive

# Retrieval-only smoke test (no LLM weights loaded):
    PYTHONPATH=. python3 -m main_pipeline.tier2_answer \
        --query "X形裂縫代表什麼？" --no-llm --show-docs

Degradation policy (mirrors Tier-I's "never fail silently"):
  * Neo4j / graph channel down -> RAGPipeline already degrades to 2-channel
    (sparse+dense) per query; we surface `route` so the caller can see it.
  * LLM unavailable            -> --no-llm returns retrieval + the assembled
    prompt; --mock-llm returns a deterministic canned answer (UI testing).
  * No Tier-I context          -> the prompt says so explicitly; retrieval
    still routes on the question alone (classifier).

GPU note: BGE-M3 encoder + bge-reranker + Breeze-7B(fp16) together need
~18 GB. On the single-GPU box they do NOT fit beside the Tier-I vLLM server
(which reserves ~70% of a 24 GB card) — either stop the vLLM server first
(`bash main_pipeline/serve_vlm.sh --stop`), pin Tier-II to another GPU
(`CUDA_VISIBLE_DEVICES=1 ...`), or run the LLM on CPU (`--device cpu`, slow).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from main_pipeline import pipeline_config as pc

log = logging.getLogger("main_pipeline.tier2")

REPO_ROOT = pc.REPO_ROOT
DEFAULT_RAG_CONFIG = REPO_ROOT / "rag_system" / "config" / "rag_config.yaml"
DEFAULT_LLM_CONFIG = REPO_ROOT / "LLM_agent" / "config" / "llm_config.yaml"

# Graph-channel passages carry this doc_id prefix in the merged corpus
# (see rag_pipeline._passage_id_to_doc_idx) — we route them into the
# 【領域知識關聯】 section of the prompt instead of the generic doc list.
GRAPH_DOCID_PREFIX = "graph_kg::"

# Prompt char budgets (chars, not tokens; zh chars ≈ 1 token under Breeze's
# tokenizer, so these keep the whole prompt safely inside max_length=2048).
DEFAULT_REPORT_BUDGET = 1000       # trimmed report.md excerpt
DEFAULT_DOC_CHAR_CAP = 500         # per retrieved passage (original used 600)
DEFAULT_HISTORY_TURNS = 2          # last N (user, assistant) pairs
DEFAULT_HISTORY_CHAR_CAP = 160     # per history message


# ═══════════════════════════════════════════════════════════════════════════
# Tier-I context loading
# ═══════════════════════════════════════════════════════════════════════════
def find_latest_run(runs_dir: Path = pc.RUNS_DIR) -> Optional[Path]:
    """Newest run directory that actually finished (has assessment.json)."""
    if not runs_dir.is_dir():
        return None
    candidates = [d for d in runs_dir.iterdir()
                  if d.is_dir() and (d / pc.OUT_ASSESSMENT).exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: (d / pc.OUT_ASSESSMENT).stat().st_mtime)


def load_tier1_context(run_dir: str | Path) -> Dict[str, Any]:
    """Read assessment.json + report.md from ONE Tier-I run directory.
    Missing files degrade to None (never raises for a partial run)."""
    run_dir = Path(run_dir)
    out: Dict[str, Any] = {"run_dir": str(run_dir), "assessment": None,
                           "report_md": None}
    ap = run_dir / pc.OUT_ASSESSMENT
    if ap.exists():
        try:
            out["assessment"] = json.loads(ap.read_text(encoding="utf-8"))
        except Exception as e:                                    # noqa: BLE001
            log.warning("could not parse %s: %s", ap, e)
    rp = run_dir / pc.OUT_REPORT_MD
    if rp.exists():
        try:
            out["report_md"] = rp.read_text(encoding="utf-8")
        except Exception as e:                                    # noqa: BLE001
            log.warning("could not read %s: %s", rp, e)
    return out


def damage_result_from_assessment(assessment: Optional[dict]) -> Optional[dict]:
    """assessment.json -> the `damage_result` dict the RAG pipeline consumes.

    Keys follow rag_system conventions (prompt_builder + metadata_filter +
    RAGPipeline._build_query): element / damage / pattern / severity / cost.
    Values are zh where the corpus metadata is zh (infer_metadata_filter
    passes unknown values through verbatim, so zh strings match directly).
    """
    if not assessment:
        return None
    cv = assessment.get("cv_findings") or {}
    pats_a = [p for p in (cv.get("patterns_a") or []) if p]
    pats_b = [p for p in (cv.get("patterns_b") or []) if p]
    dr = {
        "element": assessment.get("element_zh") or assessment.get("element") or "",
        "damage": "、".join(pats_a),
        "pattern": (pats_b[0] if pats_b else (pats_a[0] if pats_a else "")),
        "severity": assessment.get("final_grade") or "",
        "cost": int(cv.get("cost_total") or 0),
    }
    return dr


# ═══════════════════════════════════════════════════════════════════════════
# Prompt assembly (extends rag_system/rag/prompt_builder.py with the Tier-I
# assessment digest + trimmed civilian report + short chat history)
# ═══════════════════════════════════════════════════════════════════════════
def _pct(x, digits: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def assessment_digest_zh(a: dict) -> str:
    """Deterministic ~10-line zh digest of assessment.json for the prompt.
    Mirrors report_builder's vocabulary; safe on partial assessments."""
    lines: List[str] = []
    g = a.get("final_grade") or "—"
    head = {"A": "A 級（嚴重受損，危險）", "B": "B 級（中度受損，需注意）",
            "C": "C 級（輕微或未見明顯損傷）"}.get(g, g)
    lines.append(f"最終判定：{head}")
    if a.get("escalation_needed") and a.get("conservative_grade"):
        lines.append(f"安全提醒：部分模組訊號指向更嚴重的 "
                     f"{a['conservative_grade']} 級，行動建議以較嚴重等級為準。")
    basis = a.get("assessment_basis")
    if basis == pc.BASIS_CV_ONLY:
        lines.append("判定依據：僅影像偵測模組（備援模式，可靠度較低）。")
    elif basis:
        lines.append("判定依據：圖神經網路綜合模型。")
    el = a.get("element_zh") or a.get("element")
    if el:
        lines.append(f"結構類型（使用者指定）：{el}")
    cv = a.get("cv_findings") or {}
    if cv.get("available"):
        pats = (cv.get("patterns_a") or []) + (cv.get("patterns_b") or [])
        if pats:
            lines.append(f"影像偵測到的損傷：{'、'.join(dict.fromkeys(pats))}")
        lines.append(f"裂縫面積比：約 {_pct(cv.get('crack_ratio'))}"
                     f"（{cv.get('crack_bucket_zh', '')}）；"
                     f"剝落面積比：約 {_pct(cv.get('spall_ratio'))}"
                     f"（{cv.get('spall_bucket_zh', '')}）")
        if cv.get("cost_total"):
            lines.append(f"規則式初估修復費用：約 NT$ {int(cv['cost_total']):,}"
                         f"（僅供參考）")
    vlm = a.get("vlm_findings") or {}
    if vlm.get("available") and vlm.get("members"):
        worst = vlm["members"][0]
        if worst.get("grade"):
            lines.append(f"最嚴重結構：{worst.get('type_zh', '結構')} "
                         f"損傷 {worst['grade']} 級")
        if vlm.get("global_indicators"):
            lines.append(f"整體性警訊：{'、'.join(vlm['global_indicators'])}")
        if vlm.get("hazards"):
            lines.append(f"周邊危險：{'、'.join(vlm['hazards'])}")
    kgf = a.get("kg_findings") or {}
    if kgf.get("available") and kgf.get("signatures_zh"):
        lines.append(f"關鍵危險徵兆（規則交叉證實）："
                     f"{'、'.join(kgf['signatures_zh'])}")
    return "\n".join(lines)


_REPORT_KEEP_START = re.compile(r"^##\s*二、")     # keep from「二、評估結果」
_REPORT_KEEP_END = re.compile(r"^##\s*五、")       # …up to (excl.)「五、注意事項」


def trim_report_md(report_md: str, budget: int = DEFAULT_REPORT_BUDGET) -> str:
    """Cut report.md down to the informative core for the prompt:
    drop the header/disclaimer/footer boilerplate, keep sections 二–四,
    collapse blank runs, hard-cap at `budget` chars."""
    if not report_md:
        return ""
    lines = report_md.splitlines()
    start = next((i for i, l in enumerate(lines) if _REPORT_KEEP_START.match(l)), 0)
    end = next((i for i, l in enumerate(lines) if _REPORT_KEEP_END.match(l)), len(lines))
    kept: List[str] = []
    for l in lines[start:end]:
        s = l.rstrip()
        if s.startswith(">"):                     # disclaimers/banners: digest has them
            continue
        if not s and kept and not kept[-1]:
            continue                              # collapse blank runs
        kept.append(s)
    text = "\n".join(kept).strip()
    if len(text) > budget:
        text = text[:budget].rstrip() + "\n…（報告其餘內容略）"
    return text


def build_tier2_prompt(question: str,
                       docs: List[dict],
                       kg_facts: List[str],
                       damage_result: Optional[dict] = None,
                       assessment: Optional[dict] = None,
                       report_md: Optional[str] = None,
                       history: Optional[List[dict]] = None,
                       report_budget: int = DEFAULT_REPORT_BUDGET,
                       doc_char_cap: int = DEFAULT_DOC_CHAR_CAP,
                       max_docs: int = 3,
                       max_kg_facts: int = 8) -> str:
    """Assemble the full Tier-II prompt.

    Layout intentionally mirrors rag_system/rag/prompt_builder.build_prompt
    (same section markers, same system instruction) and ADDS the Tier-I
    assessment digest + trimmed report + short chat history, per the
    Tier-III integration contract in project_state.md.
    """
    from rag_system.rag.prompt_builder import SYSTEM_INSTRUCTION_ZH

    parts: List[str] = [SYSTEM_INSTRUCTION_ZH, ""]

    # ---- Tier-I assessment ------------------------------------------------
    if assessment:
        parts.append("【損傷評估結果（AI 快篩）】")
        parts.append(assessment_digest_zh(assessment))
    elif damage_result:
        dr = damage_result
        parts.append("【損傷評估結果】")
        parts.append(f"構件類型：{dr.get('element') or '未知'}")
        parts.append(f"損傷類型：{dr.get('damage') or '未知'}")
        parts.append(f"損傷型態：{dr.get('pattern') or '未知'}")
        parts.append(f"損傷等級：{dr.get('severity') or '未知'}")
        if dr.get("cost"):
            parts.append(f"費用估算：NT${int(dr['cost']):,}"
                         f"（僅供參考，實際費用請諮詢專業技師）")
    else:
        parts.append("【損傷評估結果】")
        parts.append("（使用者尚未上傳影像進行評估；請以一般性知識回答，"
                     "並提醒使用者可先進行影像快篩。）")
    parts.append("")

    # ---- Tier-I civilian report excerpt -----------------------------------
    if report_md and report_budget > 0:
        excerpt = trim_report_md(report_md, budget=report_budget)
        if excerpt:
            parts.append("【評估報告摘要】")
            parts.append(excerpt)
            parts.append("")

    # ---- retrieved passages ------------------------------------------------
    parts.append("【參考知識】")
    if docs:
        for i, d in enumerate(docs[:max_docs]):
            text = (d.get("text") or "").strip().replace("\n", " ")
            parts.append(f"[資料{i + 1}] {text[:doc_char_cap]}")
            parts.append("")
    else:
        parts.append("（本次未檢索到相關文件）")
        parts.append("")

    # ---- graph facts --------------------------------------------------------
    if kg_facts:
        parts.append("【領域知識關聯】")
        for fact in kg_facts[:max_kg_facts]:
            parts.append(f"• {str(fact).strip()}")
        parts.append("")

    # ---- short chat history --------------------------------------------------
    hist = [h for h in (history or [])
            if isinstance(h, dict) and h.get("content")]
    if hist:
        hist = hist[-(DEFAULT_HISTORY_TURNS * 2):]
        parts.append("【先前對話】")
        for h in hist:
            who = "使用者" if h.get("role") == "user" else "助理"
            parts.append(f"{who}：{str(h['content'])[:DEFAULT_HISTORY_CHAR_CAP]}")
        parts.append("")

    parts.append("【用戶問題】")
    parts.append(question.strip())
    parts.append("")
    parts.append("請根據以上資訊，用繁體中文回答用戶的問題。"
                 "回答時使用【標題】格式分段，5-7句以內。"
                 "若資訊不足以確定，請誠實說明並建議尋求專業技師協助。")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Mock LLM (UI / pipeline testing without GPU weights)
# ═══════════════════════════════════════════════════════════════════════════
class MockLLM:
    """Deterministic stand-in for BreezeLLM — exercises the WHOLE Tier-II
    path (classifier, retrieval, prompt assembly) without model weights."""
    base_model_name = "mock"

    def generate(self, prompt: str, **_) -> str:
        n_docs = prompt.count("[資料")
        return ("【測試回覆】\n"
                "這是 Tier-II 測試模式（--mock-llm）產生的固定回覆，"
                f"僅用於驗證流程：本次檢索到 {n_docs} 份參考資料，"
                "提示詞已完整組裝。正式部署時請改用 Breeze-7B 模型。\n"
                "【提醒】本系統僅供快篩參考，實際安全狀態請以專業技師評估為準。")


class Breeze4BitLLM:
    """Breeze-7B loaded in 4-bit NF4 via bitsandbytes (~4.5 GB VRAM instead
    of ~14 GB fp16). Same .generate() contract as LLM_agent's BreezeLLM.

    This is THE deployment path for a single 16 GB V100 — fp16 Breeze alone
    almost fills the card and cannot coexist with the Tier-I VLM server.
    NF4 works on compute capability 7.0 (Volta); compute dtype is float16
    because the V100 has no bfloat16. Enable via TIER2_4BIT=1 / --four-bit.
    Requires: pip install bitsandbytes accelerate
    """

    def __init__(self,
                 base_model: str = "MediaTek-Research/Breeze-7B-Instruct-v1_0",
                 lora_path: Optional[str] = None,
                 max_length: int = 2048):
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer,
                                  BitsAndBytesConfig)
        self.base_model_name = base_model + " (4-bit NF4)"
        self.max_length = max_length
        log.info("[LLM-4bit] loading tokenizer: %s", base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,   # V100: no bf16
        )
        log.info("[LLM-4bit] loading %s with NF4 quantization…", base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb, device_map="auto",
            trust_remote_code=True)
        if lora_path and Path(lora_path).exists():
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                log.info("[LLM-4bit] LoRA adapter attached: %s", lora_path)
            except Exception as e:                               # noqa: BLE001
                log.warning("[LLM-4bit] LoRA load failed (%s); base only", e)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 512,
                 temperature: float = 0.3, top_p: float = 0.9,
                 do_sample: bool = True,
                 repetition_penalty: float = 1.1) -> str:
        import torch
        messages = [{"role": "system", "content": ""},
                    {"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:                                        # noqa: BLE001
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=self.max_length
                                ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════════
# The engine (importable by the web backend; the CLI below is a thin wrapper)
# ═══════════════════════════════════════════════════════════════════════════
class Tier2Engine:
    """Loads the RAG pipeline + LLM once; answers many questions.

    answer() is stateless w.r.t. Tier-I context — pass `assessment` /
    `report_md` per call so one loaded engine can serve many runs/users.
    """

    def __init__(self,
                 rag_config_path: str | Path = DEFAULT_RAG_CONFIG,
                 llm_config_path: str | Path = DEFAULT_LLM_CONFIG,
                 device: Optional[str] = None,
                 mock_llm: bool = False,
                 no_llm: bool = False,
                 four_bit: Optional[bool] = None):
        self.rag_config_path = Path(rag_config_path)
        self.llm_config_path = Path(llm_config_path)
        self.device = device
        self.mock_llm = mock_llm
        self.no_llm = no_llm
        if four_bit is None:                      # env fallback for the webapp
            four_bit = os.environ.get("TIER2_4BIT", "0") == "1"
        self.four_bit = bool(four_bit)
        self.pipeline = None                     # rag_system RAGPipeline
        self.llm = None
        self.rag_cfg: Dict[str, Any] = {}
        self.gen_cfg: Dict[str, Any] = {}
        self.load_seconds: Dict[str, float] = {}

    # ------------------------------------------------------------------ load
    def load(self) -> "Tier2Engine":
        from rag_system.pipeline.rag_pipeline import RAGPipeline

        t0 = time.time()
        with open(self.rag_config_path, encoding="utf-8") as f:
            self.rag_cfg = yaml.safe_load(f)
        log.info("loading RAG pipeline (encoder + index + reranker + graph)…")
        self.pipeline = RAGPipeline(self.rag_cfg)
        self.load_seconds["rag"] = round(time.time() - t0, 1)
        log.info("RAG pipeline ready in %.1fs", self.load_seconds["rag"])

        llm_cfg: Dict[str, Any] = {}
        if self.llm_config_path.exists():
            with open(self.llm_config_path, encoding="utf-8") as f:
                llm_cfg = yaml.safe_load(f) or {}
        self.gen_cfg = dict(llm_cfg.get("generation") or {})

        if self.no_llm:
            log.info("LLM disabled (--no-llm): retrieval-only mode")
        elif self.mock_llm:
            self.llm = MockLLM()
            log.info("LLM mocked (--mock-llm)")
        elif self.four_bit:
            t1 = time.time()
            m = llm_cfg.get("llm") or {}
            self.llm = Breeze4BitLLM(
                base_model=m.get("base_model",
                                 "MediaTek-Research/Breeze-7B-Instruct-v1_0"),
                lora_path=m.get("lora_path"),
                max_length=int(m.get("max_length", 2048)),
            )
            self.load_seconds["llm"] = round(time.time() - t1, 1)
            log.info("LLM ready (4-bit NF4) in %.1fs", self.load_seconds["llm"])
        else:
            t1 = time.time()
            from LLM_agent.llm_inference import BreezeLLM
            m = llm_cfg.get("llm") or {}
            self.llm = BreezeLLM(
                base_model=m.get("base_model",
                                 "MediaTek-Research/Breeze-7B-Instruct-v1_0"),
                lora_path=m.get("lora_path"),
                device=self.device,
                use_fp16=bool(m.get("use_fp16", True)),
                max_length=int(m.get("max_length", 2048)),
            )
            self.load_seconds["llm"] = round(time.time() - t1, 1)
            log.info("LLM ready in %.1fs", self.load_seconds["llm"])
        return self

    # ---------------------------------------------------------------- answer
    def answer(self,
               question: str,
               assessment: Optional[dict] = None,
               report_md: Optional[str] = None,
               history: Optional[List[dict]] = None,
               qa_type: Optional[str] = None,
               top_k: Optional[int] = None,
               report_budget: int = DEFAULT_REPORT_BUDGET,
               include_report: bool = True) -> Dict[str, Any]:
        """Full Tier-II pass for one question. Returns a JSON-able dict:
        {answer, qa_type, route, sources[], prompt, prompt_chars, seconds{}}.
        """
        if self.pipeline is None:
            raise RuntimeError("Tier2Engine.load() must be called first")
        from rag_system.retrieval.metadata_filter import infer_metadata_filter

        question = (question or "").strip()
        if not question:
            raise ValueError("empty question")

        timings: Dict[str, float] = {}
        damage_result = damage_result_from_assessment(assessment)
        dr_for_pipeline = dict(damage_result) if damage_result else None
        if qa_type:
            dr_for_pipeline = dr_for_pipeline or {}
            dr_for_pipeline["qa_type"] = qa_type

        query = self.pipeline._build_query(damage_result, question)
        filters = infer_metadata_filter(damage_result) if damage_result else None

        prompt_cfg = self.rag_cfg.get("prompt") or {}
        k = top_k or int(self.rag_cfg.get("retrieval", {})
                         .get("reranker_top_k", 3))

        # ---- [1]+[2] classifier + routed retrieval ------------------------
        t0 = time.time()
        retrieved = self.pipeline.retrieve(
            query=query, metadata_filters=filters,
            damage_result=dr_for_pipeline, top_k=k)
        timings["retrieve"] = round(time.time() - t0, 2)

        route: Dict[str, Any] = {}
        try:                                   # bookkeeping only, never fatal
            r = self.pipeline._route(query=query, damage_result=dr_for_pipeline)
            route = {"qa_type": r.get("qa_type"),
                     "channels": sorted(r.get("channels") or []),
                     "weights": {c: round(float(w), 3)
                                 for c, w in (r.get("weights") or {}).items()},
                     "source": r.get("source")}
        except Exception as e:                                  # noqa: BLE001
            log.debug("route introspection failed: %s", e)

        # ---- hydrate texts; split graph passages into kg_facts ------------
        store = self.pipeline.index.doc_store
        docs: List[dict] = []
        kg_facts: List[str] = []
        sources: List[dict] = []
        for r in retrieved:
            di = r.get("doc_idx")
            doc = store.get(di) if isinstance(store, dict) else None
            if not doc:
                continue
            text = doc.get("text", "")
            doc_id = r.get("doc_id") or doc.get("doc_id") or f"doc_{di}"
            is_graph = str(doc_id).startswith(GRAPH_DOCID_PREFIX) \
                or bool(r.get("graph_provenance"))
            entry = {"text": text, **r}
            if is_graph:
                kg_facts.append(text)
            else:
                docs.append(entry)
            sources.append({
                "doc_id": doc_id,
                "text": text,
                "channel": "graph" if is_graph else "text",
                "rerank_score": round(float(r.get("rerank_score", 0.0)), 4),
                "rrf_score": round(float(r.get("rrf_score", 0.0)), 4),
                "metadata": doc.get("metadata") or {},
            })

        # ---- [3] prompt -----------------------------------------------------
        prompt = build_tier2_prompt(
            question=question, docs=docs, kg_facts=kg_facts,
            damage_result=damage_result, assessment=assessment,
            report_md=(report_md if include_report else None),
            history=history, report_budget=report_budget,
            max_docs=int(prompt_cfg.get("max_retrieved_docs", 3)),
            max_kg_facts=int(prompt_cfg.get("max_kg_facts", 8)),
        )

        # ---- [4] LLM ---------------------------------------------------------
        answer_text = None
        if self.llm is not None:
            t1 = time.time()
            gen = {k_: v for k_, v in self.gen_cfg.items()
                   if k_ in ("max_new_tokens", "temperature", "top_p",
                             "do_sample", "repetition_penalty")}
            try:
                answer_text = self.llm.generate(prompt, **gen)
            except Exception as e:                              # noqa: BLE001
                log.exception("LLM generation failed")
                answer_text = (f"【系統訊息】語言模型產生回覆時發生錯誤"
                               f"（{type(e).__name__}），請稍後再試；"
                               f"下方參考資料仍可供閱讀。")
            timings["generate"] = round(time.time() - t1, 2)

        return {
            "ok": True,
            "answer": answer_text,
            "qa_type": route.get("qa_type"),
            "route": route,
            "sources": sources,
            "n_docs": len(docs),
            "n_kg_facts": len(kg_facts),
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "seconds": timings,
            "llm_model": getattr(self.llm, "base_model_name", None),
            "used_tier1_context": bool(assessment),
        }

    # ----------------------------------------------------------------- close
    def close(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.close()
            except Exception:                                   # noqa: BLE001
                pass


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def _print_result(res: Dict[str, Any], show_docs: bool, show_prompt: bool) -> None:
    r = res.get("route") or {}
    print(f"\nroute: type={r.get('qa_type')}  channels={r.get('channels')}  "
          f"weights={r.get('weights')}  source={r.get('source')}")
    print(f"retrieved: {res['n_docs']} text passage(s) + "
          f"{res['n_kg_facts']} graph fact(s)   "
          f"[retrieve {res['seconds'].get('retrieve', '—')}s"
          + (f", generate {res['seconds']['generate']}s"
             if "generate" in res["seconds"] else "") + "]")
    if show_docs:
        for i, s in enumerate(res["sources"], 1):
            print(f"  [{i}] ({s['channel']}) {s['doc_id']}  "
                  f"rerank={s['rerank_score']}")
            print(f"      {s['text'][:160].replace(chr(10), ' ')}…")
    if show_prompt:
        print("\n" + "─" * 30 + " PROMPT " + "─" * 30)
        print(res["prompt"])
        print("─" * 68)
    if res.get("answer") is not None:
        print("\n" + "═" * 30 + " ANSWER " + "═" * 30)
        print(res["answer"])
        print("═" * 68)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", default=None,
                    help="the user question (omit with --interactive)")
    src = ap.add_argument_group("Tier-I context (pick at most one)")
    src.add_argument("--run-dir", default=None,
                     help="a Tier-I run directory (contains assessment.json)")
    src.add_argument("--latest-run", action="store_true",
                     help="use the newest finished run under main_pipeline/runs/")
    src.add_argument("--assessment", default=None,
                     help="explicit path to an assessment.json")
    src.add_argument("--report", default=None,
                     help="explicit path to a report.md")
    ap.add_argument("--rag-config", default=str(DEFAULT_RAG_CONFIG))
    ap.add_argument("--llm-config", default=str(DEFAULT_LLM_CONFIG))
    ap.add_argument("--qa-type", default=None,
                    help="force T1–T5 (bypasses the classifier)")
    ap.add_argument("--top-k", type=int, default=None,
                    help="passages after rerank (default: config reranker_top_k)")
    ap.add_argument("--report-budget", type=int, default=DEFAULT_REPORT_BUDGET,
                    help="max chars of report.md included in the prompt "
                         "(0 disables the report section)")
    ap.add_argument("--device", default=None,
                    help="LLM device override, e.g. cuda / cpu")
    ap.add_argument("--no-llm", action="store_true",
                    help="retrieval + prompt only (no model weights loaded)")
    ap.add_argument("--mock-llm", action="store_true",
                    help="canned answer (test the full path without a GPU)")
    ap.add_argument("--four-bit", action="store_true",
                    help="load Breeze-7B in 4-bit NF4 (~4.5 GB — required to "
                         "fit beside the VLM on a 16 GB V100); also via "
                         "TIER2_4BIT=1")
    ap.add_argument("--interactive", action="store_true",
                    help="chat loop; models stay loaded between questions")
    ap.add_argument("--show-docs", action="store_true")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--json", action="store_true",
                    help="print the full result as one JSON object")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if not a.query and not a.interactive:
        ap.error("--query is required (or use --interactive)")

    # ---- resolve Tier-I context ------------------------------------------
    assessment = report_md = None
    if a.run_dir or a.latest_run:
        run_dir = Path(a.run_dir) if a.run_dir else find_latest_run()
        if run_dir is None:
            print("WARNING: no finished Tier-I run found under "
                  f"{pc.RUNS_DIR} — continuing without Tier-I context.",
                  file=sys.stderr)
        else:
            ctx = load_tier1_context(run_dir)
            assessment, report_md = ctx["assessment"], ctx["report_md"]
            print(f"[tier2] Tier-I context: {run_dir.name}  "
                  f"(grade={ (assessment or {}).get('final_grade') })")
    if a.assessment:
        assessment = json.loads(Path(a.assessment).read_text(encoding="utf-8"))
    if a.report:
        report_md = Path(a.report).read_text(encoding="utf-8")

    engine = Tier2Engine(rag_config_path=a.rag_config,
                         llm_config_path=a.llm_config,
                         device=a.device, mock_llm=a.mock_llm,
                         no_llm=a.no_llm, four_bit=a.four_bit).load()

    history: List[dict] = []
    try:
        def _ask(q: str) -> None:
            res = engine.answer(q, assessment=assessment, report_md=report_md,
                                history=history, qa_type=a.qa_type,
                                top_k=a.top_k, report_budget=a.report_budget)
            if a.json:
                slim = {k: v for k, v in res.items() if k != "prompt" or a.show_prompt}
                print(json.dumps(slim, ensure_ascii=False, indent=2))
            else:
                _print_result(res, show_docs=a.show_docs,
                              show_prompt=a.show_prompt)
            if res.get("answer"):
                history.append({"role": "user", "content": q})
                history.append({"role": "assistant", "content": res["answer"]})

        if a.query:
            _ask(a.query)
        if a.interactive:
            print("\n[tier2] interactive mode — 輸入問題後 Enter，"
                  "輸入 exit / quit 離開。\n")
            while True:
                try:
                    q = input("問題> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "q"):
                    break
                _ask(q)
        return 0
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
