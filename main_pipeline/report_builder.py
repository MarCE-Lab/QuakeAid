# -*- coding: utf-8 -*-
"""
main_pipeline/report_builder.py
===============================
Stage 5 — the RULE-BASED reports (zh-TW) + the machine-readable assessment.json.

TWO audiences, one assessment dict:

  * render_engineer_report(a) -> report_engineer.md
      Everything, per module: raw enums beside their zh mapping, per-feature
      tables, EDL confidence/uncertainty, GNN probabilities, the internal
      placard value, validator verdict, stage timings and errors. NO advice,
      NO caveat prose — engineers read the numbers, not the warnings.

  * render_public_report(a)   -> report.md
      Exactly three sections:
        一、評估結果      — 1~2 sentence verdict + the detailed action list
        二、詳細說明      — ONE integrated paragraph, never split by module,
                            ordered "what we found → what it may lead to →
                            why the result is this grade (the mapping rule)"
        三、注意事項與限制 — same caveats as before
      Constraints enforced here: no module/pipeline names, no probabilities,
      no I~V or placard vocabulary. Evidence that CARRIES the headline grade is
      selected by severity (see _public_evidence) and stated in hedged wording
      ("似乎可以看到…"), because a single photo cannot support a definite claim;
      everything else is reported plainly. NOTE: the hedging is confined to the
      *descriptive* sentences — the verdict sentence and the action list in
      section 1 stay direct, so that softened description can never soften a
      safety instruction.

Design constraints this module enforces (project mandates + owner decisions):

  * Audience = general public, no engineering background. Every enum the
    components emit is mapped through fixed zh-TW dictionaries with a safe
    fallback (never crash on an unseen value, never show raw snake_case when a
    mapping exists). Numbers are rounded and bucketed into everyday words.
  * 100% deterministic ("rule-based"): fixed section templates + dictionary
    lookups + sorting/capping rules. No LLM anywhere in the report path, so the
    same inputs always render the same report.
  * NO placard vocabulary (紅單/黃單/綠單) is ever shown — the system outputs
    A/B/C only (project_state.md §"Few thing that matters" #3). The KG's
    internal placard_suggestion is translated into neutral severity wording;
    the raw value stays only in assessment.json for engineers.
  * FALSE-REASSURANCE guard in presentation: the headline stays the GNN's
    grade (the evaluated decision maker), but whenever ANY component's signal
    maps to a MORE severe grade, a prominent warning tells the reader to treat
    the building as the more severe level until professional inspection
    (assessment.json carries it as `conservative_grade`).
  * Degrades gracefully: every section renders from whatever components
    succeeded; missing pieces become explicit "無法取得" lines, and the
    CV-only fallback mode is labelled loudly.

Outputs
-------
build_assessment(bundle) -> dict          (assessment.json content)
render_report(assessment) -> str          (report.md content, zh-TW)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from main_pipeline import pipeline_config as pc
from main_pipeline.cv_infer import AREA_BUCKET_WORDS_ZH, area_bucket

# ═══════════════════════════════════════════════════════════════════════════
# zh-TW dictionaries (fixed; fallback = prettified original, never a crash)
# ═══════════════════════════════════════════════════════════════════════════
ELEMENT_ZH = {"wall": "牆", "column": "柱", "beam": "梁"}

MEMBER_TYPE_ZH = {
    "column": "柱", "beam": "梁", "beam_column_joint": "梁柱接頭",
    "shear_wall": "剪力牆", "bearing_wall": "承重牆", "rc_wall": "鋼筋混凝土牆",
    "partition_wall": "隔間牆", "curtain_wall": "帷幕牆",
    "spandrel_or_sill_wall": "窗台牆／窗間牆", "slab": "樓板", "staircase": "樓梯",
    "foundation_or_column_base": "基礎／柱腳", "brace": "斜撐", "parapet": "女兒牆",
    "non_structural_element": "非結構構件", "unknown": "",
}
MATERIAL_ZH = {
    "reinforced_concrete": "鋼筋混凝土", "steel": "鋼構", "src": "鋼骨鋼筋混凝土",
    "brick_masonry": "磚造", "confined_masonry": "加強磚造", "timber": "木造",
    "bamboo_mud": "竹木土造", "light_partition": "輕隔間", "glass": "玻璃",
    "unknown": "", "na": "",
}
ROLE_ZH = {
    "primary_load_bearing": "主要承重結構", "secondary_lateral": "次要側力結構",
    "non_structural": "非結構結構",
}
MECHANISM_ZH = {
    "flexural": "彎曲破壞", "shear": "剪力破壞（脆性）", "axial_compression": "軸壓破壞",
    "short_column_effect": "短柱效應", "joint_failure": "梁柱接頭破壞",
    "buckling": "挫屈", "connection_failure": "接合破壞",
    "non_structural_falling": "非結構物墜落", "non_structural_toppling": "非結構物傾倒",
    "settlement": "沉陷", "indeterminate": "機制無法判定", "na": "", "unknown": "",
}
DAMAGE_TYPE_ZH = {
    "hairline_crack": "髮絲裂縫", "vertical_crack": "垂直裂縫",
    "horizontal_crack": "水平裂縫", "diagonal_crack_45": "約45°斜向裂縫",
    "diagonal_crack": "斜向裂縫", "x_shape_crack": "X形交叉裂縫",
    "web_crack": "網狀裂縫", "flexural_crack": "彎曲裂縫", "shear_crack": "剪力裂縫",
    "through_crack": "貫穿裂縫", "interface_crack": "介面裂縫",
    "cover_spalling": "混凝土保護層剝落", "concrete_spalling": "混凝土剝落",
    "surface_spalling": "表面剝落", "plaster_spalling": "粉刷層剝落",
    "render_spalling": "粉刷層剝落", "concrete_crushing": "混凝土壓碎",
    "core_crushing": "核心混凝土碎裂", "rebar_exposed": "鋼筋外露",
    "rebar_buckling": "主筋挫屈", "rebar_fracture": "鋼筋斷裂",
    "stirrup_opening_or_fracture": "箍筋脫開／斷裂",
    "member_fracture": "結構斷裂", "residual_deformation": "殘餘變形",
    "tilt_or_lean": "傾斜", "joint_damage": "接頭損壞",
    "support_displacement": "支承位移", "finish_spalling": "粉刷層剝落",
    "tile_falling": "磁磚剝落", "window_or_glass_damage": "門窗／玻璃損壞",
}
POSITION_ZH = {
    "column_top": "柱頂", "column_mid": "柱中段", "column_base": "柱底／柱腳",
    "column_bottom": "柱底", "beam_top": "梁頂", "beam_bottom": "梁底",
    "wall_top": "牆頂", "wall_bottom": "牆底", "wall_base": "牆腳",
    "beam_end": "梁端", "beam_mid": "梁中段", "joint_panel": "接頭區",
    "wall_corner": "牆角", "wall_center": "牆面中央", "wall_edge": "牆緣",
    "wall_column_interface": "牆柱交界", "wall_opening_corner": "開口角隅",
    "member_full": "整支結構", "top": "頂部", "bottom": "底部", "unknown": "位置不明",
}
REBAR_STATE_ZH = {
    "not_exposed": "鋼筋未外露", "exposed_intact": "鋼筋外露但完好",
    "exposed_buckled": "主筋外露且挫屈", "exposed_ruptured": "鋼筋外露且斷裂",
    "exposed_fractured": "鋼筋外露且斷裂", "na": "", "unknown": "",
}
CONCRETE_STATE_ZH = {
    "intact": "混凝土完好", "cracked": "混凝土開裂",
    "cover_spalled": "保護層剝落", "spalled": "混凝土剝落",
    "crushed": "混凝土壓碎", "core_crushed": "核心混凝土碎裂", "na": "", "unknown": "",
}
WIDTH_QUAL_ZH = {
    "hairline": "髮絲級（<0.3mm）", "visible": "肉眼可見", "wide": "明顯偏寬",
    "very_wide": "非常寬", "na": "", "unknown": "寬度不明",
}
LOAD_CAP_ZH = {
    "intact": "承載能力未受影響", "possibly_compromised": "承載能力可能受損",
    "likely_compromised": "承載能力恐已受損", "lost": "承載能力喪失",
    "unknown": "承載狀態不明", "na": "",
}
GLOBAL_IND_ZH = {
    "overall_tilt": "建築整體傾斜", "story_offset_or_soft_story": "樓層側移／軟弱層",
    "story_collapse_or_slab_sagging": "樓層塌陷／樓板下垂",
    "foundation_separation_or_scour": "基礎分離／掏空",
    "adjacent_building_influence": "受鄰棟建築影響",
    "column_base_displacement": "柱腳位移", "ground_deformation": "地面變形",
    "differential_settlement": "不均勻沉陷",
    "drift_indirect_evidence": "樓層側向位移的間接跡象",
}
HAZARD_ZH = {
    "falling_object": "高處物件掉落危險", "toppling_object": "物件傾倒危險",
    "parapet_or_wall_falling": "女兒牆／外牆掉落危險",
    "glass_falling": "玻璃掉落危險", "gas_leak": "疑似瓦斯外洩",
    "electrical_hazard": "電線／電氣危險", "water_leak": "漏水",
}
MARKING_ZH = {
    "spray_paint": "現場噴漆標記", "paint_mark": "現場噴漆標記",
    "tape": "封鎖線／膠帶", "placard": "已張貼公告單",
    "chalk": "粉筆標記", "sticker": "貼紙標記", "none": "",
}
SIGNATURE_ZH = {
    "short_column_shear": "短柱剪力破壞徵兆", "soft_story": "軟弱層徵兆",
    "x_shape_crack": "X形交叉裂縫（往復剪力）", "joint_failure": "梁柱接頭破壞徵兆",
    "column_axial_failure": "柱軸壓破壞徵兆", "core_crushing": "核心混凝土碎裂",
    "rebar_buckling": "主筋挫屈",
}
IVGRADE_ZH = {
    "I": "I 級（輕微：髮絲裂縫等）",
    "II": "II 級（中度：明顯裂縫、粉刷層剝落）",
    "III": "III 級（明顯損壞：保護層剝落、鋼筋外露但尚完好）",
    "IV": "IV 級（嚴重：主筋挫屈或核心混凝土壓碎）",
    "V": "V 級（極嚴重：結構斷裂、承載力喪失）",
}
PATTERN_A_ZH = {"Cracks": "裂縫", "Spalling": "混凝土剝落", "Expose of rebar": "鋼筋外露"}
PATTERN_B_ZH = {
    "Diagonal": "斜向裂縫", "Diagonal_large": "大型連續斜向裂縫",
    "Horizontal": "水平裂縫", "Horizontal_large": "大型連續水平裂縫",
    "Vertical": "垂直裂縫", "Vertiacal_large": "大型連續垂直裂縫",
    "Web": "網狀裂縫", "Web_large": "大型網狀裂縫",
    "X-shape": "X形交叉裂縫", "spalling-like_cracks": "剝落狀細裂縫",
}
INFO_SUFF_ZH = {
    "sufficient": "充足", "limited": "有限", "insufficient": "不足",
    "sufficient_for_visible_scope": "就照片拍到的範圍而言算充足",
}
IMG_ISSUE_ZH = {
    "blur": "影像模糊", "motion_blur": "影像晃動模糊",
    "low_light": "光線不足", "overexposed": "過度曝光", "underexposed": "曝光不足",
    "low_resolution": "影像解析度偏低", "perspective_distortion": "拍攝角度造成畫面變形",
    "partial_view": "僅拍到局部", "occlusion": "有物體遮擋",
    "far_distance": "拍攝距離過遠", "reflection_glare": "反光眩光",
}

GRADE_HEADLINE_ZH = {
    "A": "A 級：嚴重受損（危險）",
    "B": "B 級：中度受損（需注意）",
    "C": "C 級：輕微或未見明顯損傷",
}
GRADE_MEANING_ZH = {
    "A": "AI 分析顯示，照片中的主要結構結構（如柱、梁、承重牆）可能已達嚴重受損程度，"
         "建築物的承載能力可能受到影響，具有較高危險性。",
    "B": "AI 分析顯示，建築物有中等程度的損傷，或存在物件墜落、傾倒等潛在危險；"
         "結構安全需要進一步確認。",
    "C": "AI 在照片中僅發現輕微損傷或未發現明顯的結構危險跡象。",
}
GRADE_ACTION_ZH = {
    "A": ["**請勿進入或停留在建築物內**，並與建築物保持安全距離。",
          "立即通報所在縣市政府或撥打 1999，申請專業技師到場評估。",
          "若建築物內仍有人員，請協助其儘速離開。"],
    "B": ["暫停使用該建築物（或受損區域），等待專業人員複檢。",
          "遠離外牆、女兒牆、招牌與窗戶等可能掉落物的下方。",
          "向所在縣市政府或管委會通報，安排專業技師評估。"],
    "C": ["目前未見明顯危險，但餘震後損傷可能擴大，請持續留意裂縫是否變寬、變長。",
          "若之後發現新的明顯裂縫、傾斜或掉落物，請重新拍照評估或通報專業技師。"],
}

_DISCLAIMER = ("本報告由 AI 系統自動產生，屬**第一階段快篩參考**，並非法定的建築物"
               "危險性判定，亦不取代專業技師之現場勘查。實際安全狀態請以主管機關與"
               "專業技師的正式評估為準。")


def _zh(value: Optional[str], table: Dict[str, str]) -> str:
    """Dictionary lookup with a safe prettified fallback (never crash/raw None).
    Used by the ENGINEER report, where seeing the raw enum is informative."""
    if value in (None, "", "na", "unknown", "indeterminate", "none"):
        return table.get(value or "", "")
    return table.get(value, str(value).replace("_", " "))


# every enum value that hit no dictionary entry during the last build_assessment;
# surfaced in the engineer report so the gap can be closed instead of shipping
# half-English text to a civilian reader.
_GAP_TABLES = "damage_type position member_type material role mechanism " \
              "rebar_state concrete_state width load_capacity global_indicator " \
              "hazard marking info_sufficiency image_issue signature".split()


def _zhp(value: Optional[str], table: Dict[str, str],
         kind: str = "", gaps: Optional[List[Tuple[str, str]]] = None) -> str:
    """STRICT lookup for the civilian report: an unmapped enum returns "" rather
    than leaking `plaster_spalling` / `low resolution` into Chinese prose. The
    miss is recorded in `gaps` so the engineer report can list it."""
    if value in (None, "", "na", "unknown", "none", "indeterminate"):
        return ""
    hit = table.get(value)
    if hit is None:
        if gaps is not None and (kind, str(value)) not in gaps:
            gaps.append((kind, str(value)))
        return ""
    return hit


def _pct(x: Optional[float], digits: int = 1) -> str:
    try:
        return f"{float(x) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


# ═══════════════════════════════════════════════════════════════════════════
# component-grade extraction (the cross-check inputs)
# ═══════════════════════════════════════════════════════════════════════════
def component_grades(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Pull every per-component A/B/C-comparable signal out of the bundle.
    Each entry: {"grade": A/B/C/None, "detail": str} — all rule-derived."""
    comps: Dict[str, Any] = {}
    cv = bundle.get("cv") or {}
    so = cv.get("structured_output") or {}
    derived = cv.get("derived") or {}
    kg = bundle.get("kg") or {}
    summ = kg.get("kg_summary") or {}
    gnn = bundle.get("gnn") or {}

    # 1. EDL (ViT evidential classifier) — only meaningful when CV ran
    if cv.get("ok") and so.get("damage level value") in pc.ABC:
        comps["edl"] = {
            "grade": so["damage level value"],
            "confidence": so.get("damage level confidence"),
            "uncertainty": so.get("damage level uncertainty"),
            "detail": (f"信心 {_pct(so.get('damage level confidence'))}，"
                       f"不確定度 {so.get('damage level uncertainty')}"),
        }
    # 2. CV rule table (YOLO patterns + crack types -> engineer's table)
    if derived.get("cv_rule_grade") in pc.ABC:
        comps["cv_rule"] = {"grade": derived["cv_rule_grade"],
                            "detail": "依裂縫型態／剝落／鋼筋外露之工程規則表"}
    # 3. VLM worst member (via the KG SAFETY grade = max(VLM, KG-rule) per member)
    worst_safety = None
    for m in summ.get("members", []) or []:
        g = m.get("grade_safety") or m.get("grade_effective")
        if g in pc.IV_RANK and (worst_safety is None
                                or pc.IV_RANK[g] > pc.IV_RANK[worst_safety]):
            worst_safety = g
    if worst_safety:
        comps["vlm_worst_member"] = {
            "grade": pc.iv_to_abc(worst_safety), "iv_grade": worst_safety,
            "detail": f"最嚴重結構為 {worst_safety} 級（已與知識規則交叉取嚴）"}
    # 4. KG rule engine (placard rules — reported WITHOUT placard vocabulary)
    placard = summ.get("placard_suggestion")
    if placard:
        comps["kg_rule"] = {"grade": pc.placard_to_abc(placard),
                            "raw_placard_internal": placard,
                            "detail": _kg_rule_phrase(placard)}
    # 5. GNN — the decision maker
    if gnn.get("ok"):
        comps["gnn"] = {"grade": gnn.get("grade"), "probs": gnn.get("probs"),
                        "detail": f"{gnn.get('arch','').upper()} 圖神經網路綜合判定"}
    return comps


def _kg_rule_phrase(placard: str) -> str:
    """Internal placard value -> neutral severity wording (NO placard words)."""
    head = (placard or "").split("_")[0]
    if "紅" in head:
        return "觸發『主要結構嚴重受損』規則條件（單張影像、部分觸發）"
    if "黃" in head:
        return "觸發『非結構物墜落／傾倒危險』規則條件"
    if "綠" in head:
        return "僅見輕～中度損傷，未觸發危險規則條件"
    return "影像資訊不足，規則無法明確判定"


def decide_final(bundle: Dict[str, Any], comps: Dict[str, Any]) -> Dict[str, Any]:
    """Headline grade policy (owner-approved):
      * normal path  : final = the GNN prediction (the trained decision maker);
      * fallback path: VLM/KG/GNN unavailable -> final = worst(EDL, CV rule)
        (取大不取小 — insufficient evidence must not clear a building);
      * conservative_grade = worst across ALL component signals; if it exceeds
        the headline, the report shows a prominent safety escalation note."""
    gnn = bundle.get("gnn") or {}
    if gnn.get("ok") and gnn.get("grade") in pc.ABC:
        final, basis = gnn["grade"], pc.BASIS_FULL
    else:
        cand = [comps.get("edl", {}).get("grade"), comps.get("cv_rule", {}).get("grade")]
        final = pc.worst_abc(cand)
        basis = pc.BASIS_CV_ONLY
        if final is None:                      # nothing at all succeeded
            final, basis = "B", pc.BASIS_CV_ONLY   # honest abstention ≠ safe -> B
    conservative = pc.worst_abc(
        [final] + [c.get("grade") for c in comps.values()]) or final
    return {"final_grade": final, "assessment_basis": basis,
            "conservative_grade": conservative,
            "escalation_needed": pc.SEVERITY_RANK[conservative] > pc.SEVERITY_RANK[final]}


# ═══════════════════════════════════════════════════════════════════════════
# per-component report-field extraction (all deterministic)
#   Every extractor returns BOTH the zh-TW strings and the raw enums, and
#   records unmapped enums into `gaps` (surfaced in the engineer report).
# ═══════════════════════════════════════════════════════════════════════════
def _extract_cv_fields(cv: Dict[str, Any],
                       gaps: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    if not cv or not cv.get("ok"):
        return {"available": False, "error": (cv or {}).get("error")}
    so = cv.get("structured_output") or {}
    d = cv.get("derived") or {}
    dets = []
    for det in so.get("detections", []) or []:
        if isinstance(det, dict):
            dets.append({k: det.get(k) for k in
                         ("index", "class_name", "confidence", "crack_type",
                          "area_px", "area_cm2", "cost", "box_xyxy")})
    return {
        "available": True,
        "patterns_a": [_zh(p, PATTERN_A_ZH) for p in so.get("damage pattern A", []) or []],
        "patterns_b": [_zh(p, PATTERN_B_ZH) for p in so.get("damage pattern B", []) or []],
        "patterns_a_raw": list(so.get("damage pattern A", []) or []),
        "patterns_b_raw": list(so.get("damage pattern B", []) or []),
        "crack_ratio": d.get("crack_area_ratio", 0.0),
        "spall_ratio": d.get("spalling_area_ratio", 0.0),
        "crack_bucket": d.get("crack_area_bucket", area_bucket(d.get("crack_area_ratio", 0.0))),
        "spall_bucket": d.get("spalling_area_bucket", area_bucket(d.get("spalling_area_ratio", 0.0))),
        "crack_bucket_zh": AREA_BUCKET_WORDS_ZH[area_bucket(d.get("crack_area_ratio", 0.0))],
        "spall_bucket_zh": AREA_BUCKET_WORDS_ZH[area_bucket(d.get("spalling_area_ratio", 0.0))],
        "crack_area_cm2": so.get("crack area cm2"),
        "spall_area_cm2": so.get("spalling area cm2"),
        "ratio_cm_per_px": so.get("ratio cm per px"),
        "image_size": so.get("size"),
        "edl_grade": so.get("damage level value"),
        "edl_confidence": so.get("damage level confidence"),
        "edl_uncertainty": so.get("damage level uncertainty"),
        "edl_probs": so.get("damage level probs"),
        "cv_rule_grade": d.get("cv_rule_grade"),
        "cv_rule_reasons": d.get("cv_rule_reasons") or [],
        "detections": dets,
        "detection_count": so.get("detection count", len(dets)),
        "cost_total": d.get("estimated_cost_total", 0),
        "cost_breakdown": {
            "rebar": int(so.get("estimated cost of rebar", 0) or 0),
            "crack": int(so.get("estimated cost of crack", 0) or 0),
            "spalling": int(so.get("estimated cost of spalling", 0) or 0),
        },
    }


_MAX_MEMBERS_SHOWN = 4
_MAX_FEATURES_PER_MEMBER = 3
_EVIDENCE_MAXLEN = 90


def _feature_line(f: Dict[str, Any]) -> str:
    """One damage feature -> one readable line (engineer-facing; keeps grades)."""
    parts: List[str] = []
    dt = _zh(f.get("damage_type"), DAMAGE_TYPE_ZH)
    pos = _zh(f.get("position_on_member"), POSITION_ZH)
    parts.append(f"{dt}" + (f"（{pos}）" if pos else ""))
    metrics = f.get("metrics") or {}
    if metrics.get("width_value"):
        parts.append(f"寬約 {metrics['width_value']}{metrics.get('width_unit') or 'mm'}")
    else:
        wq = _zh(metrics.get("width_qualitative"), WIDTH_QUAL_ZH)
        if wq:
            parts.append(wq)
    for key, tab in (("rebar_state", REBAR_STATE_ZH), ("concrete_state", CONCRETE_STATE_ZH)):
        z = _zh(f.get(key), tab)
        if z and f.get(key) not in ("not_exposed", "intact"):
            parts.append(z)
    mech = _zh(f.get("inferred_mechanism"), MECHANISM_ZH)
    if mech and f.get("inferred_mechanism") != "indeterminate":
        parts.append(f"研判為{mech}")
    g = f.get("feature_severity_grade")
    if g in pc.IV_RANK:
        parts.append(f"嚴重度 {g} 級")
    return "，".join(p for p in parts if p)


def _extract_vlm_fields(vlm: Dict[str, Any], kg_summary: Dict[str, Any],
                        gaps: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    """Members sorted worst-first by the KG SAFETY grade; the capped list feeds
    the old layout, `members_all` feeds the engineer report, and the strict
    zh fields (`*_pub`) feed the civilian paragraph."""
    if not vlm or not vlm.get("ok"):
        return {"available": False, "error": (vlm or {}).get("error")}
    rj = vlm.get("reasoning") or {}
    members_raw = [m for m in (rj.get("members") or []) if isinstance(m, dict)]
    kg_by_id = {m.get("member"): m for m in (kg_summary or {}).get("members", []) or []}

    def _rank(m):
        km = kg_by_id.get(m.get("id"), {})
        g = km.get("grade_safety") or km.get("grade_effective") or m.get("member_damage_grade")
        primary = 1 if m.get("structural_role") == "primary_load_bearing" else 0
        return (pc.IV_RANK.get(g or "", 0), primary)

    members_sorted = sorted(members_raw, key=_rank, reverse=True)
    members_all: List[Dict[str, Any]] = []
    for m in members_sorted:
        km = kg_by_id.get(m.get("id"), {})
        grade = (km.get("grade_safety") or km.get("grade_effective")
                 or m.get("member_damage_grade"))
        feats = sorted([f for f in (m.get("damage_features") or []) if isinstance(f, dict)],
                       key=lambda f: pc.IV_RANK.get(f.get("feature_severity_grade") or "", 0),
                       reverse=True)
        evidence_full = (m.get("member_reasoning") or "").strip().replace("\n", " ")
        evidence = (evidence_full[:_EVIDENCE_MAXLEN] + "…"
                    if len(evidence_full) > _EVIDENCE_MAXLEN else evidence_full)
        members_all.append({
            "id": m.get("id"),
            "member_type": m.get("member_type"),
            "structural_role": m.get("structural_role"),
            "type_zh": _zh(m.get("member_type"), MEMBER_TYPE_ZH),
            "type_zh_pub": _zhp(m.get("member_type"), MEMBER_TYPE_ZH, "member_type", gaps),
            "material_zh": _zh(m.get("material"), MATERIAL_ZH),
            "role_zh": _zh(m.get("structural_role"), ROLE_ZH),
            "grade": grade, "grade_zh": IVGRADE_ZH.get(grade or "", ""),
            "grade_vlm_raw": m.get("member_damage_grade"),
            "grade_kg_effective": km.get("grade_effective"),
            "grade_kg": km.get("grade_kg"),
            "grade_was_escalated": bool(km.get("undergrade")),
            "kg_conflict": km.get("conflict"),
            "load_capacity_status": m.get("load_capacity_status"),
            "load_zh": _zh(m.get("load_capacity_status"), LOAD_CAP_ZH),
            "feature_lines": [_feature_line(f) for f in feats[:_MAX_FEATURES_PER_MEMBER]],
            "feature_lines_all": [_feature_line(f) for f in feats],
            "features_raw": [{
                "damage_type": f.get("damage_type"),
                "position": f.get("position_on_member"),
                "severity": f.get("feature_severity_grade"),
                "mechanism": f.get("inferred_mechanism"),
                "rebar_state": f.get("rebar_state"),
                "concrete_state": f.get("concrete_state"),
                "width_qualitative": (f.get("metrics") or {}).get("width_qualitative"),
                "width_value": (f.get("metrics") or {}).get("width_value"),
                "width_unit": (f.get("metrics") or {}).get("width_unit"),
                "length_qualitative": (f.get("metrics") or {}).get("length_qualitative"),
                "area_extent": (f.get("metrics") or {}).get("area_extent"),
                # strict zh for the civilian paragraph (empty when unmapped)
                "damage_zh_pub": _zhp(f.get("damage_type"), DAMAGE_TYPE_ZH, "damage_type", gaps),
                "position_zh_pub": _zhp(f.get("position_on_member"), POSITION_ZH, "position", gaps),
            } for f in feats],
            "n_features_hidden": max(0, len(feats) - _MAX_FEATURES_PER_MEMBER),
            "evidence": evidence,
            "evidence_full": evidence_full,
        })
    members_out = members_all[:_MAX_MEMBERS_SHOWN]

    gi_raw = [k for k, v in (rj.get("global_indicators") or {}).items()
              if isinstance(v, dict) and v.get("observed")]
    hz_raw = [h.get("type") or h.get("hazard_type")
              for h in rj.get("secondary_hazards") or [] if isinstance(h, dict)]
    mk_raw = [h.get("type") or h.get("marking_type")
              for h in rj.get("human_markings") or [] if isinstance(h, dict)]
    summary = rj.get("assessment_inputs_summary") or {}
    quality = (rj.get("image_context") or {}).get("image_quality") or {}
    issues = quality.get("issues") or []
    return {
        "available": True,
        "members": members_out,
        "members_all": members_all,
        "n_members_total": len(members_raw),
        "global_indicators": [z for z in (_zh(k, GLOBAL_IND_ZH) for k in gi_raw) if z],
        "hazards": [z for z in (_zh(h, HAZARD_ZH) for h in hz_raw) if z],
        "markings": [z for z in (_zh(k, MARKING_ZH) for k in mk_raw) if z],
        "global_indicators_raw": gi_raw,
        "hazards_raw": hz_raw,
        "markings_raw": mk_raw,
        "global_indicators_pub": [z for z in
                                  (_zhp(k, GLOBAL_IND_ZH, "global_indicator", gaps)
                                   for k in gi_raw) if z],
        "hazards_pub": [z for z in (_zhp(h, HAZARD_ZH, "hazard", gaps) for h in hz_raw) if z],
        "markings_pub": [z for z in (_zhp(k, MARKING_ZH, "marking", gaps) for k in mk_raw) if z],
        "info_sufficiency_raw": summary.get("information_sufficiency"),
        "info_sufficiency_zh": _zh(summary.get("information_sufficiency"), INFO_SUFF_ZH),
        "info_sufficiency_pub": _zhp(summary.get("information_sufficiency"),
                                     INFO_SUFF_ZH, "info_sufficiency", gaps),
        "image_issues_raw": list(issues),
        "image_issues_zh": [_zh(i, IMG_ISSUE_ZH) for i in issues],
        "image_issues_pub": [z for z in (_zhp(i, IMG_ISSUE_ZH, "image_issue", gaps)
                                         for i in issues) if z],
        "occlusion": quality.get("occlusion_present"),
    }


def _rule_activation_line(r: Dict[str, Any]) -> str:
    """SeverityRule partial-trigger -> severity-accurate zh line."""
    m, g = r.get("member"), r.get("safety_grade")
    sev = str(r.get("severity") or "")
    if sev.startswith("乙"):
        return (f"結構 {m}：損傷達 **{g} 級**，符合『主要承重結構中度以上受損』"
                f"評估條件（單張影像僅部分觸發；正式判定需整層統計）")
    return (f"結構 {m}：損傷 {g} 級，屬『輕微受損』評估條件"
            f"（單張影像、部分觸發）")


def _extract_kg_fields(kg: Dict[str, Any],
                       gaps: Optional[List[Tuple[str, str]]] = None) -> Dict[str, Any]:
    if not kg or not kg.get("ok"):
        return {"available": False, "error": (kg or {}).get("error")}
    summ = kg.get("kg_summary") or {}
    val = kg.get("validation") or {}
    n_feat = summ.get("n_features", 0)
    n_conf = summ.get("n_grade_conflict", 0)
    n_under = summ.get("n_undergrade", 0)
    sigs = [s for s in summ.get("critical_signatures", []) or []
            if isinstance(s, dict) and s.get("corroborated")]
    rules = summ.get("rule_activations", []) or []
    return {
        "available": True,
        "verdict": kg.get("verdict"),
        "n_features": n_feat, "n_agree": max(0, n_feat - n_conf),
        "n_conflict": n_conf, "n_undergrade": n_under,
        "n_indeterminate": summ.get("n_indeterminate_kg"),
        "signatures_zh": [_zh(s.get("signature"), SIGNATURE_ZH) for s in sigs],
        "signatures_raw": [s.get("signature") for s in sigs],
        "signatures_pub": [z for z in (_zhp(s.get("signature"), SIGNATURE_ZH, "signature", gaps)
                                       for s in sigs) if z],
        "signatures_all_raw": [s for s in summ.get("critical_signatures", []) or []
                               if isinstance(s, dict)],
        "rule_lines": [_rule_activation_line(r) for r in rules],
        "rule_activations_raw": rules,
        "member_rows": summ.get("members") or [],
        "kg_rule_phrase": _kg_rule_phrase(summ.get("placard_suggestion") or ""),
        "placard_internal": summ.get("placard_suggestion"),
        "placard_basis": summ.get("placard_suggestion_basis"),
        "n_validation_warn": len(val.get("consistency_flags") or []),
        "validation_flags": list(val.get("consistency_flags") or []),
        "schema_valid": val.get("schema_valid"),
        "schema_errors": list(val.get("schema_errors") or []),
        "enum_violations": list(val.get("enum_violations") or []),
        "id_violations": list(val.get("id_violations") or []),
        "coverage": val.get("coverage"),
        "grade_crosscheck_summary": val.get("grade_crosscheck_summary"),
        "conversion_flags": list(kg.get("conversion_flags") or []),
    }


# ═══════════════════════════════════════════════════════════════════════════
# assessment.json assembly
# ═══════════════════════════════════════════════════════════════════════════
def build_assessment(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Combine every stage output into the single machine-readable result.
    `bundle` keys: run_id, image_path, element, gnn_input, gnn_arch,
                   cv, vlm, kg, gnn (each a stage-output dict or None)."""
    comps = component_grades(bundle)
    final = decide_final(bundle, comps)
    kg = bundle.get("kg") or {}
    gnn = bundle.get("gnn") or {}
    gaps: List[Tuple[str, str]] = []
    cv_f = _extract_cv_fields(bundle.get("cv") or {}, gaps)
    vlm_f = _extract_vlm_fields(bundle.get("vlm") or {}, (kg.get("kg_summary") or {}), gaps)
    kg_f = _extract_kg_fields(kg, gaps)
    return {
        "pipeline_version": pc.PIPELINE_VERSION,
        "run_id": bundle.get("run_id"),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image_path": bundle.get("image_path"),
        "element": bundle.get("element"),
        "element_zh": _zh(bundle.get("element"), ELEMENT_ZH),
        # ---- the answer -------------------------------------------------- #
        "final_grade": final["final_grade"],
        "assessment_basis": final["assessment_basis"],
        "conservative_grade": final["conservative_grade"],
        "escalation_needed": final["escalation_needed"],
        "gnn_probs": gnn.get("probs"),
        # ---- per-component signals (engineer-facing) ---------------------- #
        "component_grades": comps,
        "gnn_detail": {"grade": gnn.get("grade"), "probs": gnn.get("probs"),
                       "logits": gnn.get("logits"), "n_nodes": gnn.get("n_nodes"),
                       "n_edges": gnn.get("n_edges"), "model": gnn.get("model"),
                       "gnn_input": gnn.get("gnn_input"), "arch": gnn.get("arch")},
        "config": {"gnn_input": bundle.get("gnn_input"),
                   "gnn_arch": bundle.get("gnn_arch"),
                   "gnn_model": gnn.get("model"),
                   "vlm": {k: (bundle.get("vlm") or {}).get(k)
                           for k in ("profile", "few_shot", "element_hint", "backend")}},
        # ---- extracted report fields ------------------------------------- #
        "cv_findings": cv_f,
        "vlm_findings": vlm_f,
        "kg_findings": kg_f,
        "stage_status": {
            s: {"ok": bool((bundle.get(s) or {}).get("ok")),
                "seconds": (bundle.get(s) or {}).get("seconds"),
                "error": (bundle.get(s) or {}).get("error")}
            for s in ("cv", "vlm", "kg", "gnn")},
        # raw internals engineers may need (never rendered for civilians)
        "internal": {"kg_placard_suggestion": (kg.get("kg_summary") or {}).get("placard_suggestion"),
                     "kg_placard_basis": (kg.get("kg_summary") or {}).get("placard_suggestion_basis"),
                     "validator_verdict": kg.get("verdict"),
                     "vocab_gaps": sorted(set(gaps))},
    }


# ═══════════════════════════════════════════════════════════════════════════
# report rendering
# ---------------------------------------------------------------------------
#   render_engineer_report(a) -> engineer/developer view: every module's raw
#       output, timings, internals and vocabulary gaps. No advice, no caveats.
#   render_public_report(a)   -> civilian view: exactly three sections
#       (1) result + actions, (2) ONE integrated paragraph, (3) limits.
#   render_report(a, audience) -> back-compatible dispatcher (default: public).
# ═══════════════════════════════════════════════════════════════════════════
COMPONENT_NAME_ZH = {
    "edl": "影像分類模型（EDL）", "cv_rule": "影像規則比對",
    "vlm_worst_member": "AI 視覺觀察（最嚴重結構）",
    "kg_rule": "知識圖譜規則檢核", "gnn": "圖神經網路（最終判定）",
}


def _fmt(x: Any, dash: str = "—") -> str:
    """None/empty -> em dash; lists -> joined (tables never render 'None')."""
    if x is None or x == "" or x == [] or x == {}:
        return dash
    if isinstance(x, (list, tuple)):
        if all(isinstance(i, (int, float)) and not isinstance(i, bool) for i in x):
            return "[" + ", ".join(f"{i:g}" for i in x) + "]"
        return "、".join(str(i) for i in x)
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


def _num(x: Any, digits: int = 4) -> str:
    try:
        return f"{float(x):.{digits}g}"
    except (TypeError, ValueError):
        return "—"


# ═══════════════════════════════════════════════════════════════════════════
# [1] ENGINEER / DEVELOPER REPORT
# ═══════════════════════════════════════════════════════════════════════════
def render_engineer_report(a: Dict[str, Any]) -> str:
    """Full-detail technical dump of every stage. Deterministic, no advice."""
    L: List[str] = []
    add = L.append
    cv, vlm, kgf = (a.get("cv_findings") or {}, a.get("vlm_findings") or {},
                    a.get("kg_findings") or {})
    cfg = a.get("config") or {}
    comps = a.get("component_grades") or {}
    internal = a.get("internal") or {}
    gd = a.get("gnn_detail") or {}
    model = gd.get("model") if isinstance(gd.get("model"), dict) else {}

    # ---------- 0. run metadata ---------- #
    add("# 震後損傷快篩｜工程／開發版報告（engineer view）")
    add("")
    add("## 0. Run metadata")
    add("")
    add("| 欄位 | 值 |")
    add("|---|---|")
    add(f"| run_id | `{_fmt(a.get('run_id'))}` |")
    add(f"| image_path | `{_fmt(a.get('image_path'))}` |")
    add(f"| element (user-specified) | `{_fmt(a.get('element'))}` / {_fmt(a.get('element_zh'))} |")
    add(f"| created_at | {_fmt(a.get('created_at'))} |")
    add(f"| pipeline_version | `{_fmt(a.get('pipeline_version'))}` |")
    add(f"| gnn_input / arch | `{_fmt(cfg.get('gnn_input'))}` / `{_fmt(cfg.get('gnn_arch'))}` |")
    if model:
        add(f"| gnn checkpoint | `{_fmt(model.get('checkpoint'))}` |")
        add(f"| gnn variant | `{_fmt(model.get('variant'))}` |")
        add(f"| gnn in_dim / feature_spec | `{_fmt(model.get('in_dim'))}` / "
            f"`{_fmt(model.get('feature_spec_version'))}` |")
        add(f"| gnn trained_on_rows / epochs | `{_fmt(model.get('trained_on_rows'))}` / "
            f"`{_fmt(model.get('trained_epochs'))}` |")
    else:
        add(f"| gnn_model | `{_fmt(cfg.get('gnn_model'))}` |")
    vcfg = cfg.get("vlm") or {}
    add(f"| vlm profile / backend | `{_fmt(vcfg.get('profile'))}` / `{_fmt(vcfg.get('backend'))}` |")
    add(f"| vlm few_shot / element_hint | `{_fmt(vcfg.get('few_shot'))}` / "
        f"`{_fmt(vcfg.get('element_hint'))}` |")
    add(f"| total_seconds | {_fmt(a.get('total_seconds'))} |")
    add("")

    # ---------- 1. decision ---------- #
    add("## 1. Decision")
    add("")
    add(f"- **final_grade** = `{_fmt(a.get('final_grade'))}`")
    add(f"- **assessment_basis** = `{_fmt(a.get('assessment_basis'))}`")
    add(f"- **conservative_grade** = `{_fmt(a.get('conservative_grade'))}`  "
        f"(escalation_needed = `{_fmt(a.get('escalation_needed'))}`)")
    probs = a.get("gnn_probs")
    if probs:
        add(f"- **gnn_probs** = A {_pct(probs.get('A'), 2)} / "
            f"B {_pct(probs.get('B'), 2)} / C {_pct(probs.get('C'), 2)}")
    add("")
    add("### 1.1 Component signals")
    add("")
    add("| component | grade | 補充 | raw |")
    add("|---|---|---|---|")
    for key in ("edl", "cv_rule", "vlm_worst_member", "kg_rule", "gnn"):
        c = comps.get(key)
        if not c:
            add(f"| {COMPONENT_NAME_ZH[key]} | — | 未產生 | — |")
            continue
        raw_bits: List[str] = []
        if key == "edl":
            raw_bits = [f"confidence={_fmt(c.get('confidence'))}",
                        f"uncertainty={_fmt(c.get('uncertainty'))}",
                        f"probs={_fmt(cv.get('edl_probs'))}"]
        elif key == "cv_rule":
            raw_bits = [f"reasons={_fmt([r.get('signal') for r in cv.get('cv_rule_reasons') or []])}"]
        elif key == "vlm_worst_member":
            raw_bits = [f"iv_grade={_fmt(c.get('iv_grade'))}"]
        elif key == "kg_rule":
            raw_bits = [f"placard_internal={_fmt(c.get('raw_placard_internal'))}"]
        elif key == "gnn":
            raw_bits = [f"probs={_fmt(c.get('probs'))}"]
        add(f"| {COMPONENT_NAME_ZH[key]} | `{_fmt(c.get('grade'))}` | "
            f"{_fmt(c.get('detail'), '')} | `{'; '.join(raw_bits)}` |")
    add("")
    grades_seen = {c.get("grade") for c in comps.values() if c.get("grade") in pc.ABC}
    add(f"- component agreement: {'CONSISTENT' if len(grades_seen) <= 1 else 'SPLIT'} "
        f"({_fmt(sorted(grades_seen))})")
    add("")

    # ---------- 2. stage status ---------- #
    add("## 2. Stage status")
    add("")
    add("| stage | ok | seconds | error |")
    add("|---|---|---|---|")
    for s in ("cv", "vlm", "kg", "gnn"):
        st = (a.get("stage_status") or {}).get(s, {}) or {}
        add(f"| {s} | `{_fmt(st.get('ok'))}` | {_fmt(st.get('seconds'))} | "
            f"{_fmt(st.get('error'))} |")
    add("")

    # ---------- 3. CV branch ---------- #
    add("## 3. CV branch (YOLO / crack-ViT / U-Net / cost / EDL)")
    add("")
    if cv.get("available"):
        add(f"- `damage pattern A` = `{_fmt(cv.get('patterns_a_raw'))}` → {_fmt(cv.get('patterns_a'))}")
        add(f"- `damage pattern B` = `{_fmt(cv.get('patterns_b_raw'))}` → {_fmt(cv.get('patterns_b'))}")
        add(f"- `crack_area_ratio` = {_num(cv.get('crack_ratio'))} "
            f"({_pct(cv.get('crack_ratio'), 3)}, bucket={_fmt(cv.get('crack_bucket'))}"
            f"/{_fmt(cv.get('crack_bucket_zh'))}, {_fmt(cv.get('crack_area_cm2'))} cm²)")
        add(f"- `spalling_area_ratio` = {_num(cv.get('spall_ratio'))} "
            f"({_pct(cv.get('spall_ratio'), 3)}, bucket={_fmt(cv.get('spall_bucket'))}"
            f"/{_fmt(cv.get('spall_bucket_zh'))}, {_fmt(cv.get('spall_area_cm2'))} cm²)")
        add(f"- `size` = {_fmt(cv.get('image_size'))}；`ratio cm per px` = "
            f"{_fmt(cv.get('ratio_cm_per_px'))}")
        add(f"- `EDL` grade = `{_fmt(cv.get('edl_grade'))}`, "
            f"confidence = {_pct(cv.get('edl_confidence'), 2)}, "
            f"uncertainty = {_fmt(cv.get('edl_uncertainty'))}")
        ep = cv.get("edl_probs") or {}
        if ep:
            add(f"  - `damage level probs` = A {_pct(ep.get('A'), 2)} / "
                f"B {_pct(ep.get('B'), 2)} / C {_pct(ep.get('C'), 2)}")
        add(f"- `cv_rule_grade` = `{_fmt(cv.get('cv_rule_grade'))}`")
        for r in cv.get("cv_rule_reasons") or []:
            add(f"  - `{_fmt(r.get('signal'))}` → `{_fmt(r.get('grade'))}`")
        b = cv.get("cost_breakdown") or {}
        add(f"- `estimated_cost_total` = NT$ {int(cv.get('cost_total') or 0):,} "
            f"(rebar {int(b.get('rebar', 0)):,} / crack {int(b.get('crack', 0)):,} / "
            f"spalling {int(b.get('spalling', 0)):,})")
        dets = cv.get("detections") or []
        if dets:
            add(f"- detections ({_fmt(cv.get('detection_count'))}):")
            add("")
            add("  | # | class | conf | crack_type | area_px | area_cm² | cost | box |")
            add("  |---|---|---|---|---|---|---|---|")
            for d in dets:
                add(f"  | {_fmt(d.get('index'))} | `{_fmt(d.get('class_name'))}` | "
                    f"{_fmt(d.get('confidence'))} | `{_fmt(d.get('crack_type'))}` | "
                    f"{_fmt(d.get('area_px'))} | {_fmt(d.get('area_cm2'))} | "
                    f"{_fmt(d.get('cost'))} | `{_fmt(d.get('box_xyxy'))}` |")
            add("")
    else:
        add(f"- **FAILED / unavailable** — `{_fmt(cv.get('error'))}`")
    add("")

    # ---------- 4. VLM branch ---------- #
    add("## 4. VLM branch (two-pass extraction → reasoning)")
    add("")
    if vlm.get("available"):
        members = vlm.get("members_all") or vlm.get("members") or []
        add(f"- members detected: **{_fmt(vlm.get('n_members_total'))}** "
            f"(all listed below, worst-first by KG safety grade)")
        add(f"- `information_sufficiency` = `{_fmt(vlm.get('info_sufficiency_raw'))}` → "
            f"{_fmt(vlm.get('info_sufficiency_zh'))}")
        add(f"- `image_quality.issues` = `{_fmt(vlm.get('image_issues_raw'))}` → "
            f"{_fmt(vlm.get('image_issues_zh'))}；`occlusion_present` = "
            f"`{_fmt(vlm.get('occlusion'))}`")
        add(f"- `global_indicators` (observed) = `{_fmt(vlm.get('global_indicators_raw'))}` → "
            f"{_fmt(vlm.get('global_indicators'))}")
        add(f"- `secondary_hazards` = `{_fmt(vlm.get('hazards_raw'))}` → {_fmt(vlm.get('hazards'))}")
        add(f"- `human_markings` = `{_fmt(vlm.get('markings_raw'))}` → {_fmt(vlm.get('markings'))}")
        add("")
        for idx, m in enumerate(members, 1):
            add(f"### 4.{idx} member `{_fmt(m.get('id'))}` — "
                f"{_fmt(m.get('material_zh'), '')}{_fmt(m.get('type_zh'), '')}"
                f"（`{_fmt(m.get('member_type'))}` / `{_fmt(m.get('structural_role'))}`）")
            add("")
            add(f"- grade: VLM `{_fmt(m.get('grade_vlm_raw'))}` → "
                f"KG `{_fmt(m.get('grade_kg'))}` → effective `{_fmt(m.get('grade_kg_effective'))}` "
                f"→ **safety `{_fmt(m.get('grade'))}`**"
                + ("  ⟵ **undergrade corrected**" if m.get("grade_was_escalated") else "")
                + (f"  (conflict=`{_fmt(m.get('kg_conflict'))}`)"
                   if m.get("kg_conflict") else ""))
            add(f"- `load_capacity_status` = `{_fmt(m.get('load_capacity_status'))}`"
                + (f"（{m['load_zh']}）" if m.get("load_zh") else ""))
            feats = m.get("features_raw") or []
            if feats:
                add(f"- damage_features ({len(feats)}):")
                add("")
                add("  | # | damage_type | position | sev | mechanism | rebar | concrete "
                    "| width | length | extent |")
                add("  |---|---|---|---|---|---|---|---|---|---|")
                for i, f in enumerate(feats, 1):
                    w = (f"{f.get('width_value')}{f.get('width_unit') or ''}"
                         if f.get("width_value") else f.get("width_qualitative"))
                    add(f"  | {i} | `{_fmt(f.get('damage_type'))}` | `{_fmt(f.get('position'))}` | "
                        f"`{_fmt(f.get('severity'))}` | `{_fmt(f.get('mechanism'))}` | "
                        f"`{_fmt(f.get('rebar_state'))}` | `{_fmt(f.get('concrete_state'))}` | "
                        f"`{_fmt(w)}` | `{_fmt(f.get('length_qualitative'))}` | "
                        f"`{_fmt(f.get('area_extent'))}` |")
                add("")
            else:
                add("- damage_features: —")
            if m.get("evidence_full"):
                add(f"- `member_reasoning`: {m['evidence_full']}")
            add("")
    else:
        add(f"- **FAILED / unavailable** — `{_fmt(vlm.get('error'))}`")
        add("")

    # ---------- 5. KG ---------- #
    add("## 5. Validator + KG grounding")
    add("")
    if kgf.get("available"):
        add(f"- `validator_verdict` = `{_fmt(kgf.get('verdict'))}`；"
            f"`schema_valid` = `{_fmt(kgf.get('schema_valid'))}`；"
            f"consistency_flags = {_fmt(kgf.get('n_validation_warn'), '0')}")
        for fl in (kgf.get("validation_flags") or [])[:20]:
            if isinstance(fl, dict):
                add(f"  - [`{_fmt(fl.get('severity'))}`] `{_fmt(fl.get('rule'))}` "
                    f"@ `{_fmt(fl.get('target'))}` — {_fmt(fl.get('detail'), '')}")
            else:
                add(f"  - `{_fmt(fl)}`")
        for k in ("schema_errors", "enum_violations", "id_violations", "conversion_flags"):
            if kgf.get(k):
                add(f"- `{k}` = `{_fmt(kgf.get(k))}`")
        if kgf.get("coverage"):
            add(f"- `coverage` = `{_fmt(kgf.get('coverage'))}`")
        if kgf.get("grade_crosscheck_summary"):
            add(f"- `grade_crosscheck_summary` = `{_fmt(kgf.get('grade_crosscheck_summary'))}`")
        add(f"- features checked = {_fmt(kgf.get('n_features'), '0')}；"
            f"agree = {_fmt(kgf.get('n_agree'), '0')}；"
            f"conflict = {_fmt(kgf.get('n_conflict'), '0')}；"
            f"undergrade = {_fmt(kgf.get('n_undergrade'), '0')}；"
            f"indeterminate = {_fmt(kgf.get('n_indeterminate'), '0')}")
        rows = kgf.get("member_rows") or []
        if rows:
            add("")
            add("  | member | type | role | VLM | KG | effective | safety | conflict | undergrade |")
            add("  |---|---|---|---|---|---|---|---|---|")
            for r in rows:
                add(f"  | `{_fmt(r.get('member'))}` | `{_fmt(r.get('member_type'))}` | "
                    f"`{_fmt(r.get('structural_role'))}` | `{_fmt(r.get('grade_vlm'))}` | "
                    f"`{_fmt(r.get('grade_kg'))}` | `{_fmt(r.get('grade_effective'))}` | "
                    f"**`{_fmt(r.get('grade_safety'))}`** | `{_fmt(r.get('conflict'))}` | "
                    f"`{_fmt(r.get('undergrade'))}` |")
            add("")
        add(f"- corroborated signatures = `{_fmt(kgf.get('signatures_raw'))}` → "
            f"{_fmt(kgf.get('signatures_zh'))}")
        for s in (kgf.get("signatures_all_raw") or []):
            add(f"  - `{_fmt(s.get('signature'))}` corroborated=`{_fmt(s.get('corroborated'))}`")
        for r in (kgf.get("rule_activations_raw") or []):
            add(f"- rule `{_fmt(r.get('rule_id'))}`: member=`{_fmt(r.get('member'))}` "
                f"severity=`{_fmt(r.get('severity'))}` partial=`{_fmt(r.get('partial'))}` "
                f"safety_grade=`{_fmt(r.get('safety_grade'))}`")
        add(f"- `placard_suggestion` (internal, never shown to civilians) = "
            f"`{_fmt(kgf.get('placard_internal') or internal.get('kg_placard_suggestion'))}`")
        add(f"- `placard_suggestion_basis` = "
            f"`{_fmt(kgf.get('placard_basis') or internal.get('kg_placard_basis'))}`")
    else:
        add(f"- **FAILED / unavailable** — `{_fmt(kgf.get('error'))}`")
    add("")

    # ---------- 6. GNN ---------- #
    add("## 6. GNN")
    add("")
    if (comps.get("gnn") or gd.get("grade")):
        add(f"- grade = `{_fmt(gd.get('grade'))}`；probs = `{_fmt(gd.get('probs'))}`")
        add(f"- logits = `{_fmt(gd.get('logits'))}`")
        add(f"- graph size: n_nodes = `{_fmt(gd.get('n_nodes'))}`, "
            f"n_edges = `{_fmt(gd.get('n_edges'))}`")
        add(f"- input mode = `{_fmt(gd.get('gnn_input') or cfg.get('gnn_input'))}`；"
            f"arch = `{_fmt(gd.get('arch') or cfg.get('gnn_arch'))}`；"
            f"checkpoint = `{_fmt(model.get('checkpoint') if model else cfg.get('gnn_model'))}`")
    else:
        st = (a.get("stage_status") or {}).get("gnn", {}) or {}
        add(f"- **not available** — `{_fmt(st.get('error'))}`；"
            f"headline fell back to `{_fmt(a.get('assessment_basis'))}`")
    add("")

    # ---------- 7. vocabulary gaps ---------- #
    add("## 7. Vocabulary gaps (unmapped enums)")
    add("")
    gapv = internal.get("vocab_gaps") or []
    if gapv:
        add("These raw values had no zh-TW dictionary entry. They are SUPPRESSED in the "
            "civilian report (never printed raw); add them to the tables in "
            "`report_builder.py` to make them visible again.")
        add("")
        add("| kind | raw value |")
        add("|---|---|")
        for kind, val in gapv:
            add(f"| `{_fmt(kind)}` | `{_fmt(val)}` |")
    else:
        add("- none — every enum in this run mapped to a zh-TW term.")
    add("")
    add("---")
    add("*engineer view — deterministic dump of `assessment.json`; "
        "no advisory or caveat text by design.*")
    add("")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# [2] PUBLIC / CIVILIAN REPORT
# ═══════════════════════════════════════════════════════════════════════════
PUBLIC_RESULT_SENTENCE_ZH = {
    "A": "從這張照片來看，建築物用來支撐重量的主要結構，可能已經出現相當嚴重的損傷跡象，"
         "安全性有疑慮。在專業技師到場確認之前，請先當作不安全來處理。",
    "B": "從這張照片來看，建築物有中等程度的損傷跡象，安全性還需要進一步確認後才能放心使用。",
    "C": "從這張照片來看，目前只發現輕微的損傷跡象，沒有看到明顯的結構危險徵兆；"
         "但這只是單張照片的初步判讀，餘震後仍請持續留意。",
}
# variant used when the main risk is falling / toppling objects rather than the frame
PUBLIC_RESULT_SENTENCE_B_HAZARD = (
    "從這張照片來看，建築物有中等程度的損傷跡象，而且有可能掉落或傾倒的物件，"
    "安全性還需要進一步確認後才能放心使用。")

_RULE_INTRO = "會得到這個結果，是因為判讀的原則是這樣訂的："
# the rule text is chosen by WHAT ACTUALLY DROVE the grade, so the report never
# describes a rule that the findings do not match
PUBLIC_GRADE_RULE_ZH = {
    "A_primary": "只要照片中支撐重量的主要結構（柱、梁、承重牆）出現達到嚴重程度的跡象"
                 "——像是混凝土被壓碎、鋼筋露出來或彎折、又寬又長的斜向或交叉裂縫、"
                 "明顯傾斜或回不去的變形——就會歸到最需要警戒的這一級。",
    "A_secondary": "只要照片中的牆體或其他結構出現達到嚴重程度的破壞跡象——像是又寬又長的"
                   "斜向或交叉裂縫、混凝土壓碎、粉刷層大面積剝落——即使那一處不是直接支撐"
                   "重量的主要結構，也會歸到最需要警戒的這一級，因為這同時反映了整棟建築"
                   "在地震中承受到的力量，旁邊看不到的部分也可能受到影響。",
    "B_struct": "如果損傷看起來停留在中等程度——裂縫明顯但還沒有整個貫穿、只有表層或"
                "保護層剝落——就會歸到需要留意、暫停使用的這一級。",
    "B_hazard": "如果主要的風險來自可能掉落、傾倒的物件，或建築整體出現需要確認的跡象，"
                "就會歸到需要留意、暫停使用的這一級。",
    "C": "照片中只看到輕微、偏表面的跡象，而且沒有出現混凝土壓碎、鋼筋外露、明顯傾斜"
         "這一類比較嚴重的情況，就會歸到目前未見明顯危險的這一級。",
}
PUBLIC_RULE_TAIL_ZH = {
    "match": "前面提到的情況大致落在這個範圍裡，所以得到現在這個結果。",
    "weak": "單看前面任何一項，都很難決定嚴重程度；不過如果把照片裡的所有跡象"
            "合在一起、再考慮拍不到的部分之後，判讀結果便會落在這一級，"
            "屬於比較保守的處理方式。",
    "escalated": "不過前面提到的情況中，有一部分看起來已經接近更嚴重的那一級，"
                 "所以雖然綜合判讀落在這一級，實際行動仍請按更嚴重的等級來準備。",
    "none": "照片中沒有找到足以支持更嚴重判定的跡象，所以歸在這一級；"
            "但這也可能是因為這張照片能看到的範圍有限。",
}

_PUBLIC_DAMAGE_ABC = {
    "rebar_buckling": "A", "rebar_fracture": "A", "stirrup_opening_or_fracture": "A",
    "core_crushing": "A", "concrete_crushing": "A", "member_fracture": "A",
    "residual_deformation": "A", "tilt_or_lean": "A", "support_displacement": "A",
    "x_shape_crack": "A", "through_crack": "A", "joint_damage": "A", "rebar_exposed": "A",
    "shear_crack": "B", "diagonal_crack": "B", "diagonal_crack_45": "B", "web_crack": "B",
    "cover_spalling": "B", "concrete_spalling": "B", "surface_spalling": "B",
    "vertical_crack": "B", "horizontal_crack": "B", "flexural_crack": "B",
    "interface_crack": "B",
    "plaster_spalling": "C", "render_spalling": "C",
    "hairline_crack": "C", "finish_spalling": "C", "tile_falling": "C",
    "window_or_glass_damage": "C",
}
_PUBLIC_CONSEQ_GROUP = {
    "core_crushing": "axial", "concrete_crushing": "axial", "rebar_buckling": "axial",
    "rebar_fracture": "axial", "member_fracture": "axial",
    "stirrup_opening_or_fracture": "axial",
    "x_shape_crack": "shear", "shear_crack": "shear", "diagonal_crack": "shear",
    "diagonal_crack_45": "shear", "web_crack": "shear", "through_crack": "shear",
    "joint_damage": "shear",
    "cover_spalling": "cover", "concrete_spalling": "cover", "surface_spalling": "cover",
    "rebar_exposed": "cover",
    "residual_deformation": "deform", "tilt_or_lean": "deform",
    "support_displacement": "deform",
    "vertical_crack": "crack", "horizontal_crack": "crack", "flexural_crack": "crack",
    "interface_crack": "crack",
    "plaster_spalling": "surface", "render_spalling": "surface",
    "finish_spalling": "surface", "hairline_crack": "surface",
    "tile_falling": "nonstruct", "window_or_glass_damage": "nonstruct",
}
_PUBLIC_CONSEQ_ZH = {
    "axial": "這一類情況通常出現在結構已經不太能完全承受上方重量的時候，"
             "代表該處還剩多少支撐力並不確定，餘震時也比較可能繼續惡化",
    "shear": "這種形狀的裂縫一般和地震的左右搖晃有關，可能表示結構抵抗水平晃動的能力已經下降，"
             "餘震時裂縫也可能被拉得更開",
    "cover": "外層混凝土掉落、鋼筋露出來以後，鋼筋就少了保護，短期內可能反映內部受力偏高，"
             "時間久了也容易鏽蝕而讓情況變差",
    "deform": "變形或傾斜沒有回復，通常代表建築在搖晃過程中已經被推到回不去的位置，"
              "這種狀況需要特別小心",
    "crack": "裂縫本身不一定代表結構失去支撐力，但會讓水氣進入，"
             "也可能在後續的餘震中變寬、變長",
    "surface": "這一類多半停留在表面或裝修層，通常不直接牽涉到建築的支撐能力，"
               "不過剝落的碎塊掉下來仍可能砸到人",
    "nonstruct": "這些東西就算不影響主要結構，掉下來或倒下來時仍然可能砸傷人",
    "global": "整體性的傾斜或樓層位移，影響的是整棟建築，而不只是照片裡的這一處",
    "hazard": "這些物件即使結構本身沒問題，在餘震時仍可能鬆脫掉落",
    "axial_maybe": "這一類情況代表這一處還能承受多少重量並不確定，"
                   "需要現場檢查才能判斷，不宜先假設沒問題",
    "extent": "損傷範圍越大，代表受力影響的區域越廣，也越需要現場確認",
}
_PUBLIC_HEDGE = ["似乎可以看到", "看起來也像是有", "另外也有跡象顯示可能有",
                 "畫面上還可能存在"]

# same-finding-from-several-sources collapsing
_PUBLIC_FAMILY = {
    "x_shape_crack": "x", "shear_crack": "diag", "diagonal_crack": "diag",
    "diagonal_crack_45": "diag", "web_crack": "web", "through_crack": "through",
    "vertical_crack": "vert", "horizontal_crack": "horiz", "flexural_crack": "flex",
    "interface_crack": "iface", "hairline_crack": "hair",
    "cover_spalling": "spall", "concrete_spalling": "spall", "surface_spalling": "spall",
    "plaster_spalling": "plaster", "render_spalling": "plaster", "finish_spalling": "plaster",
    "rebar_exposed": "rebar_exp", "rebar_buckling": "rebar_buck",
    "rebar_fracture": "rebar_frac", "stirrup_opening_or_fracture": "stirrup",
    "core_crushing": "crush", "concrete_crushing": "crush", "member_fracture": "fracture",
    "tilt_or_lean": "deform", "residual_deformation": "deform",
    "support_displacement": "support", "joint_damage": "joint",
    "tile_falling": "tile", "window_or_glass_damage": "glass",
}
_SIGNATURE_FAMILY = {
    "x_shape_crack": "x", "short_column_shear": "diag", "joint_failure": "joint",
    "column_axial_failure": "crush", "core_crushing": "crush",
    "rebar_buckling": "rebar_buck", "soft_story": "soft_story",
}
_PATTERN_A_FAMILY = {"Cracks": "crack_any", "Spalling": "spall", "Expose of rebar": "rebar_exp"}
_PATTERN_B_FAMILY = {
    "X-shape": "x", "Diagonal": "diag", "Diagonal_large": "diag",
    "Horizontal": "horiz", "Horizontal_large": "horiz",
    "Vertical": "vert", "Vertiacal_large": "vert",
    "Web": "web", "Web_large": "web", "spalling-like_cracks": "spall_crack",
}
_CRACK_FAMILIES = {"x", "diag", "web", "through", "vert", "horiz", "flex", "iface", "hair"}

_MAX_SUPPORT_ITEMS = 5
_MAX_OTHER_ITEMS = 6
_MAX_CONSEQ = 3


def _cost_phrase(cost: Any) -> str:
    """Everyday-language money. NT$64 must never render as 「1 萬元」."""
    try:
        cost = int(cost or 0)
    except (TypeError, ValueError):
        return ""
    if cost <= 0:
        return ""
    if cost < 1000:
        return "新台幣一千元以內"
    if cost < 10000:
        return f"新台幣 {int(round(cost / 1000))} 千元上下"
    if cost < 100000:
        return f"新台幣 {cost / 10000:.1f} 萬元上下"
    return f"新台幣 {int(round(cost / 10000))} 萬元上下"


def _public_item_text(member_zh: str, f: Dict[str, Any]) -> str:
    """One damage feature -> plain-language phrase. Returns "" when the damage
    type has no zh mapping (never leak a raw enum into the civilian report)."""
    dt = f.get("damage_zh_pub") or ""
    if not dt:
        return ""
    pos = f.get("position_zh_pub") or ""
    where = member_zh or ""
    if pos:
        where = f"{where}（{pos}）" if where else pos
    txt = f"{where}的{dt}" if where else dt
    if (f.get("width_qualitative") in ("wide", "very_wide")
            and _PUBLIC_FAMILY.get(f.get("damage_type") or "") in _CRACK_FAMILIES):
        txt += f"，{WIDTH_QUAL_ZH.get(f['width_qualitative'], '')}"
    return txt


def _public_evidence(a: Dict[str, Any], grade: str) -> Dict[str, Any]:
    """Split every finding into (i) the evidence that carries the headline grade
    and (ii) everything else.

      * A/B  -> supporting = findings at least as severe as the headline;
      * C    -> supporting = the mild findings only, so that anything MORE severe
                is reported plainly instead of being softened away;
      * if nothing reaches the headline severity (the grade came from combining
        weak signals), the most severe findings available become the supporting
        set and `weak` is set — the paragraph then says so honestly rather than
        claiming "no notable damage" while listing damage two sentences later.
    """
    cv = a.get("cv_findings") or {}
    vlm = a.get("vlm_findings") or {}
    kgf = a.get("kg_findings") or {}
    rank = pc.SEVERITY_RANK

    items: List[Dict[str, Any]] = []
    seen_text: set = set()
    seen_family: set = set()
    load_groups: set = set()      # load-capacity verdicts -> consequence sentences

    def push(text: str, abc: str, group: str,
             family: str = "", primary: bool = False, hazard: bool = False) -> None:
        text = (text or "").strip()
        if not text or text in seen_text:
            return
        if family:
            if family in seen_family:
                return
            if family == "crack_any" and (seen_family & _CRACK_FAMILIES):
                return
            seen_family.add(family)
        seen_text.add(text)
        items.append({"text": text, "abc": abc if abc in pc.ABC else "C",
                      "group": group, "primary": primary, "hazard": hazard})

    # --- structural observations (worst member first, worst feature first) --- #
    for m in (vlm.get("members_all") or vlm.get("members") or []):
        is_primary = m.get("structural_role") == "primary_load_bearing"
        mzh = m.get("type_zh_pub") or ""
        for f in (m.get("features_raw") or []):
            dt = f.get("damage_type")
            abc = (pc.iv_to_abc(f.get("severity")) if f.get("severity") in pc.IV_RANK
                   else _PUBLIC_DAMAGE_ABC.get(dt or "", "C"))
            push(_public_item_text(mzh, f), abc,
                 _PUBLIC_CONSEQ_GROUP.get(dt or "", "crack"),
                 _PUBLIC_FAMILY.get(dt or ""), is_primary)
        lc = m.get("load_capacity_status")
        if lc in ("likely_compromised", "lost"):
            load_groups.add("axial")
        elif lc == "possibly_compromised":
            load_groups.add("axial_maybe")
    # --- corroborated critical signatures --- #
    for i, z in enumerate(kgf.get("signatures_pub") or []):
        raw = (kgf.get("signatures_raw") or [None] * (i + 1))[i]
        push(z, "A", "shear", _SIGNATURE_FAMILY.get(raw or "", ""), True)
    # --- whole-building indicators --- #
    for z in (vlm.get("global_indicators_pub") or []):
        push(z, "A", "global", f"gi:{z}", True)
    # --- image-detected patterns (specific crack types before the generic one) - #
    for p in (cv.get("patterns_b_raw") or []):
        abc = "A" if p == "X-shape" else ("B" if p.endswith("_large") else "C")
        grp = "shear" if ("iagonal" in p or p == "X-shape") else "crack"
        push(_zhp(p, PATTERN_B_ZH), abc, grp, _PATTERN_B_FAMILY.get(p, ""))
    for p in (cv.get("patterns_a_raw") or []):
        abc = {"Expose of rebar": "A", "Spalling": "B", "Cracks": "C"}.get(p, "C")
        grp = {"Expose of rebar": "cover", "Spalling": "cover", "Cracks": "crack"}.get(p, "crack")
        push(_zhp(p, PATTERN_A_ZH), abc, grp, _PATTERN_A_FAMILY.get(p, ""))
    # --- measured extent, only when it is actually large --- #
    try:
        if (cv.get("spall_bucket") or 0) >= 2 or float(cv.get("spall_ratio") or 0) >= 0.02:
            push("剝落的範圍在照片中占了不小的面積", "B", "extent", "ext_spall")
        if (cv.get("crack_bucket") or 0) >= 2 or float(cv.get("crack_ratio") or 0) >= 0.02:
            push("裂縫在照片中分布的範圍不小", "B", "extent", "ext_crack")
    except (TypeError, ValueError):
        pass
    # --- surrounding hazards / site markings --- #
    for z in (vlm.get("hazards_pub") or []):
        push(z, "B", "hazard", f"hz:{z}", False, True)
    for z in (vlm.get("markings_pub") or []):
        push(z, "C", "nonstruct", f"mk:{z}")

    gr = rank.get(grade, rank["C"])
    if grade == "C":
        support = [i for i in items if i["abc"] == "C"]
    else:
        support = [i for i in items if rank.get(i["abc"], 0) >= gr]
    weak = False
    if not support and items:
        # nothing reached the headline severity — show the strongest we have and
        # let the paragraph admit that the grade came from combining weak signals
        weak = True
        top = max(rank.get(i["abc"], 0) for i in items)
        support = [i for i in items if rank.get(i["abc"], 0) == top]
    other = [i for i in items if i not in support]
    support = sorted(support, key=lambda i: -rank.get(i["abc"], 0))[:_MAX_SUPPORT_ITEMS]
    # the incidental list is capped too, so order it worst-first: if something has
    # to be dropped it must be the mildest finding, never the most severe one
    other = sorted(other, key=lambda i: -rank.get(i["abc"], 0))[:_MAX_OTHER_ITEMS]

    conseq: List[str] = []
    for i in support:
        c = _PUBLIC_CONSEQ_ZH.get(i["group"])
        if c and c not in conseq:
            conseq.append(c)
    conseq = conseq[:_MAX_CONSEQ]
    # the load-capacity verdict is a conclusion drawn FROM the damage above, so it
    # closes the "what it leads to" clause instead of posing as an observation
    for g in ("axial", "axial_maybe"):
        if g in load_groups and _PUBLIC_CONSEQ_ZH[g] not in conseq:
            conseq.append(_PUBLIC_CONSEQ_ZH[g])
            break
    return {"support": support, "other": other, "consequences": conseq,
            "weak": weak,
            "any_primary": any(i["primary"] for i in support),
            "any_hazard": any(i["hazard"] for i in support),
            "structural_support": [i for i in support if i["group"] != "hazard"]}


def _public_rule_key(grade: str, ev: Dict[str, Any]) -> str:
    """Pick the rule wording that actually matches what drove the grade."""
    if grade == "A":
        return "A_primary" if ev["any_primary"] else "A_secondary"
    if grade == "B":
        return "B_hazard" if (ev["any_hazard"] and not ev["structural_support"]) else "B_struct"
    return "C"


def _public_paragraph(a: Dict[str, Any]) -> str:
    """Section 2 as ONE integrated paragraph:
    what we found -> what it may lead to -> why the result is this grade."""
    grade = a.get("final_grade", "B")
    ev = _public_evidence(a, grade)
    s: List[str] = []

    # (i) what we found — hedged, because these are the grade-carrying claims
    if ev["support"]:
        bits = [f"{_PUBLIC_HEDGE[i % len(_PUBLIC_HEDGE)]}{it['text']}"
                for i, it in enumerate(ev["support"])]
        s.append("在這張照片裡，" + "；".join(bits) + "。")
    else:
        s.append("在這張照片裡，沒有找到明顯的損傷跡象。")

    # (ii) what it may lead to
    if ev["consequences"]:
        s.append("這些情況代表的意義是：" + "；".join(ev["consequences"]) + "。")

    # (iii) why the result is this grade — the mapping / grading rule
    s.append(_RULE_INTRO + PUBLIC_GRADE_RULE_ZH[_public_rule_key(grade, ev)])
    if not ev["support"]:
        s.append(PUBLIC_RULE_TAIL_ZH["none"])
    elif a.get("escalation_needed"):
        s.append(PUBLIC_RULE_TAIL_ZH["escalated"])
    elif ev["weak"]:
        s.append(PUBLIC_RULE_TAIL_ZH["weak"])
    else:
        s.append(PUBLIC_RULE_TAIL_ZH["match"])

    # (iv) which grade the actions were written for, when it differs
    if a.get("escalation_needed"):
        s.append(f"上面的建議行動就是按照更嚴重的 {a.get('conservative_grade')} 級來給的，"
                 f"在專業人員到場確認之前請以此為準。")

    # (v) everything else we saw, stated plainly
    if ev["other"]:
        s.append("除此之外，照片中也看到" + "、".join(i["text"] for i in ev["other"])
                 + "，一併記錄供參考。")

    # (vi) rough repair cost of the visible damage
    cost = _cost_phrase((a.get("cv_findings") or {}).get("cost_total"))
    if cost:
        s.append(f"照片中看得到的這些損傷，若要修補，粗略換算大約是{cost}，"
                 f"這只是依面積和單價表推估的數字，也只涵蓋照片拍到的部分，"
                 f"實際金額仍要現場評估才算得準。")
    return "".join(x for x in s if x)


def render_public_report(a: Dict[str, Any]) -> str:
    """Civilian report: three sections only, no module names, no probabilities,
    and no raw enum ever printed."""
    L: List[str] = []
    add = L.append
    grade = a.get("final_grade", "B")
    act_grade = a.get("conservative_grade") if a.get("escalation_needed") else grade
    cv, vlm = a.get("cv_findings") or {}, a.get("vlm_findings") or {}

    add("# 建築物震後損傷初步評估結果")
    add("")
    add(f"> {_DISCLAIMER}")
    add("")

    # ---------- 一、result + actions ---------- #
    add("## 一、評估結果")
    add("")
    add(f"# {GRADE_HEADLINE_ZH.get(grade, grade)}")
    add("")
    if grade == "B" and (vlm.get("hazards_pub") or []):
        add(PUBLIC_RESULT_SENTENCE_B_HAZARD)
    else:
        add(PUBLIC_RESULT_SENTENCE_ZH.get(grade, ""))
    add("")
    if a.get("escalation_needed"):
        add(f"> 🚨 **安全提醒**：這次判讀中有部分跡象指向**更嚴重的 {act_grade} 級**。"
            f"基於「寧可從嚴」的原則，在專業技師到場確認之前，"
            f"**請直接以 {act_grade} 級的標準採取行動**。")
        add("")
    add("**建議採取的行動：**")
    add("")
    for i, line in enumerate(GRADE_ACTION_ZH.get(act_grade, []), 1):
        add(f"{i}. {line}")
    add("")

    # ---------- 二、one integrated paragraph ---------- #
    add("## 二、詳細說明")
    add("")
    add(_public_paragraph(a))
    add("")

    # ---------- 三、caveats & limits ---------- #
    add("## 三、注意事項與限制")
    add("")
    add("- 本次判讀只看了**一張照片**：無法掌握整棟建築、其他樓層或被擋住的部位，"
        "也無法統計「整層樓有多少比例的結構受損」這類正式評估需要的資訊。")
    if a.get("assessment_basis") == pc.BASIS_CV_ONLY:
        add("- ⚠️ 本次判讀有部分程序未能完成，這個結果是在資訊較少的情況下得出的，"
            "可靠度較低，請務必安排專業人員複檢。")
    if vlm.get("available"):
        if vlm.get("info_sufficiency_pub"):
            add(f"- 這張照片可供判讀的資訊量為：**{vlm['info_sufficiency_pub']}**。")
        if vlm.get("image_issues_pub"):
            add(f"- 影像品質提醒：{'、'.join(vlm['image_issues_pub'])}，可能影響判讀結果。")
        if vlm.get("occlusion"):
            add("- 照片中有物體遮擋，**可能有沒被拍到的損傷**。")
    elif a.get("assessment_basis") != pc.BASIS_CV_ONLY:
        add("- 這張照片只完成了部分判讀程序，可參考的資訊比平常少，請以專業複檢為準。")
    if cv.get("available") and cv.get("edl_uncertainty") is not None:
        try:
            if float(cv["edl_uncertainty"]) >= 0.5:
                add("- 這張照片的判讀**不確定性偏高**，結果僅供參考，"
                    "建議補拍更清楚、距離更近的照片再評估一次。")
        except (TypeError, ValueError):
            pass
    if _cost_phrase(cv.get("cost_total")):
        add("- 修復費用僅為粗略估算（依裂縫型態、面積與單價表換算），"
            "不含拆除、鷹架、裝修復原等費用，實際費用需由專業人員現場評估。")
    add("- 餘震可能使原有損傷擴大，請持續留意裂縫是否變寬、變長，或出現新的傾斜與掉落物。")
    add("- 如需正式的建築物安全評估，請聯繫所在縣市政府工務／建管單位，"
        "或洽詢建築師公會、結構／土木技師公會；並請保留此報告與原始照片供評估人員參考。")
    add(f"- {_DISCLAIMER}")
    add("")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════
# dispatcher (back-compatible: render_report(a) == the civilian report)
# ═══════════════════════════════════════════════════════════════════════════
def render_report(a: Dict[str, Any], audience: str = "public") -> str:
    if audience in ("engineer", "developer", "dev", "technical"):
        return render_engineer_report(a)
    return render_public_report(a)


def render_reports(a: Dict[str, Any]) -> Dict[str, str]:
    """Both renderings in one call -> {"public": md, "engineer": md}."""
    return {"public": render_public_report(a), "engineer": render_engineer_report(a)}