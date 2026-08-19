"""
main_pipeline/kg_infer.py
=========================
Stage 3 — Validator + Knowledge-Graph reasoning for ONE image.

Single-image analogue of the batch chain
    run_validator -> build_handoff_from_results -> build_kg
with everything held in memory (no handoff JSONL, no out/ rewrite):

  1. Track-2 validation of the reasoning JSON (schema / ids / consistency /
     I~V cross-check)  -> verdict + flags                (validator.report)
  2. reasoning_json (+ the LIVE cv_record, per the --gnn-input flag)
        -> instance subgraph                             (vlm_json_to_graph)
  3. grounding + rule reasoning: independent I~V re-derivation, mechanism
     risk, member SAFETY grades (max of VLM & KG — false-reassurance guard),
     severity partial-triggers, placard_suggestion FEATURE
                                                        (reasoning_rules.reason)
  4. tensor row in the EXACT training contract (feature_spec v1.3)
                                                        (export_pyg.to_tensors)

Inference-time policy on the verdict (deliberate difference from training):
the batch KG build SKIPS `rejected` bundles to keep the training set clean; a
production run instead PROCEEDS on any parseable JSON (the graph converter
already repairs duplicate ids and drops dangling relationships, recording
flags) and surfaces the degraded verdict to the report — a flagged triage
answer beats no answer, and the mandate only forbids silently *clearing* a
building. Total generation failure is still a hard stop handled upstream.

The ontology is loaded from validator/built_kg/out/ontology.pkl and built from
the CSVs on first use if absent.
"""
from __future__ import annotations

import logging
import pickle
import time
from typing import Any, Dict, Optional

from main_pipeline import pipeline_config as pc

log = logging.getLogger("main_pipeline.kg")

_ONTOLOGY = None            # loaded once per process
_ONT_CTX = None


def load_ontology():
    """Load (or build once) the static domain ontology + its export context."""
    global _ONTOLOGY, _ONT_CTX
    if _ONTOLOGY is not None:
        return _ONTOLOGY, _ONT_CTX
    from validator.built_kg import export_pyg as ex
    if pc.ONTOLOGY_PKL.exists():
        with pc.ONTOLOGY_PKL.open("rb") as fh:
            _ONTOLOGY = pickle.load(fh)
        log.info("ontology loaded: %d nodes / %d edges",
                 _ONTOLOGY.number_of_nodes(), _ONTOLOGY.number_of_edges())
    else:
        log.warning("ontology.pkl missing — building from csv_files/ (one-time)")
        from validator.built_kg.build_ontology import build_ontology
        _ONTOLOGY = build_ontology(save=True, verbose=False)
    _ONT_CTX = ex.ontology_context(_ONTOLOGY)   # also fills DAMAGE_TYPE_TO_CATEGORY
    return _ONTOLOGY, _ONT_CTX


def run_kg_stage(reasoning_json: Dict[str, Any],
                 image_id: str,
                 cv_record: Optional[Dict[str, Any]],
                 gnn_input: str = pc.DEFAULT_GNN_INPUT,
                 run_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate + reason over one image's VLM output.

    cv_record: the live record from cv_infer ({image_path, element,
    structured_output}) or None. How it is attached follows the gnn_input flag
    so the graph matches the checkpoint that will consume it:
        full     -> attach as-is
        cv_noedl -> attach with the 3 EDL columns dropped (drop_cv_edl=True)
        nocv     -> not attached (cv_present=0, zero CV block)

    Returns {"ok", "verdict", "validation", "kg_summary", "graph_jsonable",
             "tensor_row", "seconds"} — or {"ok": False, "error": ...}.
    """
    from validator import report as vreport
    from validator.built_kg.vlm_json_to_graph import to_instance_graph
    from validator.built_kg.reasoning_rules import reason
    from validator.built_kg import export_pyg as ex
    from validator.built_kg.build_kg import _graph_to_jsonable

    t0 = time.time()
    out: Dict[str, Any] = {"image_id": image_id, "gnn_input": gnn_input}
    try:
        variant_cfg = pc.GNN_VARIANTS[pc.GNN_INPUT_TO_VARIANT[gnn_input]]

        # ---- 1. Track-2 validation --------------------------------------- #
        validation = vreport.validate_image(reasoning_json, image_id=image_id)
        verdict = validation.get("verdict", "rejected")
        out["validation"] = validation
        out["verdict"] = verdict
        if verdict == "rejected":
            log.warning("validator verdict REJECTED — proceeding with repaired "
                        "graph, result reliability degraded (see report)")

        # ---- 2. instance graph (CV attachment per gnn_input) ------------- #
        attach = variant_cfg["attach_cv"] and cv_record is not None
        if variant_cfg["attach_cv"] and cv_record is None:
            log.warning("gnn_input=%s expects a CV record but the CV stage "
                        "produced none — GNN will see cv_present=0 (same as a "
                        "training-time join miss)", gnn_input)
        G, cmeta = to_instance_graph(
            reasoning_json, image_id=image_id,
            cv_record=cv_record if attach else None,
            drop_cv_edl=variant_cfg["drop_cv_edl"])
        out["conversion_flags"] = cmeta.get("conversion_flags", [])
        out["n_members"] = cmeta.get("n_members")
        out["n_features"] = cmeta.get("n_features")

        # ---- 3. grounding + reasoning ------------------------------------ #
        O, ont_ctx = load_ontology()
        G, summ = reason(G, O)
        out["kg_summary"] = summ
        out["graph_jsonable"] = _graph_to_jsonable(G)

        # ---- 4. tensor row (training contract, no label at inference) ---- #
        rm = run_meta or pc.inference_run_meta()
        row = ex.to_tensors(G, None, image_id, rm, ont_ctx)
        row["base_image_id"] = image_id
        out["tensor_row"] = row
        out["ok"] = True
    except Exception as e:  # noqa: BLE001 — fault-isolated stage
        log.exception("KG stage failed on %s", image_id)
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    out["seconds"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(
        description="Validator + KG reasoning on one *.reasoning.json")
    ap.add_argument("reasoning_json")
    ap.add_argument("--image-id", default=None)
    ap.add_argument("--gnn-input", default=pc.DEFAULT_GNN_INPUT,
                    choices=sorted(pc.GNN_INPUT_TO_VARIANT))
    ap.add_argument("--cv-record", default=None,
                    help="optional path to a JSON {image_path,element,structured_output}")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    obj = json.load(open(a.reasoning_json, encoding="utf-8"))
    iid = a.image_id or (obj.get("image_context", {}) or {}).get("id") or "IMG:cli"
    cvr = json.load(open(a.cv_record, encoding="utf-8")) if a.cv_record else None
    r = run_kg_stage(obj, iid, cvr, gnn_input=a.gnn_input)
    r.pop("tensor_row", None)
    r.pop("graph_jsonable", None)
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str)[:6000])
