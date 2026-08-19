"""
main_pipeline/gnn_infer.py
==========================
Stage 4 — GNN A/B/C prediction on ONE graph (the RUNBOOK's "Step 7 deployment
inference", implemented).

Loads a deployable checkpoint saved by main_pipeline/train_gnn_models.py
(train_eval.fit_final payload: state_dict + in_dim + num_relations +
hidden/layers/dropout + label_map + feature_spec_version), rebuilds the exact
architecture via validator/gnn/models, converts the single tensor row with the
SAME encoders training used (gnn_dataset._homogeneous_nodes / _edges_typed /
_edges_undirected — imported, not re-implemented, so the feature layout can
never drift), and runs batch_size=1.

Graphs are tiny (3-11 nodes); inference runs on CPU by design so the single
production GPU stays dedicated to the vLLM server + CV models.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from main_pipeline import pipeline_config as pc

# validator/gnn uses flat imports; expose it exactly as train_gnn_models does.
sys.path.insert(0, str(pc.GNN_DIR))

log = logging.getLogger("main_pipeline.gnn")

_MODEL_CACHE: Dict[str, Any] = {}


def _load_checkpoint(gnn_input: str, arch: str, device: str = "cpu"):
    """Load + rebuild a deployable model once per (variant, arch)."""
    key = f"{gnn_input}:{arch}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    import torch
    from models import build_model            # validator/gnn/models (flat import)

    ckpt_path = pc.gnn_checkpoint_path(gnn_input, arch)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"GNN checkpoint missing: {ckpt_path}\n"
            f"Train it first:  PYTHONPATH=. python3 -m main_pipeline.train_gnn_models "
            f"--variants {pc.GNN_INPUT_TO_VARIANT[gnn_input]} --models {arch}")
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    cfg = SimpleNamespace(hidden_dim=payload["hidden_dim"],
                          num_layers=payload["num_layers"],
                          dropout=payload["dropout"])
    model = build_model(payload["model"], payload["in_dim"], cfg,
                        num_relations=payload.get("num_relations"))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    spec_path = pc.gnn_feature_spec_path(gnn_input)
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.exists() else None

    entry = {"model": model, "payload": payload, "spec": spec, "path": str(ckpt_path)}
    _MODEL_CACHE[key] = entry
    log.info("loaded %s (%s) in_dim=%d rel=%s spec=v%s", ckpt_path.name, gnn_input,
             payload["in_dim"], payload.get("num_relations"),
             payload.get("feature_spec_version"))
    return entry


def _row_to_data(row: Dict[str, Any], spec: Dict[str, Any], arch: str):
    """Tensor row -> a single PyG Data, using the TRAINING encoders (imported
    from gnn_dataset) so inference features are bit-identical to training.
    Unlike gnn_dataset.row_to_gin/rgcn this attaches no label (inference)."""
    import torch
    from torch_geometric.data import Data
    import gnn_dataset as D                    # flat import (validator/gnn)

    x, gidx = D._homogeneous_nodes(row, spec)
    xt = (torch.tensor(x, dtype=torch.float) if x
          else torch.zeros((0, spec["homogeneous_dim"])))
    if arch == "gin":
        edges = D._edges_undirected(row, gidx)
        ei = (torch.tensor(edges, dtype=torch.long).t().contiguous()
              if edges else torch.zeros((2, 0), dtype=torch.long))
        d = Data(x=xt, edge_index=ei, num_nodes=len(x))
    elif arch == "rgcn":
        sd, et = D._edges_typed(row, gidx)
        ei = (torch.tensor(sd, dtype=torch.long).t().contiguous()
              if sd else torch.zeros((2, 0), dtype=torch.long))
        d = Data(x=xt, edge_index=ei, num_nodes=len(x))
        d.edge_type = (torch.tensor(et, dtype=torch.long)
                       if et else torch.zeros((0,), dtype=torch.long))
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return d


def run_gnn_stage(tensor_row: Dict[str, Any],
                  gnn_input: str = pc.DEFAULT_GNN_INPUT,
                  arch: str = pc.DEFAULT_GNN_ARCH,
                  device: str = "cpu") -> Dict[str, Any]:
    """Predict A/B/C for one tensor row (from kg_infer.run_kg_stage).

    Returns {"ok","grade","probs":{A,B,C},"logits","model":{...},"seconds"}
    or {"ok": False, "error": ...}. Never raises.
    """
    t0 = time.time()
    out: Dict[str, Any] = {"gnn_input": gnn_input, "arch": arch}
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Batch

        entry = _load_checkpoint(gnn_input, arch, device)
        payload = entry["payload"]
        spec = entry["spec"]
        if spec is None:
            raise FileNotFoundError(
                f"feature_spec.json missing next to the checkpoint "
                f"({pc.gnn_feature_spec_path(gnn_input)}); re-run train_gnn_models")

        # contract check: the graph we built must match what the model expects
        if int(spec["homogeneous_dim"]) != int(payload["in_dim"]):
            raise ValueError(
                f"feature dim mismatch: graph spec homogeneous_dim="
                f"{spec['homogeneous_dim']} but checkpoint in_dim={payload['in_dim']}")

        data = _row_to_data(tensor_row, spec, arch)
        batch = Batch.from_data_list([data]).to(device)
        with torch.no_grad():
            logits = entry["model"](batch)[0]
            probs = F.softmax(logits, dim=-1)

        label_map: Dict[str, int] = payload["label_map"]      # {"A":0,"B":1,"C":2}
        idx_to_label = {v: k for k, v in label_map.items()}
        pred_idx = int(probs.argmax().item())
        out.update({
            "ok": True,
            "grade": idx_to_label[pred_idx],
            "probs": {idx_to_label[i]: round(float(probs[i]), 4)
                      for i in range(len(idx_to_label))},
            "logits": [round(float(v), 4) for v in logits],
            "n_nodes": int(data.num_nodes),
            "n_edges": int(data.edge_index.size(1)),
            "model": {
                "variant": pc.GNN_INPUT_TO_VARIANT[gnn_input],
                "arch": payload["model"],
                "checkpoint": entry["path"],
                "in_dim": payload["in_dim"],
                "feature_spec_version": payload.get("feature_spec_version"),
                "trained_on_rows": payload.get("trained_on_rows"),
                "trained_epochs": payload.get("trained_epochs"),
            },
        })
    except Exception as e:  # noqa: BLE001 — fault-isolated stage
        log.exception("GNN stage failed")
        out.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    out["seconds"] = round(time.time() - t0, 2)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="GNN inference on one tensor-row JSON")
    ap.add_argument("row_json", help="a file holding one to_tensors() row")
    ap.add_argument("--gnn-input", default=pc.DEFAULT_GNN_INPUT,
                    choices=sorted(pc.GNN_INPUT_TO_VARIANT))
    ap.add_argument("--arch", default=pc.DEFAULT_GNN_ARCH, choices=pc.GNN_ARCHS)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    row = json.load(open(a.row_json, encoding="utf-8"))
    print(json.dumps(run_gnn_stage(row, a.gnn_input, a.arch),
                     ensure_ascii=False, indent=2))
