#!/usr/bin/env python3
"""
main_pipeline/train_gnn_models.py
=================================
Train and SAVE the three deployable GNN checkpoints the main pipeline needs
(the formal eval only ran cross-validation and never persisted a model):

    fewshot_full          <- validator/built_kg/out_fewshot          (--cv-cache, --cv-mode full)
    fewshot_cvnoedl_full  <- validator/built_kg/out_fewshot_cvnoedl  (--cv-cache, --cv-mode no_edl)
    fewshot_nocv_full     <- validator/built_kg/out_fewshot_nocv     (no cv cache)

For each variant this script:
  1. locates the training dataset (dataset_tensors.jsonl + feature_spec.json);
     if MISSING, rebuilds it from the existing few-shot handoff + CV cache by
     invoking validator.built_kg.build_kg with the variant's flags (exactly the
     RUNBOOK Step-3 recipe) and copying validator/built_kg/out/ aside;
  2. runs a leakage-safe grouped-stratified CV (default 5 folds x 1 repeat —
     enough to estimate the median best epoch; the full 5x5 numbers already
     exist under validator/gnn/out/) using the UNCHANGED validator/gnn stack;
  3. fits a final model on ALL rows at the CV median best epoch (train_eval.
     fit_final — same recipe as `--fit-final` in the RUNBOOK) and saves the
     deployable checkpoint + the variant's feature_spec.json + a training
     summary to  main_pipeline/models/gnn/<variant>/.

Both architectures (rgcn + gin) are trained per variant by default so
run_pipeline.py can pick either via --gnn-arch.

Run from the repo root:
    PYTHONPATH=. python3 -m main_pipeline.train_gnn_models                  # all 3, both archs
    PYTHONPATH=. python3 -m main_pipeline.train_gnn_models --variants fewshot_full --models rgcn
    PYTHONPATH=. python3 -m main_pipeline.train_gnn_models --cuda --folds 5 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from main_pipeline import pipeline_config as pc

# validator/gnn scripts use FLAT imports (import gnn_config as C, from models
# import build_model), so the gnn dir itself must be on sys.path before any of
# them are imported. This mirrors how they run inside validator/gnn/.
sys.path.insert(0, str(pc.GNN_DIR))


# --------------------------------------------------------------------------- #
# dataset (re)build — RUNBOOK_GNN Step 3, automated
# --------------------------------------------------------------------------- #
def _dataset_paths(kg_out_dir: str):
    d = pc.BUILT_KG_DIR / kg_out_dir
    return d / "dataset_tensors.jsonl", d / "feature_spec.json"


def ensure_dataset(variant: str, rebuild: bool = False) -> Dict[str, Path]:
    """Return {'tensors','spec'} for the variant, rebuilding via build_kg if
    the copy is missing (or --rebuild-datasets was passed)."""
    v = pc.GNN_VARIANTS[variant]
    tensors, spec = _dataset_paths(v["kg_out_dir"])
    if tensors.exists() and spec.exists() and not rebuild:
        print(f"[{variant}] dataset found: {tensors.parent}")
        return {"tensors": tensors, "spec": spec}

    # rebuild from the few-shot handoff (+ cv cache when the variant fuses CV)
    if not pc.FEWSHOT_HANDOFF.exists():
        raise FileNotFoundError(
            f"[{variant}] dataset missing at {tensors.parent} and cannot rebuild: "
            f"handoff not found at {pc.FEWSHOT_HANDOFF}")
    cmd = [sys.executable, "-m", "validator.built_kg.build_kg",
           "--handoff", str(pc.FEWSHOT_HANDOFF)]
    if v["cv_cache"]:
        if not pc.CV_CACHE_EVAL.exists():
            raise FileNotFoundError(
                f"[{variant}] needs the CV cache but {pc.CV_CACHE_EVAL} is missing")
        cmd += ["--cv-cache", str(pc.CV_CACHE_EVAL), "--cv-mode", v["cv_mode"]]

    print(f"[{variant}] rebuilding dataset:\n    {' '.join(cmd)}")
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(pc.REPO_ROOT) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(pc.REPO_ROOT), env=env)

    # copy validator/built_kg/out -> validator/built_kg/<kg_out_dir> (RUNBOOK step 3)
    src = pc.BUILT_KG_DIR / "out"
    dst = tensors.parent
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst,
                    ignore=shutil.ignore_patterns("reasoned_subgraphs", "samples"))
    if not (tensors.exists() and spec.exists()):
        raise RuntimeError(f"[{variant}] rebuild finished but {tensors} / {spec} missing")
    print(f"[{variant}] dataset rebuilt -> {dst}")
    return {"tensors": tensors, "spec": spec}


# --------------------------------------------------------------------------- #
# train one (variant, arch): CV for the epoch estimate, then fit-final
# --------------------------------------------------------------------------- #
def train_variant_arch(variant: str, arch: str, paths: Dict[str, Path], a) -> Dict:
    import numpy as np                      # noqa: F401 (train_eval dependency)
    import torch
    import gnn_config as C
    import train_eval as TE

    device = "cuda" if (a.cuda and torch.cuda.is_available()) else "cpu"
    out_dir = pc.GNN_MODELS_DIR / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = C.TrainConfig(
        model=arch,
        hidden_dim=a.hidden, num_layers=a.layers, dropout=a.dropout,
        lr=a.lr, weight_decay=a.weight_decay, epochs=a.epochs,
        batch_size=a.batch_size, patience=a.patience,
        n_folds=a.folds, n_repeats=a.repeats, seed=a.seed,
        class_weighting=a.class_weighting, select_metric=a.select,
        out_dir=str(out_dir), fit_final=True,
    )

    print(f"\n===== [{variant} / {arch.upper()}] CV ({cfg.n_repeats}x{cfg.n_folds}) "
          f"on {paths['tensors'].parent.name}  device={device} =====")
    res = TE.cross_validate(cfg, str(paths["tensors"]), str(paths["spec"]),
                            device=device, verbose=a.verbose)
    agg = res["summary"]["aggregate"]
    print(f"[{variant}/{arch}] CV macroF1={agg['macro_f1']['mean']:.3f}"
          f"±{agg['macro_f1']['std']:.3f}  acc={agg['accuracy']['mean']:.3f}  "
          f"A-recall={agg['recall_A']['mean']:.3f}  "
          f"FR={agg['false_reassurance_rate']['mean']:.3f}")

    ep = max(20, res["summary"]["median_best_epoch"])
    ckpt = out_dir / f"{arch}_final.pt"
    TE.fit_final(cfg, res["rows"], res["dataset"], res["in_dim"], res["spec"],
                 device, epochs=ep, out_path=ckpt)
    print(f"[{variant}/{arch}] saved deployable checkpoint -> {ckpt} "
          f"(trained {ep} epochs on all {len(res['rows'])} graphs)")

    # persist the CV evidence next to the checkpoint
    (out_dir / f"{arch}_cv_per_fold.json").write_text(
        json.dumps(res["per_fold"], indent=2), encoding="utf-8")
    (out_dir / f"{arch}_cv_summary.json").write_text(
        json.dumps(res["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    return res["summary"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=list(pc.GNN_VARIANTS),
                    choices=list(pc.GNN_VARIANTS),
                    help="which checkpoint families to (re)train")
    ap.add_argument("--models", default="both", choices=["both", *pc.GNN_ARCHS],
                    help="architectures to train per variant (default both)")
    ap.add_argument("--rebuild-datasets", action="store_true",
                    help="force re-running build_kg for each variant's dataset")
    # CV / optimisation knobs (defaults mirror validator/gnn/train_eval.py; the
    # lighter 5x1 CV is only used to estimate the fit-final epoch — the formal
    # 5x5 evaluation numbers already exist under validator/gnn/out/)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--class-weighting", choices=["none", "inverse"], default="none")
    ap.add_argument("--select", default="macro_f1",
                    choices=["macro_f1", "accuracy", "balanced_accuracy", "a_recall"])
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    archs = list(pc.GNN_ARCHS) if a.models == "both" else [a.models]
    overview: Dict[str, Dict] = {}

    for variant in a.variants:
        paths = ensure_dataset(variant, rebuild=a.rebuild_datasets)
        out_dir = pc.GNN_MODELS_DIR / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        # the checkpoint's exact input contract travels with it
        shutil.copy2(paths["spec"], out_dir / "feature_spec.json")

        summaries = {}
        for arch in archs:
            summaries[arch] = train_variant_arch(variant, arch, paths, a)
        (out_dir / "training_summary.json").write_text(
            json.dumps({"variant": variant,
                        "gnn_input": pc.GNN_VARIANTS[variant]["gnn_input"],
                        "dataset": str(paths["tensors"]),
                        "archs": summaries},
                       indent=2, ensure_ascii=False), encoding="utf-8")
        overview[variant] = {arch: s["aggregate"]["macro_f1"]["mean"]
                             for arch, s in summaries.items()}

    print("\n================ TRAINED CHECKPOINTS ================")
    for variant, per_arch in overview.items():
        for arch, f1 in per_arch.items():
            print(f"  {variant:24s} {arch:5s}  CV macroF1={f1:.3f}   "
                  f"-> {pc.GNN_MODELS_DIR / variant / (arch + '_final.pt')}")
    print("\nrun the pipeline with:  PYTHONPATH=. python3 -m main_pipeline.run_pipeline "
          "<image> --element <wall|column|beam> --gnn-input <full|cv_noedl|nocv>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
