from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
LABEL_INFER = {
    "azure": REPO_ROOT / "label" / "processor" / "AzureOpenAI.py",
    "vllm": REPO_ROOT / "label" / "processor" / "vLLM.py",
}
FEATURE_INFER = {
    "azure": REPO_ROOT / "feature" / "processor" / "AzureOpenAI.py",
    "vllm": REPO_ROOT / "feature" / "processor" / "vLLM.py",
}
LABEL_EVAL = REPO_ROOT / "label" / "processor" / "eval.py"
FEATURE_EVAL = REPO_ROOT / "feature" / "processor" / "eval.py"
POSITIVE_VALUE = 1  # feature pipeline expects 1 for true positives


@dataclass
class DatasetSpec:
    tag: str                 # “generated”, “reference”, etc.
    reports: Path            # CSV with columns: study_id, report
    label_gt: Path | None    # CSV for label evaluation and TP filtering
    feature_gt: Path | None  # JSON (or CSV) for feature evaluation


async def run_cmd(tag: str, command: list[str]) -> None:
    print(f"[{tag}] {' '.join(command)}")
    proc = await asyncio.create_subprocess_exec(*command, cwd=str(REPO_ROOT))
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"{tag} failed with exit code {rc}")


async def run_label_inference(spec: DatasetSpec, backbone: str, model: str, output_root: Path) -> tuple[Path, Path]:
    out_dir = output_root / spec.tag / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(LABEL_INFER[backbone]),
        "--model_name",
        model,
        "--input_csv",
        str(spec.reports),
        "--o",
        str(out_dir),
    ]
    await run_cmd(f"{spec.tag}-label-infer", cmd)
    pred_json = out_dir / "tmp" / f"output_labels_{model}.json"
    return out_dir, pred_json


async def run_label_evaluation(spec: DatasetSpec, out_dir: Path, model: str) -> None:
    if spec.label_gt is None:
        return
    cmd = [
        sys.executable,
        str(LABEL_EVAL),
        "--gt_dir",
        str(spec.label_gt),
        "--gen_dir",
        str(out_dir),
        "--model_name",
        model,
    ]
    await run_cmd(f"{spec.tag}-label-eval", cmd)


def build_tp_label_csv(gt_csv: Path, pred_json: Path, output_csv: Path) -> Path:
    df_gt = pd.read_csv(gt_csv).set_index("study_id")
    df_tp = pd.DataFrame(0, index=df_gt.index, columns=df_gt.columns)

    with open(pred_json, encoding="utf-8") as fh:
        preds = json.load(fh)

    for study_id, conditions in preds.items():
        if study_id not in df_tp.index:
            continue
        for condition, value in conditions.items():
            if condition not in df_tp.columns:
                continue
            if str(value).lower() == "positive" and df_gt.loc[study_id, condition] == 1:
                df_tp.loc[study_id, condition] = POSITIVE_VALUE

    df_tp.insert(0, "study_id", df_tp.index)
    df_tp.to_csv(output_csv, index=False)
    return output_csv


async def run_feature_inference(
    spec: DatasetSpec,
    backbone: str,
    model: str,
    filtered_labels: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    out_dir = output_root / spec.tag / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(FEATURE_INFER[backbone]),
        "--model",
        model,
        "--r",
        str(spec.reports),
        "--l",
        str(filtered_labels),
        "--o",
        str(out_dir),
    ]
    await run_cmd(f"{spec.tag}-feature-infer", cmd)
    gen_json = out_dir / f"output_feature_{model}.json"
    return out_dir, gen_json


async def run_feature_evaluation(spec: DatasetSpec, out_dir: Path, gen_json: Path) -> None:
    if spec.feature_gt is None:
        return
    cmd = [
        sys.executable,
        str(FEATURE_EVAL),
        "--gen_path",
        str(gen_json),
        "--gt_path",
        str(spec.feature_gt),
        "--metric_path",
        str(out_dir),
    ]
    await run_cmd(f"{spec.tag}-feature-eval", cmd)


async def orchestrate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    specs: list[DatasetSpec] = [
        DatasetSpec("generated", args.gen_reports),
        DatasetSpec("reference", args.gt_reports)
    ]


    # Stage 1: label inference for all datasets
    label_results = await asyncio.gather(
        *(run_label_inference(spec, args.label_backbone, args.label_model, output_root) for spec in specs)
    )
    label_dirs = {spec.tag: result[0] for spec, result in zip(specs, label_results)}
    pred_jsons = {spec.tag: result[1] for spec, result in zip(specs, label_results)}

    # Stage 2: label evaluation (only after all inference completed)
    await asyncio.gather(
        *(run_label_evaluation(spec, label_dirs[spec.tag], args.label_model) for spec in specs)
    )

    # Stage 3: build filtered label CSVs for the feature extractor
    filtered_csvs: dict[str, Path] = {}
    for spec in specs:
        if spec.label_gt is None:
            raise ValueError(f"{spec.tag}: label ground truth required to build feature labels.")
        filtered_csvs[spec.tag] = await asyncio.to_thread(
            build_tp_label_csv,
            spec.label_gt,
            pred_jsons[spec.tag],
            output_root / spec.tag / "labels_for_features.csv",
        )

    # Stage 4: feature inference for all datasets
    feature_results = await asyncio.gather(
        *(
            run_feature_inference(
                spec,
                args.feature_backbone,
                args.feature_model,
                filtered_csvs[spec.tag],
                output_root,
            )
            for spec in specs
        )
    )
    feature_dirs = {spec.tag: result[0] for spec, result in zip(specs, feature_results)}
    feature_jsons = {spec.tag: result[1] for spec, result in zip(specs, feature_results)}

    # Stage 5: feature evaluation
    await asyncio.gather(
        *(
            run_feature_evaluation(spec, feature_dirs[spec.tag], feature_jsons[spec.tag])
            for spec in specs
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CLEAR label and feature pipelines with stage barriers.")
    parser.add_argument("--gen-reports", type=Path, required=True, help="CSV of generated reports.")
    parser.add_argument("--gt-reports", type=Path, help="CSV of reference reports.")

    parser.add_argument("--label-backbone", choices=("azure", "vllm"), default="vllm")
    parser.add_argument("--label-model", required=True)
    parser.add_argument("--feature-backbone", choices=("azure", "vllm"), default="vllm")
    parser.add_argument("--feature-model", required=True)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs", help="Directory to store outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(orchestrate(args))


if __name__ == "__main__":
    main()
