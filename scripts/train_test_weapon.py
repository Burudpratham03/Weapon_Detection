import argparse
import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, test, and save sample detections for weapon detection."
    )
    parser.add_argument("--data", default="data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=2,
                        help="Dataloader workers")
    parser.add_argument("--device", default="",
                        help="Device, e.g. cpu, 0, 0,1")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence for samples")
    parser.add_argument(
        "--sample-count", type=int, default=8, help="How many test images for samples"
    )
    parser.add_argument(
        "--run-name", default="weapon_yolo11n", help="Run folder name under runs"
    )
    return parser.parse_args()


def _normalize_split_path(path_value: str) -> str:
    if path_value.startswith("../"):
        return path_value[3:]
    return path_value


def ensure_data_paths(data_yaml_path: Path) -> dict:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    changed = False
    for split in ("train", "val", "test"):
        if split in data_cfg and isinstance(data_cfg[split], str):
            normalized = _normalize_split_path(data_cfg[split])
            if normalized != data_cfg[split]:
                data_cfg[split] = normalized
                changed = True

    if changed:
        with data_yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data_cfg, f, sort_keys=False)

    return data_cfg


def collect_image_count(project_root: Path, split_path: str) -> int:
    split_dir = project_root / split_path
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in split_dir.rglob("*") if p.suffix.lower() in image_exts)


def collect_sample_images(project_root: Path, split_path: str, sample_count: int) -> list[str]:
    split_dir = project_root / split_path
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        [str(p) for p in split_dir.rglob("*")
         if p.suffix.lower() in image_exts]
    )
    return images[:sample_count]


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_yaml_path = (project_root / args.data).resolve()

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    data_cfg = ensure_data_paths(data_yaml_path)

    counts = {
        "train_images": collect_image_count(project_root, data_cfg["train"]),
        "val_images": collect_image_count(project_root, data_cfg["val"]),
        "test_images": collect_image_count(project_root, data_cfg["test"]),
    }

    start_time = time.time()

    model = YOLO(args.model)

    train_kwargs = {
        "data": str(data_yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(project_root / "runs"),
        "name": args.run_name,
        "exist_ok": True,
    }
    if args.device.strip():
        train_kwargs["device"] = args.device

    print("Starting training...")
    train_results = model.train(**train_kwargs)

    train_save_dir = Path(train_results.save_dir)
    best_weights = train_save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found: {best_weights}")

    print("Running test split evaluation...")
    trained_model = YOLO(str(best_weights))
    test_metrics = trained_model.val(
        data=str(data_yaml_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(project_root / "runs"),
        name=f"{args.run_name}_test",
        exist_ok=True,
    )

    sample_images = collect_sample_images(
        project_root, data_cfg["test"], args.sample_count)
    if not sample_images:
        raise RuntimeError("No test images found for sample detections")

    print(
        f"Generating sample detections on {len(sample_images)} test images...")
    pred_results = trained_model.predict(
        source=sample_images,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project=str(project_root / "runs"),
        name=f"{args.run_name}_samples",
        exist_ok=True,
    )

    sample_output_dir = project_root / "runs" / f"{args.run_name}_samples"
    pred_items = []
    for res in pred_results:
        pred_items.append(
            {
                "image": str(res.path),
                "detections": int(len(res.boxes)) if res.boxes is not None else 0,
            }
        )

    elapsed_sec = round(time.time() - start_time, 2)

    summary = {
        "dataset": {
            "yaml": str(data_yaml_path),
            "class_count": int(data_cfg.get("nc", 0)),
            "class_names": data_cfg.get("names", []),
            "image_counts": counts,
        },
        "training": {
            "base_model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "run_dir": str(train_save_dir),
            "best_weights": str(best_weights),
            "elapsed_seconds": elapsed_sec,
        },
        "test_metrics": {
            "precision": float(test_metrics.box.mp),
            "recall": float(test_metrics.box.mr),
            "map50": float(test_metrics.box.map50),
            "map50_95": float(test_metrics.box.map),
        },
        "sample_detections": {
            "source_images": sample_images,
            "output_dir": str(sample_output_dir),
            "per_image_detection_counts": pred_items,
        },
    }

    summary_json_path = train_save_dir / "results_summary.json"
    summary_txt_path = train_save_dir / "results_summary.txt"

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("Weapon Detection Training and Testing Summary\n")
        f.write("=" * 48 + "\n")
        f.write(f"Data file: {summary['dataset']['yaml']}\n")
        f.write(f"Classes: {summary['dataset']['class_names']}\n")
        f.write(f"Image counts: {summary['dataset']['image_counts']}\n")
        f.write(f"Best weights: {summary['training']['best_weights']}\n")
        f.write(f"Elapsed seconds: {summary['training']['elapsed_seconds']}\n")
        f.write("\nTest metrics\n")
        f.write(f"Precision: {summary['test_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {summary['test_metrics']['recall']:.4f}\n")
        f.write(f"mAP@0.50: {summary['test_metrics']['map50']:.4f}\n")
        f.write(f"mAP@0.50:0.95: {summary['test_metrics']['map50_95']:.4f}\n")
        f.write("\nSample detections output dir\n")
        f.write(f"{summary['sample_detections']['output_dir']}\n")

    print("Done.")
    print(f"Best weights: {best_weights}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary TXT: {summary_txt_path}")
    print(f"Sample detections: {sample_output_dir}")


if __name__ == "__main__":
    main()
