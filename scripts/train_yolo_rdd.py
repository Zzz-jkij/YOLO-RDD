"""Train YOLO-RDD with a manuscript-reported configuration and seed 0, 42, or 2025."""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO
import yaml


ROOT = Path(__file__).resolve().parents[1]
parser = ArgumentParser()
parser.add_argument("--data", default=ROOT / "configs/data/rdd2022-four-class.yaml", type=Path)
parser.add_argument("--config", default=ROOT / "configs/training/yolo-rdd-training.yaml", type=Path)
parser.add_argument("--seed", default=42, choices=(0, 42, 2025), type=int)
args = parser.parse_args()

with args.config.open(encoding="utf-8") as f:
    training = yaml.safe_load(f)

model = YOLO(ROOT / "configs/model/yolo-rdd.yaml")
model.train(
    data=str(args.data),
    seed=args.seed,
    optimizer=training["optimizer"],
    epochs=training["epochs"],
    patience=training["patience"],
    batch=training["batch"],
    imgsz=training["imgsz"],
    lr0=training["lr0"],
    momentum=training["momentum"],
    weight_decay=training["weight_decay"],
)
