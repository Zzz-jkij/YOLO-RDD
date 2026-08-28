"""Train YOLO-RDD with seed 0, 42, or 2025."""

from argparse import ArgumentParser
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
parser = ArgumentParser()
parser.add_argument("--data", default=ROOT / "configs/data/rdd2022-four-class.yaml", type=Path)
parser.add_argument("--seed", default=42, choices=(0, 42, 2025), type=int)
args = parser.parse_args()

model = YOLO(ROOT / "configs/model/yolo-rdd.yaml")
model.train(data=str(args.data), seed=args.seed, batch=16)
