# YOLO-RDD

Reproducibility resources for the YOLO-RDD pavement-distress detector.

## Reproducibility materials

This repository provides:

- `configs/model/yolo-rdd.yaml`: YOLO-RDD architecture configuration;
- `configs/data/rdd2022-four-class.yaml`: RDD2022 configuration for D00, D10, D20, and D40;
- `configs/training/yolo-rdd-training.yaml`: batch size and repeated-run seeds;
- `yolo_rdd_modules/`: RepGELAN-related, DySlim-Neck, DySample, and DynamicHead implementations;
- `scripts/train_yolo_rdd.py`: training entry point;
- `splits/`: fixed 8,000-image training and 2,000-image validation manifests;
- `integration/`: Ultralytics module-registration instructions.

## Training

The released batch size is `16`. Repeated training uses seeds `0`, `42`, and `2025`.

```bash
python scripts/train_yolo_rdd.py --data /path/to/rdd2022-four-class.yaml --seed 42
```

Set `path` in the data YAML to the four-class RDD2022 development dataset before training. See `requirements.txt` and `integration/ultralytics_integration.md` for dependencies and module registration.

## Self-collected field test set

The [versioned field-test release](https://github.com/Zzz-jkij/YOLO-RDD/releases/tag/v1.0-field-test) contains 171 pavement images acquired using UAVs and smartphones, with 513 manually annotated instances from the D00, D10, D20, and D40 categories. The release is intended only for offline evaluation and was not used for training, validation, or fine-tuning.

The release copies have had EXIF GPS metadata removed. The associated manifest records SHA-256 hashes for integrity verification.

## Citation

Please cite the associated manuscript and this repository when using these materials.

## License

Licensing terms for the released field-test data are supplied in the corresponding release package.
