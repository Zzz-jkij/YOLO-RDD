# Integrating the YOLO-RDD modules with Ultralytics 8.3.0

The four files in `yolo_rdd_modules/` provide the YOLO-RDD-specific module implementations. They must be made visible to the Ultralytics YAML parser before `configs/model/yolo-rdd.yaml` is loaded.

1. Copy `yolo_rdd_modules/` to `ultralytics/nn/yolo_rdd_modules/` in an Ultralytics 8.3.0 source checkout.
2. In `ultralytics/nn/tasks.py`, import the five exported names:

```python
from ultralytics.nn.yolo_rdd_modules import (
    DynamicHead, Dy_Sample, GSConv, RepNCSPELAN4_low, VoVGSCSPC,
)
```

3. In `parse_model`, treat `RepNCSPELAN4_low`, `VoVGSCSPC`, and `GSConv` as channel-changing modules. Each `VoVGSCSPC` entry in the released model YAML represents one aggregation stage.
4. In `parse_model`, add a `Dy_Sample` branch that prepends the input channel count to its YAML arguments:

```python
elif m is Dy_Sample:
    c2 = ch[f]
    args = [c2, *args]
```

5. In `parse_model`, add a `DynamicHead` branch that appends the input-channel list for its three input feature maps:

```python
elif m is DynamicHead:
    args.append([ch[x] for x in f])
```

6. Add `DynamicHead` to the detection-head task-dispatch checks used by the installed Ultralytics version.

These steps are intentionally limited to registration and argument routing. The architecture itself remains specified by `configs/model/yolo-rdd.yaml`.
