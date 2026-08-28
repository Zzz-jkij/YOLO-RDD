# YOLO-RDD architecture configuration

The complete model graph is defined in `configs/model/yolo-rdd.yaml`. This note records the configuration so that the architecture can be inspected directly.

## Backbone

The nano scale is selected with depth multiplier `0.50`, width multiplier `0.25`, and maximum channels `1024`. The backbone contains four `RepNCSPELAN4_low` blocks, configured at YAML layers 2, 4, 6, and 8 with nominal output channels 256, 512, 512, and 1024, respectively. Their exact implementation is in `yolo_rdd_modules/GELAN.py`.

## Neck

The top-down route uses two `Dy_Sample` layers with `scale=2` and `style="lp"`. The module defaults retained by the configuration are `groups=4` and `dyscope=False`. Three `VoVGSCSPC` nodes are each configured with three repeats and nominal output channels of 256. Two `GSConv` layers use nominal output channels 256 and 512, respectively, with a 3 x 3 kernel and stride 2. Their implementations are in `yolo_rdd_modules/Dysample_nsl.py` and `yolo_rdd_modules/SlimNeck.py`.

## Detection head

`DynamicHead` receives the three feature maps generated at YAML layers 16, 19, and 22, corresponding to P3, P4, and P5 outputs. It is initialized with the dataset class count supplied by the data configuration. Its implementation is in `yolo_rdd_modules/DynamicHead.py`.

## Dataset class count

The base model YAML retains the conventional `nc: 80` declaration used by the YOLO configuration template. During training, the four-class RDD2022 data configuration (`nc: 4`) supplies the active detection class count.
