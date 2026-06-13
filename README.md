# RPN-YOLO

Official implementation of **RPN-YOLO: Rectangular Self-Calibration for Spatial Fidelity Preservation in Small Traffic Sign Detection**, submitted to Neurocomputing.

RPN-YOLO jointly optimizes feature enhancement, scale perception, and regression supervision for small traffic sign detection through a single core mechanism—**Rectangular Self-Calibration (RSC)**. On TT100K, it achieves **79.1% mAP@0.5** and **50.12% APsmall** with only **2.62M parameters**.

## Key Results

| Model | TT100K mAP@0.5 | TT100K mAP@0.5:0.95 | GTSDB mAP@0.5 | GTSDB mAP@0.5:0.95 | Params | FPS |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| YOLO11n | 73.9 | 55.8 | 47.9 | 39.3 | 2.59M | 47.3 |
| **RPN-YOLO** | **79.1** | **59.7** | **51.9** | **42.7** | **2.62M** | **43.9** |

## Environment

- Ubuntu 22.04
- Python 3.10
- PyTorch 2.3.0 + CUDA 12.1 + cuDNN 8.9.2

```bash
pip install -r requirements.txt
```

## Weights

| File | Description |
|------|-------------|
| `baseline.pt` | YOLO11n baseline |
| `RPN-YOLO.pt` | Full RPN-YOLO |

## Validation

```bash
python val.py --weights RPN-YOLO.pt --data TT100k.yaml
python val.py --weights RPN-YOLO.pt --data GTSDB.yaml
```