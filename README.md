# RM-DETR

Official code for **RM-DETR: Real-Time Mask Detection Transformer using Decoupled Semantic Reasoning and Spatial Refinement**.

RM-DETR separates low-resolution semantic reasoning from high-resolution spatial refinement. The S2 feature bypasses the iterative encoder-decoder path and is fused once with the upsampled S3 encoder feature for final mask prediction.


## Installation

```bash
conda create -n rmdetr python=3.10 -y
conda activate rmdetr
pip install -r requirements.txt
```

ONNX and TensorRT benchmarking dependencies are optional:

```bash
pip install -r requirements-onnx.txt
```

Install TensorRT separately for the CUDA version available on the target machine.

## Dataset

Download MS COCO 2017 and arrange it as follows:

```text
data/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
└── val2017/
```

The dataset is not distributed with this repository.

## Configurations

| Group | Configurations | Paper section |
|---|---|---|
| ResNet main models | `configs/main/rmdetr_r{18,34,50,101}.yml` | Tables 1 and 2 |
| HGNetv2 real-time models | `configs/realtime/rmdetr_hg_{s,m,l,x}.yml` | Tables 1 and 3 |
| Mask embedding ablations | `configs/ablation/*.yml` | Table 4 |


## Training

Single GPU:

```bash
python tools/train.py --config configs/main/rmdetr_r50.yml --amp
```

Distributed training:

```bash
torchrun --nproc_per_node=4 tools/train.py \
  --config configs/main/rmdetr_r50.yml \
  --amp
```

All paper models were trained for 72 epochs with a total batch size of 16 and 640 x 640 inputs. When changing the number of GPUs, set the per-process batch size so that the total batch size remains 16.

## Evaluation

To evaluate a trained model, place your own RM-DETR checkpoint under `weights/` and run:

```bash
python tools/train.py \
  --config configs/main/rmdetr_r50.yml \
  --resume weights/rmdetr_r50.pth \
  --test-only

The current evaluator reports both COCO bounding-box and segmentation metrics when RMDETRPostProcessor is selected.

## Paper results

### ResNet models: PyTorch FP32, NVIDIA TITAN RTX, batch size 1

| Model | Input | FLOPs (G) | Params (M) | Latency (ms) | Mask AP |
|---|---:|---:|---:|---:|---:|
| RM-DETR-R18 | 640 | 37.1 | 22.1 | 13.8 | 39.3 |
| RM-DETR-R34 | 640 | 53.2 | 33.4 | 17.4 | 41.1 |
| RM-DETR-R50 | 640 | 74.1 | 42.8 | 26.1 | 44.4 |
| RM-DETR-R101 | 640 | 153.3 | 87.3 | 39.2 | 45.4 |

### HGNetv2 models: TensorRT FP16, NVIDIA TITAN RTX, batch size 1

| Model | FLOPs (G) | Params (M) | Latency (ms) | Mask AP | Box AP |
|---|---:|---:|---:|---:|---:|
| RM-DETR-S | 21.6 | 12.3 | 2.8 | 38.1 | 45.8 |
| RM-DETR-M | 40.6 | 22.2 | 3.9 | 42.0 | 50.4 |
| RM-DETR-L | 77.0 | 45.0 | 5.3 | 43.9 | 52.6 |
| RM-DETR-X | 94.0 | 55.0 | 7.0 | 44.9 | 53.8 |

Machine-readable summaries are available under `results/`.

## Checkpoints

Trained RM-DETR model weights and checkpoints are not distributed with this repository. TensorRT engine and ONNX files are also not distributed.

## Benchmark protocol

The paper uses 640 x 640 inputs and batch size 1. TensorRT FP16 latency is measured after 100 warm-up iterations over 100 timed iterations. The reported latency measures TensorRT inference with CUDA synchronization and excludes data preprocessing and host-to-device/device-to-host transfers.

## License and acknowledgements

RM-DETR is released under the Apache License 2.0. Third-party components remain subject to their respective licenses. See `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES/` for details.

## Citation

Citation metadata will be added after publication. The current manuscript authors are Joo-Hee Oh, Chan-Young Choi, and Sang-Woong Lee.
