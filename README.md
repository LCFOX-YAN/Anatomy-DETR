## 1. Project Structure

```text
.
├── configs/
├── engine/
│   ├── backbone/
│   ├── core/
│   ├── data/
│   ├── deim/
│   ├── optim/
│   └── solver/
├── train.py
├── train.sh
```

This model provides two training configurations. You can choose different backbones for training according to the available GPU memory:

* `configs/train_convnext_s.yml`
* `configs/train_hgnetv2_b5.yml`

## 2. Environment Setup

```bash
conda create -n anatomy-detr python=3.10 -y
conda activate anatomy-detr
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

## 3. Dataset Preparation

The default dataset configuration is located at:

```text
configs/dataset/custom_detection.yml
```

## 4. Weight Preparation

The ConvNeXt configuration reads the default pretrained weight from:

```text
weight/convnext/convnextv2_tiny_384.safetensors
```

The HGNetV2-B5 configuration reads the default pretrained weights from:

```text
weight/hgnetv2/
```

## 5. Training

Example for training on a single machine with 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=16085 \
  --nproc_per_node=4 \
  train.py \
  -c 'configs/train_convnext_s.yml' \
  --use-amp \
  --seed=0
```

You can also run:

```bash
bash train.sh
```
