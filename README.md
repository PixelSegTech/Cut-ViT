# Cut-ViT: Task-Specific Model Pruning via Gram Anchoring Subspace Consistency - ECCV 2026




## Getting Started

This code was developed using `Python 3.11`, so to run the experiments, we recommend that you create an appropriate environment and install the required dependencies as follows:

```bash
conda create -n Cut-ViT python=3.11
conda activate Cut-ViT

# Install the required Python libraries
pip install -r requirements.txt
```

## Dataset Setup
   Our work performs task-specific pruning on eight datasets, namely ImageNet, COCO, ADE20K, DAVIS, NYUv2, FG3DCAR, JODS, and SBD, using the standard dataset split protocols.

## Model Setup

Models are downloaded automatically on demand from HuggingFace, with the exception of DINOv3, for which you'll have to request a download link via [this form](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and download models as follows:

```sh
wget -P /home/$USER/scratch/cache/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth <VIT-B/16 URL>
wget -P /home/$USER/scratch/cache/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth <VIT-H+/16 URL>
```

## Reproducing Experiments

Some of the core experiments in the paper can be reproduced as-is using the scripts in the `jobs` directory. Scripts generally assume that you have a dataset cache folder rooted at `/home/$USER/scratch/cache`; make sure to change it as appropriate.


### Elastic Pruning

- `jobs/cut-vit/dino.sh`: perform elastic pruning for a DINO ViT-B/16 backbone.


