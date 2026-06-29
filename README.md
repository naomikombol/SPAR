# SPAR: Single-Pass Any-Resolution ViT for Open-vocabulary Segmentation

This repository contains the official code for the paper Naomi Kombol, Ivan Martinović, Siniša Šegvić, Giorgos Tolias,  
*"SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation"*,  
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026.  

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2503.19777-b31b1b.svg)](https://arxiv.org/abs/2604.02252)
[![Project Page](https://img.shields.io/badge/Project_Page-SPAR-blue)](https://naomikombol.github.io/SPAR/)
[![CVPR 2026](https://img.shields.io/badge/CVPR_2026-Paper-green)](https://openaccess.thecvf.com/content/CVPR2026/html/Kombol_SPAR_Single-Pass_Any-Resolution_ViT_for_Open-vocabulary_Segmentation_CVPR_2026_paper.html)

</div>

## Environment Setup and Installation
```bash
conda create -n spar python=3.10

# Activate environment
conda activate spar

# Install pytorch
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Required <2.2.0 because of compatibility with mmseg
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# Evaluation dataloading relies on MMSeg
pip install \
  mmengine==0.10.6 \
  mmsegmentation==1.2.2 \
  open_clip_torch==2.29.0 \
  torchmetrics==1.6.0 \
  scikit-learn==1.6.1 \
  openpyxl==3.1.5 \
  pycocotools==2.0.8 \
  numpy==1.26.4 \
  pytorch_lightning==2.5.3 
```

### Dinov3txt preparation
Clone the DINOv3txt repository and take only the dinov3 directory to your main SPAR directory:
```
git clone https://github.com/facebookresearch/dinov3.git
```



## Evaluation Datasets
The following dataset configurations are present in this repo: PASCAL VOC, PASCAL Context, Cityscapes, ADE20k, with two more variant datasets VOC20, Context59 (i.e., PASCAL VOC and PASCAL Context without the background category).

Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. 


## Training Dataset (SA-1B)

SPAR models are trained using the first 25K images from the SA-1B dataset.

A helper script is provided to automate downloading and extraction. Before running, set your desired `OUTPUT_DIR` in `sa1b_download.sh`, which defines where the extracted `.png` images will be stored.

```bash
sh sa1b_download.sh
```

## 🚀 SPAR
Ensure all config files are correctly set before running experiments. In particular, dataset root directories must match across image generation and training, and evaluation dataset paths must be updated to reflect your local setup.

Misaligned paths will result in errors or incomplete runs.

---


SPAR provides a single end-to-end script that handles:
- Augmented image generation  
- Model training  
- Evaluation on 6 benchmark datasets  

All steps are executed automatically for the model specified in `spar.sh`.

To run the full pipeline:

```bash
sh spar.sh
```

## Pretrained Models

SPAR models pretrained on 25K SA-1B images:
<br>

<table>
  <tr>
    <th>Model</th>
    <th>Backbone</th>
    <th>Pretraining Dataset</th>
    <th>Download</th>
  </tr>

  <tr>
    <td>SPAR SigLIP2</td>
    <td>ViT-B</td>
    <td>SA-1B (25K images)</td>
    <td><a href="https://drive.google.com/file/d/1G6qfZpOZGhkM48NuibA3-S3IEqAWNObB/view?usp=sharing">link</a></td>
  </tr>

  <tr>
    <td>SPAR ALL SigLIP2</td>
    <td>ViT-B</td>
    <td>SA-1B (25K images)</td>
    <td><a href="https://drive.google.com/file/d/1pTVcVmC_9aQ2c8Tbv2aNbtrxAjed7tBw/view?usp=sharing">link</a></td>
  </tr>

  <tr>
    <td>SPAR MaskCLIP</td>
    <td>ViT-B</td>
    <td>SA-1B (25K images)</td>
    <td><a href="https://drive.google.com/file/d/1wLqX4Pu58f7g8ktOpI83Nj4zam46czyQ/view?usp=sharing">link</a></td>
  </tr>

  <tr>
    <td>SPAR DINOv3-TXT</td>
    <td>ViT-L</td>
    <td>SA-1B (25K images)</td>
    <td><a href="https://drive.google.com/file/d/1s5G2-qBvPxXwTN61xNDiciYxcG7B3GYH/view?usp=sharing">link</a></td>
  </tr>
</table>


The above SPAR MaskCLIP model uses an updated preprocessing configuration, while all other model weights remain as reported. For completeness, we also provide the original SPAR MaskCLIP weights used in the paper.

<table>
  <tr>
    <th>Model</th>
    <th>Backbone</th>
    <th>Training Data</th>
    <th>Weights</th>
    <th>Notes</th>
  </tr>

  <tr>
    <td>SPAR MaskCLIP (Legacy)</td>
    <td>ViT-B</td>
    <td>SA-1B (25K images)</td>
    <td><a href="https://drive.google.com/file/d/11MABz9Dr3ZI7pzP6p-khQf69Wk43NDne/view?usp=sharing">link</a></td>
    <td>Paper weigths trained with legacy preprocessing</td>
  </tr>
</table>

Checkpoint evaluation is done with:
```bash
python eval.py -model DOWNLOADED_MODEL_TYPE -dataset cfg_DATASET -checkpoint PATH_TO_DOWNLOADED_WEIGHTS
```

## Inference Demo

We provide `inference_demo.py` for running SPAR models on a single image for quick qualitative evaluation. It outputs a colorized segmentation map with class predictions overlaid on the original image. We provide a Cityscapes validation image for convenience.

```bash
python inference_demo.py -model MODEL_NAME -checkpoint PATH_TO_CHECKPOINT
```

## Citation

```
@article{kombol2026spar,
  title={SPAR: Single-Pass Any-Resolution ViT for Open-vocabulary Segmentation},
  author={Kombol, Naomi and Martinovi{\'c}, Ivan and {\v{S}}egvi{\'c}, Sini{\v{s}}a and Tolias, Giorgos},
  journal={arXiv preprint arXiv:2604.02252},
  year={2026},
  note={Accepted to CVPR 2026}
}
```

## Acknowledgments

This repository is based on ["SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference"](https://github.com/wangf3014/SCLIP). Thanks to the authors!
