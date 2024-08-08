# Text To Texture: Fashion Image Generation from Text Descriptions

## Overview

This project aims to generate fashion images from textual descriptions by implementing the state-of-the-art AttnGAN model. The AttnGAN framework leverages fine-grained attention mechanisms to produce high-resolution, detailed images based on text inputs.

## Model and Framework

**AttnGAN (Python 3, Pytorch 1.0)**

AttnGAN (Attentional Generative Adversarial Networks) is designed for fine-grained text-to-image generation. This implementation follows the methodology outlined in the paper "AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks" by Tao Xu et al.

## Dependencies

- **Python**: Version 3.6 or higher
- **Pytorch**: Version 1.0 or higher

### Additional Packages

Ensure the project folder is added to the `PYTHONPATH` and install the following packages:

```bash
pip install python-dateutil easydict pandas torchfile nltk scikit-image
```

## Data

The project utilizes the DeepFashion MultiModal dataset, a widely recognized dataset in the fashion industry. This dataset supports multiple benchmarks, including Attribute Prediction, Consumer-to-shop Clothes Retrieval, In-shop Clothes Retrieval, and Landmark Detection.

- **Dataset Link**: [DeepFashion MultiModal dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

## Training Process

### Pre-train DAMSM Models

To pre-train the Deep Attentional Multimodal Similarity Model (DAMSM):

```bash
python pretrain_DAMSM.py --cfg cfg/DAMSM/fashion.yml --gpu 0
```

### Train AttnGAN Models

To train the AttnGAN model:

```bash
python main.py --cfg cfg/fashion_attn2.yml --gpu 0
```

Example configuration files for training and evaluation are provided in the `*.yml` files.

## Sampling

To generate images from captions:

```bash
python main.py --cfg cfg/eval_fashion.yml --gpu 0
```

To generate images from custom sentences, modify the file `./data/fashion/example_captions.txt` with your sentences.

## Validation

To generate images for all captions in the validation dataset, set `B_VALIDATION` to `True` in the `eval_*.yml` file, and then run:

```bash
python main.py --cfg cfg/eval_fashion.yml --gpu 0
```

### Inception Score Calculation

- For models trained on birds: Use `StackGAN-inception-model`
- For models trained on COCO: Use `improved-gan/inception_score`

## API Creation

The evaluation code is embedded in a callable containerized API available in the `eval` folder.

## References

- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://github.com/hanzhanggit/StackGAN-v2)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://github.com/carpedm20/DCGAN-tensorflow)
