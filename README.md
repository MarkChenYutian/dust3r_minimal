# DINOv2 Minimal Inference

This repository provides the dataset preprocessing scripts from the original DUSt3R project together with a minimal example to run [DINO v2](https://github.com/facebookresearch/dinov2) inference.

The preprocessing utilities remain under the original CC BY-NC-SA 4.0 license from NAVER. They allow preparing various public datasets used for research. See the scripts inside `datasets_preprocess/` for details.

The script `dinov2_inference.py` shows how to download a pretrained DINO v2 model from HuggingFace and extract features from input images.

## Installation

```bash
pip install -r requirements.txt
```

The first execution of the inference script will download the model weights automatically.

## Running inference

```bash
python dinov2_inference.py image1.jpg image2.jpg
```

The command prints the shape of the extracted feature maps for each image.

## License

The code in this repository is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license. See the `LICENSE` file for the full text.
