#!/usr/bin/env python3
"""Minimal DINO v2 inference script.

This utility loads a pretrained DINO v2 model from HuggingFace and
extracts features for a list of input images.
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def load_image(path, processor, device):
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"].to(device)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)

    features = []
    for img_path in args.images:
        tensor = load_image(Path(img_path), processor, device)
        with torch.no_grad():
            output = model(pixel_values=tensor)
        features.append(output.last_hidden_state.cpu())
        print(f"{img_path}: feature shape {features[-1].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DINO v2 inference on images")
    parser.add_argument("images", nargs="+", help="input image paths")
    parser.add_argument("--model", default="facebook/dinov2-base", help="HuggingFace model id")
    parser.add_argument("--cpu", action="store_true", help="force CPU execution")
    args = parser.parse_args()
    main(args)
