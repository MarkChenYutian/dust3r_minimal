# DUSt3R Minimal Inference

This repository provides the minimal code necessary to run inference with [DUSt3R](https://arxiv.org/abs/2312.14132). It includes the model implementation, the associated CroCo modules and a small demo script.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

(Optional) compile the optimized kernels used by CroCo:

```bash
cd croco/models/curope
python setup.py build_ext --inplace
cd ../../../
```

## Example usage

```python
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

model = AsymmetricCroCo3DStereo.from_pretrained(
    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
).to('cuda')
images = load_images([
    'croco/assets/Chateau1.png',
    'croco/assets/Chateau2.png'
], size=512)
pairs = make_pairs(images, scene_graph='complete', symmetrize=True)
output = inference(pairs, model, 'cuda')

scene = global_aligner(output, device='cuda',
                       mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init='mst', niter=300,
                               schedule='cosine', lr=0.01)
scene.show()
```

You can also start a simple Gradio UI with:

```bash
python demo.py --model_name DUSt3R_ViTLarge_BaseDecoder_512_dpt
```

## License

The code is distributed under the CC BY-NC-SA 4.0 license. See [LICENSE](LICENSE) for details.
