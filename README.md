<!-- <div align="center">
  <img src="https://github.com/Roblox/SmoothCache/blob/main/assets/TeaserFigureFlat.png" width="100%" ></img>
  <br>
  <em>
      (Accelerating Diffusion Transformer inference across multiple modalities with 50 DDIM Steps on DiT-XL-256x256, 100 DPM-Solver++(3M) SDE steps for a 10s audio sample (spectrogram shown) on Stable Audio Open, 30 Rectified Flow steps on Open-Sora 480p 2s videos) 
  </em>
</div>
<br> -->

![Accelerating Diffusion Transformer inference across multiple modalities with 50 DDIM Steps on DiT-XL-256x256, 100 DPM-Solver++(3M) SDE steps for a 10s audio sample (spectrogram shown) on Stable Audio Open, 30 Rectified Flow steps on Open-Sora 480p 2s videos](assets/TeaserFigureFlat.png)

**Figure 1. Accelerating Diffusion Transformer inference across multiple modalities with 50 DDIM Steps on DiT-XL-256x256, 100 DPM-Solver++(3M) SDE steps for a 10s audio sample (spectrogram shown) on Stable Audio Open, 30 Rectified Flow steps on Open-Sora 480p 2s videos**


# Updates

## Release v0.1

[View release notes for v0.1](https://github.com/Roblox/SmoothCache/releases/tag/v0.1)

SmoothCache now supports generating cache schedues using a zero-intrusion external helper. See [run_calibration.py](./examples/run_calibration.py) to find out how it generates a schedule compatible with [HuggingFace Diffusers DiTPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/dit/pipeline_dit.py), without requiring any changes to Diffusers implementation!


# Introduction
We introduce **SmoothCache**, a straightforward acceleration technique for DiT architecture models, that's both **training-free, flexible and performant**. By leveraging layer-wise representation error, our method identifies redundancies in the diffusion process, generates a static caching scheme to reuse output featuremaps and therefore reduces the need for computationally expensive operations. This solution works across different models and modalities, can be easily dropped into existing Diffusion Transformer pipelines, can be stacked on different solvers, and requires no additional training or datasets. **SmoothCache** consistently outperforms various solvers designed to accelerate the diffusion process, while matching or surpassing the performance of existing modality-specific caching techniques.

> 🥯[[Arxiv]](https://arxiv.org/abs/2411.10510)

![Illustration of SmoothCache. When the layer representation loss obtained from the calibration pass is below some threshold α, the corresponding layer is cached and used in place of the same computation on a future timestep. The figure on the left shows how the layer representation error impacts whether certain layers are eligible for caching. The error of the attention (attn) layer is higher in earlier timesteps, so our schedule caches the later timesteps accordingly. The figure on the right shows the application of the caching schedule to the DiT-XL architecture. The output of the attn layer at time t − 1 is cached and re-used in place of computing FFN t − 2, since the corresponding error is below α. This cached output is introduced in the model using the properties of the residual connection.](assets/SmoothCache2.png)

## Quick Start

### Install
```bash
pip install SmoothCache
```

### Usage - Inference

Inspired by [DeepCache](https://raw.githubusercontent.com/horseee/DeepCache), we have implemented drop-in SmoothCache helper classes that easily applies to [Huggingface Diffuser DiTPipeline](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/dit), and [original DiT implementations](https://github.com/facebookresearch/DiT).

Generally, only 3 additional lines needs to be added to the original sampler scripts:
```python
from SmoothCache import <DESIREDCacheHelper>
cache_helper = DiffuserCacheHelper(<MODEL_HANDLER>, schedule=schedule)
cache_helper.enable()
# Original sampler code.
cache_helper.eisable()
```

#### Usage example with Huggingface Diffuser DiTPipeline:
```python
import json
import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

# Import SmoothCacheHelper
from SmoothCache import DiffuserCacheHelper  

# Load the DiT pipeline and scheduler
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Initialize the DiffuserCacheHelper with the model
with open("smoothcache_schedules/50-N-3-threshold-0.35.json", "r") as f:
    schedule = json.load(f)
cache_helper = DiffuserCacheHelper(pipe.transformer, schedule=schedule)

# Enable the caching helper
cache_helper.enable()
# Prepare the input
words = ["Labrador retriever"]
class_ids = pipe.get_label_ids(words)

# Generate images with the pipeline
generator = torch.manual_seed(33)
image = pipe(class_labels=class_ids, num_inference_steps=50, generator=generator).images[0]

# Restore the original forward method and disable the helper
# disable() should be paired up with enable() 
cache_helper.disable()
```

#### Usage example with original DiT implementation
```python
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from SmoothCache import DiTCacheHelper  # Import DiTCacheHelper
import json

# Setup PyTorch:
torch.manual_seed(args.seed)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.ckpt is None:
    assert (
        args.model == "DiT-XL/2"
    ), "Only DiT-XL/2 models are available for auto-download."
    assert args.image_size in [256, 512]
    assert args.num_classes == 1000

# Load model:
latent_size = args.image_size // 8
model = DiT_models[args.model](
    input_size=latent_size, num_classes=args.num_classes
).to(device)
ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict)
model.eval()  # important!
with open("smoothcache_schedules/50-N-3-threshold-0.35.json", "r") as f:
    schedule = json.load(f)
cache_helper = DiTCacheHelper(model, schedule=schedule)

# number of timesteps should be consistent with provided schedules
diffusion = create_diffusion(str(len(schedule[cache_helper.components_to_wrap[0]])))

# Enable the caching helper
cache_helper.enable()

# Sample images:
samples = diffusion.p_sample_loop(
    model.forward_with_cfg,
    z.shape,
    z,
    clip_denoised=False,
    model_kwargs=model_kwargs,
    progress=True,
    device=device,
)
samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
samples = vae.decode(samples / 0.18215).sample

# Disable the caching helper after sampling
cache_helper.disable()
# Save and display images:
save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
```

### Usage - Cache Schedule Generation
See [run_calibration.py](./examples/run_calibration.py), which generates schedule for the self-attention module ([attn1](https://github.com/huggingface/diffusers/blob/37a5f1b3b69ed284086fb31fb1b49668cba6c365/src/diffusers/models/attention.py#L380)) 
from Diffusers [BasicTransformerBlock](https://github.com/huggingface/diffusers/blob/37a5f1b3b69ed284086fb31fb1b49668cba6c365/src/diffusers/models/attention.py#L261C7-L261C28) block. 

Note that only self-attention, and not cross-attention, is enabled in the stock config of Diffusers [DiT module](https://github.com/huggingface/diffusers/blob/37a5f1b3b69ed284086fb31fb1b49668cba6c365/src/diffusers/models/transformers/dit_transformer_2d.py#L72-L73). We leave this behavior
as-is for the purpose of minimal intrusion. 

We welcome all contributions aimed at expending SmoothCache's model coverage and module coverage. 

## Visualization

### 256x256 Image Generation Task

![Mosaic Image](assets/dit-mosaic.png)



## Evaluation

### Image Generation with DiT-XL/2-256x256
![Table 1. Results For DiT-XL-256x256 on using DDIM Sampling.
Note that L2C is not training free](assets/table1.png)

### Video Generation with OpenSora
![Table 2. Results For OpenSora on Rectified Flow](assets/table2.png)

### Audio Generation with Stable Audio Open
![Table 3. Results For Stable Audio Open on DPMSolver++(3M) SDE on 3 datasets](assets/table3.png)


# License
SmoothCache is licensed under the [Apache-2.0](LICENSE) license.

## Bibtex
```
@misc{liu2024smoothcacheuniversalinferenceacceleration,
      title={SmoothCache: A Universal Inference Acceleration Technique for Diffusion Transformers}, 
      author={Joseph Liu and Joshua Geddes and Ziyu Guo and Haomiao Jiang and Mahesh Kumar Nandwana},
      year={2024},
      eprint={2411.10510},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.10510}, 
}
```