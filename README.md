# Revisit Weakly-Supervised Audio-Visual Video Parsing from the Language Perspective

Code for NeurIPS 2023 paper [Revisit Weakly-Supervised Audio-Visual Video Parsing from the Language Perspective](https://arxiv.org/abs/2306.00595)

# Method Overview

![](https://github.com/fyyCS/LSLD/blob/main/fig/model.jpeg)

# Environment

You should install [CLIP](https://github.com/openai/CLIP) and [LAION-CLAP](https://github.com/LAION-AI/CLAP)

# Prepare data

1. Resnet and VGGish features can be downloaded from [Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing](https://github.com/YapengTian/AVVP-ECCV20).
We also provide [visual feature]() extracted by CLIP and [audio feature]() extracted by LAION-CLAP.
2. Put the downloaded features into data/feats/.

# Label Denoising
  python main.py --mode label_denoise --language refine_label/denoised_label.npz --refine_label refine_label/final_label.npz

# Train the model







