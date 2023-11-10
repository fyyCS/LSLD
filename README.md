# Revisit Weakly-Supervised Audio-Visual Video Parsing from the Language Perspective
Yingying Fan, [Yu Wu](http://yu-wu.net/), Bo Du and [Yutian Lin](https://vana77.github.io/)

Code for NeurIPS 2023 paper [Revisit Weakly-Supervised Audio-Visual Video Parsing from the Language Perspective](https://arxiv.org/abs/2306.00595)

## Method Overview

![](https://github.com/fyyCS/LSLD/blob/main/fig/model.jpeg)

## Environment

You should install [CLIP](https://github.com/openai/CLIP) and [LAION-CLAP](https://github.com/LAION-AI/CLAP)
and run
```python
pip install -r requirement.txt
```

## Prepare data

1. Resnet and VGGish features can be downloaded from [Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing](https://github.com/YapengTian/AVVP-ECCV20).
We also provide [visual feature](https://drive.google.com/drive/folders/1JcOT_Pm17MAb-wQERXOCLQqoyT9nObJL?usp=sharing) extracted by CLIP and [audio feature](https://drive.google.com/file/d/12U0DFxbZG0IQTbsXFi5XTYi87z59Zlg7/view?usp=drive_link) extracted by LAION-CLAP.
2. Put the downloaded features into data/feats/.
3. We use CLIP(ViT-B/16) and [LAION-CLAP](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-best.pt) pre-trained on audioset.

## Label Denoising
```script
python main.py --mode label_denoise --language refine_label/denoised_label.npz --refine_label refine_label/final_label.npz
```

## Train the model

1. Resnet and VGGish features
```script
python main.py --mode train_model --num_layers 6 --lr 8e-5 --refine_label refine_label/final_label.npz --save_model true --checkpoint LSLD.pt
```
2. CLAP and CLIP features (Recommended)
```script
python main.py --mode train_model --num_layers 4 --lr 2e-4 --refine_label refine_label/final_label.npz --save_model true --checkpoint LSLD.pt
```

## Test the model
```script
python main.py --mode test_LSLD --checkpoint LSLD.pt
```
## Citation
```script
@article{fan2023revisit,
  title={Revisit Weakly-Supervised Audio-Visual Video Parsing from the Language Perspective},
  author={Fan, Yingying and Wu, Yu and Du, Bo and Lin, Yutian },
  journal={arXiv preprint arXiv:2306.00595},
  year={2023}
}
```




