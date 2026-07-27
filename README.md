<div align="center">

# YOLOv3 Embedding Inversion

### Can you reconstruct an image from object detection output?

<img src="comparison.png" width="100%">

**Yes.** YOLOv3's detection embedding -- 904,995 numbers with **zero color or pixel data** -- can be inverted to recover recognizable images.

| Metric | Value | Meaning |
|--------|-------|---------|
| **SSIM** | 0.42 | Structural similarity (1.0 = perfect) |
| **PSNR** | 14.8 dB | Pixel-level accuracy (higher = better) |
| **Similarity** | 72% | Human-friendly perceptual match |
| **Embedding size** | 904,995 values | 3 anchors x (5 box + 80 class) per grid cell |
| **Image size** | 519,168 values | 416 x 416 x 3 RGB |

</div>

---

## The Problem

<div align="center">

**Detection outputs are assumed to be anonymous. They are not.**

</div>

When you run YOLOv3 on an image, it outputs three grids of detection features:

```
head13:  255 x 13 x 13  =   43,095 values  (large objects)
head26:  255 x 26 x 26  =  172,380 values  (medium objects)
head52:  255 x 52 x 52  =  689,520 values  (small objects)
                                   --------
                                   904,995 values total
```

Each cell has **255 numbers** = `3 anchors x (5 box params + 80 COCO classes)`. These encode:
- **Objectness**: "is something here?"
- **Box geometry**: x, y, width, height
- **Class probabilities**: person, car, dog, etc.

**No color. No texture. No pixels.** Just "what objects are where."

Yet a trained decoder can reconstruct a recognizable image from this alone.

---

## The Evidence

<div align="center">

### Original vs Reconstruction

<img src="side_by_side_000000000036.png" width="90%">

<img src="side_by_side_000000000077.png" width="90%">

<img src="side_by_side_000000525732.png" width="90%">

*Each pair: left = original image, right = reconstruction from YOLOv3 detection features only.*

</div>

---

## How It Works

<div align="center">

```
Original Image                    Frozen YOLOv3                    Trained Decoder
   416x416 RGB    --->    Detection Embedding    --->    Reconstructed Image
  519,168 pixels         904,995 features              416x416 RGB
                         (no color data)
```

</div>

The decoder has **7,316,387 parameters** and combines three architectural innovations:

### 1. GENEO Layers
Group Equivariant Non-Expansive Operators that respect C4 rotation symmetry. These provide geometric structure priors -- the decoder "knows" that rotating an object shouldn't change its fundamental layout.

### 2. Transformer Bottleneck
A 4-layer transformer encoder processes 169 spatial tokens (the 13x13 grid), establishing global relationships between distant grid locations before image upsampling.

### 3. Progressive Upsampling
U-Net-like decoder with lateral connections from the embedding:
```
256 x 13x13 -> 160 x 26x26 -> 112 x 52x52 -> 80 x 104x104 -> 48 x 208x208 -> 32 x 416x416 -> 3 x 416x416 RGB
```

Output is in YCbCr color space, then converted to RGB. The decoder learns to hallucinate plausible colors from training data statistics.

---

## Results Across Decoder Versions

<div align="center">

<img src="plots/version_progression.png" width="80%">

| Version | What Changed | Avg SSIM | Avg PSNR |
|---------|-------------|----------|----------|
| v1 | Baseline CNN | 0.314 | 8.5 dB |
| v2 | + GENEO layers | 0.347 | 11.1 dB |
| **v3** | **+ Transformer** | **0.421** | **14.8 dB** |
| v6 | + YCbCr loss | 0.279 | 6.8 dB |

</div>

v3 (GENEO + transformer) achieves the best quality. Adding the transformer for global reasoning was the key breakthrough.

---

## Privacy Risk

<div align="center">

<img src="plots/privacy_risk.png" width="70%">

</div>

Our results fall in the **moderate privacy risk** zone. SSIM above 0.2 means recognizable structure is recoverable. An attacker with:
1. Access to detection outputs (the embedding)
2. A trained decoder (this repo)

...can reconstruct approximate visual content of the original image.

**The embedding is 174% the size of the raw image** (904,995 vs 519,168 values). The privacy risk comes from the *structure* of the representation, not its size.

---

## Comparison With Literature

<div align="center">

<img src="plots/literature_comparison.png" width="80%">

</div>

This project was built **independently** before discovering the existing literature. Similar attacks exist across domains:

| Paper | Domain | Key Result |
|-------|--------|------------|
| Song & Raghunathan (2020) | Text embeddings | Recovered 50-70% of tokens |
| Morris et al. (2023) | Text embeddings | Recovered 92% of 32-token text |
| Mai et al. IdDecoder (2023) | Face embeddings | Reconstructed realistic faces |
| Kaissis et al. (2021) | Feature descriptors | RGB from SIFT/FREAK (SSIM 0.51-0.68) |
| Dosovitskiy & Brox (2016) | Visual features | Inverted DNN representations |
| Wang et al. (2024) | Detection transformers | Inverted DETR features |
| FIA-Flow (2025) | Multi-architecture | Black-box inversion with flow matching |

**Our contribution**: specific application to YOLOv3 with GENEO geometric operators. Modern approaches (diffusion priors, FIA-Flow) would likely achieve higher fidelity with less custom architecture.

---

## Run It Yourself

### Setup

```powershell
cd model
pip install -r requirements.txt
```

### Extract an embedding from any image

```powershell
python extract_embedding.py --image images\000000000036.jpg --output embedding.npz
```

### Reconstruct the image

```powershell
python reconstruct_embedding.py --embedding embedding.npz --checkpoint runs\geneo_ycbcr_v3_best.pt --output reconstruction.png
```

### Or use a pre-extracted embedding

```powershell
python reconstruct_embedding.py --embedding embeddings\test_emb_000000000036.npz --checkpoint runs\geneo_ycbcr_v3_best.pt --output recon.png
```

### Regenerate plots

```powershell
cd ..\scripts
python make_presentation.py
python make_plots.py
```

---

## Folder Structure

```
presentation/
├── README.md                   You are here
├── comparison.png              Main figure (3 rows: original | embedding | recon)
├── explanation.png             Research infographic with citations
├── side_by_side_*.png          Individual comparison pairs
├── metrics.csv                 Reconstruction metrics
├── model/                      Complete runnable pipeline
│   ├── inverse_decoder.py      Decoder (GENEO + transformer + upsampler)
│   ├── geneo_layer.py          GENEO operator implementation
│   ├── darknet_v3.py           Original Darknet YOLOv3 loader
│   ├── extract_embedding.py    Image -> embedding
│   ├── reconstruct_embedding.py  Embedding -> image
│   ├── yolov3.weights          Original Darknet weights (248 MB)
│   ├── yolov3.cfg              Original Darknet config
│   ├── runs/geneo_ycbcr_v3_best.pt  Trained decoder (85 MB)
│   ├── images/                 5 sample COCO images
│   ├── embeddings/             5 pre-extracted .npz files
│   └── requirements.txt
├── plots/                      Publication-quality figures
│   ├── version_progression.png     SSIM/PSNR across decoder versions
│   ├── privacy_risk.png            Scatter with risk zones
│   ├── literature_comparison.png   Our results vs published work
│   ├── reconstruction_gallery.png  Best recon per image
│   ├── ssim_by_version.png         Per-image SSIM bars
│   ├── psnr_by_version.png         Per-image PSNR bars
│   ├── embedding_vs_image.png      Info capacity comparison
│   └── all_metrics.csv             All 18 measurements
└── scripts/                    Generation scripts
    ├── make_presentation.py
    └── make_plots.py
```

---

## Limitations

1. **Small test set** (5 images). Larger evaluation needed for statistical significance.
2. **COCO persons only**. Generalization to other classes/datasets is untested.
3. **Modern approaches exist**. Diffusion priors (FIA-Flow 2025) would achieve higher fidelity with less custom architecture.
4. **No defense tested**. Differential privacy, noise injection, or feature masking could reduce inversion quality.

---

## Citation

```bibtex
@misc{yolov3_inversion,
  title  = {YOLOv3 Embedding Inversion: Privacy Risk of Detection Features},
  author = {Carson},
  year   = {2025},
  note   = {Independent implementation. GENEO layers based on algebraic
            representation theorem for linear GENEOs.}
}
```

---

<div align="center">

**Detection features are not anonymous. The structure is the leak.**

</div>
