
# Linearized Coupling Flow with Shortcut Constraints for One-Step Face Restoration

| arXiv (PDF) | [https://arxiv.org/pdf/2603.03648.pdf](https://arxiv.org/pdf/2603.03648.pdf) |

**Weights (Baidu Netdisk):** folder **SCFlowSR** contains **`pretrained_models`** (SwinIR, VAE, etc.) and **`checkpoints`** (**Large** + **Tiny**). Link: [https://pan.baidu.com/s/1cfXHCQewKDrWrxCyZf1TUg?pwd=t6jf](https://pan.baidu.com/s/1cfXHCQewKDrWrxCyZf1TUg?pwd=t6jf) · extraction code: `t6jf`

---

## Environment setup

### 1. Python & CUDA

- Recommended: **Python ≥ 3.10**, **CUDA** matching your PyTorch build (see `requirements.txt` for a CUDA 11.8 example).

### 2. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
pip install torchdiffeq wandb  # if not already satisfied by your stack
```

> Note: `requirements.txt` pins `torch`/`torchvision` from the PyTorch CUDA index (`cu118`). Adjust the `--extra-index-url` and package versions if your driver uses another CUDA version.

### 3. Pretrained assets (training / inference)

Typical layout (paths often referenced in `configs/*.yaml`):

```
FlowSR-lightning/
├── pretrained-models/
│   ├── swinir/              # e.g. face_swinir_v1.ckpt
│   ├── tiny_vae/            # TAESD encoder/decoder (Tiny variant)
│   └── sd_ae.ckpt           # KL AutoencoderKL (Large variant)
└── load/                     # datasets & file lists (see below)
```

Download SwinIR / VAE weights according to the **variant** you use (`swinir_path`, VAE paths, `scale_factor` in YAML), or unpack **`pretrained_models`** from the Baidu **SCFlowSR** folder (same link as at the top of this README).

---

## Data preparation (training)

### `train.list` / `valid.list` format

Both are **plain text**: **one image path per line** (UTF-8). Paths are usually **absolute**; each line is read as one HR training image (`CodeformerDataset`). Blank lines are skipped.

Example:

```
/path/to/ffhq/00001.png
/path/to/ffhq/00002.png
```

### How to build the lists

- **By hand:** Collect HR faces (e.g. FFHQ crops), then write `train.list` / `valid.list` yourself with any editor.
- **Helper script:** From repo root, `scripts/make_file_list.py` walks `--img_folder` for `.jpg`/`.png`/`.jpeg`, writes **`train.list`** (all images) and **`valid.list`** (the first `--val_size` paths only). By default training and validation sets **overlap** on those first `val_size` images; for a disjoint split, write your own lists or adjust the script.

```bash
python scripts/make_file_list.py \
  --img_folder /path/to/hr_images \
  --save_folder load/FFHQ \
  --val_size 1000
```

Writes `load/FFHQ/train.list` and `load/FFHQ/valid.list`.

### Config

In `configs/shortcutfm_face_tiny.yaml` (or the Large config), set:

- `data.params.train.params.file_list` → `load/FFHQ/train.list` (or your path)  
- `data.params.validation.params.file_list` → `load/FFHQ/valid.list`  
- `file_backend_cfg` → `HardDiskBackend` if files are on local disk.

**Batch size & workers:** tune `data.params.batch_size` and `num_workers` for your GPU (**Large** needs more VRAM than **Tiny**).

---

## Training

Entry point: **`train.py`**.

```bash
cd /path/to/FlowSR-lightning

# Tiny variant (TAESD + lighter UNet)
python train.py --config configs/shortcutfm_face_tiny.yaml --name shortcutfm_face_tiny

# Large variant (KL VAE + deeper UNet)
python train.py --config configs/shortcutfm_face_codeformer.yaml --name shortcutfm_face_large
```

Useful options:

| Option | Description |
|--------|-------------|
| `--name` | Experiment name; logs go under `logs/<name>/exp_<timestamp>/` |
| `--devices` | GPU count, e.g. `1` for single GPU (default may use all) |
| `--resume_checkpoint` | Full resume (optimizer + step) from a `.ckpt` |
| `--load_weights` | Load weights only (no optimizer state) |
| `--use_wandb` / `--use_wandb_offline` | Weights & Biases logging |

Override any config value with OmegaConf dotlist, e.g.:

```bash
python train.py --config configs/shortcutfm_face_tiny.yaml --name ablation model.params.lr=2e-5
```

Checkpoints and a resolved `config.yaml` are saved under `logs/.../checkpoints/` and the run folder.

---

## Inference

We release **two pretrained checkpoints** (different model sizes): pair **`shortcutfm_face_codeformer.yaml`** with the **Large** weight (200k steps), and **`shortcutfm_face_tiny.yaml`** with the **Tiny** weight (150k steps). Get both under **`checkpoints`** in the Baidu **SCFlowSR** folder (link at the top of this README). Use the **`config.yaml`** and **`.ckpt`** from the same training run (or matching released pair).

The script **`inference.py`** loads **ShortcutFlowModel** + EMA + **VAE** + **SwinIR** from a **training `config.yaml`** and a **Lightning `.ckpt`** (see `load_model_from_config`).

1. **Edit the paths** at the bottom of `inference.py`:

   - **`config_path`** — must match the variant: **Tiny** run uses TAESD + `scale_factor: 1.0`; **Large** run uses KL VAE + `scale_factor: 0.18215`.  
   - **`checkpoint_path`** — e.g. `step150000.ckpt` (Tiny release) or `step200000.ckpt` (Large release).  
   - **`swinir_checkpoint_path`** — SwinIR weights (required).  
   - **`lr_dataset_dir`** / **`output_dir`** — input images and save folder.

2. Run:

```bash
cd /path/to/FlowSR-lightning
python inference.py
```

If `load_vae` only supports TAESD, extend it for **Large** inference with `AutoencoderKL` + correct scaling when you switch configs.

**Other tools:** `auto_benchmark.py` / `benchmark_v2.py` for batch evaluation once paths are set.

---

## Config overview

Training configs live under `configs/` (e.g. `shortcutfm_face_tiny.yaml`, `shortcutfm_face_codeformer.yaml`). Flow model: `flowsr.flow.ShortcutFlowModel` with `segment_K: 128`, `boostrap_every: 4`.

---

## Citation

If you use this code or the method, please cite the paper (update with official venue when available):

```bibtex
@article{scflowfr2026,
  title   = {Linearized Coupling Flow with Shortcut Constraints for One-Step Face Restoration},
  journal = {arXiv preprint arXiv:2603.03648},
  year    = {2026}
}
```

---

## License

See [LICENSE](LICENSE) in this repository.


