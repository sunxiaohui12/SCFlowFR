
# Linearized Coupling Flow with Shortcut Constraints for One-Step Face Restoration

| Resource | Link |
|----------|------|
| arXiv (abstract) | [https://arxiv.org/abs/2603.03648](https://arxiv.org/abs/2603.03648) |
| arXiv (PDF) | [https://arxiv.org/pdf/2603.03648.pdf](https://arxiv.org/pdf/2603.03648.pdf) |

This repo builds on **FlowSR-style** flow matching with shortcut constraints for face restoration (see `flowsr/` for models, training loop, and configs).

---

## Released checkpoints (two model sizes)

We provide **two checkpoints** that match the paper experiments but differ in **backbone capacity**: the **latent UNet** (`EfficientShortcutUnet`) and the **first-stage encoder / decoder** (VAE).

| | **Large** | **Tiny** |
|---|-----------|----------|
| **Role** | Higher quality, heavier compute | Faster / lighter, smaller VRAM |
| **Shortcut Flow** | `ShortcutFlowModel` (`segment_K: 128`, `boostrap_every: 4`, `schedule: linear`) — same as Tiny |
| **UNet (`net_cfg`)** | Deeper: `channel_mult: [1, 2, 4, 8]`, 4 levels | Lighter: `channel_mult: [1, 2, 4]`, 3 levels |
| **First stage (latent space)** | `AutoencoderKL` — `pretrained-models/sd_ae.ckpt` | `TAESD` (Tiny VAE) — `taesd_encoder.pth` / `taesd_decoder.pth` |
| **`scale_factor`** | `0.18215` (SD-style latent scaling) | `1.0` |
| **Training steps (released)** | **200k** | **150k** |
| **Config template** | `shortcutfm_face_codeformer.yaml` (KL-VAE + large UNet) | `shortcutfm_face_tiny.yaml` |
| **Download** | *To be added (e.g. Hugging Face / Google Drive) — 200k step checkpoint* | *To be added — 150k step checkpoint* |

SwinIR context path in both training configs: `pretrained-models/swinir/face_swinir_v1.ckpt` (you still need this file locally for training; for inference the default script uses a SwinIR path you can set in `inference.py`).

> **Note:** `inference.py` is written around the **Tiny** stack (TAESD + `scale_factor: 1.0`). To run the **Large** model, point `config_path` to the **Large** run’s `config.yaml` and matching checkpoint, and ensure the script’s VAE loader matches `first_stage_cfg` (KL + `scale_factor` 0.18215) — you may need to align `load_vae` with `AutoencoderKL` if it currently assumes TAESD only.

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

Download SwinIR / VAE weights according to the **variant** you use (`swinir_path`, VAE paths, `scale_factor` in YAML).

---

## Data preparation (training)

1. **Prepare HR images** (e.g. FFHQ-style face crops) and a **file list** (one image path per line).
2. Point the dataset config to your list and backend, e.g. in `configs/shortcutfm_face_tiny.yaml` or the Large variant config:

   - `data.params.train.params.file_list` → training list  
   - `data.params.validation.params.file_list` → validation list  
   - `file_backend_cfg` → `HardDiskBackend` and correct root if needed.

3. Create parent directories as needed, e.g. `load/FFHQ/`, and place `train.list` / `valid.list` (or your names) there.

4. **Batch size & workers**: tune `data.params.batch_size` and `num_workers` for your GPU memory (**Large** needs more VRAM than **Tiny**).

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

`configs/` includes shortcut-flow face settings, e.g.:

- **`shortcutfm_face_tiny.yaml`** — **Tiny** UNet + **TAESD** (matches **150k** released checkpoint).  
- **`shortcutfm_face_codeformer.yaml`** — **Large** UNet + **KL VAE** (matches **200k** released checkpoint).

Shared flow head: `flowsr.flow.ShortcutFlowModel` with `segment_K: 128`, `boostrap_every: 4`.

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


