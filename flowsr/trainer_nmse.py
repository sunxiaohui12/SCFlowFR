import wandb
import einops
import warnings
from PIL import Image

import torch
from torch import Tensor
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from einops import rearrange

from torch.nn import functional as F
from flowsr.helpers import freeze
from flowsr.helpers import exists
from flowsr.helpers import un_normalize_ims
from flowsr.helpers import instantiate_from_config
from flowsr.helpers import load_partial_from_config


def make_grid(*log_images):
    rearranged_images = [einops.rearrange(img, "b c h w -> h (b w) c") for img in log_images]
    grid = torch.cat(rearranged_images, dim=1)
    # normalize to [0, 255]

    # print("grid", grid.min(), grid.max())
    grid = un_normalize_ims(grid).cpu().numpy()
    return grid


class NMSETrainer(LightningModule):
    def __init__(
        self,
        model_cfg: dict = None,
        first_stage_cfg: dict = None,
        scale_factor: int = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        n_images_to_vis: int = 16,
        metric_tracker_cfg: dict = None,
        lr_scheduler_cfg: dict = None,
        log_grad_norm: bool = False,
    ):
        """
        Args:
            model_cfg: Model config.
            first_stage_cfg: First stage config, if None, identity is used.
            scale_factor: Scale factor for the latent space (normalize the
                latent space, default value for SD: 0.18215).
            lr: Learning rate.
            weight_decay: Weight decay.
            n_images_to_vis: Number of images to visualize.
            noising_step: Forward diffusion noising step with linear schedule
                of Ho et al. Set to -1 to disable.
            concat_context: Whether to concatenate the low-res images as conditioning.
            ca_context: Whether to use cross-attention context.
            first_stage_cfg: First stage config, if None, identity is used.
            scale_factor: Scale factor for the latent space (normalize the
                latent space, default value for SD: 0.18215).
            lr: Learning rate.
            weight_decay: Weight decay.
            n_images_to_vis: Number of images to visualize.
            ema_rate: EMA rate.
            ema_update_every: EMA update rate (every n steps).
            ema_update_after_step: EMA update start after n steps.
            use_ema_for_sampling: Whether to use the EMA model for sampling.
            metric_tracker_cfg: Metric tracker config.
            lr_scheduler_cfg: Learning rate scheduler config.
            log_grad_norm: Whether to log the gradient norm.
        """
        super().__init__()
        self.model = instantiate_from_config(model_cfg)

        # first stage encoding
        self.scale_factor = scale_factor
        if exists(first_stage_cfg):
            self.first_stage = instantiate_from_config(first_stage_cfg)
            freeze(self.first_stage)
            self.first_stage.eval()
            if self.scale_factor == 1.0:
                warnings.warn("Using first stage with scale_factor=1.0")
        else:
            if self.scale_factor != 1.0:
                raise ValueError("Cannot use scale_factor with identity first stage")
            self.first_stage = None

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.log_grad_norm = log_grad_norm

        self.vis_samples = None
        self.metric_tracker = (
            instantiate_from_config(metric_tracker_cfg)
            if exists(metric_tracker_cfg)
            else None
        )

        self.n_images_to_vis = n_images_to_vis
        self.val_epochs = 0

        self.save_hyperparameters()

        # flag to make sure the signal is not handled at an incorrect state, e.g. during weights update
        self.stop_training = False

    def stop_training_method(self):
        # dummy function to be compatible
        pass

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        out = dict(optimizer=opt)
        if exists(self.lr_scheduler_cfg):
            sch = load_partial_from_config(self.lr_scheduler_cfg)
            sch = sch(optimizer=opt)
            out["lr_scheduler"] = sch
        return out

    def forward(self, x_source: Tensor, **kwargs):
        return self.model(x_source, **kwargs)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if not exists(self.first_stage):
            return x
        x = self.first_stage.encode(x)
        if not isinstance(x, torch.Tensor):  # hack for posterior of original VAE
            x = x.mode()
        return x * self.scale_factor

    # @torch.no_grad()
    def decode_first_stage(self, z):
        if not exists(self.first_stage):
            return z
        return self.first_stage.decode(z / self.scale_factor)

    def extract_from_batch(self, batch):
        """
        Takes batch and extracts high-res and low-res images and latent codes.

        Returns:
            hres_ims: high-res images
            hres_z: high-res latent codes (if identity first stage, this is hres_ims)
            lres_ims: low-res images
            lres_z: low-res latent codes (if identity first stage, this is lres_ims)
        """
        hr, lr = batch['hr'], batch['lr']
        hr = hr * 2.0 - 1.0  # to [-1, 1]
        lr = lr * 2.0 - 1.0  # to [-1, 1]

        with torch.no_grad():
            hres_z = self.encode_first_stage(hr)
            lres_z = self.encode_first_stage(lr)

        return hr.float(), hres_z.float(), lr.float(), lres_z.float()

    def training_step(self, batch, batch_idx):
        """extract high-res and low-res images from batch"""
        hres_ims, hres_z, lres_ims, lres_z = self.extract_from_batch(batch)

        """ loss """
        hres_z_pred = self.forward(x_source=lres_z)
        # with torch.no_grad():
        hres_pred = self.decode_first_stage(hres_z_pred)
        loss_1 = F.l1_loss(hres_pred, hres_ims)
        loss_2 = F.l1_loss(hres_z_pred, hres_z)
        loss = loss_1 + loss_2
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=lres_z.shape[0],
        )

        """ misc """
        if exists(self.lr_scheduler_cfg):
            self.lr_schedulers().step()
        if self.stop_training:
            self.stop_training_method()
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        hr_ims, hr_z, lr_ims, lr_z = self.extract_from_batch(batch)

        _vis_samples = {"hr": hr_ims, "lr": lr_ims}

        hr_z_pred = self.forward(x_source=lr_z)
        hr_pred = self.decode_first_stage(hr_z_pred)

        _vis_samples[f"hr_pred"] = hr_pred

        # track metrics
        if exists(self.metric_tracker):
            self.metric_tracker(hr_ims, hr_pred)

        if self.stop_training:
            self.stop_training_method()

        # store samples for visualization
        if self.vis_samples is None:
            self.vis_samples = _vis_samples
        elif self.vis_samples["hr"].shape[0] < self.n_images_to_vis:
            for key, val in self.vis_samples.items():
                self.vis_samples[key] = torch.cat([val, _vis_samples[key]], dim=0)

    def on_validation_epoch_end(self):
        # log low-res images, high-res images, and up-sampled images

        for key, val in self.vis_samples.items():
            if val.shape[0] > self.n_images_to_vis:
                self.vis_samples[key] = val[: self.n_images_to_vis]
            self.log_image(make_grid(val), key)

        self.vis_samples = None

        # compute metrics
        if exists(self.metric_tracker):
            metrics = self.metric_tracker.aggregate()
            for k, v in metrics.items():
                self.log(f"val/{k}", v, sync_dist=True)
            self.metric_tracker.reset()

        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs} | Optimizer step {self.global_step}")

        torch.cuda.empty_cache()

    def log_image(self, img, name):
        """
        Args:
            ims: torch.Tensor or np.ndarray of shape (h, w, c) in range [0, 255]
            name: str
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(self.logger, WandbLogger):
            img = Image.fromarray(img)
            img = wandb.Image(img)
            self.logger.experiment.log({f"{name}/samples": img}, step=self.global_step)
        else:
            img = einops.rearrange(img, "h w c -> c h w")
            self.logger.experiment.add_image(
                f"{name}/samples", img, global_step=self.global_step
            )


def get_grad_norm(model):
    total_norm = 0
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm
