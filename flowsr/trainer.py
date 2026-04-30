import wandb
import einops
import warnings
from PIL import Image

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger
from einops import rearrange
from flowsr.ema import EMA
from flowsr.diffusion import ForwardDiffusion

from flowsr.helpers import freeze
from flowsr.helpers import resize_ims
from flowsr.helpers import exists
from flowsr.helpers import un_normalize_ims
from flowsr.helpers import instantiate_from_config
from flowsr.helpers import load_partial_from_config


def make_grid(*log_images):
    rearranged_images = [einops.rearrange(img, "b c h w -> h (b w) c") for img in log_images]
    grid = torch.cat(rearranged_images, dim=1)
    # normalize to [0, 255]
    grid = un_normalize_ims(grid).cpu().numpy()
    return grid


class FlowSRTrainer(LightningModule):
    def __init__(
        self,
        fm_cfg: dict,
        swinir_cfg: dict = None,
        swinir_path: str = None,
        start_from_noise: bool = False,
        noising_step: int = -1,
        concat_context: bool = False,
        ca_context: bool = False,
        first_stage_cfg: dict = None,
        scale_factor: int = 1.0,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        n_images_to_vis: int = 16,
        ema_rate: float = 0.99,
        ema_update_every: int = 100,
        ema_update_after_step: int = 1000,
        use_ema_for_sampling: bool = True,
        metric_tracker_cfg: dict = None,
        lr_scheduler_cfg: dict = None,
        log_grad_norm: bool = False,
        validation_timesteps: list = [5, 40],
    ):
        """
        Args:
            fm_cfg: Flow matching model config.
            start_from_noise: Whether to start from noise with low-res image as
                conditioning (FM) or directly from low-res image (IC-FM).
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
        self.model = instantiate_from_config(fm_cfg)
        # self.model = torch.compile(self.model)            # TODO haven't fully debugged yet
        self.ema_model = EMA(
            self.model,
            beta=ema_rate,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every,
            power=3 / 4.0,  # recommended for trainings < 1M steps
            include_online_model=False,  # we have the online model stored here
        )

        # initialize swinir model
        self.swinir = instantiate_from_config(swinir_cfg)
        sd = torch.load(swinir_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in sd.items()
        }
        self.swinir.load_state_dict(sd, strict=True)
        for p in self.swinir.parameters():
            p.requires_grad = False
        print(f"load SwinIR from {swinir_path}")

        self.use_ema_for_sampling = use_ema_for_sampling
        self.start_from_noise = start_from_noise
        self.concat_context = concat_context
        self.ca_context = ca_context

        # forward diffusion of image
        self.noise_image = noising_step > 0
        self.noising_step = noising_step
        if self.start_from_noise and self.noise_image:
            raise ValueError("Cannot use noising step with start_from_noise=True")
        if self.noising_step > 0:
            if self.noising_step > 1 and isinstance(self.noising_step, int):
                self.diffusion = ForwardDiffusion()
            else:
                raise ValueError("Invalid noising step")
        else:
            self.diffusion = None

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
        self.validation_timesteps = validation_timesteps

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

    def forward(self, x_target: Tensor, x_source: Tensor, **kwargs):
        return self.model.training_losses(
            x1=x_target,
            x0=x_source,
            ema_model=self.ema_model.ema_model,
            **kwargs,
        )

    @torch.no_grad()
    def encode_first_stage(self, x):
        if not exists(self.first_stage):
            return x
        x = self.first_stage.encode(x)
        if not isinstance(x, torch.Tensor):  # hack for posterior of original VAE
            x = x.mode()
        return x * self.scale_factor

    @torch.no_grad()
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
        gt, lq, prompt = batch
        gt = rearrange(gt, "b h w c -> b c h w").contiguous().float()
        lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()

        with torch.no_grad():
            hres_z = self.encode_first_stage(gt)
            clean = self.swinir(lq)
            clean = clean * 2 - 1

        lres_z = self.encode_first_stage(clean)

        return gt.float(), hres_z.float(), lq.float() * 2 - 1, lres_z.float(), clean

    def training_step(self, batch, batch_idx):
        """extract high-res and low-res images from batch"""
        hres_ims, hres_z, lres_ims, lres_z, _ = self.extract_from_batch(batch)

        """ context & conditioning information """
        x_source, context, context_ca = self.get_source_and_context(lres_z)

        """ loss """
        loss = self.forward(
            x_target=hres_z, x_source=x_source, context=context, context_ca=context_ca
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=x_source.shape[0],
        )

        """ misc """
        self.ema_model.update()
        if exists(self.lr_scheduler_cfg):
            self.lr_schedulers().step()
        if self.stop_training:
            self.stop_training_method()
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

        return loss

    def get_source_and_context(self, lres_z: Tensor):
        lres_z_hr = lres_z

        # define x0
        if self.start_from_noise:
            x_source = torch.randn_like(lres_z_hr)
        else:
            x_source = lres_z_hr

        # noise the start
        if self.noise_image:
            x_source = self.diffusion.q_sample(x_start=x_source, t=self.noising_step)

        # define context (concatenated with image latent codes)
        if self.concat_context or self.start_from_noise:
            context = lres_z_hr

        context_ca = None

        return x_source, context, context_ca

    def predict_high_res_z(self, lres_z: Tensor, sample_kwargs=None):
        """
        Decode from x0 -> x1 (low-res -> high-res latent code)
        Args:
            lres_z: low-res latent codes (in low-resolution)
            z_hr_size: size of the high-res latent code
        Returns:
            hr_pred_z: high-res latent codes
        """
        z, context, context_ca = self.get_source_and_context(lres_z)

        # up-sample with flow matching
        if not exists(sample_kwargs):
            # default during training
            sample_kwargs = dict(num_steps=40, method="rk4")

        fn = (
            self.ema_model.model.generate
            if self.use_ema_for_sampling
            else self.model.generate
        )
        hr_pred_z = fn(
            x=z, context=context, context_ca=context_ca, sample_kwargs=sample_kwargs
        )

        return hr_pred_z

    def predict_high_res_img(self, lres_z: Tensor, sample_kwargs=None):
        """
        Decode from x0 -> x1 -> VAE-decode (low-res z -> high-res image)
        Args:
            lres_z: low-res latent codes (in low-resolution)
        Returns:
            hr_pred: high-res images (already decoded with first stage)
        """
        hr_pred_z = self.predict_high_res_z(lres_z, sample_kwargs=sample_kwargs)
        # decode with first stage (if no first stage, this is identity)
        hr_pred = self.decode_first_stage(hr_pred_z)
        return hr_pred

    def validation_step(self, batch, batch_idx):
        hr_ims, hr_z, lr_ims, lr_z, lr_clean = self.extract_from_batch(batch)

        _vis_samples = {"hr": hr_ims, "lr": lr_ims, "lr_clean": lr_clean}

        for num_steps in self.validation_timesteps:
            sample_kwargs = dict(num_steps=num_steps, method="euler")
            hr_pred = self.predict_high_res_img(lres_z=lr_z, sample_kwargs=sample_kwargs)
            _vis_samples[f"pred_s{num_steps}"] = hr_pred

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
