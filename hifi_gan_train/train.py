import logging
import time
import typing
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .checkpoint import Checkpoint, save_checkpoint
from .dataset import MelFunction
from .config import TrainingConfig
from .models import (
    TrainingModel,
    discriminator_loss,
    feature_loss,
    generator_loss,
    setup_model,
)
from .utils import to_gpu

_LOGGER = logging.getLogger("tacotron2_train")

# -----------------------------------------------------------------------------


def train(
    train_loader: DataLoader,
    config: TrainingConfig,
    model_dir: Path,
    mel_spectrogram: MelFunction,
    training_model: typing.Optional[TrainingModel] = None,
    global_step: int = 1,
    checkpoint_epochs: int = 1,
    rank: int = 0,
):
    """Run training for the specified number of epochs"""
    torch.manual_seed(config.seed)

    training_model = setup_model(config, training_model=training_model)

    # Gradient scaler
    scaler = GradScaler() if config.fp16_run else None

    # Begin training
    for epoch in range(1, config.epochs + 1):
        _LOGGER.debug(
            "Begin epoch %s/%s (global step=%s)", epoch, config.epochs, global_step
        )
        epoch_start_time = time.perf_counter()
        global_step = train_step(
            global_step=global_step,
            epoch=epoch,
            config=config,
            training_model=training_model,
            train_loader=train_loader,
            mel_spectrogram=mel_spectrogram,
            fp16_run=config.fp16_run,
            scaler=scaler,
        )

        if ((epoch % checkpoint_epochs) == 0) and (rank == 0):
            # Save checkpoint
            checkpoint_path = model_dir / f"checkpoint_{global_step}.pth"
            _LOGGER.debug("Saving checkpoint to %s", checkpoint_path)
            save_checkpoint(
                Checkpoint(
                    training_model=training_model,
                    epoch=epoch,
                    global_step=global_step,
                    version=config.version,
                ),
                checkpoint_path,
            )

            # Save checkpoint config
            config_path = model_dir / f"config_{global_step}.json"
            with open(config_path, "w") as config_file:
                config.save(config_file)

            _LOGGER.info("Saved checkpoint to %s", checkpoint_path)

        epoch_end_time = time.perf_counter()
        _LOGGER.debug(
            "Epoch %s complete in %s second(s) (global step=%s)",
            epoch,
            epoch_end_time - epoch_start_time,
            global_step,
        )


def train_step(
    global_step: int,
    epoch: int,
    config: TrainingConfig,
    training_model: TrainingModel,
    train_loader: DataLoader,
    mel_spectrogram: MelFunction,
    fp16_run: bool,
    scaler: typing.Optional[GradScaler] = None,
) -> int:
    steps_per_epoch = len(train_loader)

    generator, mpd, msd = (
        training_model.generator,
        training_model.mpd,
        training_model.msd,
    )
    assert generator and mpd and msd

    optim_d, optim_g = training_model.optimizer_d, training_model.optimizer_g
    assert optim_d and optim_g

    generator.train()
    mpd.train()
    msd.train()

    for batch_idx, batch in enumerate(train_loader):
        x, y, _, y_mel = batch
        x = torch.autograd.Variable(to_gpu(x))
        y = torch.autograd.Variable(to_gpu(y))
        y_mel = torch.autograd.Variable(to_gpu(y_mel))
        y = y.unsqueeze(1)

        with autocast(enabled=fp16_run):
            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), config.audio.mel_fmax_loss
            )

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, _losses_disc_f_r, _losses_disc_f_g = discriminator_loss(
                y_df_hat_r, y_df_hat_g
            )

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, _losses_disc_s_r, _losses_disc_s_g = discriminator_loss(
                y_ds_hat_r, y_ds_hat_g
            )

            loss_disc_all = loss_disc_s + loss_disc_f

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, _losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, _losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        with torch.no_grad():
            mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

        if fp16_run:
            # Float16
            assert scaler is not None

            scaler.scale(loss_disc_all).backward()
            scaler.scale(loss_gen_all).backward()

            scaler.step(optim_d)
            scaler.step(optim_g)

            scaler.update()
        else:
            # Float32
            loss_disc_all.backward()
            loss_gen_all.backward()

            optim_d.step()
            optim_g.step()

        _LOGGER.debug(
            "Loss: %s, Mel Error: %s (step=%s/%s)",
            loss_gen_all.item(),
            mel_error,
            batch_idx + 1,
            steps_per_epoch,
        )
        global_step += 1

    return global_step
