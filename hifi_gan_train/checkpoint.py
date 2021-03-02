"""Methods for saving/loading checkpoints"""
import logging
import typing
from dataclasses import dataclass
from pathlib import Path

import torch

from .config import TrainingConfig
from .models import TrainingModel, setup_model

_LOGGER = logging.getLogger("hifi_gan_train.checkpoint")

# -----------------------------------------------------------------------------


@dataclass
class Checkpoint:
    training_model: TrainingModel
    epoch: int
    global_step: int
    version: int


def save_checkpoint(checkpoint: Checkpoint, checkpoint_dir: Path):
    """Save models and training state to a directory of Torch checkpoints"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_model = checkpoint.training_model

    # Generator
    generator_path = checkpoint_dir / "generator.pth"
    torch.save(
        {
            "generator": get_state_dict(training_model.generator),
            "epoch": checkpoint.epoch,
            "global_step": checkpoint.global_step,
            "version": checkpoint.version,
        },
        generator_path,
    )
    _LOGGER.debug("Saved generator to %s", generator_path)

    # Discriminators/Optimizers
    discrim_optim_path = checkpoint_dir / "discrim_optim.pth"
    torch.save(
        {
            "mpd": get_state_dict(training_model.mpd),
            "msd": get_state_dict(training_model.msd),
            "optim_g": get_state_dict(training_model.optimizer_g),
            "optim_d": get_state_dict(training_model.optimizer_d),
            "epoch": checkpoint.epoch,
            "global_step": checkpoint.global_step,
            "version": checkpoint.version,
        },
        discrim_optim_path,
    )


def get_state_dict(model):
    """Return model state dictionary whether or not distributed training was used"""
    if hasattr(model, "module"):
        return model.module.state_dict()

    return model.state_dict()


# -----------------------------------------------------------------------------


def load_checkpoint(
    checkpoint_dir: Path,
    config: TrainingConfig,
    training_model: typing.Optional[TrainingModel] = None,
    load_discriminator_optimizer: bool = True,
    use_cuda: bool = True,
) -> Checkpoint:
    """Load models and training state from a directory of Torch checkpoints"""
    # Generator
    generator_path = checkpoint_dir / "generator.pth"

    _LOGGER.debug("Loading generator from %s", generator_path)
    generator_dict = torch.load(generator_path, map_location="cpu")

    version = int(generator_dict.get("version", 1))
    global_step = int(generator_dict.get("global_step", 1))
    epoch = int(generator_dict.get("epoch", -1))

    if not training_model:
        training_model = setup_model(
            config,
            create_discriminator_optimizer=load_discriminator_optimizer,
            last_epoch=epoch,
            use_cuda=use_cuda,
        )

    assert training_model.generator, "No generator"
    set_state_dict(training_model.generator, generator_dict)

    # Load discriminator/optimizer states
    if load_discriminator_optimizer:
        # Verify model has been set up
        assert training_model.mpd, "No multi-period discriminator"
        assert training_model.msd, "No multi-scale discriminator"
        assert training_model.optimizer_g, "No generator optimizer"
        assert training_model.optimizer_d, "No discriminator optimizer"

        # Load state dicts
        discrim_optim_path = checkpoint_dir / "discrim_optim.pth"
        _LOGGER.debug("Loading discriminator/optimizer from %s", discrim_optim_path)
        discrim_optim_dict = torch.load(discrim_optim_path, map_location="cpu")

        set_state_dict(training_model.mpd, discrim_optim_dict["mpd"])
        set_state_dict(training_model.msd, discrim_optim_dict["msd"])
        set_state_dict(training_model.optimizer_d, discrim_optim_dict["optim_d"])
        set_state_dict(training_model.optimizer_g, discrim_optim_dict["optim_g"])

        do_epoch = int(discrim_optim_dict.get("epoch", epoch))
        do_version = int(discrim_optim_dict.get("version", version))

        if do_epoch != epoch:
            _LOGGER.warning(
                "Generator and discriminator/optimizer epoch mismatch (gen=%s, do=%s)",
                epoch,
                do_epoch,
            )

        if do_version != version:
            _LOGGER.warning(
                "Generator and discriminator/optimizer version mismatch (gen=%s, do=%s)",
                version,
                do_version,
            )

    return Checkpoint(
        training_model=training_model,
        epoch=epoch,
        global_step=global_step,
        version=version,
    )


def set_state_dict(model, state_dict):
    """Load state dictionary whether or not distributed training was used"""
    if hasattr(model, "module"):
        return model.module.load_state_dict(state_dict)

    return model.load_state_dict(state_dict)
