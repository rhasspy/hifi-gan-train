#!/usr/bin/env python3
import argparse
import logging
import os
import random
import sys
import typing
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import load_checkpoint
from .config import TrainingConfig
from .dataset import MelDataset
from .models import TrainingModel, setup_model
from .train import train

_LOGGER = logging.getLogger("hifi_gan_train")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="hifi-gan-train")
    parser.add_argument("model_dir", help="Directory to store model artifacts")
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: use config)"
    )
    parser.add_argument("--checkpoint", help="Directory to restore checkpoint")
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=1,
        help="Number of epochs between checkpoints",
    )
    parser.add_argument(
        "--local_rank", type=int, help="Rank passed from torch.distributed.launch"
    )
    parser.add_argument("--git-commit", help="Git commit to store in config")
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    assert torch.cuda.is_available(), "GPU is required for training"

    is_distributed = args.local_rank is not None

    if is_distributed:
        _LOGGER.info("Setting up distributed run (rank=%s)", args.local_rank)
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # -------------------------------------------------------------------------

    # Convert to paths
    args.model_dir = Path(args.model_dir)

    if args.config:
        args.config = [Path(p) for p in args.config]

    if args.checkpoint:
        args.checkpoint = Path(args.checkpoint)

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    config.git_commit = args.git_commit

    # Create output directory
    args.model_dir.mkdir(parents=True, exist_ok=True)

    _LOGGER.debug("Setting random seed to %s", config.seed)
    random.seed(config.seed)

    # Load wav paths
    if os.isatty(sys.stdin.fileno()):
        print("Reading WAV path(s) from stdin...", file=sys.stderr)

    wav_paths = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            wav_paths.append(Path(line))

    _LOGGER.info("Loaded %s WAV path(s)", len(wav_paths))

    # Create data loader
    dataset = MelDataset(
        wav_paths=wav_paths, config=config, shuffle=(not is_distributed)
    )

    batch_size = config.batch_size if args.batch_size is None else args.batch_size
    sampler = DistributedSampler(dataset) if is_distributed else None

    train_loader = DataLoader(
        dataset,
        shuffle=(not is_distributed),
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )

    training_model: typing.Optional[TrainingModel] = None
    global_step: int = 1

    if args.checkpoint:
        _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
        checkpoint = load_checkpoint(args.checkpoint, config)
        training_model = checkpoint.training_model
        global_step = checkpoint.global_step
        _LOGGER.info(
            "Loaded checkpoint from %s (global step=%s)", args.checkpoint, global_step
        )
    else:
        training_model = setup_model(config)

    # TODO
    # if is_distributed:
    #     model = DistributedDataParallel(
    #         model, device_ids=[args.local_rank], output_device=args.local_rank
    #     )

    # Train
    _LOGGER.info("Training started (batch size=%s)", batch_size)
    # torch.autograd.set_detect_anomaly(True)
    try:
        train(
            train_loader,
            config,
            args.model_dir,
            mel_spectrogram=dataset.mel_spectrogram,
            training_model=training_model,
            global_step=global_step,
            checkpoint_epochs=args.checkpoint_epochs,
            rank=(args.local_rank if is_distributed else 0),
        )
        _LOGGER.info("Training finished")
    except KeyboardInterrupt:
        _LOGGER.info("Training stopped")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
