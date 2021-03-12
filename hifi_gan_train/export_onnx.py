#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .config import TrainingConfig
from .utils import to_gpu

_LOGGER = logging.getLogger("hifi_gan_train.export_onnx")

OPSET_VERSION = 12

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="glow-tts-export-onnx")
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pth)")
    parser.add_argument("output", help="Path to output onnx model")
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )
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

    # Convert to paths
    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)

    if args.config:
        args.config = [Path(p) for p in args.config]
    elif args.checkpoint and args.checkpoint.is_dir():
        # Look for config in checkpoint directory
        maybe_config_path = args.checkpoint / "config.json"
        if maybe_config_path.is_file():
            _LOGGER.debug("Found config in checkpoint directory: %s", maybe_config_path)
            args.config = [maybe_config_path]

    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    # Load checkpoint
    _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, config)
    generator = checkpoint.training_model.generator

    _LOGGER.info(
        "Loaded checkpoint from %s (global step=%s)",
        args.checkpoint,
        checkpoint.global_step,
    )

    # Inference only
    generator.eval()
    generator.remove_weight_norm()

    if args.output.is_dir():
        # Output to directory
        args.output.mkdir(parents=True, exist_ok=True)
        output_path = args.output / "generator.onnx"
    else:
        # Output to file
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_path = args.output

    # Create dummy input
    dummy_input = to_gpu(torch.randn((1, config.audio.num_mels, 50), dtype=torch.float))

    # Export
    torch.onnx.export(
        generator,
        dummy_input,
        str(output_path),
        opset_version=12,
        do_constant_folding=True,
        input_names=["mel"],
        output_names=["audio"],
        dynamic_axes={
            "mel": {0: "batch_size", 2: "mel_length"},
            "audio": {0: "batch_size", 1: "audio_length"},
        },
    )

    _LOGGER.info("Exported model to %s", args.output)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
