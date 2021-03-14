#!/usr/bin/env python3
"""Converts JSONL mel spectrograms to WAV audio using NVIDIA's WaveGlow"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import TrainingConfig
from .models import Generator
from .checkpoint import load_checkpoint
from .wavfile import write as wav_write

_LOGGER = logging.getLogger("hifi_gan")

MAX_WAV_VALUE = 32767.0

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="hifi-gan")
    parser.add_argument("checkpoint", help="Path to generator checkpoint")
    parser.add_argument("output_dir", help="Directory to write WAV files")
    parser.add_argument(
        "--numpy-files",
        action="store_true",
        help="stdin lines are numpy mel file names",
    )
    parser.add_argument(
        "--config", action="append", help="Path to JSON configuration file(s)"
    )

    # Spectrogram settings
    parser.add_argument("--mel-channels", type=int, default=80)
    parser.add_argument("--sampling-rate", type=int, default=22050)

    parser.add_argument("--cuda", action="store_true", help="Use GPU for inference")
    parser.add_argument(
        "--fp16", action="store_true", help="Use FP16 for GPU inference"
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

    # Convert to paths
    args.checkpoint = Path(args.checkpoint)
    args.output_dir = Path(args.output_dir)

    if args.config:
        args.config = [Path(p) for p in args.config]
    else:
        # Look for config next to checkpoint
        maybe_config_path = args.checkpoint.parent / "config.json"
        if maybe_config_path.is_file():
            _LOGGER.debug("Found config next to checkpoint: %s", maybe_config_path)
            args.config = [maybe_config_path]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load configuration
    config = TrainingConfig()
    if args.config:
        _LOGGER.debug("Loading configuration(s) from %s", args.config)
        config = TrainingConfig.load_and_merge(config, args.config)

    _LOGGER.debug(config)

    torch.manual_seed(config.seed)

    # Generator
    generator = Generator(config)

    _LOGGER.debug("Loading checkpoint from %s", args.checkpoint)
    device = "cuda" if args.cuda else "cpu"
    state_dict_g = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state_dict_g["generator"])

    generator.eval()
    generator.remove_weight_norm()
    _LOGGER.info("Loaded generator from %s", args.checkpoint)

    # -------------------------------------------------------------------------

    _LOGGER.info("Ready")

    if os.isatty(sys.stdin.fileno()):
        print("Reading JSON from stdin...", file=sys.stderr)

    # Read JSON objects from standard input.
    # Each object should have this structure:
    # {
    #   "id": "utterance id (used for output file name)",
    #   "audio": {
    #     "filter_length": length of filter,
    #     "hop_length": length of hop,
    #     "win_length": length of window,
    #     "mel_channels": number of mel channels,
    #     "sample_rate": sample rate of audio,
    #     "mel_fmin": min frequency for mel,
    #     "mel_fmax": max frequency for mel
    #   },
    #   "mel": [numpy array of shape (mel_channels, mel_windows)]
    # }
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                # Skip blank lines
                continue

            if args.numpy_files:
                # Lines are numpy mel file names instead of JSON
                mel_path = Path(line)
                mel = torch.from_numpy(
                    np.load(str(mel_path), allow_pickle=True)
                ).unsqueeze(0)
                utt_id = mel_path.stem
            else:
                # Lines are JSON
                mel_obj = json.loads(line)
                mel = torch.FloatTensor(mel_obj["mel"]).unsqueeze(0)
                utt_id = mel_obj.get("id", "")

                # Load audio settings
                # TODO: Verify audio settings
                # audio_obj = mel_obj.get("audio", {})
                # audio_settings = AudioSettings(
                #     filter_length=audio_obj.get("filter_length", args.filter_length),
                #     hop_length=audio_obj.get("hop_length", args.hop_length),
                #     win_length=audio_obj.get("win_length", args.win_length),
                #     mel_channels=audio_obj.get("mel_channels", args.mel_channels),
                #     sampling_rate=audio_obj.get("sampling_rate", args.sampling_rate),
                #     mel_fmin=audio_obj.get("mel_fmin", args.mel_fmin),
                #     mel_fmax=audio_obj.get("mel_fmax", args.mel_fmax),
                # )

            _LOGGER.debug("Mel shape: %s", mel.shape)

            if args.cuda:
                mel.cuda()

            # Run Hifi-GAN
            start_time = time.perf_counter()
            with torch.no_grad():
                _LOGGER.debug("Running inference...")
                signal = generator(mel)

            signal = signal.squeeze().cpu().numpy()
            signal = signal * MAX_WAV_VALUE
            signal = signal.astype("int16")

            end_time = time.perf_counter()

            # Save WAV data
            if not utt_id:
                # Use timestamp
                utt_id = str(time.time())

            wav_path = args.output_dir / (utt_id + ".wav")
            with open(wav_path, "wb") as wav_file:
                wav_write(wav_file, args.sampling_rate, signal)

            duration_sec = len(signal) / args.sampling_rate
            infer_sec = end_time - start_time
            real_time_factor = duration_sec / infer_sec

            _LOGGER.debug(
                "Wrote %s (%s sample(s), %0.2f second(s)) in %0.2f second(s) with real-time factor of %0.2f",
                wav_path,
                len(signal),
                duration_sec,
                infer_sec,
                real_time_factor,
            )
    except KeyboardInterrupt:
        pass


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
