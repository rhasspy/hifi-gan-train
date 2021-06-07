# Hi-Fi GAN

Version of [Hi-Fi GAN](https://github.com/jik876/hifi-gan) designed to work with

* [tacotron2-train](https://github.com/rhasspy/tacotron2-train)
* [glow-tts-train](https://github.com/rhasspy/glow-tts-train)

## Additional Features

* Models can be exported to [onnx](https://onnx.ai/) format

## Dependencies

* Python 3.7 or higher
* PyTorch 1.6 or higher
* librosa

## Installation

```sh
git clone https://github.com/rhasspy/hifi-gan-train
cd hifi-gan-train
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install --upgrade wheel setuptool
pip3 install -r requirements.txt
```

## Running

```sh
bin/hifi-gan-train --debug /path/to/model --config /path/to/config.json < /path/to/wav_paths.txt
```

See the `configs` directory for example configs and `--help` for more options.
